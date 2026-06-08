"""Node2Vec and BERT embeddings + the per-experiment feature table.

The expensive embeddings (Node2Vec walks, BERT forward passes) are cached on
disk keyed by their hyperparameters, so the three TF variants and the PyG
experiment reuse a single computation.

``build_feature_table`` returns everything a downstream GNN needs:
    final_df       - Node_id (contiguous int), feature columns, label
    feature_cols   - the embedding column names
    graph_edgelist - the citation edges remapped to the contiguous ids
    paper_idx      - mapping {original paper id -> contiguous int}
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from common import resolve_device, ensure_dir


# --------------------------------------------------------------------------- #
# Caching helpers
# --------------------------------------------------------------------------- #
def _data_tag(cfg) -> str:
    return os.path.splitext(os.path.basename(cfg.data_path))[0]


def _load_cache(path: str, use_cache: bool):
    if use_cache and os.path.exists(path):
        print(f"[cache] loading {path}")
        return pd.read_pickle(path)
    return None


def _save_cache(df: pd.DataFrame, path: str, use_cache: bool) -> None:
    if use_cache:
        ensure_dir(os.path.dirname(path))
        df.to_pickle(path)
        print(f"[cache] saved {path}")


# --------------------------------------------------------------------------- #
# Node2Vec
# --------------------------------------------------------------------------- #
def get_node2vec_embeddings(G: nx.DiGraph, all_nodes, cfg) -> pd.DataFrame:
    """128-d Node2Vec embeddings, reindexed to ``all_nodes`` (missing -> 0)."""
    cache = os.path.join(
        cfg.cache_dir,
        f"{_data_tag(cfg)}_node2vec_d{cfg.n2v_dim}_p{cfg.n2v_p}_q{cfg.n2v_q}"
        f"_wl{cfg.n2v_walk_length}_nw{cfg.n2v_num_walks}_seed{cfg.seed}.pkl",
    )
    cached = _load_cache(cache, cfg.use_cache)
    if cached is not None:
        return cached

    from node2vec import Node2Vec

    print("[node2vec] fitting random-walk embeddings ...")
    node2vec = Node2Vec(
        G,
        dimensions=cfg.n2v_dim,
        walk_length=cfg.n2v_walk_length,
        num_walks=cfg.n2v_num_walks,
        workers=cfg.n2v_workers,   # 1 => deterministic
        p=cfg.n2v_p,
        q=cfg.n2v_q,
        seed=cfg.seed,
    )
    model = node2vec.fit(window=cfg.n2v_window, min_count=1, batch_words=4, seed=cfg.seed)

    emb = pd.DataFrame(
        [model.wv[str(node)] for node in G.nodes()],
        index=[str(node) for node in G.nodes()],
        columns=[f"node2vec_dim_{i}" for i in range(cfg.n2v_dim)],
    )
    # Cover nodes dropped from the graph with zero vectors, then align order.
    for node in set(all_nodes) - set(emb.index):
        emb.loc[node] = np.zeros(cfg.n2v_dim)
    emb = emb.reindex(all_nodes)
    emb.index.name = "Source_PaperID"

    _save_cache(emb, cache, cfg.use_cache)
    return emb


# --------------------------------------------------------------------------- #
# BERT
# --------------------------------------------------------------------------- #
def get_bert_embeddings(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """BERT [CLS] embeddings per unique Source_PaperID, PCA-reduced to 128-d."""
    cache = os.path.join(
        cfg.cache_dir,
        f"{_data_tag(cfg)}_bert_{cfg.bert_model.replace('/', '-')}"
        f"_pca{cfg.bert_pca_dim}_ml{cfg.bert_max_length}_seed{cfg.seed}.pkl",
    )
    cached = _load_cache(cache, cfg.use_cache)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.decomposition import PCA
    from tqdm import tqdm

    device = resolve_device(cfg.device)
    print(f"[bert] encoding citation texts on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model)
    model = AutoModel.from_pretrained(cfg.bert_model).to(device)
    model.eval()

    unique_texts = df.groupby("Source_PaperID").first().reset_index()
    texts = unique_texts["Citation_text"].tolist()
    source_ids = unique_texts["Source_PaperID"].tolist()

    vectors = []
    for text in tqdm(texts, desc="BERT [CLS]"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=cfg.bert_max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        vectors.append(cls)
    vectors = np.array(vectors)

    reduced = PCA(n_components=cfg.bert_pca_dim, random_state=cfg.seed).fit_transform(vectors)
    emb = pd.DataFrame(
        reduced,
        index=source_ids,
        columns=[f"bert_dim_{i}" for i in range(cfg.bert_pca_dim)],
    )
    emb.index.name = "Source_PaperID"

    _save_cache(emb, cache, cfg.use_cache)
    return emb


# --------------------------------------------------------------------------- #
# Feature table assembly
# --------------------------------------------------------------------------- #
def build_feature_table(
    G: nx.DiGraph, df: pd.DataFrame, cfg, variant: str
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, dict]:
    """Assemble node features: 'node2vec', 'bert', or 'hybrid'/'fused'."""
    if variant == "fused":  # alias used by the PyG path
        variant = "hybrid"
    all_nodes = df["Source_PaperID"].unique()

    if variant == "node2vec":
        combined = get_node2vec_embeddings(G, all_nodes, cfg).copy()
    elif variant == "bert":
        combined = get_bert_embeddings(df, cfg).copy()
    elif variant == "hybrid":
        n2v = get_node2vec_embeddings(G, all_nodes, cfg)
        bert = get_bert_embeddings(df, cfg)
        combined = pd.concat([n2v, bert], axis=1)
    else:
        raise ValueError(f"Unknown variant '{variant}' (node2vec|bert|hybrid)")

    # Attach labels and a clean integer Node_id.
    sentiment = df.set_index("Source_PaperID")["Sentiment"].to_dict()
    combined["label"] = combined.index.map(sentiment)
    combined.index.name = "Node_id"
    final_df = combined.reset_index()

    paper_idx = {name: idx for idx, name in enumerate(final_df["Node_id"])}
    final_df["Node_id"] = final_df["Node_id"].map(paper_idx)

    graph_edgelist = nx.to_pandas_edgelist(G)
    graph_edgelist["source"] = graph_edgelist["source"].map(paper_idx)
    graph_edgelist["target"] = graph_edgelist["target"].map(paper_idx)

    feature_cols = [c for c in final_df.columns if c not in ("Node_id", "label")]
    return final_df, feature_cols, graph_edgelist, paper_idx
