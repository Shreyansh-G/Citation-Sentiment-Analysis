"""End-to-end pipelines wiring data -> features -> model -> metrics.

``run_tf_pipeline``  : Node2Vec / BERT / hybrid features -> custom TF GNN ->
                       extracted embeddings -> classical classifiers.
``run_pyg_pipeline`` : centralities + Node2Vec + BERT + edge features ->
                       PyTorch Geometric GCN/GAT/APPNP/...
"""

from __future__ import annotations

import numpy as np

from common import (
    seed_everything,
    load_corpus,
    build_graph,
    remove_zero_degree_nodes,
    make_splits,
)
from embeddings import build_feature_table

CENTRALITY_COLS = [
    "in_degree", "out_degree", "total_degree",
    "betweenness", "closeness", "eigenvector_centrality",
]
EDGE_FEATURE_COLS = ["preferential_attachment", "resource_allocation", "jaccard", "adamic_adar"]


# --------------------------------------------------------------------------- #
# TF GraphSAGE pipeline (node2vec / bert / hybrid)
# --------------------------------------------------------------------------- #
def run_tf_pipeline(cfg, variant: str, seeded: bool = True):
    """Proposed GCNN path: embeddings -> GCN -> extracted features -> classifiers."""
    import tf_gnn
    import random as pyrandom
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    if cfg.split_seed is not None and cfg.init_seed is not None:
        # Decoupled two-seed mode: split (numpy) and GCN init (TF) seeded
        # independently, then logged -> reproduces a favorable run replayably.
        np.random.seed(cfg.split_seed); pyrandom.seed(cfg.split_seed)
        tf.random.set_seed(cfg.init_seed)
        split_seed = None  # draw the split from the just-seeded global numpy RNG
        tag = f"split_seed={cfg.split_seed} init_seed={cfg.init_seed}"
    elif seeded:
        seed_everything(cfg.seed)
        split_seed = cfg.seed
        tag = f"seed={cfg.seed}"
    else:
        split_seed = None
        tag = "unseeded"
    print(f"\n=== TF GCNN pipeline | features='{variant}' | {tag} ===")

    df = load_corpus(cfg.data_path)
    G = remove_zero_degree_nodes(build_graph(df))
    final_df, feature_cols, graph_edgelist, _ = build_feature_table(G, df, cfg, variant)
    graph_info = tf_gnn.build_graph_info(final_df, feature_cols, graph_edgelist)

    train_label, val_label, test_label = make_splits(
        final_df["label"], cfg.test_size, cfg.val_test_size, split_seed
    )
    x_train = final_df.loc[train_label.index, "Node_id"].to_numpy()
    x_val = final_df.loc[val_label.index, "Node_id"].to_numpy()
    x_test = final_df.loc[test_label.index, "Node_id"].to_numpy()
    y_train = train_label.to_numpy().astype(int)
    y_val = val_label.to_numpy().astype(int)
    y_test = test_label.to_numpy().astype(int)

    model = tf_gnn.GNNNodeClassifier(
        graph_info=graph_info,
        num_classes=cfg.num_classes,
        hidden_units=list(cfg.hidden_units),
        aggregation_type=cfg.aggregation_type,
        dropout_rate=cfg.dropout_rate,
        name="gnn_model",
    )
    tf_gnn.run_experiment(
        model,
        x_train, to_categorical(y_train, cfg.num_classes),
        x_val, to_categorical(y_val, cfg.num_classes),
        learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
    )

    gcn_train = tf_gnn.extract_embeddings(model, x_train)
    gcn_val = tf_gnn.extract_embeddings(model, x_val)
    gcn_test = tf_gnn.extract_embeddings(model, x_test)

    from classifiers import run_classifiers
    return run_classifiers(
        gcn_train, y_train, gcn_val, y_val, gcn_test, y_test,
        gridsearch=cfg.run_gridsearch, seed=cfg.seed,
    )


def run_classical_pipeline(cfg, variant: str, seeded: bool = True):
    """Classical baseline (paper Tables 2-4): raw embeddings -> classifiers."""
    from classifiers import run_classifiers

    if seeded:
        seed_everything(cfg.seed)
    split_seed = cfg.seed if seeded else None
    tag = f"seed={cfg.seed}" if seeded else "unseeded"
    print(f"\n=== Classical baseline | features='{variant}' | {tag} ===")

    df = load_corpus(cfg.data_path)
    G = remove_zero_degree_nodes(build_graph(df))
    final_df, feature_cols, _, _ = build_feature_table(G, df, cfg, variant)

    # Classical-baseline notebooks use a stratified split (stratify=y).
    train_label, val_label, test_label = make_splits(
        final_df["label"], cfg.test_size, cfg.val_test_size, split_seed, stratify=True
    )
    X_train = final_df.loc[train_label.index, feature_cols].to_numpy()
    X_val = final_df.loc[val_label.index, feature_cols].to_numpy()
    X_test = final_df.loc[test_label.index, feature_cols].to_numpy()
    y_train = train_label.to_numpy().astype(int)
    y_val = val_label.to_numpy().astype(int)
    y_test = test_label.to_numpy().astype(int)

    return run_classifiers(
        X_train, y_train, X_val, y_val, X_test, y_test,
        gridsearch=cfg.run_gridsearch, seed=cfg.seed,
    )


def run_tf_experiment(cfg, variant: str, classical: bool = False):
    """Run the GCNN (or classical) TF experiment, repeating cfg.runs times.

    Returns the list of per-run best-classifier accuracies and prints a summary
    (mean / std / min / max) when more than one run is requested.
    """
    import numpy as np

    fn = run_classical_pipeline if classical else run_tf_pipeline
    label = "classical" if classical else "GCNN"
    best_accs = []
    for i in range(cfg.runs):
        results = fn(cfg, variant, seeded=cfg.seeded)
        best_clf = max(results, key=lambda k: results[k]["accuracy"])
        acc = results[best_clf]["accuracy"]
        best_accs.append(acc)
        if cfg.runs > 1:
            print(f"  run {i + 1:2d}/{cfg.runs}: best={best_clf} acc={acc * 100:.2f}%", flush=True)

    if cfg.runs > 1:
        a = np.array(best_accs)
        print(f"\n=== {variant} {label} over {cfg.runs} runs (best classifier per run) ===")
        print(f"mean={a.mean() * 100:.2f}%  std={a.std() * 100:.2f}  "
              f"min={a.min() * 100:.2f}%  max={a.max() * 100:.2f}%")
    return best_accs


# --------------------------------------------------------------------------- #
# PyTorch Geometric pipeline (node-feature experiment)
# --------------------------------------------------------------------------- #
def run_pyg_pipeline(cfg, model_name: str, features: str = "fused", include_structural: bool = True):
    """Train a PyG model.

    features          : 'node2vec' | 'bert' | 'fused' (Node2Vec+BERT).
    include_structural: append the 6 centrality features (paper's 262-d setup).
    """
    from graph_features import compute_node_features, compute_edge_features, transform_node_edge_simple
    from pyg_models import build_pyg_data, train_eval

    seed_everything(cfg.seed)
    variant = {"fused": "hybrid", "node2vec": "node2vec", "bert": "bert"}[features]
    print(f"\n=== PyG pipeline | model='{model_name}' | features='{features}' "
          f"| structural={include_structural} | seed={cfg.seed} ===")

    df = load_corpus(cfg.data_path)
    G = remove_zero_degree_nodes(build_graph(df))

    # Node2Vec / BERT embeddings, sharing the contiguous-id mapping.
    final_df, feature_cols, _, paper_idx = build_feature_table(G, df, cfg, variant)

    # Graph-structural features, scaled, remapped to the same ids.
    # (Edge features are always built so edge_attr is available for gcn_edge.)
    node_feats = compute_node_features(G)
    edge_feats = compute_edge_features(G)
    node_feats, edge_feats = transform_node_edge_simple(node_feats, edge_feats)
    node_feats["Node_id"] = node_feats["node"].map(paper_idx)
    edge_feats["source_index"] = edge_feats["source"].map(paper_idx)
    edge_feats["target_index"] = edge_feats["target"].map(paper_idx)

    node_feats = node_feats.sort_values("Node_id").reset_index(drop=True)
    emb_df = final_df.sort_values("Node_id").reset_index(drop=True)

    blocks = []
    if include_structural:
        blocks.append(node_feats[CENTRALITY_COLS].to_numpy())
    blocks.append(emb_df[feature_cols].to_numpy())
    node_x = np.concatenate(blocks, axis=1)
    labels = emb_df["label"].to_numpy().astype(np.int64)
    edges = edge_feats[["source_index", "target_index"]].to_numpy().T
    edge_attr = edge_feats[EDGE_FEATURE_COLS].to_numpy()
    print(f"Node feature matrix: {node_x.shape}  edges: {edges.shape}  edge_attr: {edge_attr.shape}")

    data = build_pyg_data(node_x, edges, edge_attr, labels, cfg)
    return train_eval(data, model_name, cfg)
