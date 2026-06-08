"""Pre-compute and cache the expensive embeddings once.

After this runs, every training script just loads the cached Node2Vec / BERT
tables from ``cache/`` instead of recomputing them.

    # everything (both Node2Vec walk settings + BERT)
    python src/precompute_embeddings.py --what all

    # only Node2Vec with the PyG biased walks
    python src/precompute_embeddings.py --what node2vec --n2v-pq 0.5,0.4

    # only BERT
    python src/precompute_embeddings.py --what bert
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import add_common_args, make_config
from common import seed_everything, load_corpus, build_graph, remove_zero_degree_nodes
import embeddings


def _run_node2vec(cfg, p, q):
    cfg.n2v_p, cfg.n2v_q = p, q
    df = load_corpus(cfg.data_path)
    G = remove_zero_degree_nodes(build_graph(df))
    all_nodes = df["Source_PaperID"].unique()
    emb = embeddings.get_node2vec_embeddings(G, all_nodes, cfg)
    print(f"[done] Node2Vec (p={p}, q={q}): {emb.shape}")


def _run_bert(cfg):
    df = load_corpus(cfg.data_path)
    emb = embeddings.get_bert_embeddings(df, cfg)
    print(f"[done] BERT: {emb.shape}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Node2Vec + BERT embeddings")
    add_common_args(parser)
    parser.add_argument("--what", choices=["node2vec", "bert", "all"], default="all")
    parser.add_argument(
        "--n2v-pq", default=None,
        help="Node2Vec p,q as 'p,q' (e.g. '1,1' for TF, '0.5,0.4' for PyG). "
             "Ignored when --what=all (both are generated).",
    )
    args = parser.parse_args()
    cfg = make_config(args)
    seed_everything(cfg.seed)

    if args.what in ("node2vec", "all"):
        if args.what == "all":
            # Both walk settings used in the paper: TF (p=q=1) and PyG (p=0.5,q=0.4).
            _run_node2vec(cfg, 1.0, 1.0)
            _run_node2vec(cfg, 0.5, 0.4)
        else:
            p, q = (map(float, args.n2v_pq.split(",")) if args.n2v_pq else (1.0, 1.0))
            _run_node2vec(cfg, float(p), float(q))

    if args.what in ("bert", "all"):
        _run_bert(cfg)

    print(f"\nCache directory: {os.path.abspath(cfg.cache_dir)}")


if __name__ == "__main__":
    main()
