"""Experiment: centralities + Node2Vec + BERT + edge features -> PyG GNN.

Reproduces ``Paper_NodeFeatures``. Choose the architecture with --model.

    python src/run_pyg.py --model gcn   --data-path data/citation_sentiment_corpus.txt
    python src/run_pyg.py --model gat
    python src/run_pyg.py --model gcn_aug
    python src/run_pyg.py --model appnp
    python src/run_pyg.py --model gcn_edge
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import add_common_args, make_config
from pipeline import run_pyg_pipeline


def main():
    parser = argparse.ArgumentParser(description="PyTorch Geometric citation sentiment")
    add_common_args(parser)
    parser.add_argument(
        "--model", default="gcn",
        choices=["gcn", "gat", "gcn_aug", "appnp", "gcn_edge"],
        help="Which PyG architecture to train",
    )
    parser.add_argument(
        "--features", default="fused", choices=["node2vec", "bert", "fused"],
        help="Node feature set (default: fused Node2Vec+BERT)",
    )
    parser.add_argument(
        "--no-structural", action="store_true",
        help="Exclude the 6 centrality features (paper uses them -> 262-d)",
    )
    args = parser.parse_args()

    # The node-feature experiment used biased Node2Vec walks (p=0.5, q=0.4).
    cfg = make_config(args, n2v_p=0.5, n2v_q=0.4)

    run_pyg_pipeline(
        cfg, model_name=args.model,
        features=args.features, include_structural=not args.no_structural,
    )


if __name__ == "__main__":
    main()
