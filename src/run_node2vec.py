"""Experiment: Node2Vec features -> custom TF GNN -> classifiers.

    python src/run_node2vec.py --data-path data/citation_sentiment_corpus.txt
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import add_common_args, make_config
from pipeline import run_tf_experiment


def main():
    parser = argparse.ArgumentParser(description="Node2Vec + GCN citation sentiment")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gridsearch", action="store_true", help="Tune downstream classifiers")
    args = parser.parse_args()

    overrides = {"run_gridsearch": args.gridsearch}
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    cfg = make_config(args, **overrides)

    run_tf_experiment(cfg, variant="node2vec")


if __name__ == "__main__":
    main()
