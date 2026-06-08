"""Classical baseline (paper Table 2): Node2Vec embeddings -> classifiers.

No GNN -- the raw 128-d Node2Vec embeddings are fed directly to the classical
models, using the stratified seed-42 split of ``Baseline_Node2Vec_ML.ipynb``.

    python src/run_baseline_node2vec.py
    python src/run_baseline_node2vec.py --gridsearch   # tune classifiers
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import add_common_args, make_config
from pipeline import run_tf_experiment


def main():
    parser = argparse.ArgumentParser(description="Classical baseline: Node2Vec -> classifiers (Table 2)")
    add_common_args(parser)
    parser.add_argument("--gridsearch", action="store_true", help="Tune classifiers with GridSearchCV")
    args = parser.parse_args()

    cfg = make_config(args, run_gridsearch=args.gridsearch)
    run_tf_experiment(cfg, variant="node2vec", classical=True)


if __name__ == "__main__":
    main()
