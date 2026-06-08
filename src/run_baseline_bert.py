"""Classical baseline (paper Table 3): BERT embeddings -> classifiers.

No GNN -- the raw 128-d BERT embeddings are fed directly to the classical
models, using the stratified seed-42 split of ``Baseline_BERT_ML.ipynb``.

    python src/run_baseline_bert.py
    python src/run_baseline_bert.py --gridsearch   # tune classifiers
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import add_common_args, make_config
from pipeline import run_tf_experiment


def main():
    parser = argparse.ArgumentParser(description="Classical baseline: BERT -> classifiers (Table 3)")
    add_common_args(parser)
    parser.add_argument("--gridsearch", action="store_true", help="Tune classifiers with GridSearchCV")
    args = parser.parse_args()

    cfg = make_config(args, run_gridsearch=args.gridsearch)
    run_tf_experiment(cfg, variant="bert", classical=True)


if __name__ == "__main__":
    main()
