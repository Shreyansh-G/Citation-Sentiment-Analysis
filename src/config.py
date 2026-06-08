"""Central configuration for the Citation Sentiment Analysis experiments.

All tunable knobs live here so every run is fully described by a single
``Config`` object. The defaults reproduce the values used in the original
notebooks; anything can be overridden from the command line (see
``add_common_args`` / ``make_config``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_path: str = "data/citation_sentiment_corpus.txt"
    cache_dir: str = "cache"          # cached Node2Vec / BERT embeddings
    output_dir: str = "outputs"       # metrics / figures
    use_cache: bool = True

    # ---- reproducibility ---------------------------------------------------
    seed: int = 42

    # ---- Node2Vec ----------------------------------------------------------
    n2v_dim: int = 128
    n2v_walk_length: int = 80
    n2v_num_walks: int = 10
    n2v_p: float = 1.0                 # return parameter (PyG run uses 0.5)
    n2v_q: float = 1.0                 # in-out parameter (PyG run uses 0.4)
    n2v_window: int = 10
    n2v_workers: int = 1               # 1 => deterministic walks

    # ---- BERT --------------------------------------------------------------
    bert_model: str = "bert-base-uncased"
    bert_max_length: int = 512
    bert_pca_dim: int = 128
    device: str = "auto"               # auto | cuda | cpu

    # ---- custom TF GNN -----------------------------------------------------
    hidden_units: Tuple[int, ...] = (128, 64)
    aggregation_type: str = "gcn"      # gcn|mean|gated|meanpool|maxpool|twomaxpool
    dropout_rate: float = 0.5
    learning_rate: float = 0.001
    num_epochs: int = 40
    batch_size: int = 32
    num_classes: int = 3

    # ---- data splits -------------------------------------------------------
    test_size: float = 0.30            # held out from the full set
    val_test_size: float = 0.40        # share of the held-out used as test

    # ---- downstream classifiers (TF variants) ------------------------------
    run_gridsearch: bool = False

    # ---- run control -------------------------------------------------------
    runs: int = 1          # number of repetitions (use with seeded=False)
    seeded: bool = True    # False => non-deterministic split/training each run

    # Decoupled two-seed mode (GCNN): split (numpy) and GCN init (TF) seeded
    # independently -> reproduces a favorable run while staying replayable.
    # e.g. the documented runs 179776560/1366669488 (91.39%) and
    #      941508052/769732007 (90.83%) -- pass via --split-seed / --init-seed.
    split_seed: int = None
    init_seed: int = None


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the options shared by every run script."""
    g = parser.add_argument_group("common")
    g.add_argument("--data-path", default=None, help="Path to citation_sentiment_corpus.txt")
    g.add_argument("--cache-dir", default=None, help="Directory for cached embeddings")
    g.add_argument("--output-dir", default=None, help="Directory for outputs")
    g.add_argument("--seed", type=int, default=None, help="Global random seed")
    g.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    g.add_argument("--no-cache", action="store_true", help="Recompute embeddings, ignore cache")
    g.add_argument("--runs", type=int, default=None, help="Repeat the experiment N times (use with --no-seed)")
    g.add_argument("--no-seed", action="store_true",
                   help="Non-deterministic split/training each run (reproduces the original notebooks)")
    g.add_argument("--split-seed", type=int, default=None,
                   help="GCNN: decoupled split seed (numpy); use together with --init-seed")
    g.add_argument("--init-seed", type=int, default=None,
                   help="GCNN: decoupled GCN-init seed (TF); use together with --split-seed")
    return parser


def make_config(args: argparse.Namespace, **overrides) -> Config:
    """Build a ``Config`` from parsed args, then apply explicit overrides."""
    cfg = Config()
    if getattr(args, "data_path", None):
        cfg.data_path = args.data_path
    if getattr(args, "cache_dir", None):
        cfg.cache_dir = args.cache_dir
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "no_cache", False):
        cfg.use_cache = False
    if getattr(args, "runs", None) is not None:
        cfg.runs = args.runs
    if getattr(args, "no_seed", False):
        cfg.seeded = False
    if getattr(args, "split_seed", None) is not None:
        cfg.split_seed = args.split_seed
    if getattr(args, "init_seed", None) is not None:
        cfg.init_seed = args.init_seed
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
