"""Shared utilities: seeding, data loading, graph construction, splits.

These mirror the first cells of every notebook so the data pipeline is
identical across experiments.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# Sentiment label encoding used throughout the project.
#   n (negative) -> 0,  o (objective/neutral) -> 1,  p (positive) -> 2
SENTIMENT_MAP = {"o": 1, "p": 2, "n": 0}
COLUMNS = ["Source_PaperID", "Target_PaperID", "Sentiment", "Citation_text"]


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and (if importable) TensorFlow / PyTorch.

    Heavy frameworks are imported lazily so a TF-only run does not pull in
    PyTorch and vice-versa.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
    try:  # PyTorch
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(device: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. 'auto' picks cuda when available."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def load_corpus(data_path: str) -> pd.DataFrame:
    """Load the tab-separated citation sentiment corpus into a DataFrame."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Place the corpus there or pass --data-path. See data/README.md."
        )
    df = pd.read_csv(data_path, sep="\t", header=None)
    df.columns = COLUMNS
    df["Sentiment"] = df["Sentiment"].replace(SENTIMENT_MAP)
    return df


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build the directed citation graph (Source -> Target, weight 1)."""
    edges = pd.DataFrame(
        {
            "source": df["Source_PaperID"],
            "target": df["Target_PaperID"],
            "weight": [1] * len(df),
        }
    )
    return nx.from_pandas_edgelist(
        edges, source="source", target="target",
        edge_attr=True, create_using=nx.DiGraph(),
    )


def remove_zero_degree_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """Drop nodes whose out-degree is zero (``len(G[node]) == 0``).

    Matches the notebooks, which built the leaderboard from ``len(G[x])``
    (number of successors) and removed every node with a count of 0.
    """
    zero_nodes = [node for node in list(G.nodes()) if len(G[node]) == 0]
    G.remove_nodes_from(zero_nodes)
    return G


def make_splits(
    labels: pd.Series,
    test_size: float,
    val_test_size: float,
    seed,
    stratify: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Two-stage train/val/test split (70:18:12 with the project defaults).

    First hold out ``test_size`` of the data, then split that holdout into
    validation / test using ``val_test_size`` as the test share.

    ``stratify=True`` preserves class proportions in every split, matching the
    classical-baseline notebooks; the GCNN notebooks used an unstratified split,
    so the GCNN path leaves it False. ``seed=None`` gives a random split.
    """
    from sklearn.model_selection import train_test_split

    train_label, holdout = train_test_split(
        labels, test_size=test_size, random_state=seed,
        stratify=labels if stratify else None,
    )
    val_label, test_label = train_test_split(
        holdout, test_size=val_test_size, random_state=seed,
        stratify=holdout if stratify else None,
    )
    return train_label, val_label, test_label


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
