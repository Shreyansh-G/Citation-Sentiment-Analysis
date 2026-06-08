"""Directed node- and edge-level graph features.

* Node features: in/out/total degree, betweenness, closeness, eigenvector
  centrality.
* Edge features: directed preferential attachment, resource allocation,
  Jaccard coefficient, Adamic-Adar index.

The TF GNN variants compute these for inspection only; the PyG
``Paper_NodeFeatures`` pipeline feeds them into the model, so the optional
``transform_node_edge_simple`` scaler lives here too.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------- #
# Directed edge-score functions
# --------------------------------------------------------------------------- #
def directed_preferential_attachment(graph, edges):
    """PA(u, v) = out_degree(u) * in_degree(v)."""
    return [(u, v, graph.out_degree(u) * graph.in_degree(v)) for u, v in edges]


def directed_resource_allocation_index(graph, edges):
    """RA(u, v) = sum_{w in succ(u) & pred(v)} 1 / degree(w)."""
    scores = []
    for u, v in edges:
        common = set(graph.successors(u)) & set(graph.predecessors(v))
        score = sum(1 / graph.degree(w) for w in common if graph.degree(w) > 0)
        scores.append((u, v, score))
    return scores


def directed_jaccard_coefficient(graph, edges):
    """JC(u, v) = |succ(u) & pred(v)| / |succ(u) | pred(v)|."""
    scores = []
    for u, v in edges:
        succ_u = set(graph.successors(u))
        pred_v = set(graph.predecessors(v))
        union = succ_u | pred_v
        score = len(succ_u & pred_v) / len(union) if union else 0
        scores.append((u, v, score))
    return scores


def directed_adamic_adar_index(graph, edges):
    """AA(u, v) = sum_{w in succ(u) & pred(v)} 1 / log(degree(w))."""
    scores = []
    for u, v in edges:
        common = set(graph.successors(u)) & set(graph.predecessors(v))
        score = sum(1 / math.log(graph.degree(w)) for w in common if graph.degree(w) > 1)
        scores.append((u, v, score))
    return scores


# --------------------------------------------------------------------------- #
# Feature tables
# --------------------------------------------------------------------------- #
def compute_node_features(G: nx.DiGraph) -> pd.DataFrame:
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    total_deg = {n: in_deg[n] + out_deg[n] for n in G.nodes()}
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)

    nodes = list(G.nodes())
    return pd.DataFrame(
        {
            "node": nodes,
            "in_degree": [in_deg[n] for n in nodes],
            "out_degree": [out_deg[n] for n in nodes],
            "total_degree": [total_deg[n] for n in nodes],
            "betweenness": [betweenness[n] for n in nodes],
            "closeness": [closeness[n] for n in nodes],
            "eigenvector_centrality": [eigenvector[n] for n in nodes],
        }
    )


def compute_edge_features(G: nx.DiGraph) -> pd.DataFrame:
    edges = list(G.edges())
    pa = directed_preferential_attachment(G, edges)
    ra = directed_resource_allocation_index(G, edges)
    jc = directed_jaccard_coefficient(G, edges)
    aa = directed_adamic_adar_index(G, edges)
    return pd.DataFrame(
        {
            "source": [u for u, _, _ in pa],
            "target": [v for _, v, _ in pa],
            "preferential_attachment": [s for _, _, s in pa],
            "resource_allocation": [s for _, _, s in ra],
            "jaccard": [s for _, _, s in jc],
            "adamic_adar": [s for _, _, s in aa],
        }
    )


# --------------------------------------------------------------------------- #
# Scaling pipeline (used by the PyG node-feature experiment)
# --------------------------------------------------------------------------- #
def transform_node_edge_simple(node_features, edge_features, qt_random_state=0):
    """Robust/Yeo-Johnson/Quantile scaling of the numeric feature columns.

    Returns transformed deep copies; id columns (node/source/target) are kept
    untouched. Identical to the ``Paper_NodeFeatures`` notebook.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        StandardScaler,
        RobustScaler,
        PowerTransformer,
        QuantileTransformer,
    )
    from sklearn.impute import SimpleImputer

    node_out = node_features.copy(deep=True)
    edge_out = edge_features.copy(deep=True)

    node_numeric = ["in_degree", "out_degree", "total_degree",
                    "betweenness", "closeness", "eigenvector_centrality"]
    edge_numeric = ["preferential_attachment", "resource_allocation",
                    "jaccard", "adamic_adar"]
    node_cols = [c for c in node_numeric if c in node_out.columns]
    edge_cols = [c for c in edge_numeric if c in edge_out.columns]

    def node_pipe(col):
        if col in ("in_degree", "out_degree", "total_degree"):
            return Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("log1p", FunctionTransformer(np.log1p, validate=False)),
                ("robust", RobustScaler()),
            ])
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("yeo", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("std", StandardScaler()),
        ])

    def edge_pipe(col):
        if col == "preferential_attachment":
            return Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("log1p", FunctionTransformer(np.log1p, validate=False)),
                ("robust", RobustScaler()),
            ])
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("quantile", QuantileTransformer(output_distribution="normal",
                                             random_state=qt_random_state)),
            ("std", StandardScaler()),
        ])

    for c in node_cols:
        try:
            pipe = node_pipe(c).fit(node_features[c].values.reshape(-1, 1))
            node_out[c] = pipe.transform(node_out[c].values.reshape(-1, 1)).reshape(-1)
        except Exception:
            node_out[c] = node_out[c].astype(float)

    for c in edge_cols:
        try:
            pipe = edge_pipe(c).fit(edge_features[c].values.reshape(-1, 1))
            edge_out[c] = pipe.transform(edge_out[c].values.reshape(-1, 1)).reshape(-1)
        except Exception:
            edge_out[c] = edge_out[c].astype(float)

    return node_out, edge_out
