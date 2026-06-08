#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run every experiment and print results to the terminal.
#
#   bash run_all.sh
#
# Prerequisite (one time): cache the embeddings, then this script is fast.
#   python src/precompute_embeddings.py --what all
# ---------------------------------------------------------------------------
set -u
PY=python

echo "################  CLASSICAL BASELINES (paper Tables 2-4)  ################"
$PY src/run_baseline_node2vec.py    # Table 2 (Node2Vec)
$PY src/run_baseline_bert.py        # Table 3 (BERT)
$PY src/run_baseline_hybrid.py      # Table 4 (Node2Vec+BERT)

echo "################  GCNN -> ML  (paper Table 1)  ################"
$PY src/run_node2vec.py             # Node2Vec  -> GCNN -> classifiers
$PY src/run_bert.py                 # BERT      -> GCNN -> classifiers
$PY src/run_hybrid.py               # Node2Vec+BERT -> GCNN -> classifiers

echo "################  GCNN node2vec: pinned reproducible runs  ################"
$PY src/run_node2vec.py --split-seed 179776560 --init-seed 1366669488   # -> 91.39%
$PY src/run_node2vec.py --split-seed 941508052 --init-seed 769732007    # -> 90.83%

echo "################  GCNN node2vec: 20 unseeded runs (mean/std/max)  ################"
$PY src/run_node2vec.py --no-seed --runs 20

echo "################  PyG GCNConv / GATConv  (paper Table 1)  ################"
$PY src/run_pyg.py --model gcn      # GCNConv (fused + structural, 262-d)
$PY src/run_pyg.py --model gat      # GATConv

echo "################  DONE  ################"
