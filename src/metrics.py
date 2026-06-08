"""Uniform metric reporting (macro / micro / weighted)."""

from __future__ import annotations

from typing import Dict, Iterable

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def report_metrics(
    name: str,
    y_true,
    y_pred,
    averages: Iterable[str] = ("macro", "micro", "weighted"),
    show_report: bool = False,
) -> Dict[str, float]:
    """Print and return accuracy plus P/R/F1 under each averaging scheme."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    results: Dict[str, float] = {"accuracy": acc}
    for avg in averages:
        p = precision_score(y_true, y_pred, average=avg, zero_division=0)
        r = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f = f1_score(y_true, y_pred, average=avg, zero_division=0)
        print(f"{avg:>8} | Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}")
        results[f"{avg}_precision"] = p
        results[f"{avg}_recall"] = r
        results[f"{avg}_f1"] = f
    if show_report:
        print(classification_report(y_true, y_pred, zero_division=0))
    return results
