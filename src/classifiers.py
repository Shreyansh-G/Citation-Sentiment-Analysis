"""Downstream classical classifiers on the extracted GNN embeddings.

Mirrors the "#Scores" section of the TF notebooks. Each classifier is fit on
the train split and reported on validation + test. Pass ``gridsearch=True`` to
tune with the same parameter grids the notebooks used (slower).
"""

from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from metrics import report_metrics


def _make_adaboost(seed: int) -> AdaBoostClassifier:
    """AdaBoost with SAMME, tolerant of the param's removal in sklearn>=1.6."""
    import inspect

    kwargs = dict(n_estimators=100, learning_rate=1.0, random_state=seed)
    if "algorithm" in inspect.signature(AdaBoostClassifier).parameters:
        kwargs["algorithm"] = "SAMME"
    return AdaBoostClassifier(**kwargs)


def _registry(seed: int):
    """Return {name: (estimator, param_grid)}; grid is None when not tuned."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            None,
        ),
        "DecisionTree": (
            DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=seed),
            {"max_depth": [5, 10, 15, 20], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=seed),
            {"n_estimators": [50, 100, 150], "max_depth": [10, 20, 30, None],
             "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "bootstrap": [True, False]},
        ),
        "SVM": (
            SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed),
            {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf", "poly", "sigmoid"], "gamma": ["scale", "auto"]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(n_estimators=100, random_state=seed),
            {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20, 30],
             "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"]},
        ),
        "AdaBoost": (
            _make_adaboost(seed),
            None,
        ),
        "SGD": (
            SGDClassifier(alpha=1e-4, max_iter=1000, tol=1e-3, random_state=seed),
            None,
        ),
        "Voting(LR+SGD)": (
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000)),
                    ("sgd", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=seed)),
                ],
                voting="soft",
            ),
            None,
        ),
    }


def run_classifiers(
    X_train, y_train, X_val, y_val, X_test, y_test,
    gridsearch: bool = False, seed: int = 42,
) -> Dict[str, dict]:
    """Fit every classifier and report val/test metrics. Returns test metrics."""
    results: Dict[str, dict] = {}
    for name, (estimator, grid) in _registry(seed).items():
        print(f"\n################ {name} ################")
        if gridsearch and grid is not None:
            search = GridSearchCV(estimator, grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
        else:
            model = estimator.fit(X_train, y_train)

        report_metrics(f"{name} [val]", y_val, model.predict(X_val))
        results[name] = report_metrics(f"{name} [test]", y_test, model.predict(X_test))
    return results
