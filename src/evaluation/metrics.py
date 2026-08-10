"""Classification metrics and the naive baselines the models must beat."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard binary-classification scorecard."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    support: int
    positive_rate: float
    predicted_positive_rate: float

    def as_dict(self) -> dict[str, Any]:
        """Return the scorecard as a plain dictionary."""
        return asdict(self)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute accuracy, precision, recall, F1 and ROC-AUC.

    ROC-AUC is returned as ``None`` when it is undefined (a single class present
    in ``y_true``, or no probabilities supplied), rather than being silently
    replaced by a misleading 0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    auc: float | None = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, np.asarray(y_proba, dtype=float)))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        support=int(len(y_true)),
        positive_rate=float(y_true.mean()),
        predicted_positive_rate=float(y_pred.mean()),
    )


def confusion(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> np.ndarray:
    """Return the 2x2 confusion matrix with a fixed [0, 1] label order."""
    return confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])


def roc_points(
    y_true: pd.Series | np.ndarray, y_proba: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return false-positive and true-positive rates for an ROC curve."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_proba, dtype=float))
    return fpr, tpr


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def majority_class_baseline(y_train: pd.Series, y_test: pd.Series) -> ClassificationMetrics:
    """Always predict whichever class dominated the training window.

    On daily equity data this is a genuinely hard benchmark: markets rise on
    slightly more than half of all sessions, so a constant "up" prediction
    already scores above 50%.
    """
    majority = int(pd.Series(y_train).mode().iloc[0])
    y_pred = np.full(len(y_test), majority, dtype=int)
    return compute_metrics(y_test, y_pred, y_proba=None)


def momentum_baseline(
    y_test: pd.Series, previous_day_up: pd.Series
) -> ClassificationMetrics:
    """Predict that tomorrow repeats today's direction.

    Args:
        y_test: True next-day labels over the test window.
        previous_day_up: 1 when the *current* session closed up, aligned to
            ``y_test``'s index.

    Returns:
        The scorecard for this rule.
    """
    aligned = pd.Series(previous_day_up).reindex(pd.Series(y_test).index).fillna(0)
    return compute_metrics(y_test, aligned.astype(int), y_proba=None)


def baseline_suite(
    y_train: pd.Series,
    y_test: pd.Series,
    daily_return_test: pd.Series,
) -> dict[str, ClassificationMetrics]:
    """Return every baseline evaluated on the same test window."""
    previous_up = (daily_return_test > 0).astype(int)
    return {
        "majority_class": majority_class_baseline(y_train, y_test),
        "momentum_persistence": momentum_baseline(y_test, previous_up),
    }
