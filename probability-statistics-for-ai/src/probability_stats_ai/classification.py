"""Binary classification metrics implemented for learning purposes."""

from collections.abc import Sequence
import math

import numpy as np


def _binary_labels(values: Sequence[int | bool], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if not np.all(np.isin(array, [0, 1, False, True])):
        raise ValueError(f"{name} must contain only binary labels 0 and 1.")
    return array.astype(int)


def confusion_matrix_binary(
    y_true: Sequence[int | bool], y_pred: Sequence[int | bool]
) -> dict[str, int]:
    """Return TP, FP, TN, and FN for binary labels where 1 is positive."""
    actual = _binary_labels(y_true, "y_true")
    predicted = _binary_labels(y_pred, "y_pred")
    if actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same length.")
    return {
        "tp": int(np.sum((actual == 1) & (predicted == 1))),
        "fp": int(np.sum((actual == 0) & (predicted == 1))),
        "tn": int(np.sum((actual == 0) & (predicted == 0))),
        "fn": int(np.sum((actual == 1) & (predicted == 0))),
    }


def classification_metrics(
    y_true: Sequence[int | bool], y_pred: Sequence[int | bool]
) -> dict[str, float | int]:
    """Calculate accuracy, precision, recall, specificity, and F1-score."""
    counts = confusion_matrix_binary(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def binary_cross_entropy(
    y_true: Sequence[int | bool], probabilities: Sequence[float], eps: float = 1e-15
) -> float:
    """Compute mean binary cross-entropy using natural logarithms."""
    actual = _binary_labels(y_true, "y_true")
    predicted = np.asarray(probabilities, dtype=float)
    if predicted.ndim != 1 or predicted.size != actual.size:
        raise ValueError("probabilities must be one-dimensional and match y_true length.")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("probabilities must be finite values between 0 and 1.")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5.")
    clipped = np.clip(predicted, eps, 1.0 - eps)
    loss = -(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))
    return float(np.mean(loss))


def roc_auc_score_binary(
    y_true: Sequence[int | bool], scores: Sequence[float]
) -> float:
    """Compute ROC-AUC from ranking scores, handling tied scores correctly.

    This O(n_positive * n_negative) implementation is intentionally simple
    for small teaching examples. For large datasets, use a tested library
    implementation such as sklearn.metrics.roc_auc_score.
    """
    actual = _binary_labels(y_true, "y_true")
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or score_array.size != actual.size:
        raise ValueError("scores must be one-dimensional and match y_true length.")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite numbers.")
    positives = score_array[actual == 1]
    negatives = score_array[actual == 0]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("ROC-AUC requires at least one positive and one negative label.")

    comparisons = positives[:, None] - negatives[None, :]
    wins = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)


def calibration_table(
    y_true: Sequence[int | bool],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Summarize predicted probabilities versus observed positive rates by bin.

    A well-calibrated model has observed positive rates close to mean predicted
    probabilities within bins. Empty bins are omitted from the returned table.
    """
    actual = _binary_labels(y_true, "y_true")
    predicted = np.asarray(probabilities, dtype=float)
    if predicted.ndim != 1 or predicted.size != actual.size:
        raise ValueError("probabilities must be one-dimensional and match y_true length.")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("probabilities must be finite values between 0 and 1.")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    bin_indices = np.minimum((predicted * n_bins).astype(int), n_bins - 1)
    table: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(np.sum(mask))
        if count == 0:
            continue
        table.append({
            "bin_lower": bin_index / n_bins,
            "bin_upper": (bin_index + 1) / n_bins,
            "count": count,
            "mean_predicted_probability": float(np.mean(predicted[mask])),
            "observed_positive_rate": float(np.mean(actual[mask])),
        })
    return table


def expected_calibration_error(
    y_true: Sequence[int | bool],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute binned Expected Calibration Error (ECE).

    ECE is a binning-dependent summary, not a complete description of
    calibration. Different bin counts can produce different values.
    """
    actual = _binary_labels(y_true, "y_true")
    table = calibration_table(actual, probabilities, n_bins=n_bins)
    total = actual.size
    return float(sum(
        row["count"] / total
        * abs(row["observed_positive_rate"] - row["mean_predicted_probability"])
        for row in table
    ))
