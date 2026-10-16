"""Expectation, variance, standard deviation, and sample summaries."""

from collections.abc import Sequence
import math

import numpy as np


def _validated_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return array


def expectation_discrete(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """Compute E[X] for a discrete random variable with a PMF."""
    x = _validated_array(values, "values")
    p = _validated_array(probabilities, "probabilities")
    if x.size != p.size:
        raise ValueError("values and probabilities must have the same length.")
    if np.any(p < 0):
        raise ValueError("Probabilities cannot be negative.")
    if not math.isclose(float(np.sum(p)), 1.0, abs_tol=1e-9):
        raise ValueError("Probabilities must sum to 1.")
    return float(np.sum(x * p))


def variance_discrete(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """Compute Var(X) = E[(X - E[X])^2] for a discrete random variable."""
    mean = expectation_discrete(values, probabilities)
    x = _validated_array(values, "values")
    p = _validated_array(probabilities, "probabilities")
    return float(np.sum(((x - mean) ** 2) * p))


def standard_deviation_discrete(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """Compute the square root of a discrete random variable's variance."""
    return math.sqrt(variance_discrete(values, probabilities))


def population_mean(data: Sequence[float]) -> float:
    """Compute the mean of the supplied population or dataset."""
    return float(np.mean(_validated_array(data, "data")))


def sample_mean(data: Sequence[float]) -> float:
    """Compute the sample mean (the same arithmetic formula as population mean)."""
    return float(np.mean(_validated_array(data, "data")))


def population_variance(data: Sequence[float]) -> float:
    """Compute population variance with denominator N (ddof=0)."""
    return float(np.var(_validated_array(data, "data"), ddof=0))


def sample_variance(data: Sequence[float]) -> float:
    """Compute unbiased sample variance with denominator n - 1 (ddof=1)."""
    array = _validated_array(data, "data")
    if array.size < 2:
        raise ValueError("At least two observations are required for sample variance.")
    return float(np.var(array, ddof=1))


def population_standard_deviation(data: Sequence[float]) -> float:
    """Compute population standard deviation with denominator N."""
    return math.sqrt(population_variance(data))


def sample_standard_deviation(data: Sequence[float]) -> float:
    """Compute sample standard deviation from the unbiased sample variance."""
    return math.sqrt(sample_variance(data))
