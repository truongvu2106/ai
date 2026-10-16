"""Gaussian distribution helpers and a Central Limit Theorem simulation."""

from collections.abc import Sequence
import math

import numpy as np


def normal_pdf(x: float | Sequence[float] | np.ndarray, mean: float = 0.0, std: float = 1.0):
    """Evaluate the Gaussian probability density function at x."""
    if not math.isfinite(mean):
        raise ValueError("mean must be finite.")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be a positive finite number.")
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers.")
    density = np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * math.sqrt(2.0 * math.pi))
    return float(density) if density.ndim == 0 else density


def z_score(x: float, mean: float, std: float) -> float:
    """Standardize x: z = (x - mean) / std."""
    if not all(math.isfinite(value) for value in (x, mean, std)):
        raise ValueError("x, mean, and std must be finite.")
    if std <= 0:
        raise ValueError("std must be greater than 0.")
    return (x - mean) / std


def simulate_clt(
    population: Sequence[float] | np.ndarray,
    sample_size: int,
    n_samples: int = 5_000,
    seed: int = 42,
) -> np.ndarray:
    """Return sample means from repeated samples drawn with replacement.

    This simulation illustrates the CLT using an empirical population. Under
    suitable conditions, the sampling distribution of sample means becomes
    more nearly normal as sample_size increases.
    """
    population_array = np.asarray(population, dtype=float)
    if population_array.ndim != 1 or population_array.size == 0:
        raise ValueError("population must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(population_array)):
        raise ValueError("population must contain only finite values.")
    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer.")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    rng = np.random.default_rng(seed)
    samples = rng.choice(population_array, size=(n_samples, sample_size), replace=True)
    return samples.mean(axis=1)
