"""Maximum Likelihood Estimation for Bernoulli observations."""

from collections.abc import Sequence
import math

import numpy as np


def _binary_observations(observations: Sequence[int | bool]) -> np.ndarray:
    values = np.asarray(observations)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional sequence.")
    if not np.all(np.isin(values, [0, 1, False, True])):
        raise ValueError("Bernoulli observations must contain only 0/1 or False/True.")
    return values.astype(int)


def bernoulli_mle(observations: Sequence[int | bool]) -> float:
    """Estimate Bernoulli success probability by MLE: p_hat = successes / n."""
    values = _binary_observations(observations)
    return float(np.mean(values))


def bernoulli_log_likelihood(observations: Sequence[int | bool], p: float) -> float:
    """Compute the Bernoulli sequence log-likelihood for a proposed p in [0, 1]."""
    values = _binary_observations(observations)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1.")

    successes = int(np.sum(values))
    failures = int(values.size - successes)
    if (p == 0 and successes > 0) or (p == 1 and failures > 0):
        return float("-inf")
    log_likelihood = 0.0
    if successes:
        log_likelihood += successes * math.log(p)
    if failures:
        log_likelihood += failures * math.log1p(-p)
    return log_likelihood


def bernoulli_likelihood(observations: Sequence[int | bool], p: float) -> float:
    """Compute likelihood for the observed Bernoulli sequence."""
    log_likelihood = bernoulli_log_likelihood(observations, p)
    return 0.0 if log_likelihood == float("-inf") else math.exp(log_likelihood)
