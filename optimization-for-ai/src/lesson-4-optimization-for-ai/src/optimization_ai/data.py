"""Small NumPy helpers for a reproducible linear-regression example."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def validate_xy(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert x/y to finite, non-empty one-dimensional float arrays."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if xa.size == 0 or xa.size != ya.size:
        raise ValueError("x and y must have equal, non-zero length.")
    if not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("x and y must contain only finite values.")
    return xa, ya


def make_regression_data(n: int = 96, seed: int = 42) -> tuple[FloatArray, FloatArray]:
    """Create illustrative normalized study-hour features and noisy score targets.

    The underlying relationship is y = 2*x + 1 plus reproducible Gaussian noise.
    x is normalized to [-2, 2] so the optimizer comparisons remain numerically
    stable. These are synthetic teaching data, not real exam-score evidence.
    """
    if n < 8:
        raise ValueError("n must be at least 8 to demonstrate mini-batches.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n, dtype=float)
    noise = rng.normal(loc=0.0, scale=0.55, size=n)
    y = 2.0 * x + 1.0 + noise
    return x, y


def predict(x: ArrayLike, params: ArrayLike) -> FloatArray:
    """Predict with theta=[w,b] and y_hat = w*x+b."""
    xa = np.asarray(x, dtype=float)
    theta = np.asarray(params, dtype=float)
    if xa.ndim != 1 or theta.shape != (2,):
        raise ValueError("x must be 1D and params must be [w, b].")
    if not np.isfinite(xa).all() or not np.isfinite(theta).all():
        raise ValueError("Inputs and parameters must be finite.")
    return theta[0] * xa + theta[1]


def mse(x: ArrayLike, y: ArrayLike, params: ArrayLike) -> float:
    """Return full-dataset mean squared error."""
    xa, ya = validate_xy(x, y)
    pred = predict(xa, params)
    return float(np.mean((pred - ya) ** 2))


def mse_gradient(x: ArrayLike, y: ArrayLike, params: ArrayLike) -> FloatArray:
    """Gradient of mean((w*x+b-y)**2), averaged over the supplied observations."""
    xa, ya = validate_xy(x, y)
    theta = np.asarray(params, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("params must be a finite array [w, b].")
    error = theta[0] * xa + theta[1] - ya
    return np.array([2.0 * np.mean(error * xa), 2.0 * np.mean(error)], dtype=float)
