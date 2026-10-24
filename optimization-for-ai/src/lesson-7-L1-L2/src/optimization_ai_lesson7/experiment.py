"""Reproducible experiment for Optimization for AI — Lesson 7.

Synthetic regression data and polynomial models demonstrate overfitting,
train/validation generalization, and L1/L2 regularization.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

Penalty = Literal["none", "l1", "l2"]


@dataclass(frozen=True)
class PolynomialModel:
    degree: int
    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    penalty: Penalty
    regularization: float
    iterations: int = 0

    def predict(self, x: np.ndarray | list[float] | float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(x, dtype=float))
        features = polynomial_features(values, self.degree)
        standardized = (features - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients

    @property
    def coefficient_l1_norm(self) -> float:
        return float(np.sum(np.abs(self.coefficients)))

    @property
    def coefficient_l2_norm(self) -> float:
        return float(np.linalg.norm(self.coefficients))

    @property
    def nonzero_coefficients(self) -> int:
        return int(np.count_nonzero(np.abs(self.coefficients) > 1e-7))


def true_function(x: np.ndarray | float) -> np.ndarray:
    """Underlying smooth signal used to create illustrative data."""
    return np.sin(2.0 * np.pi * np.asarray(x, dtype=float))


def make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic train and validation datasets.

    Training: 18 noisy observations, RNG seed 42.
    Validation: 100 separate noisy observations, RNG seed 123.
    The data are synthetic and are not measurements about real people.
    """
    rng_train = np.random.default_rng(42)
    x_train = np.linspace(0.02, 0.98, 18)
    y_train = true_function(x_train) + rng_train.normal(0.0, 0.22, len(x_train))

    x_valid = (np.arange(100, dtype=float) + 0.5) / 100.0
    rng_valid = np.random.default_rng(123)
    y_valid = true_function(x_valid) + rng_valid.normal(0.0, 0.22, len(x_valid))
    return x_train, y_train, x_valid, y_valid


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Build powers of z=2x-1, excluding the intercept column."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    z = 2.0 * np.asarray(x, dtype=float) - 1.0
    return np.column_stack([z**power for power in range(1, degree + 1)])


def _standardize_fit(x: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = polynomial_features(x, degree)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (raw - mean) / scale, mean, scale


def fit_ridge(
    x: np.ndarray, y: np.ndarray, degree: int, regularization: float = 0.0
) -> PolynomialModel:
    """Fit polynomial regression minimizing MSE + lambda * ||beta||_2^2.

    The intercept is not penalized. The polynomial features are standardized
    using training data only. With regularization=0 this is ordinary least
    squares solved via the normal equations.
    """
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty 1D arrays of equal length")

    standardized, mean, scale = _standardize_fit(x, degree)
    design = np.column_stack([np.ones(len(x)), standardized])
    penalty = np.diag(np.r_[0.0, np.full(degree, regularization)])
    matrix = design.T @ design / len(y) + penalty
    vector = design.T @ y / len(y)
    params = np.linalg.solve(matrix, vector)
    return PolynomialModel(
        degree=degree, intercept=float(params[0]), coefficients=params[1:].copy(),
        feature_mean=mean, feature_scale=scale,
        penalty="l2" if regularization > 0 else "none", regularization=float(regularization)
    )


def fit_lasso(
    x: np.ndarray, y: np.ndarray, degree: int, regularization: float = 0.002,
    max_iter: int = 25000, tolerance: float = 1e-10,
) -> PolynomialModel:
    """Fit with proximal gradient for MSE + lambda * ||beta||_1.

    The intercept is not penalized. Soft-thresholding encourages some
    polynomial coefficients to become exactly zero.
    """
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty 1D arrays of equal length")

    standardized, mean, scale = _standardize_fit(x, degree)
    design = np.column_stack([np.ones(len(x)), standardized])
    n = len(y)
    # Gradient Lipschitz constant for mean squared error is 2*lambda_max(A'A/n).
    largest_eigenvalue = float(np.linalg.eigvalsh(design.T @ design / n).max())
    step = 0.95 / (2.0 * largest_eigenvalue)
    params = np.zeros(degree + 1, dtype=float)
    used_iterations = 0

    for iteration in range(max_iter):
        residual = design @ params - y
        gradient = 2.0 * (design.T @ residual) / n
        intercept = params[0] - step * gradient[0]
        raw_coefficients = params[1:] - step * gradient[1:]
        coeffs = np.sign(raw_coefficients) * np.maximum(
            np.abs(raw_coefficients) - step * regularization, 0.0
        )
        updated = np.r_[intercept, coeffs]
        used_iterations = iteration + 1
        if np.linalg.norm(updated - params) < tolerance:
            params = updated
            break
        params = updated

    return PolynomialModel(
        degree=degree, intercept=float(params[0]), coefficients=params[1:].copy(),
        feature_mean=mean, feature_scale=scale, penalty="l1",
        regularization=float(regularization), iterations=used_iterations
    )


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-empty shape")
    return float(np.mean((y_true - y_pred) ** 2))


def objective_value(model: PolynomialModel, x: np.ndarray, y: np.ndarray) -> float:
    """Compute training MSE plus the model's declared L1/L2 penalty."""
    mse = mean_squared_error(y, model.predict(x))
    if model.penalty == "l1":
        return mse + model.regularization * model.coefficient_l1_norm
    if model.penalty == "l2":
        return mse + model.regularization * model.coefficient_l2_norm**2
    return mse


@lru_cache(maxsize=1)
def build_models() -> dict[str, PolynomialModel]:
    """Build every model used in prompts, examples, and Manim scenes."""
    x_train, y_train, _, _ = make_dataset()
    return {
        "simple": fit_ridge(x_train, y_train, degree=3, regularization=0.01),
        "overfit": fit_ridge(x_train, y_train, degree=12, regularization=0.0),
        "ridge_0001": fit_ridge(x_train, y_train, degree=12, regularization=0.001),
        "ridge_001": fit_ridge(x_train, y_train, degree=12, regularization=0.01),
        "ridge_01": fit_ridge(x_train, y_train, degree=12, regularization=0.1),
        "lasso_0002": fit_lasso(x_train, y_train, degree=12, regularization=0.002),
    }


def evaluate_models() -> dict[str, dict[str, float | int | str]]:
    x_train, y_train, x_valid, y_valid = make_dataset()
    results: dict[str, dict[str, float | int | str]] = {}
    for name, model in build_models().items():
        results[name] = {
            "degree": model.degree,
            "penalty": model.penalty,
            "lambda": model.regularization,
            "train_mse": mean_squared_error(y_train, model.predict(x_train)),
            "validation_mse": mean_squared_error(y_valid, model.predict(x_valid)),
            "l1_norm": model.coefficient_l1_norm,
            "l2_norm": model.coefficient_l2_norm,
            "nonzero_coefficients": model.nonzero_coefficients,
        }
    return results
