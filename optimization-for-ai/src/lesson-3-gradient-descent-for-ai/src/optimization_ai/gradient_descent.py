"""Gradient Descent and Stochastic Gradient Descent for simple linear regression.

The model is y_hat = w*x + b and the objective is mean squared error (MSE).
These functions intentionally use NumPy and explicit formulas for teaching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validated_data(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert inputs to 1D float arrays and validate their shapes and values."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if x_array.size == 0 or x_array.size != y_array.size:
        raise ValueError("x and y must have the same non-zero length.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x and y must contain only finite values.")
    return x_array, y_array


def predict(x: ArrayLike, w: float, b: float) -> FloatArray:
    """Predict targets using the linear model y_hat = w*x + b."""
    x_array = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x_array)) or not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("Inputs and parameters must be finite.")
    return w * x_array + b


def mean_squared_error(x: ArrayLike, y: ArrayLike, w: float, b: float) -> float:
    """Return the average squared prediction error for the whole dataset."""
    x_array, y_array = _validated_data(x, y)
    errors = predict(x_array, w, b) - y_array
    return float(np.mean(errors**2))


def batch_gradient(x: ArrayLike, y: ArrayLike, w: float, b: float) -> tuple[float, float]:
    """Calculate the exact MSE gradient using every training observation.

    For MSE = mean((w*x + b - y)**2):
    dMSE/dw = 2*mean((prediction - y)*x)
    dMSE/db = 2*mean(prediction - y)
    """
    x_array, y_array = _validated_data(x, y)
    errors = predict(x_array, w, b) - y_array
    dw = float(2.0 * np.mean(errors * x_array))
    db = float(2.0 * np.mean(errors))
    return dw, db


def stochastic_gradient(
    x_i: float, y_i: float, w: float, b: float
) -> tuple[float, float]:
    """Calculate the gradient of one observation's squared error."""
    if not all(np.isfinite(v) for v in (x_i, y_i, w, b)):
        raise ValueError("Inputs and parameters must be finite.")
    error = w * x_i + b - y_i
    return float(2.0 * error * x_i), float(2.0 * error)


@dataclass(frozen=True)
class TrainingResult:
    """Training trajectory, including the initial parameter values."""

    w_history: FloatArray
    b_history: FloatArray
    loss_history: FloatArray
    sample_indices: NDArray[np.int64] | None = None

    @property
    def final_w(self) -> float:
        return float(self.w_history[-1])

    @property
    def final_b(self) -> float:
        return float(self.b_history[-1])

    @property
    def final_loss(self) -> float:
        return float(self.loss_history[-1])


def train_batch_gd(
    x: ArrayLike,
    y: ArrayLike,
    *,
    learning_rate: float = 0.05,
    steps: int = 100,
    initial_w: float = 0.0,
    initial_b: float = 0.0,
) -> TrainingResult:
    """Train with full-batch Gradient Descent and record each update."""
    x_array, y_array = _validated_data(x, y)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not np.isfinite(initial_w) or not np.isfinite(initial_b):
        raise ValueError("Initial parameters must be finite.")

    w, b = float(initial_w), float(initial_b)
    w_history = [w]
    b_history = [b]
    loss_history = [mean_squared_error(x_array, y_array, w, b)]

    for _ in range(steps):
        dw, db = batch_gradient(x_array, y_array, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        w_history.append(w)
        b_history.append(b)
        loss_history.append(mean_squared_error(x_array, y_array, w, b))

    return TrainingResult(
        np.asarray(w_history),
        np.asarray(b_history),
        np.asarray(loss_history),
    )


def train_sgd(
    x: ArrayLike,
    y: ArrayLike,
    *,
    learning_rate: float = 0.005,
    steps: int = 3000,
    initial_w: float = 0.0,
    initial_b: float = 0.0,
    seed: int = 42,
) -> TrainingResult:
    """Train with single-observation SGD; select one random observation per update.

    The reported loss is still evaluated against the complete dataset after each
    update so it can be compared with the batch-GD objective.
    """
    x_array, y_array = _validated_data(x, y)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not np.isfinite(initial_w) or not np.isfinite(initial_b):
        raise ValueError("Initial parameters must be finite.")

    rng = np.random.default_rng(seed)
    w, b = float(initial_w), float(initial_b)
    w_history = [w]
    b_history = [b]
    loss_history = [mean_squared_error(x_array, y_array, w, b)]
    sample_indices: list[int] = []

    for _ in range(steps):
        index = int(rng.integers(0, x_array.size))
        dw, db = stochastic_gradient(x_array[index], y_array[index], w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        sample_indices.append(index)
        w_history.append(w)
        b_history.append(b)
        loss_history.append(mean_squared_error(x_array, y_array, w, b))

    return TrainingResult(
        np.asarray(w_history),
        np.asarray(b_history),
        np.asarray(loss_history),
        np.asarray(sample_indices, dtype=np.int64),
    )
