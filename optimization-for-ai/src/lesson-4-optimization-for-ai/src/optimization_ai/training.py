"""Mini-batch training utilities for teaching optimizer behavior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .data import mse, mse_gradient, validate_xy
from .optimizers import Optimizer

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TrainingHistory:
    """Training diagnostics recorded at initialization and after each epoch."""
    params: FloatArray
    full_loss: FloatArray
    batch_sizes: IntArray
    gradient_norms: FloatArray
    epoch_end_params: FloatArray

    @property
    def final_params(self) -> FloatArray:
        return self.epoch_end_params[-1].copy()

    @property
    def final_loss(self) -> float:
        return float(self.full_loss[-1])


def train_minibatch(
    x: FloatArray,
    y: FloatArray,
    optimizer: Optimizer,
    *,
    batch_size: int = 8,
    epochs: int = 60,
    seed: int = 42,
    initial_params: tuple[float, float] = (0.0, 0.0),
) -> TrainingHistory:
    """Train y_hat=w*x+b using shuffled mini-batches.

    Full-data MSE is recorded at epoch boundaries. `params` records every update;
    `epoch_end_params` includes initial parameters plus one state per epoch.
    """
    xa, ya = validate_xy(x, y)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= xa.size:
        raise ValueError("batch_size must be an integer between 1 and n_samples.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    params = np.asarray(initial_params, dtype=float)
    if params.shape != (2,) or not np.isfinite(params).all():
        raise ValueError("initial_params must be a finite (w, b) pair.")

    rng = np.random.default_rng(seed)
    all_params = [params.copy()]
    epoch_params = [params.copy()]
    loss_history = [mse(xa, ya, params)]
    sizes: list[int] = []
    grad_norms: list[float] = []

    for _epoch in range(epochs):
        permutation = rng.permutation(xa.size)
        for start in range(0, xa.size, batch_size):
            indices = permutation[start : start + batch_size]
            grad = mse_gradient(xa[indices], ya[indices], params)
            grad_norms.append(float(np.linalg.norm(grad)))
            sizes.append(int(indices.size))
            params = optimizer.step(params, grad)
            if not np.isfinite(params).all():
                raise FloatingPointError("Training diverged to non-finite parameters.")
            all_params.append(params.copy())
        epoch_params.append(params.copy())
        loss_history.append(mse(xa, ya, params))

    return TrainingHistory(
        params=np.asarray(all_params, dtype=float),
        full_loss=np.asarray(loss_history, dtype=float),
        batch_sizes=np.asarray(sizes, dtype=np.int64),
        gradient_norms=np.asarray(grad_norms, dtype=float),
        epoch_end_params=np.asarray(epoch_params, dtype=float),
    )


def optimizer_suite(seed: int = 42) -> dict[str, tuple[Optimizer, int]]:
    """Return a reproducible teaching configuration for optimizer comparisons.

    Every method uses mini-batches; the comparison isolates the update rule while
    keeping dataset, batch size, epochs, and sampling seed consistent.
    """
    from .optimizers import AdaGrad, Adam, Momentum, RMSProp, SGD

    return {
        "Mini-batch SGD": (SGD(learning_rate=0.03), 12),
        "Momentum": (Momentum(learning_rate=0.03, beta=0.85), 12),
        "AdaGrad": (AdaGrad(learning_rate=0.12), 12),
        "RMSProp": (RMSProp(learning_rate=0.025, beta=0.9), 12),
        "Adam": (Adam(learning_rate=0.03), 12),
    }
