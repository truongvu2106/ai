"""Explicit NumPy implementations of common first-order optimizers.

All optimizers minimize an objective by updating params -= an optimizer-specific
transformation of the gradient. The implementations are intentionally compact
for teaching, not replacements for production deep-learning frameworks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _gradient(grad: ArrayLike) -> FloatArray:
    result = np.asarray(grad, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("grad must be a non-empty finite 1D array.")
    return result


class Optimizer(ABC):
    """Base class for an optimizer that owns update state."""

    def __init__(self, learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        self.learning_rate = float(learning_rate)

    @abstractmethod
    def step(self, params: ArrayLike, grad: ArrayLike) -> FloatArray:
        """Return updated parameters; never mutate the caller's array."""


class SGD(Optimizer):
    """Plain SGD update: theta <- theta - lr * gradient."""

    def step(self, params: ArrayLike, grad: ArrayLike) -> FloatArray:
        theta = np.asarray(params, dtype=float)
        g = _gradient(grad)
        if theta.shape != g.shape or not np.isfinite(theta).all():
            raise ValueError("params and grad must have identical valid shapes.")
        return theta - self.learning_rate * g


class Momentum(Optimizer):
    """Classical momentum: velocity <- beta*velocity + gradient."""

    def __init__(self, learning_rate: float = 0.03, beta: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        self.beta = float(beta)
        self.velocity: FloatArray | None = None

    def step(self, params: ArrayLike, grad: ArrayLike) -> FloatArray:
        theta = np.asarray(params, dtype=float)
        g = _gradient(grad)
        if theta.shape != g.shape or not np.isfinite(theta).all():
            raise ValueError("params and grad must have identical valid shapes.")
        if self.velocity is None:
            self.velocity = np.zeros_like(g)
        if self.velocity.shape != g.shape:
            raise ValueError("Gradient shape cannot change after optimizer initialization.")
        self.velocity = self.beta * self.velocity + g
        return theta - self.learning_rate * self.velocity


class AdaGrad(Optimizer):
    """AdaGrad: accumulate squared gradients to scale each parameter's step."""

    def __init__(self, learning_rate: float = 0.3, eps: float = 1e-8) -> None:
        super().__init__(learning_rate)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")
        self.eps = float(eps)
        self.accumulated_sq_grad: FloatArray | None = None

    def step(self, params: ArrayLike, grad: ArrayLike) -> FloatArray:
        theta = np.asarray(params, dtype=float)
        g = _gradient(grad)
        if theta.shape != g.shape or not np.isfinite(theta).all():
            raise ValueError("params and grad must have identical valid shapes.")
        if self.accumulated_sq_grad is None:
            self.accumulated_sq_grad = np.zeros_like(g)
        if self.accumulated_sq_grad.shape != g.shape:
            raise ValueError("Gradient shape cannot change after optimizer initialization.")
        self.accumulated_sq_grad += g**2
        return theta - self.learning_rate * g / (np.sqrt(self.accumulated_sq_grad) + self.eps)


class RMSProp(Optimizer):
    """RMSProp: exponentially average squared gradients."""

    def __init__(self, learning_rate: float = 0.03, beta: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")
        self.beta = float(beta)
        self.eps = float(eps)
        self.avg_sq_grad: FloatArray | None = None

    def step(self, params: ArrayLike, grad: ArrayLike) -> FloatArray:
        theta = np.asarray(params, dtype=float)
        g = _gradient(grad)
        if theta.shape != g.shape or not np.isfinite(theta).all():
            raise ValueError("params and grad must have identical valid shapes.")
        if self.avg_sq_grad is None:
            self.avg_sq_grad = np.zeros_like(g)
        if self.avg_sq_grad.shape != g.shape:
            raise ValueError("Gradient shape cannot change after optimizer initialization.")
        self.avg_sq_grad = self.beta * self.avg_sq_grad + (1.0 - self.beta) * (g**2)
        return theta - self.learning_rate * g / (np.sqrt(self.avg_sq_grad) + self.eps)


class Adam(Optimizer):
    """Adam with first/second moment estimates and bias correction."""

    def __init__(
        self,
        learning_rate: float = 0.03,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.m: FloatArray | None = None
        self.v: FloatArray | None = None
        self.t = 0

    def step(self, params: ArrayLike, grad: ArrayLike) -> FloatArray:
        theta = np.asarray(params, dtype=float)
        g = _gradient(grad)
        if theta.shape != g.shape or not np.isfinite(theta).all():
            raise ValueError("params and grad must have identical valid shapes.")
        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)
        if self.m.shape != g.shape or self.v is None or self.v.shape != g.shape:
            raise ValueError("Gradient shape cannot change after optimizer initialization.")
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return theta - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
