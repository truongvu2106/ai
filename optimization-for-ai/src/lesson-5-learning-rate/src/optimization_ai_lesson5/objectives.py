"""Objective functions and simple gradient-descent trajectories for Lesson 5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

ScalarFunction = Callable[[float], float]


def quadratic_loss(theta: float) -> float:
    """Convex teaching objective J(theta) = (theta - 3)^2."""
    return float((theta - 3.0) ** 2)


def quadratic_gradient(theta: float) -> float:
    """Derivative of the convex teaching objective."""
    return float(2.0 * (theta - 3.0))


def nonconvex_loss(theta: float) -> float:
    """A 1D non-convex objective with several local minima.

    This is an illustrative landscape for teaching initialization sensitivity,
    not a model of a specific production neural network.
    """
    return float(0.1 * theta**2 + 2.0 * np.cos(1.5 * theta) + 0.12 * theta)


def nonconvex_gradient(theta: float) -> float:
    """Derivative of ``nonconvex_loss``."""
    return float(0.2 * theta - 3.0 * np.sin(1.5 * theta) + 0.12)


@dataclass(frozen=True)
class Trajectory:
    """One-dimensional gradient descent history, including the initial state."""

    theta: np.ndarray
    loss: np.ndarray
    gradient: np.ndarray
    converged: bool


def run_gradient_descent(
    objective: ScalarFunction,
    gradient_fn: ScalarFunction,
    initial_theta: float,
    learning_rate: float,
    *,
    max_steps: int = 200,
    tolerance: float = 1e-5,
    divergence_limit: float = 1e8,
) -> Trajectory:
    """Run scalar gradient descent and return all states.

    ``tolerance`` checks the absolute gradient magnitude. A divergence limit
    stops runs whose parameter becomes too large or non-finite.
    """
    if not np.isfinite(initial_theta):
        raise ValueError("initial_theta must be finite")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    theta_values = [float(initial_theta)]
    loss_values = [float(objective(float(initial_theta)))]
    grad_values = [float(gradient_fn(float(initial_theta)))]
    converged = abs(grad_values[-1]) <= tolerance

    for _ in range(max_steps):
        if converged:
            break
        theta = theta_values[-1]
        grad = grad_values[-1]
        next_theta = theta - learning_rate * grad
        if not np.isfinite(next_theta) or abs(next_theta) > divergence_limit:
            break
        next_loss = float(objective(float(next_theta)))
        next_grad = float(gradient_fn(float(next_theta)))
        if not np.isfinite(next_loss) or not np.isfinite(next_grad):
            break
        theta_values.append(float(next_theta))
        loss_values.append(next_loss)
        grad_values.append(next_grad)
        converged = abs(next_grad) <= tolerance

    return Trajectory(
        theta=np.asarray(theta_values, dtype=float),
        loss=np.asarray(loss_values, dtype=float),
        gradient=np.asarray(grad_values, dtype=float),
        converged=converged,
    )


def quadratic_run(initial_theta: float, learning_rate: float, steps: int) -> Trajectory:
    """Fixed-step quadratic run used by lesson examples and animations."""
    return run_gradient_descent(
        quadratic_loss, quadratic_gradient, initial_theta, learning_rate,
        max_steps=steps, tolerance=1e-12, divergence_limit=1e12,
    )
