"""Mathematical objectives and gradient-descent trajectories for Lesson 6.

The convex and non-convex functions are explicit teaching landscapes. Neither
is intended to reproduce the actual loss surface of a production neural network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

Point2D = tuple[float, float]


def convex_objective(x: float, y: float) -> float:
    """Strictly convex quadratic with unique global minimum at (0.5, -0.25)."""
    return float((x - 0.5) ** 2 + 0.5 * (y + 0.25) ** 2)


def convex_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of ``convex_objective`` as [dJ/dx, dJ/dy]."""
    return np.array([2.0 * (x - 0.5), y + 0.25], dtype=float)


def convex_hessian() -> np.ndarray:
    """Constant positive-definite Hessian for the convex teaching function."""
    return np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)


def nonconvex_objective(x: float, y: float) -> float:
    """Non-convex 2D teaching function with unequal minima and a saddle.

    f(x,y) = (x^2 - 1)^2 + 0.5*y^2 + 0.2*x

    This is a low-dimensional illustrative landscape, not a real neural-network
    loss surface.
    """
    return float((x * x - 1.0) ** 2 + 0.5 * y * y + 0.2 * x)


def nonconvex_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of ``nonconvex_objective`` as [df/dx, df/dy]."""
    return np.array([4.0 * x * (x * x - 1.0) + 0.2, y], dtype=float)


def nonconvex_hessian(x: float, y: float = 0.0) -> np.ndarray:
    """Hessian; it is diagonal for this separable teaching function."""
    del y  # The second derivative in y is constant and independent of y.
    return np.array([[12.0 * x * x - 4.0, 0.0], [0.0, 1.0]], dtype=float)


def nonconvex_stationary_points() -> dict[str, dict[str, float | str]]:
    """Return verified stationary points (computed from the cubic derivative)."""
    roots = np.roots([4.0, 0.0, -4.0, 0.2])  # 4*x^3 - 4*x + 0.2 = 0
    real_roots = sorted(float(r.real) for r in roots if abs(float(r.imag)) < 1e-9)
    points: dict[str, dict[str, float | str]] = {}
    for x in real_roots:
        h_xx = float(12.0 * x * x - 4.0)
        if h_xx < 0:
            kind = "saddle"
        elif nonconvex_objective(x, 0.0) < 0:
            kind = "global_minimum"
        else:
            kind = "local_minimum"
        key = kind if kind not in points else f"{kind}_{len(points) + 1}"
        points[key] = {
            "x": x,
            "y": 0.0,
            "value": nonconvex_objective(x, 0.0),
            "hessian_xx": h_xx,
            "hessian_yy": 1.0,
            "kind": kind,
        }
    return points


@dataclass(frozen=True)
class Trajectory2D:
    """Gradient-descent states; the initial state is included at index zero."""

    points: np.ndarray       # shape (steps + 1, 2)
    losses: np.ndarray       # full objective at every recorded state
    gradient_norms: np.ndarray
    converged: bool


def run_gradient_descent_2d(
    objective: Callable[[float, float], float],
    gradient_fn: Callable[[float, float], np.ndarray],
    initial_point: Point2D,
    learning_rate: float,
    *,
    max_steps: int = 1200,
    tolerance: float = 1e-6,
    divergence_limit: float = 1e5,
) -> Trajectory2D:
    """Run fixed-step gradient descent on a two-dimensional teaching objective."""
    point = np.asarray(initial_point, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("initial_point must contain two finite values")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    points = [point.copy()]
    losses = [float(objective(*point))]
    gradient_norms = [float(np.linalg.norm(gradient_fn(*point)))]
    converged = gradient_norms[-1] <= tolerance

    for _ in range(max_steps):
        if converged:
            break
        grad = np.asarray(gradient_fn(*points[-1]), dtype=float)
        next_point = points[-1] - learning_rate * grad
        if not np.all(np.isfinite(next_point)) or np.linalg.norm(next_point) > divergence_limit:
            break
        next_loss = float(objective(*next_point))
        next_grad_norm = float(np.linalg.norm(gradient_fn(*next_point)))
        if not np.isfinite(next_loss) or not np.isfinite(next_grad_norm):
            break
        points.append(next_point.copy())
        losses.append(next_loss)
        gradient_norms.append(next_grad_norm)
        converged = next_grad_norm <= tolerance

    return Trajectory2D(
        points=np.asarray(points, dtype=float),
        losses=np.asarray(losses, dtype=float),
        gradient_norms=np.asarray(gradient_norms, dtype=float),
        converged=converged,
    )
