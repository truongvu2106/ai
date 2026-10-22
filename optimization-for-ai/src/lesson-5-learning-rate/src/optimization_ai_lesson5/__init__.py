"""Lesson 5: Learning Rate, Convergence, and Local Minima."""
from .objectives import (
    Trajectory,
    nonconvex_gradient,
    nonconvex_loss,
    quadratic_gradient,
    quadratic_loss,
    quadratic_run,
    run_gradient_descent,
)

__all__ = [
    "Trajectory", "nonconvex_gradient", "nonconvex_loss",
    "quadratic_gradient", "quadratic_loss", "quadratic_run",
    "run_gradient_descent",
]
