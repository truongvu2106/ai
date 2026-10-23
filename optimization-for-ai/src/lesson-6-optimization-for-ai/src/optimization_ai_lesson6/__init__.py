"""Convexity and neural-network optimization teaching examples for Lesson 6."""

from .objectives import (
    Trajectory2D,
    convex_gradient,
    convex_objective,
    nonconvex_gradient,
    nonconvex_objective,
    nonconvex_stationary_points,
    run_gradient_descent_2d,
)

__all__ = [
    "Trajectory2D",
    "convex_gradient",
    "convex_objective",
    "nonconvex_gradient",
    "nonconvex_objective",
    "nonconvex_stationary_points",
    "run_gradient_descent_2d",
]
