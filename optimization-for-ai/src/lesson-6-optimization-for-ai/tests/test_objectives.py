"""Numerical checks for Lesson 6 teaching objectives."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optimization_ai_lesson6.objectives import (  # noqa: E402
    convex_gradient, convex_hessian, convex_objective, nonconvex_gradient,
    nonconvex_hessian, nonconvex_objective, nonconvex_stationary_points,
    run_gradient_descent_2d,
)


def test_convex_objective_unique_minimum_is_zero():
    assert convex_objective(0.5, -0.25) == pytest.approx(0.0)
    assert np.allclose(convex_gradient(0.5, -0.25), [0.0, 0.0])
    assert np.all(np.linalg.eigvalsh(convex_hessian()) > 0)


def test_convexity_jensen_example():
    a, b, alpha = -1.0, 2.5, 0.4
    z = alpha * a + (1.0 - alpha) * b
    left = (z - 0.5) ** 2
    right = alpha * (a - 0.5) ** 2 + (1.0 - alpha) * (b - 0.5) ** 2
    assert left == pytest.approx(0.36)
    assert right == pytest.approx(3.30)
    assert left <= right


@pytest.mark.parametrize("start", [(-2.0, 1.5), (2.0, 1.25), (-1.0, -1.5)])
def test_convex_gradient_descent_reaches_same_minimum(start):
    run = run_gradient_descent_2d(convex_objective, convex_gradient, start, 0.2, max_steps=120, tolerance=1e-6)
    assert run.converged
    assert run.points[-1, 0] == pytest.approx(0.5, abs=1e-5)
    assert run.points[-1, 1] == pytest.approx(-0.25, abs=1e-5)
    assert run.losses[-1] < run.losses[0]


def test_nonconvex_global_and_local_minima_are_distinct():
    stationary = nonconvex_stationary_points()
    global_min = stationary["global_minimum"]
    local_min = stationary["local_minimum"]
    saddle = stationary["saddle"]
    assert float(global_min["value"]) == pytest.approx(-0.2024404343, abs=1e-6)
    assert float(local_min["value"]) == pytest.approx(0.1974341529, abs=1e-6)
    assert float(saddle["value"]) == pytest.approx(1.0050062815, abs=1e-6)
    assert float(global_min["value"]) < float(local_min["value"]) < float(saddle["value"])


def test_saddle_has_opposite_curvature_signs():
    saddle = nonconvex_stationary_points()["saddle"]
    hessian = nonconvex_hessian(float(saddle["x"]))
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert eigenvalues[0] < 0 < eigenvalues[1]
    assert np.linalg.norm(nonconvex_gradient(float(saddle["x"]), 0.0)) < 1e-8


@pytest.mark.parametrize("start, expected_x", [((-1.8, 1.4), -1.0241203), ((1.8, 1.2), 0.97399435)])
def test_nonconvex_initialization_reaches_expected_valley(start, expected_x):
    run = run_gradient_descent_2d(nonconvex_objective, nonconvex_gradient, start, 0.02, max_steps=1300, tolerance=1e-6)
    assert run.converged
    assert run.points[-1, 0] == pytest.approx(expected_x, abs=1e-4)
    assert abs(run.points[-1, 1]) < 1e-5


def test_gradient_matches_finite_difference_convex():
    x, y, eps = 1.2, -0.8, 1e-6
    num_dx = (convex_objective(x + eps, y) - convex_objective(x - eps, y)) / (2 * eps)
    num_dy = (convex_objective(x, y + eps) - convex_objective(x, y - eps)) / (2 * eps)
    assert convex_gradient(x, y) == pytest.approx([num_dx, num_dy], abs=1e-6)


def test_gradient_matches_finite_difference_nonconvex():
    x, y, eps = 0.7, -0.4, 1e-6
    num_dx = (nonconvex_objective(x + eps, y) - nonconvex_objective(x - eps, y)) / (2 * eps)
    num_dy = (nonconvex_objective(x, y + eps) - nonconvex_objective(x, y - eps)) / (2 * eps)
    assert nonconvex_gradient(x, y) == pytest.approx([num_dx, num_dy], abs=1e-6)


def test_invalid_learning_rate_rejected():
    with pytest.raises(ValueError):
        run_gradient_descent_2d(convex_objective, convex_gradient, (1, 2), 0.0)
