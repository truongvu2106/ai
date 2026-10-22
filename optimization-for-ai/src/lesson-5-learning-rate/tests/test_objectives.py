import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_ai_lesson5 import (  # noqa: E402
    nonconvex_gradient, nonconvex_loss, quadratic_gradient, quadratic_loss,
    quadratic_run, run_gradient_descent,
)


def test_quadratic_objective_and_gradient():
    assert quadratic_loss(-2.0) == 25.0
    assert quadratic_gradient(-2.0) == -10.0
    assert quadratic_loss(3.0) == 0.0
    assert quadratic_gradient(3.0) == 0.0


def test_first_step_for_stable_learning_rate():
    trajectory = quadratic_run(-2.0, 0.25, steps=1)
    assert trajectory.theta[-1] == pytest.approx(0.5)
    assert trajectory.loss[-1] == pytest.approx(6.25)


def test_small_learning_rate_progresses_slowly():
    trajectory = quadratic_run(-2.0, 0.05, steps=1)
    assert trajectory.theta[-1] == pytest.approx(-1.5)
    assert trajectory.loss[-1] == pytest.approx(20.25)


def test_large_learning_rate_overshoots_and_increases_loss():
    trajectory = quadratic_run(-2.0, 1.1, steps=1)
    assert trajectory.theta[-1] == pytest.approx(9.0)
    assert trajectory.loss[-1] == pytest.approx(36.0)
    assert trajectory.loss[-1] > trajectory.loss[0]


def test_stable_rate_converges_under_gradient_tolerance():
    trajectory = run_gradient_descent(
        quadratic_loss, quadratic_gradient, -2.0, 0.25,
        max_steps=100, tolerance=1e-3,
    )
    assert trajectory.converged
    assert trajectory.theta[-1] == pytest.approx(3.0, abs=5e-4)


def test_too_large_rate_does_not_converge_within_short_limit():
    trajectory = run_gradient_descent(
        quadratic_loss, quadratic_gradient, -2.0, 1.1,
        max_steps=12, tolerance=1e-3, divergence_limit=1e8,
    )
    assert not trajectory.converged
    assert trajectory.loss[-1] > trajectory.loss[0]


def test_nonconvex_gradient_matches_finite_difference():
    x = 0.7
    eps = 1e-6
    numerical = (nonconvex_loss(x + eps) - nonconvex_loss(x - eps)) / (2 * eps)
    assert nonconvex_gradient(x) == pytest.approx(numerical, abs=1e-5)


@pytest.mark.parametrize("start, expected", [(-6.0, -6.04), (-4.0, -2.03), (1.0, 1.98)])
def test_nonconvex_initialization_reaches_expected_basin(start, expected):
    trajectory = run_gradient_descent(
        nonconvex_loss, nonconvex_gradient, start, 0.02,
        max_steps=300, tolerance=1e-5,
    )
    assert trajectory.converged
    assert trajectory.theta[-1] == pytest.approx(expected, abs=0.03)


def test_different_starts_reach_different_minima():
    left = run_gradient_descent(nonconvex_loss, nonconvex_gradient, -4.0, 0.02, max_steps=300, tolerance=1e-5)
    right = run_gradient_descent(nonconvex_loss, nonconvex_gradient, 1.0, 0.02, max_steps=300, tolerance=1e-5)
    assert not np.isclose(left.theta[-1], right.theta[-1], atol=0.1)


def test_invalid_learning_rate_rejected():
    with pytest.raises(ValueError):
        run_gradient_descent(quadratic_loss, quadratic_gradient, -2.0, 0.0)
