import numpy as np
import pytest

from optimization_ai import (
    batch_gradient,
    mean_squared_error,
    stochastic_gradient,
    train_batch_gd,
    train_sgd,
)

X = np.array([1.0, 2.0, 3.0])
Y = np.array([3.0, 5.0, 7.0])


def test_perfect_fit_has_zero_mse() -> None:
    assert mean_squared_error(X, Y, w=2.0, b=1.0) == pytest.approx(0.0)


def test_initial_mse_is_correct() -> None:
    assert mean_squared_error(X, Y, w=0.0, b=0.0) == pytest.approx(83.0 / 3.0)


def test_initial_batch_gradient_is_correct() -> None:
    dw, db = batch_gradient(X, Y, w=0.0, b=0.0)
    assert dw == pytest.approx(-68.0 / 3.0)
    assert db == pytest.approx(-10.0)


def test_one_batch_update_reduces_loss() -> None:
    first_w, first_b = 0.0 - 0.05 * (-68.0 / 3.0), 0.0 - 0.05 * (-10.0)
    assert mean_squared_error(X, Y, first_w, first_b) < mean_squared_error(X, Y, 0.0, 0.0)


def test_batch_gd_converges_on_illustrative_linear_data() -> None:
    result = train_batch_gd(X, Y, learning_rate=0.05, steps=100)
    assert result.final_loss < 0.001
    assert result.final_w == pytest.approx(2.0, abs=0.03)
    assert result.final_b == pytest.approx(1.0, abs=0.05)


def test_sgd_is_reproducible_with_seed() -> None:
    first = train_sgd(X, Y, learning_rate=0.005, steps=500, seed=19)
    second = train_sgd(X, Y, learning_rate=0.005, steps=500, seed=19)
    np.testing.assert_allclose(first.w_history, second.w_history)
    np.testing.assert_allclose(first.b_history, second.b_history)


def test_sgd_converges_with_enough_updates() -> None:
    result = train_sgd(X, Y, learning_rate=0.005, steps=3000, seed=42)
    assert result.final_loss < 0.001
    assert result.final_w == pytest.approx(2.0, abs=0.03)
    assert result.final_b == pytest.approx(1.0, abs=0.06)


def test_sgd_records_sample_indices() -> None:
    result = train_sgd(X, Y, steps=12, seed=3)
    assert result.sample_indices is not None
    assert len(result.sample_indices) == 12
    assert np.all((result.sample_indices >= 0) & (result.sample_indices < len(X)))


def test_invalid_shapes_raise_value_error() -> None:
    with pytest.raises(ValueError):
        mean_squared_error([1.0, 2.0], [1.0], 1.0, 0.0)


def test_non_positive_learning_rate_raises() -> None:
    with pytest.raises(ValueError):
        train_batch_gd(X, Y, learning_rate=0)
    with pytest.raises(ValueError):
        train_sgd(X, Y, learning_rate=-0.1)


def test_stochastic_gradient_formula() -> None:
    # Single example: x=2, y=5, w=0, b=0 -> error=-5.
    dw, db = stochastic_gradient(2.0, 5.0, 0.0, 0.0)
    assert dw == pytest.approx(-20.0)
    assert db == pytest.approx(-10.0)
