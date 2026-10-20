import numpy as np
import pytest

from optimization_ai import (
    AdaGrad, Adam, Momentum, RMSProp, SGD,
    make_regression_data, mse, mse_gradient, train_minibatch,
)


def test_regression_data_is_reproducible_and_finite():
    x1, y1 = make_regression_data(n=32, seed=7)
    x2, y2 = make_regression_data(n=32, seed=7)
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)
    assert np.isfinite(x1).all() and np.isfinite(y1).all()


def test_mse_and_gradient_for_known_case():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 5.0, 7.0])
    assert mse(x, y, [0.0, 0.0]) == pytest.approx(83.0 / 3.0)
    np.testing.assert_allclose(mse_gradient(x, y, [0.0, 0.0]), [-68.0 / 3.0, -10.0])


def test_plain_sgd_step():
    result = SGD(0.1).step([1.0, 2.0], [2.0, -4.0])
    np.testing.assert_allclose(result, [0.8, 2.4])


def test_first_momentum_step_matches_plain_sgd():
    params = np.array([1.0, 2.0])
    grad = np.array([2.0, -4.0])
    expected = SGD(0.1).step(params, grad)
    actual = Momentum(0.1, beta=0.9).step(params, grad)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("optimizer", [AdaGrad(0.2), RMSProp(0.1), Adam(0.1)])
def test_adaptive_optimizer_returns_finite_updated_parameters(optimizer):
    result = optimizer.step([0.0, 0.0], [2.0, -4.0])
    assert result.shape == (2,)
    assert np.isfinite(result).all()
    assert result[0] < 0 and result[1] > 0


def test_adam_first_step_is_close_to_sign_scaled_step():
    result = Adam(0.01).step([0.0, 0.0], [2.0, -4.0])
    np.testing.assert_allclose(result, [-0.01, 0.01], atol=1e-6)


def test_minibatch_training_shapes_and_batch_counts():
    x, y = make_regression_data(n=24, seed=3)
    history = train_minibatch(x, y, SGD(0.01), batch_size=8, epochs=3, seed=2)
    assert history.params.shape == (1 + 3 * 3, 2)
    assert history.epoch_end_params.shape == (4, 2)
    assert history.full_loss.shape == (4,)
    assert history.batch_sizes.tolist() == [8] * 9


def test_training_is_reproducible_for_seeded_batch_order():
    x, y = make_regression_data(n=32, seed=9)
    first = train_minibatch(x, y, Adam(0.02), batch_size=8, epochs=4, seed=10)
    second = train_minibatch(x, y, Adam(0.02), batch_size=8, epochs=4, seed=10)
    np.testing.assert_allclose(first.params, second.params)
    np.testing.assert_allclose(first.full_loss, second.full_loss)


def test_mini_batch_training_reduces_loss_on_illustrative_data():
    x, y = make_regression_data(n=96, seed=42)
    initial_loss = mse(x, y, [0.0, 0.0])
    history = train_minibatch(x, y, Adam(0.03), batch_size=12, epochs=30, seed=42)
    assert history.final_loss < initial_loss
    assert np.isfinite(history.params).all()


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"batch_size": 25}, {"epochs": 0},
])
def test_training_rejects_invalid_configuration(kwargs):
    x, y = make_regression_data(n=24, seed=4)
    options = {"batch_size": 8, "epochs": 2, "seed": 1}
    options.update(kwargs)
    with pytest.raises(ValueError):
        train_minibatch(x, y, SGD(0.01), **options)


def test_optimizer_rejects_invalid_learning_rate():
    with pytest.raises(ValueError):
        SGD(0.0)
