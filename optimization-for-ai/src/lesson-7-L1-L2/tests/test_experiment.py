import numpy as np
import pytest

from optimization_ai_lesson7.experiment import (
    build_models, evaluate_models, fit_lasso, fit_ridge, make_dataset,
    mean_squared_error, objective_value, true_function,
)


def test_synthetic_data_are_deterministic():
    first = make_dataset()
    second = make_dataset()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert len(first[0]) == 18
    assert len(first[2]) == 100


def test_true_function_reference_values():
    np.testing.assert_allclose(true_function(np.array([0.0, 0.25, 0.5, 0.75])), [0, 1, 0, -1], atol=1e-12)


def test_mse_zero_for_exact_predictions():
    assert mean_squared_error(np.array([1, 2]), np.array([1, 2])) == 0.0


def test_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mean_squared_error(np.array([1, 2]), np.array([1]))


def test_ridge_model_predicts_finite_values():
    xt, yt, _, _ = make_dataset()
    model = fit_ridge(xt, yt, degree=12, regularization=0.001)
    assert np.isfinite(model.predict(np.linspace(0, 1, 101))).all()


def test_regularization_strength_must_be_nonnegative():
    xt, yt, _, _ = make_dataset()
    with pytest.raises(ValueError):
        fit_ridge(xt, yt, degree=3, regularization=-0.1)
    with pytest.raises(ValueError):
        fit_lasso(xt, yt, degree=3, regularization=-0.1)


def test_overfit_model_has_lower_train_but_higher_validation_mse():
    scores = evaluate_models()
    assert scores["overfit"]["train_mse"] < scores["simple"]["train_mse"]
    assert scores["overfit"]["validation_mse"] > scores["simple"]["validation_mse"]


def test_ridge_improves_validation_in_this_fixed_example():
    scores = evaluate_models()
    assert scores["ridge_0001"]["validation_mse"] < scores["overfit"]["validation_mse"]


def test_lasso_produces_some_zero_coefficients():
    models = build_models()
    assert models["lasso_0002"].nonzero_coefficients < 12


def test_regularized_objective_includes_penalty():
    xt, yt, _, _ = make_dataset()
    model = fit_ridge(xt, yt, degree=12, regularization=0.001)
    assert objective_value(model, xt, yt) > mean_squared_error(yt, model.predict(xt))
