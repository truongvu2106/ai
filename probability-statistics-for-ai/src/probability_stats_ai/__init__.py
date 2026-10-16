"""Educational utilities for Probability & Statistics for AI."""

from .descriptive import (
    expectation_discrete,
    population_mean,
    population_standard_deviation,
    population_variance,
    sample_mean,
    sample_standard_deviation,
    sample_variance,
    standard_deviation_discrete,
    variance_discrete,
)
from .distributions import normal_pdf, simulate_clt, z_score
from .estimation import bernoulli_likelihood, bernoulli_log_likelihood, bernoulli_mle
from .inference import (
    proportion_confidence_interval_wald,
    proportion_confidence_interval_wilson,
    proportion_margin_of_error,
    proportion_standard_error,
    one_sample_proportion_z_test,
    sample_proportion,
)
from .probability import (
    bayes_theorem,
    conditional_probability,
    independent_joint_probability,
    probability_union,
)
from .classification import (
    binary_cross_entropy,
    calibration_table,
    expected_calibration_error,
    classification_metrics,
    confusion_matrix_binary,
    roc_auc_score_binary,
)

__all__ = [
    "expectation_discrete", "variance_discrete", "standard_deviation_discrete",
    "population_mean", "population_variance", "population_standard_deviation",
    "sample_mean", "sample_variance", "sample_standard_deviation",
    "normal_pdf", "z_score", "simulate_clt", "bernoulli_mle",
    "bernoulli_log_likelihood", "bernoulli_likelihood", "sample_proportion",
    "proportion_standard_error", "proportion_margin_of_error",
    "proportion_confidence_interval_wald", "proportion_confidence_interval_wilson",
    "one_sample_proportion_z_test", "probability_union", "conditional_probability",
    "independent_joint_probability", "bayes_theorem", "confusion_matrix_binary",
    "classification_metrics", "binary_cross_entropy", "calibration_table",
    "expected_calibration_error", "roc_auc_score_binary",
]
