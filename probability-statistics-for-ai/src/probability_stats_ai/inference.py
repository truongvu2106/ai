"""Introductory inference tools for a population proportion."""

from math import erfc, isfinite, sqrt
from statistics import NormalDist


def _validate_counts(successes: int, n: int) -> None:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(successes, int) or not 0 <= successes <= n:
        raise ValueError("successes must be an integer between 0 and n.")


def sample_proportion(successes: int, n: int) -> float:
    """Compute the observed sample proportion p_hat = successes / n."""
    _validate_counts(successes, n)
    return successes / n


def proportion_standard_error(p_hat: float, n: int) -> float:
    """Plug-in SE for a sample proportion: sqrt(p_hat * (1 - p_hat) / n)."""
    if not isfinite(p_hat) or not 0.0 <= p_hat <= 1.0:
        raise ValueError("p_hat must be between 0 and 1.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    return sqrt(p_hat * (1.0 - p_hat) / n)


def proportion_margin_of_error(p_hat: float, n: int, confidence_level: float = 0.95) -> float:
    """Wald margin of error for a proportion using the normal critical value."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1, exclusive.")
    z_critical = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    return z_critical * proportion_standard_error(p_hat, n)


def proportion_confidence_interval_wald(
    successes: int,
    n: int,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return an approximate Wald confidence interval for a proportion.

    This matches the simple p_hat +/- z*SE method taught in the lesson. The
    Wald interval can perform poorly for small samples or proportions near 0/1;
    use the Wilson interval below for many practical settings.
    """
    _validate_counts(successes, n)
    p_hat = sample_proportion(successes, n)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1, exclusive.")
    z_critical = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    se = proportion_standard_error(p_hat, n)
    margin = z_critical * se
    return {
        "estimate": p_hat,
        "standard_error": se,
        "z_critical": z_critical,
        "margin_of_error": margin,
        "lower": max(0.0, p_hat - margin),
        "upper": min(1.0, p_hat + margin),
        "confidence_level": confidence_level,
    }


def proportion_confidence_interval_wilson(
    successes: int,
    n: int,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return a Wilson score confidence interval for a population proportion."""
    _validate_counts(successes, n)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1, exclusive.")
    p_hat = successes / n
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = z * sqrt((p_hat * (1.0 - p_hat) / n) + z2 / (4.0 * n * n)) / denominator
    return {
        "estimate": p_hat,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
        "confidence_level": confidence_level,
    }


def one_sample_proportion_z_test(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",
) -> dict[str, float | str]:
    """Perform a one-sample normal-approximation z-test for a proportion.

    The null standard error uses p0, not the observed sample proportion.
    """
    _validate_counts(successes, n)
    if not isfinite(p0) or not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be strictly between 0 and 1.")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")

    p_hat = successes / n
    se_null = sqrt(p0 * (1.0 - p0) / n)
    z_statistic = (p_hat - p0) / se_null
    if alternative == "two-sided":
        p_value = erfc(abs(z_statistic) / sqrt(2.0))
    elif alternative == "greater":
        p_value = 0.5 * erfc(z_statistic / sqrt(2.0))
    else:
        p_value = 0.5 * erfc(-z_statistic / sqrt(2.0))
    return {
        "sample_proportion": p_hat,
        "null_proportion": p0,
        "standard_error_under_null": se_null,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "alternative": alternative,
    }
