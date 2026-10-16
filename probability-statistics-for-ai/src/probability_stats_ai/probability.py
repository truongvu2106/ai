"""Basic probability rules and Bayes' theorem."""

import math


def _validate_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def probability_union(p_a: float, p_b: float, p_a_and_b: float) -> float:
    """Return P(A ∪ B) using P(A) + P(B) - P(A ∩ B)."""
    for value, name in ((p_a, "p_a"), (p_b, "p_b"), (p_a_and_b, "p_a_and_b")):
        _validate_probability(value, name)

    lower = max(0.0, p_a + p_b - 1.0)
    upper = min(p_a, p_b)
    if not lower - 1e-12 <= p_a_and_b <= upper + 1e-12:
        raise ValueError("p_a_and_b is inconsistent with p_a and p_b.")
    return p_a + p_b - p_a_and_b


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Return P(A | B) = P(A ∩ B) / P(B)."""
    _validate_probability(p_a_and_b, "p_a_and_b")
    _validate_probability(p_b, "p_b")
    if p_b == 0:
        raise ValueError("p_b must be greater than 0 for conditional probability.")
    if p_a_and_b > p_b:
        raise ValueError("P(A ∩ B) cannot exceed P(B).")
    return p_a_and_b / p_b


def independent_joint_probability(p_a: float, p_b: float) -> float:
    """Return P(A ∩ B) = P(A)P(B), assuming A and B are independent."""
    _validate_probability(p_a, "p_a")
    _validate_probability(p_b, "p_b")
    return p_a * p_b


def bayes_theorem(
    p_b_given_a: float,
    p_a: float,
    p_b_given_not_a: float,
) -> float:
    """Compute P(A | B) from P(B | A), P(A), and P(B | not A)."""
    _validate_probability(p_b_given_a, "p_b_given_a")
    _validate_probability(p_a, "p_a")
    _validate_probability(p_b_given_not_a, "p_b_given_not_a")

    p_b = p_b_given_a * p_a + p_b_given_not_a * (1.0 - p_a)
    if p_b == 0:
        raise ValueError("P(B) is zero, so P(A | B) is undefined.")
    return p_b_given_a * p_a / p_b
