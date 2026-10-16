"""Run the end-to-end exercises for Probability & Statistics for AI."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allows running this file directly from a cloned repository without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probability_stats_ai import (  # noqa: E402
    bayes_theorem,
    bernoulli_mle,
    bernoulli_log_likelihood,
    binary_cross_entropy,
    calibration_table,
    expected_calibration_error,
    classification_metrics,
    expectation_discrete,
    normal_pdf,
    independent_joint_probability,
    probability_union,
    one_sample_proportion_z_test,
    proportion_confidence_interval_wald,
    proportion_confidence_interval_wilson,
    proportion_standard_error,
    roc_auc_score_binary,
    simulate_clt,
    standard_deviation_discrete,
    variance_discrete,
    z_score,
)


def sampling_bias_demo(output_path: Path) -> tuple[float, float, float]:
    """Compare a random sample with a convenience sample from enthusiasts.

    The constructed population has exactly 50% AI users: 40% among 80,000
    general users and 90% among 20,000 tech enthusiasts.
    """
    general_users = np.array([1] * 32_000 + [0] * 48_000)
    enthusiasts = np.array([1] * 18_000 + [0] * 2_000)
    population = np.concatenate([general_users, enthusiasts])
    rng = np.random.default_rng(19)
    random_sample = rng.choice(population, size=400, replace=False)
    convenience_sample = rng.choice(enthusiasts, size=400, replace=False)

    population_rate = float(np.mean(population))
    random_rate = float(np.mean(random_sample))
    convenience_rate = float(np.mean(convenience_sample))

    labels = ["Full population", "Random sample", "Tech-enthusiast\nsample"]
    rates = [population_rate, random_rate, convenience_rate]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, rates)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion reporting AI use")
    ax.set_title("Sampling bias: who you ask changes the result")
    ax.axhline(population_rate, linestyle="--", linewidth=1.5, label="True population rate")
    ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=4)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return population_rate, random_rate, convenience_rate


def plot_clt_simulation(output_path: Path) -> None:
    """Plot a skewed population and sample-mean distributions for several n."""
    rng = np.random.default_rng(7)
    population = rng.exponential(scale=1.0, size=100_000)
    sample_sizes = [1, 2, 10, 30]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].hist(population, bins=70, density=True, alpha=0.78)
    axes[0].set_title("Original population: Exponential")
    axes[0].set_xlabel("Observation value")
    axes[0].set_ylabel("Density")
    axes[0].axvline(1.0, linestyle="--", linewidth=1.5, label="Population mean = 1")
    axes[0].legend()

    for ax, n in zip(axes[1:], sample_sizes[1:]):
        means = simulate_clt(population, sample_size=n, n_samples=8_000, seed=100 + n)
        ax.hist(means, bins=55, density=True, alpha=0.78)
        ax.axvline(1.0, linestyle="--", linewidth=1.5, label="Population mean = 1")
        ax.set_title(f"Sampling distribution of means (n={n})")
        ax.set_xlabel("Sample mean")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Central Limit Theorem: sample means become more concentrated", fontsize=13)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confidence_interval(output_path: Path, interval: dict[str, float]) -> None:
    """Plot a single approximate confidence interval for the example."""
    estimate = interval["estimate"]
    lower = interval["lower"]
    upper = interval["upper"]
    fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    ax.errorbar(
        estimate, 1, xerr=[[estimate - lower], [upper - estimate]],
        fmt="o", capsize=8, markersize=8, linewidth=3,
    )
    ax.axvline(estimate, linestyle="--", linewidth=1.2)
    ax.set_xlim(0.45, 0.75)
    ax.set_ylim(0.6, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("Population proportion")
    ax.set_title("Approximate 95% confidence interval for the AI-use proportion")
    ax.annotate(f"{lower:.3f}", (lower, 1), xytext=(0, -22), textcoords="offset points", ha="center")
    ax.annotate(f"p̂ = {estimate:.2f}", (estimate, 1), xytext=(0, 14), textcoords="offset points", ha="center")
    ax.annotate(f"{upper:.3f}", (upper, 1), xytext=(0, -22), textcoords="offset points", ha="center")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    print("=== Lesson 9: Probability & Statistics for AI ===\n")

    print("1) Sampling bias: random sample vs convenience sample")
    bias_path = outputs_dir / "sampling_bias_demo.png"
    population_rate, random_rate, convenience_rate = sampling_bias_demo(bias_path)
    print(f"True population AI-use rate: {population_rate:.2%}")
    print(f"Random sample of 400: {random_rate:.2%}")
    print(f"Convenience sample of tech enthusiasts: {convenience_rate:.2%}")
    print(f"Saved: {bias_path.relative_to(REPO_ROOT)}\n")

    print("2) Probability rules")
    print(f"P(A or B) with P(A)=0.30, P(B)=0.20, P(A and B)=0.10: {probability_union(0.30, 0.20, 0.10):.2f}")
    print(f"P(A and B) if independent and each has probability 0.30: {independent_joint_probability(0.30, 0.30):.2f}\n")

    print("3) Bayes' theorem")
    posterior = bayes_theorem(p_b_given_a=0.80, p_a=0.10, p_b_given_not_a=0.20)
    print(f"P(spam | contains 'free') = {posterior:.4f} ({posterior:.2%})\n")

    print("4) Expectation, variance, and standard deviation")
    values = [0, 2, 4]
    probabilities = [0.25, 0.50, 0.25]
    expectation = expectation_discrete(values, probabilities)
    variance = variance_discrete(values, probabilities)
    std = standard_deviation_discrete(values, probabilities)
    print(f"E[X] = {expectation:.4f}")
    print(f"Var(X) = {variance:.4f}")
    print(f"SD(X) = {std:.4f}\n")

    print("5) Gaussian distribution and Z-score")
    print(f"z-score for x=115, mean=100, std=15: {z_score(115, 100, 15):.2f}")
    print(f"Standard normal density at z=0: {normal_pdf(0):.4f}\n")

    print("6) Maximum Likelihood Estimation")
    coin_tosses = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    p_hat = bernoulli_mle(coin_tosses)
    print(f"Observed heads: {sum(coin_tosses)}/{len(coin_tosses)}")
    print(f"Bernoulli MLE for P(heads): {p_hat:.2f}")
    print(f"Log-likelihood at p=0.70: {bernoulli_log_likelihood(coin_tosses, 0.70):.4f}")
    print(f"Log-likelihood at p=0.50: {bernoulli_log_likelihood(coin_tosses, 0.50):.4f}\n")

    print("7) Sampling, Standard Error, and Confidence Interval")
    successes, n = 240, 400
    p_hat_sample = successes / n
    se = proportion_standard_error(p_hat_sample, n)
    interval = proportion_confidence_interval_wald(successes, n, confidence_level=0.95)
    wilson = proportion_confidence_interval_wilson(successes, n, confidence_level=0.95)
    print(f"Sample proportion: {p_hat_sample:.4f}")
    print(f"Standard Error: {se:.4f} ({se:.2%})")
    print(f"Margin of Error: {interval['margin_of_error']:.4f} ({interval['margin_of_error']:.2%})")
    print(f"Wald 95% CI: [{interval['lower']:.4f}, {interval['upper']:.4f}]")
    print(f"Wilson 95% CI (practical alternative): [{wilson['lower']:.4f}, {wilson['upper']:.4f}]")
    test = one_sample_proportion_z_test(successes, n, p0=0.50, alternative="two-sided")
    print(f"One-sample z-test under H0: p=0.50: z={test['z_statistic']:.2f}, p-value={test['p_value']:.8f}\n")

    print("8) Classification metrics")
    # Ground truth: 20 spam, 80 non-spam. Predictions: TP=15, FN=5, FP=10, TN=70.
    y_true = np.array([1] * 20 + [0] * 80)
    y_pred = np.array([1] * 15 + [0] * 5 + [1] * 10 + [0] * 70)
    probabilities = np.array([0.90] * 15 + [0.40] * 5 + [0.70] * 10 + [0.10] * 70)
    metrics = classification_metrics(y_true, y_pred)
    for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "specificity", "f1"):
        value = metrics[key]
        print(f"{key:>12}: {value:.4f}" if isinstance(value, float) else f"{key:>12}: {value}")
    print(f"Binary cross-entropy: {binary_cross_entropy(y_true, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score_binary(y_true, probabilities):.4f}")
    print(f"Expected Calibration Error (10 bins): {expected_calibration_error(y_true, probabilities):.4f}")
    print("Calibration table:")
    for row in calibration_table(y_true, probabilities):
        print(f"  bin [{row['bin_lower']:.1f}, {row['bin_upper']:.1f}): n={row['count']}, "
              f"mean predicted={row['mean_predicted_probability']:.2f}, "
              f"observed positive rate={row['observed_positive_rate']:.2f}")
    print()

    print("9) Create plots")
    clt_path = outputs_dir / "clt_sampling_means.png"
    ci_path = outputs_dir / "confidence_interval.png"
    plot_clt_simulation(clt_path)
    plot_confidence_interval(ci_path, interval)
    print(f"Saved: {bias_path.relative_to(REPO_ROOT)}")
    print(f"Saved: {clt_path.relative_to(REPO_ROOT)}")
    print(f"Saved: {ci_path.relative_to(REPO_ROOT)}")
    print("\nDone. Explore the source modules and the exercises/ folder next.")


if __name__ == "__main__":
    main()
