"""Run the deterministic Lesson 7 experiment and save a static preview."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optimization_ai_lesson7.experiment import (  # noqa: E402
    build_models, evaluate_models, make_dataset, true_function,
)


def main() -> None:
    results = evaluate_models()
    print("Synthetic polynomial regression experiment (fixed random seeds)\n")
    print(f"{'Model':<14} {'Train MSE':>12} {'Validation MSE':>16} {'Nonzero β':>11}")
    for name, item in results.items():
        print(f"{name:<14} {item['train_mse']:>12.6f} {item['validation_mse']:>16.6f} {item['nonzero_coefficients']:>11}")

    x_train, y_train, x_valid, y_valid = make_dataset()
    models = build_models()
    x_grid = np.linspace(0.02, 0.98, 500)
    colors = {
        "simple": "#10B981", "overfit": "#FB7185",
        "ridge_0001": "#38BDF8", "lasso_0002": "#A855F7",
    }
    labels = {
        "simple": "Degree 3", "overfit": "Degree 12 / no penalty",
        "ridge_0001": "Degree 12 / L2 λ=0.001", "lasso_0002": "Degree 12 / L1 λ=0.002",
    }
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.scatter(x_train, y_train, s=38, color="#F59E0B", label="Training observations", zorder=5)
    ax.plot(x_grid, true_function(x_grid), linestyle="--", color="#94A3B8", linewidth=2, label="Underlying signal")
    for key in ["simple", "overfit", "ridge_0001", "lasso_0002"]:
        ax.plot(x_grid, models[key].predict(x_grid), color=colors[key], linewidth=2.2, label=labels[key])
    ax.set_title("Regularization and Generalization — illustrative synthetic data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-1.8, 1.8)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    output = ROOT / "outputs" / "lesson7_model_fits.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"\nSaved model-fit preview to: {output}")
    print("Note: data are synthetic; validation results describe this fixed experiment, not a universal ranking.")


if __name__ == "__main__":
    main()
