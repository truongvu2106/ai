"""Run the reproducible numerical experiments for Lesson 5."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from optimization_ai_lesson5 import (  # noqa: E402
    nonconvex_gradient, nonconvex_loss, quadratic_gradient,
    quadratic_loss, quadratic_run, run_gradient_descent,
)

BG = "#0B0F19"
GRID = "#1E293B"
CYAN = "#38BDF8"
AMBER = "#F59E0B"
CORAL = "#FB7185"
GREEN = "#10B981"
PURPLE = "#A855F7"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"


def print_learning_rate_comparison() -> None:
    print("\n1) LEARNING RATE: J(theta) = (theta - 3)^2; theta_0 = -2")
    print("learning_rate | theta_1 | loss_0 | loss_1 | interpretation")
    for lr, label in [(0.05, "small; slow"), (0.25, "stable in this example"), (1.1, "too large; diverges")]:
        run = quadratic_run(-2.0, lr, steps=2)
        print(f"{lr:13.2f} | {run.theta[1]:7.3f} | {run.loss[0]:6.2f} | {run.loss[1]:6.2f} | {label}")
    print("For eta=0.25: theta: -2 -> 0.5 -> 1.75; loss: 25 -> 6.25 -> 1.5625")


def print_convergence() -> None:
    print("\n2) CONVERGENCE: criterion is abs(gradient) <= 1e-3")
    for lr in (0.05, 0.25, 1.1):
        run = run_gradient_descent(
            quadratic_loss, quadratic_gradient, -2.0, lr,
            max_steps=250, tolerance=1e-3, divergence_limit=1e8,
        )
        status = "converged" if run.converged else "NOT converged (unstable/diverging trajectory or iteration limit)"
        print(f"eta={lr:0.2f}: steps={len(run.theta)-1:3d}, theta={run.theta[-1]: .6f}, loss={run.loss[-1]:.6g}, {status}")


def print_local_minima() -> None:
    print("\n3) NON-CONVEX OBJECTIVE: f(theta)=0.1*theta^2+2*cos(1.5*theta)+0.12*theta")
    for start in (-6.0, -4.0, 1.0):
        run = run_gradient_descent(
            nonconvex_loss, nonconvex_gradient, start, 0.02,
            max_steps=300, tolerance=1e-5,
        )
        print(
            f"start={start:5.1f} -> theta={run.theta[-1]:8.4f}, "
            f"f(theta)={run.loss[-1]:8.4f}, steps={len(run.theta)-1:3d}, "
            f"converged={run.converged}"
        )


def save_static_preview() -> Path:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(WHITE)

    theta = np.linspace(-5.0, 12.0, 500)
    axes[0].plot(theta, (theta - 3.0) ** 2, color=CYAN, linewidth=2)
    for lr, color, label, steps in [
        (0.05, AMBER, "small η=0.05", 18),
        (0.25, GREEN, "stable η=0.25", 18),
        (1.1, CORAL, "too large η=1.1", 8),
    ]:
        run = quadratic_run(-2.0, lr, steps=steps)
        axes[0].plot(run.theta, np.minimum(run.loss, 80), "o-", color=color, markersize=3, linewidth=1.5, label=label)
    axes[0].set_xlim(-5, 12)
    axes[0].set_ylim(0, 80)
    axes[0].set_xlabel("Parameter θ")
    axes[0].set_ylabel("Objective J(θ)")
    axes[0].set_title("Learning rate changes the update path")
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8)

    xs = np.linspace(-7.5, 7.5, 700)
    axes[1].plot(xs, [nonconvex_loss(float(x)) for x in xs], color=PURPLE, linewidth=2)
    for start, color, label in [(-6.0, CYAN, "start −6"), (-4.0, GREEN, "start −4"), (1.0, AMBER, "start 1")]:
        run = run_gradient_descent(nonconvex_loss, nonconvex_gradient, start, 0.02, max_steps=300, tolerance=1e-5)
        axes[1].plot(run.theta, run.loss, "o-", color=color, markersize=2.5, linewidth=1.4, label=label)
        axes[1].scatter([run.theta[-1]], [run.loss[-1]], color=color, s=55, edgecolors=WHITE, linewidths=0.6, zorder=5)
    axes[1].set_xlim(-7.5, 7.5)
    axes[1].set_xlabel("Parameter θ")
    axes[1].set_ylabel("Objective f(θ)")
    axes[1].set_title("A non-convex landscape depends on initialization")
    axes[1].legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8)
    fig.suptitle("Learning Rate, Convergence, and Local Minima", color=WHITE, fontsize=15, weight="bold")
    fig.tight_layout()
    out = out_dir / "lesson5_landscapes_preview.png"
    fig.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print_learning_rate_comparison()
    print_convergence()
    print_local_minima()
    print(f"\nSaved static preview: {save_static_preview()}")


if __name__ == "__main__":
    main()
