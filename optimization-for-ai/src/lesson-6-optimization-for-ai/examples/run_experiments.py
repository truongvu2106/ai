"""Run the numerical experiments referenced by Lesson 6 prompts and animations."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from optimization_ai_lesson6 import (  # noqa: E402
    convex_gradient, convex_objective, nonconvex_gradient,
    nonconvex_objective, nonconvex_stationary_points, run_gradient_descent_2d,
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


def main() -> None:
    print("LESSON 6 — CONVEXITY AND WHY NEURAL NETWORK OPTIMIZATION IS DIFFICULT")
    print("=" * 72)
    print("Convex teaching objective: J(x,y)=(x-0.5)^2 + 0.5(y+0.25)^2")
    print("Unique global minimum: (x,y)=(0.5,-0.25), J=0")
    starts_convex = [(-2.0, 1.5), (2.0, 1.25), (-1.0, -1.5)]
    for start in starts_convex:
        run = run_gradient_descent_2d(
            convex_objective, convex_gradient, start, 0.2,
            max_steps=100, tolerance=1e-6,
        )
        end = run.points[-1]
        print(
            f"  start={start!s:<18} end=({end[0]: .6f}, {end[1]: .6f}) "
            f"J={run.losses[-1]:.8f}, steps={len(run.points)-1}, converged={run.converged}"
        )

    print("\nNon-convex teaching objective: f(x,y)=(x^2-1)^2 + 0.5*y^2 + 0.2*x")
    print("Stationary points (y=0):")
    for name, item in nonconvex_stationary_points().items():
        print(
            f"  {name:<18} x={float(item['x']): .6f}, "
            f"f={float(item['value']): .6f}, "
            f"Hxx={float(item['hessian_xx']): .6f}, Hyy={float(item['hessian_yy']):.1f}"
        )

    print("\nGradient descent with eta=0.02 from different starts:")
    starts_nonconvex = [(-1.8, 1.4), (1.8, 1.2)]
    end_runs = []
    for start in starts_nonconvex:
        run = run_gradient_descent_2d(
            nonconvex_objective, nonconvex_gradient, start, 0.02,
            max_steps=1200, tolerance=1e-6,
        )
        end = run.points[-1]
        end_runs.append(run)
        print(
            f"  start={start!s:<18} end=({end[0]: .6f}, {end[1]: .6f}) "
            f"f={run.losses[-1]:.8f}, steps={len(run.points)-1}, converged={run.converged}"
        )

    # Generate a static preview that uses the exact objective functions above.
    x = np.linspace(-2.3, 2.3, 280)
    y = np.linspace(-2.0, 2.0, 240)
    X, Y = np.meshgrid(x, y)
    Zc = (X - 0.5) ** 2 + 0.5 * (Y + 0.25) ** 2
    Zn = (X**2 - 1.0) ** 2 + 0.5 * Y**2 + 0.2 * X

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=BG)
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(color=GRID, alpha=0.45, linewidth=0.5)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    ax1.contourf(X, Y, Zc, levels=35, cmap="viridis")
    convex_runs = [
        run_gradient_descent_2d(convex_objective, convex_gradient, s, 0.2, max_steps=70, tolerance=1e-5)
        for s in starts_convex
    ]
    for run, color in zip(convex_runs, [CYAN, AMBER, CORAL]):
        ax1.plot(run.points[:, 0], run.points[:, 1], color=color, linewidth=1.6)
        ax1.scatter(*run.points[0], color=color, marker="o", s=28, edgecolors=WHITE, linewidths=.5)
    ax1.scatter([0.5], [-0.25], color=GREEN, marker="*", s=180, edgecolors=WHITE, linewidths=.7)
    ax1.set_title("Convex: all shown starts reach the unique global minimum", color=WHITE, fontsize=10)
    ax1.set_xlabel("Parameter x")
    ax1.set_ylabel("Parameter y")

    ax2.contourf(X, Y, Zn, levels=35, cmap="magma")
    for run, color in zip(end_runs, [CYAN, AMBER]):
        ax2.plot(run.points[:, 0], run.points[:, 1], color=color, linewidth=1.6)
        ax2.scatter(*run.points[0], color=color, marker="o", s=30, edgecolors=WHITE, linewidths=.5)
    stationary = nonconvex_stationary_points()
    for name, item in stationary.items():
        marker = "X" if item["kind"] == "saddle" else "*"
        color = CORAL if item["kind"] == "saddle" else (GREEN if item["kind"] == "global_minimum" else AMBER)
        ax2.scatter([item["x"]], [item["y"]], color=color, marker=marker, s=140 if marker == "*" else 80,
                    edgecolors=WHITE, linewidths=.6, zorder=5)
    ax2.set_title("Non-convex teaching function: multiple critical points", color=WHITE, fontsize=10)
    ax2.set_xlabel("Parameter x")
    ax2.set_ylabel("Parameter y")
    fig.suptitle("Lesson 6 — Convexity vs. Non-Convexity", color=WHITE, fontsize=15, weight="bold")
    fig.text(0.5, 0.015, "Illustrative mathematical functions—not actual neural-network loss surfaces.",
             color=MUTED, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = ROOT / "outputs" / "lesson6_landscapes_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"\nPreview saved to: {out}")


if __name__ == "__main__":
    main()
