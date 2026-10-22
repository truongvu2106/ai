"""Generate five Lesson 5 MP4 scenes locally.

Scenes are strictly aligned with VIDEO_PROMPTS.md and objectives.py:
  learning-rate : small/stable/too-large learning rates on J(theta)=(theta-3)^2
  one-update    : four exact updates using eta=0.25
  convergence   : loss and gradient-magnitude histories for the same three rates
  local-minima  : different initializations on the same illustrative non-convex function
  recap         : visual checklist summarizing the same experiments

Requires FFmpeg on PATH. No videos are generated merely by importing this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

SCENES = ("learning-rate", "one-update", "convergence", "local-minima", "recap")


def style_axis(ax: Axes, title: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, weight="bold", pad=8)


def canvas(nrows: int = 1, ncols: int = 2, figsize: tuple[float, float] = (12.8, 7.2)) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.82, bottom=0.13, hspace=0.36, wspace=0.28)
    for ax in axes.flat:
        style_axis(ax)
    return fig, axes


def header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.075, 0.955, title, color=WHITE, fontsize=19, weight="bold", va="top")
    fig.text(0.075, 0.905, subtitle, color=MUTED, fontsize=10, va="top")


def save_animation(
    fig: Figure,
    update: Callable[[int], list],
    frames: int,
    path: Path,
    fps: int,
    dpi: int,
) -> None:
    if not writers.is_available("ffmpeg"):
        plt.close(fig)
        raise RuntimeError("FFmpeg writer unavailable. Install FFmpeg and ensure it is on PATH.")
    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=2800,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Created: {path}")


def frame_count(seconds: float, fps: int) -> int:
    return max(2, int(round(seconds * fps)))


def animate_learning_rate(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 1: compare small, suitable, and unstable learning rates."""
    rates = [(0.05, AMBER, "Small η = 0.05"), (0.25, GREEN, "Suitable here η = 0.25"), (1.1, CORAL, "Too large η = 1.10")]
    runs = {lr: quadratic_run(-2.0, lr, steps=5) for lr, _, _ in rates}
    fig, axes = canvas(1, 2)
    ax_land, ax_text = axes[0]
    header(fig, "Learning rate changes the path", "Same starting point and objective; only the learning rate changes")
    style_axis(ax_land, "Convex loss landscape: J(θ) = (θ − 3)²")
    theta_grid = np.linspace(-5.0, 13.0, 500)
    ax_land.plot(theta_grid, (theta_grid - 3.0) ** 2, color=CYAN, linewidth=2.1, label="J(θ)")
    ax_land.axvline(3.0, color=GREEN, linestyle="--", alpha=0.8, linewidth=1.2)
    ax_land.text(3.2, 67, "minimum θ*=3", color=GREEN, fontsize=9)
    ax_land.set_xlim(-6.0, 13.5)
    ax_land.set_ylim(0.0, 100.0)
    ax_land.set_xlabel("Parameter θ")
    ax_land.set_ylabel("Objective J(θ)")
    dots: dict[float, object] = {}
    trails: dict[float, object] = {}
    for lr, color, label in rates:
        trails[lr], = ax_land.plot([], [], "o-", color=color, linewidth=1.5, markersize=4, label=label)
        dots[lr], = ax_land.plot([], [], "o", color=color, markersize=9, markeredgecolor=WHITE, markeredgewidth=0.8)
    ax_land.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7.6, loc="upper left")

    style_axis(ax_text, "Observe the behavior")
    ax_text.set_axis_off()
    callout = ax_text.text(0.02, 0.90, "", transform=ax_text.transAxes, color=WHITE, fontsize=13, va="top", linespacing=1.6)
    step_text = ax_text.text(0.02, 0.22, "", transform=ax_text.transAxes, color=MUTED, fontsize=11, va="top", linespacing=1.6)

    frames = frame_count(seconds, fps)
    states_per_step = max(1, frames // 6)

    def update(frame: int) -> list:
        step = min(5, frame // states_per_step)
        alpha = min(1.0, (frame % states_per_step) / max(1, states_per_step - 1))
        for lr, color, label in rates:
            states = runs[lr].theta
            idx = min(step, len(states) - 1)
            previous = states[max(0, idx - 1)]
            current = states[idx]
            theta = previous + alpha * (current - previous) if step > 0 else current
            history = states[:idx + 1]
            if step > 0 and alpha < 1.0:
                history = np.append(states[:idx], theta)
            else:
                history = states[:idx + 1]
            y_history = np.minimum((history - 3.0) ** 2, 99.0)
            trails[lr].set_data(history, y_history)
            dots[lr].set_data([np.clip(theta, -6.0, 13.5)], [min(quadratic_loss(float(theta)), 99.0)])
        lr, color, label = rates[-1]
        fast_run = runs[1.1]
        actual_theta = fast_run.theta[min(step, len(fast_run.theta) - 1)]
        fast_loss = fast_run.loss[min(step, len(fast_run.loss) - 1)]
        if step <= 1:
            interpretation = "A small step makes steady\nbut slow progress."
        elif step <= 3:
            interpretation = "A suitable step reaches the\nminimum quickly in this example."
        else:
            interpretation = "An oversized step jumps past\nthe minimum; its loss grows."
        callout.set_text(interpretation)
        step_text.set_text(
            f"Iteration shown: {step}\n"
            f"η=1.10 parameter: {actual_theta:.3f}\n"
            f"η=1.10 loss: {fast_loss:.3f}\n\n"
            "This stability boundary is specific\nto this quadratic example."
        )
        return [*dots.values(), *trails.values(), callout, step_text]

    save_animation(fig, update, frames, out / "01_learning_rate_comparison.mp4", fps, dpi)


def animate_one_update(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 2: show exactly how one parameter update is calculated."""
    lr = 0.25
    run = quadratic_run(-2.0, lr, steps=5)
    fig, axes = canvas(1, 2)
    ax_curve, ax_math = axes[0]
    header(fig, "One update, step by step", "An optimizer uses the gradient at the current parameter value")
    style_axis(ax_curve, "J(θ) = (θ − 3)²")
    grid = np.linspace(-3.0, 4.3, 500)
    ax_curve.plot(grid, (grid - 3.0) ** 2, color=CYAN, linewidth=2.2)
    ax_curve.scatter([3], [0], color=GREEN, s=65, zorder=5, edgecolors=WHITE, linewidths=0.7)
    ax_curve.text(3.08, 1.6, "minimum θ*=3", color=GREEN, fontsize=9)
    ax_curve.set_xlim(-3, 4.3)
    ax_curve.set_ylim(0, 32)
    ax_curve.set_xlabel("Parameter θ")
    ax_curve.set_ylabel("Objective J(θ)")
    path_line, = ax_curve.plot([], [], "o-", color=AMBER, linewidth=1.8, markersize=5)
    point, = ax_curve.plot([], [], "o", color=CORAL, markersize=10, markeredgecolor=WHITE, markeredgewidth=0.8)
    step_arrow = None

    style_axis(ax_math, "Read the update")
    ax_math.set_axis_off()
    formula = ax_math.text(0.02, 0.94, r"$\theta_{t+1}=\theta_t-\eta\,J'(\theta_t)$", transform=ax_math.transAxes, color=WHITE, fontsize=19, va="top")
    step_label = ax_math.text(0.02, 0.76, "", transform=ax_math.transAxes, color=AMBER, fontsize=13, va="top", linespacing=1.5)
    interpretation = ax_math.text(0.02, 0.35, "", transform=ax_math.transAxes, color=MUTED, fontsize=11.5, va="top", linespacing=1.5)
    formulas = []
    for i in range(min(4, len(run.theta) - 1)):
        current = float(run.theta[i])
        grad = quadratic_gradient(current)
        next_theta = float(run.theta[i + 1])
        formulas.append((current, grad, next_theta, quadratic_loss(current), quadratic_loss(next_theta)))
    frames = frame_count(seconds, fps)
    frames_per_step = max(1, frames // len(formulas))

    def update(frame: int) -> list:
        idx = min(len(formulas) - 1, frame // frames_per_step)
        alpha = min(1.0, (frame % frames_per_step) / max(1, frames_per_step - 1))
        current, grad, next_theta, current_loss, next_loss = formulas[idx]
        theta = current + alpha * (next_theta - current)
        visible_steps = run.theta[:idx + 1]
        path_line.set_data(visible_steps, (visible_steps - 3.0) ** 2)
        point.set_data([theta], [quadratic_loss(theta)])
        step_label.set_text(
            f"Step {idx + 1}\n"
            f"Current θ = {current:.4f}\n"
            f"Gradient J′(θ) = {grad:.4f}\n"
            f"Learning rate η = {lr:.2f}\n"
            f"Next θ = {next_theta:.4f}"
        )
        interpretation.set_text(
            f"Loss: {current_loss:.6f} → {next_loss:.6f}\n\n"
            + ("The negative gradient points toward the minimum."
               if idx == 0 else "Each update uses the gradient at the current point.")
        )
        return [path_line, point, formula, step_label, interpretation]

    save_animation(fig, update, frames, out / "02_one_update_step_by_step.mp4", fps, dpi)


def animate_convergence(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 3: compare loss and gradient magnitude to explain convergence."""
    settings = [(0.05, AMBER, "η=0.05 · slow"), (0.25, GREEN, "η=0.25 · converges here"), (1.1, CORAL, "η=1.10 · diverges")]
    runs = {
        lr: run_gradient_descent(
            quadratic_loss, quadratic_gradient, -2.0, lr,
            max_steps=70 if lr < 1 else 28, tolerance=1e-12, divergence_limit=1e12,
        ) for lr, _, _ in settings
    }
    total_steps = max(len(run.theta) - 1 for run in runs.values())
    fig, axes = canvas(1, 2)
    ax_loss, ax_grad = axes[0]
    header(fig, "How do we recognize convergence?", "Track both the objective value and the gradient magnitude")
    style_axis(ax_loss, "Full objective value J(θ)")
    style_axis(ax_grad, "Gradient magnitude |J′(θ)|")
    line_loss = {}
    line_grad = {}
    for lr, color, label in settings:
        line_loss[lr], = ax_loss.plot([], [], color=color, linewidth=2, label=label)
        line_grad[lr], = ax_grad.plot([], [], color=color, linewidth=2, label=label)
    for ax in (ax_loss, ax_grad):
        ax.set_yscale("log")
        ax.set_xlabel("Update step")
        ax.set_xlim(0, total_steps)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7.6)
    ax_loss.set_ylabel("J(θ), log scale")
    ax_grad.set_ylabel("|gradient|, log scale")
    ax_loss.set_ylim(1e-5, 1e8)
    ax_grad.set_ylim(1e-5, 1e4)
    callout = fig.text(0.075, 0.055, "", color=MUTED, fontsize=11)
    frames = frame_count(seconds, fps)

    def update(frame: int) -> list:
        step = int(round((frame / max(1, frames - 1)) * total_steps))
        artists: list = [callout]
        for lr, color, label in settings:
            run = runs[lr]
            stop = min(step + 1, len(run.theta))
            xs = np.arange(stop)
            loss = np.maximum(run.loss[:stop], 1e-12)
            grad = np.maximum(np.abs(run.gradient[:stop]), 1e-12)
            line_loss[lr].set_data(xs, loss)
            line_grad[lr].set_data(xs, grad)
            artists.extend([line_loss[lr], line_grad[lr]])
        if step < 8:
            message = "Convergence means approaching a stable solution—not merely taking one downhill step."
        elif step < 24:
            message = "A suitable rate reduces both loss and gradient magnitude in this convex example."
        else:
            message = "A too-large rate can overshoot repeatedly and make the loss grow instead of converge."
        callout.set_text(message)
        return artists

    save_animation(fig, update, frames, out / "03_convergence_diagnostics.mp4", fps, dpi)


def animate_local_minima(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 4: different initial points settle into distinct minima."""
    starts = [(-6.0, CYAN, "Start θ₀ = −6"), (-4.0, GREEN, "Start θ₀ = −4"), (1.0, AMBER, "Start θ₀ = 1")]
    runs = {
        start: run_gradient_descent(
            nonconvex_loss, nonconvex_gradient, start, 0.02,
            max_steps=300, tolerance=1e-5,
        ) for start, _, _ in starts
    }
    max_steps = max(len(run.theta) - 1 for run in runs.values())
    fig, axes = canvas(1, 2)
    ax_land, ax_notes = axes[0]
    header(fig, "Local minima: the starting point matters", "A non-convex teaching landscape can contain several valleys")
    style_axis(ax_land, "f(θ) = 0.1θ² + 2 cos(1.5θ) + 0.12θ")
    xs = np.linspace(-7.5, 7.5, 700)
    ax_land.plot(xs, [nonconvex_loss(float(x)) for x in xs], color=PURPLE, linewidth=2.1)
    ax_land.set_xlim(-7.5, 7.5)
    ax_land.set_ylim(-2.2, 5.2)
    ax_land.set_xlabel("Parameter θ")
    ax_land.set_ylabel("Objective f(θ)")
    lines = {}
    dots = {}
    for start, color, label in starts:
        lines[start], = ax_land.plot([], [], "-", color=color, linewidth=1.7, label=label)
        dots[start], = ax_land.plot([], [], "o", color=color, markersize=9, markeredgecolor=WHITE, markeredgewidth=0.8)
    ax_land.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8, loc="upper center")
    style_axis(ax_notes, "What the experiment shows")
    ax_notes.set_axis_off()
    note = ax_notes.text(0.02, 0.92, "", transform=ax_notes.transAxes, color=WHITE, fontsize=12, va="top", linespacing=1.6)
    endpoint_note = ax_notes.text(0.02, 0.40, "", transform=ax_notes.transAxes, color=MUTED, fontsize=10.5, va="top", linespacing=1.6)
    frames = frame_count(seconds, fps)

    def update(frame: int) -> list:
        step = int(round(frame / max(1, frames - 1) * max_steps))
        for start, color, label in starts:
            run = runs[start]
            idx = min(step, len(run.theta) - 1)
            th = run.theta[:idx + 1]
            ls = run.loss[:idx + 1]
            lines[start].set_data(th, ls)
            dots[start].set_data([th[-1]], [ls[-1]])
        if step < 35:
            note.set_text("The landscape is not one simple bowl.\nIts valleys contain different local minima.")
        elif step < 110:
            note.set_text("Gradient Descent follows local slope information.\nIt does not automatically compare every valley.")
        else:
            note.set_text("Different starting points can settle in\ndifferent minima on the same objective.")
        pieces = []
        for start, color, label in starts:
            run = runs[start]
            idx = min(step, len(run.theta) - 1)
            pieces.append(f"{label}: θ≈{run.theta[idx]:.3f}, f≈{run.loss[idx]:.3f}")
        endpoint_note.set_text("\n".join(pieces) + "\n\nIllustrative 1D landscape; not a neural-network loss surface.")
        return [*lines.values(), *dots.values(), note, endpoint_note]

    save_animation(fig, update, frames, out / "04_local_minima_initialization.mp4", fps, dpi)


def animate_recap(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 5: final diagnostic checklist aligned with the lesson examples."""
    fig, axes = canvas(1, 2)
    ax_left, ax_right = axes[0]
    header(fig, "Diagnose the optimization behavior", "Ask three questions before changing the training setup")
    for ax in axes.flat:
        ax.set_axis_off()
    left_lines = [
        ("SMALL LEARNING RATE", "Loss decreases slowly", AMBER),
        ("SUITABLE LEARNING RATE", "Loss approaches the minimum", GREEN),
        ("TOO-LARGE LEARNING RATE", "Overshoots; loss may diverge", CORAL),
    ]
    right_lines = [
        ("CONVERGENCE", "Do parameters, objective, or gradient stabilize?", CYAN),
        ("LOCAL MINIMA", "Can a different start reach another valley?", PURPLE),
        ("IMPORTANT LIMIT", "Behavior depends on the objective and settings", WHITE),
    ]
    texts = []
    for ax, items in [(ax_left, left_lines), (ax_right, right_lines)]:
        for i, (title, body, color) in enumerate(items):
            y = 0.82 - i * 0.27
            title_artist = ax.text(0.02, y, title, transform=ax.transAxes, color=color, fontsize=12, weight="bold", va="top", alpha=0.08)
            body_artist = ax.text(0.02, y - 0.09, body, transform=ax.transAxes, color=WHITE, fontsize=10.5, va="top", alpha=0.08, wrap=True)
            texts.extend([(title_artist, i), (body_artist, i)])
    frames = frame_count(seconds, fps)

    def update(frame: int) -> list:
        stage = min(3, int(frame / max(1, frames - 1) * 4))
        for artist, idx in texts:
            artist.set_alpha(1.0 if idx < stage else (0.35 if idx == stage else 0.08))
        return [item[0] for item in texts]

    save_animation(fig, update, frames, out / "05_optimization_diagnostic_recap.mp4", fps, dpi)


FUNCTIONS = {
    "learning-rate": animate_learning_rate,
    "one-update": animate_one_update,
    "convergence": animate_convergence,
    "local-minima": animate_local_minima,
    "recap": animate_recap,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["all", *SCENES], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration of each MP4 scene")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps < 1 or args.dpi < 30:
        parser.error("seconds must be positive, fps >= 1, and dpi >= 30")
    selected = SCENES if args.scene == "all" else (args.scene,)
    for scene in selected:
        FUNCTIONS[scene](args.output_dir, args.fps, args.dpi, args.seconds)


if __name__ == "__main__":
    main()
