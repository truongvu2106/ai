"""Generate five Lesson 6 MP4 scenes locally; no MP4s are rendered on import.

Scenes: convexity, convex-paths, nonconvex-paths, saddle, nn-recap.
Requires FFmpeg on PATH. Numerical functions are imported from the same module
used by examples/run_experiments.py and tests/test_objectives.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

SCENES = ("convexity", "convex-paths", "nonconvex-paths", "saddle", "nn-recap")
NONCONVEX_STARTS = [((-1.8, 1.4), CYAN, "Start (-1.8, 1.4)"), ((1.8, 1.2), AMBER, "Start (1.8, 1.2)")]
CONVEX_STARTS = [((-2.0, 1.5), CYAN, "Start A"), ((2.0, 1.25), AMBER, "Start B"), ((-1.0, -1.5), CORAL, "Start C")]


def style_axis(ax: Axes, title: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10.5, weight="bold", pad=8)


def new_canvas(nrows: int = 1, ncols: int = 2, figsize: tuple[float, float] = (12.8, 7.2)) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12, hspace=0.34, wspace=0.28)
    for ax in axes.flat:
        style_axis(ax)
    return fig, axes


def header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.955, title, color=WHITE, fontsize=18, weight="bold", va="top")
    fig.text(0.07, 0.905, subtitle, color=MUTED, fontsize=9.5, va="top")


def save_animation(fig: Figure, update: Callable[[int], list], frames: int, path: Path, fps: int, dpi: int) -> None:
    if not writers.is_available("ffmpeg"):
        plt.close(fig)
        raise RuntimeError("FFmpeg writer unavailable. Install FFmpeg and ensure ffmpeg is on PATH.")
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2800,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Created: {path}")


def frame_count(seconds: float, fps: int) -> int:
    return max(2, int(round(seconds * fps)))


def mesh() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2.3, 2.3, 220)
    y = np.linspace(-2.0, 2.0, 190)
    return np.meshgrid(x, y)


def plot_contours(ax: Axes, objective: Callable[[float, float], float], title: str, levels: int = 35, cmap: str = "viridis") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = mesh()
    if objective is convex_objective:
        Z = (X - 0.5) ** 2 + 0.5 * (Y + 0.25) ** 2
    elif objective is nonconvex_objective:
        Z = (X**2 - 1.0) ** 2 + 0.5 * Y**2 + 0.2 * X
    else:
        Z = np.vectorize(objective)(X, Y)
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.contour(X, Y, Z, levels=12, colors=WHITE, linewidths=0.35, alpha=0.22)
    ax.set_xlabel("Parameter x")
    ax.set_ylabel("Parameter y")
    style_axis(ax, title)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.0, 2.0)
    return X, Y, Z


def animate_convexity(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 1: convex chord geometry and Jensen inequality on one cross-section."""
    fig, axes = new_canvas(1, 2)
    ax, notes = axes[0]
    header(fig, "Convexity: the bowl stays below its chords", "On this cross-section, every chord lies on or above the function")
    xx = np.linspace(-2.0, 3.0, 500)
    ff = (xx - 0.5) ** 2
    style_axis(ax, "Convex slice: f(x) = (x − 0.5)²")
    ax.plot(xx, ff, color=CYAN, lw=2.5)
    a, b, alpha = -1.0, 2.5, 0.4
    fa, fb = (a - 0.5) ** 2, (b - 0.5) ** 2
    z = alpha * a + (1 - alpha) * b
    fz = (z - 0.5) ** 2
    chord = alpha * fa + (1 - alpha) * fb
    ax.plot([a, b], [fa, fb], color=AMBER, lw=1.8, ls="--", label="Chord")
    ax.scatter([a, b], [fa, fb], color=AMBER, s=55, zorder=5)
    ax.scatter([z], [fz], color=GREEN, s=75, edgecolors=WHITE, zorder=6)
    chord_dot, = ax.plot([], [], "o", color=CORAL, ms=8, mec=WHITE, mew=.6, zorder=7)
    connector, = ax.plot([], [], color=CORAL, lw=1.3, ls=":")
    ax.set_xlim(-2, 3.1)
    ax.set_ylim(-0.5, 11)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8)
    notes.set_axis_off()
    note = notes.text(0.02, .95, "", transform=notes.transAxes, color=WHITE, fontsize=13, va="top", linespacing=1.6)
    detail = notes.text(0.02, .34, "", transform=notes.transAxes, color=MUTED, fontsize=11, va="top", linespacing=1.6)
    frames = frame_count(seconds, fps)

    def update(frame: int) -> list:
        p = frame / max(1, frames - 1)
        alpha_now = 0.15 + 0.7 * p
        point_x = alpha_now * a + (1 - alpha_now) * b
        point_y = alpha_now * fa + (1 - alpha_now) * fb
        curve_y = (point_x - 0.5) ** 2
        chord_dot.set_data([point_x], [point_y])
        connector.set_data([point_x, point_x], [curve_y, point_y])
        note.set_text("Convexity means:\nf(αa + (1−α)b) ≤ αf(a) + (1−α)f(b)\n\nThe curve never rises above the chord.")
        detail.set_text(
            f"a={a:g}, b={b:g}, α={alpha_now:.2f}\n"
            f"Combined point z={point_x:.2f}\n"
            f"f(z)={curve_y:.2f} ≤ chord value {point_y:.2f}\n\n"
            "For a convex objective, every local minimum\nis also a global minimum."
        )
        return [chord_dot, connector, note, detail]

    save_animation(fig, update, frames, out / "01_convexity_geometry.mp4", fps, dpi)


def animate_convex_paths(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 2: different initializations converge to one global minimum on a convex objective."""
    runs = [run_gradient_descent_2d(convex_objective, convex_gradient, start, 0.2, max_steps=100, tolerance=1e-6)
            for start, _, _ in CONVEX_STARTS]
    fig, axes = new_canvas(1, 2)
    ax, notes = axes[0]
    header(fig, "Convex objective: one global destination", "Different starts; same strictly convex quadratic; one unique minimizer")
    plot_contours(ax, convex_objective, "J(x,y) = (x − 0.5)² + 0.5(y + 0.25)²", cmap="viridis")
    ax.scatter([0.5], [-0.25], marker="*", s=190, color=GREEN, edgecolors=WHITE, linewidths=.7, zorder=6)
    lines, dots = [], []
    for (_, color, label) in CONVEX_STARTS:
        line, = ax.plot([], [], color=color, lw=1.8, label=label)
        dot, = ax.plot([], [], "o", color=color, ms=8, mec=WHITE, mew=.6)
        lines.append(line); dots.append(dot)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8)
    notes.set_axis_off()
    note = notes.text(.02, .95, "", transform=notes.transAxes, color=WHITE, fontsize=12.5, va="top", linespacing=1.6)
    detail = notes.text(.02, .37, "", transform=notes.transAxes, color=MUTED, fontsize=10.5, va="top", linespacing=1.6)
    frames = frame_count(seconds, fps)
    max_len = max(len(r.points) for r in runs)

    def update(frame: int) -> list:
        fraction = frame / max(1, frames - 1)
        step = int(fraction * (max_len - 1))
        artists = [note, detail]
        for i, run in enumerate(runs):
            idx = min(step, len(run.points) - 1)
            pts = run.points[:idx + 1]
            lines[i].set_data(pts[:, 0], pts[:, 1])
            dots[i].set_data([pts[-1, 0]], [pts[-1, 1]])
            artists.extend([lines[i], dots[i]])
        note.set_text("The objective has one unique global minimum.\n\nGradient descent from each shown start\napproaches the same point when the\nalgorithm's settings are suitable.")
        detail.set_text("J(x,y)=(x−0.5)²+0.5(y+0.25)²\nGlobal minimum: (0.5, −0.25)\nMinimum objective value: 0\n\nConvexity rules out a worse local minimum.")
        return artists

    save_animation(fig, update, frames, out / "02_convex_global_minimum.mp4", fps, dpi)


def animate_nonconvex_paths(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 3: trajectories reach a global or a higher local minimum."""
    runs = [run_gradient_descent_2d(nonconvex_objective, nonconvex_gradient, start, 0.02,
                                    max_steps=1300, tolerance=1e-6)
            for start, _, _ in NONCONVEX_STARTS]
    stationary = nonconvex_stationary_points()
    global_min = stationary["global_minimum"]
    local_min = stationary["local_minimum"]
    saddle = stationary["saddle"]
    fig, axes = new_canvas(1, 2)
    ax, notes = axes[0]
    header(fig, "Non-convex objective: multiple valleys", "The starting point can affect which minimum gradient descent reaches")
    plot_contours(ax, nonconvex_objective, "f(x,y)=(x²−1)²+0.5y²+0.2x", cmap="magma")
    ax.scatter([global_min["x"]], [0], marker="*", s=180, color=GREEN, edgecolors=WHITE, linewidths=.7, zorder=6, label="Global minimum")
    ax.scatter([local_min["x"]], [0], marker="*", s=150, color=AMBER, edgecolors=WHITE, linewidths=.7, zorder=6, label="Higher local minimum")
    ax.scatter([saddle["x"]], [0], marker="X", s=80, color=CORAL, edgecolors=WHITE, linewidths=.7, zorder=6, label="Saddle point")
    lines, dots = [], []
    for (_, color, label) in NONCONVEX_STARTS:
        line, = ax.plot([], [], color=color, lw=1.8, label=label)
        dot, = ax.plot([], [], "o", color=color, ms=8, mec=WHITE, mew=.6)
        lines.append(line); dots.append(dot)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7.4, loc="upper center")
    notes.set_axis_off()
    note = notes.text(.02, .95, "", transform=notes.transAxes, color=WHITE, fontsize=12, va="top", linespacing=1.55)
    detail = notes.text(.02, .35, "", transform=notes.transAxes, color=MUTED, fontsize=10.2, va="top", linespacing=1.55)
    frames = frame_count(seconds, fps)
    max_len = max(len(r.points) for r in runs)

    def update(frame: int) -> list:
        step = int((frame / max(1, frames - 1)) * (max_len - 1))
        artists = [note, detail]
        for i, run in enumerate(runs):
            idx = min(step, len(run.points) - 1)
            pts = run.points[:idx + 1]
            lines[i].set_data(pts[:, 0], pts[:, 1])
            dots[i].set_data([pts[-1, 0]], [pts[-1, 1]])
            artists.extend([lines[i], dots[i]])
        note.set_text("Unlike the convex example, this landscape\nhas minima with different objective values.\n\nA run can converge to a nearby local minimum\nwithout finding the global minimum.")
        detail.set_text(
            f"Global: x≈{float(global_min['x']):.3f}, f≈{float(global_min['value']):.3f}\n"
            f"Local:  x≈{float(local_min['x']):.3f}, f≈{float(local_min['value']):.3f}\n"
            f"Saddle: x≈{float(saddle['x']):.3f}, f≈{float(saddle['value']):.3f}\n\n"
            "Teaching function only—not a real neural-network loss surface."
        )
        return artists

    save_animation(fig, update, frames, out / "03_nonconvex_multiple_minima.mp4", fps, dpi)


def animate_saddle(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 4: gradient can be zero at a saddle that is not a minimum."""
    saddle = nonconvex_stationary_points()["saddle"]
    sx = float(saddle["x"])
    base = nonconvex_objective(sx, 0.0)
    hxx = float(saddle["hessian_xx"])
    hyy = float(saddle["hessian_yy"])
    fig, axes = new_canvas(1, 2)
    ax, cross = axes[0]
    header(fig, "A saddle point is not a minimum", "The gradient can be near zero while some nearby directions still decrease the objective")
    plot_contours(ax, nonconvex_objective, "Same non-convex teaching objective", cmap="magma")
    ax.scatter([sx], [0], marker="X", s=130, color=CORAL, edgecolors=WHITE, linewidths=.8, zorder=7)
    ax.annotate("Saddle: ∇f ≈ 0", (sx, 0), xytext=(0.7, 0.65), color=WHITE,
                arrowprops={"arrowstyle": "->", "color": CORAL}, fontsize=9)
    cross.set_facecolor(BG)
    style_axis(cross, "Move through the saddle along two directions")
    delta = np.linspace(-0.75, 0.75, 300)
    fx = np.array([nonconvex_objective(sx + d, 0.0) for d in delta])
    fy = np.array([nonconvex_objective(sx, d) for d in delta])
    fx_rel = fx - base
    fy_rel = fy - base
    cross.plot(delta, fx_rel, color=CORAL, lw=2.2, label="Change along x (downhill on both sides)")
    cross.plot(delta, fy_rel, color=CYAN, lw=2.2, label="Change along y (uphill on both sides)")
    moving_x, = cross.plot([], [], "o", color=CORAL, ms=8, mec=WHITE, mew=.5)
    moving_y, = cross.plot([], [], "o", color=CYAN, ms=8, mec=WHITE, mew=.5)
    cross.axhline(0, color=MUTED, linewidth=.7, alpha=.6)
    cross.axvline(0, color=MUTED, linewidth=.7, alpha=.6)
    cross.set_xlabel("Displacement from saddle")
    cross.set_ylabel("Change in objective from saddle")
    cross.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7.1, loc="upper center")
    note = fig.text(.07, .045, "", color=MUTED, fontsize=10)
    frames = frame_count(seconds, fps)

    def update(frame: int) -> list:
        phase = frame / max(1, frames - 1)
        d = -0.65 + 1.3 * phase
        ix = int(np.argmin(np.abs(delta - d)))
        moving_x.set_data([delta[ix]], [fx_rel[ix]])
        moving_y.set_data([delta[ix]], [fy_rel[ix]])
        note.set_text(
            f"Critical point x≈{sx:.6f}; objective≈{base:.6f}; gradient≈0. "
            f"Hessian directions: curvature x={hxx:.3f}, y={hyy:.1f}."
        )
        return [moving_x, moving_y, note]

    save_animation(fig, update, frames, out / "04_saddle_point.mp4", fps, dpi)


def animate_nn_recap(out: Path, fps: int, dpi: int, seconds: float) -> None:
    """Scene 5: connect teaching landscapes to the challenges of neural-network training."""
    fig, axes = new_canvas(1, 2)
    ax, notes = axes[0]
    header(fig, "Why neural-network optimization is harder", "Many interacting parameters create a high-dimensional objective")
    plot_contours(ax, nonconvex_objective, "Low-dimensional teaching analogy only", cmap="magma")
    starts = [((-1.8, 1.4), CYAN, "trajectory A"), ((1.8, 1.2), AMBER, "trajectory B")]
    runs = [run_gradient_descent_2d(nonconvex_objective, nonconvex_gradient, s, .02,
                                    max_steps=1300, tolerance=1e-6) for s, _, _ in starts]
    lines, dots = [], []
    for (_, color, label) in starts:
        line, = ax.plot([], [], color=color, lw=1.7, label=label)
        dot, = ax.plot([], [], "o", color=color, ms=8, mec=WHITE, mew=.5)
        lines.append(line); dots.append(dot)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8)
    notes.set_axis_off()
    headline = notes.text(.02, .95, "", transform=notes.transAxes, color=WHITE, fontsize=12.3, va="top", linespacing=1.6)
    details = notes.text(.02, .43, "", transform=notes.transAxes, color=MUTED, fontsize=10.5, va="top", linespacing=1.55)
    frames = frame_count(seconds, fps)
    max_len = max(len(run.points) for run in runs)

    def update(frame: int) -> list:
        step = int(frame / max(1, frames - 1) * (max_len - 1))
        artists = [headline, details]
        for i, run in enumerate(runs):
            idx = min(step, len(run.points) - 1)
            points = run.points[:idx + 1]
            lines[i].set_data(points[:, 0], points[:, 1])
            dots[i].set_data([points[-1, 0]], [points[-1, 1]])
            artists.extend([lines[i], dots[i]])
        headline.set_text("Neural-network training can be challenging because:\n\n• many parameters interact\n• directions can have different curvature\n• flat regions and saddle points can slow progress\n• objective functions are often non-convex")
        details.set_text("A contour plot like this is only a simplified analogy.\nReal networks have far more parameters; this is not\na measured or complete neural-network loss surface.\n\nConvexity helps explain guarantees—but non-convex\ndoes not mean optimization must fail.")
        return artists

    save_animation(fig, update, frames, out / "05_neural_network_optimization_recap.mp4", fps, dpi)


GENERATORS = {
    "convexity": animate_convexity,
    "convex-paths": animate_convex_paths,
    "nonconvex-paths": animate_nonconvex_paths,
    "saddle": animate_saddle,
    "nn-recap": animate_nn_recap,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["all", *SCENES], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps < 1 or args.dpi < 50:
        parser.error("seconds and fps must be positive; dpi must be at least 50")
    selected = list(GENERATORS) if args.scene == "all" else [args.scene]
    for scene in selected:
        GENERATORS[scene](args.output_dir, args.fps, args.dpi, args.seconds)


if __name__ == "__main__":
    main()
