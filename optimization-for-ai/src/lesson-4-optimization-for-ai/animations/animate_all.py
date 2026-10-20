"""Generate five Lesson 4 educational animation clips as MP4 files.

The scripts generate videos only when run locally. FFmpeg must be installed and
available on PATH. Run from the project root after `pip install -e .`:

    python animations/animate_all.py --output-dir outputs --fps 24 --dpi 120 --speed 0.5 --duration 10

Scenes are illustrative teaching visualizations, not optimizer benchmarks.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = str(PROJECT_ROOT / "src")
if SOURCE_PATH in sys.path:
    sys.path.remove(SOURCE_PATH)
sys.path.insert(0, SOURCE_PATH)

from optimization_ai import (  # noqa: E402
    AdaGrad, Adam, Momentum, RMSProp, SGD,
    make_regression_data, mse, mse_gradient, optimizer_suite, train_minibatch,
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

X, Y = make_regression_data(n=96, seed=42)


def style_axis(ax: Axes, title: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, weight="bold", pad=9)


def canvas(nrows: int = 1, ncols: int = 2, *, figsize=(12.8, 7.2)) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.80, bottom=0.13, hspace=0.35, wspace=0.28)
    for ax in axes.flat:
        style_axis(ax)
    return fig, axes


def header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.075, 0.945, title, color=WHITE, fontsize=20, weight="bold", va="top")
    fig.text(0.075, 0.885, subtitle, color=MUTED, fontsize=10.5, va="top")


def plot_data_axis(ax: Axes, title: str = "") -> None:
    style_axis(ax, title)
    ax.scatter(X, Y, color=CYAN, s=27, alpha=0.8, edgecolor="none", label="Training data")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-4.5, 6.5)
    ax.set_xlabel("Normalized study-hours feature x")
    ax.set_ylabel("Illustrative target score y")


def save_animation(
    fig: Figure,
    update: Callable[[int], list],
    frames: int,
    path: Path,
    fps: int,
    dpi: int,
    speed: float,
    duration: float | None,
) -> None:
    if not writers.is_available("ffmpeg"):
        plt.close(fig)
        raise RuntimeError("FFmpeg writer unavailable. Install FFmpeg and ensure it is on PATH.")
    source_duration = frames / fps if duration is None else duration
    output_frames = max(2, int(round(source_duration / speed * fps)))

    def update_at_playback_speed(frame: int) -> list:
        source_frame = min(
            int(frame * (frames - 1) / (output_frames - 1)),
            frames - 1,
        )
        return update(source_frame)

    ani = FuncAnimation(
        fig,
        update_at_playback_speed,
        frames=output_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Created: {path}")


def minibatch_trace(batch_size: int = 12, epochs: int = 4, lr: float = 0.03, seed: int = 11):
    rng = np.random.default_rng(seed)
    params = np.array([0.0, 0.0])
    states = [params.copy()]
    indices_list: list[np.ndarray] = []
    losses = [mse(X, Y, params)]
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            ids = order[start:start + batch_size]
            grad = mse_gradient(X[ids], Y[ids], params)
            params = params - lr * grad
            states.append(params.copy())
            indices_list.append(ids.copy())
            losses.append(mse(X, Y, params))
    return np.asarray(states), indices_list, np.asarray(losses)


def animate_minibatch_mechanism(
    out: Path, fps: int, dpi: int, speed: float, duration: float | None
) -> None:
    """Scene 1: highlight one mini-batch, compute its gradient, then update line."""
    states, batch_ids, losses = minibatch_trace(batch_size=12, epochs=3, lr=0.03, seed=11)
    fig, axes = canvas(1, 2)
    ax_data, ax_loss = axes[0]
    header(fig, "Mini-batches: learn from a small group at a time", "Each update uses one subset of observations; the loss display is still evaluated on all data")
    style_axis(ax_data, "Highlight the current mini-batch")
    ax_data.set_xlim(-2.2, 2.2)
    ax_data.set_ylim(-4.5, 6.5)
    ax_data.set_xlabel("Normalized study-hours feature x")
    ax_data.set_ylabel("Illustrative target score y")
    regular = ax_data.scatter(X, Y, s=25, color=CYAN, alpha=0.28, edgecolor="none", label="Other observations")
    selected = ax_data.scatter([], [], s=70, color=AMBER, edgecolor=WHITE, linewidth=0.6, label="Selected mini-batch")
    line_x = np.linspace(-2.2, 2.2, 120)
    (line,) = ax_data.plot([], [], color=GREEN, lw=2.7, label="Current model")
    info = ax_data.text(0.04, 0.04, "", transform=ax_data.transAxes, color=WHITE, fontsize=9,
                        bbox={"boxstyle": "round,pad=0.5", "fc": "#111827", "ec": GRID})
    ax_data.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7, loc="upper left")
    style_axis(ax_loss, "Full-dataset MSE after each update")
    ax_loss.set_xlabel("Mini-batch update")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_xlim(0, len(losses)-1)
    ax_loss.set_ylim(0, losses[0]*1.08)
    ax_loss.plot(np.arange(len(losses)), losses, color=GRID, lw=1.2)
    (loss_line,) = ax_loss.plot([], [], color=GREEN, lw=2.2)
    (loss_point,) = ax_loss.plot([], [], "o", color=AMBER, ms=7)
    text = ax_loss.text(0.05, 0.92, "", transform=ax_loss.transAxes, color=WHITE, va="top", fontsize=11)

    fpu = 5
    total = len(batch_ids) * fpu + 1

    def update(frame: int) -> list:
        step = min(frame // fpu, len(batch_ids)-1)
        alpha = (frame % fpu) / fpu
        if frame == total - 1:
            step = len(batch_ids)-1
            alpha = 1.0
        p = (1-alpha)*states[step] + alpha*states[step+1]
        ids = batch_ids[step]
        selected.set_offsets(np.column_stack([X[ids], Y[ids]]))
        line.set_data(line_x, p[0]*line_x+p[1])
        current_loss = mse(X, Y, p)
        info.set_text(f"Batch size = {len(ids)} of {len(X)}\nUpdate = {step+1}/{len(batch_ids)}\nw = {p[0]:.3f}   b = {p[1]:.3f}")
        shown = step + alpha
        xs = np.arange(step+1)
        ys = losses[:step+1]
        xvals = np.append(xs, shown)
        yvals = np.append(ys, current_loss)
        loss_line.set_data(xvals, yvals)
        loss_point.set_data([shown], [current_loss])
        text.set_text(f"Current full-data MSE: {current_loss:.3f}\nOne gradient update from 12 examples")
        return [regular, selected, line, info, loss_line, loss_point, text]

    save_animation(fig, update, total, out/"01_minibatch_mechanism.mp4", fps, dpi, speed, duration)


def animate_batch_sizes(out: Path, fps: int, dpi: int, speed: float, duration: float | None) -> None:
    """Scene 2: compare update noise for batch sizes 1, 8, 32, and full batch."""
    configs = [("Batch size 1 (SGD)", 1, 0.003, CORAL),
               ("Mini-batch 8", 8, 0.02, AMBER),
               ("Mini-batch 32", 32, 0.025, PURPLE),
               ("Full batch 96", 96, 0.03, GREEN)]
    runs = {}
    for label, size, lr, _color in configs:
        opt = SGD(lr)
        runs[label] = train_minibatch(X, Y, opt, batch_size=size, epochs=24, seed=23)
    fig, axes = canvas(1, 2)
    ax_line, ax_loss = axes[0]
    header(fig, "Batch size changes how noisy each update is", "Same data and epochs; learning rates are illustrative, not a speed benchmark")
    plot_data_axis(ax_line, "Prediction lines at the current epoch")
    line_artists = {}
    for label, _size, _lr, color in configs:
        (artist,) = ax_line.plot([], [], color=color, lw=2.0, label=label)
        line_artists[label] = artist
    ax_line.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7, loc="upper left")
    info = ax_line.text(0.04, 0.04, "", transform=ax_line.transAxes, color=WHITE, fontsize=9,
                        bbox={"boxstyle": "round,pad=0.4", "fc": "#111827", "ec": GRID})
    style_axis(ax_loss, "Full-data MSE by epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_xlim(0, 24)
    y_max = max(float(hist.full_loss[0]) for hist in runs.values()) * 1.08
    ax_loss.set_ylim(0, y_max)
    loss_artists = {}
    point_artists = {}
    for label, _size, _lr, color in configs:
        (la,) = ax_loss.plot([], [], color=color, lw=2.0, label=label)
        (pa,) = ax_loss.plot([], [], "o", color=color, ms=5)
        loss_artists[label], point_artists[label] = la, pa
    ax_loss.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7)
    (epoch_text,) = (ax_loss.text(0.04, 0.94, "", transform=ax_loss.transAxes, color=WHITE, va="top", fontsize=10),)

    def update(frame: int) -> list:
        epoch = min(frame, 24)
        artists: list = [info, epoch_text]
        for label, _size, _lr, _color in configs:
            params = runs[label].epoch_end_params[epoch]
            line_artists[label].set_data(np.linspace(-2.2, 2.2, 120), params[0]*np.linspace(-2.2,2.2,120)+params[1])
            hist = runs[label].full_loss[:epoch+1]
            loss_artists[label].set_data(np.arange(len(hist)), hist)
            point_artists[label].set_data([epoch], [hist[-1]])
            artists.extend([line_artists[label], loss_artists[label], point_artists[label]])
        info.set_text(f"Epoch: {epoch}\nSmaller batches update more often\nLarge batches average more examples")
        epoch_text.set_text("Batch-size comparison (illustrative settings)")
        return artists

    save_animation(fig, update, 25, out/"02_batch_size_comparison.mp4", fps, dpi, speed, duration)


def _quadratic_path(momentum: bool, steps: int = 90) -> np.ndarray:
    """Illustrative optimizer paths on a narrow quadratic valley, not the regression loss."""
    eta = 0.055 if not momentum else 0.02
    beta = 0.82
    point = np.array([3.4, 1.15], dtype=float)
    velocity = np.zeros(2)
    path = [point.copy()]
    for _ in range(steps):
        grad = np.array([point[0], 30.0 * point[1]])
        if momentum:
            velocity = beta * velocity + grad
            point = point - eta * velocity
        else:
            point = point - eta * grad
        path.append(point.copy())
    return np.asarray(path)


def animate_momentum(out: Path, fps: int, dpi: int, speed: float, duration: float | None) -> None:
    """Scene 3: show momentum on a narrow, illustrative loss valley."""
    gd = _quadratic_path(False, 75)
    mom = _quadratic_path(True, 75)
    fig, axes = canvas(1, 2)
    ax_path, ax_formula = axes[0]
    header(fig, "Momentum uses the history of updates", "Illustrative narrow loss valley: compare a direct gradient step with a velocity that accumulates direction")
    style_axis(ax_path, "Parameter path on a loss landscape")
    u = np.linspace(-4.0, 4.0, 180)
    v = np.linspace(-1.5, 1.5, 180)
    U, V = np.meshgrid(u, v)
    Z = 0.5*(U**2 + 30.0*V**2)
    ax_path.contour(U, V, Z, levels=[0.5, 1, 2, 4, 8, 16, 32, 48], colors=[GRID], linewidths=0.8)
    (gd_line,) = ax_path.plot([], [], color=CORAL, lw=1.8, alpha=0.9, label="Gradient Descent")
    (mom_line,) = ax_path.plot([], [], color=GREEN, lw=2.0, alpha=0.95, label="Momentum")
    (gd_point,) = ax_path.plot([], [], "o", color=CORAL, ms=7)
    (mom_point,) = ax_path.plot([], [], "o", color=GREEN, ms=7)
    ax_path.scatter([0], [0], color=WHITE, marker="*", s=130, label="Minimum")
    ax_path.set_xlim(-4, 4); ax_path.set_ylim(-1.5, 1.5)
    ax_path.set_xlabel("Parameter u"); ax_path.set_ylabel("Parameter v")
    ax_path.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8, loc="upper right")
    style_axis(ax_formula, "What does momentum add?")
    ax_formula.axis("off")
    formula_text = ax_formula.text(0.04, 0.88, "", transform=ax_formula.transAxes, color=WHITE, fontsize=12, va="top", linespacing=1.5)
    stat_text = ax_formula.text(0.04, 0.30, "", transform=ax_formula.transAxes, color=AMBER, fontsize=11, va="top", linespacing=1.6)

    def update(frame: int) -> list:
        i = min(frame, len(gd)-1)
        gd_line.set_data(gd[:i+1, 0], gd[:i+1, 1])
        mom_line.set_data(mom[:i+1, 0], mom[:i+1, 1])
        gd_point.set_data([gd[i,0]], [gd[i,1]])
        mom_point.set_data([mom[i,0]], [mom[i,1]])
        formula_text.set_text("Without momentum:\n  θ ← θ − ηg\n\nWith momentum:\n  v ← βv + g\n  θ ← θ − ηv\n\nThe velocity carries information\nfrom recent gradients.")
        z_gd = 0.5*(gd[i,0]**2+30*gd[i,1]**2)
        z_mom = 0.5*(mom[i,0]**2+30*mom[i,1]**2)
        stat_text.set_text(f"GD loss: {z_gd:.3f}\nMomentum loss: {z_mom:.3f}\n\nIllustrative settings; not a\nuniversal speed benchmark.")
        return [gd_line, mom_line, gd_point, mom_point, formula_text, stat_text]

    save_animation(fig, update, len(gd), out/"03_momentum_loss_valley.mp4", fps, dpi, speed, duration)


def animate_adaptive_steps(out: Path, fps: int, dpi: int, speed: float, duration: float | None) -> None:
    """Scene 4: visualize update sizes for parameters with different gradient scales."""
    # Synthetic gradient sequence is deliberate: feature/parameter scales can differ.
    gradients = np.array([
        [4.0, 0.25], [3.0, -0.20], [2.0, 0.18], [1.5, -0.15],
        [1.0, 0.12], [0.8, -0.10], [0.5, 0.08], [0.3, -0.06],
        [0.2, 0.05], [0.12, -0.04], [0.08, 0.03], [0.05, -0.02],
    ])
    configs = {
        "SGD": (SGD(0.08), CYAN),
        "AdaGrad": (AdaGrad(0.18), AMBER),
        "RMSProp": (RMSProp(0.06, beta=0.9), PURPLE),
        "Adam": (Adam(0.06), GREEN),
    }
    deltas: dict[str, list[np.ndarray]] = {}
    params_by_name: dict[str, np.ndarray] = {}
    for name, (optimizer, _color) in configs.items():
        params = np.zeros(2)
        seq = []
        for g in gradients:
            new_params = optimizer.step(params, g)
            seq.append(new_params-params)
            params = new_params
        deltas[name] = seq
        params_by_name[name] = np.asarray(seq)

    fig, axes = canvas(1, 2)
    ax_steps, ax_rules = axes[0]
    header(fig, "Adaptive optimizers scale each parameter's step", "A conceptual gradient sequence with different coordinate scales—not a benchmark")
    style_axis(ax_steps, "Absolute parameter-update size")
    ax_steps.set_xlabel("Gradient update")
    ax_steps.set_ylabel("Absolute step size")
    ax_steps.set_xlim(1, len(gradients))
    max_y = max(float(np.max(np.abs(np.asarray(v)))) for v in deltas.values()) * 1.1
    ax_steps.set_ylim(0, max_y)
    lines = {}
    for name, (_opt, color) in configs.items():
        (line,) = ax_steps.plot([], [], color=color, lw=2, label=f"{name}: |Δw|")
        lines[name] = line
    ax_steps.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7, ncol=2)
    style_axis(ax_rules, "What each method remembers")
    ax_rules.axis("off")
    rule_txt = ax_rules.text(0.04, 0.9, "", transform=ax_rules.transAxes, color=WHITE, fontsize=11, va="top", linespacing=1.5)
    foot_txt = ax_rules.text(0.04, 0.22, "", transform=ax_rules.transAxes, color=MUTED, fontsize=9, va="top")

    def update(frame: int) -> list:
        k = min(frame+1, len(gradients))
        artists: list = [rule_txt, foot_txt]
        for name in configs:
            vals = np.abs(np.asarray(deltas[name])[:k, 0])
            lines[name].set_data(np.arange(1, k+1), vals)
            artists.append(lines[name])
        rule_txt.set_text(
            "SGD\n  Uses the current gradient.\n\n"
            "AdaGrad\n  Accumulates squared gradients;\n  frequently updated parameters\n  tend to receive smaller steps.\n\n"
            "RMSProp\n  Tracks a moving average of\n  squared gradients.\n\n"
            "Adam\n  Combines momentum-like first\n  moments with adaptive scaling."
        )
        foot_txt.set_text(f"Gradient update {k}/{len(gradients)}\nConceptual gradients: g=[large, small] initially.")
        return artists

    save_animation(
        fig, update, len(gradients), out/"04_adaptive_optimizer_steps.mp4",
        fps, dpi, speed, duration,
    )


def animate_optimizer_comparison(
    out: Path, fps: int, dpi: int, speed: float, duration: float | None
) -> None:
    """Scene 5: compare loss curves and fitted predictions for the same mini-batches."""
    suite = optimizer_suite()
    histories = {}
    colors = {"Mini-batch SGD": CYAN, "Momentum": AMBER, "AdaGrad": PURPLE, "RMSProp": CORAL, "Adam": GREEN}
    for name, (optimizer, batch_size) in suite.items():
        histories[name] = train_minibatch(X, Y, optimizer, batch_size=batch_size, epochs=60, seed=42)

    fig, axes = canvas(1, 2)
    ax_model, ax_loss = axes[0]
    header(fig, "Same data, different optimizer behavior", "Shared data, batch size and shuffle seed; hyperparameters remain illustrative")
    plot_data_axis(ax_model, "Prediction line after each epoch")
    line_artists = {}
    line_x = np.linspace(-2.2, 2.2, 120)
    for name, color in colors.items():
        (line,) = ax_model.plot([], [], color=color, lw=1.9, label=name)
        line_artists[name] = line
    ax_model.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7, loc="upper left")
    info = ax_model.text(0.04, 0.04, "", transform=ax_model.transAxes, color=WHITE, fontsize=9,
                         bbox={"boxstyle": "round,pad=0.4", "fc": "#111827", "ec": GRID})
    style_axis(ax_loss, "Full-dataset MSE")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_xlim(0, 60)
    max_loss = max(float(h.full_loss[0]) for h in histories.values())*1.08
    ax_loss.set_ylim(0, max_loss)
    loss_lines, loss_points = {}, {}
    for name, color in colors.items():
        (line,) = ax_loss.plot([], [], color=color, lw=2.0, label=name)
        (point,) = ax_loss.plot([], [], "o", color=color, ms=5)
        loss_lines[name] = line
        loss_points[name] = point
    ax_loss.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=7)

    def update(frame: int) -> list:
        epoch = min(frame, 60)
        artists: list = [info]
        for name, hist in histories.items():
            params = hist.epoch_end_params[epoch]
            line_artists[name].set_data(line_x, params[0]*line_x+params[1])
            ys = hist.full_loss[:epoch+1]
            loss_lines[name].set_data(np.arange(len(ys)), ys)
            loss_points[name].set_data([epoch], [ys[-1]])
            artists.extend([line_artists[name], loss_lines[name], loss_points[name]])
        info.set_text(f"Epoch {epoch}\nCompare update rules, not universal rankings.")
        return artists

    save_animation(fig, update, 61, out/"05_optimizer_comparison.mp4", fps, dpi, speed, duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lesson 4 optimizer animations.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Playback speed multiplier: 0.5 is half speed, 1 is normal speed (default: 0.5).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Target duration in seconds per clip at normal speed; actual duration is duration / speed.",
    )
    parser.add_argument("--scene", choices=["all", "minibatches", "batch-sizes", "momentum", "adaptive", "comparison"], default="all")
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 50:
        parser.error("--fps must be positive and --dpi must be at least 50.")
    if not np.isfinite(args.speed) or not 0.1 <= args.speed <= 4.0:
        parser.error("--speed must be between 0.1 and 4.0.")
    if args.duration is not None and (
        not np.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be a positive number of seconds.")
    scene_functions = {
        "minibatches": animate_minibatch_mechanism,
        "batch-sizes": animate_batch_sizes,
        "momentum": animate_momentum,
        "adaptive": animate_adaptive_steps,
        "comparison": animate_optimizer_comparison,
    }
    names = list(scene_functions) if args.scene == "all" else [args.scene]
    for name in names:
        scene_functions[name](args.output_dir, args.fps, args.dpi, args.speed, args.duration)


if __name__ == "__main__":
    main()
