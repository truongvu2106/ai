"""Generate five Lesson 3 animation clips as MP4 files.

This script generates videos only when you run it. No MP4s are bundled by default.
Requires FFmpeg installed and available on PATH.

Run from the project root after `pip install -e .`:
    python animations/animate_all.py --output-dir outputs --fps 24 --dpi 120 --slowdown 2
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

# Make `python animations/animate_all.py` work without changing directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization_ai import batch_gradient, mean_squared_error, train_batch_gd, train_sgd

# Channel palette
BG = "#0B0F19"
GRID = "#1E293B"
CYAN = "#38BDF8"
AMBER = "#F59E0B"
CORAL = "#FB7185"
GREEN = "#10B981"
PURPLE = "#A855F7"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"

X = np.array([1.0, 2.0, 3.0])
Y = np.array([3.0, 5.0, 7.0])


def style_axis(ax: Axes, *, title: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if title:
        ax.set_title(title, color=WHITE, fontsize=12, pad=10, weight="bold")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def make_figure(nrows: int = 1, ncols: int = 1, *, figsize=(12, 6.75)) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.13, wspace=0.28, hspace=0.34)
    for ax in axes.flat:
        style_axis(ax)
    return fig, axes


def add_header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.94, title, color=WHITE, fontsize=20, weight="bold", va="top")
    fig.text(0.08, 0.875, subtitle, color=MUTED, fontsize=10.5, va="top")


def set_canvas(ax: Axes) -> None:
    ax.set_xlim(0.7, 3.3)
    ax.set_ylim(-0.3, 8.2)
    ax.set_xticks(X)
    ax.set_xlabel("Input x (illustrative study hours)")
    ax.set_ylabel("Target / prediction")


def draw_predictions(ax: Axes, w: float, b: float, *, title: str = "") -> tuple[list, list]:
    """Draw target points, prediction line, and vertical residual segments."""
    ax.clear()
    style_axis(ax, title=title)
    set_canvas(ax)
    ax.scatter(X, Y, s=90, color=CYAN, edgecolors=WHITE, linewidths=0.7, zorder=5, label="Target y")
    line_x = np.linspace(0.7, 3.3, 100)
    line_y = w * line_x + b
    (line,) = ax.plot(line_x, line_y, color=AMBER, linewidth=2.7, label="Prediction ŷ = wx + b")
    pred = w * X + b
    residuals = []
    for xi, yi, pi in zip(X, Y, pred):
        residuals.append(ax.plot([xi, xi], [yi, pi], color=CORAL, linewidth=2.0, alpha=0.95)[0])
    ax.scatter(X, pred, color=AMBER, s=45, marker="s", zorder=6, label="Predictions")
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8, loc="upper left")
    return [line, *residuals, *leg.get_texts()]


def save_animation(
    fig: Figure,
    update: Callable[[int], list],
    frames: int,
    output_path: Path,
    *,
    fps: int,
    dpi: int,
    slowdown: int,
    interval: int = 1000 // 24,
) -> None:
    animation = FuncAnimation(
        fig,
        lambda frame: update(frame // slowdown),
        frames=frames * slowdown,
        interval=interval,
        blit=False,
        repeat=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Created: {output_path}")


def animate_problem_hook(output_dir: Path, fps: int, dpi: int, slowdown: int) -> None:
    """Part 1: show the model improving and the whole-data MSE falling."""
    result = train_batch_gd(X, Y, learning_rate=0.05, steps=18)
    fig, axes = make_figure(1, 2, figsize=(12, 6.75))
    ax_model, ax_loss = axes[0]
    add_header(fig, "How can a model learn from its errors?", "Same data • changing parameters • measurable improvement")
    set_canvas(ax_model)
    ax_model.scatter(X, Y, s=90, color=CYAN, edgecolors=WHITE, linewidths=0.7, zorder=5, label="Target y")
    (pred_line,) = ax_model.plot([], [], color=AMBER, lw=2.8, label="Prediction")
    residual_artists = [ax_model.plot([], [], color=CORAL, lw=2.0)[0] for _ in X]
    (pred_points,) = ax_model.plot([], [], "s", color=AMBER, ms=7)
    ax_model.set_xlim(0.7, 3.3); ax_model.set_ylim(-0.5, 8.2)
    ax_model.set_xticks(X); ax_model.set_xlabel("Input x"); ax_model.set_ylabel("Target / prediction")
    ax_model.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8, loc="upper left")
    info = ax_model.text(0.04, 0.05, "", transform=ax_model.transAxes, color=WHITE, fontsize=10,
                         bbox={"boxstyle": "round,pad=0.5", "fc": "#111827", "ec": GRID})
    style_axis(ax_loss, title="Full-dataset MSE")
    ax_loss.set_xlabel("Batch update")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_xlim(0, len(result.loss_history)-1)
    ax_loss.set_ylim(0, result.loss_history[0] * 1.08)
    ax_loss.plot(np.arange(len(result.loss_history)), result.loss_history, color=GRID, lw=1.5, alpha=0.95)
    (loss_point,) = ax_loss.plot([], [], "o", color=GREEN, ms=8)
    loss_text = ax_loss.text(0.07, 0.92, "", transform=ax_loss.transAxes, color=GREEN, fontsize=12, weight="bold", va="top")

    frames_per_step = 5
    total_frames = (len(result.w_history)-1) * frames_per_step + frames_per_step

    def update(frame: int) -> list:
        step = min(frame // frames_per_step, len(result.w_history)-2)
        alpha = (frame % frames_per_step) / frames_per_step
        if frame >= (len(result.w_history)-1) * frames_per_step:
            step = len(result.w_history)-1
            w, b = result.w_history[step], result.b_history[step]
            loss = result.loss_history[step]
        else:
            w = (1-alpha)*result.w_history[step] + alpha*result.w_history[step+1]
            b = (1-alpha)*result.b_history[step] + alpha*result.b_history[step+1]
            loss = mean_squared_error(X, Y, w, b)
        pred = w*X+b
        pred_line.set_data(np.linspace(0.7,3.3,100), w*np.linspace(0.7,3.3,100)+b)
        pred_points.set_data(X, pred)
        for artist, xi, yi, pi in zip(residual_artists, X, Y, pred):
            artist.set_data([xi, xi], [yi, pi])
        info.set_text(f"w = {w:.3f}\nb = {b:.3f}\nMSE = {loss:.3f}")
        current_step = min(frame // frames_per_step, len(result.loss_history)-1)
        loss_point.set_data([current_step], [result.loss_history[current_step]])
        loss_text.set_text(f"MSE = {loss:.3f}")
        return [pred_line, *residual_artists, pred_points, info, loss_point, loss_text]

    save_animation(
        fig, update, total_frames, output_dir/"01_problem_hook.mp4",
        fps=fps, dpi=dpi, slowdown=slowdown,
    )


def animate_loss_curve(output_dir: Path, fps: int, dpi: int, slowdown: int) -> None:
    """Part 2: explain a gradient step on the 1D loss curve with b fixed at 1."""
    eta = 0.05
    w = 0.3
    history = [w]
    for _ in range(12):
        grad = (28.0/3.0) * (w-2.0)
        w = w - eta*grad
        history.append(w)
    curve_w = np.linspace(0, 3.4, 400)
    curve_loss = (14.0/3.0) * (curve_w-2.0)**2
    fig, axes = make_figure(1, 2, figsize=(12, 6.75))
    ax_curve, ax_model = axes[0]
    add_header(fig, "Gradient Descent: follow the downhill direction", "For this scene, b is fixed at 1 so we can see one parameter clearly")
    style_axis(ax_curve, title="Loss as a function of w")
    ax_curve.plot(curve_w, curve_loss, color=PURPLE, lw=3)
    ax_curve.axvline(2, color=GREEN, linestyle="--", alpha=0.8, label="Minimum at w = 2")
    ax_curve.set_xlabel("Parameter w"); ax_curve.set_ylabel("MSE J(w)")
    ax_curve.set_xlim(0,3.4); ax_curve.set_ylim(0, 20)
    ax_curve.legend(facecolor=BG, edgecolor=GRID, labelcolor=WHITE, fontsize=8)
    (point,) = ax_curve.plot([], [], "o", color=AMBER, ms=10, zorder=5)
    (tangent,) = ax_curve.plot([], [], color=CORAL, lw=2, alpha=0.9)
    (step_arrow,) = ax_curve.plot([], [], color=CYAN, lw=2.5, marker=">", markevery=[1])
    formula = ax_curve.text(0.05, 0.92, "", transform=ax_curve.transAxes, color=WHITE, va="top", fontsize=10,
                            bbox={"boxstyle":"round,pad=0.45", "fc":"#111827", "ec":GRID})

    set_canvas(ax_model)
    ax_model.set_ylim(-0.4,8.2)
    ax_model.scatter(X,Y,s=80,color=CYAN,edgecolors=WHITE,linewidths=0.6,zorder=5,label="Target y")
    line_x = np.linspace(0.7,3.3,100)
    (line,) = ax_model.plot([],[],color=AMBER,lw=2.8,label="ŷ = wx + 1")
    pred_points = ax_model.scatter([],[],color=AMBER,s=45,marker="s",label="Predictions")
    ax_model.set_title("Same parameter update changes the model", color=WHITE, fontsize=12, pad=10, weight="bold")
    ax_model.legend(facecolor=BG,edgecolor=GRID,labelcolor=WHITE,fontsize=8,loc="upper left")
    stats = ax_model.text(0.04,0.05,"",transform=ax_model.transAxes,color=WHITE,fontsize=10,
                          bbox={"boxstyle":"round,pad=0.45","fc":"#111827","ec":GRID})

    def update(frame: int) -> list:
        idx = min(frame // 6, len(history)-2)
        alpha = (frame % 6)/6
        if frame >= (len(history)-1)*6:
            idx = len(history)-1; current_w = history[idx]; next_w = current_w
        else:
            current_w = (1-alpha)*history[idx]+alpha*history[idx+1]
            next_w = history[idx+1]
        loss = (14.0/3.0)*(current_w-2.0)**2
        grad = (28.0/3.0)*(current_w-2.0)
        point.set_data([current_w],[loss])
        tangent_x = np.array([max(0.0,current_w-0.35), min(3.4,current_w+0.35)])
        tangent_y = loss + grad*(tangent_x-current_w)
        tangent.set_data(tangent_x,tangent_y)
        arrow_x = np.array([current_w,next_w])
        arrow_y = (14.0/3.0)*(arrow_x-2.0)**2
        step_arrow.set_data(arrow_x,arrow_y)
        formula.set_text(f"wₜ₊₁ = wₜ − η · dJ/dw\nη = {eta:.2f}   gradient = {grad:.2f}\nJ(w) = {loss:.3f}")
        pred = current_w*X+1.0
        line.set_data(line_x,current_w*line_x+1.0)
        pred_points.set_offsets(np.column_stack((X,pred)))
        stats.set_text(f"w = {current_w:.3f}\nb = 1.000 (fixed)\nMSE = {mean_squared_error(X,Y,current_w,1.0):.3f}")
        return [point,tangent,step_arrow,formula,line,pred_points,stats]

    save_animation(
        fig, update, len(history)*6, output_dir/"02_gradient_on_loss_curve.mp4",
        fps=fps, dpi=dpi, slowdown=slowdown,
    )


def animate_batch_gd(output_dir: Path, fps: int, dpi: int, slowdown: int) -> None:
    """Part 3: full-batch GD repeatedly uses every training observation."""
    result = train_batch_gd(X,Y,learning_rate=0.05,steps=70)
    fig, axes = make_figure(1,2,figsize=(12,6.75))
    ax_model, ax_loss = axes[0]
    add_header(fig,"Batch Gradient Descent","Each update calculates the gradient using all training observations")
    ax_model.scatter(X,Y,s=90,color=CYAN,edgecolors=WHITE,linewidths=0.6,zorder=5,label="Targets")
    line_x=np.linspace(0.7,3.3,100)
    (line,)=ax_model.plot([],[],color=AMBER,lw=2.7,label="Current model")
    pred_points=ax_model.scatter([],[],s=45,marker="s",color=AMBER,label="Predictions")
    residuals=[ax_model.plot([],[],color=CORAL,lw=2.0)[0] for _ in X]
    set_canvas(ax_model)
    ax_model.set_ylim(-0.5,8.2)
    ax_model.legend(facecolor=BG,edgecolor=GRID,labelcolor=WHITE,fontsize=8,loc="upper left")
    info=ax_model.text(0.04,0.05,"",transform=ax_model.transAxes,color=WHITE,fontsize=10,
                       bbox={"boxstyle":"round,pad=0.45","fc":"#111827","ec":GRID})
    style_axis(ax_loss,title="MSE on all three observations")
    ax_loss.set_xlabel("Batch update"); ax_loss.set_ylabel("MSE")
    ax_loss.set_xlim(0,len(result.loss_history)-1); ax_loss.set_ylim(0,result.loss_history[0]*1.05)
    ax_loss.plot(np.arange(len(result.loss_history)),result.loss_history,color=GRID,lw=1.4)
    (curve,)=ax_loss.plot([],[],color=GREEN,lw=2.4)
    (marker,)=ax_loss.plot([],[],"o",color=AMBER,ms=7)
    frames_per_step=3
    total=(len(result.w_history)-1)*frames_per_step+1

    def update(frame: int) -> list:
        idx=min(frame//frames_per_step,len(result.w_history)-1)
        if idx < len(result.w_history)-1 and frame%frames_per_step:
            alpha=(frame%frames_per_step)/frames_per_step
            w=(1-alpha)*result.w_history[idx]+alpha*result.w_history[idx+1]
            b=(1-alpha)*result.b_history[idx]+alpha*result.b_history[idx+1]
            loss=mean_squared_error(X,Y,w,b)
        else:
            w,b,loss=result.w_history[idx],result.b_history[idx],result.loss_history[idx]
        pred=w*X+b
        line.set_data(line_x,w*line_x+b); pred_points.set_offsets(np.column_stack((X,pred)))
        for artist,xi,yi,pi in zip(residuals,X,Y,pred): artist.set_data([xi,xi],[yi,pi])
        info.set_text(f"update = {idx}\nw = {w:.3f}\nb = {b:.3f}\nMSE = {loss:.4f}")
        upto=min(idx+1,len(result.loss_history))
        curve.set_data(np.arange(upto),result.loss_history[:upto])
        marker.set_data([idx],[result.loss_history[idx]])
        return [line,pred_points,*residuals,info,curve,marker]

    save_animation(
        fig, update, total, output_dir/"03_batch_gradient_descent.mp4",
        fps=fps, dpi=dpi, slowdown=slowdown,
    )


def animate_sgd(output_dir: Path, fps: int, dpi: int, slowdown: int) -> None:
    """Part 4: single-example SGD updates with the sampled point highlighted."""
    result=train_sgd(X,Y,learning_rate=0.01,steps=360,seed=7)
    fig,axes=make_figure(1,2,figsize=(12,6.75))
    ax_model,ax_loss=axes[0]
    add_header(fig,"Stochastic Gradient Descent","Each update uses one randomly selected training observation")
    ax_model.scatter(X,Y,s=90,color=CYAN,edgecolors=WHITE,linewidths=0.6,zorder=5,label="Targets")
    line_x=np.linspace(0.7,3.3,100)
    (line,)=ax_model.plot([],[],color=AMBER,lw=2.7,label="Current model")
    pred_points=ax_model.scatter([],[],s=45,marker="s",color=AMBER,label="Predictions")
    (selected,)=ax_model.plot([],[],"o",ms=16,mfc="none",mec=GREEN,mew=2.5,label="Selected sample")
    residuals=[ax_model.plot([],[],color=CORAL,lw=1.8,alpha=0.85)[0] for _ in X]
    set_canvas(ax_model); ax_model.set_ylim(-0.5,8.2)
    ax_model.legend(facecolor=BG,edgecolor=GRID,labelcolor=WHITE,fontsize=8,loc="upper left")
    info=ax_model.text(0.04,0.05,"",transform=ax_model.transAxes,color=WHITE,fontsize=10,
                       bbox={"boxstyle":"round,pad=0.45","fc":"#111827","ec":GRID})
    style_axis(ax_loss,title="Full-dataset MSE (measured after each sample update)")
    ax_loss.set_xlabel("SGD update"); ax_loss.set_ylabel("MSE")
    ax_loss.set_xlim(0,len(result.loss_history)-1); ax_loss.set_ylim(0,result.loss_history[0]*1.05)
    ax_loss.plot(np.arange(len(result.loss_history)),result.loss_history,color=GRID,lw=1.4)
    (curve,)=ax_loss.plot([],[],color=GREEN,lw=2.2)
    (marker,)=ax_loss.plot([],[],"o",color=AMBER,ms=7)
    frames=180
    indices=np.linspace(0,len(result.w_history)-1,frames,dtype=int)

    def update(frame:int)->list:
        idx=int(indices[frame])
        w,b,loss=result.w_history[idx],result.b_history[idx],result.loss_history[idx]
        pred=w*X+b
        line.set_data(line_x,w*line_x+b); pred_points.set_offsets(np.column_stack((X,pred)))
        for artist,xi,yi,pi in zip(residuals,X,Y,pred): artist.set_data([xi,xi],[yi,pi])
        if idx>0 and result.sample_indices is not None:
            chosen=int(result.sample_indices[idx-1])
            selected.set_data([X[chosen]],[Y[chosen]])
            selected.set_visible(True)
        else:
            selected.set_visible(False)
        info.set_text(f"update = {idx}\nsampled x = {X[int(result.sample_indices[max(idx-1,0)])]:.0f}\nw = {w:.3f}, b = {b:.3f}\nMSE = {loss:.4f}")
        upto=idx+1
        # Subsample the plotted trace for efficient drawing without changing training.
        take=np.unique(np.linspace(0,upto-1,min(500,upto),dtype=int))
        curve.set_data(take,result.loss_history[take])
        marker.set_data([idx],[loss])
        return [line,pred_points,selected,*residuals,info,curve,marker]

    save_animation(
        fig, update, frames, output_dir/"04_stochastic_gradient_descent.mp4",
        fps=fps, dpi=dpi, slowdown=slowdown,
    )


def animate_comparison(output_dir: Path, fps: int, dpi: int, slowdown: int) -> None:
    """Part 5: compare parameter paths and full-dataset loss traces."""
    batch=train_batch_gd(X,Y,learning_rate=0.05,steps=70)
    sgd=train_sgd(X,Y,learning_rate=0.005,steps=1800,seed=42)
    # Compare paths in the parameter plane; SGD updates use individual examples.
    w_grid=np.linspace(-0.1,2.8,100)
    b_grid=np.linspace(-0.4,2.2,100)
    W,B=np.meshgrid(w_grid,b_grid)
    Z=np.mean((W[...,None]*X+B[...,None]-Y)**2,axis=-1)
    fig,axes=make_figure(1,2,figsize=(12,6.75))
    ax_path,ax_loss=axes[0]
    add_header(fig,"Batch GD vs. SGD","Same objective • different gradient estimates and update paths")
    style_axis(ax_path,title="Parameter space (w, b)")
    contours=ax_path.contourf(W,B,Z,levels=28,cmap="viridis",alpha=0.9)
    ax_path.contour(W,B,Z,levels=10,colors="#CBD5E1",linewidths=0.45,alpha=0.38)
    ax_path.set_xlim(w_grid.min(),w_grid.max());ax_path.set_ylim(b_grid.min(),b_grid.max())
    ax_path.set_xlabel("w");ax_path.set_ylabel("b")
    ax_path.tick_params(colors=WHITE)
    (batch_path,)=ax_path.plot([],[],color=CYAN,lw=2.5,label="Batch GD path")
    (sgd_path,)=ax_path.plot([],[],color=CORAL,lw=1.25,alpha=0.95,label="SGD path")
    (batch_point,)=ax_path.plot([],[],"o",color=CYAN,ms=7)
    (sgd_point,)=ax_path.plot([],[],"o",color=CORAL,ms=7)
    ax_path.plot([2],[1],marker="*",color=GREEN,ms=12,label="Minimum (2, 1)")
    ax_path.legend(facecolor=BG,edgecolor=GRID,labelcolor=WHITE,fontsize=8,loc="upper right")
    fig.colorbar(contours,ax=ax_path,fraction=0.045,pad=0.03,label="MSE")
    style_axis(ax_loss,title="Full-dataset MSE")
    ax_loss.set_xlabel("Progress (normalized for display)");ax_loss.set_ylabel("MSE")
    # Compare trajectories on a common normalized horizontal axis, not equal compute cost.
    t_batch=np.linspace(0,1,len(batch.loss_history))
    t_sgd=np.linspace(0,1,len(sgd.loss_history))
    ax_loss.set_xlim(0,1);ax_loss.set_ylim(0, batch.loss_history[0]*1.05)
    ax_loss.plot(t_batch,batch.loss_history,color=GRID,lw=1.1)
    ax_loss.plot(t_sgd,sgd.loss_history,color=GRID,lw=1.1,alpha=0.5)
    (batch_curve,)=ax_loss.plot([],[],color=CYAN,lw=2.5,label="Batch GD")
    (sgd_curve,)=ax_loss.plot([],[],color=CORAL,lw=1.6,label="SGD")
    ax_loss.legend(facecolor=BG,edgecolor=GRID,labelcolor=WHITE,fontsize=8)
    caption=fig.text(0.5,0.035,"Batch GD uses all observations per update; SGD uses one sampled observation.",
                     color=MUTED,ha="center",fontsize=9)
    frames=200

    def update(frame:int)->list:
        progress=frame/(frames-1)
        ib=min(int(progress*(len(batch.w_history)-1)),len(batch.w_history)-1)
        isgd=min(int(progress*(len(sgd.w_history)-1)),len(sgd.w_history)-1)
        batch_path.set_data(batch.w_history[:ib+1],batch.b_history[:ib+1])
        sgd_path.set_data(sgd.w_history[:isgd+1],sgd.b_history[:isgd+1])
        batch_point.set_data([batch.w_history[ib]],[batch.b_history[ib]])
        sgd_point.set_data([sgd.w_history[isgd]],[sgd.b_history[isgd]])
        batch_curve.set_data(t_batch[:ib+1],batch.loss_history[:ib+1])
        # Reduce display samples for performance but preserve the curve's trend.
        upto=isgd+1
        take=np.unique(np.linspace(0,upto-1,min(600,upto),dtype=int))
        sgd_curve.set_data(t_sgd[take],sgd.loss_history[take])
        caption.set_text(
            f"Batch update {ib}: uses all 3 samples  |  SGD update {isgd}: uses one sample"
        )
        return [batch_path,sgd_path,batch_point,sgd_point,batch_curve,sgd_curve,caption]

    save_animation(
        fig, update, frames, output_dir/"05_batch_vs_sgd.mp4",
        fps=fps, dpi=dpi, slowdown=slowdown,
    )


def main() -> None:
    parser=argparse.ArgumentParser(description="Generate Lesson 3 Gradient Descent MP4 animations.")
    parser.add_argument("--output-dir",type=Path,default=Path("outputs"),help="Directory for generated MP4 files.")
    parser.add_argument("--fps",type=int,default=24,help="Frames per second (default: 24).")
    parser.add_argument("--dpi",type=int,default=120,help="Render DPI (default: 120).")
    parser.add_argument(
        "--slowdown",
        type=int,
        default=2,
        help="Hold each frame for this many output frames (default: 2; use 1 for normal speed).",
    )
    args=parser.parse_args()
    if args.fps<1 or args.fps>60:
        parser.error("--fps must be between 1 and 60.")
    if args.dpi<60 or args.dpi>240:
        parser.error("--dpi must be between 60 and 240.")
    if args.slowdown<1 or args.slowdown>8:
        parser.error("--slowdown must be between 1 and 8.")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError(
            "Matplotlib cannot find FFmpeg. Install FFmpeg and ensure its executable is on PATH, "
            "then rerun this script. No video was generated."
        )
    args.output_dir.mkdir(parents=True,exist_ok=True)
    animations=[animate_problem_hook,animate_loss_curve,animate_batch_gd,animate_sgd,animate_comparison]
    for animation_fn in animations:
        animation_fn(args.output_dir,args.fps,args.dpi,args.slowdown)
    print(f"\nDone. Generated {len(animations)} MP4 files in: {args.output_dir.resolve()}")


if __name__=="__main__":
    main()
