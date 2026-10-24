"""Manim Community Edition scenes for Optimization for AI — Lesson 7.

Render individual scenes with:
  manim -pqh animations/lesson7_scenes.py OverfittingHook
  manim -pqh animations/lesson7_scenes.py TrainValidationGap
  manim -pqh animations/lesson7_scenes.py L2Regularization
  manim -pqh animations/lesson7_scenes.py L1VersusL2
  manim -pqh animations/lesson7_scenes.py RegularizationStrengthAndRecap

The scenes use the exact deterministic data/model definitions in src/optimization_ai_lesson7/experiment.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optimization_ai_lesson7.experiment import (  # noqa: E402
    build_models, evaluate_models, make_dataset, true_function,
)

BG = "#0B0F19"
GRID = "#1E293B"
CYAN = "#38BDF8"
AMBER = "#F59E0B"
CORAL = "#FB7185"
GREEN = "#10B981"
PURPLE = "#A855F7"
WHITE = "#F8FAFC"
MUTED = "#CBD5E1"


def model_plot(axes: Axes, model, color: str, x_min: float = 0.02, x_max: float = 0.98):
    return axes.plot(
        lambda value: float(model.predict(np.array([value]))[0]),
        x_range=[x_min, x_max], color=color, stroke_width=3,
        use_smoothing=True,
    )


def training_dots(axes: Axes, color: str = AMBER) -> VGroup:
    x_train, y_train, _, _ = make_dataset()
    return VGroup(*[
        Dot(axes.c2p(float(x), float(y)), radius=0.045, color=color)
        for x, y in zip(x_train, y_train)
    ])


def make_axes(x_length: float = 5.7, y_length: float = 3.65) -> Axes:
    return Axes(
        x_range=[0, 1, 0.2], y_range=[-1.5, 1.5, 0.5],
        x_length=x_length, y_length=y_length,
        axis_config={"color": MUTED, "stroke_width": 1.5, "include_numbers": True, "font_size": 16},
        tips=False,
    )


def header(title: str, subtitle: str | None = None) -> VGroup:
    title_mob = Text(title, font_size=33, color=WHITE, weight="BOLD")
    title_mob.to_edge(UP, buff=0.28)
    if subtitle:
        sub = Text(subtitle, font_size=18, color=MUTED).next_to(title_mob, DOWN, buff=0.12)
        return VGroup(title_mob, sub)
    return VGroup(title_mob)


def metric_line(label: str, train: float, valid: float, color: str) -> VGroup:
    title = Text(label, font_size=19, color=color, weight="BOLD")
    vals = Text(f"Train MSE  {train:.4f}    Val MSE  {valid:.4f}", font_size=15, color=WHITE)
    vals.next_to(title, DOWN, buff=0.08, aligned_edge=LEFT)
    group = VGroup(title, vals)
    return group


class OverfittingHook(Scene):
    """Compare model complexity before explaining regularization."""
    def construct(self):
        self.camera.background_color = BG
        models = build_models()
        scores = evaluate_models()
        heading = header("A model can memorize and still generalize poorly", "Same synthetic data · different model complexity")
        self.play(FadeIn(heading, shift=DOWN * 0.15))
        axes = make_axes(x_length=6.0, y_length=3.65).shift(LEFT * 3.55 + DOWN * 0.30)
        self.play(Create(axes))
        signal = axes.plot(lambda x: float(true_function(x)), x_range=[0, 1], color=GREY_B, stroke_width=2, use_smoothing=True)
        points = training_dots(axes)
        simple = model_plot(axes, models["simple"], GREEN)
        overfit = model_plot(axes, models["overfit"], CORAL)
        self.play(Create(signal), FadeIn(points))
        self.play(Create(simple), run_time=1.4)
        self.play(Create(overfit), run_time=1.7)

        x_panel = 2.05
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.35, color=GREY_B), Text("Underlying signal", font_size=15, color=MUTED)).arrange(RIGHT, buff=0.12),
            VGroup(Line(ORIGIN, RIGHT * 0.35, color=GREEN), Text("Degree 3", font_size=15, color=GREEN)).arrange(RIGHT, buff=0.12),
            VGroup(Line(ORIGIN, RIGHT * 0.35, color=CORAL), Text("Degree 12, no penalty", font_size=15, color=CORAL)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([x_panel + 1.0, 1.2, 0])
        s = scores["simple"]
        o = scores["overfit"]
        simple_stats = metric_line("Simpler model", float(s["train_mse"]), float(s["validation_mse"]), GREEN)
        complex_stats = metric_line("High-degree model", float(o["train_mse"]), float(o["validation_mse"]), CORAL)
        simple_stats.move_to([x_panel + 0.95, -0.45, 0])
        complex_stats.move_to([x_panel + 0.95, -1.55, 0])
        note = Text("Lower training error ≠ better unseen predictions", font_size=17, color=AMBER)
        note.move_to([x_panel + 0.95, -2.65, 0])
        self.play(FadeIn(legend), FadeIn(simple_stats), FadeIn(complex_stats))
        self.play(Write(note))
        self.wait(1.0)


class TrainValidationGap(Scene):
    """Animate training-versus-validation MSE as the overfitting signal."""
    def construct(self):
        self.camera.background_color = BG
        self.play(FadeIn(header("Train loss and validation loss tell different stories", "Validation data approximates performance on unseen examples")))
        scores = evaluate_models()
        rows = [
            ("Degree 3", "simple", GREEN),
            ("Degree 12 / no penalty", "overfit", CORAL),
            ("Degree 12 / L2 λ=0.001", "ridge_0001", CYAN),
        ]
        x0, max_width = -3.9, 7.5
        top = 1.55
        for idx, (label, key, color) in enumerate(rows):
            y = top - idx * 1.45
            name = Text(label, font_size=19, color=color, weight="BOLD").move_to([-4.0, y + 0.36, 0])
            train = float(scores[key]["train_mse"])
            valid = float(scores[key]["validation_mse"])
            # Scale the bars by 0.35 MSE; cap only for drawing so visual widths stay on-canvas.
            train_w = min(train / 0.35, 1.0) * max_width
            valid_w = min(valid / 0.35, 1.0) * max_width
            train_bar = RoundedRectangle(corner_radius=0.05, width=max(train_w, 0.03), height=0.22, stroke_width=0, fill_color=color, fill_opacity=0.95)
            train_bar.move_to([x0 + train_w / 2, y, 0])
            val_bar = RoundedRectangle(corner_radius=0.05, width=max(valid_w, 0.03), height=0.22, stroke_width=0, fill_color=CORAL, fill_opacity=0.88)
            val_bar.move_to([x0 + valid_w / 2, y - 0.32, 0])
            train_label = Text(f"Train {train:.4f}", font_size=15, color=WHITE).next_to(train_bar, RIGHT, buff=0.12)
            val_label = Text(f"Validation {valid:.4f}", font_size=15, color=WHITE).next_to(val_bar, RIGHT, buff=0.12)
            self.play(FadeIn(name), GrowFromEdge(train_bar, LEFT), FadeIn(train_label), run_time=0.8)
            self.play(GrowFromEdge(val_bar, LEFT), FadeIn(val_label), run_time=0.8)
        note = Text("A large train–validation gap is a warning sign for overfitting.", font_size=20, color=AMBER)
        note.to_edge(DOWN, buff=0.35)
        self.play(Write(note))
        self.wait(1)


class L2Regularization(Scene):
    """Explain how L2 regularization changes a high-degree fit."""
    def construct(self):
        self.camera.background_color = BG
        self.play(FadeIn(header("Regularization discourages unnecessary complexity", "L2 adds a penalty for large coefficients")))
        models = build_models()
        scores = evaluate_models()
        axes = make_axes(x_length=6.0, y_length=3.6).shift(LEFT * 3.45 + DOWN * 0.28)
        self.play(Create(axes), FadeIn(training_dots(axes)))
        unreg = model_plot(axes, models["overfit"], CORAL)
        self.play(Create(unreg), run_time=1.2)
        ridge = model_plot(axes, models["ridge_0001"], GREEN)
        self.play(Transform(unreg, ridge), run_time=1.6)

        formula = MathTex(r"J(\beta)=\underbrace{\operatorname{MSE}}_{\text{fit}}+\lambda\underbrace{\sum_j\beta_j^2}_{\text{L2 penalty}}", font_size=30, color=WHITE)
        formula.move_to([2.85, 1.25, 0])
        explanation = VGroup(
            Text("λ = 0: fit training data closely", font_size=17, color=CORAL),
            Text("λ = 0.001: shrink coefficients", font_size=17, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([2.65, 0.05, 0])
        o, r = scores["overfit"], scores["ridge_0001"]
        metrics = VGroup(
            Text(f"No penalty: train {float(o['train_mse']):.4f} | val {float(o['validation_mse']):.4f}", font_size=14, color=CORAL),
            Text(f"L2 penalty:  train {float(r['train_mse']):.4f} | val {float(r['validation_mse']):.4f}", font_size=14, color=GREEN),
            Text(f"Coefficient L2 norm: {float(models['overfit'].coefficient_l2_norm):.1f} → {float(models['ridge_0001'].coefficient_l2_norm):.2f}", font_size=14, color=AMBER),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([2.85, -1.55, 0])
        self.play(Write(formula), FadeIn(explanation))
        self.play(FadeIn(metrics))
        self.wait(1)


class L1VersusL2(Scene):
    """Compare L1 sparsity and L2 shrinkage using the shared data."""
    def construct(self):
        self.camera.background_color = BG
        self.play(FadeIn(header("L1 and L2 regularization behave differently", "Both discourage overfitting, but penalize coefficients differently")))
        models = build_models()
        scores = evaluate_models()
        ridge = models["ridge_0001"]
        lasso = models["lasso_0002"]
        formula_l1 = MathTex(r"J=\operatorname{MSE}+\lambda\sum_j|\beta_j|", font_size=28, color=PURPLE)
        formula_l2 = MathTex(r"J=\operatorname{MSE}+\lambda\sum_j\beta_j^2", font_size=28, color=CYAN)
        formula_l1.move_to([-3.0, 2.25, 0])
        formula_l2.move_to([3.0, 2.25, 0])
        self.play(Write(formula_l1), Write(formula_l2))

        # Coefficient comparison uses the same degree-12 standardized polynomial features.
        for side_x, model, color, label in [(-3.25, lasso, PURPLE, "L1 / Lasso"), (3.25, ridge, CYAN, "L2 / Ridge")]:
            label_mob = Text(label, font_size=22, color=color, weight="BOLD").move_to([side_x, 1.6, 0])
            self.play(FadeIn(label_mob))
            base_y = -0.25
            max_abs = max(float(np.max(np.abs(model.coefficients))), 0.01)
            bar_groups = VGroup()
            for i, coefficient in enumerate(model.coefficients):
                height = 1.75 * abs(float(coefficient)) / max_abs
                bar = Rectangle(width=0.23, height=max(height, 0.025), stroke_width=0, fill_color=color, fill_opacity=0.92)
                x_pos = side_x - 1.55 + i * 0.28
                bar.move_to([x_pos, base_y + height / 2, 0])
                bar_groups.add(bar)
            baseline = Line([side_x - 1.7, base_y, 0], [side_x + 1.7, base_y, 0], color=MUTED, stroke_width=1.2)
            self.play(Create(baseline), LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bar_groups], lag_ratio=0.04), run_time=1.3)
            if label.startswith("L1"):
                note_text = f"Nonzero coefficients: {lasso.nonzero_coefficients}/12"
                val_mse = float(scores["lasso_0002"]["validation_mse"])
            else:
                note_text = f"Coefficients shrunk, usually not zero"
                val_mse = float(scores["ridge_0001"]["validation_mse"])
            note = Text(note_text, font_size=15, color=WHITE).move_to([side_x, -1.05, 0])
            metric = Text(f"Validation MSE: {val_mse:.4f}", font_size=16, color=AMBER).move_to([side_x, -1.55, 0])
            self.play(FadeIn(note), FadeIn(metric))
        footer = Text("L1 can produce sparsity; L2 tends to shrink weights smoothly.", font_size=19, color=WHITE).to_edge(DOWN, buff=0.25)
        self.play(Write(footer))
        self.wait(1)


class RegularizationStrengthAndRecap(Scene):
    """Show regularization strength as a trade-off, then recap the lesson."""
    def construct(self):
        self.camera.background_color = BG
        self.play(FadeIn(header("Regularization is a trade-off", "Too little can overfit; too much can underfit")))
        models = build_models()
        scores = evaluate_models()
        axes = make_axes(x_length=5.5, y_length=3.3).shift(LEFT * 3.65 + DOWN * 0.35)
        self.play(Create(axes), FadeIn(training_dots(axes)))
        keys = [("overfit", CORAL), ("ridge_0001", GREEN), ("ridge_001", CYAN), ("ridge_01", PURPLE)]
        current = model_plot(axes, models[keys[0][0]], keys[0][1])
        self.play(Create(current), run_time=1.0)
        for key, color in keys[1:]:
            new_graph = model_plot(axes, models[key], color)
            self.play(Transform(current, new_graph), run_time=0.9)

        names = ["λ=0", "λ=.001", "λ=.01", "λ=.1"]
        model_keys = ["overfit", "ridge_0001", "ridge_001", "ridge_01"]
        rows = VGroup()
        for i, (name, key) in enumerate(zip(names, model_keys)):
            sc = scores[key]
            row = Text(f"{name:<8} train {float(sc['train_mse']):.4f} | val {float(sc['validation_mse']):.4f}", font_size=15, color=[CORAL, GREEN, CYAN, PURPLE][i])
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([3.0, 0.55, 0])
        self.play(FadeIn(rows))
        summary = VGroup(
            Text("1. Train loss measures fit to training data.", font_size=17, color=WHITE),
            Text("2. Validation loss estimates performance on unseen data.", font_size=17, color=WHITE),
            Text("3. Choose regularization using validation—not training loss alone.", font_size=17, color=AMBER),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([2.8, -1.65, 0])
        self.play(FadeIn(summary))
        self.wait(1.5)
