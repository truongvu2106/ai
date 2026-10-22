# Optimization for AI — Lesson 5

**Topic:** Learning Rate, Convergence, and Local Minima  
**Series:** AI & ML — Learn • Build • Grow

This project contains the lesson prompts, reproducible numerical demonstrations, unit tests, and Python source code to generate five separate MP4 teaching animations. **The project does not include rendered MP4s**; run the animation script on your computer to produce them.

## Teaching design and consistency

The prompts, code, and animations share the same two explicitly separated objectives:

### A. Convex objective — learning rate and convergence

\[
J(\theta)=(\theta-3)^2,\qquad J'(\theta)=2(\theta-3)
\]

Initial value: \(\theta_0=-2\). The first gradient is \(-10\), and the initial loss is \(25\).

| Learning rate | First updated parameter | First new loss | Interpretation in this example |
|---:|---:|---:|---|
| 0.05 | -1.5 | 20.25 | Small step; slow progress |
| 0.25 | 0.5 | 6.25 | Stable progress toward the minimum |
| 1.10 | 9.0 | 36.0 | Overshoots and increases loss; subsequent steps diverge |

For \(\eta=0.25\), the successive states are:

- Parameter: -2 → 0.5 → 1.75 → 2.375
- Loss: 25 → 6.25 → 1.5625 → 0.390625

The learning-rate stability interpretation above is specific to this quadratic objective. It is not a universal learning-rate rule for all optimization problems.

### B. Non-convex objective — local minima and initialization

\[
f(\theta)=0.1\theta^2+2\cos(1.5\theta)+0.12\theta
\]

\[
f'(\theta)=0.2\theta-3\sin(1.5\theta)+0.12
\]

Learning rate: 0.02. Starting values: -6, -4, and 1. These starts move toward different minima. This is an illustrative one-dimensional landscape, not a representation of a particular neural network's loss surface.

## Project structure

```text
optimization-for-ai-lesson5/
├── src/optimization_ai_lesson5/
│   ├── __init__.py
│   └── objectives.py
├── examples/
│   └── run_experiments.py
├── animations/
│   └── animate_all.py
├── tests/
│   └── test_objectives.py
├── outputs/
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install **FFmpeg** separately and make sure `ffmpeg -version` works in your terminal. FFmpeg is required only to write MP4 files.

## Run the numerical examples

From this project's root directory:

```bash
python examples/run_experiments.py
```

The script prints the verified first-step values, a convergence diagnostic using `abs(gradient) <= 1e-3`, and final states for different starts on the non-convex objective. It also writes `outputs/lesson5_landscapes_preview.png`.

## Generate MP4 animation clips

Generate all five scenes (10 seconds per clip by default):

```bash
python animations/animate_all.py --scene all --output-dir outputs --seconds 10 --fps 24 --dpi 100
```

Generate one scene while tuning visuals:

```bash
python animations/animate_all.py --scene learning-rate --output-dir outputs_preview --seconds 10 --fps 15 --dpi 80
```

Scene names: `learning-rate`, `one-update`, `convergence`, `local-minima`, `recap`.

### Generated MP4s

1. `01_learning_rate_comparison.mp4` — small, suitable, and oversized steps on the convex quadratic.
2. `02_one_update_step_by_step.mp4` — exact parameter/loss updates for \(\eta=0.25\).
3. `03_convergence_diagnostics.mp4` — objective and gradient-magnitude histories.
4. `04_local_minima_initialization.mp4` — different starting values on the non-convex function.
5. `05_optimization_diagnostic_recap.mp4` — summarizes learning rate, convergence, and initialization.

## Run tests

```bash
pytest -q
```

The tests check objective/gradient values, the three learning-rate outcomes, convergence behavior, and initialization sensitivity on the non-convex function.

## Educational caveats

- A low objective value for a training example does not automatically establish generalization.
- A gradient near zero is one convergence diagnostic, not a universal guarantee of global optimality.
- Different starts can end in different local minima on a non-convex objective.
- A learning rate that works on one objective may fail on another.
- The two objectives are deliberately separate: the quadratic illustrates learning-rate stability; the non-convex function illustrates local minima.

## License

MIT. See `LICENSE`.
