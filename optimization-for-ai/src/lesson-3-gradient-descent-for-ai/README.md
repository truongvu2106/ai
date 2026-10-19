# Gradient Descent and SGD — Optimization for AI (Lesson 3)

Educational Python project for the `AI & ML — Learn • Build • Grow` channel. It implements full-batch Gradient Descent and single-observation Stochastic Gradient Descent (SGD) for a simple linear regression model, and provides source code to render five explanatory animation clips as MP4 files.

> **The videos are not included and are not generated for you.** Run the animation script locally to create the MP4 clips.

## Learning example

We use the same illustrative points as the earlier Optimization for AI lessons:

```python
x = [1, 2, 3]
y = [3, 5, 7]
y_hat = w * x + b
J(w, b) = mean((w*x + b - y)**2)
```

These points are for teaching only, not real exam data.

At `w=0, b=0`:

- Initial MSE: `83/3 ≈ 27.6667`
- Gradient with respect to `w`: `dw = -68/3 ≈ -22.6667`
- Gradient with respect to `b`: `db = -10`

With learning rate `η=0.05`, the first full-batch update is approximately `w=1.1333, b=0.5`, reducing the full-data MSE to approximately `5.4885`.

## Project layout

```text
gradient-descent-for-ai/
├── src/optimization_ai/
│   └── gradient_descent.py
├── examples/
│   └── run_demo.py
├── animations/
│   └── animate_all.py
├── tests/
│   └── test_gradient_descent.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Setup

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the numerical demo

This prints the initial gradient, one update, and final results for both optimizers. It does not create video files.

```bash
python examples/run_demo.py
```

## Render the animation clips as MP4

Install **FFmpeg** separately and make sure `ffmpeg` is available on your system `PATH`. Verify it with:

```bash
ffmpeg -version
```

Then, from the project root:

```bash
python animations/animate_all.py --output-dir outputs --fps 24 --dpi 120 --slowdown 2
```

The default `--slowdown 2` holds each frame for twice as long, making the clips play at half speed. Use `--slowdown 1` for normal speed; values from 1 to 8 are supported.

The script creates five MP4 files under `outputs/`:

1. `01_problem_hook.mp4` — prediction line improves and full-dataset MSE changes.
2. `02_gradient_on_loss_curve.mp4` — a parameter steps along a one-dimensional loss curve.
3. `03_batch_gradient_descent.mp4` — full-batch updates use every training point.
4. `04_stochastic_gradient_descent.mp4` — one data point is highlighted for each SGD update.
5. `05_batch_vs_sgd.mp4` — compares parameter paths and full-dataset loss traces.

For a lower-resolution/faster draft:

```bash
python animations/animate_all.py --output-dir outputs_preview --fps 15 --dpi 80
```

If Matplotlib reports that FFmpeg is unavailable, install FFmpeg and restart the terminal so PATH changes take effect. This project deliberately requires FFmpeg for MP4 export and does not silently substitute GIF files.

## Important interpretation notes

- Full-batch Gradient Descent calculates each gradient from all observations.
- In this project, SGD means **single-observation** updates: each update uses one randomly selected training point.
- The displayed objective is the full-dataset MSE even when SGD updates use only one sample.
- The demo uses different learning rates and update counts for the two methods to illustrate their behavior. It is not a controlled performance benchmark.
- SGD can increase the full-dataset loss on an individual update because its gradient is based on one sample.
- A learning rate that is too large can cause overshooting or divergence; a small learning rate may progress slowly. These behaviors depend on the objective and setup.

## Run tests

```bash
pytest -q
```

## License

MIT. See `LICENSE`.
