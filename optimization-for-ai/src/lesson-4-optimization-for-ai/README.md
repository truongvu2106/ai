# Optimization for AI — Lesson 4

**Mini-Batches, Momentum, and Adaptive Optimizers** for the `AI & ML — Learn • Build • Grow` education channel.

This repository contains explicit NumPy implementations, a reproducible coding demo, tests, lesson prompts, and source code that renders five explanatory MP4 animations when run locally.

> **No MP4s are rendered or bundled by default.** Run the animation generator on your machine to create the videos.

## Teaching objectives

By the end of this lesson, viewers should be able to explain:

- **Mini-batch size** controls how many observations contribute to one gradient update.
- **Momentum** accumulates recent gradient information to influence the next step.
- **AdaGrad** accumulates squared gradients to scale parameter steps.
- **RMSProp** tracks a moving average of squared gradients.
- **Adam** combines a first-moment estimate, a second-moment estimate, and bias correction.

A key distinction: mini-batching is a data/update strategy; Momentum, AdaGrad, RMSProp, and Adam are update rules. No optimizer is universally best. Results depend on data, objective, learning rate, batch size, and other hyperparameters.

## Data example

The project creates deterministic synthetic regression data:

```python
x = np.linspace(-2.0, 2.0, 96)
y = 2.0 * x + 1.0 + noise
```

`noise` is reproducible Gaussian noise controlled by seed 42. Inputs are normalized for stable numerical demonstrations. These are illustrative teaching data, not real exam-score evidence.

Model and objective:

```text
y_hat = w*x + b
J(w,b) = mean((w*x + b - y)**2)
```

## Project structure

```text
optimization-for-ai-lesson4/
├── src/optimization_ai/
│   ├── __init__.py
│   ├── data.py
│   ├── optimizers.py
│   └── training.py
├── examples/
│   └── run_comparison.py
├── animations/
│   └── animate_all.py
├── tests/
│   └── test_optimizers.py
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Setup

Python 3.10+ recommended.

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

```bash
python examples/run_comparison.py
```

This trains mini-batch SGD, Momentum, AdaGrad, RMSProp, and Adam using the same generated data and batch size. The update rules and learning rates differ, so this is a teaching comparison—not a universal optimizer ranking. A static full-dataset-MSE graph is written to `outputs/optimizer_loss_comparison.png`.

## Generate the animation clips as MP4

Install **FFmpeg** separately and make sure `ffmpeg` is available on your `PATH`:

```bash
ffmpeg -version
```

Then run from the project root:

```bash
python animations/animate_all.py --output-dir outputs --fps 24 --dpi 120 --speed 0.5 --duration 10
```

`--speed` sets playback speed (`0.5` is half speed, `1` is normal speed); it defaults to `0.5`. `--duration` sets the intended duration of each clip in seconds at normal speed, so the rendered duration is `duration / speed` (for example, 10 seconds at half speed produces a 20-second clip). If `--duration` is omitted, each scene keeps its existing base duration and playback speed scales it. Use `--speed 1 --duration 15` for 15-second clips at normal speed. Supported speed values are `0.1`–`4.0`.

Five clips will be created:

1. `01_minibatch_mechanism.mp4` — highlights the mini-batch used for each update.
2. `02_batch_size_comparison.mp4` — compares batch sizes 1, 8, 32, and full batch.
3. `03_momentum_loss_valley.mp4` — visualizes Momentum on an illustrative narrow loss valley.
4. `04_adaptive_optimizer_steps.mp4` — compares parameter-step scaling for SGD, AdaGrad, RMSProp, and Adam.
5. `05_optimizer_comparison.mp4` — compares full-dataset MSE and prediction lines over training.

Render one scene at a time to iterate quickly:

```bash
python animations/animate_all.py --scene momentum --output-dir outputs_preview --fps 15 --dpi 80
python animations/animate_all.py --scene minibatches --output-dir outputs_preview --fps 15 --dpi 80
```

Available `--scene` values: `all`, `minibatches`, `batch-sizes`, `momentum`, `adaptive`, `comparison`.

## Important interpretation notes

- Batch size 1 corresponds to single-observation SGD; a full-dataset batch corresponds to full-batch Gradient Descent.
- A mini-batch gradient estimates the full-dataset gradient using a subset.
- Momentum uses a history-dependent velocity. The exact convention varies by implementation; this project uses `v = beta*v + g` then `theta -= lr*v`.
- AdaGrad's accumulated squared gradients can make effective steps progressively smaller.
- RMSProp uses an exponential moving average of squared gradients.
- Adam combines bias-corrected first and second moments.
- The Momentum visual uses a separate illustrative quadratic landscape to reveal motion in a narrow valley; it is not the same exact dataset loss surface as the regression demo.
- Training curves depend on the chosen hyperparameters. Do not interpret this demo as evidence that one optimizer wins on every problem.
- Learning-rate selection and convergence will be examined in Lesson 5.

## Tests

```bash
pytest -q
```

## Video prompts

`VIDEO_PROMPTS.md` includes Source Research, Main Content, and Visual Director prompts, plus the YouTube title and Vietnamese description. Each prompt is under 500 words and emphasizes intuition, numerical interpretation, synchronized animation, and concise but meaningful explanation.

## License

MIT. See `LICENSE`.
