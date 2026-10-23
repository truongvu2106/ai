# Optimization for AI — Lesson 6

**Topic:** Convexity and Why Neural Network Optimization Is Difficult  
**Channel:** AI & ML — Learn • Build • Grow

This project contains synchronized video prompts, NumPy source code, unit tests, a static preview, and code to generate five separate MP4 animations. **No MP4s are rendered by default.** Run the animation generator locally to create them.

## Teaching consistency

### A. Convex objective

\[
J(x,y)=(x-0.5)^2+0.5(y+0.25)^2
\]

Gradient: `[2*(x - 0.5), y + 0.25]`. The unique global minimum is `(0.5, -0.25)` with objective `0`. With learning rate `0.2`, the three starts in the scripts all approach this same point.

Convexity chord example uses the slice `g(x)=(x-0.5)^2`, `a=-1`, `b=2.5`, and `alpha=0.4`:

- Combined point: `z = alpha*a + (1-alpha)*b = 1.1`.
- Function value: `g(z)=0.36`.
- Weighted chord value: `alpha*g(a)+(1-alpha)*g(b)=3.30`.
- Thus `g(z) <= 3.30`, illustrating the convexity inequality.

### B. Non-convex teaching objective

\[
f(x,y)=(x^2-1)^2+0.5y^2+0.2x
\]

Gradient: `[4*x*(x**2 - 1) + 0.2, y]`.

Verified critical points at `y=0`:

| Type | x | Objective | Meaning |
|---|---:|---:|---|
| Global minimum | -1.024120 | -0.202440 | Lowest of these critical points |
| Saddle point | 0.050126 | 1.005006 | Negative curvature in x, positive curvature in y |
| Local minimum | 0.973994 | 0.197434 | A local valley with higher objective than the global minimum |

With learning rate `0.02`, starts `(-1.8, 1.4)` and `(1.8, 1.2)` approach different minima. This is a toy 2D objective—not an actual neural-network loss surface.

## Structure

```text
optimization-for-ai-lesson6/
├── src/optimization_ai_lesson6/
│   ├── __init__.py
│   └── objectives.py
├── examples/run_experiments.py
├── animations/animate_all.py
├── tests/test_objectives.py
├── outputs/lesson6_landscapes_preview.png
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Setup

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

Install dependencies and editable package:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install FFmpeg separately and confirm that `ffmpeg -version` works. FFmpeg is required only for writing MP4s.

## Run numerical experiments

```bash
python examples/run_experiments.py
```

The script prints convex convergence endpoints, non-convex stationary point values, non-convex trajectory endpoints, and recreates the preview image.

## Generate MP4s

All five 10-second scenes at 24 FPS:

```bash
python animations/animate_all.py --scene all --output-dir outputs --seconds 10 --fps 24 --dpi 100
```

Render a single scene during iteration:

```bash
python animations/animate_all.py --scene saddle --output-dir outputs_preview --seconds 10 --fps 15 --dpi 80
```

Scene names: `convexity`, `convex-paths`, `nonconvex-paths`, `saddle`, `nn-recap`.

MP4 outputs:

1. `01_convexity_geometry.mp4`
2. `02_convex_global_minimum.mp4`
3. `03_nonconvex_multiple_minima.mp4`
4. `04_saddle_point.mp4`
5. `05_neural_network_optimization_recap.mp4`

## Test

```bash
pytest -q
```

## Important caveats

- For a convex objective over a convex domain, every local minimum is global; strict convexity gives at most one minimizer.
- Non-convex does not mean optimization must fail, and a local minimum is not necessarily poor.
- A saddle has directions of both positive and negative curvature; a near-zero gradient alone does not classify a point as a minimum.
- Actual neural-network loss landscapes are high-dimensional. The 2D contour here is a deliberately simple analogy, not a measurement of a neural network.

## License

MIT; see `LICENSE`.

## Further reading

- MIT OpenCourseWare, *Convex Analysis and Optimization*: https://ocw.mit.edu/courses/6-253-convex-analysis-and-optimization-spring-2012/
- Cornell, *Optimization Theory — Numerical Methods for Data Science*: https://www.cs.cornell.edu/~bindel/nmds/00-Background/05-Optimization.html
- Cambridge University Press, *Optimization Landscape of Neural Networks*: https://www.cambridge.org/core/books/mathematical-aspects-of-deep-learning/optimization-landscape-of-neural-networks/E8E68EA489DC86BC75013421EBBF824C
- Nguyen & Hein, *The Loss Surface of Deep and Wide Neural Networks* (PMLR): https://proceedings.mlr.press/v70/nguyen17a.html

These sources provide broader context. The numerical examples and expected values for this lesson are defined in the repository's source code and tests.
