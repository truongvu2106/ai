# Lesson Example — PyTorch Jupyter Notebook

## Main notebook

`notebooks/lesson_2_1_5_overfitting_underfitting_generalization.ipynb`

## What the notebook teaches

- Underfitting: a model is too simple to capture the pattern.
- Overfitting: a flexible model fits training-specific noise.
- Generalization: performance on new examples from the relevant data distribution.
- Training/validation loss curves and the generalization gap.
- Validation-based checkpoint selection and final test evaluation.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open the notebook and use **Run All**.

## Experiment design

The notebook uses 240 noisy synthetic regression examples (168 train / 36 validation / 36 test). It compares a linear model, a compact MLP, and a high-capacity MLP using PyTorch. The underlying function is only a visual teaching reference; it is never passed to the models.

All model training is CPU-compatible.

## Outputs

The notebook exports CSV/JSON files and plots to `outputs/`. These files are used by the separate Manim project.

## Tests

After running the notebook:

```powershell
pytest -q
```
