# Lesson Example — Jupyter Notebook + PyTorch

This folder contains the main coding lesson. Open `notebooks/lesson_1_what_is_machine_learning.ipynb` and run each cell in order.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/lesson_1_what_is_machine_learning.ipynb`.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

## What learners build

- A synthetic dataset with house area and house price.
- A custom PyTorch `nn.Module` using `nn.Linear(1, 1)`.
- A mean-squared-error loss function.
- An SGD optimizer.
- An explicit training loop with forward pass, `backward()`, and `step()`.
- A prediction for a new 120 m² house.

The notebook is CPU-only and does not require an NVIDIA GPU. Its data are synthetic and must not be interpreted as actual property-market data.

## Outputs for the Manim project

The final notebook cell exports `outputs/house_data.csv`, `outputs/training_loss.csv`, `outputs/new_house_predictions.csv`, and `outputs/lesson_results.json`. The video-generation project reads these files to keep its visuals consistent with the code lesson.
