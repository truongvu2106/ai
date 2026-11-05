# Lesson 2.1.3 — Lesson Example (PyTorch + Jupyter)

This folder contains the educational implementation. The notebook is the primary learning artifact; learners should run it cell by cell and read the explanations before each code section.

## Learning goals

- Understand dataset, example, feature, target, and label.
- Separate the feature matrix `X` from the target vector `y`.
- Distinguish classification labels from numeric regression targets.
- Convert features and targets into correctly typed PyTorch tensors.
- Use `TensorDataset` and `DataLoader` to inspect aligned examples and batches.

## Project layout

```text
lesson_example/
├── notebooks/
│   └── lesson_2_1_3_features_targets_datasets_labels.ipynb
├── outputs/
│   ├── customer_dataset.csv
│   ├── features_and_target.png
│   └── lesson_results.json
├── tests/
│   └── test_lesson_artifacts.py
└── requirements.txt
```

## Windows setup

Open PowerShell in this `lesson_example` directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name phase2-lesson213 --display-name "Python (Phase 2 Lesson 2.1.3)"
jupyter notebook
```

Open `notebooks/lesson_2_1_3_features_targets_datasets_labels.ipynb` and select the `Python (Phase 2 Lesson 2.1.3)` kernel. The notebook runs on CPU; no NVIDIA GPU is required.

## Run

Execute notebook cells from top to bottom. The notebook recreates the same synthetic customer dataset used in Lesson 2.1.2 and exports files to `outputs/`.

## Run tests

Run tests after executing the notebook:

```powershell
python -m pytest -q
```

## Scope note

This lesson prepares and inspects the data. It does not teach train/validation/test splitting, model evaluation, or the complete reusable ML pipeline; those topics belong to subsequent lessons in Module 2.1.

All customer records are synthetic teaching data.
