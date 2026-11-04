# Lesson 2.1.2 — Jupyter Notebook Example

This folder contains the primary educational artifact: `notebooks/lesson_2_1_2_supervised_vs_unsupervised_learning.ipynb`.

## Install on Windows (CPU only)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m ipykernel install --user --name phase2-lesson-212 --display-name "Python (Phase 2 Lesson 2.1.2)"
jupyter lab
```

Run the notebook from top to bottom. It generates the plots and `outputs/lesson_results.json` consumed by the separate video-generation project.

## Key implementation choices

- Supervised: a small `nn.Linear(2, 3)` classifier trained using `CrossEntropyLoss` and SGD.
- Unsupervised: a transparent K-means implementation using `torch.cdist`, nearest-centroid assignment, and centroid means.
- K-means receives only the feature tensor; it never receives the known category labels.
- CPU only. Synthetic data only. No formal unseen-data evaluation in this lesson.
