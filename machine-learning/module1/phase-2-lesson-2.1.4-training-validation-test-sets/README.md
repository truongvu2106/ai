# Lesson example — PyTorch Jupyter Notebook

## Main notebook

`notebooks/lesson_2_1_4_training_validation_test_sets.ipynb`

The notebook is the primary teaching artifact. It recreates the customer dataset from the previous lessons, makes a reproducible stratified split, verifies that indices do not overlap, builds PyTorch `TensorDataset`/`Subset`/`DataLoader` objects, trains a small classifier using only the training set, selects a checkpoint using validation loss, and evaluates the test set at the end.

## Install

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open the notebook and run all cells. No NVIDIA GPU is required.

## Outputs

The notebook exports:

- `customer_dataset.csv`
- `customer_split_assignments.csv`
- `training_history.csv`
- `lesson_results.json`
- `three_split_scatter.png`
- `split_sizes.png`
- `class_distribution_comparison.png`
- `training_validation_loss.png`

## Tests

After executing all notebook cells:

```bash
python -m pytest tests -q
```

All example data are synthetic teaching data.
