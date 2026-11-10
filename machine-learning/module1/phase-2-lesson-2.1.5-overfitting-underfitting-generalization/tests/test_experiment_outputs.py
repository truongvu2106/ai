from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def test_split_counts_and_no_overlap():
    data = pd.read_csv(OUT / "dataset_splits.csv")
    counts = data["split"].value_counts().to_dict()
    assert counts == {"train": 168, "validation": 36, "test": 36}
    assert len(data) == 240
    assert data["example_id"].is_unique


def test_expected_model_behavior_is_visible():
    summary = json.loads((OUT / "experiment_summary.json").read_text(encoding="utf-8"))
    models = summary["models"]
    assert models["underfit"]["best_validation_mse"] > models["compact"]["best_validation_mse"]
    assert models["high_capacity"]["final_validation_mse"] > models["high_capacity"]["best_validation_mse"]
    assert models["high_capacity"]["final_train_mse"] < models["high_capacity"]["final_validation_mse"]


def test_high_capacity_history_has_training_and_validation_losses():
    history = pd.read_csv(OUT / "loss_history.csv")
    large = history[history["model"] == "High-capacity MLP"]
    assert len(large) == 6000
    assert {"epoch", "training_mse", "validation_mse"}.issubset(large.columns)
    assert large["validation_mse"].idxmin() < large.index[-1]


def test_test_results_are_exported_after_model_selection():
    summary = json.loads((OUT / "experiment_summary.json").read_text(encoding="utf-8"))
    test_predictions = pd.read_csv(OUT / "test_predictions.csv")
    assert summary["selected_model"] == "CompactMLP"
    assert len(test_predictions) == 36
    assert summary["final_test_mse"] > 0
