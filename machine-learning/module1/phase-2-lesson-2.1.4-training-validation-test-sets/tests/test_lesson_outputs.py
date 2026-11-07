import json
from pathlib import Path

import pandas as pd

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def load_assignments():
    path = OUTPUTS / "customer_split_assignments.csv"
    assert path.exists(), "Run the Jupyter notebook before running tests."
    return pd.read_csv(path)


def test_split_sizes_and_total():
    results = json.loads((OUTPUTS / "lesson_results.json").read_text(encoding="utf-8"))
    assert results["number_of_examples"] == 180
    assert results["split_sizes"] == {"train": 126, "validation": 27, "test": 27}
    assert sum(results["split_sizes"].values()) == 180


def test_each_example_belongs_to_exactly_one_split():
    assignments = load_assignments()
    assert len(assignments) == 180
    assert assignments["customer_id"].is_unique
    assert set(assignments["split"]) == {"train", "validation", "test"}


def test_stratified_class_counts_are_preserved():
    results = json.loads((OUTPUTS / "lesson_results.json").read_text(encoding="utf-8"))
    for split_name, expected_count in {"train": 42, "validation": 9, "test": 9}.items():
        assert set(results["stratified_class_counts"][split_name].values()) == {expected_count}


def test_training_history_and_test_metrics_exist():
    results = json.loads((OUTPUTS / "lesson_results.json").read_text(encoding="utf-8"))
    history = pd.read_csv(OUTPUTS / "training_history.csv")
    assert len(history) == results["epochs"]
    assert results["best_epoch"] >= 1
    assert 0.0 <= results["final_test_accuracy"] <= 1.0
    assert results["no_overlap"] is True
    assert results["full_coverage"] is True
