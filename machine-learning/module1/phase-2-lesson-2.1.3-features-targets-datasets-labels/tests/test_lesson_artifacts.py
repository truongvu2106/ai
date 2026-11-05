import json
from pathlib import Path

import pandas as pd

LESSON_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = LESSON_DIR / "outputs"


def load_artifacts():
    csv_path = OUTPUT_DIR / "customer_dataset.csv"
    json_path = OUTPUT_DIR / "lesson_results.json"
    assert csv_path.exists(), "Run the notebook first to generate customer_dataset.csv"
    assert json_path.exists(), "Run the notebook first to generate lesson_results.json"
    frame = pd.read_csv(csv_path)
    results = json.loads(json_path.read_text(encoding="utf-8"))
    return frame, results


def test_dataset_has_expected_examples_and_columns():
    frame, results = load_artifacts()
    assert len(frame) == 180
    assert frame.columns.tolist() == [
        "customer_id", "monthly_visit_score", "spending_score", "customer_segment"
    ]
    assert results["number_of_examples"] == 180


def test_features_and_target_are_separated():
    frame, results = load_artifacts()
    features = results["feature_columns"]
    target = results["target_column"]
    assert features == ["monthly_visit_score", "spending_score"]
    assert target == "customer_segment"
    assert target not in features
    assert "customer_id" not in features
    assert len(frame[features]) == len(frame[target])


def test_class_labels_and_mapping_are_consistent():
    frame, results = load_artifacts()
    expected = {"Tier_A", "Tier_B", "Tier_C"}
    assert set(frame["customer_segment"].unique()) == expected
    assert set(results["label_mapping"].keys()) == expected
    assert sum(results["label_counts"].values()) == 180


def test_tensor_shapes_and_dtypes_are_documented():
    _, results = load_artifacts()
    assert results["feature_shape"] == [180, 2]
    assert results["target_shape"] == [180]
    assert results["feature_dtype"] == "torch.float32"
    assert results["target_dtype"] == "torch.int64"


def test_batch_dimensions_are_consistent():
    _, results = load_artifacts()
    assert results["batch_size"] == 8
    assert results["first_batch_feature_shape"] == [8, 2]
    assert results["first_batch_target_shape"] == [8]
