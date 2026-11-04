from pathlib import Path
import ast
import json
import nbformat

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "lesson_2_1_2_supervised_vs_unsupervised_learning.ipynb"
OUTPUTS = Path(__file__).parents[1] / "outputs"


def test_notebook_has_markdown_and_code_cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert any(cell.cell_type == "markdown" for cell in notebook.cells)
    assert any(cell.cell_type == "code" for cell in notebook.cells)
    assert len(notebook.cells) >= 25


def test_notebook_code_cells_are_valid_python():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook-cell-{index}")


def test_notebook_teaches_both_learning_types_with_pytorch():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "nn.Linear(in_features=2, out_features=3)" in source
    assert "nn.CrossEntropyLoss()" in source
    assert "loss.backward()" in source
    assert "optimizer.step()" in source
    assert "torch.cdist" in source
    assert "X_unlabeled" in source


def test_exported_results_are_consistent_after_notebook_execution():
    result_file = OUTPUTS / "lesson_results.json"
    assert result_file.exists(), "Run the notebook before running output tests."
    results = json.loads(result_file.read_text(encoding="utf-8"))
    assert results["phase"] == 2
    assert results["lesson"] == "2.1.2"
    assert results["dataset"]["sample_count"] == 180
    assert results["supervised"]["epochs"] == 350
    assert results["unsupervised"]["k"] == 3
    assert len(results["dataset"]["points"]) == 180
