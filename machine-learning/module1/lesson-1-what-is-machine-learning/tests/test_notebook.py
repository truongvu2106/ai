from pathlib import Path
import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "lesson_3_1_1_what_is_machine_learning.ipynb"


def test_notebook_is_valid_and_has_expected_steps():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert "Step 1" in markdown
    assert "Step 16" in markdown
    assert "supervised learning" in markdown.lower()
    assert "feature" in markdown.lower()
    assert "target" in markdown.lower()


def test_notebook_demonstrates_the_pytorch_training_loop():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for required in [
        "class HousePriceModel(nn.Module)",
        "nn.Linear(in_features=1, out_features=1)",
        "nn.MSELoss()",
        "torch.optim.SGD",
        "optimizer.zero_grad()",
        "loss.backward()",
        "optimizer.step()",
    ]:
        assert required in source


def test_code_cells_compile():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")
