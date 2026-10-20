"""Train and compare mini-batch SGD, Momentum, AdaGrad, RMSProp, and Adam."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = str(PROJECT_ROOT / "src")
if SOURCE_PATH in sys.path:
    sys.path.remove(SOURCE_PATH)
sys.path.insert(0, SOURCE_PATH)

import numpy as np

from optimization_ai import make_regression_data, mse, optimizer_suite, train_minibatch


def main() -> None:
    x, y = make_regression_data(n=96, seed=42)
    print("Illustrative synthetic regression data: y ≈ 2x + 1 + noise")
    print(f"Training observations: {len(x)}")
    print(f"Initial full-data MSE (w=0,b=0): {mse(x, y, [0.0, 0.0]):.4f}\n")

    results = {}
    for name, (optimizer, batch_size) in optimizer_suite().items():
        history = train_minibatch(
            x, y, optimizer, batch_size=batch_size, epochs=60, seed=42,
        )
        results[name] = history
        w, b = history.final_params
        print(
            f"{name:16s} | batch_size={batch_size:2d} | "
            f"w={w:8.4f} | b={b:8.4f} | "
            f"final MSE={history.final_loss:.5f}"
        )

    print("\nInterpretation: all optimizers use the same dataset and batch size. "
          "Their learning rates and update rules differ, so this is an educational "
          "comparison rather than a universal performance ranking.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed; skipping the static plot.")
        return

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, history in results.items():
        ax.plot(np.arange(len(history.full_loss)), history.full_loss, label=name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Full-dataset MSE")
    ax.set_title("Optimizer comparison on the same illustrative dataset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimizer_loss_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved static comparison plot to: {output_dir / 'optimizer_loss_comparison.png'}")


if __name__ == "__main__":
    main()
