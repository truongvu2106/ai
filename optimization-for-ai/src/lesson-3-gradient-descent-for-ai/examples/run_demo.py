"""Run the Lesson 3 numerical demonstration without creating any video."""

from __future__ import annotations

import numpy as np

from optimization_ai import mean_squared_error, train_batch_gd, train_sgd


def main() -> None:
    # Illustrative data reused from the Optimization for AI lessons.
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 5.0, 7.0])

    initial_loss = mean_squared_error(x, y, w=0.0, b=0.0)
    first_dw, first_db = -68.0 / 3.0, -10.0
    learning_rate = 0.05
    first_w = 0.0 - learning_rate * first_dw
    first_b = 0.0 - learning_rate * first_db
    first_loss = mean_squared_error(x, y, first_w, first_b)

    batch = train_batch_gd(x, y, learning_rate=0.05, steps=100)
    sgd = train_sgd(x, y, learning_rate=0.005, steps=3000, seed=42)

    print("Gradient Descent and Stochastic Gradient Descent")
    print("Illustrative data: x=[1, 2, 3], y=[3, 5, 7]")
    print("Model: y_hat = w*x + b; objective: full-data MSE\n")
    print(f"Initial state: w=0.0000, b=0.0000, MSE={initial_loss:.6f}")
    print(f"Initial gradient: dw={first_dw:.6f}, db={first_db:.6f}")
    print(
        f"First batch update (eta={learning_rate}): "
        f"w={first_w:.6f}, b={first_b:.6f}, MSE={first_loss:.6f}\n"
    )

    print("Full-batch Gradient Descent:")
    print(
        f"  final w={batch.final_w:.6f}, b={batch.final_b:.6f}, "
        f"MSE={batch.final_loss:.8f} after 100 updates"
    )

    print("Single-observation SGD:")
    print(
        f"  final w={sgd.final_w:.6f}, b={sgd.final_b:.6f}, "
        f"MSE={sgd.final_loss:.8f} after 3000 updates"
    )
    print("\nNote: learning rates and update counts differ; this is a conceptual demo,")
    print("not a controlled benchmark of optimizer speed or compute efficiency.")


if __name__ == "__main__":
    main()
