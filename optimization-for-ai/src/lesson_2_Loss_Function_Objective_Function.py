import numpy as np
import matplotlib.pyplot as plt

# Illustrative data only.
x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 5.0, 7.0])


def calculate_losses(
    w: float,
    b: float,
) -> tuple[np.ndarray, float, float]:
    """Return per-example squared errors, MSE, and MAE."""
    predictions = w * x + b
    errors = predictions - y

    squared_errors = errors**2
    mse = float(np.mean(squared_errors))
    mae = float(np.mean(np.abs(errors)))

    return squared_errors, mse, mae


candidates = [
    (1.0, 1.0),
    (2.0, 0.0),
    (2.0, 1.0),
]

for w, b in candidates:
    squared_errors, mse, mae = calculate_losses(w, b)
    print(
        f"w={w:.1f}, b={b:.1f} | "
        f"squared errors={squared_errors.tolist()} | "
        f"MSE={mse:.4f} | MAE={mae:.4f}"
    )

# Visualize the objective over candidate parameter values.
w_values = np.linspace(0.0, 3.0, 121)
b_values = np.linspace(-1.0, 3.0, 121)
W, B = np.meshgrid(w_values, b_values)

J = np.mean(
    (W[..., None] * x + B[..., None] - y) ** 2,
    axis=-1,
)

plt.figure(figsize=(8, 6))
contours = plt.contourf(W, B, J, levels=30, cmap="viridis")
plt.colorbar(contours, label="Mean Squared Error")

plt.scatter(
    [1.0, 2.0, 2.0],
    [1.0, 0.0, 1.0],
    marker="x",
    color="red",
    label="Candidate parameters",
)

plt.xlabel("w")
plt.ylabel("b")
plt.title("Objective Function J(w, b)")
plt.legend()
plt.tight_layout()
plt.show()