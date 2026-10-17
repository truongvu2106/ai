import numpy as np

# Illustrative data only—not real evidence about exam performance.
x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 5.0, 7.0])


def mean_squared_error(w: float, b: float) -> float:
    """Calculate the mean squared prediction error."""
    predictions = w * x + b
    return float(np.mean((predictions - y) ** 2))


candidates = [
    (1.0, 1.0),
    (2.0, 0.0),
    (2.0, 1.0),
]

for w, b in candidates:
    loss = mean_squared_error(w, b)
    print(f"w={w:.1f}, b={b:.1f} -> MSE={loss:.4f}")