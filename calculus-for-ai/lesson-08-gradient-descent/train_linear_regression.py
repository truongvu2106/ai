import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CREATE A SYNTHETIC DATASET
# ============================================================

# Use a fixed seed so the experiment is reproducible.
rng = np.random.default_rng(42)

# Generate 100 input samples.
X = np.linspace(-1, 1, 100)

# The underlying relationship is approximately y = 3x + 2.
# Gaussian noise makes the dataset more realistic.
noise = rng.normal(loc=0.0, scale=0.2, size=X.shape)
y = 3 * X + 2 + noise

# ============================================================
# 2. INITIALIZE MODEL PARAMETERS
# ============================================================

# Linear Regression model:
# y_hat = w * X + b

w = 0.0
b = 0.0

learning_rate = 0.1
iterations = 1000


# ============================================================
# 3. DEFINE PREDICTION AND LOSS FUNCTIONS
# ============================================================

def predict(X, w, b):
    """Calculate predictions for all input samples."""
    return w * X + b


def mean_squared_error(y_pred, y_true):
    """Calculate Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


# Record the initial Loss before training.
initial_predictions = predict(X, w, b)
initial_loss = mean_squared_error(initial_predictions, y)

loss_history = []


# ============================================================
# 4. TRAIN THE MODEL USING GRADIENT DESCENT
# ============================================================

for iteration in range(iterations):

    # Forward Pass: calculate predictions.
    y_pred = predict(X, w, b)

    # Calculate prediction errors.
    error = y_pred - y

    # Calculate gradients of MSE with respect to w and b.
    m = len(X)

    dw = (2 / m) * np.sum(error * X)
    db = (2 / m) * np.sum(error)

    # Update both parameters using gradients from
    # the same original parameter state.
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Recalculate Loss after the update.
    updated_predictions = predict(X, w, b)
    current_loss = mean_squared_error(updated_predictions, y)

    loss_history.append(current_loss)

    # Print progress periodically.
    if iteration % 100 == 0:
        print(
            f"Iteration {iteration:4d} | "
            f"Loss: {current_loss:.6f} | "
            f"w: {w:.6f} | "
            f"b: {b:.6f}"
        )


# ============================================================
# 5. DISPLAY TRAINING RESULTS
# ============================================================

final_predictions = predict(X, w, b)
final_loss = mean_squared_error(final_predictions, y)

print("\n========== TRAINING RESULTS ==========")
print(f"Initial Loss: {initial_loss:.6f}")
print(f"Final Loss:   {final_loss:.6f}")
print(f"Learned Weight (w): {w:.6f}")
print(f"Learned Bias (b):   {b:.6f}")

# Test the model on several input values.
test_inputs = np.array([-1.0, 0.0, 0.5, 1.0])
test_predictions = predict(test_inputs, w, b)

print("\n========== SAMPLE PREDICTIONS ==========")
for x_value, y_value in zip(test_inputs, test_predictions):
    print(f"x = {x_value:5.2f} -> predicted y = {y_value:.6f}")


# ============================================================
# 6. PLOT THE TRAINING LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(loss_history, label="Training MSE")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent: Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 7. VISUALIZE THE FITTED REGRESSION LINE
# ============================================================

plt.figure(figsize=(9, 5))

plt.scatter(
    X,
    y,
    label="Training Data",
    alpha=0.7
)

plt.plot(
    X,
    final_predictions,
    label="Learned Regression Line",
    linewidth=2
)

plt.xlabel("Input X")
plt.ylabel("Target y")
plt.title("Linear Regression Trained from Scratch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 8. OPTIONAL: VERIFY WITH SCIKIT-LEARN
# ============================================================

try:
    from sklearn.linear_model import LinearRegression

    # scikit-learn expects a two-dimensional feature array.
    X_2d = X.reshape(-1, 1)

    reference_model = LinearRegression()
    reference_model.fit(X_2d, y)

    sklearn_w = float(reference_model.coef_[0])
    sklearn_b = float(reference_model.intercept_)

    print("\n========== SCIKIT-LEARN COMPARISON ==========")
    print(f"NumPy Weight:      {w:.6f}")
    print(f"scikit-learn Weight: {sklearn_w:.6f}")
    print(f"NumPy Bias:        {b:.6f}")
    print(f"scikit-learn Bias:   {sklearn_b:.6f}")

    print(
        "\nThe coefficients should be close because both models "
        "fit the same Linear Regression objective."
    )

except ImportError:
    print(
        "\nscikit-learn is not installed. "
        "The NumPy model trained successfully without it."
    )