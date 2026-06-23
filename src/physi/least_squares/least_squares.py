"""Linear regression via least squares (normal equation)."""

import numpy as np
import matplotlib.pyplot as plt


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear model y ≈ Xθ using the normal equation.

    θ = (XᵀX)⁻¹Xᵀy

    X: (n, d) design matrix
    y: (n,) target vector
    Returns: θ coefficients (d+1,) including intercept
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Add intercept column
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])

    # Normal equation: θ = (XᵀX)⁻¹Xᵀy
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    theta = np.linalg.solve(XtX, Xty)

    return theta


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict ŷ = Xθ (includes intercept)."""
    X = np.asarray(X)
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])
    return X_aug @ theta


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y - y_pred) ** 2))


if __name__ == "__main__":
    # Demo: y = 2 + 3x + noise
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 + 3 * X.ravel() + 0.5 * np.random.randn(100)

    theta = fit(X, y)
    print("Coefficients [intercept, slope]:", theta)

    y_pred = predict(X, theta)
    print("MSE:", mse(y, y_pred))

    # Plot
    x_flat = X.ravel()
    x_line = np.linspace(x_flat.min(), x_flat.max(), 100)
    y_line = predict(x_line.reshape(-1, 1), theta)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_flat, y, alpha=0.6, label="data")
    ax.plot(x_line, y_line, "r-", lw=2, label=rf"fit: $\hat{{y}} = {theta[0]:.2f} + {theta[1]:.2f}x$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Least squares linear regression")
    plt.tight_layout()
    plt.show()
