"""Minimal explicit Euler for y' = tan(x) (same as y' - tan(x) = 0)."""

import matplotlib.pyplot as plt
import numpy as np


def euler_tan(x0: float, x1: float, y0: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    h = (x1 - x0) / n
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    for _ in range(n):
        y = y + h * np.tan(x)
        x = x + h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    # Stay inside (-π/2, π/2) where tan is smooth on the grid.
    xa, ya = euler_tan(0.0, 1.2, 0.0, 400)
    plt.plot(xa, ya, label="Euler")
    plt.plot(xa, -np.log(np.cos(xa)), "--", label=r"$-\ln\cos x$ (exact, $y(0)=0$)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
