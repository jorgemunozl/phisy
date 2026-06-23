"""Problema 2(c): y = x + 1/x + x^m/(m^2-1),  m ∈ {0,2,3},  x ∈ [0.1, 4]."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0.1, 4.0, 400)
for m in (0, 2, 3):
    ys = xs + 1 / xs + xs**m / (m * m - 1)
    plt.plot(xs, ys, label=rf"$m={m}$")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(r"$y=x+\frac{1}{x}+\frac{x^m}{m^2-1}$")
plt.tight_layout()
img = Path(__file__).resolve().parent / "images"
img.mkdir(parents=True, exist_ok=True)
path = img / "lab2_euler_c.pdf"
plt.savefig(path, bbox_inches="tight", dpi=150)
print(path)
plt.show()
