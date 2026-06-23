"""Problema 1: verificación SymPy (a) y familias con (y(0), y'(0)) (b)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols("x", real=True)
y = sp.Function("y")
ode1 = sp.Eq(y(x).diff(x, 2) + 7 * y(x).diff(x), sp.exp(-7 * x))
ode2 = sp.Eq(y(x).diff(x, 2) - 7 * y(x).diff(x), (x - 1) ** 2)

s1 = sp.dsolve(ode1, y(x))
s2 = sp.dsolve(ode2, y(x))
print("--- SymPy dsolve (comparar con el desarrollo a mano / fig. p1) ---")
print("(i)  y'' + 7y' = e^{-7x}")
sp.pprint(s1)
print("(ii) y'' - 7y' = (x-1)^2")
sp.pprint(s2)


def y_curve(ode: sp.Eq, y0: float, yp0: float):
    yx = y(x)
    sol = sp.dsolve(ode, yx, ics={yx.subs(x, 0): y0, yx.diff(x).subs(x, 0): yp0})
    return sp.lambdify(x, sol.rhs, modules="numpy")


xs = np.linspace(-0.5, 3.0, 500)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ics = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, -0.5))
for y0, yp0 in ics:
    lab = rf"$y(0)={y0}$, $y'(0)={yp0}$"
    ax[0].plot(xs, y_curve(ode1, y0, yp0)(xs), label=lab)
    ax[1].plot(xs, y_curve(ode2, y0, yp0)(xs), label=lab)

ax[0].set_title(r"$y'' + 7y' = e^{-7x}$")
ax[1].set_title(r"$y'' - 7y' = (x-1)^2$")
for a in ax:
    a.set_xlabel("$x$")
    a.set_ylabel("$y$")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=8)
fig.tight_layout()

out = Path(__file__).resolve().parent / "images"
out.mkdir(parents=True, exist_ok=True)
path = out / "lab1_familias_edos.pdf"
fig.savefig(path, bbox_inches="tight", dpi=150)
print(path)
plt.show()
