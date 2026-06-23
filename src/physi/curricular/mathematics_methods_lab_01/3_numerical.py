"""Problema 3 (b): x²y''−2xy'+2y=x²−2x+2, y(1)=1, y'(1)=0 en [0.5, 5]."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

x_sym = sp.symbols("x", real=True)
y_fn = sp.Function("y")
ode = sp.Eq(
    x_sym**2 * y_fn(x_sym).diff(x_sym, 2)
    - 2 * x_sym * y_fn(x_sym).diff(x_sym)
    + 2 * y_fn(x_sym),
    x_sym**2 - 2 * x_sym + 2,
)
ics = {y_fn(1): 1, y_fn(x_sym).diff(x_sym).subs(x_sym, 1): 0}
sol = sp.dsolve(ode, y_fn(x_sym), ics=ics)
sp.pprint(sp.Eq(y_fn(x_sym), sol.rhs))
y_anal = sp.lambdify(x_sym, sol.rhs, modules="numpy")


def rhs(t: float, u: list[float]) -> list[float]:
    yv, yp = u
    return [yp, (2 / t) * yp - (2 / (t * t)) * yv + (t * t - 2 * t + 2) / (t * t)]


x_lo, x_hi = 0.5, 5.0
ic = [1.0, 0.0]
kw = {"dense_output": True, "rtol": 1e-9, "atol": 1e-12}
s_lo = solve_ivp(rhs, (1.0, x_lo), ic, **kw)
s_hi = solve_ivp(rhs, (1.0, x_hi), ic, **kw)

xs = np.linspace(x_lo, x_hi, 5000)
y_num = np.empty_like(xs)
m = xs < 1.0
y_num[m] = s_lo.sol(xs[m])[0]
y_num[~m] = s_hi.sol(xs[~m])[0]
err = float(np.max(np.abs(y_num - y_anal(xs))))
print(f"Error máximo absoluto (malla {len(xs)} pts): {err:.3e}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, y_num, label="Numérica (RK45)")
ax.plot(xs, y_anal(xs), "--", label="Analítica (SymPy)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(r"$x^2 y'' - 2x y' + 2y = x^2 - 2x + 2$,  $y(1)=1$, $y'(1)=0$")
fig.tight_layout()

out = Path(__file__).resolve().parent / "images"
out.mkdir(parents=True, exist_ok=True)
path = out / "lab3_numerico_vs_analitico.pdf"
fig.savefig(path, bbox_inches="tight", dpi=150)
print(path)
plt.show()
