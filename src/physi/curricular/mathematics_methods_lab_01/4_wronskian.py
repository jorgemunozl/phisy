"""Problema 4 (b) iii: Wronskiano; comparar resultado a mano W=2e^(3x) con sympy.wronskian."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols("x", real=True)
fns = [sp.exp(x), x * sp.exp(x), x**2 * sp.exp(x)]
w_sym = sp.simplify(sp.wronskian(fns, x))
w_hand = 2 * sp.exp(3 * x)

print("SymPy (wronskian):", w_sym)
print("A mano (informe): ", w_hand)
print("Diferencia simbólica W_sym - W_manuscrito =", sp.simplify(w_sym - w_hand))

xs = np.linspace(-2.0, 2.0, 800)
fn_sym = sp.lambdify(x, w_sym, modules="numpy")
fn_hand = sp.lambdify(x, w_hand, modules="numpy")
ys_sym = fn_sym(xs)
ys_hand = fn_hand(xs)
print("max |W_sym(x) - W_manuscrito(x)| en la malla:", float(np.max(np.abs(ys_sym - ys_hand))))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, ys_sym, label=r"$W_{\mathrm{SymPy}}(x)$ (sympy.wronskian)")
ax.plot(xs, ys_hand, "--", lw=1.5, label=r"$2\mathrm{e}^{3x}$ (analítico a mano)")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$W[y_1,y_2,y_3]$")
ax.set_title(r"$y_1=e^x,\; y_2=x e^x,\; y_3=x^2 e^x$")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

out = Path(__file__).resolve().parent / "images"
out.mkdir(parents=True, exist_ok=True)
path = out / "lab4_wronskiano.pdf"
fig.savefig(path, bbox_inches="tight", dpi=150)
print(path)
plt.show()
