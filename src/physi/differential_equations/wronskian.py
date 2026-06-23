"""Wronskian of three sufficiently smooth functions (numerical or symbolic)."""

from collections.abc import Callable

import numpy as np
import sympy as sp


def wronskian_at(
    f1: Callable[[float], float],
    f2: Callable[[float], float],
    f3: Callable[[float], float],
    x: float,
    *,
    h: float = 1e-5,
) -> float:
    """Determinant of [[f_i], [f_i'], [f_i'']] at x (central finite differences)."""

    def d1(f: Callable[[float], float], t: float) -> float:
        return (f(t + h) - f(t - h)) / (2.0 * h)

    def d2(f: Callable[[float], float], t: float) -> float:
        return (f(t + h) - 2.0 * f(t) + f(t - h)) / (h * h)

    m = np.array(
        [
            [f1(x), f2(x), f3(x)],
            [d1(f1, x), d1(f2, x), d1(f3, x)],
            [d2(f1, x), d2(f2, x), d2(f3, x)],
        ],
        dtype=float,
    )
    return float(np.linalg.det(m))


def wronskian_function(
    f1: Callable[[float], float],
    f2: Callable[[float], float],
    f3: Callable[[float], float],
    *,
    h: float = 1e-5,
) -> Callable[[float], float]:
    """Return W(x) using the same finite-difference step h."""

    def w(x: float) -> float:
        return wronskian_at(f1, f2, f3, x, h=h)

    return w


def wronskian_symbolic(y1: sp.Expr, y2: sp.Expr, y3: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """Exact Wronskian det for expressions in t (rows:0th, 1st, 2nd derivatives)."""
    rows = [
        [y1, y2, y3],
        [sp.diff(y1, t), sp.diff(y2, t), sp.diff(y3, t)],
        [sp.diff(y1, t, 2), sp.diff(y2, t, 2), sp.diff(y3, t, 2)],
    ]
    return sp.Matrix(rows).det()


if __name__ == "__main__":
    # W(1, x, x^2) = 2
    wnum = wronskian_at(lambda _: 1.0, lambda t: t, lambda t: t * t, 0.5)
    assert np.isclose(wnum, 2.0, rtol=0, atol=1e-6)

    x = sp.symbols("x", real=True)
    wsym = sp.simplify(wronskian_symbolic(1, x, x**2, x))
    assert wsym == 2
