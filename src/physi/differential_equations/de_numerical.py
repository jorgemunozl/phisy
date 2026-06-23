"""Numerical integration of y'' + 7 y' = f(x) via SciPy (first-order reduction)."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def _first_order_rhs(
    f: Callable[[float], float],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """State z = [y, y']; return dz/dx for y'' + 7 y' = f(x)."""

    def rhs(x: float, z: np.ndarray) -> np.ndarray:
        y, yp = z
        return np.array([yp, -7.0 * yp + f(x)])

    return rhs


def integrate_ode(
    f: Callable[[float], float],
    y0: float,
    yp0: float,
    x_span: tuple[float, float],
    *,
    n_eval: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve IVP from x_span[0] with y=y0, y'=yp0; returns (x, y)."""
    x0, x1 = x_span
    z0 = np.array([y0, yp0], dtype=float)
    rhs = _first_order_rhs(f)

    sol = solve_ivp(
        rhs,
        (x0, x1),
        z0,
        t_eval=np.linspace(x0, x1, n_eval),
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[0]


def ics_from_constants_ode1(c1: float, c2: float) -> tuple[float, float]:
    """Match SymPy general solution at x=0: y'' + 7y' = exp(-7x)."""
    y0 = c1 + c2
    yp0 = -7.0 * c2 - 1.0 / 7.0
    return y0, yp0


def ics_from_constants_ode2(c1: float, c2: float) -> tuple[float, float]:
    """Match SymPy general solution at x=0: y'' + 7y' = (x-1)^2."""
    y0 = c1 + c2
    yp0 = -7.0 * c2 + 65.0 / 343.0
    return y0, yp0


def rhs_exp(x: float) -> float:
    return float(np.exp(-7.0 * x))


def rhs_poly(x: float) -> float:
    return (x - 1.0) ** 2


def plot_numerical_solutions(
    xlim: tuple[float, float] = (-0.5, 3.0),
    constant_pairs: list[tuple[float, float]] | None = None,
) -> None:
    if constant_pairs is None:
        constant_pairs = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, -0.5),
        ]

    x0, x1 = xlim
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for c1, c2 in constant_pairs:
        y0_a, yp0_a = ics_from_constants_ode1(c1, c2)
        xa, ya = integrate_ode(rhs_exp, y0_a, yp0_a, (x0, x1))
        axes[0].plot(xa, ya, label=rf"$C_1={c1}$, $C_2={c2}$")

        y0_b, yp0_b = ics_from_constants_ode2(c1, c2)
        xb, yb = integrate_ode(rhs_poly, y0_b, yp0_b, (x0, x1))
        axes[1].plot(xb, yb, label=rf"$C_1={c1}$, $C_2={c2}$")

    axes[0].set_title(r"$y'' + 7y' = e^{-7x}$ (numerical)")
    axes[1].set_title(r"$y'' + 7y' = (x-1)^2$ (numerical)")
    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_numerical_solutions()
