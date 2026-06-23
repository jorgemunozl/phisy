"""Curvas teóricas de dispersión n(λ) mediante la fórmula de Cauchy.

    n(λ) = A + B/λ²   (λ en micrómetros)

Coeficientes aproximados para vidrios ópticos comunes (solo ilustrativo).
Genera ``lab/n_vs_lambda.png`` para el informe LaTeX.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (A, B) en Cauchy con λ en µm. Orden: mayor n típico → menor.
VIDRIOS: dict[str, tuple[float, float]] = {
    "LaSF9 (flint denso de lantano)": (1.8301, 0.01710),
    "SF10 (flint denso)": (1.7280, 0.01342),
    "F2 (flint)": (1.6053, 0.00824),
    "BaK4 (crown de bario)": (1.5570, 0.00540),
    "BK7 (crown borosilicato)": (1.5046, 0.00420),
}

RANGO_VISIBLE_UM = (0.38, 0.78)

# Paleta con buen contraste (accesible)
COLORES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def cauchy(lambda_um: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b / lambda_um**2


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
        }
    )

    lambdas = np.linspace(0.35, 1.6, 500)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), layout="constrained")

    ax.axvspan(
        *RANGO_VISIBLE_UM,
        color="#B0B0B0",
        alpha=0.45,
        zorder=0,
        label="Espectro visible",
    )

    for (nombre, (a, b)), color in zip(VIDRIOS.items(), COLORES, strict=True):
        n_vals = cauchy(lambdas, a, b)
        ax.plot(lambdas, n_vals, color=color, linewidth=2.4, label=nombre, zorder=2)

    ax.set_xlim(0.35, 1.6)
    ax.set_ylim(1.50, 1.96)
    ax.set_xlabel(r"Longitud de onda $\lambda$ ($\mu$m)")
    ax.set_ylabel(r"Índice de refracción $n$")
    ax.set_title(r"Dispersión cromática: $n(\lambda)$ (modelo de Cauchy)")

    leg = ax.legend(
        loc="upper right",
        framealpha=0.95,
        edgecolor="0.75",
        title="Vidrio óptico",
        alignment="left",
    )
    leg.get_title().set_fontsize(10)

    ax.text(
        0.5 * (RANGO_VISIBLE_UM[0] + RANGO_VISIBLE_UM[1]),
        1.505,
        "luz visible",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.35",
        style="italic",
        zorder=1,
    )

    salida = Path(__file__).resolve().parents[2] / "lab" / "n_vs_lambda.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {salida}")


if __name__ == "__main__":
    main()
