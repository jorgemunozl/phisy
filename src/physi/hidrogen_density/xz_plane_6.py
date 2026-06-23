"""Plot six hydrogen probability densities |ψ_{n,l,m}|² in the xz-plane."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from math import factorial


def hydrogen_radial(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """Radial wave function R_{nl}(r) in atomic units (a₀ = 1)."""
    rho = 2 * r / n
    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)(rho)
    coeff = factorial(n - l - 1) / (2 * n * factorial(n + l))
    norm = np.sqrt((2 / n) ** 3 * coeff)
    return norm * np.exp(-rho / 2) * rho**l * laguerre


def hydrogen_density(n: int, l: int, m: int, r: np.ndarray,
                     theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Probability density |ψ_{n,l,m}(r,θ,φ)|²."""
    R = hydrogen_radial(n, l, r)
    Y = sp.sph_harm_y(l, m, theta, phi)
    return np.abs(R * Y) ** 2


def plot_six_densities(states: list[tuple[int, int, int]],
                       save: str | None = None) -> None:
    """Plot six |ψ|² cross-sections in the xz-plane (y=0)."""
    pts = 400
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for ax, (n, l, m) in zip(axes.flat, states):
        limit = 2 * n * n
        x = np.linspace(-limit, limit, pts)
        z = np.linspace(-limit, limit, pts)
        X, Z = np.meshgrid(x, z)

        r = np.sqrt(X**2 + Z**2)
        theta = np.arctan2(np.abs(X), Z)
        phi = np.where(X >= 0, 0.0, np.pi)

        density = hydrogen_density(n, l, m, r, theta, phi)
        vmax = float(np.percentile(density, 99.5))
        im = ax.pcolormesh(
            X, Z, density, cmap="inferno",
            shading="auto", vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_title(rf"${n},{l},{m}$")

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


STATES = [(1, 0, 0), (2, 1, 0), (2, 1, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2)]
SAVE = "xz_plane_6.png"


if __name__ == "__main__":
    plot_six_densities(STATES, save=SAVE)
