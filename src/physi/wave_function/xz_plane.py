"""Plot hydrogen atom probability density |ψ_{n,l,m}|² in the xz-plane."""

import argparse
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


def plot_density(n: int, l: int, m: int, save: str | None = None) -> None:
    """Plot |ψ|² cross-section in the xz-plane (y=0)."""
    limit = 2 * n * (n)
    pts = 600
    x = np.linspace(-limit, limit, pts)
    z = np.linspace(-limit, limit, pts)
    X, Z = np.meshgrid(x, z)

    r = np.sqrt(X**2 + Z**2)
    theta = np.arctan2(np.abs(X), Z)          # polar angle from z-axis
    phi = np.where(X >= 0, 0.0, np.pi)        # azimuthal angle (y=0 slice)

    density = hydrogen_density(n, l, m, r, theta, phi)

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = float(np.percentile(density, 99.5))  # clip hot-spots
    im = ax.pcolormesh(
        X, Z, density, cmap="inferno",
        shading="auto", vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x  (a₀)")
    ax.set_ylabel("z  (a₀)")
    ax.set_title(rf"$|\psi_{{{n},{l},{m}}}|^2$   (xz-plane)")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$", shrink=0.8)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot hydrogen probability density |ψ_{n,l,m}|²")
    parser.add_argument(
        "n", type=int, help="Principal quantum number (n ≥ 1)")

    parser.add_argument(
        "l", type=int, help="Angular momentum (0 ≤ l < n)")
    parser.add_argument(
        "m", type=int, help="Magnetic QN (|m| ≤ l)")
    parser.add_argument("--save", type=str, help="Path to save figure")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("n must be ≥ 1")
    if not 0 <= args.l < args.n:
        parser.error(
            f"l must satisfy 0 ≤ l < n,"
            f" got l={args.l}, n={args.n}"
        )
    if abs(args.m) > args.l:
        parser.error(f"|m| must be ≤ l, got m={args.m}, l={args.l}")

    plot_density(args.n, args.l, args.m, save=args.save)


if __name__ == "__main__":
    main()
