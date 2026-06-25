"""3D isosurface plot of hydrogen probability density |ψ_{n,l,m}|²."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as sp
from math import factorial


def hydrogen_radial(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """Radial wave function R_{nl}(r) in atomic units (a₀ = 1)."""
    rho = 2 * r / n
    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)(rho)
    coeff = (
        factorial(n - l - 1) / (2 * n * factorial(n + l))
    )
    norm = np.sqrt((2 / n) ** 3 * coeff)
    return norm * np.exp(-rho / 2) * rho**l * laguerre


def find_isosurfaces(n, l, m, iso_frac=0.05, n_ang=80):
    """
    Find isosurface sheets r(θ,φ) where |ψ|² = iso_val.

    Returns (THETA, PHI, R_sheets, iso_val) where
    R_sheets has shape (n_sheets, n_ang, n_ang) with NaN
    where no crossing exists.
    """
    r_max = 4 * n**2
    r_arr = np.linspace(1e-6, r_max, 2000)
    R2 = hydrogen_radial(n, l, r_arr) ** 2

    theta = np.linspace(0, np.pi, n_ang)
    phi = np.linspace(0, 2 * np.pi, n_ang)
    THETA, PHI = np.meshgrid(theta, phi)
    Y2 = np.abs(sp.sph_harm_y(l, m, THETA, PHI)) ** 2

    iso_val = iso_frac * R2.max() * Y2.max()

    # R²(r) has up to (n-l) peaks → up to 2(n-l) crossings
    max_cross = 2 * (n - l)
    R_sheets = np.full(
        (max_cross, n_ang, n_ang), np.nan
    )

    for i in range(n_ang):
        for j in range(n_ang):
            y2 = Y2[i, j]
            if y2 < 1e-30:
                continue
            target = iso_val / y2
            diff = R2 - target
            signs = np.sign(diff)
            idx = np.where(np.diff(signs) != 0)[0]
            for k, ci in enumerate(idx[:max_cross]):
                r0, r1 = r_arr[ci], r_arr[ci + 1]
                d0, d1 = diff[ci], diff[ci + 1]
                R_sheets[k, i, j] = (
                    r0 - d0 * (r1 - r0) / (d1 - d0)
                )

    return THETA, PHI, R_sheets, iso_val


def plot_isosurface(
    n: int, l: int, m: int,
    iso_frac: float = 0.05,
    save: str | None = None,
) -> None:
    """Plot 3D isosurface of |ψ_{n,l,m}|²."""
    THETA, PHI, R_sheets, iso_val = find_isosurfaces(
        n, l, m, iso_frac=iso_frac
    )

    # Color by Re(Y_lm) → shows orbital phase
    Y_real = np.real(sp.sph_harm_y(l, m, THETA, PHI))
    vmax = np.abs(Y_real).max() or 1.0
    normalizer = plt.Normalize(-vmax, vmax)
    cmap = plt.colormaps["RdBu_r"]
    colors = cmap(normalizer(Y_real))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for s in range(R_sheets.shape[0]):
        R = R_sheets[s]
        if np.all(np.isnan(R)):
            continue
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        ax.plot_surface(
            X, Y, Z,
            facecolors=colors,
            rstride=1, cstride=1,
            antialiased=True, alpha=0.85,
        )

    r_lim = np.nanmax(R_sheets) * 1.1
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_zlim(-r_lim, r_lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x (a₀)")
    ax.set_ylabel("y (a₀)")
    ax.set_zlabel("z (a₀)")
    ax.set_title(
        rf"$|\psi_{{{n},{l},{m}}}|^2$ "
        f"isosurface ({str(iso_frac)} of max)"
    )

    mappable = cm.ScalarMappable(
        cmap="RdBu_r", norm=normalizer
    )
    mappable.set_array(Y_real)
    cbar = fig.colorbar(
        mappable, ax=ax, shrink=0.6, pad=0.1
    )
    cbar.set_label(r"Re($Y_\ell^m$)")
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    # plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D isosurface of |ψ_{n,l,m}|²"
    )
    parser.add_argument(
        "n", type=int,
        help="Principal quantum number (n ≥ 1)")
    parser.add_argument(
        "l", type=int,
        help="Angular momentum (0 ≤ l < n)")
    parser.add_argument(
        "m", type=int,
        help="Magnetic QN (|m| ≤ l)")
    parser.add_argument(
        "--iso", type=float, default=0.05,
        help="Iso-level as fraction of max (default 0.05)"
    )
    parser.add_argument(
        "--save", type=str,
        help="Path to save figure")
    # args = parser.parse_args()
    n = 3
    l = 2
    m = 0
    iso = 0.05
    save = f"isosurface/{n}_{l}_{m}.png"
    
    args = argparse.Namespace(**{
        "n": n,
        "l": l,
        "m": m,
        "iso": iso,
        "save": save,
    })
    if args.n < 1:
        parser.error("n must be ≥ 1")
    if not 0 <= args.l < args.n:
        parser.error(
            f"l must satisfy 0 ≤ l < n,"
            f" got l={args.l}, n={args.n}"
        )
    if abs(args.m) > args.l:
        parser.error(
            f"|m| must be ≤ l,"
            f" got m={args.m}, l={args.l}"
        )

    plot_isosurface(
        args.n, args.l, args.m,
        iso_frac=args.iso, save=args.save,
    )


if __name__ == "__main__":
    main()
