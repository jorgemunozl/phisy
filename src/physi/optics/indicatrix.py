#!/usr/bin/env python3
"""
Interactive 3D visualization of the optical indicatrix (index ellipsoid).

The indicatrix equation in principal coordinates:
    x²/n_x² + y²/n_y² + z²/n_z² = 1

where n_x, n_y, n_z are the principal refractive indices of an anisotropic
crystal.  The distance from the origin to the surface in any direction gives
the refractive index for light polarised along that direction.

Crystal families
----------------
* n_x = n_y = n_z  →  isotropic  (sphere)
* n_x = n_y ≠ n_z  →  uniaxial   (ellipsoid of revolution)
    * n_z > n_x  →  positive uniaxial
    * n_z < n_x  →  negative uniaxial
* n_x ≠ n_y ≠ n_z →  biaxial    (triaxial ellipsoid) — two optic axes exist

Run:
  uv run python src/optics/indicatrix.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ellipsoid_mesh(nx: float, ny: float, nz: float, n: int = 60):
    """Return (X, Y, Z) coordinate arrays for the indicatrix surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    cosu, sinu = np.cos(u), np.sin(u)
    cosv, sinv = np.cos(v), np.sin(v)
    X = nx * np.outer(cosu, sinv)
    Y = ny * np.outer(sinu, sinv)
    Z = nz * np.outer(np.ones_like(u), cosv)
    return X, Y, Z


def _crystal_type(nx: float, ny: float, nz: float) -> str:
    """Classify the crystal based on principal indices."""
    eps = 1e-9
    if abs(nx - ny) < eps and abs(ny - nz) < eps:
        return "isotropic"
    if abs(nx - ny) < eps:
        return "positive uniaxial" if nz > nx else "negative uniaxial"
    if abs(ny - nz) < eps:
        return "positive uniaxial" if nx > ny else "negative uniaxial"
    return "biaxial"


def _optic_axes(nx: float, ny: float, nz: float):
    """
    Return the optic axis direction(s) for a biaxial crystal.

    For a biaxial crystal with nx < ny < nz, there are two optic axes
    symmetric about the z-axis (the largest index direction).

    Returns the angle (in radians) from the z-axis, or None if not biaxial.
    """
    eps = 1e-9
    if nx < ny - eps and ny < nz - eps:
        # Biaxial case: tan(V) = sqrt( (1/nx^2 - 1/ny^2) / (1/ny^2 - 1/nz^2) )
        inv_nx2 = 1.0 / nx**2
        inv_ny2 = 1.0 / ny**2
        inv_nz2 = 1.0 / nz**2
        num = inv_nx2 - inv_ny2
        den = inv_ny2 - inv_nz2
        if den <= 0 or num < 0:
            return None
        V = math.atan(math.sqrt(num / den))
        return V
    return None


@dataclass
class IndicatrixInfo:
    nx: float
    ny: float
    nz: float
    crystal: str
    optic_axis_angle_deg: float | None  # half-angle between optic axes (biaxial only)
    delta_n: float  # birefringence = |n_max - n_min|

    @classmethod
    def compute(cls, nx: float, ny: float, nz: float) -> IndicatrixInfo:
        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            crystal=_crystal_type(nx, ny, nz),
            optic_axis_angle_deg=(
                math.degrees(_optic_axes(nx, ny, nz))
                if _optic_axes(nx, ny, nz) is not None
                else None
            ),
            delta_n=max(nx, ny, nz) - min(nx, ny, nz),
        )


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------


def build_figure(initial_nx: float, initial_ny: float, initial_nz: float):
    """Set up the figure with 3D axes, information panel, and slider controls."""
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(left=0.05, right=0.65, top=0.95, bottom=0.22)

    # -- 3D axes -----------------------------------------------------------
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Indicatrix (index ellipsoid)")

    # -- Information axes (text) -------------------------------------------
    ax_info = fig.add_axes([0.72, 0.35, 0.26, 0.58])
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    # -- Slider axes -------------------------------------------------------
    slider_color = "lightgoldenrodyellow"
    ax_nx = fig.add_axes([0.12, 0.10, 0.30, 0.03])
    ax_ny = fig.add_axes([0.12, 0.06, 0.30, 0.03])
    ax_nz = fig.add_axes([0.12, 0.02, 0.30, 0.03])

    slider_nx = Slider(
        ax=ax_nx,
        label="$n_x$",
        valmin=1.0,
        valmax=3.0,
        valinit=initial_nx,
        valstep=0.001,
        color=slider_color,
    )
    slider_ny = Slider(
        ax=ax_ny,
        label="$n_y$",
        valmin=1.0,
        valmax=3.0,
        valinit=initial_ny,
        valstep=0.001,
        color=slider_color,
    )
    slider_nz = Slider(
        ax=ax_nz,
        label="$n_z$",
        valmin=1.0,
        valmax=3.0,
        valinit=initial_nz,
        valstep=0.001,
        color=slider_color,
    )

    # -- Preset buttons ----------------------------------------------------
    ax_iso = fig.add_axes([0.72, 0.07, 0.08, 0.05])
    ax_uni_pos = fig.add_axes([0.81, 0.07, 0.08, 0.05])
    ax_uni_neg = fig.add_axes([0.72, 0.01, 0.08, 0.05])
    ax_biax = fig.add_axes([0.81, 0.01, 0.08, 0.05])

    btn_iso = _make_button(ax_iso, "Isotropic")
    btn_uni_pos = _make_button(ax_uni_pos, "Uniaxial +")
    btn_uni_neg = _make_button(ax_uni_neg, "Uniaxial −")
    btn_biax = _make_button(ax_biax, "Biaxial")

    return (
        fig,
        ax3d,
        ax_info,
        slider_nx,
        slider_ny,
        slider_nz,
        btn_iso,
        btn_uni_pos,
        btn_uni_neg,
        btn_biax,
    )


def _make_button(ax, label: str):
    from matplotlib.widgets import Button

    return Button(ax, label, color="0.85", hovercolor="0.95")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_principal_axes(ax3d, nx: float, ny: float, nz: float):
    """Draw coloured arrows for the three principal semi-axes."""
    limit = max(nx, ny, nz) * 1.25
    ax3d.set_xlim(-limit, limit)
    ax3d.set_ylim(-limit, limit)
    ax3d.set_zlim(-limit, limit)

    # Remove old arrow artists if they exist.
    for artist in getattr(ax3d, "_principal_arrows", []):
        artist.remove()
    ax3d._principal_arrows = []

    colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    for axis, val in [("x", nx), ("y", ny), ("z", nz)]:
        vec = np.zeros(3)
        idx = ord(axis) - ord("x")
        vec[idx] = val
        arr = ax3d.quiver(
            0,
            0,
            0,
            vec[0],
            vec[1],
            vec[2],
            color=colors[axis],
            linewidth=2.5,
            arrow_length_ratio=0.12,
            alpha=0.85,
        )
        ax3d._principal_arrows.append(arr)

    # Add labels at the tips.
    for artist in getattr(ax3d, "_principal_labels", []):
        artist.remove()
    ax3d._principal_labels = []
    for axis, val in [("x", nx), ("y", ny), ("z", nz)]:
        vec = np.zeros(3)
        idx = ord(axis) - ord("x")
        vec[idx] = val
        label = ax3d.text(
            vec[0] * 1.08,
            vec[1] * 1.08,
            vec[2] * 1.08,
            f"$n_{{{axis}}}$",
            color=colors[axis],
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax3d._principal_labels.append(label)


def _draw_optic_axes(ax3d, nx: float, ny: float, nz: float):
    """Draw optic axes for a biaxial crystal (lines through the origin)."""
    for artist in getattr(ax3d, "_optic_axis_artists", []):
        artist.remove()
    ax3d._optic_axis_artists = []

    if nx < ny < nz:
        V = _optic_axes(nx, ny, nz)
        if V is not None:
            length = max(nx, ny, nz) * 1.15
            for sign in (+1, -1):
                dx = length * math.sin(V) * sign
                dz = length * math.cos(V)
                (line,) = ax3d.plot(
                    [0, dx],
                    [0, 0],
                    [0, dz],
                    color="#aa3377",
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.7,
                )
                ax3d._optic_axis_artists.append(line)


def _draw_cross_section_circles(ax3d, nx: float, ny: float, nz: float):
    """Draw the three principal cross-section circles (planes xy, xz, yz)."""
    for artist in getattr(ax3d, "_cross_sections", []):
        artist.remove()
    ax3d._cross_sections = []

    theta = np.linspace(0, 2 * np.pi, 100)

    # XY plane (z=0)
    cx, cy = nx * np.cos(theta), ny * np.sin(theta)
    (l1,) = ax3d.plot(cx, cy, np.zeros_like(theta), "k--", alpha=0.25, linewidth=0.8)
    ax3d._cross_sections.append(l1)

    # XZ plane (y=0)
    cx, cz = nx * np.cos(theta), nz * np.sin(theta)
    (l2,) = ax3d.plot(cx, np.zeros_like(theta), cz, "k--", alpha=0.25, linewidth=0.8)
    ax3d._cross_sections.append(l2)

    # YZ plane (x=0)
    cy, cz = ny * np.cos(theta), nz * np.sin(theta)
    (l3,) = ax3d.plot(np.zeros_like(theta), cy, cz, "k--", alpha=0.25, linewidth=0.8)
    ax3d._cross_sections.append(l3)


def redraw(
    ax3d,
    ax_info,
    nx: float,
    ny: float,
    nz: float,
    n_mesh: int = 60,
    alpha: float = 0.75,
):
    """Update the 3D plot and info panel for the current indices."""
    # Clear existing surface.
    for artist in getattr(ax3d, "_surface", []):
        artist.remove()

    # Generate & plot new ellipsoid.
    X, Y, Z = _ellipsoid_mesh(nx, ny, nz, n=n_mesh)
    surf = ax3d.plot_surface(
        X,
        Y,
        Z,
        rstride=2,
        cstride=2,
        color="#1f77b4",
        alpha=alpha,
        edgecolor="gray",
        linewidth=0.15,
        antialiased=True,
    )
    ax3d._surface = [surf]

    # Principal semi-axes.
    _draw_principal_axes(ax3d, nx, ny, nz)

    # Cross-section circles.
    _draw_cross_section_circles(ax3d, nx, ny, nz)

    # Optic axes (biaxial only).
    _draw_optic_axes(ax3d, nx, ny, nz)

    # Information panel.
    info = IndicatrixInfo.compute(nx, ny, nz)
    _update_info_panel(ax_info, info)


def _update_info_panel(ax_info, info: IndicatrixInfo):
    ax_info.clear()
    ax_info.axis("off")

    lines = [
        ("Principal refractive indices", ""),
        (f"  n_x = {info.nx:.4f}", ""),
        (f"  n_y = {info.ny:.4f}", ""),
        (f"  n_z = {info.nz:.4f}", ""),
        ("", ""),
        ("Crystal type", f"{info.crystal}"),
        ("Birefringence Δn", f"{info.delta_n:.4f}"),
    ]

    if info.optic_axis_angle_deg is not None:
        lines.append(
            (
                "Optic axis half-angle V",
                f"{info.optic_axis_angle_deg:.2f}°",
            )
        )
        lines.append(
            (
                "Optic axes (2V)",
                f"{2 * info.optic_axis_angle_deg:.2f}°",
            )
        )

    # Additional derived info.
    if info.crystal == "isotropic":
        lines.append(("", ""))
        lines.append("—— Single refractive index in all directions", "")
    elif "uniaxial" in info.crystal:
        no = (
            min(info.nx, info.nz)
            if abs(info.nx - info.ny) < 1e-9
            else min(info.ny, info.nz)
        )
        ne = (
            max(info.nx, info.nz)
            if abs(info.nx - info.ny) < 1e-9
            else max(info.ny, info.nz)
        )
        sign = "+" if "positive" in info.crystal else "−"
        lines.append(("", ""))
        lines.append((f"n_o (ordinary) = {no:.4f}", ""))
        lines.append((f"n_e (extraordinary) = {ne:.4f}", ""))
        lines.append((f"Crystal sign: {sign}", ""))
    elif info.crystal == "biaxial":
        axes = sorted([info.nx, info.ny, info.nz])
        lines.append(("", ""))
        lines.append((f"n_α (smallest) = {axes[0]:.4f}", ""))
        lines.append((f"n_β (middle)   = {axes[1]:.4f}", ""))
        lines.append((f"n_γ (largest)  = {axes[2]:.4f}", ""))

    y = 0.98
    for left, right in lines:
        ax_info.text(
            0.02,
            y,
            left,
            transform=ax_info.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontweight="bold" if left and not left.startswith("  ") else "normal",
        )
        if right:
            ax_info.text(
                0.98,
                y,
                right,
                transform=ax_info.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                fontfamily="monospace",
            )
        y -= 0.055


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Interactive 3D visualization of the optical indicatrix."
    )
    p.add_argument("--nx", type=float, default=1.5, help="Principal index n_x")
    p.add_argument("--ny", type=float, default=1.7, help="Principal index n_y")
    p.add_argument("--nz", type=float, default=2.0, help="Principal index n_z")
    p.add_argument("--mesh", type=int, default=60, help="Mesh resolution")
    p.add_argument("--alpha", type=float, default=0.75, help="Surface opacity")
    args = p.parse_args()

    # --- Build figure -----------------------------------------------------
    (
        fig,
        ax3d,
        ax_info,
        slider_nx,
        slider_ny,
        slider_nz,
        btn_iso,
        btn_uni_pos,
        btn_uni_neg,
        btn_biax,
    ) = build_figure(args.nx, args.ny, args.nz)

    # Initial draw.
    redraw(ax3d, ax_info, args.nx, args.ny, args.nz, n_mesh=args.mesh, alpha=args.alpha)

    # --- Connect sliders --------------------------------------------------
    def _on_change(val):
        redraw(
            ax3d,
            ax_info,
            slider_nx.val,
            slider_ny.val,
            slider_nz.val,
            n_mesh=args.mesh,
            alpha=args.alpha,
        )
        fig.canvas.draw_idle()

    slider_nx.on_changed(_on_change)
    slider_ny.on_changed(_on_change)
    slider_nz.on_changed(_on_change)

    # --- Connect preset buttons -------------------------------------------
    def _set_preset(nx: float, ny: float, nz: float):
        slider_nx.set_val(nx)
        slider_ny.set_val(ny)
        slider_nz.set_val(nz)
        # on_changed will fire automatically.

    btn_iso.on_clicked(lambda _: _set_preset(1.5, 1.5, 1.5))
    btn_uni_pos.on_clicked(lambda _: _set_preset(1.5, 1.5, 2.0))
    btn_uni_neg.on_clicked(lambda _: _set_preset(2.0, 2.0, 1.5))
    btn_biax.on_clicked(lambda _: _set_preset(1.5, 1.7, 2.0))

    plt.show()


if __name__ == "__main__":
    main()
