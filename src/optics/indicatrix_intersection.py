#!/usr/bin/env python3
"""
Interactive visualization of the intersection of the indicatrix ellipsoid
with an arbitrary plane through its centre.

Key fact: the intersection of an ellipsoid with a plane is always an ellipse.
This is the fundamental geometry behind the index ellipse (or "vibration
ellipse") used to determine the two allowed refractive indices and
polarisation directions for a given wave-vector direction in a crystal.

The plane is defined by its unit normal n(θ, φ):
    n = (sin θ cos φ, sin θ sin φ, cos θ)

Orthonormal basis spanning the plane:
    u = (cos θ cos φ, cos θ sin φ, -sin θ)
    v = (-sin φ, cos φ, 0)

Any point in the plane:  p(s, t) = s·u + t·v

Substituting into the indicatrix equation  x²/nx² + y²/ny² + z²/nz² = 1
gives a quadratic form in (s, t):
    A·s² + 2B·s·t + C·t² = 1

The eigenvalues of [[A, B], [B, C]] give the semi-axis lengths of the
intersection ellipse; the eigenvectors give their orientation within the plane.

Run:
  uv run python src/optics/indicatrix_intersection.py
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ellipsoid_mesh(nx: float, ny: float, nz: float, n: int = 50):
    """Spherical-coordinate parametrisation of the ellipsoid surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    cosu, sinu = np.cos(u), np.sin(u)
    cosv, sinv = np.cos(v), np.sin(v)
    X = nx * np.outer(cosu, sinv)
    Y = ny * np.outer(sinu, sinv)
    Z = nz * np.outer(np.ones_like(u), cosv)
    return X, Y, Z


def _plane_basis(theta: float, phi: float):
    """
    Return orthonormal basis (n, u, v) where n is the plane normal,
    and u, v span the plane.
    """
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    n = np.array([st * cp, st * sp, ct])
    u = np.array([ct * cp, ct * sp, -st])
    v = np.array([-sp, cp, 0.0])
    return n, u, v


def _intersection_ellipse(
    nx: float,
    ny: float,
    nz: float,
    theta: float,
    phi: float,
):
    """
    Compute the intersection ellipse of the indicatrix with the plane
    through the origin having normal n(theta, phi).

    Returns
    -------
    axes_3d : list of two 3D vectors (semi-axis directions in world coords)
    lengths : tuple (a, b)  semi-axis lengths
    s_t_ellipse : (s_pts, t_pts)  parametric ellipse in plane coordinates
    A, B, C   : coefficients of the quadratic form
    n, u, v   : the plane basis
    """
    n, u, v = _plane_basis(theta, phi)

    # Quadratic form coefficients from substituting p = s·u + t·v
    diag = np.array([1.0 / nx**2, 1.0 / ny**2, 1.0 / nz**2])
    A = np.dot(diag, u * u)  # u_x²/nx² + u_y²/ny² + u_z²/nz²
    B = np.dot(diag, u * v)  # u_x v_x / nx² + ...
    C = np.dot(diag, v * v)  # v_x²/nx² + ...

    M = np.array([[A, B], [B, C]])
    eigvals, eigvecs = np.linalg.eigh(M)  # ascending eigenvalues

    # Semi-axes in (s, t) plane coordinates.
    a = 1.0 / math.sqrt(eigvals[0])  # long semi-axis
    b = 1.0 / math.sqrt(eigvals[1])  # short semi-axis
    e1 = eigvecs[:, 0]  # direction of a  (in s,t)
    e2 = eigvecs[:, 1]  # direction of b  (in s,t)

    # Parametric ellipse in (s, t).
    alpha = np.linspace(0, 2 * np.pi, 200)
    s_pts = a * e1[0] * np.cos(alpha) + b * e2[0] * np.sin(alpha)
    t_pts = a * e1[1] * np.cos(alpha) + b * e2[1] * np.sin(alpha)

    # 3D coordinates of the two semi-axis tips.
    axes_3d = [e1[0] * a * u + e1[1] * a * v, e2[0] * b * u + e2[1] * b * v]

    return axes_3d, (a, b), (s_pts, t_pts), (A, B, C), (n, u, v)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(
    initial_nx: float,
    initial_ny: float,
    initial_nz: float,
    initial_theta_deg: float,
    initial_phi_deg: float,
):
    """Create figure with 3D scene, 2D ellipse panel, sliders, and info."""
    fig = plt.figure(figsize=(16, 8.5))
    fig.subplots_adjust(left=0.04, right=0.68, top=0.96, bottom=0.26)

    # ----- 3D axes --------------------------------------------------------
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Ellipsoid ∩ plane = ellipse")

    # ----- 2D ellipse axes ------------------------------------------------
    ax2d = fig.add_axes([0.70, 0.38, 0.28, 0.56])
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("s (axis u)")
    ax2d.set_ylabel("t (axis v)")
    ax2d.set_title("Intersection ellipse (in plane coordinates)")
    ax2d.grid(True, alpha=0.3)

    # ----- Sliders --------------------------------------------------------
    slider_color = "lightgoldenrodyellow"
    ax_nx = fig.add_axes([0.12, 0.13, 0.22, 0.025])
    ax_ny = fig.add_axes([0.12, 0.10, 0.22, 0.025])
    ax_nz = fig.add_axes([0.12, 0.07, 0.22, 0.025])

    ax_theta = fig.add_axes([0.45, 0.13, 0.20, 0.025])
    ax_phi = fig.add_axes([0.45, 0.10, 0.20, 0.025])

    s_nx = Slider(
        ax_nx, "$n_x$", 1.0, 3.0, valinit=initial_nx, valstep=0.001, color=slider_color
    )
    s_ny = Slider(
        ax_ny, "$n_y$", 1.0, 3.0, valinit=initial_ny, valstep=0.001, color=slider_color
    )
    s_nz = Slider(
        ax_nz, "$n_z$", 1.0, 3.0, valinit=initial_nz, valstep=0.001, color=slider_color
    )

    s_theta = Slider(
        ax_theta,
        r"$\theta$ (polar) [°]",
        0.0,
        90.0,
        valinit=initial_theta_deg,
        valstep=0.5,
        color=slider_color,
    )
    s_phi = Slider(
        ax_phi,
        r"$\phi$ (azim.) [°]",
        -180.0,
        180.0,
        valinit=initial_phi_deg,
        valstep=0.5,
        color=slider_color,
    )

    # ----- Info axes ------------------------------------------------------
    ax_info = fig.add_axes([0.02, 0.01, 0.64, 0.045])
    ax_info.axis("off")

    return fig, ax3d, ax2d, ax_info, s_nx, s_ny, s_nz, s_theta, s_phi


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _clear_artists(ax, attr: str):
    for a in getattr(ax, attr, []):
        a.remove()
    setattr(ax, attr, [])


def redraw(
    ax3d,
    ax2d,
    ax_info,
    nx: float,
    ny: float,
    nz: float,
    theta_deg: float,
    phi_deg: float,
):
    """Full redraw of the 3D scene, 2D ellipse, and info bar."""
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    n, u, v = _plane_basis(theta, phi)

    # ---- 3D: ellipsoid surface -------------------------------------------
    _clear_artists(ax3d, "_surf_artists")
    X, Y, Z = _ellipsoid_mesh(nx, ny, nz, n=50)
    surf = ax3d.plot_surface(
        X,
        Y,
        Z,
        rstride=2,
        cstride=2,
        color="#1f77b4",
        alpha=0.25,
        edgecolor="gray",
        linewidth=0.12,
        antialiased=True,
    )
    ax3d._surf_artists.append(surf)

    # ---- 3D: principal axes ----------------------------------------------
    _clear_artists(ax3d, "_ax_artists")
    colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    for axis, val in [("x", nx), ("y", ny), ("z", nz)]:
        idx = ord(axis) - ord("x")
        vec = np.zeros(3)
        vec[idx] = val
        q = ax3d.quiver(
            0,
            0,
            0,
            vec[0],
            vec[1],
            vec[2],
            color=colors[axis],
            linewidth=2.0,
            arrow_length_ratio=0.10,
            alpha=0.6,
        )
        ax3d._ax_artists.append(q)

    # ---- 3D: plane patch -------------------------------------------------
    _clear_artists(ax3d, "_plane_artists")
    limit = max(nx, ny, nz) * 1.3
    s_vals = np.linspace(-limit, limit, 10)
    t_vals = np.linspace(-limit, limit, 10)
    S, T = np.meshgrid(s_vals, t_vals)
    PX = S * u[0] + T * v[0]
    PY = S * u[1] + T * v[1]
    PZ = S * u[2] + T * v[2]

    # Semi-transparent plane.
    verts = []
    for i in range(len(s_vals) - 1):
        for j in range(len(t_vals) - 1):
            quad = [
                [PX[i, j], PY[i, j], PZ[i, j]],
                [PX[i + 1, j], PY[i + 1, j], PZ[i + 1, j]],
                [PX[i + 1, j + 1], PY[i + 1, j + 1], PZ[i + 1, j + 1]],
                [PX[i, j + 1], PY[i, j + 1], PZ[i, j + 1]],
            ]
            verts.append(quad)
    plane_poly = Poly3DCollection(
        verts,
        alpha=0.12,
        color="#ff7f0e",
        zorder=1,
    )
    ax3d.add_collection(plane_poly)
    ax3d._plane_artists.append(plane_poly)

    # Normal vector arrow.
    arr_n = ax3d.quiver(
        0,
        0,
        0,
        n[0] * limit * 0.8,
        n[1] * limit * 0.8,
        n[2] * limit * 0.8,
        color="#d62728",
        linewidth=2.5,
        arrow_length_ratio=0.12,
        alpha=0.9,
    )
    ax3d._plane_artists.append(arr_n)
    # Normal label.
    txt_n = ax3d.text(
        n[0] * limit * 0.85,
        n[1] * limit * 0.85,
        n[2] * limit * 0.85,
        "n",
        color="#d62728",
        fontsize=11,
        fontweight="bold",
    )
    ax3d._plane_artists.append(txt_n)

    # ---- 3D: intersection ellipse ----------------------------------------
    _clear_artists(ax3d, "_ell_artists")
    (
        axes_3d,
        (a_len, b_len),
        (s_pts, t_pts),
        quad_coeffs,
        (n_basis, u_basis, v_basis),
    ) = _intersection_ellipse(nx, ny, nz, theta, phi)
    # Recompute eigenvectors for 2D panel.
    A_, B_, C_ = quad_coeffs
    M_ = np.array([[A_, B_], [B_, C_]])
    eigvals, eigvecs = np.linalg.eigh(M_)
    a_len_ = 1.0 / math.sqrt(eigvals[0])
    b_len_ = 1.0 / math.sqrt(eigvals[1])
    e1 = eigvecs[:, 0]
    e2 = eigvecs[:, 1]

    # 3D curve.
    ell_x = s_pts * u[0] + t_pts * v[0]
    ell_y = s_pts * u[1] + t_pts * v[1]
    ell_z = s_pts * u[2] + t_pts * v[2]

    (ell_curve,) = ax3d.plot(
        ell_x, ell_y, ell_z, color="#ff7f0e", linewidth=3.0, zorder=5
    )
    ax3d._ell_artists.append(ell_curve)

    # Semi-axis arrows in the plane.
    for vec, col, lbl in [
        (axes_3d[0], "#2ca02c", f"a = {a_len:.3f}"),
        (axes_3d[1], "#d62728", f"b = {b_len:.3f}"),
    ]:
        q = ax3d.quiver(
            0,
            0,
            0,
            vec[0],
            vec[1],
            vec[2],
            color=col,
            linewidth=2.8,
            arrow_length_ratio=0.12,
            zorder=6,
        )
        ax3d._ell_artists.append(q)

    # ---- 3D: limits ------------------------------------------------------
    lim = max(nx, ny, nz) * 1.35
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)

    # ---- 2D: ellipse in plane coordinates --------------------------------
    _clear_artists(ax2d, "_2d_artists")
    ax2d.cla()
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("s (axis u)")
    ax2d.set_ylabel("t (axis v)")
    ax2d.set_title("Intersection ellipse (in plane coordinates)")
    ax2d.grid(True, alpha=0.3)

    # Axes lines.
    lim2d = max(a_len, b_len) * 1.3
    ax2d.set_xlim(-lim2d, lim2d)
    ax2d.set_ylim(-lim2d, lim2d)
    ax2d.axhline(0, color="gray", linewidth=0.5)
    ax2d.axvline(0, color="gray", linewidth=0.5)

    # The ellipse.
    ax2d.plot(s_pts, t_pts, color="#ff7f0e", linewidth=2.5, zorder=3)
    ax2d.plot(s_pts[0], t_pts[0], "o", color="#ff7f0e", markersize=4, zorder=3)

    # Semi-axes directions in the (s, t) plane coordinates.
    s1, t1 = e1[0] * a_len_, e1[1] * a_len_
    s2, t2 = e2[0] * b_len_, e2[1] * b_len_

    for (sss, ttt), col in [
        ((s1, t1), "#2ca02c"),
        ((-s1, -t1), "#2ca02c"),
        ((s2, t2), "#d62728"),
        ((-s2, -t2), "#d62728"),
    ]:
        ax2d.annotate(
            "",
            xy=(sss, ttt),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=col, lw=2.2),
        )

    ax2d.text(
        s1 * 1.08,
        t1 * 1.08,
        f"a = {a_len_:.3f}",
        color="#2ca02c",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax2d.text(
        s2 * 1.08,
        t2 * 1.08,
        f"b = {b_len_:.3f}",
        color="#d62728",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    ax2d._2d_artists = []  # We cleared via cla(), nothing to track.

    # ---- Info bar --------------------------------------------------------
    ax_info.clear()
    ax_info.axis("off")
    n1_, n2_ = sorted([a_len_, b_len_], reverse=True)
    biref = n1_ - n2_
    info_lines = (
        f"  n_x={nx:.3f}  n_y={ny:.3f}  n_z={nz:.3f}  "
        f"|  θ={theta_deg:.1f}°  φ={phi_deg:.1f}°  "
        f"|  n₁'={n1_:.4f}  n₂'={n2_:.4f}  Δn'={biref:.4f}  "
        f"|  semi-axes:  a={n1_:.3f}  b={n2_:.3f}"
    )
    ax_info.text(0.0, 0.5, info_lines, va="center", fontsize=10, fontfamily="monospace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Interactive ellipsoid–plane intersection (index ellipse)."
    )
    p.add_argument("--nx", type=float, default=1.5)
    p.add_argument("--ny", type=float, default=1.7)
    p.add_argument("--nz", type=float, default=2.0)
    p.add_argument(
        "--theta",
        type=float,
        default=35.0,
        help="Polar angle of plane normal (degrees)",
    )
    p.add_argument(
        "--phi",
        type=float,
        default=45.0,
        help="Azimuthal angle of plane normal (degrees)",
    )
    args = p.parse_args()

    fig, ax3d, ax2d, ax_info, s_nx, s_ny, s_nz, s_theta, s_phi = build_figure(
        args.nx, args.ny, args.nz, args.theta, args.phi
    )

    redraw(ax3d, ax2d, ax_info, args.nx, args.ny, args.nz, args.theta, args.phi)

    def _update(_val=None):
        redraw(
            ax3d, ax2d, ax_info, s_nx.val, s_ny.val, s_nz.val, s_theta.val, s_phi.val
        )
        fig.canvas.draw_idle()

    s_nx.on_changed(_update)
    s_ny.on_changed(_update)
    s_nz.on_changed(_update)
    s_theta.on_changed(_update)
    s_phi.on_changed(_update)

    plt.show()


if __name__ == "__main__":
    main()
