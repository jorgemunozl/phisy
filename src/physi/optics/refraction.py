#!/usr/bin/env python3
"""
Refraction at a planar interface (Snell's law).

Run:
  uv run python src/optics/refraction.py --show
  uv run python src/optics/refraction.py --n1 1.0 --n2 1.5 --theta1 50
  uv run python src/optics/refraction.py --n1 1.5 --n2 1.0 --animate --outfile tir.gif
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class SnellResult:
    theta1_rad: float
    theta2_rad: float | None  # None => total internal reflection
    critical_rad: float | None  # None => no critical angle (n1 <= n2)


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def snell(n1: float, n2: float, theta1_rad: float) -> SnellResult:
    """
    Angles are measured from the surface normal.

    Returns theta2_rad=None when n1 > n2 and theta1 exceeds the critical angle,
    i.e. when total internal reflection occurs.
    """
    if not (n1 > 0.0 and n2 > 0.0):
        raise ValueError("Refractive indices must be positive.")

    # Critical angle exists only for n1 > n2.
    critical = None
    if n1 > n2:
        critical = math.asin(_clamp(n2 / n1, -1.0, 1.0))

    s1 = math.sin(theta1_rad)
    s2 = (n1 / n2) * s1
    if abs(s2) > 1.0:
        return SnellResult(theta1_rad=theta1_rad, theta2_rad=None, critical_rad=critical)

    theta2 = math.asin(_clamp(s2, -1.0, 1.0))
    return SnellResult(theta1_rad=theta1_rad, theta2_rad=theta2, critical_rad=critical)


def _ray_endpoints(theta_rad: float, length: float, medium: str) -> tuple[float, float]:
    """
    Return (x, y) endpoint at given length from interface.

    medium:
      - "incident": endpoint in medium 1 (y>0) on left side
      - "reflected": endpoint in medium 1 (y>0) on right side
      - "refracted": endpoint in medium 2 (y<0) on right side
    """
    sx = math.sin(theta_rad)
    cy = math.cos(theta_rad)
    if medium == "incident":
        return (-length * sx, +length * cy)
    if medium == "reflected":
        return (+length * sx, +length * cy)
    if medium == "refracted":
        return (+length * sx, -length * cy)
    raise ValueError(f"Unknown medium={medium!r}")


def _format_deg(x_rad: float | None) -> str:
    if x_rad is None:
        return "—"
    return f"{math.degrees(x_rad):.2f}°"


def _validate_theta_deg(name: str, theta_deg: float) -> float:
    if not (0.0 <= theta_deg < 90.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 90 degrees (got {theta_deg}).")
    return theta_deg


def build_figure(n1: float, n2: float) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax_ray, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.1]},
        constrained_layout=True,
    )

    # Ray diagram styling (interface at y=0).
    ax_ray.axhspan(0, 2, color="#e8f1ff", alpha=0.9, zorder=0)
    ax_ray.axhspan(-2, 0, color="#fff3e6", alpha=0.9, zorder=0)
    ax_ray.axhline(0, color="k", linewidth=1.2, zorder=2)
    ax_ray.axvline(0, color="k", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax_ray.set_aspect("equal", adjustable="box")
    ax_ray.set_xlim(-2.2, 2.2)
    ax_ray.set_ylim(-2.0, 2.0)
    ax_ray.set_xticks([])
    ax_ray.set_yticks([])
    ax_ray.set_title("Ray diagram (angles from normal)")
    ax_ray.text(-2.1, 1.7, f"medium 1: n₁={n1:g}", ha="left", va="center")
    ax_ray.text(-2.1, -1.7, f"medium 2: n₂={n2:g}", ha="left", va="center")

    # Curve axes: theta2(theta1).
    ax_curve.set_title("Snell map: θ₂(θ₁)")
    ax_curve.set_xlabel("incident angle θ₁ (deg)")
    ax_curve.set_ylabel("refracted angle θ₂ (deg)")
    ax_curve.set_xlim(0, 90)
    ax_curve.set_ylim(0, 90)
    ax_curve.grid(True, alpha=0.25)

    theta1_grid = np.linspace(0.0, math.radians(89.9), 600)
    s2 = (n1 / n2) * np.sin(theta1_grid)
    valid = np.abs(s2) <= 1.0
    theta2_grid = np.empty_like(theta1_grid)
    theta2_grid[:] = np.nan
    theta2_grid[valid] = np.arcsin(s2[valid])

    ax_curve.plot(
        np.degrees(theta1_grid[valid]),
        np.degrees(theta2_grid[valid]),
        color="#2a6fbb",
        linewidth=2.0,
        label="θ₂ = asin((n₁/n₂) sin θ₁)",
    )

    if n1 > n2:
        theta_c = math.degrees(math.asin(n2 / n1))
        ax_curve.axvline(theta_c, color="#aa3377", linestyle="--", linewidth=1.5, label="critical angle")
        ax_curve.axvspan(theta_c, 90, color="#aa3377", alpha=0.08, label="TIR region")

    ax_curve.legend(loc="lower right", frameon=True, framealpha=0.95)
    return fig, ax_ray, ax_curve


def draw_state(
    ax_ray: plt.Axes,
    n1: float,
    n2: float,
    theta1_rad: float,
    *,
    length: float = 1.8,
) -> SnellResult:
    # Keep the first two lines (interface + normal) and first two texts (medium labels).
    for line in list(ax_ray.lines[2:]):
        line.remove()
    for text in list(ax_ray.texts[2:]):
        text.remove()

    res = snell(n1=n1, n2=n2, theta1_rad=theta1_rad)

    # Incident ray.
    xi, yi = _ray_endpoints(res.theta1_rad, length, "incident")
    ax_ray.plot([xi, 0], [yi, 0], color="#1f77b4", linewidth=2.2, label="incident")

    # Reflected ray (always exists).
    xr, yr = _ray_endpoints(res.theta1_rad, length, "reflected")
    ax_ray.plot([0, xr], [0, yr], color="#1f77b4", linewidth=1.7, alpha=0.6, label="reflected")

    # Refracted ray (only if not TIR).
    if res.theta2_rad is not None:
        xt, yt = _ray_endpoints(res.theta2_rad, length, "refracted")
        ax_ray.plot([0, xt], [0, yt], color="#d62728", linewidth=2.2, label="refracted")
        status = "refraction"
    else:
        status = "total internal reflection"

    ax_ray.text(
        -2.1,
        1.35,
        f"θ₁={_format_deg(res.theta1_rad)}   θ₂={_format_deg(res.theta2_rad)}   θc={_format_deg(res.critical_rad)}",
        ha="left",
        va="center",
        fontsize=10,
    )
    ax_ray.text(-2.1, 1.15, status, ha="left", va="center", fontsize=10, alpha=0.85)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize refraction at a planar interface (Snell's law).")
    p.add_argument("--n1", type=float, default=1.0, help="Refractive index in medium 1 (y>0).")
    p.add_argument("--n2", type=float, default=1.5, help="Refractive index in medium 2 (y<0).")
    p.add_argument("--theta1", type=float, default=40.0, help="Incident angle in degrees (from normal).")
    p.add_argument("--outfile", type=str, default="refraction.png", help="Output path (PNG for static, GIF for animation).")
    p.add_argument("--dpi", type=int, default=160, help="Output DPI (static).")
    p.add_argument("--show", action="store_true", help="Show interactively instead of saving.")

    p.add_argument("--animate", action="store_true", help="Animate sweeping θ₁ and save a GIF.")
    p.add_argument("--theta1-min", type=float, default=0.0, help="Animation start θ₁ in degrees.")
    p.add_argument("--theta1-max", type=float, default=80.0, help="Animation end θ₁ in degrees.")
    p.add_argument("--frames", type=int, default=180, help="Number of animation frames.")
    p.add_argument("--fps", type=int, default=30, help="Animation frames per second.")

    args = p.parse_args()

    fig, ax_ray, ax_curve = build_figure(args.n1, args.n2)

    if not args.animate:
        _validate_theta_deg("theta1", args.theta1)
        res = draw_state(ax_ray, args.n1, args.n2, math.radians(args.theta1))
        print(
            "Snell:",
            f"n1={args.n1:g}",
            f"n2={args.n2:g}",
            f"theta1={_format_deg(res.theta1_rad)}",
            f"theta2={_format_deg(res.theta2_rad)}",
            f"critical={_format_deg(res.critical_rad)}",
        )

        if args.show:
            plt.show()
            return

        fig.savefig(args.outfile, dpi=args.dpi)
        print(f"Saved {args.outfile}")
        return

    # Animation.
    _validate_theta_deg("theta1_min", args.theta1_min)
    _validate_theta_deg("theta1_max", args.theta1_max)
    if not (args.theta1_min < args.theta1_max):
        raise ValueError("--theta1-min must be < --theta1-max for --animate.")
    theta_min = math.radians(args.theta1_min)
    theta_max = math.radians(args.theta1_max)
    theta_seq = np.linspace(theta_min, theta_max, args.frames)

    def init() -> tuple:
        draw_state(ax_ray, args.n1, args.n2, float(theta_seq[0]))
        return tuple(ax_ray.lines) + tuple(ax_ray.texts)

    def update(i: int) -> tuple:
        draw_state(ax_ray, args.n1, args.n2, float(theta_seq[i]))
        return tuple(ax_ray.lines) + tuple(ax_ray.texts)

    anim = FuncAnimation(fig, update, init_func=init, frames=args.frames, interval=1000 / args.fps, blit=False)

    if args.show:
        plt.show()
        return

    if not args.outfile.lower().endswith(".gif"):
        raise SystemExit("--animate expects --outfile to end with .gif")

    anim.save(args.outfile, writer=PillowWriter(fps=args.fps))
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()
