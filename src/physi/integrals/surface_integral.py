from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Vec3 = NDArray[np.floating]
ScalarField = Callable[[float, float, float], float]
VectorField = Callable[[float, float, float], tuple[float, float, float] | Vec3]
SurfaceMap = Callable[[float, float], tuple[float, float, float] | Vec3]
SurfaceTangent = Callable[[float, float], tuple[float, float, float] | Vec3]


def _vec3(x: tuple[float, float, float] | Vec3) -> Vec3:
    a = np.asarray(x, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {a.shape}")
    return a


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return np.cross(a, b)


def _finite_difference_tangents(
    r: SurfaceMap,
    u: float,
    v: float,
    *,
    du: float,
    dv: float,
) -> tuple[Vec3, Vec3]:
    ru = (_vec3(r(u + du, v)) - _vec3(r(u - du, v))) / (2.0 * du)
    rv = (_vec3(r(u, v + dv)) - _vec3(r(u, v - dv))) / (2.0 * dv)
    return ru, rv


@dataclass(frozen=True)
class Surface:
    """
    Parametric surface r(u,v) with optional tangents.

    If `ru`/`rv` are not provided, they are approximated by central differences.
    """

    r: SurfaceMap
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    ru: SurfaceTangent | None = None
    rv: SurfaceTangent | None = None
    fd_step: float = 1e-6

    def tangents(self, u: float, v: float) -> tuple[Vec3, Vec3]:
        if self.ru is not None and self.rv is not None:
            return _vec3(self.ru(u, v)), _vec3(self.rv(u, v))
        h = float(self.fd_step)
        return _finite_difference_tangents(self.r, u, v, du=h, dv=h)

    def area_element(self, u: float, v: float) -> float:
        ru, rv = self.tangents(u, v)
        return float(np.linalg.norm(_cross(ru, rv)))

    def normal_times_area(self, u: float, v: float) -> Vec3:
        ru, rv = self.tangents(u, v)
        return _cross(ru, rv)


def surface_integral_scalar(
    surface: Surface,
    f: ScalarField,
    *,
    method: Literal["quad", "grid"] = "quad",
    grid_n: int = 300,
    epsabs: float = 1e-9,
    epsrel: float = 1e-9,
) -> float:
    """
    Compute ∬_S f(x,y,z) dS over a parametric surface.
    """

    (u0, u1) = surface.u_range
    (v0, v1) = surface.v_range

    if method == "grid":
        return _surface_integral_scalar_grid(surface, f, grid_n=grid_n)
    if method != "quad":
        raise ValueError(f"Unknown method: {method!r}")

    # Prefer scipy when available (project dependency), but keep import local.
    from scipy import integrate  # type: ignore

    def integrand(v: float, u: float) -> float:
        x, y, z = _vec3(surface.r(u, v))
        return float(f(float(x), float(y), float(z)) * surface.area_element(u, v))

    val, _err = integrate.dblquad(
        integrand,
        u0,
        u1,
        lambda _u: v0,
        lambda _u: v1,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return float(val)


def surface_integral_flux(
    surface: Surface,
    F: VectorField,
    *,
    method: Literal["quad", "grid"] = "quad",
    grid_n: int = 300,
    epsabs: float = 1e-9,
    epsrel: float = 1e-9,
    orientation: Literal["+","-"] = "+",
) -> float:
    """
    Compute flux ∬_S F · n dS.

    For parametric r(u,v), this equals ∬ F(r(u,v)) · (r_u × r_v) dudv.
    `orientation="-"` flips the sign (useful if your parametrization is inward).
    """

    sign = 1.0 if orientation == "+" else -1.0
    (u0, u1) = surface.u_range
    (v0, v1) = surface.v_range

    if method == "grid":
        return sign * _surface_integral_flux_grid(surface, F, grid_n=grid_n)
    if method != "quad":
        raise ValueError(f"Unknown method: {method!r}")

    from scipy import integrate  # type: ignore

    def integrand(v: float, u: float) -> float:
        x, y, z = _vec3(surface.r(u, v))
        Fx, Fy, Fz = _vec3(F(float(x), float(y), float(z)))
        nA = surface.normal_times_area(u, v)
        return float(sign * (Fx * nA[0] + Fy * nA[1] + Fz * nA[2]))

    val, _err = integrate.dblquad(
        integrand,
        u0,
        u1,
        lambda _u: v0,
        lambda _u: v1,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return float(val)


def _surface_integral_scalar_grid(surface: Surface, f: ScalarField, *, grid_n: int) -> float:
    (u0, u1) = surface.u_range
    (v0, v1) = surface.v_range
    u = np.linspace(u0, u1, grid_n, dtype=float)
    v = np.linspace(v0, v1, grid_n, dtype=float)

    # Evaluate on grid (robust for non-vectorized user functions).
    integrand = np.empty((grid_n, grid_n), dtype=float)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            x, y, z = _vec3(surface.r(float(ui), float(vj)))
            integrand[i, j] = float(f(float(x), float(y), float(z)) * surface.area_element(float(ui), float(vj)))

    tmp = np.trapezoid(integrand, v, axis=1)
    val = np.trapezoid(tmp, u, axis=0)
    return float(val)


def _surface_integral_flux_grid(surface: Surface, F: VectorField, *, grid_n: int) -> float:
    (u0, u1) = surface.u_range
    (v0, v1) = surface.v_range
    u = np.linspace(u0, u1, grid_n, dtype=float)
    v = np.linspace(v0, v1, grid_n, dtype=float)

    integrand = np.empty((grid_n, grid_n), dtype=float)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            x, y, z = _vec3(surface.r(float(ui), float(vj)))
            Fx, Fy, Fz = _vec3(F(float(x), float(y), float(z)))
            nA = surface.normal_times_area(float(ui), float(vj))
            integrand[i, j] = float(Fx * nA[0] + Fy * nA[1] + Fz * nA[2])

    tmp = np.trapezoid(integrand, v, axis=1)
    val = np.trapezoid(tmp, u, axis=0)
    return float(val)


def _demo(method: Literal["quad", "grid"], grid_n: int) -> None:
    # Unit sphere: u=theta in [0,pi], v=phi in [0,2pi]
    def r(theta: float, phi: float) -> Vec3:
        return np.array(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ],
            dtype=float,
        )

    def ru(theta: float, phi: float) -> Vec3:
        return np.array(
            [
                math.cos(theta) * math.cos(phi),
                math.cos(theta) * math.sin(phi),
                -math.sin(theta),
            ],
            dtype=float,
        )

    def rv(theta: float, phi: float) -> Vec3:
        return np.array(
            [
                -math.sin(theta) * math.sin(phi),
                math.sin(theta) * math.cos(phi),
                0.0,
            ],
            dtype=float,
        )

    S = Surface(r=r, ru=ru, rv=rv, u_range=(0.0, math.pi), v_range=(0.0, 2.0 * math.pi))

    area = surface_integral_scalar(S, lambda x, y, z: 1.0, method=method, grid_n=grid_n)
    x2 = surface_integral_scalar(S, lambda x, y, z: x * x, method=method, grid_n=grid_n)

    flux = surface_integral_flux(
        S,
        lambda x, y, z: (x, y, z),
        method=method,
        grid_n=grid_n,
        orientation="+",
    )

    print("Unit sphere checks")
    print(f"  area      ≈ {area:.12g}   (exact 4π = {4*math.pi:.12g})")
    print(f"  ∬ x^2 dS  ≈ {x2:.12g}     (exact 4π/3 = {4*math.pi/3:.12g})")
    print(f"  flux(x,y,z) ≈ {flux:.12g} (exact 4π = {4*math.pi:.12g})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Numerical surface integrals over parametric surfaces.")
    p.add_argument("--demo", action="store_true", help="Run built-in sanity checks on the unit sphere.")
    p.add_argument("--method", choices=["quad", "grid"], default="quad", help="Integration backend.")
    p.add_argument("--grid-n", type=int, default=400, help="Grid resolution if method=grid.")
    args = p.parse_args(argv)

    if args.demo:
        _demo(method=args.method, grid_n=args.grid_n)
        return 0

    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
