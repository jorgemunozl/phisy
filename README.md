<h1 align="center">Physi</h1>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/numpy-✓-013243?logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/scipy-✓-8CAAE6?logo=scipy" alt="SciPy">
  <img src="https://img.shields.io/badge/matplotlib-✓-11557c?logo=python" alt="Matplotlib">
  <img src="https://img.shields.io/badge/sympy-✓-3B5526?logo=sympy" alt="SymPy">
  <img src="https://img.shields.io/badge/typst-✓-239DAD?logo=typst" alt="Typst">
  <img src="https://img.shields.io/badge/manim-✓-FF6B6B" alt="Manim">
</p>
<p align="center"><i>Computational physics experiments &mdash; vector calculus, complex analysis, dynamical systems, and more.</i></p>

---

## Gallery

<details open>
<summary><b>🧲 Stokes &amp; Divergence Theorems</b></summary>
<br>

**Magnetic dipole field** &mdash; a quiver plot of **B** in the xz-plane, computed from the vector potential and verified against Stokes' theorem.

<p align="center"><img src="images/prob4_B_field.png" width="550" alt="B field quiver plot"></p>

*Purely radial field, strongest at the poles, vanishing at the equator &mdash; classic dipolar morphology.*

<br>

**Wave function cross-sections** &mdash; slices through 3D quantum states reveal nodal surfaces and symmetry patterns.

<p>
  <img src="images/xz_plane_6.png" width="370" alt="Wave function xz-plane slice">
  <img src="images/3_2_0.png" width="370" alt="Wave function isosurface">
</p>

*Left: xz-plane cut of the |n=6⟩ state. Right: 3D isosurface of the |3,2,0⟩ hydrogen orbital.*

</details>

<details open>
<summary><b>🔢 Complex Analysis</b></summary>
<br>

**Cauchy&ndash;Riemann orthogonality** &mdash; level curves of the real and imaginary parts of an analytic function intersect at right angles everywhere.

<p align="center"><img src="images/prob1_contours.png" width="460" alt="Cauchy-Riemann contours"></p>

<br>

**Conformal mapping** &mdash; the Joukowsky transform maps circles to airfoil-like curves.

<p>
  <img src="images/prob2_roots.png" width="270" alt="Roots in Argand plane">
  <img src="images/prob2_mapping.png" width="270" alt="Conformal mapping">
</p>

*Left: sixth roots of -64 in the Argand plane. Right: images of circles under the Joukowsky map.*

</details>

<details open>
<summary><b>🎯 Double Pendulum</b></summary>
<br>

**Chaotic trajectory** &mdash; the path of the second mass never closes, a signature of deterministic chaos.

<p align="center"><img src="images/prob4_trajectory.png" width="420" alt="Double pendulum trajectory"></p>

</details>

<details open>
<summary><b>🌊 Brownian Motion</b></summary>
<br>

**3D random walk** &mdash; 10⁴ steps of a Wiener process in three dimensions.

<p align="center"><img src="images/brownian_motion_3d.gif" width="420" alt="Brownian motion 3D"></p>

</details>

<details open>
<summary><b>🔗 Coupled Oscillators</b></summary>
<br>

**Two masses, three springs** &mdash; normal modes and energy sloshing between the two bodies.

<p align="center"><img src="images/coupled_mas.gif" width="460" alt="Coupled masses"></p>

</details>

---

## Project structure

```
physi/
├── images/                  # README images &amp; GIFs
├── src/physi/
│   ├── brownian/           # Brownian motion (1D, 2D, 3D)
│   ├── coupled_mas/        # Coupled spring-mass system
│   ├── double_pendulum/    # Manim double pendulum animation
│   ├── magnetic_field/     # Magnetic field computation
│   ├── vector_fields/      # Vector field visualization
│   ├── curricular/
│   │   ├── mathematics_methods_lab_01/   # ODEs
│   │   ├── mathematics_methods_lab_03/   # Complex analysis, catenary, pendulum
│   │   └── mm_lab_04/                    # Stokes, divergence, magnetic potential
│   └── ...                 # DFT, KANs, Monte Carlo, optics, splines, wave functions
└── pyproject.toml
```

---

<p align="center">
  <sub>Python &middot; NumPy &middot; SciPy &middot; Matplotlib &middot; SymPy &middot; Manim &middot; Typst</sub>
</p>
