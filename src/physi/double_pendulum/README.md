# Double Pendulum

Numerical simulation of the chaotic double pendulum with multiple integration methods, energy analysis, and Manim visualizations.

## Files

| File | Description |
|------|-------------|
| `double_pendulum.py` | Core simulation — solvers, energy analysis, plotting |
| `visual.py` | Manim animations and visualizations |
| `print_data.py` | Inspect saved `.npy` files |
| `test.py` | Scratchpad / sanity checks |
| `main.py` (parent dir) | Example entry point |

## Gallery

<div align="center">

### Three-way comparison

![Three-way comparison](media/images/DoublePendulumThreeWay.png)

Three initial conditions side by side — chaos in action.

### Energy visualization

![Energy](media/images/DoublePendulumEnergy.png)

Kinetic, potential, and total energy over time.

### Side-by-side replay

![Side-by-side](media/images/DoublePendulumSideBySide.png)

Two solvers running the same initial condition for comparison.

### Overlay comparison

![Overlay](media/images/DoublePendulumOverlay.png)

Direct trajectory overlay of different solvers.

</div>

## Quick Start

```python
from physi.double_pendulum.double_pendulum import DoublePendulumSolver

solver = DoublePendulumSolver(method="rk8", preset="hard")
solution = solver.solve()       # shape (steps+1, 4): [θ₁, ω₁, θ₂, ω₂]
```

### Presets

| Preset      | Time | Steps     | Use                 |
|-------------|------|-----------|---------------------|
| `"testing"` | 10s  | 1 000     | Quick validation    |
| `"hard"`    | 40s  | 100 000   | Standard simulation |
| `"extreme"` | 100s | 10 000 000| High precision      |

### Integration Methods

| `method` field | Solver | Description |
|----------------|--------|-------------|
| `"rk4"` | Classical RK4 | Fixed-step 4th-order Runge–Kutta |
| `"rk8"` | DOP853 | High-order adaptive solver via SciPy |
| `"rk4_adaptive"` | RKF45 | Embedded 4(5) pair with automatic step-size control |

### Forcing re-computation

`solve()` returns cached results when the `.npy` file already exists. Recompute with:

```python
solution = solver.solve(dt=0.01, force=True)
```

### Energy analysis

```python
energy = solver.get_energy(solution)           # total energy at each step
delta_E = solver.get_delta_E()                 # energy drift (final - initial)
```

### Plotting

```python
# Energy vs time for one or more solvers
solver.plot_e_vs_time(method="rk4", preset="hard")

# Energy vs time for multiple step sizes
solver.plot_delta_e_vs_h(h_list=[1.0, 0.1, 0.01])

# Compare two solvers head-to-head
solver.compare_solutions(method="rk4", preset="hard")
```

### Manim animations

```bash
manim -pql visual.py DoublePendulumEnergy
manim -pql visual.py DoublePendulumSideBySide
manim -pql visual.py DoublePendulumThreeWay
manim -pql visual.py DoublePendulumOverlay
```

## Solver comparison

```python
from physi.double_pendulum.double_pendulum import DoublePendulumSolver

ref = DoublePendulumSolver(method="rk8", preset="hard")
approx = DoublePendulumSolver(method="rk4", preset="hard")

ref.solve()
approx.solve()
ref.compare_solutions("rk4", "hard")
```

## Data format

Solutions are saved as `.npy` files in `data/`. The filename encodes the parameters:

```
{time}s_{steps}_{theta1}_{theta2}_{omega1}_{omega2}_{method}.npy
```

Example: `40s_10e5_pi_2_pi_2_1.0_-1.0_rk8.npy`

Each array has shape `(N, 4)` with columns `[θ₁, ω₁, θ₂, ω₂]`.
