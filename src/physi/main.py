import matplotlib.pyplot as plt

from physi.double_pendulum.double_pendulum import (
    SLIDES_PATH,
    DoublePendulumSolver,
)


def precision():
    # ── Convergence study: |ΔE| vs h for RK8 ──────────────────────
    h_list = [
        0.1,
        0.05,
        0.025,
        0.01,
        0.0075,
        0.005,
        0.0035,
        0.002,
        0.001,
        0.0005,
        0.00025,
        0.00001,
        0.000001,
    ]

    # Create a single figure for both precisions
    fig, ax = plt.subplots(figsize=(8, 6))

    print("=" * 60)
    print("FLOAT64 convergence")
    print("=" * 60)
    solver_f64 = DoublePendulumSolver(
        method="rk8",
        preset="testing",
        use_longdouble=False,
    )
    solver_f64.plot_delta_e_vs_h(
        h_list=h_list,
        ax=ax,
        label="float64  |ΔE|",
    )

    # float128 — pushes the roundoff floor ~3 orders of magnitude lower
    print("\n" + "=" * 60)
    print("FLOAT128 (longdouble) convergence")
    print("=" * 60)
    solver_f128 = DoublePendulumSolver(
        method="rk8",
        preset="testing",
        use_longdouble=True,
    )
    solver_f128.plot_delta_e_vs_h(
        h_list=h_list,
        ax=ax,
        label="float128  |ΔE|",
    )

    ax.invert_xaxis()  # smaller h (more accurate) → right
    ax.legend(fontsize=16, loc="lower left")
    fig.tight_layout()
    path = SLIDES_PATH / "rk8_log_log_convergence_combined.pdf"
    fig.savefig(path, format="pdf")
    print(f"Saved combined convergence plot → {path}")


def plots():
    solver = DoublePendulumSolver(
        method="rk8",
        preset="hard",
    )
    solver.plot_e_vs_time()


def energy_drift():
    solver = DoublePendulumSolver(
        method="rk4",
        preset="hard",
    )
    solver.plot_e_vs_time()


if __name__ == "__main__":
    # plots()
    precision()
    # energy_drift()
