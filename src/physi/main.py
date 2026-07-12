from physi.double_pendulum.double_pendulum import DoublePendulumSolver


def precision():
    # ── Convergence study: |ΔE| vs h for RK8 ──────────────────────
    # float64 first
    h_list = [0.01, 0.0075, 0.005, 0.0035, 0.002, 0.001]
    print("=" * 60)
    print("FLOAT64 convergence")
    print("=" * 60)
    solver_f64 = DoublePendulumSolver(
        method="rk8",
        preset="testing",  # T = 5 s, h = 0.01
        use_longdouble=False,
    )
    solver_f64.plot_delta_e_vs_h(
        h_list=h_list,
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
        h_list=[0.01, 0.005, 0.002, 0.001],
    )


def plots():
    solver = DoublePendulumSolver(
        method="rk8",
        preset="hard",
    )
    solver.plot_e_vs_time()


if __name__ == "__main__":
    # plots()
    precision()
