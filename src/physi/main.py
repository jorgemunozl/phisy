from physi.double_pendulum.double_pendulum import DoublePendulumSolver


def main():
    # ── Convergence study: |ΔE| vs h for RK8 ──────────────────────
    # Uses "testing" preset (T = 5 s) to keep run-times manageable.
    # h_list = [0.01, 0.005, 0.002, 0.001, 0.0005]
    solver = DoublePendulumSolver(
        method="rk4",
        preset="testing",  # T = 5 s, h = 0.01
    )
    solver.plot_delta_e_vs_h(
        h_list=[0.01, 0.005, 0.002, 0.001],
    )


if __name__ == "__main__":
    main()
