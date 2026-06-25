from physi.double_pendulum.config import DoublePendulumSolver


def main():
    solver = DoublePendulumSolver(
        method="rk4",
        preset="hard",
    )
    # solver.compare_solutions("rk8", "hard")
    solver.plot_e_vs_time(method="rk8", preset="hard")
    # solver_rk8.compare_solutions("rk4", "hard")


if __name__ == "__main__":
    main()
