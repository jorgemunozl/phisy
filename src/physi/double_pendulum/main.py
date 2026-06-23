from physi.double_pendulum.config import DoublePendulumSolver


def main():
    solver_rk8 = DoublePendulumSolver(
        method="rk8",
        preset="hard",
    )
    solver_rk8.compare_solutions("rk4", "hard")


if __name__ == "__main__":
    main()
