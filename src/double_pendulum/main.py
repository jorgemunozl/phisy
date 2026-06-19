from subprocess import run

from constants import DoublePendulumSolver


def main():
    solver = DoublePendulumSolver(
        method="rk4",
        preset="hard",
    )
    solver.solve()


if __name__ == "__main__":
    main()
