from physi.double_pendulum.double_pendulum import DoublePendulumSolver


def main():
    solver = DoublePendulumSolver(
        method="rk4",
        preset="testing",
    )
    # solver.plot_e_vs_time(method="rk4", preset="hard")
    # solver.compare_solutions(method="rk8", preset="hard")
    # solver.plot_e_vs_time(method="rk8", preset="hard")
    # solver.plot_delta_e_vs_h(h_num=range(3, 6))
    # solver.plot_e_vs_time(scale="normal")
    # solver.plot_e_vs_time(scale="normal", method="rk8", preset="hard")
    # solver.plot_e_vs_time(scale="normal")


if __name__ == "__main__":
    main()
