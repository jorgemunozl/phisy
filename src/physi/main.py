from physi.double_pendulum.double_pendulum import DoublePendulumSolver


def main():
    solver = DoublePendulumSolver(
        method="rk4",
        preset="hard",
    )
    # solver.plot_e_vs_time(method="rk4", preset="hard")
    # solver.plot_delta_e_vs_h(h_list=[0.001, 0.0001])
    solver.plot_e_vs_time(scale="normal")
    # solver.plot_e_vs_time(scale="normal", method="rk8", preset="hard")
    # solver.plot_e_vs_time(scale="normal")


if __name__ == "__main__":
    main()
