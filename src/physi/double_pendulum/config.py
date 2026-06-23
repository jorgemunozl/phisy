from _typeshed import Self
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

DOUBLE_PENDULUM_PATH = Path(__file__).parent

# Physical parameters for the double pendulum
M1 = 1.0
M2 = 1.0
L1 = 1.0
L2 = 1.0
GRAVITY = 9.81
TOTAL_MASS = M1 + M2


def d2theta1(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = M1 + M2 * np.sin(diff) ** 2
    # θ₁'' = [m₂·g·sin(θ₂)·cos(Δ) - m₂·l₁·ω₁²·sin(Δ)·cos(Δ) - m₂·l₂·ω₂²·sin(Δ) - (m₁+m₂)·g·sin(θ₁)] / [l₁·(m₁ + m₂·sin²(Δ))]
    numerator = (
        M2 * GRAVITY * np.sin(theta_2) * np.cos(diff)
        - M2 * L1 * omega_1**2 * np.sin(diff) * np.cos(diff)
        - M2 * L2 * omega_2**2 * np.sin(diff)
        - TOTAL_MASS * GRAVITY * np.sin(theta_1)
    )
    return numerator / (alpha * L1)


def d2theta2(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = M1 + M2 * np.sin(diff) ** 2
    first_term = np.sin(diff) * (
        TOTAL_MASS * L1 * omega_1**2 + M2 * L2 * omega_2**2 * np.cos(diff)
    )
    second_term = GRAVITY * (
        TOTAL_MASS * np.sin(theta_1) * np.cos(diff) - TOTAL_MASS * np.sin(theta_2)
    )
    return (first_term + second_term) / (alpha * L2)


def rk4_step(f, t, u, h):
    k1 = h * f(t, u, d2theta1, d2theta2)
    k2 = h * f(t + h / 2, u + k1 / 2, d2theta1, d2theta2)
    k3 = h * f(t + h / 2, u + k2 / 2, d2theta1, d2theta2)
    k4 = h * f(t + h, u + k3, d2theta1, d2theta2)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Theta update
def F(t, u, second_theta_1, second_theta_2):
    # u = [theta_1, omega_1, theta_2, omega_2]
    theta_1, omega_1, theta_2, omega_2 = u

    d_omega1 = second_theta_1(theta_1, theta_2, omega_1, omega_2)
    d_omega2 = second_theta_2(theta_1, theta_2, omega_1, omega_2)

    return np.array([omega_1, d_omega1, omega_2, d_omega2])


def calc_kinetic_energy(m1, l1, omega_1, m2, l2, omega_2, diff) -> np.ndarray:
    """
    KE of the double pendulum.
    KE = 1/2 m1 l1^2 w1^2  +  1/2 m2 (l1^2 w1^2 + l2^2 w2^2 + 2 l1 l2 w1 w2 cos(dθ))
    """
    return 0.5 * m1 * l1**2 * omega_1**2 + 0.5 * m2 * (
        l1**2 * omega_1**2
        + l2**2 * omega_2**2
        + 2 * l1 * l2 * omega_1 * omega_2 * np.cos(diff)
    )


def calc_potential_energy(m1, l1, m2, l2, theta_1, theta_2) -> np.ndarray:
    """PE = -(m1+m2) g l1 cos theta1  -  m2 g l2 cos theta2"""
    return -(m1 + m2) * GRAVITY * l1 * np.cos(theta_1) - m2 * GRAVITY * l2 * np.cos(
        theta_2
    )


def rhs(t, u):
    """Right-hand side with the signature required by solve_ivp."""
    return F(t, u, d2theta1, d2theta2)


@dataclass
class DoublePendulumSolver:
    """
    present to dont change every time
    """

    time: float = field(
        default=10.0,
        metadata={"description": "Total simulation time in seconds"},
    )
    preset: str = field(
        default="hard",
        metadata={"description": "Preset to use for simulation"},
    )
    steps: int = field(
        default=int(10e2),
        metadata={"description": "Number of simulation steps"},
    )
    theta_1: float = field(
        default=np.pi / 2,
        metadata={"description": "Initial angle of the first pendulum"},
    )
    theta_2: float = field(
        default=np.pi / 2,
        metadata={"description": "Initial angle of the second pendulum"},
    )
    omega_1: float = field(
        default=1.0,
        metadata={"description": "Initial angular velocity of the first pendulum"},
    )
    omega_2: float = field(
        default=-1.0,
        metadata={"description": "Initial angular velocity of the second pendulum"},
    )
    time_str: str = field(
        default="10s",
        metadata={"description": "Total simulation time as a string"},
    )
    steps_str: str = field(
        default="10e2",
        metadata={"description": "Number of simulation steps as a string"},
    )
    path_numpy: Path = field(
        default=Path(""),
        metadata={"description": "Path to save the simulation results as a NumPy file"},
    )
    path_animation: Path = field(
        default=Path(""),
        metadata={"description": "Path to save the simulation results as a plot"},
    )
    method: Literal["rk4", "rk8"] = field(
        default="rk4",
        metadata={"description": "Solver method to use"},
    )

    def __post_init__(self):
        if self.preset:
            self.set_preset(self.preset)
        self._build_paths()

    def _build_paths(self):
        # map pi to symbol
        pi = "pi_2"
        prefix = f"{self.time_str}_{self.steps_str}_{pi}_{pi}_{self.omega_1}_{self.omega_2}_{self.method}"
        self.path_numpy = DOUBLE_PENDULUM_PATH / f"data/{prefix}.npy"
        self.path_animation = DOUBLE_PENDULUM_PATH / f"data/{prefix}.png"

    def solve(self):
        if self.preset:
            self.set_preset(self.preset)
            self._build_paths()

        if self.path_numpy.exists():
            print("Solution already exists!!")
        else:
            if self.method == "rk4":
                self.solve_rk4()
            elif self.method == "rk8":
                self.solve_rk8()
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def set_preset(self, preset: str):
        if preset == "testing":
            self.time = 10
            self.steps = 1000
            self.steps_str = "10e3"
            self.time_str = "10s"
        elif preset == "hard":
            self.time = 30
            self.steps = 100000
            self.steps_str = "10e5"
            self.time_str = "30s"
        elif preset == "extreme":
            self.time = 100
            self.steps = 1000000
            self.steps_str = "10e6"
            self.time_str = "100s"
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def build_times(self):
        return (0, self.time), np.linspace(0, self.time, self.steps)

    def build_initial_conditions(self):
        return np.array([self.theta_1, self.omega_1, self.theta_2, self.omega_2])

    def save_solution(self, u):
        np.save(self.path_numpy, u)

    def solve_rk4(self):
        u_0 = self.build_initial_conditions()
        dt = self.time / self.steps
        u = np.zeros((self.steps + 1, 4))
        u[0] = u_0
        for i in range(self.steps):
            u[i + 1] = rk4_step(F, i * dt, u[i], dt)
        # u = [theta_1, omega_1, theta_2, omega_2]
        self.save_solution(u)
        print(f"Saved solution to {self.path_numpy}")

    def solve_rk8(self):
        from scipy.integrate import solve_ivp

        u_0 = self.build_initial_conditions()
        d_span, t_eval = self.build_times()
        sol = solve_ivp(
            rhs,
            t_span,
            u_0,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )
        self.save_solution(sol.y.transpose())
        print(f"Saved solution to {self.path_numpy}")

    def get_energy(self, u: np.ndarray) -> np.ndarray:
        # u = [theta_1, omega_1, theta_2, omega_2]
        return calc_kinetic_energy(M1, L1, u[1], M2, L2, u[3], u[0] - u[2]) + calc_potential_energy(M1, L1, M2, L2, u[0], u[2])

    def get_delta_E():
        u = np.load(self.path_numpy)
        u_0, u_f = u[0], u[-1]
        return self.get_energy(u_f) - self.get_energy(u_0)

    def plot_difference(self, u_self: np.ndarray, u_odd: np.ndarray):
        """
        Delta E against H
        """
        import matplotlib.pyplot as plt

        for i in range(10):




    def compare_solutions(self, method: Literal["rk4", "rk8"], preset: str):
        solver_odd = DoublePendulumSolver(method=method, preset=preset)
        if not self.path_numpy.exists():
            self.solve()
        if not solver_odd.path_numpy.exists():
            solver_odd.solve()
        u_self = np.load(self.path_numpy)
        u_odd = np.load(solver_odd.path_numpy)

        diffs = np.zeros(u_self.shape)
        print("Resting", self.method, "against", solver_odd.method)
        for i in range(u_self.shape[0]):
            diff = u_self[i] - u_odd[i]
            diffs[i] = diff

        u_0_self = u_self[0]
        u_0_odd = u_odd[0]

        u_f_self = u_self[-1]
        u_f_odd = u_odd[-1]

        e_0_self = self.kinetic_energy(u_0_self) + self.potential_energy(u_0_self)
        print(f"Energy at t=0 (self): {e_0_self}")
        e_0_odd = solver_odd.kinetic_energy(u_0_odd) + solver_odd.potential_energy(
            u_0_odd
        )
        print(f"Energy at t=0 (odd): {e_0_odd}")
        e_0_diff = e_0_self - e_0_odd
        print(f"Energy difference at t=0: {e_0_diff}")

        e_f_self = self.kinetic_energy(u_f_self) + self.potential_energy(u_f_self)
        print(f"Energy at t=final (self): {e_f_self}")
        e_f_odd = solver_odd.kinetic_energy(u_f_odd) + solver_odd.potential_energy(
            u_f_odd
        )
        print(f"Energy at t=final (odd): {e_f_odd}")
        e_self_f_diff = e_f_self - e_0_self
        e_odd_f_diff = e_f_odd - e_0_odd

        print(f"Energy difference at t=final (self): {e_self_f_diff}")
        print(f"Energy difference at t=final (odd): {e_odd_f_diff}")

        mse = np.mean(diffs**2)
        deviation = np.sqrt(mse)
        mean = np.mean(diffs)
        print(f"MSE: {mse}, deviation: {deviation}, mean: {mean}")
