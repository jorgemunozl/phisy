import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
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


# --- DOP853 Butcher tableau (Hairer, Nørsett, Wanner 1993) ---
# 12-stage explicit RK method, order 8 (FSAL: 13 effective stages).
# Coefficients from scipy.integrate._ivp.dop853_coefficients.

_RK8_C = np.array(
    [
        0.0,
        0.0526001519587677318785587544488,
        0.0789002279381515978178381316732,
        0.118350341907227396726757197510,
        0.281649658092772603273242802490,
        0.333333333333333333333333333333,
        0.25,
        0.307692307692307692307692307692,
        0.651282051282051282051282051282,
        0.6,
        0.857142857142857142857142857142,
        1.0,
    ]
)

_RK8_A = np.zeros((12, 12))

_RK8_A[1, 0] = 5.26001519587677318785587544488e-2

_RK8_A[2, 0] = 1.97250569845378994544595329183e-2
_RK8_A[2, 1] = 5.91751709536136983633785987549e-2

_RK8_A[3, 0] = 2.95875854768068491816892993775e-2
_RK8_A[3, 2] = 8.87627564304205475450678981324e-2

_RK8_A[4, 0] = 2.41365134159266685502369798665e-1
_RK8_A[4, 2] = -8.84549479328286085344864962717e-1
_RK8_A[4, 3] = 9.24834003261792003115737966543e-1

_RK8_A[5, 0] = 3.7037037037037037037037037037e-2
_RK8_A[5, 3] = 1.70828608729473871279604482173e-1
_RK8_A[5, 4] = 1.25467687566822425016691814123e-1

_RK8_A[6, 0] = 3.7109375e-2
_RK8_A[6, 3] = 1.70252211019544039314978060272e-1
_RK8_A[6, 4] = 6.02165389804559606850219397283e-2
_RK8_A[6, 5] = -1.7578125e-2

_RK8_A[7, 0] = 3.70920001185047927108779319836e-2
_RK8_A[7, 3] = 1.70383925712239993810214054705e-1
_RK8_A[7, 4] = 1.07262030446373284651809199168e-1
_RK8_A[7, 5] = -1.53194377486244017527936158236e-2
_RK8_A[7, 6] = 8.27378916381402288758473766002e-3

_RK8_A[8, 0] = 6.24110958716075717114429577812e-1
_RK8_A[8, 3] = -3.36089262944694129406857109825
_RK8_A[8, 4] = -8.68219346841726006818189891453e-1
_RK8_A[8, 5] = 2.75920996994467083049415600797e1
_RK8_A[8, 6] = 2.01540675504778934086186788979e1
_RK8_A[8, 7] = -4.34898841810699588477366255144e1

_RK8_A[9, 0] = 4.77662536438264365890433908527e-1
_RK8_A[9, 3] = -2.48811461997166764192642586468
_RK8_A[9, 4] = -5.90290826836842996371446475743e-1
_RK8_A[9, 5] = 2.12300514481811942347288949897e1
_RK8_A[9, 6] = 1.52792336328824235832596922938e1
_RK8_A[9, 7] = -3.32882109689848629194453265587e1
_RK8_A[9, 8] = -2.03312017085086261358222928593e-2

_RK8_A[10, 0] = -9.3714243008598732571704021658e-1
_RK8_A[10, 3] = 5.18637242884406370830023853209
_RK8_A[10, 4] = 1.09143734899672957818500254654
_RK8_A[10, 5] = -8.14978701074692612513997267357
_RK8_A[10, 6] = -1.85200656599969598641566180701e1
_RK8_A[10, 7] = 2.27394870993505042818970056734e1
_RK8_A[10, 8] = 2.49360555267965238987089396762
_RK8_A[10, 9] = -3.0467644718982195003823669022

_RK8_A[11, 0] = 2.27331014751653820792359768449
_RK8_A[11, 3] = -1.05344954667372501984066689879e1
_RK8_A[11, 4] = -2.00087205822486249909675718444
_RK8_A[11, 5] = -1.79589318631187989172765950534e1
_RK8_A[11, 6] = 2.79488845294199600508499808837e1
_RK8_A[11, 7] = -2.85899827713502369474065508674
_RK8_A[11, 8] = -8.87285693353062954433549289258
_RK8_A[11, 9] = 1.23605671757943030647266201528e1
_RK8_A[11, 10] = 6.43392746015763530355970484046e-1

# 8th-order weights (FSAL property: B = full-A[12, :12])
_RK8_B = np.array(
    [
        5.42937341165687622380535766363e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        4.45031289275240888144113950566,
        1.89151789931450038304281599044,
        -5.8012039600105847814672114227,
        3.1116436695781989440891606237e-1,
        -1.52160949662516078556178806805e-1,
        2.01365400804030348374776537501e-1,
        4.47106157277725905176885569043e-2,
    ]
)


def rk8_step(f, t, u, h):
    """Fixed-step DOP853 (8th-order Runge-Kutta) — 12 stages."""
    n = len(u)
    k = np.zeros((12, n))

    k[0] = h * f(t, u, d2theta1, d2theta2)

    for i in range(1, 12):
        u_i = u + np.dot(_RK8_A[i, :i], k[:i])
        k[i] = h * f(t + _RK8_C[i] * h, u_i, d2theta1, d2theta2)

    return u + np.dot(_RK8_B, k)


def adaptive_rk4_step(f, t, u, h):
    a2, a3, a4, a5, a6 = 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2

    b21 = 1 / 4
    b31, b32 = 3 / 32, 9 / 32
    b41, b42, b43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
    b51, b52, b53, b54 = 439 / 216, -8.0, 3680 / 513, -845 / 4104
    b61, b62, b63, b64, b65 = -8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40

    c1, c3, c4, c5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5
    d1, d3, d4, d5, d6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55

    k1 = h * f(t, u, d2theta1, d2theta2)
    k2 = h * f(t + a2 * h, u + b21 * k1, d2theta1, d2theta2)
    k3 = h * f(t + a3 * h, u + b31 * k1 + b32 * k2, d2theta1, d2theta2)
    k4 = h * f(t + a4 * h, u + b41 * k1 + b42 * k2 + b43 * k3, d2theta1, d2theta2)
    k5 = h * f(
        t + a5 * h,
        u + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4,
        d2theta1,
        d2theta2,
    )
    k6 = h * f(
        t + a6 * h,
        u + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5,
        d2theta1,
        d2theta2,
    )

    # 4th-order approximation (used for error estimation)
    u4 = u + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    # 5th-order approximation (accepted solution)
    u5 = u + d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6

    error = np.abs(u5 - u4)
    return u5, error


# Vector State for the double pendulum
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
    preset: Literal["testing", "hard", "extreme"] = field(
        default="hard",
        metadata={"description": "Preset to use for simulation"},
    )
    steps: int = field(
        default=int(10e2),
        metadata={"description": "Number of simulation steps"},
    )
    h: float = field(
        default=0.01,
        metadata={"description": "Step size"},
    )
    theta_1: float = field(
        default=np.pi / 2,
        metadata={"description": "Initial angle of the first pendulum"},
    )
    theta_1_str: str = field(
        default="pi/2",
        metadata={"description": "Initial angle of the first pendulum as a string"},
    )
    theta_2: float = field(
        default=np.pi / 6,
        metadata={"description": "Initial angle of the second pendulum"},
    )
    theta_2_str: str = field(
        default="pi/6",
        metadata={"description": "Initial angle of the second pendulum as a string"},
    )
    omega_1: float = field(
        default=1.0,
        metadata={"description": "Initial angular velocity of the first pendulum"},
    )
    omega_2: float = field(
        default=0.0,
        metadata={"description": "Initial angular velocity of the second pendulum"},
    )
    time_str: str = field(
        default="10s",
        metadata={"description": "Total simulation time as a string"},
    )
    h_str: str = field(
        default="1e-2",
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
    method: Literal["rk4", "rk8", "rk4_adaptive"] = field(
        default="rk4",
        metadata={"description": "Solver method to use"},
    )
    # — internal, set by solver methods —
    _feval_count: int | None = field(
        default=None,
        init=False,
        metadata={"description": "Number of RHS evaluations in the last solve"},
    )

    def __post_init__(self):
        if self.preset:
            self.set_preset(self.preset)

        os.makedirs(DOUBLE_PENDULUM_PATH / "plots", exist_ok=True)
        self._build_paths()
        self.solve()

    def _build_paths(self):
        # map pi to symbol
        pi_2 = "pi_2" if self.theta_1 == np.pi / 2 else "pi_1"
        pi_6 = "pi_6" if self.theta_2 == np.pi / 6 else "pi_1"
        prefix = f"{self.time_str}_{self.h_str}_{pi_2}_{pi_6}_{self.omega_1}_{self.omega_2}_{self.method}"
        self.path_numpy = DOUBLE_PENDULUM_PATH / f"data/{prefix}.npy"
        self.path_animation = DOUBLE_PENDULUM_PATH / f"data/{prefix}.png"

    def solve(self, force=False):
        if self.preset:
            self.set_preset(self.preset)
            self._build_paths()

        if force or not self.path_numpy.exists():
            print("Computing solution on", self.path_numpy)
            if self.method == "rk4":
                self.solve_rk4()
            elif self.method == "rk8":
                self.solve_rk8()
            elif self.method == "rk4_adaptive":
                self.solve_adaptive_rk4()
            else:
                raise ValueError(f"Unknown method: {self.method}")
        elif self.path_numpy.exists():
            print("Solution already exists!!", self.path_numpy)
        return np.load(self.path_numpy)

    def set_preset(self, preset: str):
        if preset == "testing":
            self.time = 5
            self.h = 0.01
            self.h_str = "10e2"
            self.time_str = "5s"

        elif preset == "hard":
            self.time = 50
            self.h = 0.001
            self.h_str = "10e3"
            self.time_str = "50s"
        elif preset == "extreme":
            self.time = 60
            self.h = 0.00001
            self.h_str = "10e5"
            self.time_str = "60s"
        else:
            raise ValueError(f"Unknown preset: {preset}")
        self.steps = int(self.time / self.h)

    def build_times(self):
        # t_span, t_eval
        return (0, self.time), np.linspace(0, self.time, self.steps)

    def build_initial_conditions(self):
        return np.array([self.theta_1, self.omega_1, self.theta_2, self.omega_2])

    def save_solution(self, u):
        np.save(self.path_numpy, u)

    def solve_rk4(self):
        num_steps = self.steps
        u_0 = self.build_initial_conditions()
        u = np.zeros((num_steps, 4))
        u[0] = u_0
        for i in range(num_steps - 1):
            u[i + 1] = rk4_step(F, i * self.h, u[i], self.h)
        # u = [theta_1, omega_1, theta_2, omega_2]
        self._feval_count = 4 * self.steps  # 4 stages per RK4 step
        self.save_solution(u)
        print(f"Saved solution to {self.path_numpy}")
        print(f"Saved solution to {self.path_numpy}")

    def solve_rk8(self):
        num_steps = self.steps
        u_0 = self.build_initial_conditions()
        u = np.zeros((num_steps, 4))
        u[0] = u_0
        for i in range(num_steps - 1):
            u[i + 1] = rk8_step(F, i * self.h, u[i], self.h)
        # u = [theta_1, omega_1, theta_2, omega_2]
        self._feval_count = 12 * self.steps  # 12 stages per RK8 step
        self.save_solution(u)
        print(f"Saved solution to {self.path_numpy}")

    def solve_adaptive_rk4(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        h0: float | None = None,
        h_min: float = 1e-12,
        h_max: float | None = None,
        max_steps: int = 1_000_000,
    ):
        """
        Solve the double pendulum using adaptive RK4 (RKF45).

        The step size is automatically adjusted so that the estimated local
        error stays within the prescribed tolerances.  At the end the
        solution is interpolated back onto the uniform time grid
        ``np.linspace(0, self.time, self.steps + 1)`` for full compatibility
        with the rest of the class.

        Parameters
        ----------
        rtol : float
            Relative tolerance for error control.
        atol : float
            Absolute tolerance for error control.
        h0 : float or None
            Initial step size (defaults to ``self.h``).
        h_min : float
            Minimum allowed step size.
        h_max : float or None
            Maximum allowed step size (defaults to ``time / 10``).
        max_steps : int
            Maximum number of accepted steps before stopping.
        """
        u_0 = self.build_initial_conditions()
        tf = self.time

        # Uniform output grid (same shape as solve_rk4 produces)
        t_eval = np.linspace(0.0, tf, self.steps + 1)

        if h0 is None:
            h0 = self.h
        if h_max is None:
            h_max = tf / 10.0

        h = min(h0, h_max)
        t = 0.0
        u_current = u_0.copy()

        # Store every accepted step for later interpolation
        times = [t]
        solutions = [u_current.copy()]

        safety = 0.9
        max_factor = 5.0
        min_factor = 0.1

        accepted = 0
        rejected = 0

        while t < tf and accepted < max_steps:
            # Don't overshoot the final time
            if t + h > tf:
                h = tf - t

            u_next, error = adaptive_rk4_step(F, t, u_current, h)

            # Weighted error norm (mix of relative & absolute)
            scale = atol + rtol * np.maximum(np.abs(u_current), np.abs(u_next))
            error_ratio = np.max(error / scale)

            if error_ratio <= 1.0:
                # --- accept step ---
                t += h
                u_current = u_next.copy()
                times.append(t)
                solutions.append(u_current.copy())
                accepted += 1

                # Increase step size (exponent 1/5 for the 5th-order method)
                if error_ratio > 1e-15:
                    h *= min(
                        max_factor, max(min_factor, safety * error_ratio ** (-1 / 5))
                    )
                else:
                    h *= max_factor
                h = min(h, h_max)

            else:
                # --- reject step, reduce step size ---
                rejected += 1
                if error_ratio > 1e-15:
                    h *= max(min_factor, safety * error_ratio ** (-1 / 4))
                else:
                    h *= min_factor

                if h < h_min:
                    # Cannot shrink any further — accept the step anyway
                    t += h
                    u_current = u_next.copy()
                    times.append(t)
                    solutions.append(u_current.copy())
                    accepted += 1
                    h = max(h, h_min)

        # Interpolate the adaptive solution back onto the uniform grid
        times_arr = np.array(times)
        solutions_arr = np.array(solutions)

        u = np.zeros((self.steps + 1, 4))
        for i in range(4):
            u[:, i] = np.interp(t_eval, times_arr, solutions_arr[:, i])

        # Ensure the initial condition is exact
        u[0] = u_0

        self._feval_count = 6 * (accepted + rejected)  # 6 stages per RKF45 attempt
        self.save_solution(u)
        print(
            f"Adaptive RK4 completed: {accepted} accepted, {rejected} rejected "
            f"steps (target grid: {self.steps} points)"
        )

    def get_energy(self, u: np.ndarray) -> np.ndarray:
        # u = [theta_1, omega_1, theta_2, omega_2]
        if u.ndim == 1:
            # Single state vector: shape (4,)
            return calc_kinetic_energy(
                M1, L1, u[1], M2, L2, u[3], u[0] - u[2]
            ) + calc_potential_energy(M1, L1, M2, L2, u[0], u[2])
        # Batch of state vectors: shape (N, 4)
        return calc_kinetic_energy(
            M1, L1, u[:, 1], M2, L2, u[:, 3], u[:, 0] - u[:, 2]
        ) + calc_potential_energy(M1, L1, M2, L2, u[:, 0], u[:, 2])

    def get_delta_E(self) -> np.ndarray:
        u = np.load(self.path_numpy)
        u_0, u_f = u[0], u[-1]
        return self.get_energy(u_f) - self.get_energy(u_0)

    def measure_compute(self) -> dict:
        """
        Force-recompute the solution and measure both wall-clock time and
        the number of right-hand-side evaluations (work done).

        Returns
        -------
        dict
            Keys: method, steps, time_s, feval_count, solution.
            ``feval_count`` is None for rk8 (scipy black-box).
        """
        import time

        t_start = time.perf_counter()
        solution = self.solve(force=True)
        elapsed = time.perf_counter() - t_start

        feval = self._feval_count
        result = {
            "method": self.method,
            "steps": len(solution),
            "time_s": elapsed,
            "feval_count": feval,
            "solution": solution,
        }

        feval_str = f"{feval:>10d}" if feval is not None else "     (n/a)"
        print(
            f"{'method':>14s}  |  {'steps':>8s}  |  {'time':>8s}  |  "
            f"{'steps/s':>10s}  |  {'f-evals':>10s}"
        )
        print(
            f"{self.method:>14s}  |  {len(solution):>8d}  |  "
            f"{elapsed:>7.3f}s  |  {len(solution) / elapsed:>10.0f}  |  "
            f"{feval_str}"
        )
        return result

    def plot_e_vs_time(self, method=None, preset=None, scale="normal"):
        """
        Delta E against H
        """
        t = np.linspace(0, self.time, self.steps)
        u = np.load(self.path_numpy)
        e_self = self.get_energy(u)
        if scale == "log":
            print(f"Using log scale for {self.method}")
            plt.plot(t, np.log(e_self), label=f"{self.method}, {self.h} log")
        else:
            plt.plot(t, e_self, label=f"{self.method}, {self.h}")
        # plt.axhline(e_self[0], color="r", linestyle="--")
        path_save = (
            f"{DOUBLE_PENDULUM_PATH}/plots/{self.method}_{self.h}_energy_{scale}.png"
        )
        if method is not None and preset is not None:
            solver = DoublePendulumSolver(method=method, preset=preset)
            u_odd = np.load(solver.path_numpy)
            plt.plot(t, self.get_energy(u_odd), label=f"{method}")
            plt.axhline(self.get_energy(u_odd)[0], color="r", linestyle="--")

        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title(
            f"{self.method}, {self.h}, {self.theta_1_str}, {self.theta_2_str}, {self.omega_1}, {self.omega_2}"
        )
        plt.legend()
        plt.savefig(path_save)
        print(f"Saved energy plot to {path_save}")

    def plot_delta_e_vs_h(self, h_num, h_list=None):
        # Total time fixed, h_variable then num_h steps variable
        if h_list is None:
            h_list = [(0.1) ** i for i in h_num]

        for h_step in h_list:

            def clean_string(h: float | str) -> str:
                return f"{h:.10f}".rstrip("0").rstrip(".")

            # Clean h string for filenames (e.g. 0.001, not 0.0010000000000002)
            h_fmt = clean_string(h_step)
            # Bypass presets so h=h_step isn't overwritten by set_preset()
            solver_h = DoublePendulumSolver(
                method=self.method,
                time=self.time,
                time_str=self.time_str,
                steps=int(self.time / h_step),
                theta_1=self.theta_1,
                theta_2=self.theta_2,
                omega_1=self.omega_1,
                omega_2=self.omega_2,
                h=h_step,
                h_str=h_fmt,
                preset="",
            )
            # __post_init__ already solved; just load the result
            vector_state_h = np.load(solver_h.path_numpy)
            energy_h = self.get_energy(vector_state_h)
            t = np.linspace(0, self.time, len(energy_h))
            plt.plot(t, energy_h, label=f"h={h_step}")

        if h_list is not None:
            prefix = "_".join(clean_string(h) for h in h_list)

        else:
            prefix = clean_string(str(h_num))
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        path = f"{DOUBLE_PENDULUM_PATH}/plots/{self.method}_{prefix}_delta_e_vs_h.png"
        print(f"Saved delta energy plot to {path}")
        plt.savefig(path)

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
        for i in range(u_self.shape[0] - 1):
            diff = u_self[i] - u_odd[i]
            diffs[i] = diff

        u_0_self = u_self[0]
        u_0_odd = u_odd[0]

        u_f_self = u_self[-1]
        u_f_odd = u_odd[-1]

        e_0_self = self.get_energy(u_0_self)
        print(f"Energy at t=0 (self): {e_0_self}")
        e_0_odd = solver_odd.get_energy(u_0_odd)
        print(f"Energy at t=0 (odd): {e_0_odd}")
        e_0_diff = e_0_self - e_0_odd
        print(f"Energy difference at t=0: {e_0_diff}")

        e_f_self = self.get_energy(u_f_self)
        print(f"Energy at t=final (self): {e_f_self}")
        e_f_odd = solver_odd.get_energy(u_f_odd)
        print(f"Energy at t=final (odd): {e_f_odd}")
        e_self_f_diff = e_f_self - e_0_self
        e_odd_f_diff = e_f_odd - e_0_odd

        print(f"Energy difference at t=final (self): {e_self_f_diff}")
        print(f"Energy difference at t=final (odd): {e_odd_f_diff}")

        mse = np.mean(diffs**2)
        deviation = np.sqrt(mse)
        mean = np.mean(diffs)
        print(f"MSE: {mse}, deviation: {deviation}, mean: {mean}")
