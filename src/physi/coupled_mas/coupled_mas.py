"""Two fixed-end chains of N equal masses connected by equal springs."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

N = 3
M = 1.0
K = 1.0
OMEGA0 = np.sqrt(K / M)

T0, TF, H = 0.0, 40.0, 0.01
X0 = np.zeros(N)
X0[0] = 1.0
V0 = np.zeros(N)


def stiffness_matrix(n: int) -> np.ndarray:
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def rhs(_t: float, state: np.ndarray) -> np.ndarray:
    x = state[:N]
    v = state[N:]
    a = -(K / M) * (stiffness_matrix(N) @ x)
    return np.concatenate([v, a])


def rk4_step(f, t: float, u: np.ndarray, h: float) -> np.ndarray:
    k1 = h * f(t, u)
    k2 = h * f(t + h / 2, u + k1 / 2)
    k3 = h * f(t + h / 2, u + k2 / 2)
    k4 = h * f(t + h, u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(T0, TF, H)
    state = np.zeros((len(t), 2 * N))
    state[0] = np.concatenate([X0, V0])
    for i in range(len(t) - 1):
        state[i + 1] = rk4_step(rhs, t[i], state[i], H)
    return t, state


def modal_solution(t: np.ndarray) -> np.ndarray:
    """Analytical solution via normal-mode decomposition."""
    matrix = stiffness_matrix(N)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    omegas = np.sqrt(K / M * eigvals)

    norms = np.sum(eigvecs**2, axis=0)
    amplitudes = (eigvecs.T @ X0) / norms
    phases = (eigvecs.T @ V0) / (omegas * norms)

    x = np.zeros((len(t), N))
    for j in range(N):
        x += (
            amplitudes[j] * np.cos(omegas[j] * t)[:, None]
            + phases[j] * np.sin(omegas[j] * t)[:, None]
        ) * eigvecs[:, j]
    return x


def main() -> None:
    t, state = integrate()
    x_num = state[:, :N]
    x_ana = modal_solution(t)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i in range(N):
        axes[0].plot(t, x_num[:, i], label=rf"$x_{i + 1}$ (RK4)")
        axes[1].plot(t, x_ana[:, i] - x_num[:, i], label=rf"$x_{i + 1}$ error")

    axes[0].set_ylabel("displacement")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        rf"Coupled oscillators ($N={N}$, $m={M}$, $k={K}$, $\omega_0={OMEGA0:.2f}$)"
    )

    axes[1].set_xlabel("time")
    axes[1].set_ylabel("RK4 − modal")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coupled_mas_displacements.png", dpi=150)
    print("Saved coupled_mas_displacements.png")

    # Animation: masses on a horizontal chain with fixed walls.
    spacing = 1.0
    eq_positions = spacing * np.arange(1, N + 1)

    fig_anim, ax = plt.subplots(figsize=(8, 3))
    ax.axvline(0, color="k", lw=2)
    ax.axvline((N + 1) * spacing, color="k", lw=2)
    ax.set_xlim(-0.5, (N + 1) * spacing + 0.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Coupled mass-spring chain")
    ax.set_xlabel("position")
    ax.set_yticks([])

    (springs,) = ax.plot([], [], "-", color="C2", lw=2)
    (masses,) = ax.plot([], [], "o", color="C0", markersize=12)
    time_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes, fontsize=11, verticalalignment="top"
    )

    def init():
        springs.set_data([], [])
        masses.set_data([], [])
        time_text.set_text("")
        return springs, masses, time_text

    def animate(frame_idx: int):
        positions = eq_positions + x_num[frame_idx]
        wall_left = 0.0
        wall_right = (N + 1) * spacing
        xs = np.concatenate([[wall_left], positions, [wall_right]])
        ys = np.zeros_like(xs)
        springs.set_data(xs, ys)
        masses.set_data(positions, np.zeros(N))
        time_text.set_text(f"t = {t[frame_idx]:.2f} s")
        return springs, masses, time_text

    skip = 5
    frames = range(0, len(t), skip)
    ani = animation.FuncAnimation(
        fig_anim, animate, frames=frames, init_func=init, interval=20, blit=True
    )
    ani.save("coupled_mas.gif", writer="pillow", fps=30)
    print("Saved coupled_mas.gif")


if __name__ == "__main__":
    main()
