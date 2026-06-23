import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

m1 = m2 = 1.0
l1 = l2 = 1.0
g = 9.81


def ode(t, y):
    th1, th2, w1, w2 = y
    d = th1 - th2
    cd, sd = np.cos(d), np.sin(d)
    # Matriz M * [a1, a2]^T = F
    M = np.array([[(m1 + m2) * l1, m2 * l2 * cd], [l1 * cd, l2]])
    F = np.array(
        [
            -m2 * l2 * w2**2 * sd - (m1 + m2) * g * np.sin(th1),
            l1 * w1**2 * sd - g * np.sin(th2),
        ]
    )
    a1, a2 = np.linalg.solve(M, F)
    return [w1, w2, a1, a2]


t_span = (0, 20)
y0 = [np.pi / 2, np.pi / 2, 0.0, 0.0]
sol = solve_ivp(ode, t_span, y0, max_step=0.01, method="RK45")
t, th1, th2 = sol.t, sol.y[0], sol.y[1]

# Posiciones de m2
x2 = l1 * np.sin(th1) + l2 * np.sin(th2)
y2 = -l1 * np.cos(th1) - l2 * np.cos(th2)

# Ángulos vs tiempo
plt.figure(figsize=(10, 5))
plt.plot(t, th1, label=r"$\theta_1(t)$")
plt.plot(t, th2, label=r"$\theta_2(t)$")
plt.xlabel("$t$ (s)")
plt.ylabel("$\theta$ (rad)")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Evolución de los ángulos — Péndulo doble")
plt.tight_layout()
plt.savefig("prob4_angles.png", dpi=300)
plt.show()

# Trayectoria de m2
plt.figure(figsize=(7, 7))
plt.plot(x2, y2, "b-", lw=0.5)
plt.xlabel("$x_2$ (m)")
plt.ylabel("$y_2$ (m)")
plt.gca().set_aspect("equal")
plt.grid(alpha=0.3)
plt.title("Trayectoria de la masa inferior $m_2$")
plt.tight_layout()
plt.savefig("prob4_trajectory.png", dpi=300)
plt.show()
