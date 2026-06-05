import matplotlib.pyplot as plt
import numpy as np


def F(t, u, gamma=0.1, omega0=1.0):
    u1, u2 = u
    du1 = u2
    du2 = -u1
    return np.array([du1, du2])


def rk4_step(f, t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + h / 2, u + k1 / 2)
    k3 = h * f(t + h / 2, u + k2 / 2)
    k4 = h * f(t + h, u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Initial conditions: y(0) = 1, y'(0) = 0
t0, tf, h = 0.0, 20.0, 0.01
t = np.arange(t0, tf, h)
u = np.zeros((len(t), 2))
u[0] = [1.0, 0.0]  # [y0, y'0]

for i in range(len(t) - 1):
    u[i + 1] = rk4_step(F, t[i], u[i], h)

plt.plot(t, u[:, 0], label="y(t)")
plt.plot(t, u[:, 1], label="y'(t)")
plt.legend()
plt.grid()
plt.show()
