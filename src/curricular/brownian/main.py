import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0
N = 1000
dt = T / N

# Simple 1D Brownian motion: W(t+dt) = W(t) + sqrt(dt) * Z, Z ~ N(0,1)
t = np.linspace(0, T, N + 1)
dW = np.sqrt(dt) * np.random.standard_normal(N)
W = np.concatenate([[0], np.cumsum(dW)])

plt.plot(t, W)
plt.xlabel("t")
plt.ylabel("W(t)")
plt.title("Brownian motion")
plt.show()
