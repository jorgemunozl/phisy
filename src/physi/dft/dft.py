import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Grid
r_max = 20.0
N = 2000
r = np.linspace(1e-5, r_max, N)
dr = r[1] - r[0]

# Kinetic energy operator (finite differences)
diag = np.ones(N)
T = (-0.5 / dr**2) * (
    np.diag(-2 * diag) +
    np.diag(diag[:-1], 1) +
    np.diag(diag[:-1], -1)
)

# External potential: hydrogen nucleus
V_ext = -1.0 / r
V_ext_mat = np.diag(V_ext)

# Initial density guess (1s-like)
u = 2 * r * np.exp(-r)
u /= np.sqrt(np.trapezoid(u**2, r))

def lda_exchange_potential(n):
    """LDA exchange only (spin unpolarized)"""
    return -(3 / np.pi)**(1/3) * n**(1/3)

# Self-consistent loop
for _ in range(20):
    n = (u**2) / (4 * np.pi * r**2)
    n[n < 1e-10] = 1e-10

    V_xc = lda_exchange_potential(n)
    V_xc_mat = np.diag(V_xc)

    H = T + V_ext_mat + V_xc_mat
    eps, vecs = eigh(H)

    u = vecs[:, 0]
    u /= np.sqrt(np.trapezoid(u**2, r))

energy = eps[0]

print(f"Kohn–Sham energy (Hartree): {energy}")
print(f"Exact hydrogen energy:     {-0.5}")

# Plot density
plt.plot(r, n, label="DFT density")
plt.plot(r, np.exp(-2*r)/np.pi, "--", label="Exact density")
plt.xlim(0, 10)
plt.legend()
plt.show()
