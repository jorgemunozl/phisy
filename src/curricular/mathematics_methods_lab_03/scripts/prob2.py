import matplotlib.pyplot as plt
import numpy as np

# --- (a) Raíces de z^6 + 64 = 0 ---
k = np.arange(6)
roots = 2 * np.exp(1j * (np.pi / 6 + k * np.pi / 3))

print("Raíces de z^6 + 64 = 0:")
for i, z in enumerate(roots):
    print(f"  z_{i} = {z.real:.6f} + ({z.imag:.6f})i")

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(roots.real, roots.imag, color="red", s=80, zorder=5)
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(2 * np.cos(theta), 2 * np.sin(theta), "gray", "--", lw=0.8, label="$|z|=2$")
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("Re($z$)")
ax.set_ylabel("Im($z$)")
ax.set_title("$z^6+64=0$ — Plano de Argand")
for i, z in enumerate(roots):
    ax.annotate(
        f"$z_{i}$",
        (z.real, z.imag),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )
ax.legend()
plt.tight_layout()
plt.savefig("prob2_roots.png", dpi=300)
plt.show()


# --- (b) Mapeo w = z + 1/z ---
def w_map(z):
    return z + 1 / z


print("\nPunto singular: z = 0 (polo simple)\n")

fig, ax = plt.subplots(figsize=(8, 8))
theta = np.linspace(0, 2 * np.pi, 2000)
for r, c in zip([1.1, 1.5, 2.0], ["blue", "green", "orange"]):
    w = w_map(r * np.exp(1j * theta))
    ax.plot(w.real, w.imag, color=c, lw=1.5, label=f"$|z|={r}$")
w1 = w_map(1.0 * np.exp(1j * theta))
ax.plot(w1.real, w1.imag, "red", "--", lw=1.5, label="$|z|=1$", alpha=0.7)
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.set_xlabel("$u$")
ax.set_ylabel("$v$")
ax.set_title(r"$w = z + 1/z$ — Imagen de $|z|=r$")
ax.legend()
plt.tight_layout()
plt.savefig("prob2_mapping.png", dpi=300)
plt.show()
