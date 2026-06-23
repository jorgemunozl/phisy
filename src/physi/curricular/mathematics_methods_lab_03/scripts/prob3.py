import matplotlib.pyplot as plt
import numpy as np

a, b = 1.0, 0.0
x = np.linspace(-2, 2, 400)
y = a * np.cosh((x - b) / a)

plt.figure(figsize=(8, 5))
plt.plot(x, y, "b-", lw=2, label=f"$y(x) = {a}\\cosh((x-{b})/{a})$")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.grid(alpha=0.3)
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.title("Catenaria — superficie de revolución mínima")
plt.legend()
plt.tight_layout()
plt.savefig("prob3_catenary.png", dpi=300)
plt.show()
