import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)

U = X**3 - 3 * X * Y**2 + 2 * X
V = 3 * X**2 * Y - Y**3 + 2 * Y

plt.figure(figsize=(8, 8))
cs_u = plt.contour(
    X,
    Y,
    U,
    levels=np.arange(-10, 11, 2),
    colors="blue",
    linestyles="dashed",
    linewidths=1.2,
)
plt.clabel(cs_u, inline=True, fontsize=8, fmt="%.0f")

cs_v = plt.contour(
    X,
    Y,
    V,
    levels=np.arange(-10, 11, 2),
    colors="red",
    linestyles="solid",
    linewidths=1.2,
)
plt.clabel(cs_v, inline=True, fontsize=8, fmt="%.0f")

plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.gca().set_aspect("equal")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$u(x,y)=c_1$ (azul) y $v(x,y)=c_2$ (rojo) para $w(z)=z^3+2z$")
plt.legend(
    handles=[
        Line2D([0], [0], color="blue", linestyle="dashed", lw=1.2, label=r"$u=c_1$"),
        Line2D([0], [0], color="red", linestyle="solid", lw=1.2, label=r"$v=c_2$"),
    ],
    loc="upper right",
)
plt.tight_layout()
plt.savefig("prob1_contours.png", dpi=300)
plt.show()
