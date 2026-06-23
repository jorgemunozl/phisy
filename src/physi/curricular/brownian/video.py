import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
T = 5.0
N = 500
dt = T / N
fps = 30

# 2D Brownian motion: independent Wiener in x and y
dW = np.sqrt(dt) * np.random.standard_normal((N, 2))
path = np.concatenate([[[0, 0]], np.cumsum(dW, axis=0)])

fig, ax = plt.subplots()
ax.set_xlim(path[:, 0].min() - 0.5, path[:, 0].max() + 0.5)
ax.set_ylim(path[:, 1].min() - 0.5, path[:, 1].max() + 0.5)
ax.set_aspect("equal")
ax.set_title("Brownian motion")

trail, = ax.plot([], [], "b-", alpha=0.4, linewidth=0.8)
dot, = ax.plot([], [], "ro", markersize=10)

def init():
    trail.set_data([], [])
    dot.set_data([], [])
    return trail, dot

def update(frame):
    trail.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
    dot.set_data([path[frame, 0]], [path[frame, 1]])
    return trail, dot

nframes = min(N + 1, int(T * fps))
interval = 1000 / fps
anim = FuncAnimation(fig, update, init_func=init, frames=nframes, interval=interval, blit=True)

anim.save("brownian_motion.gif", writer=PillowWriter(fps=fps))
print("Saved brownian_motion.gif")
