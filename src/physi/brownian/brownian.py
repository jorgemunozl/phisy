import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


class BrownianMotion:
    def __init__(self, T, N, dimension: int, fps=30):
        self.T = T
        self.N = N
        self.dimension = dimension
        self.dt = T / N
        self.t = np.linspace(0, T, N + 1)
        self.fps = fps

    def generate(self):
        if self.dimension == 1:
            self.dW = np.sqrt(self.dt) * np.random.standard_normal(self.N)
            self.W = np.concatenate([[0], np.cumsum(self.dW)])
        elif self.dimension == 2:
            self.dW = np.sqrt(self.dt) * np.random.standard_normal(
                (self.N, self.dimension)
            )
            self.W = np.concatenate([[[0, 0]], np.cumsum(self.dW, axis=0)])
        else:
            raise ValueError("dimension must be 1 or 2")

    def plot(self):
        plt.plot(self.t, self.W)
        plt.xlabel("t")
        plt.ylabel("W(t)")
        plt.title("Brownian motion")
        plt.show()

    def animate(self):

        fig, ax = plt.subplots()
        ax.set_xlim(path[:, 0].min() - 0.5, path[:, 0].max() + 0.5)
        ax.set_ylim(path[:, 1].min() - 0.5, path[:, 1].max() + 0.5)
        ax.set_aspect("equal")
        ax.set_title("Brownian motion")

        (trail,) = ax.plot([], [], "b-", alpha=0.4, linewidth=0.8)
        (dot,) = ax.plot([], [], "ro", markersize=10)

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
        anim = FuncAnimation(
            fig, update, init_func=init, frames=nframes, interval=interval, blit=True
        )

        anim.save("brownian_motion.gif", writer=PillowWriter(fps=fps))
        print("Saved brownian_motion.gif")



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def brownian_path_3d(T: float, N: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / N
    dW = np.sqrt(dt) * rng.standard_normal((N, 3))
    return np.concatenate([[[0.0, 0.0, 0.0]], np.cumsum(dW, axis=0)])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Animate a 3D Brownian motion (Wiener process)."
    )
    p.add_argument("--T", type=float, default=5.0, help="Total time horizon.")
    p.add_argument("--N", type=int, default=500, help="Number of steps.")
    p.add_argument("--fps", type=int, default=30, help="Frames per second.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    p.add_argument(
        "--outfile",
        type=str,
        default="brownian_motion_3d.gif",
        help="Output GIF path.",
    )
    p.add_argument(
        "--show", action="store_true", help="Show interactively instead of saving."
    )
    args = p.parse_args()

    path = brownian_path_3d(T=args.T, N=args.N, seed=args.seed)

    mins = path.min(axis=0)
    maxs = path.max(axis=0)
    pad = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Brownian motion")
    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)
    try:
        ax.set_box_aspect((maxs - mins) + 2 * pad)  # matplotlib >= 3.3
    except Exception:
        pass

    (trail,) = ax.plot([], [], [], "b-", alpha=0.4, linewidth=0.8)
    (dot,) = ax.plot([], [], [], "ro", markersize=6)

    def init():
        trail.set_data([], [])
        trail.set_3d_properties([])
        dot.set_data([], [])
        dot.set_3d_properties([])
        return (trail, dot)

    def update(frame: int):
        xs = path[: frame + 1, 0]
        ys = path[: frame + 1, 1]
        zs = path[: frame + 1, 2]
        trail.set_data(xs, ys)
        trail.set_3d_properties(zs)
        dot.set_data([path[frame, 0]], [path[frame, 1]])
        dot.set_3d_properties([path[frame, 2]])
        return (trail, dot)

    interval = 1000 / args.fps
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=path.shape[0],
        interval=interval,
        blit=False,  # blitting is unreliable for 3D axes
    )

    if args.show:
        plt.show()
        return

    anim.save(args.outfile, writer=PillowWriter(fps=args.fps))
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()
