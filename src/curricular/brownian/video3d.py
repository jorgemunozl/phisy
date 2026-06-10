import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def brownian_path_3d(T: float, N: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / N
    dW = np.sqrt(dt) * rng.standard_normal((N, 3))
    return np.concatenate([[[0.0, 0.0, 0.0]], np.cumsum(dW, axis=0)])


def main() -> None:
    p = argparse.ArgumentParser(description="Animate a 3D Brownian motion (Wiener process).")
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
    p.add_argument("--show", action="store_true", help="Show interactively instead of saving.")
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
