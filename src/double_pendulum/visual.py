from pathlib import Path

import numpy as np

# Physical parameters — kept in sync with main.py
from constants import GRAVITY, L1, L2, M1, M2

from manim import (
    BLUE,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    Dot,
    Line,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

# ── Settings ──────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "30_100000_pi_2_pi_2_0.0_0.0_rk4.npy"
ANIM_DURATION = 30.0  # desired total playback duration in scene-seconds
RENDER_FPS = 60  # must match the -ql/--fps flag you pass to manim
#   -ql → 15 fps | -qm → 30 fps | -qh/-qk → 60 fps
SUBSAMPLING = 1  # extra trail thinning (1 = every rendered step)

# The pendulum lives in the left half of the screen; shift its pivot here.
PENDULUM_OFFSET = np.array([-3.5, 0.0, 0.0])


def calc_kinetic_energy(m1, l1, omega_1, m2, l2, omega_2, diff) -> np.ndarray:
    """
    KE of the double pendulum.
    KE = 1/2 m1 l1^2 w1^2  +  1/2 m2 (l1^2 w1^2 + l2^2 w2^2 + 2 l1 l2 w1 w2 cos(dθ))
    """
    return 0.5 * m1 * l1**2 * omega_1**2 + 0.5 * m2 * (
        l1**2 * omega_1**2
        + l2**2 * omega_2**2
        + 2 * l1 * l2 * omega_1 * omega_2 * np.cos(diff)
    )


def calc_potential_energy(m1, l1, m2, l2, theta_1, theta_2) -> np.ndarray:
    """PE = -(m1+m2) g l1 cos theta1  -  m2 g l2 cos theta2"""
    return -(m1 + m2) * GRAVITY * l1 * np.cos(theta_1) - m2 * GRAVITY * l2 * np.cos(
        theta_2
    )


# ══════════════════════════════════════════════════════════════════════
#  Manim scene
# ══════════════════════════════════════════════════════════════════════
class DoublePendulum(Scene):
    """
    Left half  → double-pendulum animation.
    Right half → live bar chart: kinetic (blue), potential (red), total (yellow).
    """

    def construct(self):
        # ── Load trajectory ──────────────────────────────────────────
        # Shape: (N, 4)  columns: [theta1, omega1, theta2, omega2]
        traj = np.load(DATA_PATH)
        num_frames = len(traj)

        # ── Cartesian coordinates ────────────────────────────────────
        x1 = L1 * np.sin(traj[:, 0])
        y1 = -L1 * np.cos(traj[:, 0])
        x2 = x1 + L2 * np.sin(traj[:, 2])
        y2 = y1 - L2 * np.cos(traj[:, 2])

        # ── Energy arrays ────────────────────────────────────────────
        ke_arr = calc_kinetic_energy(
            M1, L1, traj[:, 1], M2, L2, traj[:, 3], traj[:, 0] - traj[:, 2]
        )
        pe_arr = calc_potential_energy(M1, L1, M2, L2, traj[:, 0], traj[:, 2])
        total_arr = ke_arr + pe_arr

        # ── Pendulum: scale to fit inside the left half ──────────────
        max_extent = max(np.abs(x2).max(), np.abs(y2).max(), 0.1) + 0.3
        scale = 2.5 / max_extent

        def pt(xx: float, yy: float) -> np.ndarray:
            """Physical (x, y) → Manim 3-D point, shifted to the left half."""
            return np.array([xx * scale, yy * scale, 0.0]) + PENDULUM_OFFSET

        pivot_pt = pt(0.0, 0.0)
        pts_p1 = [pt(x1[i], y1[i]) for i in range(num_frames)]
        pts_p2 = [pt(x2[i], y2[i]) for i in range(num_frames)]

        # ── Pendulum scene objects ────────────────────────────────────
        pivot = Dot(pivot_pt, color=WHITE, radius=0.08)
        mass1 = Dot(pts_p1[0], color=BLUE, radius=0.15 * (M1 ** (1 / 3)))
        mass2 = Dot(pts_p2[0], color=GREEN, radius=0.15 * (M2 ** (1 / 3)))
        rod1 = Line(pivot_pt, pts_p1[0], color=BLUE, stroke_width=6)
        rod2 = Line(pts_p1[0], pts_p2[0], color=GREEN, stroke_width=5)
        trail = VGroup()

        # ── Vertical divider ──────────────────────────────────────────
        divider = Line(
            np.array([0.0, -4.2, 0.0]),
            np.array([0.0, 4.2, 0.0]),
            color=WHITE,
            stroke_width=1.0,
            stroke_opacity=0.35,
        )

        # ── Energy chart layout (right half: x in [0.5 … 7]) ─────────
        # Three bars: KE at 2.0, PE at 3.8, Total at 5.6
        BAR_W = 0.85
        BAR_X = {"ke": 2.0, "pe": 3.8, "total": 5.6}

        # Screen y-range for the chart
        CHART_Y_MIN = -3.0
        CHART_Y_MAX = 3.2

        # Map all energy values to that range (with a bit of padding)
        all_vals = np.concatenate([ke_arr, pe_arr, total_arr])
        e_min = float(all_vals.min())
        e_max = float(all_vals.max())
        pad = 0.1 * max(e_max - e_min, 1.0)
        e_min -= pad
        e_max += pad

        def e2y(e: float) -> float:
            """Map an energy value to a Manim y-coordinate."""
            return CHART_Y_MIN + (e - e_min) / (e_max - e_min) * (
                CHART_Y_MAX - CHART_Y_MIN
            )

        zero_y = e2y(0.0)

        def make_bar(x: float, val: float, color) -> Rectangle:
            """
            Build a filled Rectangle representing *val* on the energy axis.
            Bars above zero_y for positive values, below for negative.
            """
            y_top = e2y(float(val))
            h = max(abs(y_top - zero_y), 0.02)
            cy = (y_top + zero_y) / 2.0
            bar = Rectangle(
                width=BAR_W,
                height=h,
                color=color,
                fill_color=color,
                fill_opacity=0.75,
                stroke_width=1,
                stroke_color=color,
            )
            bar.move_to(np.array([x, cy, 0.0]))
            return bar

        # Initial bars
        ke_bar = make_bar(BAR_X["ke"], ke_arr[0], BLUE)
        pe_bar = make_bar(BAR_X["pe"], pe_arr[0], RED)
        tot_bar = make_bar(BAR_X["total"], total_arr[0], YELLOW)

        # Zero reference line that spans all three bars
        x_left = min(BAR_X.values()) - BAR_W * 0.7
        x_right = max(BAR_X.values()) + BAR_W * 0.7
        zero_line = Line(
            np.array([x_left, zero_y, 0.0]),
            np.array([x_right, zero_y, 0.0]),
            color=WHITE,
            stroke_width=1.5,
            stroke_opacity=0.55,
        )
        zero_label = Text("0", font_size=18, color=WHITE).move_to(
            np.array([x_left - 0.25, zero_y, 0.0])
        )

        # Bar labels (below the chart)
        lbl_y = CHART_Y_MIN - 0.45
        ke_lbl = Text("KE", font_size=22, color=BLUE).move_to(
            np.array([BAR_X["ke"], lbl_y, 0.0])
        )
        pe_lbl = Text("PE", font_size=22, color=RED).move_to(
            np.array([BAR_X["pe"], lbl_y, 0.0])
        )
        tot_lbl = Text("Total", font_size=22, color=YELLOW).move_to(
            np.array([BAR_X["total"], lbl_y, 0.0])
        )

        # Section title
        title = Text("Energy", font_size=26, color=WHITE).move_to(
            np.array([(BAR_X["ke"] + BAR_X["total"]) / 2, CHART_Y_MAX + 0.45, 0.0])
        )

        # ── Add everything to the scene ───────────────────────────────
        self.add(
            # left: pendulum
            pivot,
            trail,
            rod1,
            rod2,
            mass1,
            mass2,
            # separator
            divider,
            # right: energy chart
            ke_bar,
            pe_bar,
            tot_bar,
            zero_line,
            zero_label,
            ke_lbl,
            pe_lbl,
            tot_lbl,
            title,
        )
        self.wait(0.3)

        # ── Frame-by-frame playback ───────────────────────────────────
        # Each self.wait() must be >= 1/RENDER_FPS or manim silently rounds
        # it up, making the video much longer than intended.  We stride
        # through the data so that exactly one rendered frame passes per
        # loop iteration.
        min_wait = 1.0 / RENDER_FPS
        ideal_wait = ANIM_DURATION / max(num_frames - 1, 1)
        stride = max(1, int(np.ceil(min_wait / ideal_wait)))
        indices = range(0, num_frames, stride)
        time_per_step = stride * ideal_wait  # now always >= min_wait

        for i in indices:
            p1 = pts_p1[i]
            p2 = pts_p2[i]

            # ── Pendulum ──────────────────────────────────────────────
            mass1.move_to(p1)
            mass2.move_to(p2)
            rod1.become(Line(pivot_pt, p1, color=BLUE, stroke_width=6))
            rod2.become(Line(p1, p2, color=GREEN, stroke_width=5))

            # Trailing dot on mass 2
            if i % (stride * SUBSAMPLING) == 0:
                alpha = 0.15 + 0.65 * (i / max(num_frames - 1, 1))
                dot = Dot(p2, color=YELLOW, radius=0.025)
                dot.set_opacity(alpha)
                trail.add(dot)

            # ── Energy bars ───────────────────────────────────────────
            ke_bar.become(make_bar(BAR_X["ke"], ke_arr[i], BLUE))
            pe_bar.become(make_bar(BAR_X["pe"], pe_arr[i], RED))
            tot_bar.become(make_bar(BAR_X["total"], total_arr[i], YELLOW))

            self.wait(time_per_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from manim import config

    config.progress_bar = "display"  # always show the progress bar
    config.quality = "low_quality"  # -ql  (change to medium/high as needed)
    config.preview = True  # open the video when done
    config.disable_caching = True  # always re-render, never use stale cache
    scene = DoublePendulum()
    scene.render()
