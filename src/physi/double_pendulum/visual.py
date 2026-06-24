"""
Visualisations for the double pendulum.

Available scenes (set via SCENE variable at the bottom of this file):
  - DoublePendulumEnergy    : single pendulum + live energy bar chart (original)
  - DoublePendulumSideBySide: RK4 vs RK8 side by side
  - DoublePendulumOverlay   : RK4 vs RK8 superimposed on the same pivot
"""

from pathlib import Path

import numpy as np

# Physical parameters — kept in sync with main.py
from config import (
    GRAVITY,
    L1,
    L2,
    M1,
    M2,
    calc_kinetic_energy,
    calc_potential_energy,
)
from manim import (
    BLUE,
    GREEN,
    ORANGE,
    RED,
    RIGHT,
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
RK4_PATH = Path(__file__).parent / "data" / "30s_10e5_pi_2_pi_2_1.0_-1.0_rk4.npy"
RK8_PATH = Path(__file__).parent / "data" / "30s_10e5_pi_2_pi_2_1.0_-1.0_rk8.npy"
ANIM_DURATION = 30.0  # desired total playback duration in scene-seconds
RENDER_FPS = 30  # must match config.quality below (ql→15, qm→30, qh/qk→60)
SUBSAMPLING = 1  # extra trail thinning (1 = every rendered step)


# ── Shared helpers ────────────────────────────────────────────────────


def load_aligned_trajectories():
    """Load RK4 and RK8 data, returning (traj_rk4, traj_rk8, num_frames)
    trimmed to the same number of time steps."""
    traj_rk4 = np.load(RK4_PATH)  # (100001, 4)
    traj_rk8 = np.load(RK8_PATH)  # (100000, 4)
    num_frames = min(len(traj_rk4), len(traj_rk8))
    return traj_rk4[:num_frames], traj_rk8[:num_frames], num_frames


def cartesian_from_traj(traj):
    """Return (x1, y1, x2, y2) cartesian coordinates from a trajectory
    array of shape (N, 4) with columns [theta1, omega1, theta2, omega2]."""
    x1 = L1 * np.sin(traj[:, 0])
    y1 = -L1 * np.cos(traj[:, 0])
    x2 = x1 + L2 * np.sin(traj[:, 2])
    y2 = y1 - L2 * np.cos(traj[:, 2])
    return x1, y1, x2, y2


def compute_scale(x2, y2, screen_half_width=3.5):
    """Compute a uniform scale so that the pendulum fits inside
    a region of width ``screen_half_width``."""
    max_extent = max(np.abs(x2).max(), np.abs(y2).max(), 0.1) + 0.3
    return screen_half_width / max_extent


def build_pendulum_objects(
    traj,
    offset,
    rod1_color,
    rod2_color,
    mass1_color,
    mass2_color,
    trail_color,
    *,
    scale,
):
    """Return (pts_p1, pts_p2, pivot, rod1, rod2, mass1, mass2, trail)
    for one pendulum instance.  ``scale`` is the pre-computed uniform scale."""
    x1, y1, x2, y2 = cartesian_from_traj(traj)

    def pt(xx, yy):
        return np.array([xx * scale, yy * scale, 0.0]) + offset

    pivot_pt = pt(0.0, 0.0)
    pts_p1 = [pt(x1[i], y1[i]) for i in range(len(traj))]
    pts_p2 = [pt(x2[i], y2[i]) for i in range(len(traj))]

    pivot = Dot(pivot_pt, color=WHITE, radius=0.08)
    mass1 = Dot(pts_p1[0], color=mass1_color, radius=0.15 * (M1 ** (1 / 3)))
    mass2 = Dot(pts_p2[0], color=mass2_color, radius=0.15 * (M2 ** (1 / 3)))
    rod1 = Line(pivot_pt, pts_p1[0], color=rod1_color, stroke_width=6)
    rod2 = Line(pts_p1[0], pts_p2[0], color=rod2_color, stroke_width=5)
    trail = VGroup()

    return pts_p1, pts_p2, pivot, rod1, rod2, mass1, mass2, trail


def compute_stride(num_frames):
    """Return (indices, time_per_step) for frame-by-frame playback."""
    min_wait = 1.0 / RENDER_FPS
    ideal_wait = ANIM_DURATION / max(num_frames - 1, 1)
    stride = max(1, int(np.ceil(min_wait / ideal_wait)))
    indices = range(0, num_frames, stride)
    return indices, stride * ideal_wait


def add_trail_dot(trail, pt, i, num_frames, color, stride):
    """Append a fading dot to a trail VGroup."""
    alpha = 0.15 + 0.65 * (i / max(num_frames - 1, 1))
    dot = Dot(pt, color=color, radius=0.025)
    dot.set_opacity(alpha)
    trail.add(dot)


# ══════════════════════════════════════════════════════════════════════
#  Scene 1 — original: pendulum + live energy bar chart
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumEnergy(Scene):
    """
    Left half  → double-pendulum animation.
    Right half → live bar chart: kinetic (blue), potential (red), total (yellow).
    """

    def construct(self):
        traj = np.load(RK4_PATH)
        num_frames = len(traj)

        # Cartesian coordinates
        x1, y1, x2, y2 = cartesian_from_traj(traj)

        # Energy arrays
        ke_arr = calc_kinetic_energy(
            M1, L1, traj[:, 1], M2, L2, traj[:, 3], traj[:, 0] - traj[:, 2]
        )
        pe_arr = calc_potential_energy(M1, L1, M2, L2, traj[:, 0], traj[:, 2])
        total_arr = ke_arr + pe_arr

        # Scale & offset for the left-half pendulum
        scale = compute_scale(x2, y2, screen_half_width=2.5)
        offset = np.array([-3.5, 0.0, 0.0])

        pts_p1, pts_p2, pivot, rod1, rod2, mass1, mass2, trail = build_pendulum_objects(
            traj,
            offset,
            rod1_color=BLUE,
            rod2_color=GREEN,
            mass1_color=BLUE,
            mass2_color=GREEN,
            trail_color=YELLOW,
            scale=scale,
        )

        # Vertical divider
        divider = Line(
            np.array([0.0, -4.2, 0.0]),
            np.array([0.0, 4.2, 0.0]),
            color=WHITE,
            stroke_width=1.0,
            stroke_opacity=0.35,
        )

        # ── Energy chart (right half) ────────────────────────────────
        BAR_W = 0.85
        BAR_X = {"ke": 2.0, "pe": 3.8, "total": 5.6}
        CHART_Y_MIN = -3.0
        CHART_Y_MAX = 3.2

        all_vals = np.concatenate([ke_arr, pe_arr, total_arr])
        e_min = float(all_vals.min())
        e_max = float(all_vals.max())
        pad = 0.1 * max(e_max - e_min, 1.0)
        e_min -= pad
        e_max += pad

        def e2y(e):
            return CHART_Y_MIN + (e - e_min) / (e_max - e_min) * (
                CHART_Y_MAX - CHART_Y_MIN
            )

        zero_y = e2y(0.0)

        def make_bar(x, val, color):
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

        ke_bar = make_bar(BAR_X["ke"], ke_arr[0], BLUE)
        pe_bar = make_bar(BAR_X["pe"], pe_arr[0], RED)
        tot_bar = make_bar(BAR_X["total"], total_arr[0], YELLOW)

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
        title = Text("Energy", font_size=26, color=WHITE).move_to(
            np.array([(BAR_X["ke"] + BAR_X["total"]) / 2, CHART_Y_MAX + 0.45, 0.0])
        )

        self.add(
            pivot,
            trail,
            rod1,
            rod2,
            mass1,
            mass2,
            divider,
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

        indices, t_step = compute_stride(num_frames)
        for i in indices:
            p1, p2 = pts_p1[i], pts_p2[i]

            mass1.move_to(p1)
            mass2.move_to(p2)
            rod1.become(Line(pivot.get_center(), p1, color=BLUE, stroke_width=6))
            rod2.become(Line(p1, p2, color=GREEN, stroke_width=5))

            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    trail, p2, i, num_frames, YELLOW, int(np.ceil(t_step * RENDER_FPS))
                )

            ke_bar.become(make_bar(BAR_X["ke"], ke_arr[i], BLUE))
            pe_bar.become(make_bar(BAR_X["pe"], pe_arr[i], RED))
            tot_bar.become(make_bar(BAR_X["total"], total_arr[i], YELLOW))

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 2 — RK4 vs RK8 side by side
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumSideBySide(Scene):
    """
    Left side  → RK4 solver
    Right side → DOP853 (RK8) solver
    """

    def construct(self):
        traj_rk4, traj_rk8, num_frames = load_aligned_trajectories()

        # Use the larger extent from both trajectories for a unified scale
        _, _, x2_4, y2_4 = cartesian_from_traj(traj_rk4)
        _, _, x2_8, y2_8 = cartesian_from_traj(traj_rk8)
        scale = compute_scale(
            np.concatenate([x2_4, x2_8]),
            np.concatenate([y2_4, y2_8]),
            screen_half_width=3.0,
        )

        offset_L = np.array([-3.6, 0.0, 0.0])
        offset_R = np.array([3.6, 0.0, 0.0])

        pts4_p1, pts4_p2, piv4, r4_1, r4_2, m4_1, m4_2, t4 = build_pendulum_objects(
            traj_rk4,
            offset_L,
            rod1_color=BLUE,
            rod2_color=GREEN,
            mass1_color=BLUE,
            mass2_color=GREEN,
            trail_color=YELLOW,
            scale=scale,
        )
        pts8_p1, pts8_p2, piv8, r8_1, r8_2, m8_1, m8_2, t8 = build_pendulum_objects(
            traj_rk8,
            offset_R,
            rod1_color=RED,
            rod2_color=ORANGE,
            mass1_color=RED,
            mass2_color=ORANGE,
            trail_color=WHITE,
            scale=scale,
        )

        # Labels
        lbl4 = Text("RK4", font_size=26, color=BLUE).move_to(
            np.array([offset_L[0], 3.5, 0.0])
        )
        lbl8 = Text("DOP853", font_size=26, color=RED).move_to(
            np.array([offset_R[0], 3.5, 0.0])
        )
        caption = Text(
            "Same initial conditions — different solvers → chaos!",
            font_size=20,
            color=WHITE,
        ).move_to(np.array([0.0, -3.6, 0.0]))

        divider = Line(
            np.array([0.0, -4.0, 0.0]),
            np.array([0.0, 4.0, 0.0]),
            color=WHITE,
            stroke_width=1.0,
            stroke_opacity=0.35,
        )

        self.add(
            piv4,
            t4,
            r4_1,
            r4_2,
            m4_1,
            m4_2,
            piv8,
            t8,
            r8_1,
            r8_2,
            m8_1,
            m8_2,
            divider,
            lbl4,
            lbl8,
            caption,
        )
        self.wait(0.3)

        indices, t_step = compute_stride(num_frames)
        for i in indices:
            p4_1, p4_2 = pts4_p1[i], pts4_p2[i]
            p8_1, p8_2 = pts8_p1[i], pts8_p2[i]

            # RK4
            m4_1.move_to(p4_1)
            m4_2.move_to(p4_2)
            r4_1.become(Line(piv4.get_center(), p4_1, color=BLUE, stroke_width=6))
            r4_2.become(Line(p4_1, p4_2, color=GREEN, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t4, p4_2, i, num_frames, YELLOW, int(np.ceil(t_step * RENDER_FPS))
                )

            # RK8
            m8_1.move_to(p8_1)
            m8_2.move_to(p8_2)
            r8_1.become(Line(piv8.get_center(), p8_1, color=RED, stroke_width=6))
            r8_2.become(Line(p8_1, p8_2, color=ORANGE, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t8, p8_2, i, num_frames, WHITE, int(np.ceil(t_step * RENDER_FPS))
                )

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 3 — RK4 vs RK8 superimposed (transposed) on the same pivot
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumOverlay(Scene):
    """
    Both RK4 and RK8 pendulums drawn on top of each other at the same
    pivot point.  They start in sync and gradually diverge — the clearest
    visualisation of chaotic sensitivity to numerical errors.
    """

    def construct(self):
        traj_rk4, traj_rk8, num_frames = load_aligned_trajectories()

        # Use the combined extent for a single scale
        _, _, x2_4, y2_4 = cartesian_from_traj(traj_rk4)
        _, _, x2_8, y2_8 = cartesian_from_traj(traj_rk8)
        scale = compute_scale(
            np.concatenate([x2_4, x2_8]),
            np.concatenate([y2_4, y2_8]),
            screen_half_width=6.0,  # full width available
        )

        center = np.array([0.0, 0.0, 0.0])

        # RK4 — blue/green with yellow trail
        pts4_p1, pts4_p2, piv4, r4_1, r4_2, m4_1, m4_2, t4 = build_pendulum_objects(
            traj_rk4,
            center,
            rod1_color=BLUE,
            rod2_color=GREEN,
            mass1_color=BLUE,
            mass2_color=GREEN,
            trail_color=YELLOW,
            scale=scale,
        )

        # RK8 — red/orange with white trail, slightly translucent
        pts8_p1, pts8_p2, piv8, r8_1, r8_2, m8_1, m8_2, t8 = build_pendulum_objects(
            traj_rk8,
            center,
            rod1_color=RED,
            rod2_color=ORANGE,
            mass1_color=RED,
            mass2_color=ORANGE,
            trail_color=WHITE,
            scale=scale,
        )

        # Make RK8 objects slightly transparent so RK4 shows through
        for obj in (r8_1, r8_2, m8_1, m8_2):
            obj.set_opacity(0.7)

        # Labels
        legend = VGroup(
            Dot(radius=0.08, color=BLUE).move_to(np.array([-4.5, 3.2, 0.0])),
            Text("RK4", font_size=22, color=BLUE).next_to(
                np.array([-4.5, 3.2, 0.0]), RIGHT
            ),
            Dot(radius=0.08, color=RED).move_to(np.array([-4.5, 2.7, 0.0])),
            Text("DOP853", font_size=22, color=RED).next_to(
                np.array([-4.5, 2.7, 0.0]), RIGHT
            ),
        )

        caption = Text(
            "Same IC — numerical errors diverge → chaos",
            font_size=20,
            color=WHITE,
        ).move_to(np.array([0.0, -3.6, 0.0]))

        # Both share the same pivot — only add one
        self.add(
            piv4,  # single pivot point
            t4,
            t8,  # trails (RK4 on top)
            r4_1,
            r4_2,
            r8_1,
            r8_2,
            m4_1,
            m4_2,
            m8_1,
            m8_2,
            legend,
            caption,
        )
        self.wait(0.3)

        indices, t_step = compute_stride(num_frames)
        for i in indices:
            p4_1, p4_2 = pts4_p1[i], pts4_p2[i]
            p8_1, p8_2 = pts8_p1[i], pts8_p2[i]

            # RK4
            m4_1.move_to(p4_1)
            m4_2.move_to(p4_2)
            r4_1.become(Line(piv4.get_center(), p4_1, color=BLUE, stroke_width=6))
            r4_2.become(Line(p4_1, p4_2, color=GREEN, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t4, p4_2, i, num_frames, YELLOW, int(np.ceil(t_step * RENDER_FPS))
                )

            # RK8
            m8_1.move_to(p8_1)
            m8_2.move_to(p8_2)
            r8_1.become(Line(piv4.get_center(), p8_1, color=RED, stroke_width=6))
            r8_2.become(Line(p8_1, p8_2, color=ORANGE, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t8, p8_2, i, num_frames, WHITE, int(np.ceil(t_step * RENDER_FPS))
                )

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Entry point  —  pick which scene to render by setting SCENE below
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from manim import config

    # ── Choose scene ──────────────────────────────────────────────────
    SCENE = "DoublePendulumOverlay"
    # SCENE = "DoublePendulumSideBySide"
    # SCENE = "DoublePendulumEnergy"
    # ──────────────────────────────────────────────────────────────────

    config.progress_bar = "display"
    config.quality = "medium_quality"  # -qm  (30 fps)
    config.preview = True
    config.disable_caching = True

    scene_class = {
        "DoublePendulumEnergy": DoublePendulumEnergy,
        "DoublePendulumSideBySide": DoublePendulumSideBySide,
        "DoublePendulumOverlay": DoublePendulumOverlay,
    }[SCENE]

    scene = scene_class()
    scene.render()
