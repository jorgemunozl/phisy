"""
Visualisations for the double pendulum.

Available scenes (set via SCENE variable at the bottom of this file):
  - DoublePendulumEnergy    : two overlaid pendulums + live energy-vs-time plots
  - DoublePendulumSideBySide: RK4 vs RK8 side by side
  - DoublePendulumOverlay   : RK4 vs RK8 superimposed on the same pivot
"""

from pathlib import Path

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Axes,
    DashedLine,
    Dot,
    Line,
    Scene,
    Text,
    VGroup,
    VMobject,
)

# Physical parameters — kept in sync with main.py
from physi.double_pendulum.double_pendulum import (
    L1,
    L2,
    M1,
    M2,
    calc_kinetic_energy,
    calc_potential_energy,
)

# ── Settings ──────────────────────────────────────────────────────────
# RK4_PATH = Path(__file__).parent / "data" / "40s_10e4_pi_2_pi_6_1.0_0.0_rk4.npy"
# RK8_PATH = Path(__file__).parent / "data" / "40s_10e4_pi_2_pi_6_1.0_0.0_rk8.npy"

RK4_PATH = Path(__file__).parent / "data" / "10s_10e2_pi_2_pi_6_1.0_0.0_rk4.npy"
RK8_PATH = Path(__file__).parent / "data" / "10s_10e2_pi_2_pi_6_1.0_0.0_rk8.npy"


# RK4_ADAPTIVE_PATH = (
#    Path(__file__).parent / "data" / "40s_10e5_pi_2_pi_2_1.0_-1.0_rk4_adaptive.npy"
# )

RK4_ADAPTIVE_PATH = RK4_PATH

ANIM_DURATION = 10.0  # desired total playback duration in scene-seconds
RENDER_FPS = 15  # must match config.quality below (ql→15, qm→30, qh/qk→60)
SUBSAMPLING = 1  # extra trail thinning (1 = every rendered step)
SIM_TIME = 4.0  # total simulation time (must match solver preset)


# ── Shared helpers ────────────────────────────────────────────────────


def load_aligned_trajectories():
    """Load RK4 and RK8 data, returning (traj_rk4, traj_rk8, num_frames)
    trimmed to the same number of time steps."""
    traj_rk4 = np.load(RK4_PATH)  # (100001, 4)
    traj_rk4_adaptive = np.load(RK4_ADAPTIVE_PATH)  # (100001, 4)
    traj_rk8 = np.load(RK8_PATH)  # (100000, 4)
    num_frames = min(len(traj_rk4), len(traj_rk4_adaptive), len(traj_rk8))
    return (
        traj_rk4[:num_frames],
        traj_rk4_adaptive[:num_frames],
        traj_rk8[:num_frames],
        num_frames,
    )


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
#  Scene 1 — two overlaid pendulums + live energy-vs-time line plots
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumEnergy(Scene):
    """
    Left half  → two overlaid pendulums (RK4 blue/green, RK8 red/orange).
    Top right  → live line plot: total energy vs time (RK4).
    Bottom right → live line plot: total energy vs time (RK8).
    """

    def construct(self):
        traj_rk4, _, traj_rk8, num_frames = load_aligned_trajectories()

        # ── Pendulum setup (left half) ─────────────────────────────────
        _, _, x2_4, y2_4 = cartesian_from_traj(traj_rk4)
        _, _, x2_8, y2_8 = cartesian_from_traj(traj_rk8)
        scale = compute_scale(
            np.concatenate([x2_4, x2_8]),
            np.concatenate([y2_4, y2_8]),
            screen_half_width=3.0,
        )
        offset = np.array([-3.5, 0.0, 0.0])

        # RK4 pendulum (blue/green, yellow trail)
        pts4_p1, pts4_p2, piv4, r4_1, r4_2, m4_1, m4_2, t4 = build_pendulum_objects(
            traj_rk4,
            offset,
            rod1_color=BLUE,
            rod2_color=GREEN,
            mass1_color=BLUE,
            mass2_color=GREEN,
            trail_color=YELLOW,
            scale=scale,
        )
        # RK8 pendulum (red/orange, white trail, slightly translucent)
        pts8_p1, pts8_p2, piv8, r8_1, r8_2, m8_1, m8_2, t8 = build_pendulum_objects(
            traj_rk8,
            offset,
            rod1_color=RED,
            rod2_color=ORANGE,
            mass1_color=RED,
            mass2_color=ORANGE,
            trail_color=WHITE,
            scale=scale,
        )
        for obj in (r8_1, r8_2, m8_1, m8_2):
            obj.set_opacity(0.7)

        # Legend (top-left corner)
        legend = VGroup(
            Dot(radius=0.08, color=BLUE).move_to(np.array([-6.5, 3.5, 0.0])),
            Text("RK4", font_size=18, color=BLUE).next_to(
                np.array([-6.5, 3.5, 0.0]), RIGHT, buff=0.15
            ),
            Dot(radius=0.08, color=RED).move_to(np.array([-6.5, 3.1, 0.0])),
            Text("RK8", font_size=18, color=RED).next_to(
                np.array([-6.5, 3.1, 0.0]), RIGHT, buff=0.15
            ),
        )

        # ── Energy arrays ──────────────────────────────────────────────
        def total_energy(traj):
            return calc_kinetic_energy(
                M1, L1, traj[:, 1], M2, L2, traj[:, 3], traj[:, 0] - traj[:, 2]
            ) + calc_potential_energy(M1, L1, M2, L2, traj[:, 0], traj[:, 2])

        e_rk4 = total_energy(traj_rk4)
        e_rk8 = total_energy(traj_rk8)
        e_init = e_rk4[0]

        # Time axis
        t_arr = np.linspace(0, SIM_TIME, num_frames)

        # Shared y-range for both plots (pad ±0.5 around the data range)
        e_all = np.concatenate([e_rk4, e_rk8])
        e_min = float(e_all.min()) - 0.5
        e_max = float(e_all.max()) + 0.5

        # ── Axes (right half) ──────────────────────────────────────────
        AX_W, AX_H = 3.8, 1.8

        ax_rk4 = Axes(
            x_range=[0, SIM_TIME, SIM_TIME / 4],
            y_range=[e_min, e_max, (e_max - e_min) / 4],
            x_length=AX_W,
            y_length=AX_H,
            axis_config={"color": WHITE, "font_size": 14},
            tips=False,
        )
        ax_rk4.move_to(np.array([3.0, 2.0, 0.0]))

        ax_rk8 = Axes(
            x_range=[0, SIM_TIME, SIM_TIME / 4],
            y_range=[e_min, e_max, (e_max - e_min) / 4],
            x_length=AX_W,
            y_length=AX_H,
            axis_config={"color": WHITE, "font_size": 14},
            tips=False,
        )
        ax_rk8.move_to(np.array([3.0, -2.0, 0.0]))

        # Labels
        rk4_title = Text("RK4 — Total Energy", font_size=20, color=BLUE)
        rk4_title.next_to(ax_rk4, UP, buff=0.12)
        rk8_title = Text("RK8 — Total Energy", font_size=20, color=RED)
        rk8_title.next_to(ax_rk8, UP, buff=0.12)

        t_label = Text("t (s)", font_size=16, color=WHITE)
        t_label_rk4 = t_label.copy().next_to(ax_rk4, DOWN, buff=0.15)
        t_label_rk8 = t_label.copy().next_to(ax_rk8, DOWN, buff=0.15)

        # Horizontal reference line at initial energy
        def make_ref_line(axes):
            x0, y0, _ = axes.coords_to_point(0, e_init)
            x1, y1, _ = axes.coords_to_point(SIM_TIME, e_init)
            return DashedLine(
                np.array([x0, y0, 0.0]),
                np.array([x1, y1, 0.0]),
                color=WHITE,
                stroke_opacity=0.35,
                dash_length=0.1,
            )

        ref_rk4 = make_ref_line(ax_rk4)
        ref_rk8 = make_ref_line(ax_rk8)

        # Energy curves (grown frame by frame)
        graph_rk4 = VMobject(color=BLUE, stroke_width=2)
        graph_rk4.start_new_path(ax_rk4.coords_to_point(0, e_rk4[0]))

        graph_rk8 = VMobject(color=RED, stroke_width=2)
        graph_rk8.start_new_path(ax_rk8.coords_to_point(0, e_rk8[0]))

        # Moving dots at the current energy value
        dot_rk4 = Dot(ax_rk4.coords_to_point(0, e_rk4[0]), color=BLUE, radius=0.06)
        dot_rk8 = Dot(ax_rk8.coords_to_point(0, e_rk8[0]), color=RED, radius=0.06)

        # Vertical divider
        divider = Line(
            np.array([-0.6, -4.2, 0.0]),
            np.array([-0.6, 4.2, 0.0]),
            color=WHITE,
            stroke_width=1.0,
            stroke_opacity=0.35,
        )

        # ── Add to scene ───────────────────────────────────────────────
        self.add(
            piv4,
            t4,
            t8,
            r4_1,
            r4_2,
            m4_1,
            m4_2,
            r8_1,
            r8_2,
            m8_1,
            m8_2,
            legend,
            divider,
            ax_rk4,
            ax_rk8,
            rk4_title,
            rk8_title,
            t_label_rk4,
            t_label_rk8,
            graph_rk4,
            graph_rk8,
            dot_rk4,
            dot_rk8,
            ref_rk4,
            ref_rk8,
        )
        self.wait(0.3)

        # ── Animation loop ─────────────────────────────────────────────
        indices, t_step = compute_stride(num_frames)
        for i in indices:
            # --- Pendulum positions ---
            p4_1, p4_2 = pts4_p1[i], pts4_p2[i]
            p8_1, p8_2 = pts8_p1[i], pts8_p2[i]

            m4_1.move_to(p4_1)
            m4_2.move_to(p4_2)
            r4_1.become(Line(piv4.get_center(), p4_1, color=BLUE, stroke_width=6))
            r4_2.become(Line(p4_1, p4_2, color=GREEN, stroke_width=5))

            m8_1.move_to(p8_1)
            m8_2.move_to(p8_2)
            r8_1.become(Line(piv4.get_center(), p8_1, color=RED, stroke_width=6))
            r8_2.become(Line(p8_1, p8_2, color=ORANGE, stroke_width=5))

            # --- Trails ---
            stride_a = int(np.ceil(t_step * RENDER_FPS))
            if i % (stride_a * SUBSAMPLING) == 0:
                add_trail_dot(t4, p4_2, i, num_frames, YELLOW, stride_a)
                add_trail_dot(t8, p8_2, i, num_frames, WHITE, stride_a)

            # --- Energy graphs: extend line to current point ---
            pt_rk4 = ax_rk4.coords_to_point(t_arr[i], e_rk4[i])
            pt_rk8 = ax_rk8.coords_to_point(t_arr[i], e_rk8[i])

            graph_rk4.add_line_to(pt_rk4)
            dot_rk4.move_to(pt_rk4)

            graph_rk8.add_line_to(pt_rk8)
            dot_rk8.move_to(pt_rk8)

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 2 — RK4 vs RK8 side by side
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumSideBySide(Scene):
    """
    Left side  → RK4 solver
    Right side → DOP853 (RK8) solver

    Note: use ``DoublePendulumThreeWay`` to see all three solvers together.
    """

    def construct(self):
        traj_rk4, traj_rk4_adaptive, traj_rk8, num_frames = load_aligned_trajectories()

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
#  Scene 3 — RK4 vs RK4 Adaptive vs RK8, side by side
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumThreeWay(Scene):
    """
    Three pendulums side by side:

      Left   → RK4 (fixed step)
      Centre → RK4 Adaptive
      Right  → DOP853 (RK8)

    All start from the same initial conditions and evolve in lock-step
    so you can watch the trajectories diverge due to numerical errors.
    """

    def construct(self):
        traj_rk4, traj_rk4_adaptive, traj_rk8, num_frames = load_aligned_trajectories()

        # Unified scale based on the combined extent of all three
        _, _, x2_4, y2_4 = cartesian_from_traj(traj_rk4)
        _, _, x2_a, y2_a = cartesian_from_traj(traj_rk4_adaptive)
        _, _, x2_8, y2_8 = cartesian_from_traj(traj_rk8)

        scale = compute_scale(
            np.concatenate([x2_4, x2_a, x2_8]),
            np.concatenate([y2_4, y2_a, y2_8]),
            screen_half_width=2.2,
        )

        # Three column centres
        offset_L = np.array([-5.0, 0.0, 0.0])
        offset_C = np.array([0.0, 0.0, 0.0])
        offset_R = np.array([5.0, 0.0, 0.0])

        # ── RK4 (blue/green, yellow trail) ───────────────────────────
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

        # ── RK4 Adaptive (purple/pink, white trail) ──────────────────
        PURPLE = "#9B59B6"
        PINK = "#FF69B4"
        (
            pts_a_p1,
            pts_a_p2,
            piv_a,
            r_a_1,
            r_a_2,
            m_a_1,
            m_a_2,
            t_a,
        ) = build_pendulum_objects(
            traj_rk4_adaptive,
            offset_C,
            rod1_color=PURPLE,
            rod2_color=PINK,
            mass1_color=PURPLE,
            mass2_color=PINK,
            trail_color=WHITE,
            scale=scale,
        )

        # ── RK8 / DOP853 (red/orange, white trail) ──────────────────
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

        # ── Labels ───────────────────────────────────────────────────
        label_y = 3.6
        lbl4 = Text("RK4", font_size=24, color=BLUE).move_to(
            np.array([offset_L[0], label_y, 0.0])
        )
        lbl_a = Text("RK4 Adaptive", font_size=24, color=PURPLE).move_to(
            np.array([offset_C[0], label_y, 0.0])
        )
        lbl8 = Text("DOP853", font_size=24, color=RED).move_to(
            np.array([offset_R[0], label_y, 0.0])
        )
        caption = Text(
            "Same IC — three solvers, three trajectories",
            font_size=20,
            color=WHITE,
        ).move_to(np.array([0.0, -3.6, 0.0]))

        # Vertical dividers
        div_L = Line(
            np.array([-2.5, -4.0, 0.0]),
            np.array([-2.5, 4.0, 0.0]),
            color=WHITE,
            stroke_width=0.8,
            stroke_opacity=0.25,
        )
        div_R = Line(
            np.array([2.5, -4.0, 0.0]),
            np.array([2.5, 4.0, 0.0]),
            color=WHITE,
            stroke_width=0.8,
            stroke_opacity=0.25,
        )

        self.add(
            piv4,
            t4,
            r4_1,
            r4_2,
            m4_1,
            m4_2,
            piv_a,
            t_a,
            r_a_1,
            r_a_2,
            m_a_1,
            m_a_2,
            piv8,
            t8,
            r8_1,
            r8_2,
            m8_1,
            m8_2,
            div_L,
            div_R,
            lbl4,
            lbl_a,
            lbl8,
            caption,
        )
        self.wait(0.3)

        indices, t_step = compute_stride(num_frames)
        for i in indices:
            p4_1, p4_2 = pts4_p1[i], pts4_p2[i]
            p_a_1, p_a_2 = pts_a_p1[i], pts_a_p2[i]
            p8_1, p8_2 = pts8_p1[i], pts8_p2[i]

            # ── RK4 ──
            m4_1.move_to(p4_1)
            m4_2.move_to(p4_2)
            r4_1.become(Line(piv4.get_center(), p4_1, color=BLUE, stroke_width=6))
            r4_2.become(Line(p4_1, p4_2, color=GREEN, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t4,
                    p4_2,
                    i,
                    num_frames,
                    YELLOW,
                    int(np.ceil(t_step * RENDER_FPS)),
                )

            # ── RK4 Adaptive ──
            m_a_1.move_to(p_a_1)
            m_a_2.move_to(p_a_2)
            r_a_1.become(Line(piv_a.get_center(), p_a_1, color=PURPLE, stroke_width=6))
            r_a_2.become(Line(p_a_1, p_a_2, color=PINK, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t_a,
                    p_a_2,
                    i,
                    num_frames,
                    WHITE,
                    int(np.ceil(t_step * RENDER_FPS)),
                )

            # ── RK8 ──
            m8_1.move_to(p8_1)
            m8_2.move_to(p8_2)
            r8_1.become(Line(piv8.get_center(), p8_1, color=RED, stroke_width=6))
            r8_2.become(Line(p8_1, p8_2, color=ORANGE, stroke_width=5))
            if i % int(np.ceil(t_step * RENDER_FPS) * SUBSAMPLING) == 0:
                add_trail_dot(
                    t8,
                    p8_2,
                    i,
                    num_frames,
                    WHITE,
                    int(np.ceil(t_step * RENDER_FPS)),
                )

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 4 — RK4 vs RK8 superimposed (transposed) on the same pivot
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumOverlay(Scene):
    """
    Both RK4 and RK8 pendulums drawn on top of each other at the same
    pivot point.  They start in sync and gradually diverge — the clearest
    visualisation of chaotic sensitivity to numerical errors.
    """

    def construct(self):
        traj_rk4, _, traj_rk8, num_frames = load_aligned_trajectories()

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
    # SCENE = "DoublePendulumThreeWay"
    # SCENE = "DoublePendulumOverlay"
    # SCENE = "DoublePendulumSideBySide"
    SCENE = "DoublePendulumEnergy"
    # ──────────────────────────────────────────────────────────────────

    config.progress_bar = "display"
    config.quality = "low_quality"  # -qh  (60 fps)
    config.preview = True
    config.disable_caching = False

    scene_class = {
        "DoublePendulumEnergy": DoublePendulumEnergy,
        "DoublePendulumSideBySide": DoublePendulumSideBySide,
        "DoublePendulumOverlay": DoublePendulumOverlay,
        "DoublePendulumThreeWay": DoublePendulumThreeWay,
    }[SCENE]

    scene = scene_class()
    scene.render()
