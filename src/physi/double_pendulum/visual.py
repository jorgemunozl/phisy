"""
Visualisations for the double pendulum.

Available scenes (set via SCENE variable at the bottom of this file):
  - DoublePendulumEnergy    : two overlaid pendulums + live energy-vs-time plots
  - DoublePendulumSideBySide: RK4 vs RK8 side by side
  - DoublePendulumOverlay   : RK4 vs RK8 superimposed on the same pivot
"""

from pathlib import Path

import manim
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
    # MathTex,  # (temporarily disabled)
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

RK4_PATH = Path(__file__).parent / "data" / "50s_10e3_pi_2_pi_6_1.0_0.0_rk4.npy"
RK8_PATH = Path(__file__).parent / "data" / "50s_10e3_pi_2_pi_6_1.0_0.0_rk8.npy"


# RK4_ADAPTIVE_PATH = (
#    Path(__file__).parent / "data" / "40s_10e5_pi_2_pi_2_1.0_-1.0_rk4_adaptive.npy"
# )

RK4_ADAPTIVE_PATH = RK4_PATH

# ── Adaptive step-size paths (hard preset, 50 s) ──────────────────────
DATA = Path(__file__).parent / "data"
RK45_ADAPTIVE_PATH = DATA / "50s_10e3_pi_2_pi_6_1.0_0.0_rk45_adaptive.npy"
RK45_STEPS_PATH = DATA / "50s_10e3_pi_2_pi_6_1.0_0.0_rk45_adaptive_steps.npz"
DOP853_ADAPTIVE_PATH = DATA / "50s_10e3_pi_2_pi_6_1.0_0.0_dop853_adaptive.npy"
DOP853_STEPS_PATH = DATA / "50s_10e3_pi_2_pi_6_1.0_0.0_dop853_adaptive_steps.npz"
DOP853_LOOSE_NPY = DATA / "50s_10e3_pi_2_pi_6_1.0_0.0_dop853_loose.npy"
DOP853_LOOSE_STEPS = DATA / "50s_10e3_pi_2_pi_6_1.0_0.0_dop853_loose_steps.npz"

RENDER_FPS = 60  # must match config.quality below (ql→15, qm→30, qh/qk→60)
SUBSAMPLING = 1  # extra trail thinning (1 = every rendered step)
SIM_TIME = 50.0  # total duration of the saved simulation data (seconds)
ANIM_DURATION = 2.5  # seconds of simulation to show — also sets video length

E_MARGIN_rk4 = 1e-8  # For rk4
E_MARGIN_rk8 = 1e-13  # For rk8


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


def compute_stride(num_frames, duration=None):
    """Return (indices, time_per_step) for frame-by-frame playback.

    ``duration`` is the simulation time covered by ``num_frames``.
    Defaults to ANIM_DURATION so scenes that slice their data already
    get the right timing without having to pass anything extra.
    """
    if duration is None:
        duration = ANIM_DURATION
    min_wait = 1.0 / RENDER_FPS
    ideal_wait = duration / max(num_frames - 1, 1)
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
            rod2_color=BLUE,
            mass1_color=BLUE,
            mass2_color=BLUE,
            trail_color=BLUE,
            scale=scale,
        )
        # RK8 pendulum (all red, slightly translucent)
        pts8_p1, pts8_p2, piv8, r8_1, r8_2, m8_1, m8_2, t8 = build_pendulum_objects(
            traj_rk8,
            offset,
            rod1_color=RED,
            rod2_color=RED,
            mass1_color=RED,
            mass2_color=RED,
            trail_color=RED,
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

        # Plot delta E = E(t) - E0 so the y-axis is centred at 0.
        # This avoids floating-point precision issues in Manim when the
        # absolute values (e.g. -7.5) dwarf the tiny window (e.g. 1e-8).
        de_rk4 = e_rk4 - e_init
        de_rk8 = e_rk8 - e_init

        # Normalise: plot ΔE / margin so both axes are always [-1, 1].
        # This means Manim never has to render tick labels like "5e-13" —
        # it always renders plain numbers like -1, -0.5, 0, 0.5, 1.
        # The actual scale is shown in the plot title instead.
        norm_rk4 = de_rk4 / E_MARGIN_rk4
        norm_rk8 = de_rk8 / E_MARGIN_rk8

        # Time axis
        t_arr = np.linspace(0, SIM_TIME, num_frames)

        # ── Axes (right half) — fixed [-1, 1] range regardless of margin
        AX_W, AX_H = 3.8, 1.8
        FIXED_Y = [-1.0, 1.0, 0.5]

        ax_rk4 = Axes(
            x_range=[0, SIM_TIME, SIM_TIME / 4],
            y_range=FIXED_Y,
            x_length=AX_W,
            y_length=AX_H,
            axis_config={"color": WHITE, "font_size": 14},
            tips=False,
        )
        ax_rk4.move_to(np.array([3.0, 2.0, 0.0]))

        ax_rk8 = Axes(
            x_range=[0, SIM_TIME, SIM_TIME / 4],
            y_range=FIXED_Y,
            x_length=AX_W,
            y_length=AX_H,
            axis_config={"color": WHITE, "font_size": 14},
            tips=False,
        )
        ax_rk8.move_to(np.array([3.0, -2.0, 0.0]))

        # Titles include the scale so the viewer knows the real units
        rk4_title = Text(
            f"RK4  ΔE / {E_MARGIN_rk4:.0e}", font_size=18, color=BLUE
        ).next_to(ax_rk4, UP, buff=0.12)
        rk8_title = Text(
            f"RK8  ΔE / {E_MARGIN_rk8:.0e}", font_size=18, color=RED
        ).next_to(ax_rk8, UP, buff=0.12)

        t_label = Text("t (s)", font_size=16, color=WHITE)
        t_label_rk4 = t_label.copy().next_to(ax_rk4, DOWN, buff=0.15)
        t_label_rk8 = t_label.copy().next_to(ax_rk8, DOWN, buff=0.15)

        # Zero reference line (ΔE = 0 ≡ perfect conservation)
        def make_ref_line(axes):
            x0, y0, _ = axes.coords_to_point(0, 0)
            x1, y1, _ = axes.coords_to_point(SIM_TIME, 0)
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
        graph_rk4.start_new_path(ax_rk4.coords_to_point(0, norm_rk4[0]))

        graph_rk8 = VMobject(color=RED, stroke_width=2)
        graph_rk8.start_new_path(ax_rk8.coords_to_point(0, norm_rk8[0]))

        # Moving dots at the current normalised ΔE value
        dot_rk4 = Dot(ax_rk4.coords_to_point(0, norm_rk4[0]), color=BLUE, radius=0.06)
        dot_rk8 = Dot(ax_rk8.coords_to_point(0, norm_rk8[0]), color=RED, radius=0.06)

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
            # margin_label_rk4,
            # margin_label_rk8,
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
            r4_2.become(Line(p4_1, p4_2, color=BLUE, stroke_width=5))

            m8_1.move_to(p8_1)
            m8_2.move_to(p8_2)
            r8_1.become(Line(piv4.get_center(), p8_1, color=RED, stroke_width=6))
            r8_2.become(Line(p8_1, p8_2, color=RED, stroke_width=5))

            # --- Trails ---
            stride_a = int(np.ceil(t_step * RENDER_FPS))
            if i % (stride_a * SUBSAMPLING) == 0:
                add_trail_dot(t4, p4_2, i, num_frames, BLUE, stride_a)
                add_trail_dot(t8, p8_2, i, num_frames, RED, stride_a)

            # --- Energy graphs: extend line to current normalised ΔE point ---
            pt_rk4 = ax_rk4.coords_to_point(t_arr[i], norm_rk4[i])
            pt_rk8 = ax_rk8.coords_to_point(t_arr[i], norm_rk8[i])

            graph_rk4.add_line_to(pt_rk4)
            dot_rk4.move_to(pt_rk4)

            graph_rk8.add_line_to(pt_rk8)
            dot_rk8.move_to(pt_rk8)

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 2 — RK45 vs DOP853: live step-size h(t) plot
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumStepSize(Scene):
    """
    Left  → RK45 (blue) and DOP853 (red) pendulums overlaid.
    Top right    → live h(t) plot for RK45.
    Bottom right → live h(t) plot for DOP853.
    """

    def construct(self):
        # ── Load data ──────────────────────────────────────────────────
        traj_rk45 = np.load(RK45_ADAPTIVE_PATH)  # uniform grid (N, 4)
        traj_dop853 = np.load(DOP853_ADAPTIVE_PATH)  # uniform grid (N, 4)

        steps_rk45 = np.load(RK45_STEPS_PATH)
        steps_dop853 = np.load(DOP853_STEPS_PATH)

        t45 = steps_rk45["t"]  # (N45,)
        h45 = steps_rk45["h"]  # (N45-1,)  step sizes
        t853 = steps_dop853["t"]  # (N853,)
        h853 = steps_dop853["h"]  # (N853-1,)

        total_frames = min(len(traj_rk45), len(traj_dop853))

        # Slice to ANIM_DURATION seconds so the video runs at 1:1 simulation speed
        n_use = max(2, int(ANIM_DURATION / SIM_TIME * total_frames))
        traj_rk45 = traj_rk45[:n_use]
        traj_dop853 = traj_dop853[:n_use]
        num_frames = n_use
        t_uniform = np.linspace(0, ANIM_DURATION, num_frames)

        # Slice step data to the same window
        mask45 = t45 <= ANIM_DURATION
        mask853 = t853 <= ANIM_DURATION
        t45 = t45[mask45]
        t853 = t853[mask853]
        h45 = np.diff(t45)  # recompute after slicing
        h853 = np.diff(t853)

        # ── Pendulums (left half) ──────────────────────────────────────
        _, _, x2_45, y2_45 = cartesian_from_traj(traj_rk45)
        _, _, x2_853, y2_853 = cartesian_from_traj(traj_dop853)
        scale = compute_scale(
            np.concatenate([x2_45, x2_853]),
            np.concatenate([y2_45, y2_853]),
            screen_half_width=3.0,
        )
        offset = np.array([-3.5, 0.0, 0.0])

        pts45_p1, pts45_p2, piv45, r45_1, r45_2, m45_1, m45_2, t4 = (
            build_pendulum_objects(
                traj_rk45,
                offset,
                rod1_color=BLUE,
                rod2_color=BLUE,
                mass1_color=BLUE,
                mass2_color=BLUE,
                trail_color=BLUE,
                scale=scale,
            )
        )
        pts853_p1, pts853_p2, piv853, r853_1, r853_2, m853_1, m853_2, t8 = (
            build_pendulum_objects(
                traj_dop853,
                offset,
                rod1_color=RED,
                rod2_color=RED,
                mass1_color=RED,
                mass2_color=RED,
                trail_color=RED,
                scale=scale,
            )
        )
        for obj in (r853_1, r853_2, m853_1, m853_2):
            obj.set_opacity(0.7)

        legend = VGroup(
            Dot(radius=0.08, color=BLUE).move_to(np.array([-6.5, 3.5, 0.0])),
            Text("RK45", font_size=18, color=BLUE).next_to(
                np.array([-6.5, 3.5, 0.0]), RIGHT, buff=0.15
            ),
            Dot(radius=0.08, color=RED).move_to(np.array([-6.5, 3.1, 0.0])),
            Text("DOP853", font_size=18, color=RED).next_to(
                np.array([-6.5, 3.1, 0.0]), RIGHT, buff=0.15
            ),
        )

        # ── Step-size axes (right half) ────────────────────────────────
        AX_W, AX_H = 3.8, 1.8

        def make_h_axes(max_h, center_y, color):
            y_top = float(max_h) * 1.3
            y_step = y_top / 4
            ax = Axes(
                x_range=[0, ANIM_DURATION, ANIM_DURATION / 4],
                y_range=[0, y_top, y_step],
                x_length=AX_W,
                y_length=AX_H,
                axis_config={"color": WHITE, "font_size": 14},
                tips=False,
            )
            ax.move_to(np.array([3.0, center_y, 0.0]))
            return ax

        ax_rk45 = make_h_axes(h45.max(), 2.0, BLUE)
        ax_dop853 = make_h_axes(h853.max(), -2.0, RED)

        rk45_title = Text("RK45 — step size h(t)", font_size=18, color=BLUE)
        rk45_title.next_to(ax_rk45, UP, buff=0.12)
        dop853_title = Text("DOP853 — step size h(t)", font_size=18, color=RED)
        dop853_title.next_to(ax_dop853, UP, buff=0.12)

        t_lbl = Text("t (s)", font_size=16, color=WHITE)
        t_lbl_45 = t_lbl.copy().next_to(ax_rk45, DOWN, buff=0.15)
        t_lbl_853 = t_lbl.copy().next_to(ax_dop853, DOWN, buff=0.15)

        # ── Growing line graphs ────────────────────────────────────────
        # Each step point: (t45[i+1], h45[i]) — time AFTER the step, size used.
        graph_rk45 = VMobject(color=BLUE, stroke_width=2)
        graph_rk45.start_new_path(ax_rk45.coords_to_point(t45[1], h45[0]))

        graph_dop853 = VMobject(color=RED, stroke_width=2)
        graph_dop853.start_new_path(ax_dop853.coords_to_point(t853[1], h853[0]))

        dot_rk45 = Dot(ax_rk45.coords_to_point(t45[1], h45[0]), color=BLUE, radius=0.06)
        dot_dop853 = Dot(
            ax_dop853.coords_to_point(t853[1], h853[0]), color=RED, radius=0.06
        )

        divider = Line(
            np.array([-0.6, -4.2, 0.0]),
            np.array([-0.6, 4.2, 0.0]),
            color=WHITE,
            stroke_width=1.0,
            stroke_opacity=0.35,
        )

        # ── Add everything to scene ────────────────────────────────────
        self.add(
            piv45,
            t4,
            t8,
            r45_1,
            r45_2,
            m45_1,
            m45_2,
            r853_1,
            r853_2,
            m853_1,
            m853_2,
            legend,
            divider,
            ax_rk45,
            ax_dop853,
            rk45_title,
            dop853_title,
            t_lbl_45,
            t_lbl_853,
            graph_rk45,
            graph_dop853,
            dot_rk45,
            dot_dop853,
        )
        self.wait(0.3)

        # ── Animation loop ─────────────────────────────────────────────
        indices, t_step = compute_stride(num_frames)
        ptr45 = 0  # next index into h45 to reveal
        ptr853 = 0  # next index into h853 to reveal

        for i in indices:
            t_curr = t_uniform[i]

            # --- Pendulums ---
            p45_1, p45_2 = pts45_p1[i], pts45_p2[i]
            p853_1, p853_2 = pts853_p1[i], pts853_p2[i]

            m45_1.move_to(p45_1)
            m45_2.move_to(p45_2)
            r45_1.become(Line(piv45.get_center(), p45_1, color=BLUE, stroke_width=6))
            r45_2.become(Line(p45_1, p45_2, color=BLUE, stroke_width=5))

            m853_1.move_to(p853_1)
            m853_2.move_to(p853_2)
            r853_1.become(Line(piv45.get_center(), p853_1, color=RED, stroke_width=6))
            r853_2.become(Line(p853_1, p853_2, color=RED, stroke_width=5))

            # --- Trails ---
            stride_a = int(np.ceil(t_step * RENDER_FPS))
            if i % (stride_a * SUBSAMPLING) == 0:
                add_trail_dot(t4, p45_2, i, num_frames, BLUE, stride_a)
                add_trail_dot(t8, p853_2, i, num_frames, RED, stride_a)

            # --- RK45 h(t) graph: reveal all steps up to t_curr ---
            while ptr45 < len(h45) and t45[ptr45 + 1] <= t_curr:
                pt = ax_rk45.coords_to_point(t45[ptr45 + 1], h45[ptr45])
                graph_rk45.add_line_to(pt)
                dot_rk45.move_to(pt)
                ptr45 += 1

            # --- DOP853 h(t) graph ---
            while ptr853 < len(h853) and t853[ptr853 + 1] <= t_curr:
                pt = ax_dop853.coords_to_point(t853[ptr853 + 1], h853[ptr853])
                graph_dop853.add_line_to(pt)
                dot_dop853.move_to(pt)
                ptr853 += 1

            self.wait(t_step)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 2b — raw adaptive steps (no interpolation)
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumRawSteps(Scene):
    """
    Raw adaptive steps — the pendulum jumps at solver step points.

    * Slow-motion playback (SLOW_MO) so the viewer can follow the dynamics.
    * Dynamic y-axis that zooms in/out following recent step-size variation.
    """

    SLOW_MO = 5  # multiplier → slower playback
    WINDOW_S = 5.0  # rolling look-back window (simulation seconds)
    Y_MARGIN = 1.5  # y_max = max_recent_h * Y_MARGIN

    def construct(self):
        # ── Load raw step data ─────────────────────────────────────────
        data45 = np.load(RK45_STEPS_PATH)
        data853 = np.load(DOP853_STEPS_PATH)

        t45 = data45["t"]
        u45 = data45["u"]
        t853 = data853["t"]
        u853 = data853["u"]

        # Slice both arrays to ANIM_DURATION
        mask45 = t45 <= ANIM_DURATION
        mask853 = t853 <= ANIM_DURATION
        t45, u45 = t45[mask45], u45[mask45]
        t853, u853 = t853[mask853], u853[mask853]
        h45 = np.diff(t45)
        h853 = np.diff(t853)

        # Merged time sequence: every moment either solver stepped
        all_times = np.union1d(t45[1:], t853[1:])
        n_merged = len(all_times)
        wait_time = (
            max(1.0 / RENDER_FPS, ANIM_DURATION / max(n_merged, 1)) * self.SLOW_MO
        )

        # ── Pendulums built from raw u ─────────────────────────────────
        _, _, x2_45, y2_45 = cartesian_from_traj(u45)
        _, _, x2_853, y2_853 = cartesian_from_traj(u853)
        scale = compute_scale(
            np.concatenate([x2_45, x2_853]),
            np.concatenate([y2_45, y2_853]),
            screen_half_width=3.0,
        )
        offset = np.array([-3.5, 0.0, 0.0])

        pts45_p1, pts45_p2, piv45, r45_1, r45_2, m45_1, m45_2, trail45 = (
            build_pendulum_objects(
                u45,
                offset,
                rod1_color=BLUE,
                rod2_color=BLUE,
                mass1_color=BLUE,
                mass2_color=BLUE,
                trail_color=BLUE,
                scale=scale,
            )
        )
        pts853_p1, pts853_p2, _, r853_1, r853_2, m853_1, m853_2, trail853 = (
            build_pendulum_objects(
                u853,
                offset,
                rod1_color=RED,
                rod2_color=RED,
                mass1_color=RED,
                mass2_color=RED,
                trail_color=RED,
                scale=scale,
            )
        )
        for obj in (r853_1, r853_2, m853_1, m853_2):
            obj.set_opacity(0.7)

        legend = VGroup(
            Dot(radius=0.08, color=BLUE).move_to(np.array([-6.5, 3.5, 0.0])),
            Text("RK45", font_size=18, color=BLUE).next_to(
                np.array([-6.5, 3.5, 0.0]), RIGHT, buff=0.15
            ),
            Dot(radius=0.08, color=RED).move_to(np.array([-6.5, 3.1, 0.0])),
            Text("DOP853", font_size=18, color=RED).next_to(
                np.array([-6.5, 3.1, 0.0]), RIGHT, buff=0.15
            ),
        )

        # ── Step-size axes (right half) — starting with full range ─────
        AX_W, AX_H = 3.8, 1.8

        def build_axes(ymax, center_y):
            y_top = float(max(ymax, 1e-12)) * self.Y_MARGIN
            y_step = y_top / 4
            return Axes(
                x_range=[0, ANIM_DURATION, ANIM_DURATION / 4],
                y_range=[0, y_top, y_step],
                x_length=AX_W,
                y_length=AX_H,
                axis_config={"color": WHITE, "font_size": 14},
                tips=False,
            ).move_to(np.array([3.0, center_y, 0.0]))

        ax_rk45 = build_axes(h45.max(), 2.0)
        ax_dop853 = build_axes(h853.max(), -2.0)

        rk45_title = Text("RK45 — h(t) [raw]", font_size=18, color=BLUE).next_to(
            ax_rk45, UP, buff=0.12
        )
        dop853_title = Text("DOP853 — h(t) [raw]", font_size=18, color=RED).next_to(
            ax_dop853, UP, buff=0.12
        )
        t_lbl = Text("t (s)", font_size=16, color=WHITE)
        t_lbl_45 = t_lbl.copy().next_to(ax_rk45, DOWN, buff=0.15)
        t_lbl_853 = t_lbl.copy().next_to(ax_dop853, DOWN, buff=0.15)

        graph_rk45 = VMobject(color=BLUE, stroke_width=2)
        graph_rk45.start_new_path(ax_rk45.coords_to_point(t45[1], h45[0]))
        graph_dop853 = VMobject(color=RED, stroke_width=2)
        graph_dop853.start_new_path(ax_dop853.coords_to_point(t853[1], h853[0]))

        dot_rk45 = Dot(ax_rk45.coords_to_point(t45[1], h45[0]), color=BLUE, radius=0.06)
        dot_dop853 = Dot(
            ax_dop853.coords_to_point(t853[1], h853[0]), color=RED, radius=0.06
        )

        divider = Line(
            np.array([-0.6, -4.2, 0.0]),
            np.array([-0.6, 4.2, 0.0]),
            color=WHITE,
            stroke_width=1.0,
            stroke_opacity=0.35,
        )

        self.add(
            piv45,
            trail45,
            trail853,
            r45_1,
            r45_2,
            m45_1,
            m45_2,
            r853_1,
            r853_2,
            m853_1,
            m853_2,
            legend,
            divider,
            ax_rk45,
            ax_dop853,
            rk45_title,
            dop853_title,
            t_lbl_45,
            t_lbl_853,
            graph_rk45,
            graph_dop853,
            dot_rk45,
            dot_dop853,
        )
        self.wait(0.3)

        # ── Helper to rebuild a graph on new axes ──────────────────────
        def rebuild_graph_on(ax, pts_t, pts_h, color):
            if len(pts_t) == 0:
                g = VMobject(color=color, stroke_width=2)
                return g
            g = VMobject(color=color, stroke_width=2)
            g.start_new_path(ax.coords_to_point(pts_t[0], pts_h[0]))
            for j in range(1, len(pts_t)):
                g.add_line_to(ax.coords_to_point(pts_t[j], pts_h[j]))
            return g

        def maybe_update_axes(
            ax,
            title_obj,
            t_curr,
            color,
            center_y,
            revealed_t,
            revealed_h,
            graph_obj,
            dot_obj,
        ):
            """Rebuild axes + graph if the recent y-range drifts far."""
            if len(revealed_t) < 2:
                return ax, graph_obj, dot_obj

            # Max h in the look-back window
            in_window = np.where(
                (revealed_t >= t_curr - self.WINDOW_S) & (revealed_t <= t_curr)
            )[0]
            if len(in_window) == 0:
                return ax, graph_obj, dot_obj

            recent_max = float(revealed_h[in_window].max())
            new_ymax = recent_max * self.Y_MARGIN

            # Only update if the new y-range differs by > 25 %
            _, old_ymax, _ = ax.y_range
            old_ymax = float(old_ymax)
            if abs(new_ymax / max(old_ymax, 1e-15) - 1.0) < 0.25:
                return ax, graph_obj, dot_obj

            # Build new axes
            new_ax = build_axes(recent_max, center_y)
            new_graph = rebuild_graph_on(new_ax, revealed_t, revealed_h, color)
            new_dot = Dot(
                new_ax.coords_to_point(revealed_t[-1], revealed_h[-1]),
                color=color,
                radius=0.06,
            )

            # Replace in scene
            self.remove(ax, graph_obj, dot_obj)
            self.add(new_ax, new_graph, new_dot)
            # Reposition title
            title_obj.next_to(new_ax, UP, buff=0.12)
            return new_ax, new_graph, new_dot

        # ── Animation loop — driven by merged step times ─────────────────
        ptr45 = 0
        ptr853 = 0
        hptr45 = 0
        hptr853 = 0
        rev45_t = []
        rev45_h = []  # revealed points for RK45
        rev853_t = []
        rev853_h = []  # revealed points for DOP853
        trail_every = max(1, n_merged // 200)
        UPDATE_EVERY = max(1, n_merged // 30)  # check axes ~30 times

        for idx, t_curr in enumerate(all_times):
            # Advance each solver to the most recent step ≤ t_curr
            while ptr45 < len(t45) - 1 and t45[ptr45 + 1] <= t_curr:
                ptr45 += 1
            while ptr853 < len(t853) - 1 and t853[ptr853 + 1] <= t_curr:
                ptr853 += 1

            # Update pendulums
            p45_1, p45_2 = pts45_p1[ptr45], pts45_p2[ptr45]
            p853_1, p853_2 = pts853_p1[ptr853], pts853_p2[ptr853]

            m45_1.move_to(p45_1)
            m45_2.move_to(p45_2)
            r45_1.become(Line(piv45.get_center(), p45_1, color=BLUE, stroke_width=6))
            r45_2.become(Line(p45_1, p45_2, color=BLUE, stroke_width=5))

            m853_1.move_to(p853_1)
            m853_2.move_to(p853_2)
            r853_1.become(Line(piv45.get_center(), p853_1, color=RED, stroke_width=6))
            r853_2.become(Line(p853_1, p853_2, color=RED, stroke_width=5))

            # Trails
            if idx % trail_every == 0:
                add_trail_dot(trail45, p45_2, idx, n_merged, BLUE, trail_every)
                add_trail_dot(trail853, p853_2, idx, n_merged, RED, trail_every)

            # h(t) plots: reveal steps up to t_curr
            while hptr45 < len(h45) and t45[hptr45 + 1] <= t_curr:
                rev45_t.append(t45[hptr45 + 1])
                rev45_h.append(h45[hptr45])
                pt = ax_rk45.coords_to_point(t45[hptr45 + 1], h45[hptr45])
                graph_rk45.add_line_to(pt)
                dot_rk45.move_to(pt)
                hptr45 += 1

            while hptr853 < len(h853) and t853[hptr853 + 1] <= t_curr:
                rev853_t.append(t853[hptr853 + 1])
                rev853_h.append(h853[hptr853])
                pt = ax_dop853.coords_to_point(t853[hptr853 + 1], h853[hptr853])
                graph_dop853.add_line_to(pt)
                dot_dop853.move_to(pt)
                hptr853 += 1

            # Dynamic y-axis zoom — checked periodically
            if idx % UPDATE_EVERY == 0:
                ra = np.array(rev45_t)
                rh = np.array(rev45_h)
                ax_rk45, graph_rk45, dot_rk45 = maybe_update_axes(
                    ax_rk45,
                    rk45_title,
                    t_curr,
                    BLUE,
                    2.0,
                    ra,
                    rh,
                    graph_rk45,
                    dot_rk45,
                )
                ra = np.array(rev853_t)
                rh = np.array(rev853_h)
                ax_dop853, graph_dop853, dot_dop853 = maybe_update_axes(
                    ax_dop853,
                    dop853_title,
                    t_curr,
                    RED,
                    -2.0,
                    ra,
                    rh,
                    graph_dop853,
                    dot_dop853,
                )

            self.wait(wait_time)

        self.wait(0.01)


# ══════════════════════════════════════════════════════════════════════
#  Scene 3 — compute-speed visualisation (coloured trail by step size)
# ══════════════════════════════════════════════════════════════════════


class DoublePendulumComputeSpeed(Scene):
    """
    One DOP853 pendulum with looser tolerance (rtol=1e-4).
    Every solver step = one video frame → animation slows down
    automatically when h is small (hard region) and speeds up
    when h is large (easy region).

    Trail dots are coloured red (small h, intense compute)
    through yellow to blue (large h, cheap compute).
    """

    def construct(self):
        # ── Load raw step data (loose tolerance) ──────────────────────
        data = np.load(DOP853_LOOSE_STEPS)
        t_full = data["t"]  # (394,)
        u_full = data["u"]  # (394, 4)
        h_full = np.diff(t_full)  # (393,)

        # Keep the first ANIM_DURATION seconds
        mask = t_full <= ANIM_DURATION
        t = t_full[mask]
        u = u_full[mask]
        h_used = h_full[: len(t) - 1]  # (N-1,)
        n_steps = len(t)  # includes t=0

        # Handle no steps case
        if n_steps < 2:
            self.add(Text("No steps in animation window.", font_size=28))
            self.wait(1)
            return

        TARGET_VIDEO_S = ANIM_DURATION  # make video roughly this long
        wait_time = max(1.0 / RENDER_FPS, TARGET_VIDEO_S / (n_steps - 1))

        # ── Pendulum from raw u (full width, centered) ────────────────
        _, _, x2, y2 = cartesian_from_traj(u)
        scale = compute_scale(x2, y2, screen_half_width=5.5)
        offset = np.array([0.0, 0.0, 0.0])

        pts_p1, pts_p2, pivot, rod1, rod2, m1, m2, trail = build_pendulum_objects(
            u,
            offset,
            rod1_color=WHITE,
            rod2_color=WHITE,
            mass1_color=WHITE,
            mass2_color=WHITE,
            trail_color=WHITE,
            scale=scale,
        )

        # ── Colour mapping: log-scale h → red (small) to blue (large) ─
        h_log_min = np.log10(h_used.min())
        h_log_max = np.log10(h_used.max())

        def h_colour(h):
            """Red (h small) → yellow → blue (h large), RGBA."""
            tt = np.clip(
                (np.log10(h) - h_log_min) / (h_log_max - h_log_min + 1e-30),
                0.0,
                1.0,
            )
            return np.array([1.0 - tt, tt * 0.7, tt, 1.0])  # no green→yellow→blue

        # ── Step-size display (top-left) ──────────────────────────────
        h_label = Text("h = ...", font_size=22, color=WHITE)
        h_label.move_to(np.array([-5.8, 3.4, 0.0]))

        # Colour legend (top-right)
        bar_w, bar_h = 1.6, 0.12
        bar_left = np.array([4.5, 3.45, 0.0])

        # Small gradient bar
        n_bar = 40
        for j in range(n_bar):
            frac = j / (n_bar - 1)
            x = bar_left[0] + frac * bar_w
            col = np.array([1.0 - frac, frac * 0.7, frac, 1.0])
            seg = Line(
                np.array([x, bar_left[1] - bar_h / 2, 0.0]),
                np.array([x, bar_left[1] + bar_h / 2, 0.0]),
                color=manim.utils.color.rgb_to_color(col[:3]),
                stroke_width=3,
            )
            self.add(seg)

        Text("small h", font_size=13, color=RED).move_to(
            np.array([bar_left[0] - 0.05, bar_left[1] - 0.25, 0.0])
        )
        Text("large h", font_size=13, color=BLUE).move_to(
            np.array([bar_left[0] + bar_w + 0.05, bar_left[1] - 0.25, 0.0])
        )

        # ── Add to scene ──────────────────────────────────────────────
        self.add(pivot, trail, rod1, rod2, m1, m2, h_label)
        self.wait(0.3)

        # ── Animation loop — one video frame per solver step ──────────
        #  i=0 is the initial condition (already shown via wait above).
        #  We advance i from 1 to n_steps-1.

        for i in range(1, n_steps):
            p1, p2 = pts_p1[i], pts_p2[i]

            m1.move_to(p1)
            m2.move_to(p2)
            rod1.become(Line(pivot.get_center(), p1, color=WHITE, stroke_width=6))
            rod2.become(Line(p1, p2, color=WHITE, stroke_width=5))

            # Trail dot coloured by the step size that produced this position
            if i > 0:
                col_rgba = h_colour(h_used[i - 1])
                trail_dot = Dot(p2, radius=0.035 if i % 3 == 0 else 0.025)
                trail_dot.set_color(manim.utils.color.rgb_to_color(col_rgba[:3]))
                trail_dot.set_opacity(0.2 + 0.6 * (i / (n_steps - 1)))
                trail.add(trail_dot)

            # Show current h
            h_curr = h_used[i - 1] if i > 0 else 0.0
            new_label = Text(
                f"h = {h_curr:.4f}",
                font_size=22,
                color=manim.utils.color.rgb_to_color(h_colour(h_curr)[:3]),
            )
            new_label.move_to(np.array([-5.8, 3.4, 0.0]))
            h_label.become(new_label)

            self.wait(wait_time)

        self.wait(0.5)


# ══════════════════════════════════════════════════════════════════════
#  Scene 4 — RK4 vs RK8 side by side
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
    # SCENE = "DoublePendulumStepSize"
    # SCENE = "DoublePendulumComputeSpeed"
    # SCENE = "DoublePendulumRawSteps"
    # ──────────────────────────────────────────────────────────────────

    config.progress_bar = "display"
    config.quality = "high_quality"  # -qh  (60 fps)
    # config.quality = "low_quality"  # -qm  (30 fps)
    config.preview = True
    config.disable_caching = False
    config.custom_folders = "media/"

    scene_class = {
        "DoublePendulumEnergy": DoublePendulumEnergy,
        "DoublePendulumStepSize": DoublePendulumStepSize,
        "DoublePendulumRawSteps": DoublePendulumRawSteps,
        "DoublePendulumComputeSpeed": DoublePendulumComputeSpeed,
        "DoublePendulumSideBySide": DoublePendulumSideBySide,
        "DoublePendulumOverlay": DoublePendulumOverlay,
        "DoublePendulumThreeWay": DoublePendulumThreeWay,
    }[SCENE]

    scene = scene_class()
    scene.render()
