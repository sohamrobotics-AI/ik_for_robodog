"""
quadruped_leg.py  -  3-DOF leg IK, touchdown-matched Bezier gait, trot, limit audit.

Leg frame (origin = hip-abduction axis, which points along +x):
    x forward, y left, z up.
Joints:
    q1  hip abduction : rotation about +x (right-hand rule). Positive swings a
                        foot hanging below the hip toward +y (to the robot's left).
    q2  hip pitch     : angle of the thigh from straight-down, positive = foot forward.
    q3  knee          : angle of the shank relative to the thigh, positive = knee
                        points BACKWARD (Mini-Cheetah style  ">"  stance).
The thigh/shank plane is offset sideways from the abduction axis by  side*l0.
    side = +1 left leg, -1 right leg.

Run:  python quadruped_leg.py        (writes PNGs + a GIF next to this file)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb
import numpy as np

# ----------------------------------------------------------------------------
# 1. GEOMETRY + INVERSE KINEMATICS
# ----------------------------------------------------------------------------
# Link-length justification
#   * L1 = L2 = 0.22 m. Equal links give the largest circular reach for a given
#     total length (0.44 m) and the most isotropic Jacobian near the working
#     point, and make both links the same part (cheaper, symmetric loads).
#   * A knee limited to +-90 deg can only reach D >= sqrt(L1^2+L2^2) = 0.311 m.
#     The foot is highest (closest to the hip) at swing apex, so
#         stand_height - step_height >= 0.311.
#     (First attempt: 0.36 - 0.05 = 0.31 -> the audit caught the knee at 91.2 deg,
#     because the lateral tuck shortens the in-plane reach a few mm.)
#     Standing at 0.37 m = 84 % of max reach leaves ~9 mm of margin at apex,
#     puts the standing knee at ~65 deg (far from the 180 deg singularity) and
#     gives the 0.05 m swing clearance. Shrinking links 0.22 -> 0.20 would cap
#     clearance at ~0.03 m for the same +-90 deg knee.
#   * l0 = 0.06 m abduction offset: keeps the thigh plane clear of the body
#     and gives the abduction joint leverage; typical of 12-15 kg quadrupeds.


@dataclass(frozen=True)
class LegGeometry:
    L1: float = 0.22   # thigh [m]
    L2: float = 0.22   # shank [m]
    l0: float = 0.06   # abduction-axis -> leg-plane offset [m]

    @property
    def reach(self) -> float:
        return self.L1 + self.L2

    @property
    def knee_90_distance(self) -> float:
        return float(np.hypot(self.L1, self.L2))


GEOM = LegGeometry()


class IKError(ValueError):
    """Foot target is outside the leg's workspace."""


def leg_ik(foot, side: int = +1, geom: LegGeometry = GEOM,
           on_unreachable: str = "raise") -> np.ndarray:
    """
    Closed-form IK for one 3-DOF leg.

    Parameters
    ----------
    foot : (..., 3) foot position [x, y, z] in the leg frame (z negative = below hip).
    side : +1 left leg, -1 right leg (sign of the abduction offset).
    on_unreachable : "raise" | "nan".  Vectorised: works on (N,3) arrays.

    Returns
    -------
    q : (..., 3) = [q1 abduction, q2 hip pitch, q3 knee] in radians.

    Derivation
    ----------
    1) Abduction. Rotating the leg plane about x by q1 maps the planar foot point
       (y', z') = (d, -Lp) to (y, z), d = side*l0:
           y = d cos q1 + Lp sin q1,    z = d sin q1 - Lp cos q1
       so  (y, z) has angle atan2(z, y) = q1 + atan2(-Lp, d), and
           q1 = atan2(z, y) - atan2(-Lp, d),   Lp = sqrt(y^2 + z^2 - d^2)
       (Lp is the foot's "drop" inside the leg plane; needs y^2+z^2 >= d^2).
    2) Knee (law of cosines) with D^2 = x^2 + Lp^2:
           cos q3 = (D^2 - L1^2 - L2^2) / (2 L1 L2),   q3 = +acos(.)  [knee back]
    3) Hip pitch. The foot vector makes angle atan2(x, Lp) with "down" and sits
       beta = atan2(L2 sin q3, L1 + L2 cos q3) past the thigh direction, so
           q2 = atan2(x, Lp) - beta
    """
    foot = np.asarray(foot, dtype=float)
    x, y, z = foot[..., 0], foot[..., 1], foot[..., 2]
    d = side * geom.l0

    Lp_sq = y * y + z * z - d * d
    D_sq = x * x + Lp_sq
    c3 = (D_sq - geom.L1 ** 2 - geom.L2 ** 2) / (2.0 * geom.L1 * geom.L2)
    bad = (Lp_sq < 0.0) | (np.abs(c3) > 1.0)
    if np.any(bad):
        if on_unreachable == "raise":
            raise IKError(f"{int(np.sum(bad))} foot target(s) outside workspace "
                          f"(first: {foot[bad][0] if foot.ndim > 1 else foot})")

    Lp = np.sqrt(np.clip(Lp_sq, 0.0, None))
    q3 = np.arccos(np.clip(c3, -1.0, 1.0))
    beta = np.arctan2(geom.L2 * np.sin(q3), geom.L1 + geom.L2 * np.cos(q3))
    q2 = np.arctan2(x, Lp) - beta
    q1 = np.arctan2(z, y) - np.arctan2(-Lp, d)
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi          # wrap to (-pi, pi]

    q = np.stack([q1, q2, q3], axis=-1)
    q[bad] = np.nan
    return q


def leg_points(q, side: int = +1, geom: LegGeometry = GEOM):
    """Forward kinematics. Returns (knee, foot), each (...,3), in the leg frame."""
    q = np.asarray(q, dtype=float)
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    d = side * geom.l0

    def lift(px, pl):  # planar (forward, drop) -> 3-D through abduction
        return np.stack([px, d * np.cos(q1) + pl * np.sin(q1),
                         d * np.sin(q1) - pl * np.cos(q1)], axis=-1)

    knee = lift(geom.L1 * np.sin(q2), geom.L1 * np.cos(q2))
    foot = lift(geom.L1 * np.sin(q2) + geom.L2 * np.sin(q2 + q3),
                geom.L1 * np.cos(q2) + geom.L2 * np.cos(q2 + q3))
    return knee, foot


def leg_fk(q, side: int = +1, geom: LegGeometry = GEOM):
    return leg_points(q, side, geom)[1]


# ----------------------------------------------------------------------------
# 2. FOOT TRAJECTORY  (straight stance + touchdown-matched Bezier swing)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GaitParams:
    step_length: float = 0.12    # [m] peak-to-peak foot travel in the body frame
    step_height: float = 0.05    # [m] swing apex above the ground
    cycle_time: float = 0.60     # [s] full gait cycle
    duty: float = 0.50           # stance fraction (trot = 0.5)
    stand_height: float = 0.37   # [m] hip height above ground
    tuck: float = 0.012          # [m] foot pulled toward the centreline at apex


def bezier(ctrl: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Bernstein-form Bezier curve. ctrl (n+1, d), s (N,) in [0,1] -> (N, d)."""
    n = len(ctrl) - 1
    k = np.arange(n + 1)
    b = np.array([comb(n, i) for i in k])[None, :] * s[:, None] ** k * (1 - s[:, None]) ** (n - k)
    return b @ ctrl


def swing_control_points(p: GaitParams) -> np.ndarray:
    """
    9 control points (degree 8) in (x, z).

    The trick: in the body frame the foot is moving BACKWARD at the stance speed
    v_s = step_length / (duty*T) at lift-off and touchdown. Choosing
        P1 = P0 - delta,  P7 = P8 + delta,   delta = v_s*T_swing/8
    makes the swing C1-continuous with the stance line, so the foot touches
    down with zero WORLD velocity (no scuffing, no velocity step in the joints).
    z has zero slope at both ends (P1.z = P7.z = 0).
    """
    h = p.step_length / 2
    v_s = p.step_length / (p.duty * p.cycle_time)
    t_sw = (1 - p.duty) * p.cycle_time
    delta = v_s * t_sw / 8
    x = np.array([-h, -h - delta, -0.8 * h, -0.4 * h, 0, 0.4 * h, 0.8 * h, h + delta, h])
    z = np.array([0, 0, 1.0, 1.3, 1.3, 1.3, 1.0, 0, 0])
    zz = bezier(np.stack([x, z], 1), np.linspace(0, 1, 401))[:, 1].max()
    return np.stack([x, z * p.step_height / zz], 1)      # apex == step_height exactly


def foot_xz(phase: np.ndarray, p: GaitParams) -> np.ndarray:
    """Foot (x, z-lift) vs gait phase in [0,1): stance first, then swing."""
    phase = np.asarray(phase) % 1.0
    h = p.step_length / 2
    out = np.zeros((phase.size, 2))
    st = phase < p.duty
    out[st, 0] = h - p.step_length * phase[st] / p.duty          # straight, constant speed
    sw = ~st
    s = (phase[sw] - p.duty) / (1 - p.duty)
    out[sw] = bezier(swing_control_points(p), s)
    return out


def foot_path_leg_frame(phase, side: int, p: GaitParams, geom: LegGeometry = GEOM):
    """Full 3-D foot target in the leg frame (adds lateral tuck + stand height)."""
    xz = foot_xz(phase, p)
    lift = xz[:, 1]
    y = side * (geom.l0 - p.tuck * lift / max(p.step_height, 1e-9))
    z = -p.stand_height + lift
    return np.stack([xz[:, 0], y, z], axis=1)


# ----------------------------------------------------------------------------
# 3. TROT ACROSS 4 LEGS + LIMIT AUDIT
# ----------------------------------------------------------------------------
LEGS = ["FL", "FR", "RL", "RR"]
SIDE = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}
TROT_OFFSET = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}   # diagonal pairs in sync
JOINTS = ["abduction", "hip", "knee"]
ANGLE_LIMIT = np.deg2rad(90.0)
SPEED_LIMIT = 6.0                                           # rad/s


@dataclass
class TrotResult:
    t: np.ndarray              # (N,)
    q: np.ndarray              # (N, 4, 3)  rad
    qd: np.ndarray             # (N, 4, 3)  rad/s
    foot: np.ndarray           # (N, 4, 3)  leg-frame targets
    stance: np.ndarray         # (N, 4) bool
    angle_flag: np.ndarray     # (N, 4, 3) bool
    speed_flag: np.ndarray     # (N, 4, 3) bool
    params: GaitParams
    geom: LegGeometry

    @property
    def ok(self) -> bool:
        return not (self.angle_flag.any() or self.speed_flag.any())

    @property
    def body_speed(self) -> float:
        return self.params.step_length / (self.params.duty * self.params.cycle_time)


def run_trot(p: GaitParams = GaitParams(), geom: LegGeometry = GEOM, n: int = 720) -> TrotResult:
    t = np.arange(n) * p.cycle_time / n                      # one cycle, endpoint excluded
    dt = p.cycle_time / n
    q = np.zeros((n, 4, 3)); foot = np.zeros((n, 4, 3)); stance = np.zeros((n, 4), bool)
    for i, leg in enumerate(LEGS):
        phase = (t / p.cycle_time + TROT_OFFSET[leg]) % 1.0
        foot[:, i] = foot_path_leg_frame(phase, SIDE[leg], p, geom)
        stance[:, i] = phase < p.duty
        q[:, i] = leg_ik(foot[:, i], SIDE[leg], geom, on_unreachable="nan")
    # periodic central difference (the gait is exactly periodic)
    qd = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2 * dt)
    with np.errstate(invalid="ignore"):
        angle_flag = np.abs(q) > ANGLE_LIMIT
        speed_flag = np.abs(qd) > SPEED_LIMIT
    nanmask = np.isnan(q)                                    # unreachable = flagged too
    return TrotResult(t, q, qd, foot, stance, angle_flag | nanmask, speed_flag | nanmask, p, geom)


def _runs(mask: np.ndarray, t: np.ndarray, T: float):
    """Contiguous True runs (cyclic) -> list of (t_start, t_end)."""
    if not mask.any():
        return []
    if mask.all():
        return [(0.0, T)]
    m = np.roll(mask, -int(np.argmin(mask)))                 # start on a False
    t0 = np.roll(t, -int(np.argmin(mask)))
    dt = t[1] - t[0]
    out, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = t0[i]
        if not v and start is not None:
            out.append((start % T, (t0[i] - dt / 2 + dt / 2) % T)); start = None
    if start is not None:
        out.append((start % T, (t0[-1] + dt) % T))
    return out


def report(r: TrotResult, title: str = "") -> str:
    T = r.params.cycle_time
    lines = [f"=== {title} | v_body={r.body_speed:.2f} m/s  L={r.params.step_length*100:.0f} cm  "
             f"H={r.params.step_height*100:.0f} cm  T={T:.2f} s ==="]
    lines.append(f"{'joint':<10}{'max|q| deg':>11}{'max|w| rad/s':>14}  flags")
    n_bad = 0
    for i, leg in enumerate(LEGS):
        for j, jn in enumerate(JOINTS):
            qmax = np.nanmax(np.abs(r.q[:, i, j]))
            wmax = np.nanmax(np.abs(r.qd[:, i, j]))
            ev = []
            for a, b in _runs(r.angle_flag[:, i, j], r.t, T):
                ev.append(f"ANGLE>90 {a:.3f}-{b:.3f}s"); n_bad += 1
            for a, b in _runs(r.speed_flag[:, i, j], r.t, T):
                ev.append(f"SPEED>6 {a:.3f}-{b:.3f}s"); n_bad += 1
            lines.append(f"{leg+' '+jn:<10}{np.rad2deg(qmax):>11.1f}{wmax:>14.2f}  "
                         f"{'; '.join(ev) if ev else 'ok'}")
    lines.append(f"-> {'ALL CLEAR' if n_bad == 0 else str(n_bad) + ' limit violation event(s)'}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------------
def plot_joints(r: TrotResult, path: str, title: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = r.params.cycle_time
    tt = np.append(r.t, T)
    fig, axs = plt.subplots(4, 3, figsize=(13, 9), sharex=True, constrained_layout=True)
    cols = {"FL": "#1b9e77", "FR": "#d95f02", "RL": "#7570b3", "RR": "#e7298a"}
    for i, leg in enumerate(LEGS):
        for j, jn in enumerate(JOINTS):
            ax = axs[i, j]
            ang = np.rad2deg(np.append(r.q[:, i, j], r.q[0, i, j]))
            # shade stance
            st = np.append(r.stance[:, i], r.stance[0, i])
            ax.fill_between(tt, -200, 200, where=st, color="0.92", step="post", lw=0)
            ax.axhline(90, color="r", ls="--", lw=.8); ax.axhline(-90, color="r", ls="--", lw=.8)
            ax.plot(tt, ang, color=cols[leg], lw=2)
            af = np.append(r.angle_flag[:, i, j], r.angle_flag[0, i, j])
            sf = np.append(r.speed_flag[:, i, j], r.speed_flag[0, i, j])
            ax.scatter(tt[af], ang[af], c="red", s=14, zorder=5, label="|q|>90°")
            ax.scatter(tt[sf], ang[sf], c="orange", s=14, zorder=4, marker="s", label="|ω|>6 rad/s")
            lim = max(100, np.nanmax(np.abs(ang)) * 1.15)
            ax.set_ylim(-lim, lim); ax.set_xlim(0, T)
            wmax = np.nanmax(np.abs(r.qd[:, i, j]))
            ax.set_title(f"{leg} {jn}   max|ω|={wmax:.1f} rad/s", fontsize=9,
                         color="red" if (af.any() or sf.any()) else "black")
            if j == 0: ax.set_ylabel("deg")
            if i == 3: ax.set_xlabel("time [s]")
    h, l = axs[0, 0].get_legend_handles_labels()
    handles = {}
    for a in axs.ravel():
        for hh, ll in zip(*a.get_legend_handles_labels()):
            handles[ll] = hh
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(f"{title}  (grey = stance, red dashes = ±90°)", fontsize=12)
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_trajectory(p: GaitParams, path: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = np.linspace(0, 1, 800, endpoint=False)
    xz = foot_xz(ph, p)
    ctrl = swing_control_points(p)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax[0].plot(xz[:, 0], xz[:, 1], lw=2.5, color="#1b9e77")
    ax[0].plot(ctrl[:, 0], ctrl[:, 1], "o--", color="0.5", ms=4, lw=.8, label="Bézier control polygon")
    ax[0].annotate("stance (straight)", (0, 0), (0, -0.012), ha="center")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("lift [m]")
    ax[0].set_title("Foot path in the body frame")
    dt = p.cycle_time / 800
    vx = np.gradient(xz[:, 0], dt); vz = np.gradient(xz[:, 1], dt)
    ax[1].plot(ph, vx, label="vx"); ax[1].plot(ph, vz, label="vz")
    ax[1].axhline(-p.step_length / (p.duty * p.cycle_time), color="r", ls=":", lw=1,
                  label="-v_stance (ground speed)")
    ax[1].axvline(p.duty, color="0.6", lw=.8)
    ax[1].set_xlabel("gait phase"); ax[1].set_ylabel("m/s"); ax[1].legend()
    ax[1].set_title("Velocity is continuous; touchdown has zero world speed")
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_envelopes(geom: LegGeometry, base: GaitParams, marks: dict, path: str):
    """Two feasibility maps: where the +-90 deg and 6 rad/s limits bite."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # (a) stand height vs step height -> max joint angle
    sh = np.linspace(0.28, 0.43, 31); st = np.linspace(0.0, 0.12, 31)
    A = np.zeros((st.size, sh.size))
    for a, hs in enumerate(st):
        for b, hh in enumerate(sh):
            r = run_trot(replace(base, step_height=hs, stand_height=hh), geom, n=120)
            A[a, b] = np.rad2deg(np.nanmax(np.abs(r.q))) if not np.isnan(r.q).any() else 180
    c = ax[0].contourf(sh, st * 100, np.clip(A, 0, 130), levels=np.linspace(40, 130, 19), cmap="viridis")
    ax[0].contour(sh, st * 100, A, levels=[90], colors="r", linewidths=2.5)
    ax[0].set_xlabel("stand height [m]"); ax[0].set_ylabel("step height [cm]")
    ax[0].set_title("Max |joint angle| [deg]  -  red line = 90° (knee-limited)")
    fig.colorbar(c, ax=ax[0])

    # (b) cycle time vs step length -> max joint speed
    Ts = np.linspace(0.25, 1.0, 31); Ls = np.linspace(0.04, 0.26, 31)
    S = np.zeros((Ls.size, Ts.size)); Ang = np.zeros_like(S)
    for a, ll in enumerate(Ls):
        for b, tc in enumerate(Ts):
            r = run_trot(replace(base, step_length=ll, cycle_time=tc), geom, n=160)
            S[a, b] = np.nanmax(np.abs(r.qd)) if not np.isnan(r.q).any() else 99
            Ang[a, b] = np.rad2deg(np.nanmax(np.abs(r.q))) if not np.isnan(r.q).any() else 180
    c = ax[1].contourf(Ts, Ls * 100, np.clip(S, 0, 14), levels=np.linspace(0, 14, 29), cmap="magma")
    ax[1].contour(Ts, Ls * 100, S, levels=[SPEED_LIMIT], colors="cyan", linewidths=2.5)
    ax[1].contourf(Ts, Ls * 100, Ang, levels=[90, 200], colors="none", hatches=["///"])
    TT, LL = np.meshgrid(Ts, Ls)
    cs = ax[1].contour(Ts, Ls * 100, LL / (base.duty * TT), levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.4],
                       colors="w", linewidths=.7, linestyles=":")
    ax[1].clabel(cs, fmt="%.1f m/s", fontsize=8)
    ax[1].set_xlabel("cycle time [s]"); ax[1].set_ylabel("step length [cm]")
    ax[1].set_title("Max |joint speed| [rad/s]  -  cyan = 6 rad/s servo limit")
    fig.colorbar(c, ax=ax[1])
    for name, (tc, ll, col) in marks.items():
        ax[1].plot(tc, ll * 100, "*", ms=16, color=col, mec="k"); ax[1].annotate(
            name, (tc, ll * 100), (6, 6), textcoords="offset points", color=col, weight="bold")
    fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------
# 5. CREATIVE EXTRA: trotting dog GIF, legs turn red when a limit is broken
# ----------------------------------------------------------------------------
def make_gif(r: TrotResult, path: str, cycles: int = 3, slowmo: float = 0.5, fps: int = 25):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyBboxPatch
    p, g = r.params, r.geom
    n = len(r.t); T = p.cycle_time
    hip_x = {"FL": .18, "FR": .18, "RL": -.18, "RR": -.18}
    v = r.body_speed
    n_frames = int(cycles * T / slowmo * fps)
    colors = {"FL": "#1b9e77", "FR": "#d95f02", "RL": "#7570b3", "RR": "#e7298a"}

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=70)
    fig.patch.set_facecolor("#10131a"); ax.set_facecolor("#10131a")
    ax.set_xlim(-0.55, 0.75); ax.set_ylim(-0.06, 0.62); ax.set_aspect("equal"); ax.axis("off")
    ax.axhline(0, color="#46506b", lw=2)
    ticks = ax.plot([], [], "|", color="#46506b", ms=10)[0]
    body = FancyBboxPatch((-.26, p.stand_height - .035), .52, .07, boxstyle="round,pad=0.01,rounding_size=.03",
                          fc="#2b3350", ec="#8ea0ff", lw=1.6); ax.add_patch(body)
    lines = {l: ax.plot([], [], "-o", lw=5 if SIDE[l] > 0 else 3, ms=5, color=colors[l],
                        alpha=1 if SIDE[l] > 0 else .55, solid_capstyle="round")[0] for l in LEGS}
    trail = ax.plot([], [], "s", color="#ffd166", ms=3)[0]
    label = ax.text(-.5, .56, "", color="w", fontsize=10, family="monospace")
    steps = []  # world footprints (x)

    def frame(k):
        tt = (k * slowmo / fps) % T
        i = int(tt / T * n) % n
        bx = v * (k * slowmo / fps)
        body.set_x(-.26)                                   # camera follows the body
        for li, l in enumerate(LEGS):
            kn, ft = leg_points(r.q[i, li], SIDE[l], g)
            hx, hz = hip_x[l], p.stand_height
            lines[l].set_data([hx, hx + kn[0], hx + ft[0]], [hz, hz + kn[2], hz + ft[2]])
            bad = r.angle_flag[i, li].any() or r.speed_flag[i, li].any()
            lines[l].set_color("red" if bad else colors[l])
            if r.stance[i, li] and (not r.stance[i - 1, li]):
                steps.append(bx + hx + ft[0])
        sx = np.array([s for s in steps if -.6 < s - bx < .8])
        trail.set_data(sx - bx, np.zeros_like(sx))
        gx = (np.arange(-3, 4) * 0.1) - (bx % 0.1)
        ticks.set_data(gx, np.full_like(gx, -0.015))
        label.set_text(f"trot  {v:.2f} m/s   t={tt:4.2f}s   {'ALL OK' if r.ok else 'LIMITS BROKEN -> red'}")
        return []

    FuncAnimation(fig, frame, frames=n_frames, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    out = os.path.dirname(os.path.abspath(__file__))

    # --- sanity: FK(IK(x)) == x over a random workspace sample, both sides ----
    rng = np.random.default_rng(0)
    for side in (+1, -1):
        pts = np.stack([rng.uniform(-.12, .12, 5000), side * rng.uniform(.03, .12, 5000),
                        rng.uniform(-.40, -.30, 5000)], 1)
        q = leg_ik(pts, side, on_unreachable="nan"); ok = ~np.isnan(q[:, 0])
        err = np.abs(leg_fk(q[ok], side) - pts[ok]).max()
        print(f"IK->FK round trip side={side:+d}: {ok.sum()} pts, max error {err:.2e} m")

    nominal = GaitParams()
    aggressive = GaitParams(step_length=0.20, step_height=0.08, cycle_time=0.35)

    for name, gp in [("NOMINAL", nominal), ("AGGRESSIVE", aggressive)]:
        r = run_trot(gp)
        print(report(r, name), "\n")
        plot_joints(r, f"{out}/joints_{name.lower()}.png", f"{name} trot")
        make_gif(r, f"{out}/trot_{name.lower()}.gif")

    plot_trajectory(nominal, f"{out}/foot_trajectory.png")
    plot_envelopes(GEOM, nominal,
                   {"nominal": (nominal.cycle_time, nominal.step_length, "lime"),
                    "aggressive": (aggressive.cycle_time, aggressive.step_length, "red")},
                   f"{out}/envelope_maps.png")
    print("wrote figures + GIFs to", out)
