"""Parametric big-air tricks, each with an exact label.

A trick's ground truth is mostly the root turning about an axis that tilts
away from vertical, plus a body shape on top (tuck, grab, wind-up). So tricks
are generated from parameters rather than animated by hand, and every label
is computed from the generated motion itself, never copied from the request.

The world is Blender's: **Z up**, the rider travels along **+Y**. The rider
rig's rest faces -Y with its left side at +X (MPFB's A-pose).

How a cork is made
------------------
The root's angular velocity is ``theta'(t) * a(t)``: a spin rate along an axis
``a`` tilted ``alpha(t)`` off vertical. About a fixed axis tilted ``alpha``,
the body's up vector is furthest from vertical, ``2 * alpha``, each time the
spin passes 180 + 360k degrees. So a cork is an axis that leans past 45
degrees around *one* of those moments and a double cork around two
consecutive ones, and ``alpha`` returns to near vertical before the landing.
The amount of lean is solved per trick (``solve_tilt``) by counting the
inversions the integrated motion really produces, so "double cork" is a
property of the generated motion, not a name attached to it.

The axis is not fixed while it leans, so the integrated orientation does not
land exactly where a clean landing needs it. A correction rotation is
blended in over the air time; it also carries the pitch from the kicker's
angle to the landing's. Labels are measured after that blend.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..camera import rotation_angle, rotation_vector

UP = np.array([0.0, 0.0, 1.0])
GRAVITY = 9.81
OUTPUT_FPS = 30

KINDS = ("flat", "cork", "double_cork")
CAMERA_ANGLES = ("side", "front_quarter", "back_quarter", "low_landing")
GRABS = ("none", "indy", "melon", "mute", "stalefish", "tail", "nose")


@dataclass
class Trick:
    name: str
    spin_deg: int                 # nominal: 360, 540, ... 1440
    kind: str                     # flat | cork | double_cork
    direction: str                # frontside | backside
    stance: str                   # regular | goofy
    grab: str
    camera_angle: str
    slowmo: float                 # conform factor: 4.0 = 120 fps shown at 30
    seed: int
    air_s: float = 2.8
    approach_s: float = 0.8
    rideout_s: float = 0.8
    kicker_deg: float = 28.0
    landing_deg: float = 32.0
    takeoff_speed: float = 16.0
    prewind_deg: float = 30.0
    tuck_depth: float = 0.6
    grab_window: tuple[float, float] = (0.3, 0.7)
    tilt_azimuth_deg: float = 0.0
    tilt_max_deg: float | None = None   # solved, see solve_tilt
    body: dict = field(default_factory=dict)   # MPFB macro values
    camera: dict = field(default_factory=dict)
    # Split group: clips sharing a rider (body, outfit, board) and a venue
    # (dressing, sun). Train/test splits must keep a group whole. ``None``
    # seeds fall back to ``seed``, so the first 20 clips are each their own.
    group: str = ""
    rider_seed: int | None = None
    venue_seed: int | None = None
    # Ground time (see ``_ground``). ``terrain=False`` is the original model:
    # a short straight run up the kicker and a straight slide down the
    # landing, which only holds for approaches under about 0.8 s.
    terrain: bool = False
    standing_s: float = 0.0       # stood still at the top before riding in
    carve_m: float = 0.0          # half-width of the turns on deck and outrun
    carve_period_s: float = 2.0   # one left-right turn pair, at takeoff speed
    jitter_deg: float = 0.0       # balance wobble of the whole body

    @property
    def spin_sign(self) -> int:
        """+1 counter-clockwise seen from above, -1 clockwise.

        A regular rider (left foot forward) faces +X while travelling +Y, so a
        frontside spin, which opens the chest down the hill first, turns the
        facing from +X toward +Y: counter-clockwise from above.
        """
        frontside = self.direction == "frontside"
        regular = self.stance == "regular"
        return 1 if frontside == regular else -1

    @property
    def render_fps(self) -> float:
        """Frames per *real* second; output plays them at 30 fps."""
        return OUTPUT_FPS * self.slowmo

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Trick":
        raw = json.loads(text)
        raw["grab_window"] = tuple(raw["grab_window"])
        return cls(**raw)


# --- rotation helpers -------------------------------------------------------

def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: rotation of ``angle`` radians about unit ``axis``."""
    x, y, z = axis / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rot_x(deg: float) -> np.ndarray:
    return axis_angle(np.array([1.0, 0, 0]), np.deg2rad(deg))


def rot_z(deg: float) -> np.ndarray:
    return axis_angle(UP, np.deg2rad(deg))


def slerp_identity(rotation: np.ndarray, fraction: float) -> np.ndarray:
    """``fraction`` of the way from identity to ``rotation``, on the shortest arc."""
    vector = rotation_vector(rotation)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    return axis_angle(vector, fraction * angle)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def stance_rotation(stance: str) -> np.ndarray:
    """Rig rest (facing -Y) -> riding sideways, nose of the board along +Y."""
    return rot_z(90.0 if stance == "regular" else -90.0)


# --- the spin ---------------------------------------------------------------

def spin_progress(s: np.ndarray) -> np.ndarray:
    """Fraction of the spin done at air fraction ``s``.

    The rate is flat through the middle and eases at both ends, ``1 - (2s-1)^4``
    integrated and normalised: riders carry a near-constant spin rate, and
    neither leave the lip nor land at full speed.
    """
    s = np.clip(s, 0.0, 1.0)
    u = 2 * s - 1
    integral = (s - (u ** 5 + 1) / 10) / 0.8
    return integral


def inversion_targets(spin_deg: int, kind: str) -> list[float]:
    """Spin angles (deg) at which the body should be inverted."""
    wanted = {"flat": 0, "cork": 1, "double_cork": 2}[kind]
    if wanted == 0:
        return []
    peaks = [180.0 + 360.0 * k for k in range(8) if 180.0 + 360.0 * k < spin_deg - 90]
    if len(peaks) < wanted:
        raise ValueError(f"a {spin_deg} cannot carry {wanted} inversions")
    # The consecutive run closest to the middle of the spin: riders open up
    # before the landing rather than invert just above it.
    middle = spin_deg / 2.0
    best = min(
        range(len(peaks) - wanted + 1),
        key=lambda i: abs(np.mean(peaks[i:i + wanted]) - middle) + 1e-3 * i,
    )
    return peaks[best:best + wanted]


def tilt_window(theta_deg: np.ndarray, targets: list[float]) -> np.ndarray:
    """0..1 lean of the axis as a function of spin angle: on around the targets."""
    if not targets:
        return np.ones_like(theta_deg)
    lo, hi = min(targets) - 150.0, max(targets) + 150.0
    ramp = 100.0
    return smoothstep((theta_deg - (lo - ramp)) / ramp) * (
        1.0 - smoothstep((theta_deg - hi) / ramp)
    )


def _air_rotations(trick: Trick, tilt_max_deg: float, s: np.ndarray) -> np.ndarray:
    """Root rotation through the air, correction blended in. Rotates rig rest."""
    theta = np.deg2rad(trick.spin_deg) * spin_progress(s)
    theta_deg = np.rad2deg(theta)
    alpha = np.deg2rad(tilt_max_deg) * tilt_window(
        theta_deg, inversion_targets(trick.spin_deg, trick.kind)
    )
    azimuth = np.deg2rad(trick.tilt_azimuth_deg)
    lean = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])

    spin = np.empty((len(s), 3, 3))
    spin[0] = np.eye(3)
    for i in range(1, len(s)):
        mid_alpha = 0.5 * (alpha[i] + alpha[i - 1])
        axis = np.cos(mid_alpha) * UP + np.sin(mid_alpha) * lean
        step = (theta[i] - theta[i - 1]) * trick.spin_sign
        spin[i] = axis_angle(axis, step) @ spin[i - 1]

    error = rot_z(trick.spin_sign * trick.spin_deg) @ spin[-1].T
    blend = smoothstep(s)
    # Pitch is applied last, about the world's lateral axis, so it levels the
    # rider from the kicker's angle to the landing's without being swung
    # around by the spin (pitch first would turn into roll half a turn later).
    pitch = trick.kicker_deg + (-trick.landing_deg - trick.kicker_deg) * blend
    stance = stance_rotation(trick.stance)
    out = np.empty((len(s), 3, 3))
    for i in range(len(s)):
        out[i] = rot_x(pitch[i]) @ slerp_identity(error, blend[i]) @ spin[i] @ stance
    return out


def tilt_series(rotations: np.ndarray) -> np.ndarray:
    """Degrees between the body's up axis (rig +Z) and world up, per frame."""
    up = rotations[:, :, 2]
    return np.rad2deg(np.arccos(np.clip(up @ UP, -1.0, 1.0)))


def count_inversions(tilt_deg: np.ndarray) -> int:
    """Separate stretches with the head below the board (tilt over 90)."""
    inverted = tilt_deg > 90.0
    return int(np.sum(inverted[1:] & ~inverted[:-1]) + inverted[0])


def solve_tilt(trick: Trick, samples: int = 400) -> None:
    """Set the axis lean (and its direction) so ``kind``'s inversions happen.

    Every lean on a 1-degree grid is integrated and its inversions counted,
    and the lean is drawn (seeded) from the middle of the longest working run:
    a run's edges are a degree from gaining or losing an inversion, which is
    not a clean example. Some lean directions never give a clean single cork
    (the landing pitch adds to leftover lean and grazes a second inversion),
    so further directions are tried, 30 degrees apart, before giving up.
    Flat spins draw 0-10 degrees and must never pass 60 degrees of tilt.
    """
    rng = np.random.default_rng(trick.seed + 7)
    s = np.linspace(0.0, 1.0, samples)
    if trick.kind == "flat":
        lean = float(rng.uniform(0.0, 10.0))
        tilt = tilt_series(_air_rotations(trick, lean, s))
        if tilt.max() > 60.0:
            raise ValueError(f"{trick.name}: flat spin tilts {tilt.max():.0f} deg")
        trick.tilt_max_deg = round(lean, 2)
        return
    wanted = len(inversion_targets(trick.spin_deg, trick.kind))
    first = trick.tilt_azimuth_deg
    for attempt in range(12):
        trick.tilt_azimuth_deg = round((first + 30.0 * attempt) % 360.0, 1)
        runs: list[list[float]] = [[]]
        for lean in np.arange(35.0, 86.0, 1.0):
            tilt = tilt_series(_air_rotations(trick, lean, s))
            # A clear inversion, head well below the board, not a 95 degree graze.
            if count_inversions(tilt) == wanted and 110.0 <= tilt.max() < 165.0:
                runs[-1].append(float(lean))
            elif runs[-1]:
                runs.append([])
        best = max(runs, key=len)
        if len(best) >= 3:
            inner = best[1:-1]
            trick.tilt_max_deg = round(float(rng.uniform(inner[0], inner[-1])), 2)
            return
    raise ValueError(f"{trick.name}: no axis lean gives {wanted} clean inversion(s)")


# --- the whole clip ---------------------------------------------------------

@dataclass
class Motion:
    """Per rendered frame. Frame ``i`` is output frame ``i`` at 30 fps."""

    t: np.ndarray            # real seconds, 0 at takeoff
    position: np.ndarray     # (n, 3) pelvis, world metres
    rotation: np.ndarray     # (n, 3, 3) root, rotates rig rest into world
    tuck: np.ndarray         # 0..1 knees and hips drawn up
    grab: np.ndarray         # 0..1 hand on the board
    wind: np.ndarray         # degrees of upper body twist, signed like the spin
    takeoff_frame: int
    landing_frame: int

    def save(self, dest: Path) -> None:
        np.savez_compressed(
            dest, t=self.t, position=self.position, rotation=self.rotation,
            tuck=self.tuck, grab=self.grab, wind=self.wind,
            takeoff_frame=self.takeoff_frame, landing_frame=self.landing_frame,
        )

    @classmethod
    def load(cls, src: Path) -> "Motion":
        z = np.load(src)
        return cls(
            t=z["t"], position=z["position"], rotation=z["rotation"],
            tuck=z["tuck"], grab=z["grab"], wind=z["wind"],
            takeoff_frame=int(z["takeoff_frame"]), landing_frame=int(z["landing_frame"]),
        )


PELVIS_HEIGHT = 0.95  # above the board's base in a riding stance, metres


def motion(trick: Trick) -> Motion:
    if trick.tilt_max_deg is None:
        raise ValueError("solve_tilt first; the lean is part of the trick's identity")
    fps = trick.render_fps
    takeoff = int(round(trick.approach_s * fps))
    landing = takeoff + int(round(trick.air_s * fps))
    count = landing + int(round(trick.rideout_s * fps)) + 1
    t = (np.arange(count) - takeoff) / fps
    air = trick.air_s

    kicker = np.deg2rad(trick.kicker_deg)
    v = trick.takeoff_speed
    vy, vz = v * np.cos(kicker), v * np.sin(kicker)
    lip = np.array([0.0, 0.0, PELVIS_HEIGHT])

    position = np.empty((count, 3))
    rotation = np.empty((count, 3, 3))
    start = rot_x(trick.kicker_deg) @ stance_rotation(trick.stance)
    s_air = np.clip(t / air, 0.0, 1.0)
    in_air = _air_rotations(trick, trick.tilt_max_deg, s_air[takeoff:landing + 1])
    land_speed = np.array([vy, vz - GRAVITY * air])
    land_dir = np.array([0.0, np.cos(np.deg2rad(trick.landing_deg)),
                         -np.sin(np.deg2rad(trick.landing_deg))])
    land_point = lip + np.array([0.0, vy * air, vz * air - 0.5 * GRAVITY * air ** 2])
    rideout_speed = float(np.hypot(*land_speed)) * 0.8

    for i, ti in enumerate(t):
        if i < takeoff:
            # Up the kicker at constant speed, standing square to the ramp.
            position[i] = lip + np.array([0.0, vy * ti, vz * ti])
            rotation[i] = start
        elif i <= landing:
            position[i] = lip + np.array([0.0, vy * ti, vz * ti - 0.5 * GRAVITY * ti ** 2])
            rotation[i] = in_air[i - takeoff]
        else:
            position[i] = land_point + land_dir * rideout_speed * (ti - air)
            rotation[i] = in_air[-1]

    s = t / air
    # Crouched riding, pop, tuck through the air, absorb the landing.
    tuck = np.full(count, 0.2)
    tuck += (trick.tuck_depth - 0.2) * smoothstep(s / 0.25) * (1 - smoothstep((s - 0.8) / 0.15))
    tuck += 0.35 * np.exp(-(((t - air) - 0.12) / 0.12) ** 2) * (t > air - 0.05)
    tuck -= 0.15 * np.exp(-((t + 0.05) / 0.08) ** 2)  # extension at the pop
    tuck = np.clip(tuck, 0.0, 1.0)

    if trick.grab == "none":
        grab = np.zeros(count)
    else:
        g0, g1 = trick.grab_window
        grab = smoothstep((s - g0) / 0.08) * (1 - smoothstep((s - g1) / 0.08))

    # Wind-up: shoulders turn against the spin on the approach, then lead it.
    sign = trick.spin_sign
    wind = -sign * trick.prewind_deg * smoothstep((t + 0.5) / 0.45) * (t < 0)
    wind += sign * trick.prewind_deg * 0.6 * smoothstep(s / 0.15) * (1 - smoothstep((s - 0.25) / 0.35)) * (t >= 0)
    wind += -sign * trick.prewind_deg * (t >= 0) * (1 - smoothstep(t / 0.12))

    if trick.terrain:
        _ground(trick, t, takeoff, landing, position, rotation, tuck)
    return Motion(t=t, position=position, rotation=rotation, tuck=tuck,
                  grab=grab, wind=wind, takeoff_frame=takeoff, landing_frame=landing)


# Pelvis above the board's base, upright, at tuck 0 and 1. Mirrors
# blender/animate.py, which cannot be imported without bpy; a test pins them.
DROP_STANDING = 0.88
DROP_TUCKED = 0.34
DECK_BELOW_LIP = 6.5   # venue.build: the deck is this far below the lip's board
LANDING_RUN_M = 40.0   # venue.build: landing slope length before the outrun


def _drop(tuck) -> np.ndarray:
    return DROP_STANDING + (DROP_TUCKED - DROP_STANDING) * np.asarray(tuck)


def _smooth_noise(rng: np.random.Generator, n: int, fps: float, scale: float,
                  corr_s: float = 0.4, dims: int = 3) -> np.ndarray:
    """Slow random wander, (n, dims), standard deviation about ``scale``."""
    keep = np.exp(-1.0 / max(1.0, corr_s * fps))
    raw = rng.normal(0.0, 1.0, (n, dims))
    # Two passes: one alone is a random walk, rough from frame to frame,
    # and its sway rates read as hundreds of degrees a second.
    for _ in range(2):
        out = np.zeros((n, dims))
        for i in range(1, n):
            out[i] = keep * out[i - 1] + raw[i]
        raw = out
    sd = out.std(axis=0)
    return out * (scale / np.where(sd > 0, sd, 1.0))


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with the ends held, so a stretch keeps its length."""
    radius = int(3 * sigma)
    if radius < 1 or len(x) < 2:
        return x
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    padded = np.concatenate([np.full(radius, x[0]), x, np.full(radius, x[-1])])
    return np.convolve(padded, k / k.sum(), mode="valid")


def _carve(distance: np.ndarray, amplitude: float, wavelength: float, phase: float):
    """Lateral offset (m) of S-turns, periodic in distance travelled.

    Periodic in distance, not time, so a rider who slows down turns less
    often instead of swinging sideways on the spot. The caller derives
    heading and edge roll from it along the path.
    """
    return amplitude * np.sin(2 * np.pi * distance / wavelength + phase)


def _ground(trick: Trick, t, takeoff: int, landing: int, position, rotation, tuck) -> None:
    """Replace the ground frames with riding that follows the venue's terrain.

    Before the takeoff: stood still at the top of the deck for
    ``standing_s``, then riding the flat deck while gathering speed (in
    S-turns of ``carve_m`` that straighten out before the kicker), then up
    the kicker. After the landing: down the landing slope and onto the
    outrun, carving and slowing to a stop. The board's base stays on the
    snow surface ``venue.build`` makes from the same path, so the approach
    and ride-out can be as long as wanted without the rider passing through
    the hill. A slow balance wobble (``jitter_deg``) rides on top. Air frames
    are left exactly as they are.
    """
    fps = trick.render_fps
    n = len(t)
    rng = np.random.default_rng(trick.seed + 31)
    stance = stance_rotation(trick.stance)
    v = trick.takeoff_speed
    k = np.deg2rad(trick.kicker_deg)
    up_slope = np.array([0.0, np.cos(k), np.sin(k)])

    # Riding crouch with a little life in it; tall while stood still.
    pre = np.arange(n) < takeoff
    post = np.arange(n) > landing
    standing = pre & (t < -trick.approach_s + trick.standing_s)
    tuck[standing] = 0.05 + 0.15 * smoothstep((t[standing] + trick.approach_s
                                               - trick.standing_s + 0.4) / 0.4)
    wobble = _smooth_noise(rng, n, fps, 0.04, dims=1)[:, 0]
    tuck[pre | post] = np.clip(tuck[pre | post] + wobble[pre | post], 0.0, 1.0)

    # Board base at takeoff, as venue.board_path reads it.
    board0 = position[takeoff] - rotation[takeoff][:, 2] * _drop(tuck[takeoff])
    deck_z = board0[2] - DECK_BELOW_LIP
    ramp_t = DECK_BELOW_LIP / np.sin(k) / v       # seconds on the kicker

    # --- approach: distance before the lip along the run, per frame ----
    t_pre = t[:takeoff]
    t_start = -trick.approach_s + trick.standing_s
    deck_time = max(1e-3, -ramp_t - t_start)
    u0 = 0.0 if trick.standing_s > 0 else 0.6 * v
    speed = np.where(t_pre < -ramp_t,
                     u0 + (v - u0) * smoothstep((t_pre - t_start) / deck_time), v)
    speed[t_pre < t_start] = 0.0
    # Distance remaining to the lip, integrated backwards from the takeoff.
    dist = np.zeros(len(t_pre))
    for i in range(len(t_pre) - 2, -1, -1):
        dist[i] = dist[i + 1] + 0.5 * (speed[i] + speed[i + 1]) / fps
    if takeoff:
        dist += speed[-1] / fps   # the last frame before the lip is one step out
    on_ramp = dist <= DECK_BELOW_LIP / np.sin(k)
    board = np.empty((n, 3))
    board[:takeoff] = np.where(
        on_ramp[:, None],
        board0 - dist[:, None] * up_slope,
        np.column_stack([np.zeros_like(dist),
                         board0[1] - DECK_BELOW_LIP / np.tan(k)
                         - (dist - DECK_BELOW_LIP / np.sin(k)),
                         np.full_like(dist, deck_z)]))
    # Pitch eases from flat to the kicker's angle over 8 m, about a real
    # transition's 15 m radius (60 degrees a second at takeoff speed).
    ramp_len = DECK_BELOW_LIP / np.sin(k)
    pitch_pre = trick.kicker_deg * smoothstep((ramp_len + 4.0 - dist) / 8.0)
    # S-turns on the deck, straight for the last 25 m before the kicker.
    wavelength = v * trick.carve_period_s
    carve_pre = _carve(dist, trick.carve_m, wavelength, rng.uniform(0, 2 * np.pi))
    carve_pre *= 1 - smoothstep((ramp_len + 25.0 - dist) / 15.0)
    carve_pre *= smoothstep((dist[0] - dist) / 15.0) if takeoff else 1.0   # ease into the first turn
    board[:takeoff, 0] += carve_pre

    # --- ride-out: down the landing, onto the outrun, slowing to a stop ---
    k_land = np.deg2rad(trick.landing_deg)
    touch = position[landing] - rotation[landing][:, 2] * _drop(tuck[landing])
    t_post = t[landing + 1:] - t[landing]
    land_v = float(np.hypot(v * np.cos(k), v * np.sin(k) - GRAVITY * trick.air_s)) * 0.8
    reach = 48.0                                    # stop before the crowd
    decay = min(reach / land_v, max(trick.rideout_s, 0.5))
    run = land_v * decay * (1 - np.exp(-t_post / decay))
    # Terrain profile from the touchdown: landing slope, a knee, flat outrun.
    prof_s = np.array([0.0, LANDING_RUN_M, LANDING_RUN_M + 15.2, 400.0])
    prof_y = touch[1] + np.array([0.0, LANDING_RUN_M * np.cos(k_land),
                                  LANDING_RUN_M * np.cos(k_land) + 15.0,
                                  LANDING_RUN_M * np.cos(k_land) + 15.0 + 384.8])
    prof_z = touch[2] + np.array([0.0, -LANDING_RUN_M * np.sin(k_land),
                                  -LANDING_RUN_M * np.sin(k_land) - 3.0,
                                  -LANDING_RUN_M * np.sin(k_land) - 3.0])
    board[landing + 1:, 0] = touch[0]
    board[landing + 1:, 1] = np.interp(run, prof_s, prof_y)
    board[landing + 1:, 2] = np.interp(run, prof_s, prof_z)
    slope_post = np.rad2deg(np.interp(run, [0, LANDING_RUN_M - 3, LANDING_RUN_M + 12, 400],
                                      [k_land, k_land, 0.0, 0.0]))
    carve_post = _carve(run, trick.carve_m, wavelength, rng.uniform(0, 2 * np.pi))
    carve_post *= smoothstep((run - 6.0) / 20.0)
    board[landing + 1:, 0] += carve_post

    # Heading follows the turns; the body leans into them.
    yaw = np.zeros(n)
    roll = np.zeros(n)
    for sl, lateral in ((slice(0, takeoff), board[:takeoff, 0]),
                        (slice(landing + 1, n), board[landing + 1:, 0])):
        forward = board[sl, 1]
        if len(forward) < 3:
            continue
        dx = np.gradient(lateral)
        dy = np.maximum(np.gradient(forward), 1e-4)
        yaw[sl] = -np.rad2deg(np.arctan2(dx, dy))
        accel = np.gradient(np.gradient(lateral)) * fps * fps
        r = np.clip(np.rad2deg(np.arctan2(accel, GRAVITY)), -45.0, 45.0)
        # Finite differences are one-sided at a stretch's ends: hold the
        # neighbouring value there rather than read a spike.
        h = yaw[sl]
        r[:3], r[-3:] = r[3], r[-4]
        h[:2], h[-2:] = h[2], h[-3]
        # A rider takes a fraction of a second to change edge: low-pass both.
        roll[sl], yaw[sl] = _gauss(r, 0.15 * fps), _gauss(h, 0.15 * fps)

    jitter = None
    if trick.jitter_deg:
        # Faded out around the air, which is left exactly as generated.
        near = np.minimum(np.abs(t - t[takeoff]), np.abs(t - t[landing]))
        jitter = _smooth_noise(rng, n, fps, np.deg2rad(trick.jitter_deg))
        jitter *= smoothstep(near / 0.3)[:, None]
    final = rot_x(trick.landing_deg) @ rotation[landing]     # upright, landing facing
    for i in range(n):
        if takeoff <= i <= landing:
            continue
        if i < takeoff:
            base = rot_x(pitch_pre[i]) @ rot_z(yaw[i]) @ axis_angle(
                np.array([0.0, 1.0, 0.0]), np.deg2rad(roll[i])) @ stance
        else:
            base = rot_x(-slope_post[i - landing - 1]) @ rot_z(yaw[i]) @ axis_angle(
                np.array([0.0, 1.0, 0.0]), np.deg2rad(roll[i])) @ final
        if jitter is not None and np.linalg.norm(jitter[i]) > 1e-9:
            base = axis_angle(jitter[i], float(np.linalg.norm(jitter[i]))) @ base
        rotation[i] = base
        position[i] = board[i] + base[:, 2] * _drop(tuck[i])


def labels(trick: Trick, m: Motion) -> dict:
    """Exact answers, measured from the generated root motion.

    ``axis_spin_deg`` and ``axis_tilt_deg`` use the same definitions as
    ``features.py`` (dominant axis = normalised sum of per-frame rotation
    vectors over the clip), so a reading that disagrees with them is a
    measurement error, not a difference of definition. They are *not* the
    nominal spin: projecting onto one axis drops whatever rotation was not
    about it (a cork's axis moves; the rider levels from the kicker's pitch to
    the landing's), so even exact motion reads below ``spin_nominal_deg``.
    Vertical yaw is not labelled: it is undefined while the rider is inverted.
    """
    steps = np.array([
        rotation_vector(m.rotation[i + 1] @ m.rotation[i].T)
        for i in range(len(m.t) - 1)
    ])
    net = steps.sum(axis=0)
    axis = net / np.linalg.norm(net)
    axis_spin = float(np.rad2deg((steps @ axis).sum()))
    axis_tilt = float(np.rad2deg(np.arccos(min(1.0, abs(float(axis @ UP))))))
    tilt = tilt_series(m.rotation[m.takeoff_frame:m.landing_frame + 1])
    held = np.flatnonzero(m.grab > 0.5)
    return {
        "group": trick.group or trick.name,
        "name": trick.name,
        "spin_nominal_deg": trick.spin_deg,
        "kind": trick.kind,
        "direction": trick.direction,
        "stance": trick.stance,
        "spin_sign": trick.spin_sign,
        "rotation_sense": "counter-clockwise" if trick.spin_sign > 0 else "clockwise",
        "grab": trick.grab,
        "camera_angle": trick.camera_angle,
        "slowmo_factor": trick.slowmo,
        "axis_tilt_target_deg": trick.tilt_max_deg,
        "inversions": count_inversions(tilt),
        "max_tilt_deg": round(float(tilt.max()), 2),
        "axis_spin_deg": round(abs(axis_spin), 2),
        "axis_tilt_deg": round(axis_tilt, 2),
        "total_rotation_deg": round(float(np.rad2deg(np.linalg.norm(steps, axis=1).sum())), 2),
        "takeoff_frame": m.takeoff_frame,
        "landing_frame": m.landing_frame,
        "air_frames": m.landing_frame - m.takeoff_frame,
        "air_s_real": trick.air_s,
        # Frames with the hand more than half way onto the board; -1 if none.
        "grab_start_frame": int(held[0]) if len(held) else -1,
        "grab_end_frame": int(held[-1]) if len(held) else -1,
        "frames": len(m.t),
        "output_fps": OUTPUT_FPS,
    }


# --- the validation set -----------------------------------------------------

# (spin, kind, camera angle, direction). Twenty clips: every flat spin from
# 360 to 1440, corks 540-1080, double corks 1080-1440, and four frontside
# double cork 1080s from every camera, because that is the trick the open
# discrepancy in docs/validation.md is about.
PLAN = [
    (360, "flat", "side", "frontside"),
    (540, "flat", "front_quarter", "backside"),
    (720, "flat", "back_quarter", "frontside"),
    (900, "flat", "low_landing", "backside"),
    (1080, "flat", "side", "frontside"),
    (1440, "flat", "low_landing", "backside"),
    (720, "flat", "low_landing", "frontside"),
    (540, "cork", "side", "backside"),
    (720, "cork", "front_quarter", "frontside"),
    (900, "cork", "back_quarter", "backside"),
    (1080, "cork", "low_landing", "frontside"),
    (720, "cork", "side", "frontside"),
    (1080, "cork", "front_quarter", "backside"),
    (1080, "double_cork", "side", "frontside"),
    (1080, "double_cork", "front_quarter", "frontside"),
    (1080, "double_cork", "back_quarter", "frontside"),
    (1080, "double_cork", "low_landing", "frontside"),
    (1260, "double_cork", "side", "backside"),
    (1440, "double_cork", "front_quarter", "frontside"),
    (1440, "double_cork", "back_quarter", "backside"),
]


def camera_params(angle: str, rng: np.random.Generator) -> dict:
    """Where a broadcast camera for ``angle`` stands, relative to the lip.

    Distances are those of fixed broadcast positions at a big-air venue: tens
    of metres away on long lenses, so the camera pans and zooms to follow.
    """
    base = {
        "side": (0.0, 50.0, 4.0),             # square to the jump
        "front_quarter": (45.0, 55.0, 2.0),   # off the side, down the hill
        "back_quarter": (-45.0, 45.0, 8.0),   # off the side, behind the kicker
        "low_landing": (70.0, 70.0, -25.0),   # at the landing, looking up
    }[angle]
    azimuth, distance, height = base
    return {
        # Azimuth from +X (square to the jump) toward +Y (down the landing).
        "azimuth_deg": float(azimuth + rng.uniform(-10, 10)),
        "distance_m": float(distance * rng.uniform(0.85, 1.2)),
        "height_m": float(height + rng.uniform(-2, 2)),
        # Rider height / frame height. Broadcast runs from wide to tight, and
        # the tight end is a long lens, far from the focal the NIM and
        # camera.py assume; the spread is what lets that error be measured.
        "frame_fill": float(rng.uniform(0.15, 0.45)),
        "lag_s": float(rng.uniform(0.08, 0.22)),        # operator reaction
        "jitter_deg": float(rng.uniform(0.02, 0.08)),   # fluid head wobble
        "zoom_lag_s": float(rng.uniform(0.3, 0.7)),
    }


def body_params(rng: np.random.Generator) -> dict:
    """MPFB macro targets: varied body shape so no one physique is learned."""
    return {
        "gender": float(rng.uniform(0.0, 1.0)),
        "age": float(rng.uniform(0.4, 0.6)),
        "muscle": float(rng.uniform(0.35, 0.75)),
        "weight": float(rng.uniform(0.35, 0.7)),
        "height": float(rng.uniform(0.35, 0.7)),
        "proportions": float(rng.uniform(0.4, 0.7)),
    }


def plan(seed: int = 0) -> list[Trick]:
    """The 20-trick validation set, with seeded random style per trick."""
    tricks = []
    for index, (spin, kind, angle, direction) in enumerate(PLAN):
        rng = np.random.default_rng(seed * 1000 + index)
        grab = str(rng.choice(GRABS[1:])) if rng.uniform() < 0.8 else "none"
        start = float(rng.uniform(0.2, 0.4))
        trick = Trick(
            name=f"synth{index + 1:02d}_{kind}_{spin}",
            spin_deg=spin, kind=kind, direction=direction,
            stance=str(rng.choice(["regular", "goofy"])),
            grab=grab, camera_angle=angle,
            slowmo=float(rng.choice([1.0, 2.0, 3.0, 4.0])),
            seed=seed * 1000 + index,
            air_s=round(float(rng.uniform(2.5, 3.2)), 2),
            prewind_deg=round(float(rng.uniform(15, 45)), 1),
            tuck_depth=round(float(rng.uniform(0.35, 0.85)), 2),
            grab_window=(round(start, 2), round(start + float(rng.uniform(0.25, 0.45)), 2)),
            tilt_azimuth_deg=round(float(rng.uniform(-30, 30) + rng.choice([0, 180])), 1),
            body=body_params(rng),
            camera=camera_params(angle, rng),
        )
        solve_tilt(trick)
        tricks.append(trick)
    return tricks


# --- the training set -------------------------------------------------------

# Spins each kind can carry. A cork needs its inversion clear of the landing
# (180 < spin - 90), a double cork two of them; past 1440 the air times here
# would need spin rates no rider holds.
SPINS = {
    "flat": (180, 360, 540, 720, 900, 1080, 1260, 1440),
    "cork": (540, 720, 900, 1080, 1260, 1440),
    "double_cork": (900, 1080, 1260, 1440),
}
SLOWMO = (1.0, 2.0, 3.0, 4.0)
SCALE_PREFIX = "synthb"


def _balanced(values, count: int, rng: np.random.Generator) -> list:
    """``count`` draws covering ``values`` evenly, in shuffled blocks.

    Each run of ``len(values)`` draws is a permutation of ``values``, so any
    prefix is balanced to within one per value: a set cut short by its
    budget is still balanced.
    """
    out: list = []
    while len(out) < count:
        out += [values[i] for i in rng.permutation(len(values))]
    return out[:count]


def scale_plan(count: int, seed: int = 1, start: int = 1, group_size: int = 4,
               prefix: str = SCALE_PREFIX, long_ground: bool = False) -> list[Trick]:
    """A balanced, randomised training set of ``count`` tricks.

    Balanced across the factored labels the classifier learns: kind (flat,
    cork, double cork) in thirds and spin evenly within each kind; direction,
    stance, grab, camera position and slow-motion factor each evenly. Each
    factor is shuffled on its own, so no two are correlated by construction.

    Clips come in **split groups** of ``group_size``: one rider (body, outfit,
    board) at one venue (dressing, sun), as one athlete's runs at one event.
    A group must stay on one side of a train/test split, or a model can learn
    the rider and the venue instead of the trick.

    Everything else is drawn per clip and wider than ``plan``: air time,
    takeoff speed, kicker and landing angles, how long the approach and the
    ride-out last (so every clip has frames on the ground either side of the
    air), pre-wind, tuck, grab timing, and the camera's placement and framing.
    Deterministic in ``seed``, and every prefix of the list is balanced, so
    plan the whole set at once and render it in order. A second call with a
    new ``start`` grows a set (numbers and groups continue from it) without
    renaming what exists.

    ``long_ground`` gives each clip far more time on the snow than in the
    air (about 30% airborne against about 60%), riding the venue's terrain
    (``_ground``): half stand still at the top first, most carve S-turns
    on the deck and the outrun, and every rider wobbles a little. A
    segmenter trained only on short run-ins learns that riding is flying.
    """
    rng = np.random.default_rng(seed * 7919 + start)
    kinds = _balanced(KINDS, count, rng)
    spin_queue = {k: _balanced(SPINS[k], kinds.count(k), rng) for k in KINDS}
    directions = _balanced(("frontside", "backside"), count, rng)
    stances = _balanced(("regular", "goofy"), count, rng)
    grabs = _balanced(GRABS, count, rng)
    angles = _balanced(CAMERA_ANGLES, count, rng)
    slowmos = _balanced(SLOWMO, count, rng)

    tricks = []
    for i in range(count):
        number = start + i
        g = (number - 1) // group_size
        group_rng = np.random.default_rng(seed * 100_003 + g)
        body = body_params(group_rng)
        rider_seed = int(group_rng.integers(1, 2**31))
        venue_seed = int(group_rng.integers(1, 2**31))
        kind = kinds[i]
        spin = int(spin_queue[kind].pop(0))
        clip_seed = seed * 1_000_003 + number
        r = np.random.default_rng(clip_seed)
        grab_start = float(r.uniform(0.1, 0.55))
        grab_len = float(r.uniform(0.15, 0.45))
        # Bigger spins need more air: a 1440 in under 2.5 s is not ridden.
        air_lo = 2.0 + 0.5 * (spin - 180) / 1260
        trick = Trick(
            name=f"{prefix}{number:04d}_{kind}_{spin}",
            spin_deg=spin, kind=kind, direction=directions[i], stance=stances[i],
            grab=grabs[i], camera_angle=angles[i], slowmo=float(slowmos[i]),
            seed=clip_seed,
            air_s=round(float(r.uniform(air_lo, 3.3)), 2),
            approach_s=round(float(r.uniform(0.5, 1.3)), 2),
            rideout_s=round(float(r.uniform(0.5, 1.3)), 2),
            kicker_deg=round(float(r.uniform(25.0, 31.0)), 1),
            landing_deg=round(float(r.uniform(29.0, 35.0)), 1),
            takeoff_speed=round(float(r.uniform(14.5, 17.5)), 2),
            prewind_deg=round(float(r.uniform(5, 60)), 1),
            tuck_depth=round(float(r.uniform(0.25, 0.9)), 2),
            grab_window=(round(grab_start, 2), round(min(0.85, grab_start + grab_len), 2)),
            tilt_azimuth_deg=round(float(r.uniform(-40, 40) + r.choice([0, 180])), 1),
            body=body,
            camera=camera_params(angles[i], r),
            group=f"{prefix}g{g:03d}",
            rider_seed=rider_seed,
            venue_seed=venue_seed,
        )
        if long_ground:
            g = np.random.default_rng(clip_seed + 5)
            trick.terrain = True
            trick.standing_s = round(float(g.uniform(0.5, 1.5)), 2) if g.uniform() < 0.5 else 0.0
            trick.approach_s = round(trick.standing_s + float(g.uniform(1.6, 4.0)), 2)
            trick.rideout_s = round(float(g.uniform(2.0, 4.5)), 2)
            # Turns sized by how hard the rider leans into them (lateral
            # acceleration a, lean atan(a / g) of 17 to 39 degrees): at speed
            # v and wavelength v * period, a half-width A needs a = A (2 pi / period)^2.
            trick.carve_period_s = round(float(g.uniform(2.0, 4.0)), 2)
            lateral = float(g.uniform(3.0, 8.0))
            half = lateral * (trick.carve_period_s / (2 * np.pi)) ** 2
            trick.carve_m = round(min(3.5, half), 2) if g.uniform() < 0.7 else 0.0
            trick.jitter_deg = round(float(g.uniform(0.5, 3.0)), 2)
        _solve_or_redraw(trick, r)
        tricks.append(trick)
    return tricks


def _solve_or_redraw(trick: Trick, rng: np.random.Generator, tries: int = 6) -> None:
    """``solve_tilt``, redrawing the air time if no lean works for this one."""
    for attempt in range(tries):
        try:
            solve_tilt(trick)
            return
        except ValueError:
            if attempt == tries - 1:
                raise
            trick.air_s = round(float(rng.uniform(2.4, 3.3)), 2)
            trick.tilt_azimuth_deg = round(float(rng.uniform(0, 360)), 1)
