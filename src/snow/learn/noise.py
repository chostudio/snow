"""A noise model for the NIM on real footage, measured, and applied to renders.

Rendered riders come out of the NIM far steadier than broadcast ones. A model
trained on renders then reads ordinary broadcast jitter as a rider turning in
the air: on `run1` the first segmenters called the rider standing at the top
of the halfpipe airborne. This module measures what real NIM output looks
like on stretches where the rider is plainly on the snow, compares it with
the synthetic clips' ground frames, and turns the difference into a training
augmentation in the segmenter's own input channels (``inputs.CHANNELS``).

What is measured (``measure``), per stretch of frames with a body:

* ``rot_step``: the body frame's rotation per frame, in degrees: percentiles
  and the autocorrelation of its fast part (the step minus a 9-frame running
  median), which says whether the jitter is white or slow wander.
* ``elev_step`` / ``azim_step``: the ray to the rider in the
  camera-compensated world frame; on real footage this carries the camera
  estimate's own error.
* keypoint jitter: body-frame joint positions (torso lengths) minus their
  7-frame running mean.
* ``grab`` and ``tilt``: medians, i.e. systematic offsets rather than noise.
* dropouts: frames without a body inside a shot, and the mean hole length.

The fitted model (``fit``) is a per-axis AR(1) rotation-vector noise added to
the true step, AR(1) noise on the two ray rates, a per-segment grab offset and
reference tilt, and holes at the measured rate. ``REAL_NOISE`` holds the
numbers the shipped weights were trained with; docs/segmenter.md says which
frames they came from. They were measured on three clips, two of which are
also the only real sanity checks there are, so treat a real-footage
improvement as indicative, not as a measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import inputs as I

ROT = I.CHANNELS.index("rot_step")
ELEV = I.CHANNELS.index("elev_step")
AZIM = I.CHANNELS.index("azim_step")
GRAB = I.CHANNELS.index("grab")
TILT = I.CHANNELS.index("cos_tilt")
POSE = [k for k, c in enumerate(I.CHANNELS) if c.startswith("pose")]
# Channel units back to physical ones (see inputs._ROT_STEP_SCALE etc).
ROT_DEG = 10.0      # rot_step channel -> degrees per frame
RAY_RAD = 0.01      # ray step channel -> radians per frame


@dataclass
class NoiseModel:
    """Parameters in physical units; ``apply`` converts to channels."""

    rot_sigma_deg: float = 0.0     # per-axis std of the rotation-vector noise, degrees
    rot_phi: float = 0.0           # its AR(1) coefficient frame to frame
    ray_sigma_rad: float = 0.0     # std of the extra noise on each ray rate, radians per frame
    ray_phi: float = 0.0
    grab_offset: float = 0.0       # real minus synthetic ground grab, torso lengths
    grab_sigma: float = 0.0        # extra per-frame grab jitter, torso lengths
    tilt_offset_deg: float = 0.0   # reference tilt drawn per segment, up to this
    hole_rate: float = 0.0         # holes started per frame with a body
    hole_mean: float = 0.0         # mean hole length, frames
    source: str = ""

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, src: Path) -> "NoiseModel":
        return cls(**json.loads(Path(src).read_text()))


# Measured with `snow segmenter noise run1:460-495 run1:590-628 trick:262-303`
# against the long-ground synthetic clips (docs/segmenter.md, "Calibrated
# noise"); dropouts over the whole of run1 and trick. run1 0-200, the
# standing start and deck riding, was kept out for the sanity check. The ray
# rates need nothing: the renders' camera estimate is already noisier.
REAL_NOISE = NoiseModel(
    rot_sigma_deg=1.5, rot_phi=0.8, ray_sigma_rad=0.0, ray_phi=0.621,
    grab_offset=-0.079, grab_sigma=0.0357, tilt_offset_deg=8.1,
    hole_rate=0.0024, hole_mean=58.0,
    source="fitted on run1:460-495, run1:590-628, trick:262-303",
)
# Share of training segments that get it; the rest stay as rendered.
APPLY_PROBABILITY = 0.8


# --- measuring -----------------------------------------------------------------


def _running(values: np.ndarray, width: int, fn) -> np.ndarray:
    half = width // 2
    out = np.empty_like(values)
    for k in range(len(values)):
        out[k] = fn(values[max(0, k - half):k + half + 1], axis=0)
    return out


def acf(values: np.ndarray, lags: int = 5) -> list[float]:
    v = values - values.mean()
    denom = float((v * v).sum()) or 1.0
    return [round(float((v[:-k] * v[k:]).sum() / denom), 3) for k in range(1, lags + 1)]


def stretch_stats(x: np.ndarray) -> dict:
    """Raw per-stretch series from one contiguous run of channel rows."""
    present = x[:, I.PRESENT] > 0
    rows = x[present]
    rot = rows[:, ROT] * ROT_DEG
    fast = rot - _running(rot, 9, np.median)
    pose = rows[:, POSE]
    resid = pose - _running(pose, 7, np.mean)
    return {
        "rot_deg": rot, "rot_fast": fast,
        "elev": rows[:, ELEV] * RAY_RAD, "azim": rows[:, AZIM] * RAY_RAD,
        "kp_resid": resid, "grab": rows[:, GRAB],
        "grab_fast": rows[:, GRAB] - _running(rows[:, GRAB], 7, np.mean),
        "tilt_deg": np.rad2deg(np.arccos(np.clip(rows[:, TILT], -1, 1))),
        # Interior holes: absent rows between present ones.
        "holes": [b - a for a, b in _holes(present)], "frames": int(present.sum()),
    }


def _holes(present: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(present)
    if len(idx) < 2:
        return []
    inner = ~present[idx[0]:idx[-1] + 1]
    out, start = [], None
    for k, gap in enumerate(inner):
        if gap and start is None:
            start = k
        if not gap and start is not None:
            out.append((start, k))
            start = None
    return out


def summarise(stretches: list[dict]) -> dict:
    cat = {k: np.concatenate([s[k] for s in stretches]) for k in
           ("rot_deg", "rot_fast", "elev", "azim", "grab", "grab_fast", "tilt_deg")}
    resid = np.concatenate([s["kp_resid"] for s in stretches])
    holes = [h for s in stretches for h in s["holes"]]
    frames = sum(s["frames"] for s in stretches)
    pct = (10, 25, 50, 75, 90, 99)
    return {
        "frames": frames,
        "rot_deg_percentiles": dict(zip(map(str, pct),
                                        np.round(np.percentile(cat["rot_deg"], pct), 3).tolist())),
        "rot_fast_acf": np.round(np.mean([acf(s["rot_fast"]) for s in stretches
                                          if len(s["rot_fast"]) > 8], axis=0), 3).tolist(),
        "rot_deg_acf": np.round(np.mean([acf(s["rot_deg"]) for s in stretches
                                         if len(s["rot_deg"]) > 8], axis=0), 3).tolist(),
        "elev_std_rad": round(float(cat["elev"].std()), 5),
        "azim_std_rad": round(float(cat["azim"].std()), 5),
        "azim_acf": np.round(np.mean([acf(s["azim"]) for s in stretches
                                      if len(s["azim"]) > 8], axis=0), 3).tolist(),
        "keypoint_jitter_torso": round(float(np.median(resid.std(axis=0))), 4),
        "keypoint_jitter_p90_joint": round(float(np.percentile(resid.std(axis=0), 90)), 4),
        "grab_median": round(float(np.median(cat["grab"])), 3),
        "grab_jitter": round(float(cat["grab_fast"].std()), 4),
        "tilt_median_deg": round(float(np.median(cat["tilt_deg"])), 1),
        "holes_per_frame": round(len(holes) / max(frames, 1), 4),
        "hole_mean_frames": round(float(np.mean(holes)), 1) if holes else 0.0,
    }


# --- fitting ------------------------------------------------------------------


def ar1(n: int, sigma: float, phi: float, rng: np.random.Generator, dims: int = 1) -> np.ndarray:
    """Stationary AR(1) noise with marginal std ``sigma``."""
    out = np.empty((n, dims))
    innov = sigma * np.sqrt(max(1 - phi * phi, 1e-6))
    out[0] = rng.normal(0, sigma, dims)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + rng.normal(0, innov, dims)
    return out


def noisy_rot(rot_deg: np.ndarray, sigma: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """|true step + 3D AR(1) rotation-vector noise|, in degrees.

    The channel keeps only the step's size, so the true step is placed on one
    axis; a random direction would change nothing in distribution.
    """
    n = ar1(len(rot_deg), sigma, phi, rng, 3)
    n[:, 0] += rot_deg
    return np.linalg.norm(n, axis=1)


def fit(real: dict, synth_ground: list[dict], seed: int = 0) -> NoiseModel:
    """Choose the noise that turns synthetic ground into real ground.

    Grid search on the rotation step: match the real median, 90th
    percentile and lag-1 autocorrelation when added to the synthetic ground
    frames. The ray rates take the variance real has in excess of synthetic,
    and the lag-1 autocorrelation real shows.
    """
    rng = np.random.default_rng(seed)
    target = np.array([real["rot_deg_percentiles"]["50"], real["rot_deg_percentiles"]["90"]])
    target_acf = real["rot_deg_acf"][0]
    best = None
    for sigma in np.arange(0.1, 4.01, 0.1):
        for phi in (0.0, 0.2, 0.4, 0.6, 0.8):
            sims = [noisy_rot(s["rot_deg"], sigma, phi, rng) for s in synth_ground
                    if len(s["rot_deg"]) > 8]
            cat = np.concatenate(sims)
            got = np.percentile(cat, [50, 90])
            got_acf = float(np.mean([acf(v)[0] for v in sims]))
            err = float(np.sum(((got - target) / target) ** 2) + (got_acf - target_acf) ** 2)
            if best is None or err < best[0]:
                best = (err, sigma, phi)
    synth = summarise(synth_ground)
    ray_var = max(real["azim_std_rad"] ** 2 - synth["azim_std_rad"] ** 2,
                  real["elev_std_rad"] ** 2 - synth["elev_std_rad"] ** 2, 0.0)
    return NoiseModel(
        rot_sigma_deg=round(float(best[1]), 3), rot_phi=float(best[2]),
        ray_sigma_rad=round(float(np.sqrt(ray_var)), 5),
        ray_phi=float(np.clip(real["azim_acf"][0], 0.0, 0.9)),
        grab_offset=round(real["grab_median"] - synth["grab_median"], 3),
        grab_sigma=round(float(np.sqrt(max(real["grab_jitter"] ** 2 - synth["grab_jitter"] ** 2,
                                            0.0))), 4),
        tilt_offset_deg=round(max(real["tilt_median_deg"] - synth["tilt_median_deg"], 0.0), 1),
        hole_rate=real["holes_per_frame"], hole_mean=real["hole_mean_frames"],
    )


def clip_dropouts(names: list[str]) -> dict:
    """Tracker losses over whole real clips, inside shots: dropout is not a
    property of ground stretches, and the stretches were chosen with a body."""
    holes, frames = [], 0
    for name in names:
        x, segment = real_channels(name)
        for seg in np.unique(segment):
            present = x[segment == seg, I.PRESENT] > 0
            frames += int(present.sum())
            holes += [b - a for a, b in _holes(present)]
    return {"holes_per_frame": round(len(holes) / max(frames, 1), 4),
            "hole_mean_frames": round(float(np.mean(holes)), 1) if holes else 0.0,
            "holes": holes, "frames": frames}


# --- applying -----------------------------------------------------------------


def apply(x: np.ndarray, rng: np.random.Generator, seen: np.ndarray,
          model: NoiseModel = REAL_NOISE, strength: float | None = None) -> None:
    """Add real-footage NIM noise to one segment's raw channels, in place.

    ``strength`` scales every term (drawn from 0.5 to 1.5 when None), so the
    model sees a spread around the measured level, not one value.
    """
    n = len(x)
    if n == 0:
        return
    s = rng.uniform(0.5, 1.5) if strength is None else strength
    if model.rot_sigma_deg:
        rot = noisy_rot(x[:, ROT] * ROT_DEG, s * model.rot_sigma_deg, model.rot_phi, rng)
        x[:, ROT] = rot / ROT_DEG
    if model.ray_sigma_rad:
        ray = ar1(n, s * model.ray_sigma_rad, model.ray_phi, rng, 2) / RAY_RAD
        x[:, ELEV] += ray[:, 0]
        x[:, AZIM] += ray[:, 1]
    if model.grab_offset or model.grab_sigma:
        x[:, GRAB] = np.clip(x[:, GRAB] + s * model.grab_offset
                             + rng.normal(0, s * model.grab_sigma + 1e-9, n), 0.0, I._GRAB_CLIP)
    if model.tilt_offset_deg:
        # A reference "up" that is not level reads as tilt the rider never had.
        tilt = np.rad2deg(np.arccos(np.clip(x[:, TILT], -1, 1)))
        tilt = np.abs(tilt + rng.choice((-1, 1)) * rng.uniform(0, s * model.tilt_offset_deg))
        x[:, TILT] = np.cos(np.deg2rad(np.minimum(tilt, 180.0)))
    if model.hole_rate and model.hole_mean:
        _punch_holes(x, rng, seen, model.hole_rate * s, model.hole_mean)
    x[~seen] = 0.0


def _punch_holes(x: np.ndarray, rng: np.random.Generator, seen: np.ndarray,
                 rate: float, mean: float) -> None:
    """Frames the tracker lost, as ``frame_channels`` writes them: present 0,
    levels interpolated, no body rotation across the hole."""
    idx = np.flatnonzero(seen)
    if len(idx) < 3:
        return
    lo, hi = idx[0] + 1, idx[-1] - 1
    for start in np.flatnonzero(rng.random(hi - lo + 1) < rate) + lo:
        length = int(rng.geometric(1.0 / max(mean, 1.0)))
        a, b = start, min(start + length, hi + 1)
        if b <= a:
            continue
        for c in [TILT, GRAB] + POSE:
            x[a:b, c] = np.linspace(x[a - 1, c], x[b, c], b - a + 2)[1:-1]
        for c in (ELEV, AZIM):
            x[a:b, c] = x[a:b, c].mean()
        x[a:b, ROT] = 0.0
        x[b, ROT] = 0.0
        x[a:b, I.PRESENT] = 0.0
        x[a:b, I.CHANNELS.index("confidence")] = 0.0


# --- the data -----------------------------------------------------------------


def real_channels(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Channels and segment ids for a real clip, from its pipeline output.

    Poses and camera are read from ``data/poses`` or, for clips posed only
    for this measurement, ``data/models/segmenter/real``.
    """
    import pandas as pd

    from .. import config as cfg
    from .. import features as F
    from .. import meta as mt
    from ..camera import CameraTrack

    extra = cfg.DATA_DIR / "models" / "segmenter" / "real"
    poses = extra / f"{name}.json"
    poses = poses if poses.exists() else cfg.pose_path(name)
    cam = extra / f"{name}.camera.npz"
    cam = cam if cam.exists() else cfg.POSES_DIR / f"{name}.camera.npz"
    track = CameraTrack.load(cam) if cam.exists() else None
    table = cfg.POSES_DIR / f"{name}.features.csv"
    df = (pd.read_csv(table) if table.exists() and poses.parent == cfg.POSES_DIR
          else F.build(poses, mt.load(name), camera=track, segmenter="off"))
    x = I.frame_channels(F.load_poses(poses), df, track.rotations if track else None)
    return x, df["segment"].to_numpy(dtype=int)


def parse_span(text: str) -> tuple[str, int, int]:
    """``run1:460-495`` -> ("run1", 460, 495), end exclusive."""
    name, frames = text.split(":")
    a, b = frames.split("-")
    return name, int(a), int(b)


def real_stretches(spans: list[str]) -> list[dict]:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out = []
    for text in spans:
        name, a, b = parse_span(text)
        if name not in cache:
            cache[name] = real_channels(name)
        x, segment = cache[name]
        # A stretch never spans a cut: split it at every segment change.
        for seg in np.unique(segment[a:b]):
            rows = np.flatnonzero(segment[a:b] == seg) + a
            out.append(stretch_stats(x[rows]))
    return out


def synthetic_ground(clips: list[dict], variants: tuple[str, ...] = ("long-ground",),
                     min_frames: int = 10) -> list[dict]:
    """Every observed ground run of the synthetic clips, per segment."""
    from .events import runs

    out = []
    for c in clips:
        if variants and c.get("variant") not in variants:
            continue
        seen = I.observed(c["x"][:, I.PRESENT], c["segment"])
        for a, b in runs(seen & ~c["y"], c["segment"]):
            if b - a >= min_frames:
                out.append(stretch_stats(c["x"][a:b]))
    return out
