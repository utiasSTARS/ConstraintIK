"""Joint trajectories for the real-robot recordings on the project page.

For each recorded condition this re-decodes, from the released checkpoint and at a finer resolution, the
two motions that the page's widgets show (the pair and anchor are read from the web export of
scripts/export_web_data.py, so the robot executes exactly what the page draws):

  {robot}_{c}_line.csv   the straight latent path x_a -> x_b at the page's default t_bar
  {robot}_{c}_knob.csv   the sweep of the leading tangent knob from -1 to +1 (Score-SVD direction at
                         t = 15, applied at t_dec = 50 and decoded from there)
  ur5_se3_branches.csv   (only with --robots ur5) the static IK branches of the UR5 SE(3) target

Each CSV has columns t [s], s (path position: alpha in [0, 1] for lines, knob value in [-1, 1] for knobs),
q1..qn [rad] and qd1..qdn [rad/s], sampled at --rate Hz. The path is followed at a constant rate in s,
scaled so that the fastest joint moves at --vmax, with smooth ramps of --ramp seconds at both ends
(velocity follows a smoothstep, so acceleration is continuous). summary.json records per trajectory the
duration, peak joint speed, joint-limit and table-clearance checks, the start pose, a suggested camera
pose matching the page's 3D view, and `timing` (normalized clip time at 21 evenly spaced path positions),
which the page uses to scrub the recorded video by path position.

    python scripts/export_robot_trajectories.py --web_data outputs/web_data --out outputs/robot_trajectories
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from constraint_ik import Target, errors, get_robot, load_checkpoint
from constraint_ik.constraints import codim, pos_axes
from constraint_ik.id_estimation import score_matrix
from constraint_ik.interpolation import lerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_web_data import (CKPT, CLEARANCE, FRANKA_HW, K_KNOB, KNOB_STEP, LIMIT_MARGIN, T_DEC, T_KNOB,  # noqa: E402
                             TARGETS, clearance, hw_margin, joint_centres, knob_noise, near, unwrap_rows)

RECORD = {
    "franka": ["plane", "posonly", "line_orient", "se3"],
    "ur5": ["plane", "line", "line_orient", "se3"],   # only with --robots ur5
}
# Page camera: azimuth / elevation (rad) of the 3D widgets, about the IKFast base frame.
CAMERA = {"ur5": (-0.75, 0.38), "franka": (-0.95, 0.38)}


def smooth_time_scaling(T_c: float, ramp: float, rate: float):
    """x(t) in [0, 1] and dx/dt: speed 1 / T_c between smoothstep velocity ramps of `ramp` s at both
    ends (continuous acceleration); total duration T = T_c + ramp."""
    T = T_c + ramp
    t = np.append(np.arange(0.0, T, 1.0 / rate), T)
    tf = np.linspace(0.0, T, 20001)
    u = np.clip(tf / ramp, 0, 1)
    w = np.clip((T - tf) / ramp, 0, 1)
    v = np.minimum(u * u * (3 - 2 * u), w * w * (3 - 2 * w))
    x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tf))])
    area = x[-1]
    return t, np.interp(t, tf, x / area), np.interp(t, tf, v / area), T


def parametrize(q_path: np.ndarray, s_grid: np.ndarray, vmax: float, ramp: float, rate: float):
    """Time-parametrize a joint path q(s), s increasing, at constant rate in s with smooth ramps, so the
    fastest joint peaks at vmax. Returns t, s(t), q(t), qd(t), duration and the normalized clip time at
    21 evenly spaced path positions."""
    x_grid = (s_grid - s_grid[0]) / (s_grid[-1] - s_grid[0])
    spl = CubicSpline(x_grid, q_path, axis=0)
    dqdx = float(np.abs(spl(np.linspace(0, 1, 4001), 1)).max())
    T_c = max(dqdx / vmax, 1.0)
    t, x, dxdt, T = smooth_time_scaling(T_c, ramp, rate)
    q = spl(x)
    qd = spl(x, 1) * dxdt[:, None]
    s_val = s_grid[0] + x * (s_grid[-1] - s_grid[0])
    tau = np.interp(np.linspace(0, 1, 21), x, t) / T
    return t, s_val, q, qd, T, tau


def shift_home(q: np.ndarray, robot: str) -> np.ndarray:
    """Shift each joint by a multiple of 2 pi so the start lies in (-pi, pi] (UR5 only; continuity kept)."""
    if robot != "ur5":
        return q
    k = np.round(q[0] / (2 * math.pi))
    return q - 2 * math.pi * k[None]


def checks(robot: str, q: np.ndarray, qd: np.ndarray) -> dict:
    out = {"min_clearance_m": round(clearance(robot, q), 3),
           "peak_joint_speed_rad_s": round(float(np.abs(qd).max()), 3),
           "hw_limit_margin_rad": round(hw_margin(robot, q), 3)}
    if robot == "franka":
        for name, (lo, hi) in FRANKA_HW.items():
            out[f"margin_{name}_rad"] = round(float(np.minimum(q - np.array(lo), np.array(hi) - q).min()), 3)
    return out


def safe_knob_range(robot: str, q: np.ndarray, s_grid: np.ndarray) -> float:
    """Largest a <= 1 such that the knob sweep over [-a, a] keeps the limit margin and the clearance."""
    for a in np.linspace(1.0, 0.2, 17):
        m = np.abs(s_grid) <= a + 1e-9
        if hw_margin(robot, q[m]) >= LIMIT_MARGIN and clearance(robot, q[m]) >= CLEARANCE:
            return float(round(a, 3))
    return 0.0


def camera(robot: str, q: np.ndarray, hfov_deg: float = 60.0, aspect: float = 16 / 9) -> dict:
    """Camera pose that frames the motion like the page's 3D view (IKFast base frame, metres)."""
    pts = joint_centres(robot, q[:: max(1, len(q) // 40)]).reshape(-1, 3)
    pts = np.vstack([pts, np.zeros((1, 3))])
    c = 0.5 * (pts.min(0) + pts.max(0))
    r = float(np.linalg.norm(pts - c, axis=1).max()) + 0.12
    az, el = CAMERA[robot]
    vfov = 2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect)
    dist = 1.1 * r / math.sin(vfov / 2)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return {"look_at": np.round(c, 3).tolist(), "position": np.round(c + dist * d, 3).tolist(),
            "distance_m": round(dist, 2), "azimuth_deg": round(math.degrees(az), 1),
            "elevation_deg": round(math.degrees(el), 1), "assumed_hfov_deg": hfov_deg}


def write_csv(path: str, t, s, q, qd) -> None:
    n = q.shape[1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s"] + [f"q{i + 1}" for i in range(n)] + [f"qd{i + 1}" for i in range(n)])
        for row in zip(t, s, q, qd):
            w.writerow([f"{row[0]:.4f}", f"{row[1]:.5f}"] + [f"{v:.6f}" for v in row[2]] + [f"{v:.6f}" for v in row[3]])


@torch.no_grad()
def line_path(ik, robot: str, c: str, web: str, n: int) -> tuple[np.ndarray, dict]:
    meta = json.load(open(os.path.join(web, robot, c, "interp.json")))
    src = meta["source"]
    data = np.load(TARGETS.format(robot=robot, c=c))
    k = src["pair_index"]
    target = Target.from_pose(c, data["pos"][data["pair_target"][k]], data["R"][data["pair_target"][k]])
    cond = ik.cond(target)
    t_bar = meta["t_default"]
    z_a, z_b = (ik.invert(data[key][k], t_bar, cond) for key in ("q_a", "q_b"))
    q = ik.decode_from(lerp(z_a, z_b, n), t_bar, cond).cpu().numpy().astype(np.float64)
    q = unwrap_rows(q)
    q = q + (near(q[0], np.array(meta["anchors"]["a"])) - q[0])
    return q, {"target": target, "t_bar": t_bar, "pair_index": k}


@torch.no_grad()
def knob_path(ik, robot: str, c: str, web: str, n: int) -> tuple[np.ndarray, dict]:
    K = json.load(open(os.path.join(web, robot, c, "knobs.json")))
    lead = K["knobs"]["lead"]
    tg = K["target"]
    target = Target.from_pose(c, np.array(tg["pos"]), np.array(tg["R"]))
    cond = ik.cond(target)
    q0 = np.array(K["q0"])
    S = score_matrix(ik, ik.encode(q0), cond, T_KNOB, K_KNOB, knob_noise(ik))
    Vh = torch.linalg.svd(S, full_matrices=False)[2]
    t_dec = K.get("t_dec", T_DEC)
    z = ik.invert(q0, t_dec, cond)
    a = torch.linspace(-1, 1, n, device=ik.device)[:, None]
    q = ik.decode_from(z + KNOB_STEP * a * Vh[lead][None], t_dec, cond).cpu().numpy().astype(np.float64)
    q = unwrap_rows(near(q, q0))
    q = q + (near(q[n // 2], q0) - q[n // 2])
    ref = np.array(K["knobs"]["dirs"][lead]["q"])
    check = float(np.abs(wrap_diff(q[:: (n - 1) // 20], ref)).max())
    return q, {"target": target, "lead": lead, "max_diff_vs_page_rad": round(check, 5)}


def wrap_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--web_data", default="outputs/web_data")
    ap.add_argument("--out", default="outputs/robot_trajectories")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vmax", type=float, default=0.3, help="peak joint speed, rad/s")
    ap.add_argument("--ramp", type=float, default=1.5, help="length of the start and stop ramps, s")
    ap.add_argument("--rate", type=float, default=100.0, help="sample rate of the CSVs, Hz")
    ap.add_argument("--frames", type=int, default=241, help="decoded frames per path before the spline")
    ap.add_argument("--robots", nargs="+", default=["franka"], choices=list(RECORD),
                    help="robots to export (default: franka)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {"vmax_rad_s": args.vmax, "ramp_s": args.ramp, "rate_hz": args.rate, "trajectories": {},
               "timing": {}}
    for robot, cells in ((r, RECORD[r]) for r in args.robots):
        ik = load_checkpoint(CKPT.format(robot=robot), args.device)
        robot_obj = get_robot(robot)
        for c in cells:
            if (robot, c) == ("ur5", "se3"):
                K = json.load(open(os.path.join(args.web_data, robot, c, "knobs.json")))
                br = shift_home(np.array(K["branches"]), robot) if K["branches"] else np.zeros((0, 6))
                br = np.array([b - 2 * math.pi * np.round(b / (2 * math.pi)) for b in br])
                tg = K["target"]
                target = Target.from_pose(c, np.array(tg["pos"]), np.array(tg["R"]))
                pe, oe = errors(robot_obj, br, target)
                rows = []
                path = os.path.join(args.out, "ur5_se3_branches.csv")
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["branch"] + [f"q{i + 1}" for i in range(6)] + ["clearance_m", "pos_err_mm", "orn_err_deg"])
                    for i, b in enumerate(br):
                        cl = clearance(robot, b[None])
                        w.writerow([i] + [f"{v:.6f}" for v in b] + [f"{cl:.3f}", f"{pe[i]:.4f}", f"{oe[i]:.4f}"])
                        rows.append({"branch": i, "clearance_m": round(cl, 3), "safe": cl >= 0.08})
                summary["trajectories"]["ur5/se3:branches"] = {"file": os.path.basename(path), "branches": rows,
                                                               "camera": camera(robot, br)}
                print(f"ur5/se3 branches: {len(br)} ({sum(r['safe'] for r in rows)} above the table)")
                continue
            for kind in ("line", "knob"):
                if kind == "line":
                    q, info = line_path(ik, robot, c, args.web_data, args.frames)
                    s_grid = np.linspace(0, 1, len(q))
                else:
                    q, info = knob_path(ik, robot, c, args.web_data, args.frames)
                    s_grid = np.linspace(-1, 1, len(q))
                    a = safe_knob_range(robot, shift_home(q, robot), s_grid)
                    info["knob_range"] = [-a, a]
                    keep = np.abs(s_grid) <= a + 1e-9
                    if a > 0:
                        q, s_grid = q[keep], s_grid[keep]
                q = shift_home(q, robot)
                t, s, qq, qd, T, tau = parametrize(q, s_grid, args.vmax, args.ramp, args.rate)
                pe, oe = errors(robot_obj, qq, info["target"])
                e = pe if pos_axes(c) else oe
                name = f"{robot}_{c}_{kind}.csv"
                write_csv(os.path.join(args.out, name), t, s, qq, qd)
                chk = checks(robot, qq, qd)
                entry = {"file": name, "duration_s": round(T, 2), "start_q": np.round(qq[0], 5).tolist(),
                         "end_q": np.round(qq[-1], 5).tolist(),
                         "max_constraint_error": round(float(np.nanmax(e)), 3),
                         "error_unit": "mm" if pos_axes(c) else "deg",
                         **chk, "camera": camera(robot, qq),
                         **{k: v for k, v in info.items() if k != "target"}}
                entry["safe"] = chk["hw_limit_margin_rad"] >= LIMIT_MARGIN and chk["min_clearance_m"] >= CLEARANCE
                summary["trajectories"][f"{robot}/{c}:{kind}"] = entry
                # The page scrubs by alpha in [0, 1]; for a clipped knob sweep that range is (1 +- a) / 2.
                lo = 0.0 if kind == "line" else (1 - info["knob_range"][1]) / 2
                summary["timing"][f"{robot}/{c}:{kind}"] = {
                    "alpha": np.round(np.linspace(lo, 1 - lo, 21), 4).tolist(), "tau": np.round(tau, 4).tolist()}
                print(f"{robot}/{c}:{kind:4s} {T:5.1f} s  peak {chk['peak_joint_speed_rad_s']:.2f} rad/s  "
                      f"clearance {chk['min_clearance_m']:.2f} m  limit margin {chk['hw_limit_margin_rad']:.2f} rad  "
                      f"{'range ' + str(info['knob_range']) + '  ' if kind == 'knob' else ''}"
                      f"err {entry['max_constraint_error']:.2f} {entry['error_unit']}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(README.format(vmax=args.vmax, ramp=args.ramp, rate=args.rate))
    print("wrote", args.out)


README = """# Real-robot trajectories for the project page

Written by `scripts/export_robot_trajectories.py` from the page data of `scripts/export_web_data.py`, so each
motion is exactly the one the page draws, re-decoded at a finer resolution from the released checkpoint.

| File | Motion |
|---|---|
| `{{robot}}_{{c}}_line.csv` | the straight latent path x_a -> x_b at the page's default noise level (Widget "Walk the latent space") |
| `{{robot}}_{{c}}_knob.csv` | the leading tangent knob swept from -1 to +1 (a Score-SVD direction at t = 15, applied to the latent at t_dec = 50) (Widget "Count the knobs") |
| `summary.json` | per trajectory: duration, peak joint speed, joint-limit and table-clearance checks, start pose, suggested camera pose, and `timing` for the page |

CSV columns: `t` [s], `s` (alpha in [0, 1] for lines, knob value in [-1, 1] for knobs), `q1..qn` [rad], `qd1..qdn` [rad/s],
sampled at {rate:g} Hz. Knob sweeps are clipped to the largest symmetric range [-a, a] that stays 0.05 rad inside the
hardware limits and 8 cm above the table (`knob_range` in `summary.json`); the Franka limits are the intersection of the
Panda and FR3 datasheet limits. Joint order and zero are the robot's own (UR5: shoulder_pan .. wrist_3; Franka: joint 1 .. 7), the
same as the model's q.

Timing: constant rate in s, scaled so the fastest joint peaks at {vmax:g} rad/s, with {ramp:g} s smoothstep velocity ramps at
both ends (acceleration is continuous, so the Franka's motion generator accepts it). Resample to the controller rate with a
cubic spline if needed.

Before running on hardware: move to `start_q` slowly (teach pendant, MoveIt, or a separate joint move), confirm the
`safe` flag and the clearance in `summary.json`, dry-run in simulation (URSim / franka_ros with Gazebo) and at reduced speed.
"""


if __name__ == "__main__":
    main()
