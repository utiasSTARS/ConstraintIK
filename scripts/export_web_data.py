"""Export the data behind the project page (papers.starslab.ca/diffusion-ik-geometry/).

For every robot and constraint this writes, from the released checkpoints,

  {robot}/{c}/interp.json          target, the three IK solutions x_a, x_b, x_c, levels, layout
  {robot}/{c}/interp_t{T}.bin      the latent plane through x_a, x_b, x_c at noise level T, decoded on a
                                   G x G grid, plus the straight segment a -> b (L frames)
  {robot}/{c}/knobs.json           score spectra at t = 0..95, d-hat(t), and 1-D sweeps along every
                                   singular direction of the score matrix at t = 15, applied to the latent
                                   inverted to t_dec = 50 and decoded from there
  manifest.json, results.json      index of all files, and the numbers of the results section

The endpoint pair of each condition is picked from results/interpolation/targets (smooth, clearly visible
motion that a real arm on a table can execute; see pick_pair) and recorded in interp.json, so the page and
scripts/export_robot_trajectories.py show the same motion. The page's default noise level is the
linear-interpolation t_bar of configs/tbar.json.

    python scripts/export_web_data.py --device cuda:1 --out outputs/web_data
    python scripts/export_web_data.py --only franka/se3 --pick franka/se3=12   # force a pair index

Binary layout of interp_t{T}.bin (little-endian float32): G*G*n joint angles (row-major over v, then u),
G*G errors, [G*G branch ids for the UR5 under SE(3)], L*n joint angles of the segment, L segment errors.
Pad coordinates (u, v) mean x = x_a + u (x_b - x_a) + v (x_c - x_a) at level T.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

from constraint_ik import Target, errors, get_robot, load_checkpoint
from constraint_ik.constraints import PAPER_ORDER, codim, n_orient_bits, pos_axes
from constraint_ik.id_estimation import gap_rank, score_matrix
from constraint_ik.interpolation import invert_levels, lerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import interp_full_stats  # noqa: E402

CKPT = "checkpoints/{robot}.pt"
TARGETS = "results/interpolation/targets/{robot}_{c}.npz"
CANONICAL = "results/interpolation/canonical/{robot}_{c}.csv"
TBAR = "configs/tbar.json"

G = 31                       # pad nodes per side; u = 0 and u = 1 fall on nodes
RANGE = (-0.25, 1.25)
L = 49                       # frames of the straight segment
LEVELS = (0, 20, 40, 60, 80, 95)
T_KNOB, K_KNOB = 15, 50      # Score-SVD setting of the paper (directions and spectrum)
T_DEC = 50                   # knobs move the DDIM-inverted latent at this level and decode from it
                             # (same scheme as scripts/viser_explorer.py and render_media.py)
T_GRID = tuple(range(0, 100, 5))
AMPS = np.linspace(-1.0, 1.0, 21)
KNOB_STEP = 1.2              # latent step (normalized sincos units) at knob value 1. Up to about 1.2 every
                             # normal direction moves the arm by at most 0.2 rad; from about 1.4 on, some
                             # normal directions snap to another IK solution (checked on all 14 conditions)
PAD2D = 13
SEED = 0
CLEARANCE = 0.08             # m, minimum height of the joint centres beyond the shoulder above the table
LIMIT_MARGIN = 0.05          # rad, required distance to the hardware joint limits
# Hardware joint limits (rad). The model's Franka limits (robots.py) are the wider data-generation box;
# a real arm must stay inside both the Panda and the FR3 datasheet limits, whichever is recorded.
FRANKA_HW = {
    "panda": ([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
              [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    "fr3": ([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
            [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]),
}
UR5_HW = 2 * math.pi         # every UR5 joint accepts [-2 pi, 2 pi]

ROBOTS = {
    "franka": {"label": "Franka", "long": "Franka Emika Panda", "color": "#2A4F7D", "light": "#527299"},
    "ur5": {"label": "UR5", "long": "Universal Robots UR5", "color": "#6E1622", "light": "#CA7981"},
}
CONSTRAINT_TEXT = {
    "plane": ("Plane", "one position coordinate", "plane"),
    "line": ("Line", "two position coordinates", "line"),
    "posonly": ("Position", "three position coordinates", "point"),
    "orient": ("Orientation", "three rotation coordinates", "frame-free"),
    "plane_orient": ("Plane + Orientation", "one position coordinate + full rotation", "plane-frame"),
    "line_orient": ("Line + Orientation", "two position coordinates + two rotation coordinates", "line-axis"),
    "se3": ("SE(3)", "full pose", "frame"),
}
TEASER = [("franka", "se3", "Franka SE(3), self-motion"), ("ur5", "line_orient", "UR5 Line + Orientation")]
REAL_ROBOTS = ("franka",)    # robots with real-robot recordings


# ---------------------------------------------------------------- geometry helpers

def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def unwrap_rows(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=np.float64)
    for i in range(1, len(q)):
        q[i] = q[i - 1] + wrap(q[i] - q[i - 1])
    return q


def near(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return ref + wrap(q - ref)


TOOL_LENGTH = {"ur5": 0.17, "franka": 0.16}   # m past the flange: Robotiq 2F-85 + coupler, Franka hand + fingers


def joint_centres(robot: str, q: np.ndarray) -> np.ndarray:
    """(..., m + 1, 3) positions of the link frames (joints, plus the flange on the Franka) and of the
    gripper tip (TOOL_LENGTH along the last frame's z axis), in the IKFast base frame."""
    F = get_robot(robot).link_frames(torch.as_tensor(np.asarray(q, dtype=np.float64)))
    pts = F[..., :3, 3]
    tip = pts[..., -1, :] + TOOL_LENGTH[robot] * F[..., -1, :3, 2]
    return torch.cat([pts, tip[..., None, :]], dim=-2).numpy()


def clearance(robot: str, q: np.ndarray) -> float:
    """Lowest joint centre beyond the shoulder, or gripper tip, over all frames of a path (m above the
    base plane)."""
    return float(joint_centres(robot, q)[..., 2:, 2].min())


def hw_margin(robot: str, q: np.ndarray) -> float:
    """Smallest distance (rad) to the hardware limits over a path; negative when outside. The Franka
    uses the intersection of the Panda and FR3 limits; UR5 paths are first shifted by 2 pi multiples."""
    q = np.atleast_2d(q)
    if robot == "ur5":
        q = q - 2 * math.pi * np.round(q[:1] / (2 * math.pi))
        return float(UR5_HW - np.abs(q).max())
    lo = np.maximum(*[np.array(v[0]) for v in FRANKA_HW.values()])
    hi = np.minimum(*[np.array(v[1]) for v in FRANKA_HW.values()])
    return float(np.minimum(q - lo, hi - q).min())


def primary_error(robot_obj, q: np.ndarray, target: Target, c: str) -> tuple[np.ndarray, str]:
    pos, orn = errors(robot_obj, q, target)
    return (pos, "mm") if pos_axes(c) else (orn, "deg")


# ---------------------------------------------------------------- choosing the pair

def read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["method"] == "linear"]


@torch.no_grad()
def decode_segments(ik, data, idx, targets, t_bar: int, n: int = L) -> np.ndarray:
    out = []
    for k in idx:
        cond = ik.cond(targets[data["pair_target"][k]])
        z_a, z_b = (ik.invert(data[key][k], t_bar, cond) for key in ("q_a", "q_b"))
        out.append(ik.decode_from(lerp(z_a, z_b, n), t_bar, cond).cpu().numpy())
    return np.stack(out)


def pick_pair(ik, robot_obj, robot: str, c: str, data, targets, t_bar: int, log: dict) -> int:
    """A pair whose decoded path at t_bar is accurate, smooth, clearly visible, inside the joint limits
    and above the table. UR5 SE(3) paths switch IK branches by design, so jumps are allowed there."""
    rows = read_rows(CANONICAL.format(robot=robot, c=c))
    key = {(int(r["target"]), int(r["pair_width"])): r for r in rows}
    err_col = "mean_pos_err_mm" if pos_axes(c) else "mean_orn_err_deg"
    med_err = np.nanmedian([float(r[err_col]) for r in rows])
    cands = [k for k in range(len(data["q_a"])) if int(data["pair_width"][k]) >= 50]
    segs = decode_segments(ik, data, cands, targets, t_bar)
    best, best_score = None, -np.inf
    table = []
    for k, seg in zip(cands, segs):
        r = key[(int(data["pair_target"][k]), int(data["pair_width"][k]))]
        seg_u = unwrap_rows(seg)
        step = float(np.abs(np.diff(seg_u, axis=0)).max())
        ok_limits = hw_margin(robot, seg_u) >= LIMIT_MARGIN
        clear = clearance(robot, seg)
        err = float(r[err_col])
        strict = float(r["valid_pct_strict"])
        jumps_ok = (robot, c) == ("ur5", "se3") or step < 0.2
        score = float(data["q_dist"][k]) / (1.0 + err / max(med_err, 1e-6))
        feasible = ok_limits and clear >= CLEARANCE and jumps_ok and strict >= 99.0
        score += 0.0 if feasible else -100.0
        table.append((k, round(score, 3), int(data["pair_width"][k]), round(err, 2), round(step, 3), round(clear, 3), ok_limits, strict))
        if score > best_score:
            best, best_score = k, score
    log["pick_table_top"] = sorted(table, key=lambda x: -x[1])[:5]
    return int(best)


# ---------------------------------------------------------------- latent plane

@torch.no_grad()
def export_interp(ik, robot_obj, robot: str, c: str, data, targets, k: int, t_bar: int, out_dir: str) -> dict:
    target = targets[data["pair_target"][k]]
    cond = ik.cond(target)
    levels = sorted(set(LEVELS) | {t_bar})
    q_a, q_b, q_c = (data[key][k] for key in ("q_a", "q_b", "q_c"))
    zs = {key: invert_levels(ik, data[key][k], levels, cond) for key in ("q_a", "q_b", "q_c")}
    us = torch.linspace(RANGE[0], RANGE[1], G, device=ik.device)
    V, U = torch.meshgrid(us, us, indexing="ij")
    U, V = U.reshape(-1, 1), V.reshape(-1, 1)
    branch_mode = (robot, c) == ("ur5", "se3")
    branches = robot_obj.ik_all(target.pos, target.R) if branch_mode else None
    files, summary = {}, {}
    unit = "mm"
    n = robot_obj.n_joints
    for t in levels:
        z_a, z_b, z_c = zs["q_a"][t], zs["q_b"][t], zs["q_c"][t]
        X = z_a + U * (z_b - z_a) + V * (z_c - z_a)
        q = ik.decode_from(X, t, cond).cpu().numpy().astype(np.float64)
        q = near(q, q_a)
        err, unit = primary_error(robot_obj, q, target, c)
        seg = ik.decode_from(lerp(z_a, z_b, L), t, cond).cpu().numpy().astype(np.float64)
        seg = unwrap_rows(seg)
        seg = seg + (near(seg[0], q_a) - seg[0])
        seg_err, _ = primary_error(robot_obj, seg, target, c)
        blobs = [q.ravel(), np.nan_to_num(err, nan=0.0)]
        if branch_mode:
            d = np.linalg.norm(wrap(q[:, None, :] - branches[None]), axis=2)
            blobs.append(np.argmin(d, axis=1).astype(np.float64))
        blobs += [seg.ravel(), np.nan_to_num(seg_err, nan=0.0)]
        name = f"interp_t{t}.bin"
        with open(os.path.join(out_dir, name), "wb") as f:
            for b in blobs:
                f.write(np.asarray(b, dtype="<f4").tobytes())
        files[str(t)] = name
        summary[str(t)] = {"pad_median": round(float(np.nanmedian(err)), 3),
                           "segment_mean": round(float(np.nanmean(seg_err)), 3)}
    color_mode = "branch" if branch_mode else ("selfmotion" if c == "se3" else "error")
    meta = {
        "mock": False,
        "target": {"pos": target.pos.tolist(), "R": target.R.tolist(), "mask": [int(v) for v in target.mask]},
        "anchors": {"a": q_a.tolist(), "b": near(q_b, q_a).tolist(), "c": near(q_c, q_a).tolist()},
        "t_levels": levels, "t_default": t_bar, "grid": G, "range": list(RANGE), "L": L, "n": n,
        "error_unit": unit, "color_mode": color_mode, "files": files,
        "branches": None if branches is None else near(branches, q_a).tolist(),
        "layout": ("little-endian float32: G*G*n joint angles (row-major over v, then u), G*G errors, "
                   "[G*G branch ids if color_mode = branch], L*n segment joint angles, L segment errors"),
        "source": {"targets": TARGETS.format(robot=robot, c=c), "pair_index": k,
                   "target_index": int(data["pair_target"][k]), "pair_width": int(data["pair_width"][k]),
                   "q_dist": round(float(data["q_dist"][k]), 4), "t_bar_file": TBAR,
                   "checkpoint": CKPT.format(robot=robot), "summary": summary},
    }
    with open(os.path.join(out_dir, "interp.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    return meta


# ---------------------------------------------------------------- knobs

def knob_noise(ik) -> torch.Tensor:
    """The K perturbations of Score-SVD, drawn on the CPU from SEED (as scripts/viser_explorer.py)."""
    return torch.randn(K_KNOB, ik.dim, generator=torch.Generator().manual_seed(SEED)).to(ik.device)


@torch.no_grad()
def spectrum(ik, x0: torch.Tensor, cond, t: int, noise: torch.Tensor):
    S = score_matrix(ik, x0, cond, t, K_KNOB, noise)
    _, s, Vh = torch.linalg.svd(S, full_matrices=False)
    return s.cpu().numpy().astype(np.float64), Vh


def knob_anchor(ik, robot: str, c: str, data, targets, k: int, noise):
    """A configuration of the same target whose t = 15 spectrum has a clean gap at the analytical
    dimension (the per-target median is exact, single anchors can have one dominant sigma_1), preferring
    configurations at least 0.3 rad inside the hardware joint limits."""
    n = ik.n_joints
    d_true = n - codim(c)
    ti = int(data["pair_target"][k])
    target = targets[ti]
    cond = ik.cond(target)
    pairs = [j for j in range(len(data["q_a"])) if int(data["pair_target"][j]) == ti]
    cands = [data["q_a"][k], data["q_b"][k]] + [data[key][j] for j in pairs for key in ("q_a", "q_b", "q_c")]
    best, best_q = None, None
    for q in cands:
        s, _ = spectrum(ik, ik.encode(q), cond, T_KNOB, noise)
        ok = ik.dim - gap_rank(s, dagger=True) == d_true
        m = ik.dim
        gap = (s[m - d_true - 1] - (s[m - d_true] if d_true else 0.0)) / s[0]
        dom = s[0] / s[1]
        score = (gap if ok else -1.0) - 0.2 * max(0.0, dom - 1.5)
        score -= 2.0 * max(0.0, 0.3 - hw_margin(robot, q))   # leave room for the knob sweep on a real arm
        if best is None or score > best:
            best, best_q = score, q
    return best_q, target, cond


@torch.no_grad()
def export_knobs(ik, robot_obj, robot: str, c: str, data, targets, k: int, vs_t: dict, out_dir: str) -> dict:
    n, m = ik.n_joints, ik.dim
    d_true = n - codim(c)
    noise = knob_noise(ik)
    q0, target, cond = knob_anchor(ik, robot, c, data, targets, k, noise)
    x0 = ik.encode(q0)
    spectra, dh_svd, dh_dag = [], [], []
    for t in T_GRID:
        s, _ = spectrum(ik, x0, cond, t, noise)
        spectra.append([round(float(v / s[0]), 5) for v in s])
        dh_svd.append(m - gap_rank(s))
        dh_dag.append(m - gap_rank(s, dagger=True))
    s15, Vh = spectrum(ik, x0, cond, T_KNOB, noise)
    d_anchor = m - gap_rank(s15, dagger=True)
    z_dec = ik.invert(q0, T_DEC, cond)
    amps = torch.as_tensor(AMPS, dtype=torch.float32, device=ik.device)[:, None]
    X = torch.cat([z_dec + KNOB_STEP * amps * Vh[i][None] for i in range(m)])
    Q = ik.decode_from(X, T_DEC, cond).cpu().numpy().astype(np.float64).reshape(m, len(AMPS), n)
    q0n = ik.decode_from(z_dec, T_DEC, cond).cpu().numpy()[0].astype(np.float64)
    dirs, motion = [], []
    for i in range(m):
        qs = unwrap_rows(near(Q[i], q0))
        qs = qs + (near(qs[len(AMPS) // 2], q0) - qs[len(AMPS) // 2])
        pe, oe = errors(robot_obj, qs, target)
        e = pe if pos_axes(c) else oe
        kind = "tangent" if i >= m - d_true else "normal"
        dq = float(np.linalg.norm(qs[-1] - qs[0]))
        motion.append(dq)
        dirs.append({"kind": kind, "sigma": round(float(s15[i] / s15[0]), 5), "q": np.round(qs, 5).tolist(),
                     "err_end": round(float(np.nanmax(e[[0, -1]])), 3)})
    tangent = [i for i in range(m) if dirs[i]["kind"] == "tangent"]
    order = sorted(tangent, key=lambda i: -motion[i] / (1.0 + dirs[i]["err_end"]))
    pad2d = None
    if len(tangent) >= 2:
        i, j = order[0], order[1]
        a = torch.as_tensor(np.linspace(-1, 1, PAD2D), dtype=torch.float32, device=ik.device)
        B, A = torch.meshgrid(a, a, indexing="ij")
        Xp = z_dec + KNOB_STEP * (A.reshape(-1, 1) * Vh[i][None] + B.reshape(-1, 1) * Vh[j][None])
        qp = ik.decode_from(Xp, T_DEC, cond).cpu().numpy().astype(np.float64)
        qp = near(qp, q0)
        pad2d = {"amps": np.round(np.linspace(-1, 1, PAD2D), 4).tolist(), "dirs": [i, j],
                 "q": np.round(qp, 5).tolist(), "order": "row-major over the second knob, then the first"}
    branches = None
    if d_true == 0:
        br = robot_obj.ik_all(target.pos, target.R)
        branches = np.round(near(br, q0), 6).tolist()
    by_c = vs_t["by_constraint"][c]["uniform"]
    doc = {
        "mock": False,
        "t_levels": list(T_GRID), "spectra": spectra, "spectra_note": "normalized by sigma_1 at each t",
        "dhat_svd": by_c["svd"]["median"], "dhat_dagger": by_c["dagger"]["median"],
        "dhat_dagger_band": [by_c["dagger"]["lo"], by_c["dagger"]["hi"]],
        "dhat_anchor_svd": dh_svd, "dhat_anchor_dagger": dh_dag,
        "id_true": d_true, "ambient": m, "t_canonical": T_KNOB, "K": K_KNOB, "seed": SEED, "t_dec": T_DEC,
        "q0": np.round(q0, 6).tolist(), "q0_decoded": np.round(near(q0n, q0), 6).tolist(),
        "target": {"pos": target.pos.tolist(), "R": target.R.tolist(), "mask": [int(v) for v in target.mask]},
        "knobs": {"amps": np.round(AMPS, 4).tolist(), "step": KNOB_STEP, "lead": order[0] if order else None,
                  "dirs": dirs},
        "pad2d": pad2d, "branches": branches,
        "source": {"anchor_dhat_t15": int(d_anchor), "checkpoint": CKPT.format(robot=robot),
                   "dhat_curves": f"results/id/{robot}_vs_t.json (median over targets, uniform anchors)"},
    }
    with open(os.path.join(out_dir, "knobs.json"), "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    return doc


# ---------------------------------------------------------------- results section

def export_results(out: str) -> None:
    est = [("mle10", "mle_k10", "Local MLE", "NN = 10", True), ("pca90", "pca90", "Local PCA", "90% variance", True),
           ("svd", "score_svd", "Score-SVD", "Stanczuk et al.", True), ("dagger", "score_svd_dagger", "Score-SVD†", "ours", True),
           ("mle2", "mle_k2", "Local MLE", "NN = 2", False), ("pca99", "pca99", "Local PCA", "99% variance", False),
           ("twonn", "twonn", "TwoNN", "", False)]
    doc = {"note": ("Generated by scripts/export_web_data.py. id: results/id/{robot}.json (medians over targets, "
                    "95% bootstrap CIs for the Fig. 2 estimators). interp: tab:interp-full from "
                    "results/interpolation/canonical (median over 150 paths of the per-path mean error, 95% CI)."),
           "order": list(PAPER_ORDER),
           "id": {"estimators": [{"key": k, "label": lab, "detail": det, "fig": fig} for k, _, lab, det, fig in est]},
           "interp": {"status": None, "source": "results/interpolation/canonical, straight-line (lin) and spherical (sph)"}}
    for robot in ROBOTS:
        with open(f"results/id/{robot}.json") as f:
            idj = json.load(f)
        bars = idj["fig_bars"]
        rows = {}
        for c in PAPER_ORDER:
            s = idj["by_constraint"][c]["summary"]
            row = {"id": int(idj["by_constraint"][c]["id"])}
            for k, src, *_ in est:
                row[k] = round(float(s[src]["median"]), 3)
            ci = {}
            for k, src, _, _, fig in est:
                if fig and src in bars:
                    j = bars["order"].index(c)
                    ci[k] = [round(float(bars[src]["lo"][j]), 3), round(float(bars[src]["hi"][j]), 3)]
            row["ci"] = ci
            rows[c] = row
        doc["id"][robot] = rows
    stats = interp_full_stats()
    for robot in ROBOTS:
        doc["interp"][robot] = {}
        for c in PAPER_ORDER:
            cell = stats[robot][c]
            doc["interp"][robot][c] = {key: (None if cell[key] is None else [round(float(v), 4) for v in cell[key]])
                                       for key in ("pos_lin", "orn_lin", "pos_sph", "orn_sph")}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(doc, f, indent=1)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/web_data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=None, help="robot/constraint cells to (re)export")
    ap.add_argument("--pick", nargs="*", default=[], help="robot/constraint=pair_index overrides")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    picks = {p.split("=")[0]: int(p.split("=")[1]) for p in args.pick}
    with open(TBAR) as f:
        tbar = json.load(f)
    manifest_path = os.path.join(args.out, "manifest.json")
    conditions = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            conditions = json.load(f).get("conditions", {})
    log = {}
    for robot in ROBOTS:
        ik = load_checkpoint(CKPT.format(robot=robot), args.device)
        robot_obj = get_robot(robot)
        with open(f"results/id/{robot}_vs_t.json") as f:
            vs_t = json.load(f)
        for c in PAPER_ORDER:
            cell = f"{robot}/{c}"
            if args.only and cell not in args.only:
                continue
            t0 = time.time()
            data = dict(np.load(TARGETS.format(robot=robot, c=c)))
            targets = [Target.from_pose(c, p, R) for p, R in zip(data["pos"], data["R"])]
            t_bar = int(tbar[robot][c]["linear"])
            log[cell] = {}
            k = picks[cell] if cell in picks else pick_pair(ik, robot_obj, robot, c, data, targets, t_bar, log[cell])
            out_dir = os.path.join(args.out, robot, c)
            os.makedirs(out_dir, exist_ok=True)
            meta = export_interp(ik, robot_obj, robot, c, data, targets, k, t_bar, out_dir)
            knobs = export_knobs(ik, robot_obj, robot, c, data, targets, k, vs_t, out_dir)
            n = robot_obj.n_joints
            conditions[cell] = {
                "n": n, "k": codim(c), "id": n - codim(c), "t_levels": meta["t_levels"], "t_default": t_bar,
                "interp": f"{cell}/interp.json", "knobs": f"{cell}/knobs.json",
                "files": {t: f"{cell}/{f}" for t, f in meta["files"].items()},
                "color_mode": meta["color_mode"], "error_unit": meta["error_unit"],
                "pick": meta["source"],
            }
            log[cell].update(pair=k, t_bar=t_bar, anchor_dhat=knobs["source"]["anchor_dhat_t15"],
                             summary=meta["source"]["summary"], seconds=round(time.time() - t0, 1))
            print(f"{cell:20s} pair {k:3d}  t_bar {t_bar}  anchor d-hat {knobs['source']['anchor_dhat_t15']}  "
                  f"{log[cell]['seconds']} s", flush=True)
    export_results(args.out)
    media = "media/real/"
    keys = [f"{r}/{c}" for r in ROBOTS for c in PAPER_ORDER]
    real_keys = [k for k in keys if k.split("/")[0] in REAL_ROBOTS]
    manifest = {
        "mock": False,
        "version": 2,
        "note": "Generated by scripts/export_web_data.py from checkpoints/{robot}.pt.",
        "robots": {r: dict(v, n=get_robot(r).n_joints, chain=f"robots/{r}/chain.json") for r, v in ROBOTS.items()},
        "constraints": [{"key": c, "label": CONSTRAINT_TEXT[c][0], "fixed": CONSTRAINT_TEXT[c][1], "k": codim(c),
                         "mask": list(Target.from_pose(c, np.zeros(3), np.eye(3)).mask.astype(int).tolist()),
                         "gizmo": CONSTRAINT_TEXT[c][2]} for c in PAPER_ORDER],
        "conditions": {k: conditions[k] for k in keys if k in conditions},
        "teaser": [{"robot": r, "constraint": c, "caption": cap} for r, c, cap in TEASER],
        "real_media": {
            "robots": list(REAL_ROBOTS),
            "available": False,
            "line": {k: media + "%s_%s_line.mp4" % tuple(k.split("/")) for k in real_keys},
            "knob": {k: media + "%s_%s_knob.mp4" % tuple(k.split("/")) for k in real_keys},
            "timing": {},
            "gallery": [
                {"robot": "franka", "title": "Franka, latent interpolation", "src": media + "gallery_franka.mp4",
                 "poster": media + "gallery_franka.jpg", "available": False},
            ],
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out, "export_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(args.out) for fn in fns)
    print(f"wrote {args.out} ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
