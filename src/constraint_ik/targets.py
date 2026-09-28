"""Interpolation targets: target synthesis from analytical IK and the endpoint-pair pickers.

A target is seeded by the FK pose of a random configuration q_rand. Around it we build an IK pool that
spans the free task coordinates, pick a seed solution, and sweep one free coordinate with IK to get a
1-D chain of solutions on one branch. Five endpoint pairs per target are then picked at pair widths
10..100 by the cell's picker:

  chain      chain-offset ranks 50 -/+ w/2 (percent of the chain)
  j7         the same ranks along the longest branch-flip-free self-motion segment, sorted by joint 7
  diameter   the pair whose joint distance is closest to w% of the diameter of chain + pool
  branches   the pair at rank w% of the sorted pairwise (wrapped) distances between IK branches
"""
from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from constraint_ik.robots import Robot, get_robot

PAIR_WIDTHS = (10, 25, 50, 75, 100)
UR5_HOME = np.array([0.0, 3 * math.pi / 2, 0.0, 3 * math.pi / 2, 0.0, 0.0])
ROT_MARGIN = 0.15      # joint-limit margin of the configurations whose FK rotation seeds a random orientation


@dataclass(frozen=True)
class Recipe:
    """Synthesis settings of one (robot, constraint) cell."""
    pool: str                                   # plane | line | plane_orient | line_orient | ball | orientations | se3
    margin: float = 0.15                        # joint-limit margin of q_rand
    seed_shell: tuple | None = (0.25, 0.65)     # r_min < ||p|| < r_max for the seed position
    in_workspace: bool = False                  # also require the seed inside the robot's cylindrical shell
    dedup_tol: float = 0.05
    pool_cap: int | None = 500                  # shuffled dedup capped at this size; None: in-order, no cap
    n_grid: int = 30                            # x (and y) grid points of the position pools
    n_rot: int = 0                              # random rotations per cell (line, plane), Rz grid size (line_orient)
    x_clip: float = 0.5
    grid_n: int = 12                            # ball pool: grid_n^3 positions within +-ball_r
    ball_r: float = 0.25
    n_orient: int = 50                          # orientations pool
    sweep: tuple | None = ("pos", 0, 0.01, 1.5)  # (pos|rot, world axis, step, max distance per direction)
    jump: float = 0.3                           # branch-flip threshold (wrapped joint distance, rad)
    min_size: int = 10                          # minimum pool and chain size (branches for UR5 SE(3))
    picker: str = "chain"
    n_tries: int = 1000


_RECIPES = {
    "plane":        Recipe("plane", n_rot=3),
    "line":         Recipe("line", n_rot=30),
    "plane_orient": Recipe("plane_orient"),
    "line_orient":  Recipe("line_orient", n_rot=30),
    "orient":       Recipe("ball", seed_shell=None, in_workspace=True, pool_cap=None),
    "posonly":      Recipe("orientations", margin=0.05, seed_shell=None, in_workspace=True, dedup_tol=0.1,
                           pool_cap=None, sweep=("rot", 2, 0.02, 6.5)),
    "se3":          Recipe("se3", margin=0.05, seed_shell=None, in_workspace=True, dedup_tol=0.1,
                           pool_cap=None, sweep=None, picker="j7"),
}
_UR5_SHELL = (0.25, 0.62)
_UR5 = {
    **{c: {"seed_shell": _UR5_SHELL} for c in ("plane", "line", "plane_orient", "line_orient", "orient")},
    "posonly": {"seed_shell": _UR5_SHELL, "picker": "diameter"},
    "se3": {"min_size": 4, "picker": "branches"},
}


def recipe(robot: str, constraint: str) -> Recipe:
    r = _RECIPES[constraint]
    return replace(r, **_UR5.get(constraint, {})) if robot == "ur5" else r


# ---------------------------------------------------------------- helpers

def wrap_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Joint-space L2 distance with angle differences wrapped to (-pi, pi], over the last axis."""
    d = np.asarray(a) - np.asarray(b)
    w = np.arctan2(np.sin(d), np.cos(d))
    return np.linalg.norm(w, axis=-1) if w.ndim > 1 else np.linalg.norm(w)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def dedup(sols, tol: float, n_joints: int, cap: int | None = None,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Greedy dedup: keep a solution farther than tol (L2) from every kept one. With rng the candidates
    are shuffled first; with cap the scan stops once cap solutions are kept."""
    if len(sols) == 0:
        return np.zeros((0, n_joints))
    S = np.asarray(sols, dtype=np.float64)
    if rng is not None:
        S = S[rng.permutation(len(S))]
    keep = np.empty_like(S)
    keep[0], k, tol2 = S[0], 1, tol * tol
    for i in range(1, len(S)):
        d = keep[:k] - S[i]
        if np.einsum("ij,ij->i", d, d).min() > tol2:
            keep[k] = S[i]
            k += 1
        if cap is not None and k >= cap:
            break
    return keep[:k].copy()


def sweep_free_axis(robot: Robot, q_seed: np.ndarray, mode: str, axis: int, step: float,
                    max_dist: float, jump: float, direction: float) -> list[tuple[float, np.ndarray]]:
    """IK walk from q_seed along a free world axis (translate for 'pos', rotate for 'rot'), holding the
    other task coordinates of q_seed; each step takes the IK solution nearest the previous one and the
    walk stops on IK failure or a jump larger than `jump`. Returns [(offset, q), ...] from (0, q_seed)."""
    pos0, R0 = robot.fk(q_seed)
    last = q_seed.copy()
    chain = [(0.0, last.copy())]
    e = np.eye(3)[axis]
    for k in range(1, int(max_dist / step) + 1):
        off = direction * step * k
        if mode == "pos":
            p, R = pos0.copy(), R0
            p[axis] = pos0[axis] + off
        else:
            p, R = pos0, Rotation.from_rotvec(off * e).as_matrix() @ R0
        sols = robot.ik_all(p, R)
        if len(sols) == 0:
            break
        d = [float(wrap_dist(s, last)) for s in sols]
        i = int(np.argmin(d))
        if d[i] > jump:
            break
        last = sols[i]
        chain.append((off, last.copy()))
    return chain


# ---------------------------------------------------------------- IK pools

def _grid_cells(robot: Robot, rec: Recipe, pos: np.ndarray) -> list[np.ndarray] | None:
    """Pool positions: an x grid on the line through pos (line families) or an xy grid at pos_z (plane
    families), keeping cells inside the workspace annulus. None if the line's x range is too short."""
    ws = robot.shell
    if rec.pool in ("line", "line_orient"):
        x0, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        half = float(np.sqrt(max(0.0, rec.seed_shell[1] ** 2 - y ** 2 - max(0.0, z) ** 2)))
        lo, hi = max(-rec.x_clip, x0 - half), min(rec.x_clip, x0 + half)
        if hi - lo < 0.05:
            return None
        return [np.array([float(x), y, z]) for x in np.linspace(lo, hi, rec.n_grid)
                if ws.r_min <= float(np.hypot(x, y)) <= ws.r_max]
    half = min(ws.r_max, rec.x_clip)
    g = np.linspace(-half, half, rec.n_grid)
    return [np.array([float(x), float(y), float(pos[2])]) for x in g for y in g
            if ws.r_min <= float(np.hypot(x, y)) <= ws.r_max]


def _random_rotation(robot: Robot, rng: np.random.Generator) -> np.ndarray:
    return robot.fk(robot.sample_q(1, rng, ROT_MARGIN)[0])[1]


def _pool(robot: Robot, rec: Recipe, pos: np.ndarray, R: np.ndarray,
          rng: np.random.Generator) -> np.ndarray:
    """Deduplicated IK solutions of the target seeded at (pos, R), spanning its free coordinates."""
    n, raw = robot.n_joints, []
    if rec.pool == "se3":
        return dedup(robot.ik_all(pos, R), rec.dedup_tol, n)
    if rec.pool == "orientations":
        for _ in range(rec.n_orient):
            raw.extend(robot.ik_all(pos, _random_rotation(robot, rng)))
        return dedup(raw, rec.dedup_tol, n)
    if rec.pool == "ball":
        raw.extend(robot.ik_all(pos, R))
        ax = np.linspace(-rec.ball_r, rec.ball_r, rec.grid_n)
        for dx in ax:
            for dy in ax:
                for dz in ax:
                    p = pos + np.array([dx, dy, dz], dtype=np.float64)
                    if robot.shell.contains(p):
                        raw.extend(robot.ik_all(p, R))
        return dedup(raw, rec.dedup_tol, n)
    cells = _grid_cells(robot, rec, pos)
    if cells is None:
        return np.zeros((0, n))
    if rec.pool == "line_orient":
        rz = np.linspace(0.0, 2 * np.pi, rec.n_rot, endpoint=False)
        rz = (rz + rng.uniform(0.0, 2 * np.pi / rec.n_rot)) % (2 * np.pi)
    for p in cells:
        if rec.pool in ("line", "plane"):
            rots = [_random_rotation(robot, rng) for _ in range(rec.n_rot)]
        elif rec.pool == "line_orient":
            rots = [R @ _rot_z(float(d)) for d in rz]
        else:
            rots = [R]
        for Rt in rots:
            raw.extend(robot.ik_all(p, Rt))
    return dedup(raw, rec.dedup_tol, n, cap=rec.pool_cap, rng=rng)


# ---------------------------------------------------------------- one target

def _seed_ok(robot: Robot, rec: Recipe, pos: np.ndarray) -> bool:
    if rec.in_workspace and not robot.shell.contains(pos):
        return False
    if rec.seed_shell is not None:
        r = float(np.linalg.norm(pos))
        return rec.seed_shell[0] < r < rec.seed_shell[1]
    return True


def _chain(robot: Robot, rec: Recipe, q_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mode, axis, step, max_dist = rec.sweep
    fwd, bwd = (sweep_free_axis(robot, q_seed, mode, axis, step, max_dist, rec.jump, d) for d in (1.0, -1.0))
    full = bwd[:0:-1] + fwd
    return np.stack([q for _, q in full]).astype(np.float64), np.array([o for o, _ in full])


def _longest_run(sols: np.ndarray, jump: float) -> np.ndarray:
    """Indices of the longest run of consecutive rows without a wrapped jump above `jump`."""
    runs = [[0]]
    for i in range(1, len(sols)):
        if wrap_dist(sols[i], sols[i - 1]) > jump:
            runs.append([i])
        else:
            runs[-1].append(i)
    return np.asarray(max(runs, key=len), dtype=np.int64)


def synthesize_target(robot: Robot, rec: Recipe, rng: np.random.Generator) -> dict | None:
    """One target: seed pose, IK pool, seed solution and chain (the ordered set the picker reads).

    For SE(3) the chain is the longest joint-7-sorted self-motion segment (Franka, offset = q_7) or the
    IK branches sorted by distance from the home pose (UR5). None if no seed in n_tries passes."""
    for _ in range(rec.n_tries):
        q_rand = robot.sample_q(1, rng, rec.margin)[0]
        pos, R = robot.fk(q_rand)
        if not _seed_ok(robot, rec, pos):
            continue
        pool = _pool(robot, rec, pos, R, rng)
        if len(pool) < rec.min_size:
            continue
        if rec.pool == "se3":
            if rec.picker == "branches":
                home = np.array([wrap_dist(q, UR5_HOME) for q in pool])
                chain = pool[np.argsort(home, kind="stable")]
            else:
                ordered = pool[np.argsort(pool[:, 6], kind="stable")]
                chain = ordered[_longest_run(ordered, rec.jump)]
                if len(chain) < rec.min_size:
                    continue
            return dict(pos=pos, R=R, q_seed=q_rand.copy(), pool=pool, chain=chain,
                        offset=chain[:, 6].copy() if rec.picker == "j7" else np.zeros(len(chain)))
        if rec.pool == "orientations":
            q_seed = pool[0].copy()
        else:
            d = np.linalg.norm(robot.fk(pool)[0] - pos, axis=1)
            i = int(np.argmin(d))
            q_seed = pool[i].copy()
            if rec.pool == "ball" and d[i] > 1e-3:
                q_seed = q_rand.copy()
                if not any(np.linalg.norm(q_seed - s) <= rec.dedup_tol for s in pool):
                    pool = np.vstack([q_seed[None], pool])
        chain, offset = _chain(robot, rec, q_seed)
        if len(chain) < rec.min_size:
            continue
        if rec.pool in ("plane", "line", "orientations"):
            R = robot.fk(q_seed)[1]
        return dict(pos=pos, R=R, q_seed=q_seed, pool=pool, chain=chain, offset=offset)
    return None


# ---------------------------------------------------------------- pair pickers

def pick_rank(offset: np.ndarray, width: float) -> tuple[int, int]:
    """Chain indices at offset-rank percentiles 50 - w/2 and 50 + w/2."""
    S, w = len(offset), float(width)
    order = np.argsort(offset, kind="stable")
    i_a = int(min(S - 1, np.floor(max(0.0, 50.0 - w / 2.0) / 100.0 * (S - 1))))
    i_b = int(min(S - 1, np.ceil(min(100.0, 50.0 + w / 2.0) / 100.0 * (S - 1))))
    if i_a == i_b:
        i_b = min(S - 1, i_a + 1)
    return int(order[i_a]), int(order[i_b])


def pick_diameter(cands: np.ndarray, width: float) -> tuple[int, int]:
    """The pair (i < j) whose L2 joint distance is closest to width% of the candidates' diameter."""
    D = np.linalg.norm(cands[:, None, :] - cands[None, :, :], axis=-1)
    iu, ju = np.triu_indices(len(cands), k=1)
    dist = D[iu, ju]
    k = int(np.argmin(np.abs(dist - float(width) / 100.0 * float(dist.max()))))
    return int(iu[k]), int(ju[k])


def pick_branches(branches: np.ndarray, widths) -> list[tuple[int, int]]:
    """Per width, the pair at rank round(w/100 (P-1)) of the P sorted pairwise wrapped distances; a rank
    already taken moves up (or, at the top, down) to the nearest free one."""
    D = wrap_dist(branches[:, None, :], branches[None, :, :])
    iu, ju = np.triu_indices(len(branches), k=1)
    order = np.argsort(D[iu, ju], kind="stable")
    P, used, out = len(order), set(), []
    for w in widths:
        r0 = r = max(0, min(P - 1, int(round(float(w) / 100.0 * (P - 1)))))
        while r in used and r < P - 1:
            r += 1
        if r in used:
            r = r0
            while r in used and r > 0:
                r -= 1
        used.add(r)
        out.append((int(iu[order[r]]), int(ju[order[r]])))
    return out


def pick_third(cands: np.ndarray, q_a: np.ndarray, q_b: np.ndarray) -> int:
    """Index of the candidate closest to completing an equilateral triangle with q_a and q_b in
    (wrapped) joint space: argmin max(|d(c,a) - d(a,b)|, |d(c,b) - d(a,b)|), excluding q_a and q_b."""
    d_ab = float(wrap_dist(q_a, q_b))
    d_a, d_b = wrap_dist(cands, q_a), wrap_dist(cands, q_b)
    score = np.maximum(np.abs(d_a - d_ab), np.abs(d_b - d_ab))
    score[(d_a < 1e-9) | (d_b < 1e-9)] = np.inf
    return int(np.argmin(score))


def pick_pairs(t: dict, rec: Recipe, widths=PAIR_WIDTHS) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(q_a, q_b, q_c) per pair width for one synthesized target.

    q_c is a third solution for latent planes (web page); it does not affect q_a, q_b. Candidates: the
    chain and pool (chain, diameter pickers), the segment (j7) or the branches (branches)."""
    chain, pool = t["chain"], t["pool"]
    if rec.picker == "diameter":
        cands = np.vstack([chain, pool]) if len(pool) else chain
        idx = [pick_diameter(cands, w) for w in widths]
    elif rec.picker == "branches":
        cands = chain
        idx = pick_branches(chain, widths)
    else:
        cands = chain if rec.picker == "j7" else np.vstack([chain, pool])
        idx = [pick_rank(t["offset"], w) for w in widths]
    out = []
    for i_a, i_b in idx:
        q_a, q_b = cands[i_a].copy(), cands[i_b].copy()
        out.append((q_a, q_b, cands[pick_third(cands, q_a, q_b)].copy()))
    return out


# ---------------------------------------------------------------- target sets

def _attempt(job: tuple[str, str, int]) -> dict | None:
    robot, constraint, seed = job
    return synthesize_target(get_robot(robot), recipe(robot, constraint), np.random.default_rng(seed))


def synthesize_targets(robot: str, constraint: str, n_targets: int = 30, seed: int = 0,
                       workers: int = 1) -> list[dict]:
    """The first n_targets successful attempts; attempt k uses default_rng(seed + k)."""
    jobs = ((robot, constraint, seed + k) for k in range(n_targets * 100))
    out: list[dict] = []
    if workers <= 1:
        results = map(_attempt, jobs)
        for t in results:
            if t is not None:
                out.append(t)
            if len(out) == n_targets:
                break
    else:
        with mp.get_context("fork").Pool(workers) as p:
            for t in p.imap(_attempt, jobs):
                if t is not None:
                    out.append(t)
                if len(out) == n_targets:
                    break
    if len(out) < n_targets:
        raise RuntimeError(f"{robot}/{constraint}: only {len(out)} of {n_targets} targets")
    return out


def build_targets(robot: str, constraint: str, n_targets: int = 30, seed: int = 0,
                  widths=PAIR_WIDTHS, workers: int = 1) -> tuple[dict, list[dict]]:
    """Compact target set {pos, R, q_seed, pair_target, pair_width, q_a, q_b, q_c, q_dist} and the full
    per-target synthesis records (pools, chains)."""
    rec = recipe(robot, constraint)
    targets = synthesize_targets(robot, constraint, n_targets, seed, workers)
    rows = [(i, w, *abc) for i, t in enumerate(targets) for w, abc in zip(widths, pick_pairs(t, rec, widths))]
    q_a, q_b = np.stack([r[2] for r in rows]), np.stack([r[3] for r in rows])
    return {
        "pos": np.stack([t["pos"] for t in targets]),
        "R": np.stack([t["R"] for t in targets]),
        "q_seed": np.stack([t["q_seed"] for t in targets]),
        "pair_target": np.array([r[0] for r in rows], dtype=np.int64),
        "pair_width": np.array([r[1] for r in rows], dtype=np.int64),
        "q_a": q_a, "q_b": q_b, "q_c": np.stack([r[4] for r in rows]),
        "q_dist": wrap_dist(q_a, q_b),
    }, targets
