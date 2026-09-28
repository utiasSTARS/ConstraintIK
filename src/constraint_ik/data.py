"""Training data: per-family target sampling with IK pooling, near-duplicate removal, dataset
bundles and the stratified batch sampler; also the Franka ground-truth (GT) targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from constraint_ik.constraints import CONSTRAINTS, MASKS
from constraint_ik.encoding import Normalizer, R_to_R6d, q_to_sincos
from constraint_ik.robots import Robot, Shell

I_R6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
SEED_MARGIN = 0.15            # joint-range margin of the random seed configurations
SEED_NORM = (0.25, 0.65)      # accepted |p_seed| (m) for Orientation and Line+Orient. seeds
LINE_X_MAX = 0.50             # |p_x| bound (m) of the Line+Orient. sweep


# ---------------------------------------------------------------- near-duplicate removal

def dedup(sols: np.ndarray, tol: float, cap: int | None = None) -> np.ndarray:
    """Greedy near-duplicate removal on raw joint angles: in order, keep a row if its L2 distance to
    every kept row exceeds tol; stop once `cap` rows are kept. (m, n) -> (k, n)."""
    X = np.asarray(sols, dtype=np.float64)
    if len(X) == 0:
        return X
    tree, kept, keep = cKDTree(X), np.zeros(len(X), dtype=bool), []
    for i, x in enumerate(X):
        near = np.asarray(tree.query_ball_point(x, tol + 1e-6), dtype=np.int64)  # superset of the tol-ball
        if all(np.linalg.norm(x - X[j]) > tol for j in near[kept[near]]):
            kept[i] = True
            keep.append(i)
            if cap is not None and len(keep) >= cap:
                break
    return X[keep]


def _shuffled_dedup(raw: np.ndarray, cap: int, tol: float, rng: np.random.Generator) -> np.ndarray:
    """dedup after a random permutation, so the kept set is not biased toward the first swept values."""
    if len(raw) == 0:
        return raw
    return dedup(raw[rng.permutation(len(raw))], tol, cap)


# ---------------------------------------------------------------- IK pooling

def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _in_annulus(shell: Shell, x: float, y: float) -> bool:
    return shell.r_min <= np.hypot(x, y) <= shell.r_max


def _seed(robot: Robot, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """FK pose of a random configuration: gives reachable orientations."""
    return robot.fk(robot.sample_q(1, rng, margin=SEED_MARGIN)[0])


def _shell_pos(shell: Shell, rng: np.random.Generator) -> np.ndarray:
    """Uniform position in the workspace shell (rejection sampling)."""
    while True:
        x = rng.uniform(-shell.r_max, shell.r_max)
        y = rng.uniform(-shell.r_max, shell.r_max)
        z = rng.uniform(shell.z_min, shell.z_max)
        if _in_annulus(shell, x, y):
            return np.array([x, y, z])


def _plane_draw(shell: Shell, z: float) -> Callable:
    """(x, y) uniform in the shell annulus at height z; None if rejected."""
    def draw(rng):
        x = rng.uniform(-shell.r_max, shell.r_max)
        y = rng.uniform(-shell.r_max, shell.r_max)
        return np.array([x, y, z]) if _in_annulus(shell, x, y) else None
    return draw


def _line_draw(shell: Shell, y: float, z: float) -> Callable:
    """x uniform on the line (y, z) inside the annulus; None if rejected."""
    def draw(rng):
        x = rng.uniform(-shell.r_max, shell.r_max)
        return np.array([x, y, z]) if _in_annulus(shell, x, y) else None
    return draw


def _random_queries(robot: Robot, rng: np.random.Generator, draw: Callable,
                    R: np.ndarray | None, attempts: int) -> Iterator:
    """(pos, R) IK queries from `attempts` position draws; a random reachable R per query if R is None."""
    for _ in range(attempts):
        p = draw(rng)
        if p is not None:
            yield p, (R if R is not None else _seed(robot, rng)[1])


def _ball_queries(shell: Shell, center: np.ndarray, R: np.ndarray, n: int,
                  radius: float = 0.25) -> Iterator:
    """n^3 grid over a cube of half-width `radius` about center, kept inside the (spherical) shell radii."""
    ax = np.linspace(-radius, radius, n)
    for dx in ax:
        for dy in ax:
            for dz in ax:
                x, y, z = float(center[0]) + dx, float(center[1]) + dy, float(center[2]) + dz
                if shell.r_min <= float(np.sqrt(x**2 + y**2 + max(0.0, z)**2)) <= shell.r_max:
                    yield np.array([x, y, z]), R


def _line_rz_queries(shell: Shell, y: float, z: float, R_axes: np.ndarray,
                     xs: np.ndarray, rzs: np.ndarray) -> Iterator:
    """x along the line (y, z) times rotations about the pinned body z axis."""
    for x in xs:
        if _in_annulus(shell, x, y):
            for a in rzs:
                yield np.array([x, y, z]), R_axes @ _Rz(float(a))


def _pool(robot: Robot, queries: Iterable, max_raw: float = np.inf) -> np.ndarray:
    """All IK solutions of the queries, stopping once at least max_raw are pooled. (m, n)."""
    blocks, m = [], 0
    for p, R in queries:
        blocks.append(robot.ik_all(p, R))
        m += len(blocks[-1])
        if m >= max_raw:
            break
    return np.concatenate(blocks) if blocks else np.empty((0, robot.n_joints))


def _line_x_range(p_seed: np.ndarray, x_seed: float) -> tuple[float, float]:
    """Line+Orient. sweep range about the seed x, inside a 0.65 m ball and |x| <= 0.5 m."""
    y, z = float(p_seed[1]), float(p_seed[2])
    half = float(np.sqrt(max(0, SEED_NORM[1]**2 - y**2 - max(0, z)**2)))
    return max(-LINE_X_MAX, x_seed - half), min(LINE_X_MAX, x_seed + half)


def _seed_ok(p: np.ndarray) -> bool:
    return SEED_NORM[0] < float(np.linalg.norm(p)) < SEED_NORM[1]


# ---------------------------------------------------------------- training-set generators
# Each returns [(q (m, n), pos (3,), R6d (6,), target id)]; pos has zeros on free axes.

def _se3(robot, rng, n_targets, tol=0.1):
    """Random reachable pose, all IK branches (Franka: joint-7 sweep)."""
    out, tried = [], 0
    while len(out) < n_targets and tried < n_targets * 5:
        tried += 1
        pos, R = _seed(robot, rng)
        sols = dedup(robot.ik_all(pos, R), tol)
        if len(sols):
            out.append((sols, pos, R_to_R6d(R).astype(np.float32), len(out)))
    return out


def _posonly(robot, rng, n_targets, cap, n_orient=30, tol=0.05):
    """Random shell position, IK over n_orient random reachable orientations."""
    out, tried = [], 0
    while len(out) < n_targets and tried < n_targets * 5:
        tried += 1
        p = _shell_pos(robot.shell, rng)
        raw = _pool(robot, _random_queries(robot, rng, lambda _: p, None, n_orient))
        sols = dedup(raw, tol, cap)
        if len(sols) >= 3:
            out.append((sols, p, I_R6D, len(out)))
    return out


def _plane(robot, rng, n_targets, cap, tol=0.05):
    """z* on a uniform grid; (x, y) in the annulus and random R."""
    out = []
    for i, z in enumerate(np.linspace(robot.shell.z_min, robot.shell.z_max, n_targets)):
        raw = _pool(robot, _random_queries(robot, rng, _plane_draw(robot.shell, z), None, cap * 10),
                    cap * 3)
        out.append((_shuffled_dedup(raw, cap, tol, rng), np.array([0, 0, z]), I_R6D, i))
    return out


def _line(robot, rng, n_targets, cap, tol=0.05):
    """(y*, z*) uniform in the shell box; x along the line and random R."""
    out, sh = [], robot.shell
    for i in range(n_targets):
        y = rng.uniform(-sh.r_max, sh.r_max)
        z = rng.uniform(sh.z_min, sh.z_max)
        raw = _pool(robot, _random_queries(robot, rng, _line_draw(sh, y, z), None, cap * 10), cap * 3)
        out.append((_shuffled_dedup(raw, cap, tol, rng), np.array([0, y, z]), I_R6D, i))
    return out


def _orient(robot, rng, n_targets, cap, grid_n=12, tol=0.05):
    """R* from a random seed pose; positions on a grid ball about the seed position."""
    out = []
    for _ in range(n_targets):
        p_seed, R = _seed(robot, rng)
        if not _seed_ok(p_seed):
            continue
        raw = _pool(robot, _ball_queries(robot.shell, p_seed, R, grid_n))
        out.append((_shuffled_dedup(raw, cap, tol, rng), np.zeros(3), R_to_R6d(R).astype(np.float32),
                    len(out)))
    return out


def _plane_orient(robot, rng, n_targets, cap, tol=0.05):
    """z* uniform, R* from a random seed pose; (x, y) in the annulus."""
    out, tried = [], 0
    while len(out) < n_targets and tried < n_targets * 10:
        tried += 1
        z = rng.uniform(robot.shell.z_min, robot.shell.z_max)
        R = _seed(robot, rng)[1]
        raw = _pool(robot, _random_queries(robot, rng, _plane_draw(robot.shell, z), R, cap * 10),
                    cap * 3)
        sols = _shuffled_dedup(raw, cap, tol, rng)
        if len(sols) >= 2:
            out.append((sols, np.array([0, 0, z]), R_to_R6d(R).astype(np.float32), len(out)))
    return out


def _line_orient(robot, rng, n_targets, cap, n_x=30, n_rz=30, tol=0.05):
    """(y*, z*, R* e_z) from a random seed pose; x and the rotation about e_z swept on a grid."""
    out = []
    for _ in range(n_targets):
        p_seed, R = _seed(robot, rng)
        if not _seed_ok(p_seed):
            continue
        x_lo, x_hi = _line_x_range(p_seed, float(p_seed[0]))
        if x_hi - x_lo < 0.05:
            continue
        rzs = np.linspace(0, 2 * np.pi, n_rz, endpoint=False)
        rzs = (rzs + rng.uniform(0, 2 * np.pi / n_rz)) % (2 * np.pi)
        y, z = float(p_seed[1]), float(p_seed[2])
        raw = _pool(robot, _line_rz_queries(robot.shell, y, z, R, np.linspace(x_lo, x_hi, n_x), rzs))
        out.append((_shuffled_dedup(raw, cap, tol, rng), np.array([0, y, z]),
                    R_to_R6d(R).astype(np.float32), len(out)))
    return out


# Haar-uniform targets used only by the Franka Orientation and Line+Orient. size-sweep models.

def _uniform_so3(n: int, rng: np.random.Generator) -> np.ndarray:
    return Rotation.random(n, random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()


def _cylinder_grid(shell: Shell, n: int, rng: np.random.Generator) -> np.ndarray:
    """About n jittered points on an (r, theta, z) grid inside the shell, shuffled."""
    cell = ((shell.r_max**2 - shell.r_min**2) * np.pi * (shell.z_max - shell.z_min) / max(1, n)) ** (1 / 3)
    rs = np.linspace(shell.r_min, shell.r_max, max(2, int(round((shell.r_max - shell.r_min) / cell))))
    zs = np.linspace(shell.z_min, shell.z_max, max(2, int(round((shell.z_max - shell.z_min) / cell))))
    pts = []
    for r in rs:
        n_th = max(1, int(round(2 * np.pi * max(r, 1e-6) / cell)))
        ths = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
        ths = (ths + rng.uniform(0, 2 * np.pi / n_th)) % (2 * np.pi)
        pts += [(r * np.cos(th), r * np.sin(th), z) for z in zs for th in ths]
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts + rng.uniform(-0.5 * cell, 0.5 * cell, size=pts.shape)
    pts = pts[shell.contains(pts)]
    rng.shuffle(pts)
    return pts[:n]


def _yz_grid(shell: Shell, n: int, rng: np.random.Generator) -> np.ndarray:
    """About n jittered (y, z) points on a grid over [-r_max, r_max] x [z_min, z_max], shuffled."""
    cell = ((2 * shell.r_max) * (shell.z_max - shell.z_min) / max(1, n)) ** 0.5
    ys = np.linspace(-shell.r_max, shell.r_max, max(2, int(round(2 * shell.r_max / cell))))
    zs = np.linspace(shell.z_min, shell.z_max, max(2, int(round((shell.z_max - shell.z_min) / cell))))
    pts = np.array([(y, z) for y in ys for z in zs], dtype=np.float64)
    pts += rng.uniform(-0.5 * cell, 0.5 * cell, size=pts.shape)
    rng.shuffle(pts)
    return pts[:n]


def _orient_grid(robot, rng, n_targets, cap, n_pos=400, tol=0.05):
    """Haar R*; positions drawn from a workspace grid pool."""
    Rs = _uniform_so3(n_targets * 3, rng)
    pool = _cylinder_grid(robot.shell, n_pos * 10, rng)
    out, tried = [], 0
    while len(out) < n_targets and tried < len(Rs):
        R = Rs[tried]
        tried += 1
        pick = rng.choice(len(pool), size=min(n_pos, len(pool)), replace=False)
        raw = _pool(robot, ((p, R) for p in pool[pick]), cap * 4)
        sols = _shuffled_dedup(raw, cap, tol, rng)
        if len(sols) >= 4:
            out.append((sols, np.zeros(3), R_to_R6d(R).astype(np.float32), len(out)))
    return out


def _line_orient_grid(robot, rng, n_targets, cap, n_x=25, n_rz=25, tol=0.05):
    """(y*, z*) on a grid and Haar R*; x and the rotation about e_z swept on a grid."""
    sh = robot.shell
    yz = _yz_grid(sh, n_targets * 3, rng)
    Rs = _uniform_so3(len(yz), rng)
    out, tried = [], 0
    while len(out) < n_targets and tried < len(yz):
        y, z, R = float(yz[tried, 0]), float(yz[tried, 1]), Rs[tried]
        tried += 1
        if np.hypot(0.0, y) > sh.r_max:
            continue
        half = float(np.sqrt(max(0, sh.r_max**2 - y**2 - max(0, z)**2)))
        if 2 * half < 0.05:
            continue
        rzs = np.linspace(0, 2 * np.pi, n_rz, endpoint=False)
        rzs = (rzs + rng.uniform(0, 2 * np.pi / n_rz)) % (2 * np.pi)
        raw = _pool(robot, _line_rz_queries(sh, y, z, R, np.linspace(-half, half, n_x), rzs))
        sols = _shuffled_dedup(raw, cap, tol, rng)
        if len(sols) >= 4:
            out.append((sols, np.array([0, y, z]), R_to_R6d(R).astype(np.float32), len(out)))
    return out


# (generator, kwargs) per family. "2x" is the paper's training set (Table target-sampling);
# "baseline" and "grid" are the Franka sources of the per-family size-sweep models.
RECIPES: dict[str, dict[str, tuple[Callable, dict]]] = {
    "2x": {
        "se3":          (_se3, dict(n_targets=8000)),
        "posonly":      (_posonly, dict(n_targets=3000, cap=200)),
        "plane":        (_plane, dict(n_targets=400, cap=600)),
        "line":         (_line, dict(n_targets=600, cap=400)),
        "orient":       (_orient, dict(n_targets=800, cap=600)),
        "plane_orient": (_plane_orient, dict(n_targets=1000, cap=300)),
        "line_orient":  (_line_orient, dict(n_targets=800, cap=500)),
    },
    "baseline": {
        "se3":          (_se3, dict(n_targets=4000)),
        "posonly":      (_posonly, dict(n_targets=1500, cap=200)),
        "plane":        (_plane, dict(n_targets=200, cap=600)),
        "line":         (_line, dict(n_targets=300, cap=400)),
        "plane_orient": (_plane_orient, dict(n_targets=500, cap=300)),
    },
    "grid": {
        "orient":       (_orient_grid, dict(n_targets=800, cap=200)),
        "line_orient":  (_line_orient_grid, dict(n_targets=800, cap=250)),
    },
}


def generate_shard(robot: Robot, constraint: str, recipe: str = "2x", seed: int = 0) -> dict:
    """Per-family shard: q_ends (N, n), cond_pos, cond_R6d, cond_mask, cluster_ids (target id),
    ee_pos (FK position; the target position for SE(3)), constraint_type, T (unused, 32), C = n."""
    fn, kw = RECIPES[recipe][constraint]
    parts = fn(robot, np.random.default_rng(seed), **kw)
    m = [len(p[0]) for p in parts]
    q = np.concatenate([p[0] for p in parts])
    pos = np.repeat(np.array([p[1] for p in parts], dtype=np.float64), m, axis=0)
    ee = pos if constraint == "se3" else robot.fk(q)[0]
    return {
        "q_ends": torch.from_numpy(q.astype(np.float32)),
        "cond_pos": torch.from_numpy(pos.astype(np.float32)),
        "cond_R6d": torch.from_numpy(np.repeat(np.array([p[2] for p in parts]), m, axis=0)),
        "cond_mask": torch.from_numpy(np.tile(np.asarray(MASKS[constraint], dtype=np.float32), (len(q), 1))),
        "cluster_ids": torch.from_numpy(np.repeat(np.array([p[3] for p in parts], dtype=np.int64), m)),
        "ee_pos": torch.from_numpy(ee.astype(np.float32)),
        "constraint_type": constraint,
        "T": 32,
        "C": int(q.shape[1]),
    }


# ---------------------------------------------------------------- bundles and batches

def build_bundle(shards: list[dict], robot: str) -> dict:
    """Merge per-family shards into one training bundle, families in CONSTRAINTS order.
    q_normalizer (joint space) is informational; the model's sincos normalizer is fit at training."""
    by_type = {s["constraint_type"]: s for s in shards}
    types = [c for c in CONSTRAINTS if c in by_type]
    sizes = [len(by_type[c]["q_ends"]) for c in types]
    offsets = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    cat = {k: torch.cat([by_type[c][k] for c in types]).float()
           for k in ("q_ends", "cond_pos", "cond_R6d", "cond_mask")}
    return {
        **cat,
        "type_indices": {c: torch.arange(offsets[i], offsets[i + 1]) for i, c in enumerate(types)},
        "type_names": types,
        "type_idx": torch.cat([torch.full((m,), i, dtype=torch.long) for i, m in enumerate(sizes)]),
        "q_normalizer": Normalizer.fit(cat["q_ends"]).to_dict(),
        "pos_normalizer": Normalizer.fit(cat["cond_pos"]).to_dict(),
        "R6d_normalizer": None,
        "T": 32,
        "C": int(cat["q_ends"].shape[1]),
        "robot": robot,
    }


@dataclass
class TrainingSet:
    """Model inputs of a bundle on one device: x = z-scored sincos (N, 2n), cond (N, 15)."""
    x: torch.Tensor
    cond: torch.Tensor
    type_indices: list[torch.Tensor]
    type_names: list[str]
    x_normalizer: Normalizer
    pos_normalizer: Normalizer
    robot: str


def load_training_set(path: str, device: str | torch.device) -> TrainingSet:
    """Fits the sincos normalizer on `device` (the paper models' normalizers were fit on CUDA;
    CPU reductions differ in the last bit)."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    sc = q_to_sincos(b["q_ends"].to(device))
    x_norm = Normalizer.fit(sc)
    pos_norm = Normalizer.from_dict(b["pos_normalizer"])
    cond = torch.cat([pos_norm.apply(b["cond_pos"].to(device)), b["cond_R6d"].to(device),
                      b["cond_mask"].to(device)], dim=-1)
    return TrainingSet(
        x=x_norm.apply(sc), cond=cond,
        type_indices=[b["type_indices"][c] for c in b["type_names"]],
        type_names=list(b["type_names"]), x_normalizer=x_norm, pos_normalizer=pos_norm,
        robot=b["robot"])


class StratifiedSampler:
    """Index batches with batch // K draws (with replacement) per family, one extra for the first
    batch % K families, then shuffled. Uses torch's global CPU RNG unless a generator is given."""

    def __init__(self, type_indices: list[torch.Tensor], batch_size: int,
                 generator: torch.Generator | None = None):
        if batch_size < len(type_indices):
            raise ValueError("batch_size must be at least the number of families")
        self.pools = type_indices
        self.batch_size = batch_size
        self.gen = generator

    def __iter__(self) -> Iterator[torch.Tensor]:
        K = len(self.pools)
        per, extra = divmod(self.batch_size, K)
        while True:
            idx = torch.cat([pool[torch.randint(0, len(pool), (per + (i < extra),), generator=self.gen)]
                             for i, pool in enumerate(self.pools)])
            yield idx[torch.randperm(len(idx), generator=self.gen)]


# ---------------------------------------------------------------- ground-truth targets (Franka)
# Held-out evaluation targets with dense solution sets (no cap): 200 IK queries per Plane/Line
# target, 50 orientations per Position target, 8^3 ball grid for Orientation and a 15 x 15 grid
# without random offset for Line+Orient.

def _gt_entry(c: str, pos: np.ndarray, R: np.ndarray, sols: np.ndarray) -> dict:
    k = sum(MASKS[c])
    return {
        "constraint_type": c, "codimension": k, "fiber_dim": sols.shape[1] - k,
        "mask": np.array(MASKS[c], dtype=np.float32),
        "cond_pos": np.array(pos, dtype=np.float32),
        "cond_R6d": R_to_R6d(R).astype(np.float32),
        "R": R.astype(np.float64),
        "n_raw": len(sols), "n_dedup": len(sols), "sols_dedup": sols,
        "best_k_6d": None, "silhouette_6d": None, "dbscan_max_clusters": None,
        "n_exact_branches": len(sols),
    }


def _gt_family(robot: Robot, c: str, n: int, rng: np.random.Generator) -> list[dict]:
    sh, out, tried = robot.shell, [], 0
    while len(out) < n and tried < n * 20:
        tried += 1
        if c == "se3":
            pos, R = _seed(robot, rng)
            sols, min_sols = dedup(robot.ik_all(pos, R), 0.1), 2
        elif c == "posonly":
            pos, R = _shell_pos(sh, rng), np.eye(3)
            raw = _pool(robot, _random_queries(robot, rng, lambda _: pos, None, 50))
            sols, min_sols = dedup(raw, 0.05), 3
        elif c in ("plane", "plane_orient"):
            z = rng.uniform(sh.z_min, sh.z_max)
            pos, R = np.array([0, 0, z]), (_seed(robot, rng)[1] if c == "plane_orient" else np.eye(3))
            raw = _pool(robot, _random_queries(robot, rng, _plane_draw(sh, z),
                                               R if c == "plane_orient" else None, 200))
            sols, min_sols = dedup(raw, 0.05), 2 if c == "plane_orient" else 3
        elif c == "line":
            y, z = rng.uniform(-sh.r_max, sh.r_max), rng.uniform(sh.z_min, sh.z_max)
            pos, R = np.array([0, y, z]), np.eye(3)
            sols, min_sols = dedup(_pool(robot, _random_queries(robot, rng, _line_draw(sh, y, z),
                                                                None, 200)), 0.05), 3
        elif c == "orient":
            p_seed, R = _seed(robot, rng)
            if not _seed_ok(p_seed):
                continue
            pos = np.zeros(3)
            sols, min_sols = dedup(_pool(robot, _ball_queries(sh, p_seed, R, 8)), 0.05), 3
        else:
            p_seed, R = _seed(robot, rng)
            y, z = float(p_seed[1]), float(p_seed[2])
            if not _seed_ok(p_seed):
                continue
            x_lo, x_hi = _line_x_range(p_seed, float(p_seed[0]))
            if x_hi - x_lo < 0.05:
                continue
            pos = np.array([0, y, z])
            queries = _line_rz_queries(sh, y, z, R, np.linspace(x_lo, x_hi, 15),
                                       np.linspace(0, 2 * np.pi, 15, endpoint=False))
            sols, min_sols = dedup(_pool(robot, queries), 0.05), 2
        if len(sols) >= min_sols:
            out.append(_gt_entry(c, pos, R, sols))
    return out


def generate_gt(robot: Robot, n_per_type: int = 50, seed: int = 0) -> list[dict]:
    """GT targets for all families (CONSTRAINTS order) from one RNG stream."""
    rng = np.random.default_rng(seed)
    return [e for c in CONSTRAINTS for e in _gt_family(robot, c, n_per_type, rng)]
