"""Intrinsic-dimension estimators: Score-SVD and Score-SVD-dagger on the learned score, and the
neighbourhood baselines (local MLE, local PCA, TwoNN) on joint-space samples. Also the GT anchor rules."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from constraint_ik.constraints import CONSTRAINTS, LOOSE, Target, is_valid
from constraint_ik.encoding import q_to_sincos
from constraint_ik.pipeline import DiffusionIK
from constraint_ik.robots import Robot

# Nearest-neighbour distance (rad) above which a joint-space sample counts as isolated.
ISO_THRESHOLD = 1.5
GT_PATH = {"ur5": "data/ground_truth_ur5.pt", "franka": "data/ground_truth_franka.pt"}
CKPT_PATH = "checkpoints/{robot}.pt"


# ---------------------------------------------------------------- ground-truth targets and anchors

def load_gt(path: str, n_per_constraint: int | None = None) -> dict[str, list[dict]]:
    """GT file (list of dicts with cond_pos, cond_R6d, R, sols_dedup) grouped by constraint, file order kept."""
    by_c = defaultdict(list)
    for e in torch.load(path, map_location="cpu", weights_only=False):
        by_c[e["constraint_type"]].append(e)
    return {c: by_c[c][:n_per_constraint] for c in CONSTRAINTS if by_c[c]}


def valid_samples(ik: DiffusionIK, robot: Robot, target: Target, n: int, seed: int) -> np.ndarray:
    """n model samples (seeded) filtered by LOOSE validity."""
    q = ik.sample(target, n, seed=seed)
    return q[is_valid(robot, q, target, LOOSE)]


def uniform_anchors(n_sols: int, n: int, rng) -> np.ndarray:
    """n GT solution indices uniformly without replacement (rng: np.random.Generator or RandomState)."""
    return rng.choice(n_sols, min(n, n_sols), replace=False)


def model_anchors(sols: np.ndarray, q_valid: np.ndarray, n: int, rng) -> np.ndarray:
    """The n GT solutions nearest (L2 in sincos space) to a valid model sample; uniform from `rng`
    when there are no valid samples or at most n solutions."""
    n_anc = min(n, len(sols))
    if len(q_valid) == 0 or len(sols) <= n_anc:
        return uniform_anchors(len(sols), n, rng)
    sols_sc = q_to_sincos(torch.tensor(sols, dtype=torch.float32)).numpy()
    qv_sc = q_to_sincos(torch.tensor(np.asarray(q_valid, dtype=np.float32))).numpy()
    d_min = np.concatenate([np.linalg.norm(s[:, None, :] - qv_sc[None], axis=-1).min(axis=1)
                            for s in np.array_split(sols_sc, max(1, len(sols_sc) // 1024))])
    return np.argpartition(d_min, n_anc - 1)[:n_anc]


# ---------------------------------------------------------------- Score-SVD

@torch.no_grad()
def score_matrix(ik: DiffusionIK, x0: torch.Tensor, cond: torch.Tensor, t: int, K: int,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
    """(K, D) scores at x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, eps ~ N(0, I) (global RNG unless given)."""
    a = ik.alphas[t]
    sa, sigma = float(a.sqrt().item()), float((1 - a).sqrt().item())
    if noise is None:
        noise = torch.randn(K, x0.numel(), device=ik.device)
    return ik.score(sa * x0.expand(K, -1) + sigma * noise, t, cond)


def singular_values(S: torch.Tensor) -> np.ndarray:
    """Descending singular values of a score matrix (or a batch of them)."""
    return torch.linalg.svd(S, full_matrices=False)[1].cpu().numpy()


def gap_rank(sv: np.ndarray, dagger: bool = False) -> int:
    """k_hat = argmax_i (sigma_i - sigma_{i+1}), i = 1..D-1; dagger appends sigma_{D+1} = 0 and searches i = 1..D."""
    sv = np.asarray(sv, dtype=np.float64)
    sv = sv[sv > 1e-12]
    if len(sv) < 2:
        return len(sv)
    if dagger:
        sv = np.append(sv, 0.0)
    return int(np.argmax(sv[:-1] - sv[1:])) + 1


def id_from_spectrum(sv: np.ndarray, D: int) -> tuple[int, int]:
    """(d_hat Score-SVD, d_hat Score-SVD-dagger) = D - k_hat."""
    return D - gap_rank(sv), D - gap_rank(sv, dagger=True)


def score_svd(ik: DiffusionIK, x0: torch.Tensor, cond: torch.Tensor, t: int = 15,
              K: int = 50) -> tuple[np.ndarray, int, int]:
    """Spectrum and both estimates at one anchor x0 (D,)."""
    sv = singular_values(score_matrix(ik, x0, cond, t, K))
    return (sv, *id_from_spectrum(sv, ik.dim))


# ---------------------------------------------------------------- neighbourhood baselines

def _knn(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Distances and indices of the k nearest neighbours of every row, self excluded."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    d, idx = nn.kneighbors(X)
    return d[:, 1:], idx[:, 1:]


def _kept(d1: np.ndarray, iso: float | None) -> np.ndarray:
    return np.ones(len(d1), bool) if iso is None else ~(d1 > iso)


def local_mle(X: np.ndarray, k: int = 10, iso: float | None = ISO_THRESHOLD,
              clip_max: float = 50.0) -> float:
    """Levina-Bickel MLE with the MacKay-Ghahramani aggregate (Pope et al. 2021, eq. 2):
    the number of log-ratio terms log(T_k / T_j) over their sum, isolated points excluded."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 2:
        return 0.0
    d, _ = _knn(X, min(k, len(X) - 1))
    d = np.sort(d[_kept(d[:, 0], iso)], axis=1)
    logs = np.log(np.maximum(d[:, -1:], 1e-10) / np.maximum(d[:, :-1], 1e-10))
    total = 0.0
    for row_sum in logs.sum(axis=1):
        total += float(row_sum)
    return float(np.clip(logs.size / total, 0.0, clip_max)) if total > 0 and logs.size else 0.0


def local_pca(X: np.ndarray, k: int = 10, var_threshold: float = 0.9,
              iso: float | None = ISO_THRESHOLD) -> float:
    """Median over points of the number of local principal components (point + k neighbours)
    reaching `var_threshold` cumulative variance; isolated points count as 0."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 2:
        return 0.0
    d, idx = _knn(X, min(k, len(X) - 1))
    kept = _kept(d[:, 0], iso)
    per_point = np.zeros(len(X), dtype=np.int32)
    n_comp = min(idx.shape[1], X.shape[1])
    if n_comp >= 2:
        for i in np.flatnonzero(kept):
            local = np.vstack([X[i:i + 1], X[idx[i]]])
            cumvar = np.cumsum(PCA(n_components=n_comp).fit(local).explained_variance_ratio_)
            per_point[i] = min(int(np.searchsorted(cumvar, var_threshold) + 1), n_comp)
    return float(np.median(per_point))


def twonn(X: np.ndarray, k: int = 10, iso: float | None = ISO_THRESHOLD) -> float:
    """Median over points of the TwoNN estimate 1 / log(r2 / r1) (Facco et al. 2017); isolated
    or degenerate points count as 0. `k` only sizes the neighbour query."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 3:
        return 0.0
    d, _ = _knn(X, max(2, min(k, len(X) - 1)))
    r1, r2 = d[:, 0], d[:, 1]
    ok = _kept(r1, iso) & (r1 > 1e-10) & (r2 > r1 * 1.00001)
    per_point = np.zeros(len(X))
    per_point[ok] = 1.0 / np.log(r2[ok] / r1[ok])
    return float(np.median(per_point))
