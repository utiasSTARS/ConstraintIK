"""Pair pickers, dedup, lerp / slerp and the t_bar rule on synthetic inputs (CPU, no IKFast)."""
import numpy as np
import pytest
import torch

from constraint_ik.interpolation import lerp, pick_t_bar, slerp, t_bar_scores
from constraint_ik.targets import (
    dedup, pick_branches, pick_diameter, pick_rank, pick_third, recipe, wrap_dist,
)


def test_pick_rank_percentiles():
    offset = np.linspace(-1.0, 1.0, 11)[::-1].copy()      # sorted by offset: index 10 first
    assert pick_rank(offset, 100) == (10, 0)
    assert pick_rank(offset, 10) == (6, 4)
    assert pick_rank(np.arange(11.0), 0) == (5, 6)          # equal ranks are nudged apart


def test_pick_diameter_extremes():
    cands = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.5, 0.0]])
    assert pick_diameter(cands, 100) == (0, 3)
    assert pick_diameter(cands, 15) == (2, 3)               # 0.5 is closest to 15% of 3.5


def test_pick_branches_distinct_ranks():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 2 * np.pi, (4, 6))
    picks = pick_branches(b, (10, 25, 50, 75, 100))
    assert len(set(picks)) == 5 and all(i < j for i, j in picks)
    D = wrap_dist(b[:, None], b[None])
    assert D[picks[-1]] == D[np.triu_indices(4, 1)].max()


def test_pick_third_excludes_endpoints():
    cands = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8], [0.5, 0.1]])
    assert pick_third(cands, cands[0], cands[1]) == 2


def test_dedup_greedy_cap_and_shuffle():
    sols = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.0], [2.0, 0.0], [2.02, 0.0]])
    assert np.array_equal(dedup(sols, 0.05, 2), sols[[0, 2, 3]])
    assert len(dedup(sols, 0.05, 2, cap=2)) == 2
    shuffled = dedup(sols, 0.05, 2, rng=np.random.default_rng(0))
    assert len(shuffled) == 3 and dedup([], 0.05, 2).shape == (0, 2)


def test_recipes():
    assert recipe("ur5", "posonly").picker == "diameter" and recipe("franka", "posonly").picker == "chain"
    assert recipe("ur5", "se3").picker == "branches" and recipe("franka", "se3").picker == "j7"
    assert recipe("ur5", "line").seed_shell == (0.25, 0.62) and recipe("franka", "orient").seed_shell is None


def test_lerp_slerp_endpoints():
    torch.manual_seed(0)
    z_a, z_b = torch.randn(1, 12), 2 * torch.randn(1, 12)
    path = lerp(z_a, z_b, 24)
    assert path.shape == (24, 12)
    assert torch.equal(path[0], z_a[0]) and torch.allclose(path[-1], z_b[0])
    s = slerp(z_a, z_b, 24)
    r = 0.5 * (z_a.norm() + z_b.norm())
    assert torch.allclose(s.norm(dim=1), r.expand(24), rtol=1e-5)
    assert torch.allclose(s[0] / s[0].norm(), z_a[0] / z_a.norm(), atol=1e-6)
    assert torch.allclose(s[-1] / s[-1].norm(), z_b[0] / z_b.norm(), atol=1e-6)


def _rows(t, valid, pos, orn, step):
    return {"t_bar": t, "valid_pct_loose": valid, "mean_pos_err_mm": pos, "mean_orn_err_deg": orn,
            "p95_ee_step_mm": step}


def test_t_bar_rule():
    rows = [_rows(0, 100, 10.0, 2.0, 10.0), _rows(40, 100, 2.0, 5.0, 10.0), _rows(95, 90, 1.0, 0.1, 10.0)]
    assert t_bar_scores(rows, "plane") == {0: 94.0, 40: 98.0, 95: 88.5}
    assert pick_t_bar(rows, "plane") == 40
    assert t_bar_scores(rows, "plane_orient") == {0: 92.0, 40: 93.0, 95: pytest.approx(88.4)}
    assert t_bar_scores(rows, "orient")[95] == pytest.approx(90 - 0.1 - 1.0)
    tie = [_rows(10, 100, 1.0, np.nan, 0.0), _rows(20, 100, 1.0, np.nan, 0.0)]
    assert pick_t_bar(tie, "line") == 10
