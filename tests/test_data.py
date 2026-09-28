"""Training-data helpers: exact greedy dedup, stratified batches, bundle layout."""
import numpy as np
import torch

from constraint_ik.constraints import CONSTRAINTS
from constraint_ik.data import StratifiedSampler, build_bundle, dedup


def _greedy(S, tol, cap=None):
    keep = [0]
    for i in range(1, len(S)):
        if all(np.linalg.norm(S[i] - S[j]) > tol for j in keep):
            keep.append(i)
        if cap is not None and len(keep) >= cap:
            break
    return S[keep]


def test_dedup_matches_greedy_at_the_tolerance():
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 0.3, (400, 7))
    u = rng.normal(size=(50, 7))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    base = S[rng.integers(0, 400, 50)]
    S = np.vstack([S, base + 0.05 * u] + [base + np.nextafter(0.05, s) * u for s in (0, 1)])
    S = S[rng.permutation(len(S))]
    for cap in (None, 2, 100):
        assert np.array_equal(dedup(S, 0.05, cap), _greedy(S, 0.05, cap))
    assert dedup(np.empty((0, 6)), 0.05).shape == (0, 6)


def test_stratified_sampler():
    pools = [torch.arange(0, 10), torch.arange(10, 1000), torch.arange(1000, 1003)]
    torch.manual_seed(0)
    idx = next(iter(StratifiedSampler(pools, 512)))
    assert [int(((idx >= p[0]) & (idx <= p[-1])).sum()) for p in pools] == [171, 171, 170]
    torch.manual_seed(0)
    assert torch.equal(idx, next(iter(StratifiedSampler(pools, 512))))


def test_bundle_layout():
    shards = [{"q_ends": torch.rand(5 + i, 6), "cond_pos": torch.rand(5 + i, 3), "cond_R6d": torch.rand(5 + i, 6),
               "cond_mask": torch.ones(5 + i, 6), "constraint_type": c} for i, c in enumerate(reversed(CONSTRAINTS))]
    b = build_bundle(shards, "ur5")
    assert b["type_names"] == list(CONSTRAINTS)
    for i, c in enumerate(CONSTRAINTS):
        idx = b["type_indices"][c]
        assert torch.equal(b["type_idx"][idx], torch.full((len(idx),), i))
        assert torch.equal(b["q_ends"][idx], next(s for s in shards if s["constraint_type"] == c)["q_ends"])
    assert torch.equal(b["pos_normalizer"]["mean"], b["cond_pos"].mean(0))
