import math

import numpy as np
import pytest
import torch

from constraint_ik.robots import Franka, UR5, get_robot


def test_limits_and_shells():
    assert np.array_equal(UR5.lo, np.zeros(6)) and np.array_equal(UR5.hi, np.full(6, 2 * math.pi))
    assert Franka.lo[3] == -3.1416 and Franka.hi[5] == 3.8223 and Franka.hi[3] == 0.0
    assert UR5.shell == (0.20, 0.70, -0.10, 0.80)
    assert Franka.shell == (0.10, 0.85, -0.10, 1.10)
    with pytest.raises(ValueError):
        get_robot("kuka")


def test_shell_contains():
    pts = np.array([[0.5, 0.0, 0.3], [0.1, 0.0, 0.3], [0.5, 0.0, 0.9]])
    assert UR5.shell.contains(pts).tolist() == [True, False, False]


def test_sample_q_within_margin():
    r = get_robot("franka")
    q = r.sample_q(500, np.random.default_rng(0), margin=0.05)
    span = r.hi - r.lo
    assert np.all(q >= r.lo + 0.05 * span) and np.all(q <= r.hi - 0.05 * span)


def test_fk_torch_differentiable_and_batched():
    r = get_robot("ur5")
    q = torch.rand(5, 3, 6, dtype=torch.float64, requires_grad=True)
    pos, R = r.fk_torch(q)
    assert pos.shape == (5, 3, 3) and R.shape == (5, 3, 3, 3)
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(R @ R.transpose(-1, -2), eye.expand_as(R), atol=1e-12)
    (pos.sum() + R.sum()).backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().max() > 1e-3


@pytest.mark.parametrize("name", ["ur5", "franka"])
def test_fk_torch_matches_ikfast(name):
    pytest.importorskip(f"ikfast_{'ur5' if name == 'ur5' else 'franka_panda'}")
    r = get_robot(name)
    q = r.sample_q(200, np.random.default_rng(1), margin=0.0)
    pos, R = r.fk(q)
    pos_t, R_t = r.fk_torch(torch.as_tensor(q))
    assert np.abs(pos_t.numpy() - pos).max() < 1e-12
    assert np.abs(R_t.numpy() - R).max() < 1e-12
    p1, R1 = r.fk(q[0])
    assert p1.shape == (3,) and R1.shape == (3, 3) and np.array_equal(p1, pos[0])


@pytest.mark.parametrize("name", ["ur5", "franka"])
def test_ik_all_reproduces_pose(name):
    pytest.importorskip(f"ikfast_{'ur5' if name == 'ur5' else 'franka_panda'}")
    r = get_robot(name)
    q = r.sample_q(5, np.random.default_rng(2))
    pos, R = r.fk(q)
    for p, Rm in zip(pos, R):
        sols = r.ik_all(p, Rm)
        assert sols.ndim == 2 and sols.shape[1] == r.n_joints and len(sols) > 0
        assert np.all(r.in_limits(sols))
        ps, Rs = r.fk(sols)
        assert np.abs(ps - p).max() < 1e-6 and np.abs(Rs - Rm).max() < 1e-6
