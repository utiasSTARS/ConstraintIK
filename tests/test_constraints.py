import numpy as np
import pytest
import torch

from constraint_ik.constraints import (
    CONSTRAINTS, DISPLAY, LOOSE, MASKS, PAPER_ORDER, STRICT, ConstraintMap, Target, codim, errors,
    intrinsic_dim, is_valid, pinned_axis,
)
from constraint_ik.robots import get_robot

UR5_DIMS = {"se3": 0, "posonly": 3, "plane": 5, "line": 4, "orient": 3,
            "plane_orient": 2, "line_orient": 2}


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_tables():
    assert CONSTRAINTS == ("se3", "posonly", "plane", "line", "orient", "plane_orient", "line_orient")
    assert sorted(PAPER_ORDER) == sorted(CONSTRAINTS) and PAPER_ORDER[0] == "plane"
    assert set(DISPLAY) == set(MASKS) == set(CONSTRAINTS)
    assert MASKS["line_orient"] == (0, 1, 1, 1, 1, 0) and pinned_axis("line_orient") == 2
    assert pinned_axis("se3") is None and pinned_axis("plane") is None
    for c in CONSTRAINTS:
        assert intrinsic_dim(c, 6) == UR5_DIMS[c]
        assert intrinsic_dim(c, 7) == UR5_DIMS[c] + 1
        assert codim(c) + intrinsic_dim(c, 6) == 6


def test_target_canonical_fill():
    R = _rz(0.3)
    t = Target.from_pose("line", [0.1, 0.2, 0.3], R)
    assert t.pos.tolist() == [0.0, 0.2, 0.3] and np.array_equal(t.R, np.eye(3))
    t = Target.from_pose("line_orient", [0.1, 0.2, 0.3], R)
    assert t.pos.tolist() == [0.0, 0.2, 0.3] and np.array_equal(t.R, R)
    t = Target.from_pose("orient", [0.1, 0.2, 0.3], R)
    assert t.pos.tolist() == [0.0, 0.0, 0.0]
    e = {"constraint_type": "plane", "cond_pos": np.float32([0, 0, 0.25]),
         "cond_R6d": [1, 0, 0, 0, 1, 0], "mask": [0, 0, 1, 0, 0, 0]}
    t = Target.from_entry(e)
    assert t.mask.tolist() == [0, 0, 1, 0, 0, 0] and np.array_equal(t.R, np.eye(3))


def test_tolerance_sets():
    assert LOOSE.per_axis and not STRICT.per_axis
    assert LOOSE.pos_mm["posonly"] == 100.0 and LOOSE.pos_mm["line"] == 80.0
    assert np.radians(LOOSE.orn_deg["se3"]) == 0.3
    assert "orient" not in LOOSE.pos_mm and "plane" not in LOOSE.orn_deg
    assert set(STRICT.pos_mm.values()) == {10.0} and set(STRICT.orn_deg.values()) == {5.0}


@pytest.fixture(scope="module")
def ur5():
    pytest.importorskip("ikfast_ur5")
    return get_robot("ur5")


def _pose(robot, seed):
    q = robot.sample_q(1, np.random.default_rng(seed))[0]
    pos, R = robot.fk(q)
    return q, pos, R


def test_errors_zero_on_manifold_and_nan_pattern(ur5):
    q, pos, R = _pose(ur5, 0)
    for c in CONSTRAINTS:
        t = Target.from_pose(c, pos, R)
        p, o = errors(ur5, np.stack([q, q]), t)
        assert p.shape == (2,)
        assert np.isnan(p).all() == (c == "orient")
        assert np.isnan(o).all() == (c in ("posonly", "plane", "line"))
        assert np.nan_to_num(p).max() < 1e-9 and np.nan_to_num(o).max() < 1e-5
        assert is_valid(ur5, q, t, STRICT) and is_valid(ur5, q, t, LOOSE)


def test_loose_is_per_axis_strict_is_l2(ur5):
    q, pos, R = _pose(ur5, 1)
    t = Target.from_pose("line", pos + np.array([0.0, 0.045, 0.045]), R)
    p, _ = errors(ur5, q, t)
    assert abs(p - 1000 * np.hypot(0.045, 0.045)) < 1e-9
    assert is_valid(ur5, q, t, LOOSE) and not is_valid(ur5, q, t, STRICT)
    t = Target.from_pose("orient", pos, R @ _rz(np.radians(8.0)))
    assert abs(errors(ur5, q, t)[1] - 8.0) < 1e-9
    assert is_valid(ur5, q, t, LOOSE) and not is_valid(ur5, q, t, STRICT)


def test_line_orient_pins_body_z_direction(ur5):
    q, pos, R = _pose(ur5, 3)
    t = Target.from_pose("line_orient", pos, R)
    q_rot = ur5.ik_all(pos, R @ _rz(0.4))          # rotation about the body z-axis is free
    q_shift = ur5.ik_all(pos + [0.03, 0, 0], R)     # translation along world x is free
    cm = ConstraintMap(ur5, t)
    for qs in (q_rot, q_shift):
        assert len(qs) and cm.h(torch.as_tensor(qs)).abs().max() < 1e-9
        assert errors(ur5, qs, t)[1].max() < 1e-5


@pytest.mark.parametrize("c", CONSTRAINTS)
def test_constraint_map_jacobian(ur5, c):
    q, pos, R = _pose(ur5, 3)
    cm = ConstraintMap(ur5, Target.from_pose(c, pos, R))
    qt = torch.as_tensor(q).unsqueeze(0)
    assert cm.h(qt).abs().max() < 1e-9
    J = cm.jac(qt)
    assert J.shape == (1, codim(c), 6) and torch.linalg.matrix_rank(J[0]) == codim(c)
    fd = torch.stack([(cm.h(qt + 1e-6 * e) - cm.h(qt - 1e-6 * e))[0] / 2e-6
                      for e in torch.eye(6, dtype=torch.float64)], dim=1)
    assert (fd - J[0]).abs().max() < 1e-6
    if codim(c) < 6:
        T = cm.tangent_basis(qt)
        assert T.shape == (1, 6, 6 - codim(c)) and (J @ T).abs().max() < 1e-10
        P = cm.tangent_projector(qt)
        assert torch.allclose(P @ P, P, atol=1e-12)
