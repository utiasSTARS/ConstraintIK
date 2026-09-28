"""The viewer URDFs reproduce the IKFast end-effector pose used by the model."""
import numpy as np
import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("ikfast_franka_panda")
pytest.importorskip("ikfast_ur5")

from constraint_ik import get_robot  # noqa: E402
from constraint_ik.viz import SPECS, load_urdf  # noqa: E402


@pytest.mark.parametrize("name", ["ur5", "franka"])
def test_urdf_ee_matches_ikfast(name):
    robot, urdf = get_robot(name), load_urdf(name)
    q = robot.sample_q(100, np.random.default_rng(0), margin=0.0)
    pos, R = robot.fk(q)
    for qi, p, r in zip(q, pos, R):
        urdf.update_cfg(qi)
        T = urdf.get_transform(SPECS[name].ee_link, urdf.base_link)
        assert np.abs(T[:3, 3] - p).max() < 1e-6
        assert np.abs(T[:3, :3] - r).max() < 1e-6
