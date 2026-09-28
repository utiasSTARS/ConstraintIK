import numpy as np
import torch

from constraint_ik.constraints import Target
from constraint_ik.encoding import Normalizer, R6d_to_R, R_to_R6d, build_cond, q_to_sincos, sincos_to_q


def _random_R(n, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, 3, 3)))
    return Q * np.sign(np.linalg.det(Q))[:, None, None]


def test_sincos_layout_and_roundtrip():
    q = np.array([[0.1, -2.0, 3.0]])
    sc = q_to_sincos(q)
    assert np.allclose(sc, np.concatenate([np.sin(q), np.cos(q)], axis=-1))
    assert np.allclose(sincos_to_q(sc), q)
    qt = torch.tensor(q)
    assert torch.allclose(sincos_to_q(q_to_sincos(qt)), qt)
    wrapped = sincos_to_q(q_to_sincos(np.array([2 * np.pi - 0.1])))
    assert np.allclose(wrapped, -0.1)


def test_R6d_columns_and_gram_schmidt():
    R = _random_R(10)
    r6 = R_to_R6d(R)
    assert np.array_equal(r6[:, :3], R[:, :, 0]) and np.array_equal(r6[:, 3:], R[:, :, 1])
    assert np.allclose(R6d_to_R(r6), R)
    assert np.allclose(R6d_to_R(r6[0]), R[0])
    assert torch.allclose(R6d_to_R(torch.tensor(r6)), torch.tensor(R))
    noisy = r6 + 0.05 * np.random.default_rng(1).normal(size=r6.shape)
    Rn = R6d_to_R(noisy)
    assert np.allclose(Rn @ np.swapaxes(Rn, 1, 2), np.eye(3), atol=1e-12)


def test_normalizer():
    x = torch.randn(1000, 4) * torch.tensor([1.0, 2.0, 3.0, 0.0]) + 5
    n = Normalizer.fit(x)
    assert n.std[3] == 1e-6
    assert torch.allclose(n.invert(n.apply(x)), x, atol=1e-5)
    assert torch.equal(Normalizer.from_dict(n.to_dict()).mean, n.mean)
    assert Normalizer.from_dict(None) is None


def test_build_cond():
    pn = Normalizer(mean=torch.tensor([0.0, 0.0, 0.2]), std=torch.tensor([0.5, 0.5, 0.25]))
    R = _random_R(1)[0]
    t = Target.from_pose("line_orient", [0.3, 0.1, 0.45], R)
    c = build_cond(t, pn)
    assert c.shape == (15,) and c.dtype == torch.float32
    assert torch.allclose(c[:3], torch.tensor([-0.0, 0.2, 1.0]))
    assert torch.allclose(c[3:9], torch.tensor(R_to_R6d(R), dtype=torch.float32))
    assert c[9:].tolist() == [0, 1, 1, 1, 1, 0]
    t2 = Target.from_pose("plane", [0.3, 0.1, 0.45], R)
    cb = build_cond([t, t2], pn)
    assert cb.shape == (2, 15) and torch.equal(cb[0], c)
    assert cb[1, 3:9].tolist() == [1, 0, 0, 0, 1, 0]
