import numpy as np
import pytest
import torch

from constraint_ik.checkpoint import load_checkpoint, save_checkpoint
from constraint_ik.constraints import CONSTRAINTS, Target
from constraint_ik.diffusion import make_schedule
from constraint_ik.encoding import Normalizer, q_to_sincos
from constraint_ik.model import EMA, CondEndpointDDPM
from constraint_ik.pipeline import DiffusionIK


def test_paper_model_size_and_names():
    for in_dim, n in ((12, 25_561_100), (14, 25_563_150)):
        m = CondEndpointDDPM(in_dim)
        assert sum(p.numel() for p in m.parameters()) == n
    keys = CondEndpointDDPM(12).state_dict().keys()
    assert {"time_mlp.2.weight", "cond_mlp.4.bias", "input_proj.weight", "blocks.9.film.weight",
            "blocks.0.mlp.3.bias", "out_proj.2.weight"} <= set(keys)
    cfg = dict(in_dim=12, cond_dim=15, base_channels=64, dim_mult=(1, 2, 4), emb_dim=32,
               kernel=3, dropout=0.0, depth=2)
    assert CondEndpointDDPM.from_config(cfg).config["base_channels"] == 64


def test_ema_warmup_then_decay():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, decay=0.5, warmup=1)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m, 0)
    assert torch.equal(ema.model.weight, m.weight)
    with torch.no_grad():
        m.weight.fill_(3.0)
    ema.update(m, 1)
    assert torch.allclose(ema.model.weight, torch.full((2, 2), 2.0))


@pytest.fixture
def tiny_path(tmp_path) -> str:
    torch.manual_seed(0)
    model = CondEndpointDDPM(12, base_channels=32, emb_dim=16, depth=2)
    q = torch.rand(256, 6) * 6.28
    xn = Normalizer.fit(q_to_sincos(q))
    pn = Normalizer(mean=torch.tensor([0.0, 0.0, 0.24]), std=torch.tensor([0.19, 0.26, 0.28]))
    path = tmp_path / "tiny.pt"
    save_checkpoint(str(path), model=model, ema_state=model.state_dict(), x_normalizer=xn,
                    pos_normalizer=pn, alphas=make_schedule("sigmoid"), robot="ur5", step=7)
    return str(path)


@pytest.fixture
def tiny(tiny_path) -> DiffusionIK:
    return load_checkpoint(tiny_path)


def test_checkpoint_roundtrip(tiny_path, tiny):
    raw = torch.load(tiny_path, weights_only=False)
    assert {"schema_version", "model_class", "encoding", "model_cfg", "model_state", "ema_state",
            "T_diff", "alphas_cumprod", "normalizer", "pos_normalizer", "R6d_normalizer",
            "type_names", "step", "args"} <= set(raw)
    assert raw["schema_version"] == 1 and raw["R6d_normalizer"] is None
    assert tiny.robot == "ur5" and tiny.type_names == list(CONSTRAINTS)
    assert tiny.dim == 12 and tiny.n_joints == 6 and tiny.T == 100 and tiny.meta["step"] == 7
    assert not any(p.requires_grad for p in tiny.model.parameters())
    # Older layout: unused cfg keys, robot only inferable from the input dimension.
    raw["model_cfg"].update(dim_mult=(1, 2, 4), kernel=3)
    del raw["robot"]
    torch.save(raw, tiny_path)
    assert load_checkpoint(tiny_path).robot == "ur5"


def test_pipeline_shapes_and_roundtrips(tiny):
    t = Target.from_pose("plane", [0.3, 0.1, 0.2], np.eye(3))
    q = np.random.default_rng(0).uniform(-3, 3, size=(5, 6))
    x0 = tiny.encode(q)
    assert x0.shape == (5, 12)
    assert np.allclose(tiny.decode(x0).numpy(), q, atol=1e-5)
    cond = tiny.cond(t)
    assert cond.shape == (15,) and tiny.cond([t, t]).shape == (2, 15)
    assert torch.equal(tiny.invert(q, 0, cond), x0)
    z = tiny.invert(q, 40, cond)
    assert z.shape == (5, 12)
    q_dec = tiny.decode_from(z, 40, t)
    assert q_dec.shape == (5, 6)
    assert torch.equal(q_dec, tiny.decode_from(z, 40, cond, grad=True).detach())
    assert torch.equal(tiny.score(z, 40, cond), -tiny.eps(z, 40, cond) / (1 - tiny.alphas[40]).sqrt().item())
    s1, s2 = tiny.sample(t, 8, n_steps=10, seed=3), tiny.sample(t, 8, n_steps=10, seed=3)
    assert s1.shape == (8, 6) and np.array_equal(s1, s2) and np.all(np.abs(s1) <= np.pi)
    g = torch.Generator().manual_seed(3)
    assert np.array_equal(tiny.sample(t, 8, n_steps=10, generator=g), s1)


def test_decoder_jacobian(tiny):
    t = Target.from_pose("se3", [0.3, 0.1, 0.2], np.eye(3))
    z = tiny.invert(np.zeros(6), 60, t)[0]
    J = torch.autograd.functional.jacobian(
        lambda zs: tiny.decode_from(zs.unsqueeze(0), 60, t, n_steps=10, grad=True)[0], z)
    assert J.shape == (6, 12) and torch.isfinite(J).all()
