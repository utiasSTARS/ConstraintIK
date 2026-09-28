import pytest
import torch

from constraint_ik.diffusion import (
    ddim_decode_partial, ddim_invert_partial, ddim_run, ddim_sample, ddim_step, ddim_timesteps,
    make_schedule, q_sample,
)


def test_schedule_values():
    s = make_schedule("sigmoid", 100)
    assert s.dtype == torch.float32 and s.shape == (100,)
    for t, v in {0: 0.9998508, 15: 0.99615, 40: 0.94998, 99: 0.36296}.items():
        assert abs(float(s[t]) - v) < 5e-6
    assert abs(float(make_schedule("linear")[99]) - 0.3636) < 1e-4
    c = make_schedule("cosine")
    assert abs(float(c[0]) - 0.99937) < 1e-5 and float(c[15]) == pytest.approx(0.9332, abs=1e-4)
    for name in ("cosine", "linear", "sigmoid"):
        assert torch.all(make_schedule(name)[1:] < make_schedule(name)[:-1])
    with pytest.raises(ValueError):
        make_schedule("quadratic")


def test_grids():
    assert ddim_timesteps(99, 0, 100) == list(range(99, -1, -1))
    assert ddim_timesteps(99, 0, 25) == list(range(99, -1, -4)) + [0]
    assert ddim_timesteps(0, 99, 25) == list(range(0, 100, 4)) + [99]
    assert ddim_timesteps(0, 40, 10) == [0, 10, 20, 30, 40]
    assert ddim_timesteps(0, 95, 10) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    assert ddim_timesteps(95, 0, 10) == [95, 85, 75, 65, 55, 45, 35, 25, 15, 5, 0]
    assert ddim_timesteps(0, 0, 100) == [0]
    assert ddim_timesteps(0, 0, 1000) == [0]


class _Const(torch.nn.Module):
    """eps = c * x (analytic stand-in for the denoiser)."""

    def __init__(self, c=0.1):
        super().__init__()
        self.c = c

    def forward(self, x, t, cond):
        return self.c * x + 0 * cond.sum(-1, keepdim=True)


def test_step_and_loops():
    a = make_schedule("sigmoid")
    model = _Const()
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    cond = torch.zeros(3, 15)
    y = ddim_step(model, x, 10, 20, cond, a)
    eps = 0.1 * x
    x0 = (x - (1 - a[10]).sqrt() * eps) / a[10].sqrt()
    assert torch.equal(y, a[20].sqrt() * x0 + (1 - a[20]).sqrt() * eps)
    assert torch.equal(ddim_invert_partial(model, x, cond, a, 0), x)
    assert torch.equal(ddim_decode_partial(model, x, cond, a, 0), x)
    assert torch.equal(ddim_sample(model, x, cond, a, 25),
                       ddim_run(model, x, cond, a, ddim_timesteps(99, 0, 25)))
    big = ddim_step(_Const(0.0), 100 * x, 99, 0, cond, a)
    assert big.abs().max() <= 5.0 * a[0].sqrt() + 1e-6          # x0 clamp at +-5


def test_grad_variant_matches():
    a = make_schedule("sigmoid")
    lin = torch.nn.Linear(4, 4).double()
    model = lambda x, t, c: lin(x)  # noqa: E731
    z = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    cond = torch.zeros(2, 15, dtype=torch.float64)
    out = ddim_decode_partial(model, z, cond, a.double(), 40, grad=True)
    assert out.requires_grad
    assert torch.equal(out.detach(), ddim_decode_partial(model, z, cond, a.double(), 40))
    assert not ddim_decode_partial(model, z, cond, a.double(), 40).requires_grad


def test_q_sample():
    a = make_schedule("sigmoid")
    x0, noise = torch.randn(4, 6), torch.randn(4, 6)
    t = torch.tensor([0, 15, 40, 99])
    xt = q_sample(x0, t, noise, a)
    assert torch.allclose(xt[2], a[40].sqrt() * x0[2] + (1 - a[40]).sqrt() * noise[2])
