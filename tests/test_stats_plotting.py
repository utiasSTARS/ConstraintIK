import numpy as np

from constraint_ik import plotting
from constraint_ik.stats import bootstrap_ci, median_ci, pctl_band


def test_median_ci():
    v = np.random.default_rng(0).lognormal(size=150)
    med, lo, hi = median_ci(v)
    assert med == np.median(v) and lo < med < hi
    assert median_ci(v) == (med, lo, hi)
    assert median_ci(np.append(v, [np.nan, np.inf])) == (med, lo, hi)
    assert np.isnan(median_ci([np.nan])).all()
    assert bootstrap_ci(v, np.mean)[0] == np.mean(v)


def test_pctl_band():
    M = np.arange(12, dtype=float).reshape(4, 3)
    med, lo, hi = pctl_band(M, 0, 100)
    assert med.tolist() == [4.5, 5.5, 6.5] and lo.tolist() == [0, 1, 2] and hi.tolist() == [9, 10, 11]


def test_palette_and_style():
    assert plotting.robot_color("franka") == "#2A4F7D"
    assert plotting.robot_color("ur5", light=True) == "#CA7981"
    shades = plotting.robot_shades("ur5", 5)
    assert len(shades) == 5 and shades[-1].lower() == "#6e1622"
    assert plotting.setup_style(usetex=False) is False
    assert plotting.constraint_label("line_orient") == "Line + Orient."
    plotting.setup_style(usetex=True)
    assert plotting.constraint_label("plane") == r"\textsc{Plane}"
    assert plotting.robot_label("ur5") == r"\textsc{UR5}"
    plotting.setup_style(usetex=False)
