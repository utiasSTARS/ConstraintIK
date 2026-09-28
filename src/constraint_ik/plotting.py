"""Paper palette and matplotlib style (serif, small-caps labels; LaTeX text when `latex` is on PATH)."""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb

FRANKA = "#2A4F7D"
FRANKA_LIGHT = "#527299"
UR5 = "#6E1622"
UR5_LIGHT = "#CA7981"
GREY = "#6E6E6E"
GREY_LIGHT = "#BDBDBD"
BLACK = "#222222"

_COLORS = {"franka": (FRANKA, FRANKA_LIGHT), "ur5": (UR5, UR5_LIGHT)}
# Pale end of the per-robot ramp (e.g. pair widths 10 -> 100 go pale -> base colour).
_RAMP_PALE = {"franka": "#C9DDEE", "ur5": "#E89AA1"}

_TEX_LABELS = {
    "plane": r"\textsc{Plane}",
    "line": r"\textsc{Line}",
    "posonly": r"\textsc{Position}",
    "orient": r"\textsc{Orient.}",
    "plane_orient": r"\textsc{Plane\,+\,Orient.}",
    "line_orient": r"\textsc{Line\,+\,Orient.}",
    "se3": r"$\mathrm{SE}(3)$",
}
_TEX_SHORT = {**_TEX_LABELS, "plane_orient": r"\textsc{Plane+Or.}",
              "line_orient": r"\textsc{Line+Or.}"}
_PLAIN_LABELS = {
    "plane": "Plane", "line": "Line", "posonly": "Position", "orient": "Orient.",
    "plane_orient": "Plane + Orient.", "line_orient": "Line + Orient.",
    "se3": r"$\mathrm{SE}(3)$",
}
_PLAIN_SHORT = {**_PLAIN_LABELS, "plane_orient": "Plane+Or.", "line_orient": "Line+Or."}


def robot_color(robot: str, light: bool = False) -> str:
    return _COLORS[robot][int(light)]


def robot_shades(robot: str, n: int = 5) -> list[str]:
    """n colours from pale to the robot's base colour."""
    pale, base = np.array(to_rgb(_RAMP_PALE[robot])), np.array(to_rgb(robot_color(robot)))
    return [to_hex(pale * (1 - s) + base * s) for s in np.linspace(0.0, 1.0, n)]


def muted_shades(robot: str, n: int) -> list[str]:
    """n colours from a muted tint to the robot's base colour (appendix grids, size sweep)."""
    base = np.array(to_rgb(robot_color(robot)))
    tint = 0.45 + 0.55 * base
    return [to_hex(tint * (1 - s) + base * s) for s in np.linspace(0.3, 1.0, n)]


def setup_style(compact: bool = False, usetex: bool | None = None) -> bool:
    """Apply the paper rcParams; returns whether LaTeX text rendering is on."""
    if usetex is None:
        usetex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 7 if compact else 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in" if compact else "out",
        "ytick.direction": "in" if compact else "out",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return usetex


def constraint_label(c: str, short: bool = False) -> str:
    """Figure label for a constraint family, small caps under LaTeX."""
    tex = plt.rcParams["text.usetex"]
    table = (_TEX_SHORT if short else _TEX_LABELS) if tex else (_PLAIN_SHORT if short else _PLAIN_LABELS)
    return table[c]


def robot_label(robot: str) -> str:
    name = {"franka": "Franka", "ur5": "UR5"}[robot]
    return rf"\textsc{{{name}}}" if plt.rcParams["text.usetex"] else name
