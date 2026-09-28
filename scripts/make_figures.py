"""Paper figures from the committed results (CPU only).

    python scripts/make_figures.py                  # every figure
    python scripts/make_figures.py id_comparison size_sweep

Each figure writes figures/{name}.pdf and figures/{name}.png. The README figure (readme_id) is PNG only;
`bash reproduce.sh media` copies it to docs/media/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
from scipy.special import expit, logit
from scipy.stats import beta, gaussian_kde

from constraint_ik import get_robot
from constraint_ik.constraints import PAPER_ORDER, codim, n_orient_bits, pos_axes
from constraint_ik.plotting import (BLACK, GREY, GREY_LIGHT, constraint_label, muted_shades, robot_color,
                                    robot_label, robot_shades, setup_style)
from constraint_ik.stats import median_ci, pctl_band

FIG_DIR = "figures"
ROBOTS = ("franka", "ur5")
PAIR_WIDTHS = (10, 25, 50, 75, 100)
FAMILY_MARKERS = {"plane": "o", "line": "s", "posonly": "^", "orient": "v", "plane_orient": "P",
                  "line_orient": "X", "se3": "D"}
ID_T = "results/id/{robot}_vs_t_model_anchored.json"


@lru_cache(maxsize=None)
def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_csv(path: str) -> dict[str, np.ndarray]:
    """Columns of a results CSV (numeric columns as float arrays)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        try:
            cols[key] = np.array(vals, dtype=float)
        except ValueError:
            cols[key] = np.array(vals)
    return cols


def tex(latex: str, plain: str) -> str:
    return latex if plt.rcParams["text.usetex"] else plain


def error_metric(c: str) -> tuple[str, str]:
    """Plotted interpolation error: position (mm) where the family fixes position, else orientation (deg)."""
    return ("mean_pos_err_mm", "[mm]") if pos_axes(c) else ("mean_orn_err_deg", r"[$^\circ$]")


def error_vs_t(robot: str, c: str, width: int, method: str = "linear"):
    """t_bar grid and (median, lo, hi) over targets of the per-path mean error, 95% bootstrap CI."""
    d = load_csv(f"results/interpolation/sweep/{robot}_{c}.csv")
    col = error_metric(c)[0]
    sel = (d["method"] == method) & (d["pair_width"] == width)
    ts = np.unique(d["t_bar"][sel])
    return ts, np.array([median_ci(d[col][sel & (d["t_bar"] == t)]) for t in ts]).T


def departure_t(robot: str, c: str, t_min: int = 15) -> int | None:
    """First noise level t >= t_min at which the median Score-SVD-dagger estimate differs from n - k."""
    d = load(ID_T.format(robot=robot))
    med = d["by_constraint"][c]["model"]["dagger"]["median"]
    return next((t for t, m in zip(d["values"], med) if t >= t_min and m != d["id"][c]), None)


# ---------------------------------------------------------------- shared panels

def draw_id_vs_t(ax, robot: str, c: str, lw: float, ylim: tuple | None = None, svd_band: bool = True) -> None:
    """Fig. 3 top row: analytical ID (dashed), Score-SVD with uniform anchors (dotted) and Score-SVD-dagger
    with model anchors (solid), medians over anchors with 95% bootstrap CIs."""
    d = load(ID_T.format(robot=robot))
    t, color = d["values"], robot_color(robot, light=True)
    svd, dagger = d["by_constraint"][c]["uniform"]["svd"], d["by_constraint"][c]["model"]["dagger"]
    clip = (lambda v: np.clip(v, *ylim)) if ylim else np.asarray
    if svd_band:
        ax.fill_between(t, clip(svd["lo"]), clip(svd["hi"]), color=color, alpha=0.12, lw=0, zorder=1.5)
    # (0.1, 1.9) rather than (0, 2): Agg drops zero-length dashes
    ax.plot(t, svd["median"], color=color, lw=lw, ls=(0, (0.1, 1.9)), dash_capstyle="round", zorder=2.5)
    ax.fill_between(t, clip(dagger["lo"]), clip(dagger["hi"]), color=color, alpha=0.22, lw=0, zorder=2)
    ax.plot(t, dagger["median"], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.axhline(d["id"][c], color=color, lw=0.8 * lw, ls=(0, (3.0, 1.8)), zorder=2)


def draw_error_vs_t(ax, robot: str, c: str, shades: list, band_alpha: float, lw: float, ms: float,
                    mew: float | None = None) -> None:
    """Five pair widths (light to dark = near to far): median over targets with 95% bootstrap CI."""
    for w, color in zip(PAIR_WIDTHS, shades):
        t, (med, lo, hi) = error_vs_t(robot, c, w)
        ax.fill_between(t, lo, hi, color=color, alpha=band_alpha, lw=0)
        ax.plot(t, med, color=color, lw=lw, marker="o", ms=ms, mew=mew)


def grid(xlabel: str, sharex: bool = True):
    """2 x 7 appendix grid (Franka / UR5 rows, one constraint per column) with fonts sized for \\linewidth."""
    plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15})
    fig, axes = plt.subplots(2, 7, figsize=(13.6, 4.0), sharex=sharex, constrained_layout=True)
    for ci, c in enumerate(PAPER_ORDER):
        axes[0, ci].set_title(constraint_label(c, short=True))
    fig.supxlabel(xlabel, fontsize=17)
    return fig, axes


def row_label(ax, robot: str, text: str, color: str | None = None) -> None:
    ax.set_ylabel(robot_label(robot) + "\n" + text, color=color or robot_color(robot))


def unit_label(ax, c: str, left: bool) -> None:
    ax.text(0.05 if left else 0.96, 0.93, error_metric(c)[1], transform=ax.transAxes, fontsize=14, va="top",
            ha="left" if left else "right", color="0.3", bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)


# ---------------------------------------------------------------- Fig. 2

ID_METHODS = (("mle_k10", "Local MLE", "///"), ("pca90", "PCA-90", "..."), ("score_svd", "Score-SVD", "xxx"),
              ("score_svd_dagger", r"Score-SVD$^{\dagger}$", ""))
ID_TICKS = {"plane": r"\textsc{Plane}", "line": r"\textsc{Line}", "posonly": r"\textsc{Position}",
            "orient": r"\textsc{Orientation}", "plane_orient": r"\textsc{Plane + orient.}",
            "line_orient": r"\textsc{Line + orient.}", "se3": r"$\textsc{se}(3)$"}


def lighten(color: str, amount: float) -> tuple:
    return tuple(v + (1 - v) * amount for v in to_rgb(color))


def id_comparison():
    """Fig. 2: per-target median ID of four estimators (95% bootstrap CI) against the analytical ID (black)."""
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                         "ytick.major.width": 0.7, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
                         "hatch.linewidth": 0.55})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.68), sharey=True)
    x, width, eps = np.arange(len(PAPER_ORDER)), 0.19, 0.1
    for ax, robot in zip(axes, ROBOTS):
        res = load(f"results/id/{robot}.json")
        bars, color = res["fig_bars"], robot_color(robot)
        assert tuple(bars["order"]) == PAPER_ORDER
        truth = np.array([res["by_constraint"][c]["id"] for c in PAPER_ORDER])
        for j, (key, _, hatch) in enumerate(ID_METHODS):
            med, lo, hi = (np.array(bars[key][s]) for s in ("median", "lo", "hi"))
            xs, up = x + (j - 1.5) * width, np.clip(hi - med, 0, None)
            ax.bar(xs, np.maximum(med, eps), width, facecolor=color if j == 3 else lighten(color, 0.78),
                   edgecolor=color, hatch=hatch, linewidth=0.7, zorder=2)
            ax.errorbar(xs, med, yerr=[np.clip(med - lo, 0, None), up], fmt="none", ecolor=color,
                        elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
            for xi, m, u, d in zip(xs, med, up, truth):
                if abs(m - d) < 0.25:
                    ax.text(xi, max(m, eps) + u + 0.2, r"$\checkmark$", ha="center", va="bottom", fontsize=8,
                            color=color, zorder=4)
        ax.hlines(truth, x - 2 * width - 0.04, x + 2 * width + 0.04, colors="black", linewidth=2.4, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([tex(ID_TICKS[c], constraint_label(c)) for c in PAPER_ORDER], rotation=30, ha="right",
                           fontsize=9)
        ax.set_ylim(-0.25, 7.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        name = {"franka": "Franka", "ur5": "UR5"}[robot]
        ax.set_title(tex(rf"\textbf{{{name}}}", name), color=color, fontsize=10.5, fontweight="bold", y=1.0,
                     pad=0, va="top")
        ax.tick_params(axis="both", labelsize=8.5)
    axes[0].set_ylabel("Intrinsic dimension", fontsize=10)
    fig.tight_layout()
    return fig


def id_comparison_legend():
    """Stand-alone legend of Fig. 2."""
    plt.rcParams.update({"hatch.linewidth": 0.55})
    handles = [Line2D([0], [0], color="black", lw=2.4, label="Ground truth")]
    handles += [Patch(facecolor="dimgray" if not hatch else "white", edgecolor="dimgray", hatch=hatch, lw=0.7,
                      label=label) for _, label, hatch in ID_METHODS]
    handles.append(Line2D([0], [0], marker=r"$\checkmark$", color="dimgray", ms=9, ls="", label="Matches truth"))
    fig = plt.figure(figsize=(7.0, 0.45))
    fig.legend(handles=handles, loc="center", ncol=6, frameon=False, fontsize=9, handlelength=1.6,
               columnspacing=1.2)
    return fig


# ---------------------------------------------------------------- Fig. 3 and its 14-condition split

MAIN_CONSTRAINTS = ("plane", "posonly", "line_orient", "se3")


def error_svd_tmid_main():
    """Fig. 3: ID estimate vs noise level (top, both robots) and position error of the decoded straight-line
    paths vs the inversion level (Franka, UR5). Dotted verticals: earliest ID departure of the two robots."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "ytick.labelsize": 9,
                         "xtick.direction": "in", "ytick.direction": "in", "ytick.major.size": 2.0,
                         "ytick.minor.size": 1.2, "ytick.minor.width": 0.4, "ytick.major.pad": 2.0})
    fig = plt.figure(figsize=(4.9, 3.25))
    outer = fig.add_gridspec(2, 1, height_ratios=(0.95, 1.55), left=0.115, right=0.985, top=0.925, bottom=0.13,
                             hspace=0.2)
    top = outer[0].subgridspec(1, 4, wspace=0.15)
    bottom = outer[1].subgridspec(2, 4, wspace=0.15, hspace=0.3)
    axes = np.array([[fig.add_subplot(top[0, i]) for i in range(4)]]
                    + [[fig.add_subplot(bottom[r, i]) for i in range(4)] for r in range(2)])
    ylims = ((-0.5, 7), (0, 100), (0, 100))
    for i, c in enumerate(MAIN_CONSTRAINTS):
        for robot in ROBOTS:
            draw_id_vs_t(axes[0, i], robot, c, lw=1.1, ylim=ylims[0])
        for r, robot in ((1, "franka"), (2, "ur5")):
            draw_error_vs_t(axes[r, i], robot, c, robot_shades(robot), band_alpha=0.22, lw=1.1, ms=2.8, mew=0)
        axes[0, i].set_title(constraint_label(c))
        for r, ax in enumerate(axes[:, i]):
            ax.set_xlim(-2, 100)
            ax.set_ylim(*ylims[r])
            ax.set_xticks([])
            ax.set_yticks([0, 2, 4, 6] if r == 0 else [0, 50, 100])
            ax.set_yticks([1, 3, 5, 7] if r == 0 else [25, 75], minor=True)
            if i:
                ax.tick_params(labelleft=False)
        departures = [t for t in (departure_t(robot, c) for robot in ROBOTS) if t is not None]
        if departures:
            y0, y1 = axes[2, i].get_position().y0, axes[0, i].get_position().y1
            fig.add_artist(Line2D([min(departures)] * 2, [y0, y1], color="0.3", lw=0.6, ls=(0, (1, 2.2)),
                                  transform=blended_transform_factory(axes[0, i].transData, fig.transFigure)))
    x_lab = 0.02
    fig.text(x_lab, np.mean([axes[0, 0].get_position().y0, axes[0, 0].get_position().y1]), "ID",
             rotation=90, ha="center", va="center", fontsize=10)
    fig.text(x_lab, np.mean([axes[2, 0].get_position().y0, axes[1, 0].get_position().y1]),
             "position error [mm]", rotation=90, ha="center", va="center", fontsize=10)
    left, right, y = axes[2, 0].get_position().x0, axes[2, -1].get_position().x1, 0.075
    fig.text(left - 0.01, y, tex(r"0\%", "0%"), ha="right", va="center", fontsize=10)
    fig.text(right + 0.005, y, tex(r"100\%", "100%"), ha="left", va="center", fontsize=10)
    fig.add_artist(FancyArrowPatch((left + 0.005, y), (right - 0.005, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=9, lw=1.0, color="black"))
    fig.text((left + right) / 2, 0.02, r"noise level $t$", ha="center", va="center", fontsize=10)
    return fig


def id_vs_t_all():
    """Fig. 3 top row split over all 14 conditions; dotted vertical at the ID departure level."""
    fig, axes = grid(r"noise level $t$")
    for ri, robot in enumerate(ROBOTS):
        D = load(ID_T.format(robot=robot))["in_dim"]
        for ci, c in enumerate(PAPER_ORDER):
            ax = axes[ri, ci]
            draw_id_vs_t(ax, robot, c, lw=1.6, svd_band=False)
            t_dep = departure_t(robot, c)
            if t_dep is not None:
                ax.axvline(t_dep, color="0.45", ls=":", lw=0.9, zorder=0)
            ax.set_xlim(-2, 101)
            ax.set_xticks([0, 50, 100])
            ax.set_ylim(-0.5, D + 0.5)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        row_label(axes[ri, 0], robot, r"$\hat{d}$")
    return fig


def interp_error_vs_t_all():
    """Fig. 3 bottom rows split over all 14 conditions (position error; orientation for Orientation)."""
    fig, axes = grid(r"noise level $t$")
    for ri, robot in enumerate(ROBOTS):
        shades = robot_shades(robot)
        for ci, c in enumerate(PAPER_ORDER):
            ax = axes[ri, ci]
            draw_error_vs_t(ax, robot, c, shades, band_alpha=0.2, lw=1.2, ms=2.6, mew=0)
            t_dep = departure_t(robot, c)
            if t_dep is not None:
                ax.axvline(t_dep, color="0.45", ls=":", lw=0.9, zorder=0)
            ax.set_xlim(-2, 101)
            ax.set_xticks([0, 50, 100])
            ax.set_ylim(bottom=0)
            unit_label(ax, c, left=False)
        row_label(axes[ri, 0], robot, "median error")
    return fig


# ---------------------------------------------------------------- App. D (ID hyperparameters and spectra)

def _id_sweep(which: str):
    """Score-SVD-dagger median over targets (solid, IQR band) and Score-SVD (dotted) against t or K."""
    fig, axes = grid(r"noise level $t$" if which == "t" else r"perturbations $K$")
    for ri, robot in enumerate(ROBOTS):
        d = load(f"results/id/{robot}_vs_{which}.json")
        xs, color = d["values"], robot_color(robot, light=True)
        for ci, c in enumerate(PAPER_ORDER):
            ax, b = axes[ri, ci], d["by_constraint"][c]["uniform"]
            ax.axhline(d["id"][c], color=color, ls="--", lw=1.0, zorder=0)
            ax.fill_between(xs, b["dagger"]["lo"], b["dagger"]["hi"], color=color, alpha=0.16, lw=0, zorder=1)
            ax.plot(xs, b["svd"]["median"], color=color, lw=1.2, ls=":", zorder=2)
            ax.plot(xs, b["dagger"]["median"], color=color, lw=1.7, marker="o", ms=2.4, zorder=3)
            ax.axvline(15 if which == "t" else 50, color="0.45", ls=":", lw=0.8, zorder=0)
            if which == "k":
                ax.set_xscale("log")
                ax.set_xticks([10, 100])
                ax.set_xticklabels([r"$10$", r"$100$"])
            else:
                ax.set_xlim(-2, 97)
                ax.set_xticks([0, 50, 95])
            ax.set_ylim(-0.5, d["in_dim"] + 0.5)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        row_label(axes[ri, 0], robot, r"$\hat{d}$")
    return fig


def id_noise_sensitivity():
    return _id_sweep("t")


def id_k_sensitivity():
    return _id_sweep("k")


def id_score_spectra():
    """Mean score-matrix spectrum (+- 1 std) at t = 15, K = 50; gap markers read off the mean spectrum."""
    fig, axes = grid(r"singular value index $i$", sharex=False)
    for ri, robot in enumerate(ROBOTS):
        res = load(f"results/id/{robot}.json")
        D, color = res["in_dim"], robot_color(robot, light=True)
        for ci, c in enumerate(PAPER_ORDER):
            ax, blk = axes[ri, ci], res["by_constraint"][c]
            mean, std = np.array(blk["spectrum"]["mean"]), np.array(blk["spectrum"]["std"])
            i = np.arange(1, D + 1)
            ax.fill_between(i, mean - std, mean + std, color="0.30", alpha=0.18, lw=0, zorder=1)
            ax.plot(i, mean, color="0.30", lw=1.3, marker="o", ms=2.6, zorder=2)
            gap_svd = int(np.argmax(mean[:-1] - mean[1:])) + 1
            gap_dagger = int(np.argmax(-np.diff(np.append(mean, 0.0)))) + 1
            ax.axvline(D - blk["id"] + 0.5, color=color, ls="--", lw=1.4, zorder=3)
            ax.axvline(gap_svd + 0.5, color=color, ls=":", lw=1.4, zorder=4)
            ax.axvline(gap_dagger + 0.5, color=color, ls="-", lw=1.9, alpha=0.85, zorder=5)
            ax.set_xlim(0.5, D + 0.5)
            ax.set_xticks([1, D // 2, D])
            ax.set_ylim(bottom=0)
            ax.set_yticks([])
        row_label(axes[ri, 0], robot, r"$\sigma_i$")
    return fig


# ---------------------------------------------------------------- App. E (interpolation)

def interp_noise_ablation():
    """Error of the decoded straight-line paths vs inversion level, 5 pair widths; dotted = table t_bar."""
    tbar = load("configs/tbar.json")
    fig, axes = grid(r"noise level $t$")
    for ri, robot in enumerate(ROBOTS):
        shades = muted_shades(robot, len(PAIR_WIDTHS))
        for ci, c in enumerate(PAPER_ORDER):
            ax = axes[ri, ci]
            draw_error_vs_t(ax, robot, c, shades, band_alpha=0.13, lw=1.1, ms=2.0)
            ax.axvline(tbar[robot][c]["linear"], color="0.45", ls=":", lw=0.8, zorder=0)
            ax.set_xlim(-2, 97)
            ax.set_xticks([0, 50, 95])
            ax.set_ylim(bottom=0)
            unit_label(ax, c, left=False)
        row_label(axes[ri, 0], robot, "median error", color=shades[-1])
    return fig


def interp_gap_error():
    """Median path error vs pair width at each method's t_bar: linear (solid) and slerp (dashed)."""
    fig, axes = grid("separation percentile")
    for ri, robot in enumerate(ROBOTS):
        color = robot_color(robot)
        for ci, c in enumerate(PAPER_ORDER):
            ax, d = axes[ri, ci], load_csv(f"results/interpolation/canonical/{robot}_{c}.csv")
            col = error_metric(c)[0]
            for method, ls, marker in (("linear", "-", "o"), ("slerp", "--", None)):
                sel = d["method"] == method
                med, lo, hi = np.array([median_ci(d[col][sel & (d["pair_width"] == w)]) for w in PAIR_WIDTHS]).T
                ax.fill_between(PAIR_WIDTHS, lo, hi, color=color, alpha=0.13, lw=0)
                ax.plot(PAIR_WIDTHS, med, ls=ls, color=color, lw=1.3, marker=marker, ms=2.8)
            ax.set_xlim(2, 108)
            ax.set_xticks([10, 50, 100])
            ax.set_ylim(bottom=0)
            unit_label(ax, c, left=True)
        row_label(axes[ri, 0], robot, "median error")
    return fig


def interp_se3_branch_rho():
    """UR5 SE(3): rho(alpha) median over pairs with the 2.5-97.5 percentile band (log scale)."""
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    d = load_csv("results/flatness/ur5_se3.csv")
    alphas = np.unique(d["alpha"])
    rho = np.full((int(d["pair"].max()) + 1, len(alphas)), np.nan)
    rho[d["pair"].astype(int), np.searchsorted(alphas, d["alpha"])] = d["rho"]
    med, lo, hi = pctl_band(rho, 2.5, 97.5)
    color = robot_color("ur5")
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    ax.fill_between(alphas, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(alphas, med, color=color, lw=1.8)
    ax.axvline(0.5, color="0.5", ls=":", lw=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylabel(tex(r"$\rho(\alpha)=\lVert Dg_{\mathbf c}\,\widehat{\mathbf v}\rVert$",
                      r"$\rho(\alpha)=\Vert Dg_{\mathbf{c}}\,\hat{\mathbf{v}}\Vert$") + "\n[rad / latent unit]")
    ax.set_xlabel(r"interpolation parameter $\alpha$")
    fig.tight_layout()
    return fig


def tau_distributions():
    """tau^q over every (pair, alpha) sample of a condition (upper half, robot colour) against the overlap of a
    uniformly random direction, tau^2 ~ Beta((n - k) / 2, k / 2) (lower half, grey), on a logit axis.
    Markers: median (dot) and 5th percentile (bar); UR5 SE(3) has no tangent space and is left out."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "xtick.labelsize": 8.5, "ytick.labelsize": 9.5,
                         "axes.labelsize": 9.5, "legend.fontsize": 9})
    summary = load("results/flatness/summary.json")
    lo_x, hi_x, half = 0.01, 1 - 1e-6, 0.42
    z = np.linspace(logit(lo_x), logit(hi_x), 600)
    tau = expit(z)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, robot in zip(axes, ROBOTS):
        color, n = robot_color(robot), get_robot(robot).n_joints
        for row, c in enumerate(PAPER_ORDER):
            y, s, k = len(PAPER_ORDER) - 1 - row, summary[robot][c], codim(c)
            if n - k == 0:
                ax.text(0.5, y, r"ID $=0$: no tangent space", ha="center", va="center", color=GREY, fontsize=8.5)
                continue
            t_emp = np.clip(load_csv(f"results/flatness/{robot}_{c}.csv")["tau"], 1e-12, 1 - 1e-12)
            dens_emp = gaussian_kde(logit(t_emp), bw_method=0.25)(z)
            # density of logit(tau) for tau = sqrt(b), b ~ Beta((n - k) / 2, k / 2)
            dens_rand = beta.pdf(tau ** 2, (n - k) / 2, k / 2) * 2 * tau * tau * (1 - tau)
            halves = ((1, dens_emp, color, s["tau_med"], s["tau_p5"], color),
                      (-1, dens_rand, GREY_LIGHT, s["tau_rand_med"], s["tau_rand_p5"], GREY))
            for sign, dens, fill, med, p5, ink in halves:
                ax.fill_between(tau, y, y + sign * half * dens / dens.max(), color=fill, lw=0, alpha=0.9)
                ax.plot([p5, p5], [y, y + sign * 0.3], color=ink, lw=1.4, solid_capstyle="butt")
                ax.plot(min(med, hi_x), y + sign * 0.14, "o", ms=3.6, mfc="white", mec=ink, mew=1.1)
            ax.axhline(y, color="0.75", lw=0.4, zorder=0)
        ax.set_xscale("logit")
        ax.set_xlim(lo_x, hi_x)
        ticks = [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(-0.6, len(PAPER_ORDER) - 0.4)
        ax.set_title(robot_label(robot), color=color)
        ax.set_xlabel(r"tangent overlap $\tau^{\mathbf{q}}$")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(PAPER_ORDER)))
    axes[0].set_yticklabels([constraint_label(c) for c in PAPER_ORDER][::-1])
    handles = [(Patch(color=robot_color("franka")), Patch(color=robot_color("ur5"))), Patch(color=GREY_LIGHT),
               Line2D([0], [0], marker="o", ls="", mfc="white", mec=BLACK, ms=4.5),
               Line2D([0], [0], marker="|", ls="", color=BLACK, ms=8, mew=1.4)]
    fig.legend(handles, [r"decoded path directions", r"random direction (Beta)", "median", "5th percentile"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# ---------------------------------------------------------------- App. C (size sweep)

def size_sweep():
    """Per-family models: mean position / orientation error of the valid samples vs training-set size N."""
    sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000]
    recs = {(r["robot"], r["constraint"], r["N"]): r for r in load("results/size_sweep/per_family.json")["records"]}
    metrics = (("pos_err_mm_mean", "position error (mm)", pos_axes),
               ("orn_err_deg_mean", r"orientation error ($^\circ$)", n_orient_bits))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for ri, robot in enumerate(ROBOTS):
        shade = dict(zip(PAPER_ORDER, muted_shades(robot, len(PAPER_ORDER))))
        for ci, (key, ylabel, has) in enumerate(metrics):
            ax = axes[ri, ci]
            for c in filter(has, PAPER_ORDER):
                ns = [N for N in sizes if (robot, c, N) in recs and recs[robot, c, N].get(key) is not None]
                ax.plot(ns, [recs[robot, c, N][key] for N in ns], color=shade[c], marker=FAMILY_MARKERS[c],
                        ls="--" if c == "se3" else "-", ms=3.4, lw=1.3, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
            ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.5)
            if ci == 0:
                ax.set_ylabel(robot_label(robot) + "\n" + ylabel, color=robot_color(robot))
            else:
                ax.set_ylabel(ylabel)
            if ri == 1:
                ax.set_xlabel(r"training-set size $N$ ($\times 10^3$)")
    handles = [Line2D([0], [0], color="0.3", marker=FAMILY_MARKERS[c], ms=4, lw=1.4,
                      ls=(0, (2.2, 1.4)) if c == "se3" else "-",
                      label=constraint_label(c) + r" ($k{=}%d$)" % codim(c)) for c in PAPER_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False,
               handlelength=3.0, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# ---------------------------------------------------------------- perturbation study

def perturbation():
    """Fraction of anchors whose Score-SVD-dagger estimate equals n - k once the anchor is moved off M_c by a
    constraint violation delta, along a normal (solid) or isotropic (dashed) direction."""
    robots = [r for r in ROBOTS if os.path.exists(f"results/id/{r}_perturbation.json")]
    if not robots:
        return None
    fig, axes = grid(r"constraint violation $\delta$ [mm, $^\circ$]")
    for ri, robot in enumerate(ROBOTS):
        if robot not in robots:
            for ax in axes[ri]:
                ax.set_axis_off()
            continue
        d = load(f"results/id/{robot}_perturbation.json")
        delta = np.array(d["deltas"], dtype=float)
        for ci, c in enumerate(PAPER_ORDER):
            ax, b = axes[ri, ci], d["by_constraint"][c]
            ax.plot(delta, b["isotropic"]["frac_dagger"], color=robot_color(robot, light=True), ls="--", lw=1.4)
            ax.plot(delta, b["normal"]["frac_dagger"], color=robot_color(robot), lw=1.7, marker="o", ms=3)
            ax.set_xscale("symlog", linthresh=1, linscale=0.5)
            ax.set_xticks([0, 1, 10, 50])
            ax.set_xticklabels(["0", "1", "10", "50"])
            ax.minorticks_off()
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
        row_label(axes[ri, 0], robot, tex(r"$\Pr[\hat d = n{-}k]$", r"$\Pr[\hat{d} = n-k]$"))
        axes[ri, 0].legend([Line2D([0], [0], color=robot_color(robot), lw=1.7, marker="o", ms=3),
                            Line2D([0], [0], color=robot_color(robot, light=True), lw=1.4, ls="--")],
                           ["normal", "isotropic"], loc="lower left", fontsize=12, frameon=False,
                           handlelength=1.8, borderaxespad=0.2)
    return fig


# ---------------------------------------------------------------- README figures

def _readme_fonts():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8})


def readme_id():
    """Clean Fig. 2: median ID per constraint of two neighbourhood baselines and the two score estimators."""
    _readme_fonts()
    methods = (("mle_k10", "Local MLE"), ("pca90", "PCA-90"), ("score_svd", "Score-SVD"),
               ("score_svd_dagger", r"Score-SVD$^{\dagger}$ (ours)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True)
    y, h = np.arange(len(PAPER_ORDER))[::-1], 0.19
    for ax, robot in zip(axes, ROBOTS):
        res = load(f"results/id/{robot}.json")
        colors = ("#D5D5D5", "#A3A3A3", robot_color(robot, light=True), robot_color(robot))
        truth = np.array([res["by_constraint"][c]["id"] for c in PAPER_ORDER])
        for j, ((key, _), fc) in enumerate(zip(methods, colors)):
            ax.barh(y + (1.5 - j) * h, np.maximum(res["fig_bars"][key]["median"], 0.15), h * 0.9, color=fc, lw=0)
        ax.vlines(truth, y - 2 * h - 0.05, y + 2 * h + 0.05, colors=BLACK, lw=2.2, zorder=4)
        ax.set_xlim(-0.3, 7.4)
        ax.set_xticks(range(8))
        ax.set_xlabel("intrinsic dimension")
        ax.set_title(robot_label(robot) + f" ($n = {res['n_joints']}$)", color=robot_color(robot), pad=6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([constraint_label(c) for c in PAPER_ORDER])
    handles = [Line2D([0], [0], color=BLACK, lw=2.2), Patch(color="#D5D5D5"), Patch(color="#A3A3A3")]
    handles += [tuple(Patch(color=robot_color(r, light=light)) for r in ROBOTS) for light in (True, False)]
    labels = ["analytical ID", methods[0][1], methods[1][1], methods[2][1], methods[3][1]]
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


FIGURES = {
    "id_comparison": id_comparison,
    "id_comparison_legend": id_comparison_legend,
    "error_svd_tmid_main": error_svd_tmid_main,
    "id_vs_t_all": id_vs_t_all,
    "interp_error_vs_t_all": interp_error_vs_t_all,
    "id_noise_sensitivity": id_noise_sensitivity,
    "id_k_sensitivity": id_k_sensitivity,
    "id_score_spectra": id_score_spectra,
    "interp_noise_ablation": interp_noise_ablation,
    "interp_gap_error": interp_gap_error,
    "interp_se3_branch_rho": interp_se3_branch_rho,
    "tau_distributions": tau_distributions,
    "size_sweep": size_sweep,
    "perturbation": perturbation,
    "readme_id": readme_id,
}


def save(fig, name: str) -> None:
    readme = name.startswith("readme_")
    if readme:
        paths = [os.path.join(FIG_DIR, f"{name}.png")]
    else:
        paths = [os.path.join(FIG_DIR, f"{name}.{ext}") for ext in ("pdf", "png")]
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dpi = (210 if readme else 200) if path.endswith(".png") else None   # README PNGs ~1400 px wide
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print("wrote", ", ".join(paths))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figures", nargs="*", help=f"any of {', '.join(FIGURES)} (default: all)")
    p.add_argument("--no-tex", action="store_true", help="matplotlib text rendering even if LaTeX is installed")
    args = p.parse_args()
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        p.error(f"unknown figures {sorted(unknown)}")
    for name in args.figures or FIGURES:
        with plt.rc_context():
            setup_style(usetex=False if args.no_tex else None)
            fig = FIGURES[name]()
            if fig is None:
                print(f"skipped {name} (missing results)")
                continue
            save(fig, name)


if __name__ == "__main__":
    main()
