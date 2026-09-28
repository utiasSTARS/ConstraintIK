"""LaTeX bodies of the paper tables from the committed results (CPU only).

    python scripts/make_tables.py            # every table
    python scripts/make_tables.py interp_full

Each table writes tables/{name}.tex and, where the paper values are in results/paper_values/{name}.json,
tables/{name}_vs_paper.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from constraint_ik.constraints import PAPER_ORDER, n_orient_bits, pos_axes
from constraint_ik.stats import median_ci

TABLE_DIR = "tables"
PAPER_DIR = "results/paper_values"
TEX_NAME = {
    "plane": r"\textsc{Plane}",
    "line": r"\textsc{Line}",
    "posonly": r"\textsc{Position}",
    "orient": r"\textsc{Orientation}",
    "plane_orient": r"\textsc{Plane} $+$ \textsc{Orientation}",
    "line_orient": r"\textsc{Line} $+$ \textsc{Orientation}",
    "se3": r"$\LieGroupSE{3}$",
}
TEX_ROBOT = {"franka": r"\textsc{Franka}", "ur5": r"\textsc{UR5}"}


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(name: str, text: str) -> str:
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------- Table tab:interp-full

INTERP_METRICS = (("pos", "mean_pos_err_mm"), ("orn", "mean_orn_err_deg"))
INTERP_METHODS = (("lin", "linear"), ("sph", "slerp"))


def interp_full_stats(results_dir: str = "results/interpolation/canonical") -> dict:
    """{robot: {c: {'pos_lin': (med, lo, hi) or None, ...}}}: median over the 150 paths at each method's
    t_bar of the per-path mean error, with its 95% bootstrap CI (4000 resamples, seed 0)."""
    out: dict = {}
    for robot in ("franka", "ur5"):
        for c in PAPER_ORDER:
            rows = read_csv(os.path.join(results_dir, f"{robot}_{c}.csv"))
            has = {"pos": bool(pos_axes(c)), "orn": bool(n_orient_bits(c))}
            cells = {}
            for metric, col in INTERP_METRICS:
                for short, method in INTERP_METHODS:
                    vals = np.array([float(r[col]) for r in rows if r["method"] == method])
                    cells[f"{metric}_{short}"] = median_ci(vals, n_boot=4000, seed=0) if has[metric] else None
            out.setdefault(robot, {})[c] = cells
    return out


def _cell(v, bold: bool) -> str:
    if v is None:
        return "--"
    med = rf"\mathbf{{{v[0]:.2f}}}" if bold else f"{v[0]:.2f}"
    return f"${med}_{{[{v[1]:.2f},{v[2]:.2f}]}}$"


def _bold(cells: dict, key: str) -> bool:
    metric = key.split("_")[0]
    lin, sph = cells[f"{metric}_lin"], cells[f"{metric}_sph"]
    return lin is not None and cells[key][0] == min(lin[0], sph[0])


def interp_full(results_dir: str = "results/interpolation/canonical") -> None:
    """Appendix table tab:interp-full: median (95% CI) FK error of the decoded lin / sph paths."""
    stats = interp_full_stats(results_dir)
    keys = ("pos_lin", "pos_sph", "orn_lin", "orn_sph")
    lines = []
    for robot in ("franka", "ur5"):
        if lines:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{7}}{{*}}{{\rotatebox[origin=c]{{90}}{{{TEX_ROBOT[robot]}}}}}")
        for c in PAPER_ORDER:
            cells = stats[robot][c]
            row = [_cell(cells[k], cells[k] is not None and _bold(cells, k)) for k in keys]
            lines.append(f" & {TEX_NAME[c]:<40s}& " + " & ".join(f"{x:<29s}" for x in row) + r" \\")
    print("wrote", write("interp_full.tex", "\n".join(lines) + "\n"))

    paper_path = os.path.join(PAPER_DIR, "interp_full.json")
    if not os.path.exists(paper_path):
        return
    paper = json.load(open(paper_path))["cells"]
    md = ["| Robot | Constraint | Cell | Paper | Release | Match |", "|---|---|---|---|---|---|"]
    n = n_ok = 0
    for robot in ("franka", "ur5"):
        for c in PAPER_ORDER:
            cells = stats[robot][c]
            for k in keys:
                p = paper[robot][c][k]
                mine = cells[k]
                if p is None and mine is None:
                    continue
                ps = "--" if p is None else f"{p['median']} [{p['lo']}, {p['hi']}]" + (" (bold)" if p["bold"] else "")
                ms = "--" if mine is None else (f"{mine[0]:.2f} [{mine[1]:.2f}, {mine[2]:.2f}]"
                                                + (" (bold)" if _bold(cells, k) else ""))
                n += 1
                n_ok += ps == ms
                md.append(f"| {robot} | {c} | {k} | {ps} | {ms} | {'yes' if ps == ms else '**no**'} |")
    md.insert(0, f"Table tab:interp-full, release vs paper: {n_ok}/{n} cells identical (median, CI, bold).\n")
    print("wrote", write("interp_full_vs_paper.md", "\n".join(md) + "\n"))


# ---------------------------------------------------------------- Table tab:id-full

ID_COLUMNS = ("mle_k2", "mle_k10", "pca90", "pca99", "twonn", "score_svd", "score_svd_dagger")
ID_ONE_DECIMAL = ("mle_k2", "mle_k10", "twonn")


def _id_cell(v: float, d: int, m: str) -> str:
    """Median over targets; bold when it rounds to the analytical ID d."""
    s = f"{v:.1f}" if m in ID_ONE_DECIMAL else f"{v:g}"
    return rf"$\mathbf{{{s}}}$" if abs(v - d) < 0.5 else f"${s}$"


def id_full(results_dir: str = "results/id") -> None:
    """Appendix table tab:id-full: ID column and the median over targets of every estimator."""
    lines, cells = [], {}
    for robot in ("franka", "ur5"):
        res = json.load(open(os.path.join(results_dir, f"{robot}.json")))["by_constraint"]
        if lines:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{7}}{{*}}{{\rotatebox[origin=c]{{90}}{{{TEX_ROBOT[robot]}}}}}")
        for c in PAPER_ORDER:
            d = res[c]["id"]
            row = [str(d)] + [_id_cell(res[c]["summary"][m]["median"], d, m) for m in ID_COLUMNS]
            cells.setdefault(robot, {})[c] = row
            lines.append(f" & {TEX_NAME[c]:<40s}& " + " & ".join(f"{x:<14s}" for x in row) + r" \\")
    print("wrote", write("id_full.tex", "\n".join(lines) + "\n"))

    paper_path = os.path.join(PAPER_DIR, "id_full.json")
    if not os.path.exists(paper_path):
        return
    paper = json.load(open(paper_path))["cells"]
    md = ["| Robot | Constraint | Column | Paper | Release | Match |", "|---|---|---|---|---|---|"]
    n = n_ok = 0
    for robot in ("franka", "ur5"):
        for c in PAPER_ORDER:
            for col, p, mine in zip(("ID",) + ID_COLUMNS, paper[robot][c], cells[robot][c]):
                n += 1
                n_ok += p == mine
                md.append(f"| {robot} | {c} | {col} | `{p}` | `{mine}` | {'yes' if p == mine else '**no**'} |")
    md.insert(0, f"Table tab:id-full, release vs paper: {n_ok}/{n} cells identical (value and bold).\n")
    print("wrote", write("id_full_vs_paper.md", "\n".join(md) + "\n"))


# ---------------------------------------------------------------- Table tab:size-sweep-compare

SIZE_SWEEP_COLUMNS = ("per_family_pos", "per_family_orn", "joint_pos", "joint_orn")


def size_sweep_compare(results_dir: str = "results/size_sweep", N: int = 40000) -> None:
    """Appendix table tab:size-sweep-compare: mean position (mm) and orientation (deg) error of the valid
    samples, per-family models at N = 40k against the jointly-trained model."""
    recs = {}
    for model in ("per_family", "joint"):
        for r in json.load(open(os.path.join(results_dir, f"{model}.json")))["records"]:
            if r["N"] in (None, N):
                recs[model, r["robot"], r["constraint"]] = r
    lines, cells = [], {}
    for robot in ("franka", "ur5"):
        if lines:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{7}}{{*}}{{\rotatebox[origin=c]{{90}}{{{TEX_ROBOT[robot]}}}}}")
        for c in PAPER_ORDER:
            has = {"pos": bool(pos_axes(c)), "orn": bool(n_orient_bits(c))}
            key = {"pos": "pos_err_mm_mean", "orn": "orn_err_deg_mean"}
            row = {f"{m}_{e}": f"{recs[m, robot, c][key[e]]:.1f}" if has[e] else None
                   for m in ("per_family", "joint") for e in ("pos", "orn")}
            cells.setdefault(robot, {})[c] = row
            tex = ["--" if row[k] is None else f"${row[k]}$" for k in SIZE_SWEEP_COLUMNS]
            lines.append(f" & {TEX_NAME[c]:<40s}& {recs['joint', robot, c]['intrinsic_dim']} & "
                         + "& ".join(f"{x:<{7 if i % 2 == 0 else 6}s}" for i, x in enumerate(tex)) + r"\\")
    print("wrote", write("size_sweep_compare.tex", "\n".join(lines) + "\n"))

    paper_path = os.path.join(PAPER_DIR, "size_sweep_compare.json")
    if not os.path.exists(paper_path):
        return
    paper = json.load(open(paper_path))["cells"]
    md = ["| Robot | Constraint | Column | Paper | Release | Match |", "|---|---|---|---|---|---|"]
    n = n_ok = 0
    for robot in ("franka", "ur5"):
        for c in PAPER_ORDER:
            pairs = [("ID", paper[robot][c]["id"], recs["joint", robot, c]["intrinsic_dim"])]
            pairs += [(k, paper[robot][c][k], cells[robot][c][k]) for k in SIZE_SWEEP_COLUMNS]
            for col, p, mine in pairs:
                n += 1
                n_ok += p == mine
                md.append(f"| {robot} | {c} | {col} | `{p}` | `{mine}` | {'yes' if p == mine else '**no**'} |")
    md.insert(0, f"Table tab:size-sweep-compare, release vs paper: {n_ok}/{n} cells identical.\n")
    print("wrote", write("size_sweep_compare_vs_paper.md", "\n".join(md) + "\n"))


# ---------------------------------------------------------------- Table tab:sensitivity

SENS_KEYS = ("rho_med", "nu_med", "tau_med", "tau_p5", "tau_rand_med", "tau_rand_p5", "delta_rel")


def sensitivity(summary_path: str = "results/flatness/summary.json") -> None:
    """Main table tab:sensitivity: medians of rho and nu, tau^q median and 5th percentile against the
    matching quantiles of the random-direction baseline (Beta), and the relative gain of the median."""
    summary = json.load(open(summary_path))
    lines = []
    for robot in ("franka", "ur5"):
        if lines:
            lines.append(r"\midrule")
        for i, c in enumerate(PAPER_ORDER):
            s = summary[robot][c]
            head = rf"\multirow{{7}}{{*}}{{\rotatebox[origin=c]{{90}}{{{TEX_ROBOT[robot]}}}}}" if i == 0 else ""
            if s["tau_rand_med"] == 0:
                tau = [f"{s['tau_med']:.3f}", rf"\tcnear{{{s['tau_p5']:.3f}}}", "0.000", "0.000", "--"]
            else:
                shade = r"\tcabove" if s["tau_p5"] > s["tau_rand_p5"] else r"\tcbelow"
                tau = [f"{s['tau_med']:.3f}", rf"{shade}{{{s['tau_p5']:.3f}}}", f"{s['tau_rand_med']:.3f}",
                       f"{s['tau_rand_p5']:.3f}", rf"\tcabove{{{100 * s['delta_rel']:+.0f}\%}}"]
            row = [rf"\tclowrho{{{s['rho_med']:.3f}}}", f"{s['nu_med']:.2f}"] + tau
            lines.append(f"{head} & {TEX_NAME[c]} & " + " & ".join(row) + r" \\")
    print("wrote", write("sensitivity.tex", "\n".join(lines) + "\n"))

    paper_path = os.path.join(PAPER_DIR, "sensitivity.json")
    if not os.path.exists(paper_path):
        return
    paper = json.load(open(paper_path))["cells"]
    md = ["| Robot | Constraint | rho_med | nu_med | tau_med | tau_p5 | baseline (med / p5) |",
          "|---|---|---|---|---|---|---|"]
    for robot in ("franka", "ur5"):
        for c in PAPER_ORDER:
            s, p = summary[robot][c], paper[robot][c]
            md.append(f"| {robot} | {c} | {p['rho_med']} -> {s['rho_med']:.3f} | {p['nu_med']} -> {s['nu_med']:.2f} | "
                      f"{p['tau_med']} -> {s['tau_med']:.3f} | {p['tau_p5']} -> {s['tau_p5']:.3f} | "
                      f"{p['tau_rand']} (RMS) -> {s['tau_rand_med']:.3f} / {s['tau_rand_p5']:.3f} |")
    rho = [summary[r][c]["rho_med"] for r in ("franka", "ur5") for c in PAPER_ORDER]
    nu = [summary[r][c]["nu_med"] for r in ("franka", "ur5") for c in PAPER_ORDER]
    md.insert(0, "Table tab:sensitivity, paper -> release (paper nu column printed sigma_max of Dg_c; paper tau_rand "
                 f"was the RMS baseline).\n\nMedian of per-cell rho_med {np.median(rho):.3f}; nu_med range "
                 f"{min(nu):.2f} to {max(nu):.2f}, median {np.median(nu):.2f}.\n")
    print("wrote", write("sensitivity_vs_paper.md", "\n".join(md) + "\n"))


TABLES = {"interp_full": interp_full, "id_full": id_full, "size_sweep_compare": size_sweep_compare,
          "sensitivity": sensitivity}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="*", help=f"any of {', '.join(TABLES)} (default: all)")
    args = p.parse_args()
    unknown = set(args.tables) - set(TABLES)
    if unknown:
        p.error(f"unknown tables {sorted(unknown)}")
    for name in args.tables or TABLES:
        TABLES[name]()


if __name__ == "__main__":
    main()
