"""Render the README media headlessly: a viser scene captured through headless Chrome (Playwright).

  docs/media/hero.gif         Franka SE(3) and UR5 Line+Orientation latent interpolations
  docs/media/ghost_grid.png   ghost trails, 2 robots x 4 constraints (paper Fig. 4 layout)
  docs/media/knobs.gif        Score-SVD directions at a Franka Plane solution (tangent vs normal)
  docs/media/manifest.json    robot, constraint, target, pair width, t_bar and camera of every file

With --paper it renders the paper's ghost-trail panels instead (print quality, linear t_bar of configs/tbar.json):
  docs/paper_renders/{robot}_{constraint}.png   all 14 cells
  docs/paper_renders/ur5_se3_branch_{i}.png     UR5 SE(3) widest pairs of four targets (appendix branch figure)
  docs/paper_renders/manifest.json

Frames go to outputs/media/<name>/. Headless Chrome renders WebGL on the GPU through ANGLE/Vulkan
(--gl gpu) or on the CPU with SwiftShader (--gl swiftshader, slow); no display is needed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import viser  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from constraint_ik import errors, get_robot, load_checkpoint, viz  # noqa: E402
from constraint_ik.constraints import DISPLAY, PAPER_ORDER  # noqa: E402
from constraint_ik.id_estimation import id_from_spectrum, score_matrix  # noqa: E402
from constraint_ik.interpolation import interpolate  # noqa: E402
from constraint_ik.plotting import constraint_label, robot_color, robot_label, setup_style  # noqa: E402

GL_ARGS = {"swiftshader": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
           "gpu": ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist",
                   "--enable-unsafe-swiftshader"]}
GHOSTS = 24          # animations show (GHOSTS - 1) * SUB + 1 frames of the decoded path
SUB = 3
FOV = 36.0           # vertical field of view (deg); cameras are fitted to the swept meshes, see viz.fit_camera
WHITE = 0.25         # fraction of white mixed into the light robot colour of the ghosts
STILL_GHOSTS = 6     # ghosts spaced evenly by link displacement, opacity STILL_RAMP, last pose untinted
STILL_FRAMES = 241   # decoded frames of a still; the ghosts are picked among them
STILL_RAMP = (0.08, 0.6)
SETTINGS = {"ghosts": f"{STILL_GHOSTS}, evenly spaced by mean link displacement, opacity "
                      f"{STILL_RAMP[0]} -> {STILL_RAMP[1]}, last pose untinted; no end-effector trace or orientation gizmo",
            "ghost_tint": "robot light colour + 25% white", "fov_deg": FOV, "grid": "2 x 2 m, 0.1 m cells",
            "method": "linear"}
NAMES = {"franka": "Franka", "ur5": "UR5"}
GRID = ("plane", "posonly", "line_orient", "se3")


@dataclass
class Clip:
    robot: str
    constraint: str
    target: int = 0
    pair_width: int = 100
    azim: float = -35.0
    elev: float = 22.0
    method: str = "linear"


# Target index and pair width of results/interpolation/targets, and the camera direction, per panel
# (chosen for a readable view: arm above the ground, final pose in front).
CLIPS = {
    ("franka", "plane"): Clip("franka", "plane", 1, 75, -100, 20),
    ("franka", "posonly"): Clip("franka", "posonly", 2, 75, 100, 20),
    ("franka", "line_orient"): Clip("franka", "line_orient", 1, 50, -110, 18),
    ("franka", "se3"): Clip("franka", "se3", 1, 100, 20, 20),
    ("ur5", "plane"): Clip("ur5", "plane", 15, 75, 100, 26),
    ("ur5", "posonly"): Clip("ur5", "posonly", 17, 50, 30, 22),
    ("ur5", "line_orient"): Clip("ur5", "line_orient", 1, 50, -40, 20),
    ("ur5", "se3"): Clip("ur5", "se3", 2, 50, -40, 18),
}
# The remaining cells of the paper renders, and the four UR5 SE(3) branch-switching targets (pair width 100).
PAPER_EXTRA = {
    ("franka", "line"): Clip("franka", "line", 12, 75, -40, 20),
    ("franka", "orient"): Clip("franka", "orient", 8, 100, 30, 20),
    ("franka", "plane_orient"): Clip("franka", "plane_orient", 2, 75, -40, 20),
    ("ur5", "line"): Clip("ur5", "line", 26, 50, -40, 20),
    ("ur5", "orient"): Clip("ur5", "orient", 7, 100, -40, 20),
    ("ur5", "plane_orient"): Clip("ur5", "plane_orient", 12, 75, -40, 20),
}
SE3_BRANCH = [Clip("ur5", "se3", i, 100, -40, 20) for i in (13, 7, 25, 23)]
KNOBS = Clip("franka", "plane", 1, 75, 195, 30)


def t_bar_of(clip: Clip) -> int:
    t = viz.canonical_tbar(clip.robot, clip.constraint, clip.method)
    if t is None:
        raise FileNotFoundError(f"{viz.TBAR_FILE} has no t_bar for {clip.robot} {clip.constraint} {clip.method}")
    return int(t)


class Stage:
    """A viser server and one headless Chrome client that renders it."""

    def __init__(self, port: int, browser: str | None, gl: str = "gpu"):
        self.server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        self._pw = sync_playwright().start()
        bundled = Path(self._pw.chromium.executable_path).exists()
        exe = browser or (None if bundled else shutil.which("google-chrome"))
        self._browser = self._pw.chromium.launch(executable_path=exe, headless=True, args=GL_ARGS[gl])
        # get_render sizes must fit the page canvas (2x supersampled 1200 px paper panels).
        self._page = self._browser.new_page(viewport={"width": 2600, "height": 2600})
        self._page.goto(f"http://127.0.0.1:{port}")
        t0 = time.time()
        while not self.server.get_clients():
            if time.time() - t0 > 60:
                raise RuntimeError("the headless browser did not connect to viser")
            time.sleep(0.1)
        self.client = next(iter(self.server.get_clients().values()))

    def reset(self) -> None:
        self.server.scene.reset()
        viz.setup_scene(self.server)

    def shot(self, cam: dict, width: int, height: int, ss: int = 1) -> np.ndarray:
        """RGB frame over a white background; ss > 1 renders larger and downsamples."""
        viz.apply_camera(self.client, cam)
        rgba = self.client.get_render(height=height * ss, width=width * ss, transport_format="png")
        img = Image.alpha_composite(Image.new("RGBA", (width * ss, height * ss), "white"),
                                    Image.fromarray(rgba).convert("RGBA")).convert("RGB")
        return np.asarray(img.resize((width, height), Image.LANCZOS) if ss > 1 else img)

    def settle(self, cam: dict, seconds: float = 2.0) -> None:
        time.sleep(seconds)
        self.shot(cam, 64, 48)

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()
        self.server.stop()


class Models:
    def __init__(self, ckpt_dir: str, device: str):
        self.ckpt_dir, self.device, self._ik = ckpt_dir, device, {}

    def path(self, robot: str) -> str:
        return f"{self.ckpt_dir}/{self.name(robot)}"

    @staticmethod
    def name(robot: str) -> str:
        return f"{robot}.pt"

    def __call__(self, robot: str):
        if robot not in self._ik:
            self._ik[robot] = load_checkpoint(self.path(robot), self.device)
        return self._ik[robot]


def _mean(x: np.ndarray) -> float | None:
    return round(float(np.nanmean(x)), 3) if np.isfinite(x).any() else None


# ---------------------------------------------------------------- scenes

def clip_path(models: Models, clip: Clip, n: int):
    target, q_a, q_b, _ = viz.load_pair(clip.robot, clip.constraint, clip.target, clip.pair_width)
    t_bar = t_bar_of(clip)
    q = interpolate(models(clip.robot), q_a, q_b, target, t_bar, clip.method, n)
    return target, viz.unwrap(q), t_bar


def still_ghosts(robot: str, q: np.ndarray, k: int = STILL_GHOSTS) -> np.ndarray:
    """k path indices spaced evenly by the mean displacement of the link frames (first pose included)."""
    F = get_robot(robot).link_frames(torch.as_tensor(q, dtype=torch.float64))[..., :3, 3].numpy()
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(F, axis=0), axis=-1).mean(-1))])
    return np.unique(np.searchsorted(s, np.linspace(0.0, s[-1], k + 1)[:-1]))


def trail_panel(stage: Stage, models: Models, clip: Clip, size: tuple[int, int], animate: bool,
                ss: int = 1) -> tuple[list[np.ndarray], dict]:
    """Ghost-trail interpolation, decoded at STILL_FRAMES points: the arm moving along alpha with each ghost
    appearing once the arm has passed it (animate, (GHOSTS - 1) * SUB + 1 frames), or only the final still.
    Ghosts sit at STILL_GHOSTS poses spaced evenly by link displacement, with opacity STILL_RAMP; the final
    pose is the untinted robot."""
    target, q, t_bar = clip_path(models, clip, STILL_FRAMES)
    ee = get_robot(clip.robot).fk(q)[0]
    center, side = viz.gizmo_size(target, ee)
    pts = np.vstack([viz.link_points(clip.robot, q[:: STILL_FRAMES // GHOSTS]),
                     viz.gizmo_points(target, center, side)])
    cam = viz.fit_camera(pts, clip.azim, clip.elev, FOV, size[0] / size[1])
    stage.reset()
    viz.add_target_gizmo(stage.server, target, "/target", robot_color(clip.robot), center=center, size=side,
                         orientation=False)
    arm = viz.add_robot(stage.server, clip.robot, "/arm")
    # Ghosts that coincide with the final pose (max joint difference < 0.05 rad) would only hide it.
    idx = [i for i in still_ghosts(clip.robot, q) if np.abs(q[i] - q[-1]).max() >= 0.05]
    # Fewer ghosts keep the top of the ramp, so a lone ghost (the start branch under SE(3)) stays visible.
    ramp = np.linspace(*STILL_RAMP, STILL_GHOSTS)[STILL_GHOSTS - len(idx):]
    steps = np.round(np.linspace(0, STILL_FRAMES - 1, (GHOSTS - 1) * SUB + 1)).astype(int) if animate \
        else [STILL_FRAMES - 1]
    frames, added = [], 0
    for j in steps:
        while added < len(idx) and idx[added] < j:
            viz.add_robot(stage.server, clip.robot, f"/ghosts/{idx[added]:03d}", viz.SPECS[clip.robot].light,
                          float(ramp[added]), WHITE).update_cfg(q[idx[added]])
            added += 1
        arm.update_cfg(q[j])
        if j == steps[0]:
            stage.settle(cam, 3.0)
        frames.append(stage.shot(cam, *size, ss))
    pos_mm, orn_deg = errors(get_robot(clip.robot), q, target)
    meta = {**asdict(clip), "t_bar": t_bar, "n_frames_path": STILL_FRAMES, "ghost_indices": [int(i) for i in idx],
            "camera": cam, "mean_pos_err_mm": _mean(pos_mm), "mean_orn_err_deg": _mean(orn_deg),
            "constraint_display": DISPLAY[clip.constraint]}
    return frames, meta


# ---------------------------------------------------------------- composition and encoding

def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_manager.findfont("DejaVu Serif"), size)


def label(img: np.ndarray, text: str, color: str, xy=(14, 10), size: int = 17) -> np.ndarray:
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, text, fill=viz.rgb255(color), font=font(size))
    return np.asarray(im)


def timeline(frames: list[np.ndarray], fps: int, hold_start: float = 0.6, hold_end: float = 1.6) -> list:
    return [frames[0]] * int(hold_start * fps) + frames + [frames[-1]] * int(hold_end * fps)


def encode(frames: list[np.ndarray], out: Path, fps: int, frames_dir: Path, colors: int = 160) -> dict:
    """Write PNG frames and a palette-optimized GIF; returns the GIF size."""
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(frames_dir / f"{i:04d}.png")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ["-framerate", str(fps), "-i", str(frames_dir / "%04d.png")]
    graph = (f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
             "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", *src, "-filter_complex", graph, "-loop", "0",
                    str(out.with_suffix(".gif"))], check=True)
    return {".gif": out.with_suffix(".gif").stat().st_size}


def tile(panels: list[list[np.ndarray]], cols: int, gap: int = 8) -> list[np.ndarray]:
    """Frame-wise grid of equally long panel sequences with white gutters."""
    h, w, _ = panels[0][0].shape
    rows = (len(panels) + cols - 1) // cols
    out = []
    for fr in zip(*panels):
        canvas = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 255, np.uint8)
        for k, f in enumerate(fr):
            r, c = divmod(k, cols)
            canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = f
        out.append(canvas)
    return out


# ---------------------------------------------------------------- media

def make_hero(stage, models, out_dir, fps):
    size = (356, 300)
    pair = [CLIPS["franka", "se3"], CLIPS["ur5", "line_orient"]]
    panels, metas = [], []
    for clip in pair:
        frames, meta = trail_panel(stage, models, clip, size, animate=True)
        text = f"{NAMES[clip.robot]}  ·  {DISPLAY[clip.constraint]}"
        panels.append([label(f, text, robot_color(clip.robot)) for f in timeline(frames, fps)])
        metas.append(meta)
    return "hero", encode(tile(panels, 2), out_dir / "hero", fps, Path("outputs/media/hero")), metas


def make_grid(stage, models, out_dir):
    size = (560, 420)
    imgs, metas = {}, []
    for robot in ("franka", "ur5"):
        for c in GRID:
            frames, meta = trail_panel(stage, models, CLIPS[robot, c], size, animate=False, ss=2)
            imgs[robot, c] = frames[-1]
            metas.append(meta)
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 3.3), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for i, robot in enumerate(("franka", "ur5")):
        for j, c in enumerate(GRID):
            ax = axes[i, j]
            ax.imshow(imgs[robot, c])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(robot_color(robot))
                s.set_linewidth(1.4)
            if i == 0:
                ax.set_title(constraint_label(c), fontsize=10)
            if j == 0:
                ax.set_ylabel(robot_label(robot), fontsize=12, color=robot_color(robot))
    path = out_dir / "ghost_grid.png"
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return "ghost_grid", {".png": path.stat().st_size}, metas


def make_knobs(stage, models, out_dir, fps, seed: int = 0, t: int = 15, K: int = 50, t_dec: int = 50,
               s_max: float = 1.2):
    """Score-SVD directions at x0 = encode(q_a) (t = 15, K = 50); a knob adds s v to the DDIM inversion of x0
    at level t_dec and decodes from there. Left, the tangent direction that moves the arm most: the arm slides
    along the constraint. Right, the normal direction whose plain step x0 + s v (no model) leaves the
    constraint most: the faint arm shows that step, the solid arm the decoded configuration."""
    clip = KNOBS
    ik, robot = models(clip.robot), get_robot(clip.robot)
    target, q0, _, _ = viz.load_pair(clip.robot, clip.constraint, clip.target, clip.pair_width)
    cond = ik.cond(target)
    x0 = ik.encode(q0)
    noise = torch.randn(K, ik.dim, generator=torch.Generator().manual_seed(seed)).to(ik.device)
    S = score_matrix(ik, x0, cond, t, K, noise)
    _, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    d_std, d_dag = id_from_spectrum(sv.cpu().numpy(), ik.dim)
    x_t = ik.invert(q0, t_dec, cond)
    s = s_max * np.sin(np.linspace(0, 2 * np.pi, 80, endpoint=False))
    s_t = torch.as_tensor(s, dtype=x_t.dtype, device=x_t.device)[:, None]

    def unwrap(q):
        return viz.unwrap(np.vstack([q0[None], q]))[1:]

    def decoded(i):
        return unwrap(ik.decode_from(x_t + s_t * Vh[i][None], t_dec, cond).cpu().numpy())

    def plain(i):
        return unwrap(ik.decode(x0.reshape(1, -1) + s_t * Vh[i][None]).detach().cpu().numpy())

    def err(q):
        pos_mm, orn_deg = errors(robot, q, target)
        return pos_mm if np.isfinite(pos_mm).any() else orn_deg

    k_n = ik.dim - d_dag
    tangent = {i: decoded(i) for i in range(k_n, ik.dim)}
    i_tan = max(tangent, key=lambda i: np.abs(tangent[i] - q0).max())
    raw = {i: plain(i) for i in range(k_n)}
    i_nor = max(raw, key=lambda i: np.nanmax(err(raw[i])))
    q_nor = decoded(i_nor)
    unit = "mm" if np.isfinite(errors(robot, q0[None], target)[0]).any() else "deg"
    place = {"plane": "the plane", "line": "the line", "posonly": "the point"}.get(clip.constraint, "the constraint")
    size = (356, 320)
    ee = robot.fk(np.vstack([tangent[i_tan], q_nor]))[0]
    center, side = viz.gizmo_size(target, ee)
    pts = np.vstack([viz.link_points(clip.robot, np.vstack([tangent[i_tan], raw[i_nor], q_nor])),
                     viz.gizmo_points(target, center, side)])
    cam = viz.fit_camera(pts, clip.azim, clip.elev, FOV, size[0] / size[1])
    color = robot_color(clip.robot)
    panels = []
    for kind, i, q, q_raw in (("tangent", i_tan, tangent[i_tan], None), ("normal", i_nor, q_nor, raw[i_nor])):
        stage.reset()
        viz.add_target_gizmo(stage.server, target, "/target", color, center=center, size=side, orientation=False)
        rest = viz.add_robot(stage.server, clip.robot, "/rest", viz.SPECS[clip.robot].light, 0.25, WHITE)
        rest.update_cfg(q0)
        ghost = None if q_raw is None else viz.add_robot(stage.server, clip.robot, "/plain", "#6B6F76", 0.45)
        arm = viz.add_robot(stage.server, clip.robot, "/arm")
        e_dec, e_raw = err(q), None if q_raw is None else err(q_raw)
        title = "tangent direction" if q_raw is None else "normal direction"
        caption = f"slides along {place}" if q_raw is None else f"pulled back onto {place}"
        frames = []
        for j, qj in enumerate(q):
            arm.update_cfg(qj)
            if ghost is not None:
                ghost.update_cfg(q_raw[j])
            if j == 0:
                stage.settle(cam, 3.0)
            f = label(label(stage.shot(cam, *size), title, color, size=15), caption, "#444444", (14, 31), 12)
            if q_raw is not None:
                f = label(f, f"grey, without the model  {e_raw[j]:.0f} {unit}", "#6b6f76", (14, size[1] - 44), 12)
            frames.append(label(f, f"decoded  {e_dec[j]:.1f} {unit}", "#444444", (14, size[1] - 26), 12))
        panels.append(frames)
    meta = [{**asdict(clip), "q0": "q_a", "t_score": t, "K": K, "seed": seed, "t_decode": t_dec, "s_max": s_max,
             "camera": cam, "d_hat_score_svd": int(d_std), "d_hat_score_svd_dagger": int(d_dag),
             "singular_values": [round(float(v), 2) for v in sv],
             "tangent": {"direction": i_tan + 1, "max_joint_move_rad": round(float(np.abs(tangent[i_tan] - q0).max()), 3),
                         "max_error": round(float(np.nanmax(err(tangent[i_tan]))), 2)},
             "normal": {"direction": i_nor + 1, "max_plain_error": round(float(np.nanmax(err(raw[i_nor]))), 2),
                        "max_decoded_error": round(float(np.nanmax(err(q_nor))), 2)}, "error_unit": unit}]
    return "knobs", encode(tile(panels, 2), out_dir / "knobs", fps, Path("outputs/media/knobs")), meta


def make_paper(stage, models, out_dir: Path, size: int = 1200):
    """Print-quality ghost trails (2x supersampled) of every cell and of the UR5 SE(3) branch targets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = {**CLIPS, **PAPER_EXTRA}
    jobs = [(f"{r}_{c}", cells[r, c]) for r in ("franka", "ur5") for c in PAPER_ORDER]
    jobs += [(f"ur5_se3_branch_{i}", clip) for i, clip in enumerate(SE3_BRANCH)]
    metas = []
    for name, clip in jobs:
        frames, meta = trail_panel(stage, models, clip, (size, size), animate=False, ss=2)
        Image.fromarray(frames[-1]).save(out_dir / f"{name}.png")
        metas.append({"file": f"{name}.png", **meta})
        print(name, meta["t_bar"], flush=True)
    return "paper_renders", {m["file"]: (out_dir / m["file"]).stat().st_size for m in metas}, metas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="hero,grid,knobs")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="docs/media")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--port", type=int, default=8097)
    p.add_argument("--browser", default=None,
                   help="Chrome/Chromium executable (default: Playwright's Chromium, else google-chrome on PATH)")
    p.add_argument("--gl", default="gpu", choices=list(GL_ARGS),
                   help="WebGL backend: gpu (ANGLE/Vulkan; DRI_PRIME=<i> picks the GPU) or swiftshader (CPU)")
    p.add_argument("--paper", action="store_true", help="render docs/paper_renders instead of the README media")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    out_dir = Path("docs/paper_renders" if args.paper else args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    models = Models(args.ckpt_dir, args.device)
    stage = Stage(args.port, args.browser, args.gl)
    jobs = [lambda: make_paper(stage, models, out_dir)] if args.paper else []
    only = [] if args.paper else args.only.split(",")
    if "hero" in only:
        jobs.append(lambda: make_hero(stage, models, out_dir, args.fps))
    if "grid" in only:
        jobs.append(lambda: make_grid(stage, models, out_dir))
    if "knobs" in only:
        jobs.append(lambda: make_knobs(stage, models, out_dir, args.fps))
    try:
        for job in jobs:
            t0 = time.time()
            name, sizes, metas = job()
            files = sizes if args.paper else {f"{name}{k}": v for k, v in sizes.items()}
            manifest[name] = {"files": files, "fps": None if args.paper else args.fps,
                              "checkpoints": {m["robot"]: f"checkpoints/{models.name(m['robot'])}" for m in metas},
                              "settings": SETTINGS, "panels": metas, "seconds": round(time.time() - t0, 1)}
            print(f"{name}: {len(files)} files ({time.time() - t0:.0f} s)")
            manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    finally:
        stage.close()


if __name__ == "__main__":
    main()
