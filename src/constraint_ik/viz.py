"""Viser scene helpers shared by the explorer and the media renderer: robots, ghost trails, constraint gizmos."""
from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf
import yourdfpy
from matplotlib.colors import to_rgb
from viser.extras import ViserUrdf

from constraint_ik.constraints import Target, n_orient_bits, pinned_axis, pos_axes
from constraint_ik.plotting import FRANKA, FRANKA_LIGHT, UR5, UR5_LIGHT
from constraint_ik.targets import PAIR_WIDTHS, pick_pairs, recipe, synthesize_targets

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robots"
TARGETS_DIR = Path("results/interpolation/targets")
TBAR_FILE = Path("configs/tbar.json")


@dataclass(frozen=True)
class RobotSpec:
    urdf: str
    joints: tuple[str, ...]
    ee_link: str   # URDF link whose pose equals the IKFast end-effector pose (robots.fk)
    color: str
    light: str


SPECS = {
    "ur5": RobotSpec("ur5/ur5.urdf",
                     ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
                     "ee_link", UR5, UR5_LIGHT),
    "franka": RobotSpec("panda/panda.urdf", tuple(f"panda_joint{i}" for i in range(1, 8)),
                        "panda_link8", FRANKA, FRANKA_LIGHT),
}


def rgb(color: str, white: float = 0.0) -> tuple[float, float, float]:
    """Hex colour as floats in [0, 1], optionally mixed with a fraction of white."""
    c = np.asarray(to_rgb(color))
    return tuple(float(v) for v in c + white * (1.0 - c))


def rgb255(color: str, white: float = 0.0) -> tuple[int, int, int]:
    return tuple(int(round(255 * v)) for v in rgb(color, white))


@functools.lru_cache(maxsize=None)
def load_urdf(robot: str) -> yourdfpy.URDF:
    spec = SPECS[robot]
    urdf = yourdfpy.URDF.load(str(ASSETS / spec.urdf), build_collision_scene_graph=False,
                              load_collision_meshes=False)
    assert tuple(urdf.actuated_joint_names) == spec.joints, urdf.actuated_joint_names
    return urdf


def unwrap(q_path: np.ndarray) -> np.ndarray:
    """Shift each frame by multiples of 2 pi so consecutive frames differ by at most pi per joint."""
    q = np.asarray(q_path, dtype=np.float64).copy()
    for i in range(1, len(q)):
        q[i] = q[i - 1] + (q[i] - q[i - 1] + np.pi) % (2 * np.pi) - np.pi
    return q


def setup_scene(server: viser.ViserServer, grid: bool = True) -> None:
    """White background, +z up, and the paper's light 2 x 2 m grid with 0.1 m cells."""
    server.scene.set_up_direction("+z")
    if grid:
        server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1, plane="xy",
                              cell_color=(215, 215, 215), section_color=(165, 165, 165),
                              section_size=1.0, shadow_opacity=0.12)


def add_robot(server: viser.ViserServer, robot: str, name: str = "/robot",
              color: str | None = None, opacity: float = 1.0, white: float = 0.0) -> ViserUrdf:
    """Robot with its own materials, or flat-tinted with `color` at `opacity` (a ghost)."""
    override = None if color is None else (*rgb(color, white), float(opacity))
    return ViserUrdf(server, load_urdf(robot), root_node_name=name, mesh_color_override=override)


def ghost_opacities(K: int, start: float = 0.15) -> np.ndarray:
    """Paper ramp: ghost k of K-1 gets start + k/(K-2) (1 - start); the last pose is the real robot."""
    if K <= 2:
        return np.full(max(K - 1, 0), start)
    return start + np.arange(K - 1) / (K - 2) * (1.0 - start)


def add_ghost_trail(server: viser.ViserServer, robot: str, q_path: np.ndarray, name: str = "/ghosts",
                    color: str | None = None, start: float = 0.15, white: float = 0.0,
                    real_last: bool = True) -> list[ViserUrdf]:
    """Tinted ghosts along q_path with the opacity ramp; the last pose untinted (or tinted if not real_last)."""
    color = color or SPECS[robot].light
    q_path = np.asarray(q_path)
    handles = []
    for k, op in enumerate(ghost_opacities(len(q_path), start)):
        h = add_robot(server, robot, f"{name}/{k:03d}", color, op, white)
        h.update_cfg(q_path[k])
        handles.append(h)
    last = add_robot(server, robot, f"{name}/{len(q_path) - 1:03d}",
                     None if real_last else color, 1.0, white)
    last.update_cfg(q_path[-1])
    return handles + [last]


def add_ee_trace(server: viser.ViserServer, points: np.ndarray, name: str = "/ee_trace",
                 color: str = "#222222", thickness: float = 0.006):
    """Polyline through end-effector positions (N, 3)."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        pts = np.repeat(pts[:1], 2, axis=0)
    return server.scene.add_spline_catmull_rom(name, pts, color=rgb255(color), thickness=thickness)


def _arrow(server: viser.ViserServer, name: str, origin: np.ndarray, direction: np.ndarray,
           length: float, color, radius: float = 0.006) -> None:
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    shaft = trimesh.creation.cylinder(radius=radius, height=0.8 * length)
    shaft.apply_translation((0, 0, 0.4 * length))
    head = trimesh.creation.cone(radius=2.6 * radius, height=0.2 * length)
    head.apply_translation((0, 0, 0.8 * length))
    mesh = trimesh.util.concatenate([shaft, head])
    wxyz = _rotation_to(np.array([0.0, 0.0, 1.0]), d)
    server.scene.add_mesh_simple(name, mesh.vertices, mesh.faces, color=color, wxyz=wxyz,
                                 position=tuple(origin), cast_shadow=False)


def _rotation_to(a: np.ndarray, b: np.ndarray) -> tuple[float, ...]:
    """Quaternion (wxyz) of the shortest rotation taking unit a to unit b."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        return tuple(tf.SO3.exp(np.pi * axis / np.linalg.norm(axis)).wxyz)
    q = np.array([1.0 + c, *v])
    return tuple(q / np.linalg.norm(q))


def gizmo_size(target: Target, ee: np.ndarray, lo: float = 0.3, hi: float = 0.9) -> tuple[np.ndarray, float]:
    """Centre and side length of the plane square / line segment that covers end-effector positions (N, 3)."""
    free = [i for i in range(3) if i not in pos_axes(target.constraint)]
    extent = float(np.ptp(ee[:, free], axis=0).max()) if free else 0.0
    return ee.mean(0), float(np.clip(1.3 * extent + 0.1, lo, hi))


def gizmo_points(target: Target, center: np.ndarray, size: float) -> np.ndarray:
    """Extreme points of the gizmo's plane square or line (for camera framing); (M, 3)."""
    fixed = pos_axes(target.constraint)
    free = [i for i in range(3) if i not in fixed]
    p = np.asarray(center, dtype=np.float64).copy()
    p[fixed] = target.pos[fixed]
    if len(free) not in (1, 2):
        return p[None]
    offsets = np.array(np.meshgrid(*[[-size / 2, size / 2]] * len(free))).reshape(len(free), -1).T
    pts = np.tile(p, (len(offsets), 1))
    pts[:, free] += offsets
    return pts


def add_target_gizmo(server: viser.ViserServer, target: Target, name: str = "/target",
                     color: str = "#444444", center: np.ndarray | None = None,
                     size: float = 0.6, orientation: bool = True) -> viser.FrameHandle:
    """Draw M_c's task-space set: plane -> translucent square, line -> thin cylinder, point -> sphere,
    full orientation -> frame axes, pinned body axis -> arrow (both skipped with orientation=False).
    Free position axes are placed at `center`."""
    root = server.scene.add_frame(name, show_axes=False)
    fixed = pos_axes(target.constraint)
    free = [i for i in range(3) if i not in fixed]
    p = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64).copy()
    p[fixed] = target.pos[fixed]
    col = rgb255(color)
    if len(free) == 2:
        dims = np.full(3, 0.002)
        dims[free] = size
        server.scene.add_box(f"{name}/plane", color=rgb255(color, 0.35), dimensions=tuple(dims),
                             position=tuple(p), opacity=0.22, side="double", cast_shadow=False,
                             receive_shadow=False)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * size / 2
        pts = np.tile(p, (5, 1))
        pts[:, free] += corners
        server.scene.add_line_segments(f"{name}/edge", np.stack([pts[:-1], pts[1:]], axis=1),
                                       colors=col, thickness=0.003)
    elif len(free) == 1:
        axis = np.zeros(3)
        axis[free[0]] = 1.0
        server.scene.add_cylinder(f"{name}/line", radius=0.004, height=size, color=col,
                                  wxyz=_rotation_to(np.array([0.0, 0.0, 1.0]), axis),
                                  position=tuple(p), cast_shadow=False, receive_shadow=False)
    elif len(free) == 0:
        server.scene.add_icosphere(f"{name}/point", radius=0.018, color=col, position=tuple(p),
                                   cast_shadow=False, receive_shadow=False)
    nb = n_orient_bits(target.constraint) if orientation else 0
    if nb == 3:
        server.scene.add_frame(f"{name}/frame", axes_length=0.12, axes_radius=0.006,
                               origin_radius=0.0, wxyz=tuple(tf.SO3.from_matrix(target.R).wxyz),
                               position=tuple(p))
    elif nb == 2:
        _arrow(server, f"{name}/axis", p, target.R[:, pinned_axis(target.constraint)], 0.16, col)
    return root


# ---------------------------------------------------------------- scene data and camera


def load_pair(robot: str, constraint: str, index: int = 0, width: int = 100,
              targets_dir: Path = TARGETS_DIR) -> tuple[Target, np.ndarray, np.ndarray, np.ndarray]:
    """(target, q_a, q_b, q_c) of an interpolation target set entry; synthesized with the same seed
    (identical to generate_targets.py) when the target file is missing."""
    path = Path(targets_dir) / f"{robot}_{constraint}.npz"
    if path.exists():
        d = np.load(path)
        k = int(np.flatnonzero((d["pair_target"] == index) & (d["pair_width"] == width))[0])
        pos, R, abc = d["pos"][index], d["R"][index], (d["q_a"][k], d["q_b"][k], d["q_c"][k])
    else:
        t = synthesize_targets(robot, constraint, index + 1, seed=0)[index]
        pos, R = t["pos"], t["R"]
        abc = pick_pairs(t, recipe(robot, constraint))[PAIR_WIDTHS.index(width)]
    return (Target.from_pose(constraint, pos, R), *abc)


def canonical_tbar(robot: str, constraint: str, method: str = "linear") -> int | None:
    """t_bar selected by eval_interpolation.py (configs/tbar.json), or None."""
    if not TBAR_FILE.exists():
        return None
    return json.loads(TBAR_FILE.read_text()).get(robot, {}).get(constraint, {}).get(method)


def link_points(robot: str, q_path: np.ndarray) -> np.ndarray:
    """Corners of every visual mesh's bounding box over a path, (N, 3); used to frame the camera."""
    urdf = load_urdf(robot)
    graph, geoms = urdf.scene.graph, urdf.scene.geometry
    pts = []
    for q in np.atleast_2d(q_path):
        urdf.update_cfg(q)
        for node in graph.nodes_geometry:
            T, g = graph.get(node)
            c = trimesh.bounds.corners(geoms[g].bounds)
            pts.append(c @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts)


def view_dir(azim_deg: float, elev_deg: float) -> np.ndarray:
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def fit_camera(points: np.ndarray, azim: float, elev: float, fov_deg: float, aspect: float,
               margin: float = 0.1) -> dict:
    """Perspective camera looking along -view_dir(azim, elev) that fits `points` inside the frame with
    `margin` (fraction of the half-extent) on the tighter axis; the look-at point centres the projection."""
    d = view_dir(azim, elev)
    f = -d
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    ty = np.tan(np.radians(fov_deg) / 2)
    tx = ty * aspect
    look = 0.5 * (points.min(0) + points.max(0))
    dist = 2.0
    for _ in range(6):
        def extent(D):
            rel = points - (look + D * d)
            z = rel @ f
            return (rel @ r) / (z * tx), (rel @ u) / (z * ty), z
        lo, hi = 0.2, 20.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            x, y, _ = extent(mid)
            lo, hi = (mid, hi) if max(np.abs(x).max(), np.abs(y).max()) > 1 - margin else (lo, mid)
        dist = hi
        x, y, z = extent(dist)
        zm = float(np.median(z))
        look = look + r * 0.5 * (x.max() + x.min()) * zm * tx + u * 0.5 * (y.max() + y.min()) * zm * ty
    pos = look + dist * d
    return {"position": [round(float(v), 4) for v in pos], "look_at": [round(float(v), 4) for v in look],
            "fov_deg": fov_deg}


def apply_camera(client: viser.ClientHandle, cam: dict) -> None:
    client.camera.up_direction = (0.0, 0.0, 1.0)
    client.camera.fov = float(np.radians(cam["fov_deg"]))
    client.camera.position = tuple(cam["position"])
    client.camera.look_at = tuple(cam["look_at"])
