"""Interactive viser explorer of the constraint-conditioned diffusion IK model, decoded live.

  Interpolate   latent interpolation between two IK solutions q_a, q_b at noise level t_bar (lerp or slerp)
  Latent plane  x(u, v) = x_a + u (x_b - x_a) + v (x_c - x_a) through three inverted solutions
  Knobs         Score-SVD at the current pose (t = 15, K = 50); one slider per right singular vector

Targets come from results/interpolation/targets/{robot}_{c}.npz when present, otherwise from the FK pose of a
random configuration and its IKFast solutions.

    python scripts/viser_explorer.py [--device cuda] [--port 8080]
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch
import viser

from constraint_ik import errors, get_robot, load_checkpoint, viz
from constraint_ik.constraints import DISPLAY, PAPER_ORDER, Target, intrinsic_dim
from constraint_ik.id_estimation import id_from_spectrum, score_matrix
from constraint_ik.interpolation import INTERPOLANTS
from constraint_ik.plotting import robot_color
from constraint_ik.targets import PAIR_WIDTHS, dedup, pick_diameter, pick_third

N_FRAMES = 47           # decoded frames per path; every second one is a ghost (the paper's 24 frames)
T_SCORE, K_SCORE = 15, 50


def fresh_target(robot, constraint: str, width: int, rng: np.random.Generator):
    """Target from the FK pose of a random in-limits configuration; endpoints among its IK solutions."""
    while True:
        pos, R = robot.fk(robot.sample_q(1, rng)[0])
        pool = dedup(robot.ik_all(pos, R), 0.1, robot.n_joints)
        if len(pool) >= 3:
            i, j = pick_diameter(pool, width)
            return (Target.from_pose(constraint, pos, R), pool[i], pool[j],
                    pool[pick_third(pool, pool[i], pool[j])])


class Explorer:
    def __init__(self, server: viser.ViserServer, ckpt_dir: str, device: str, seed: int):
        self.server, self.ckpt_dir, self.device = server, ckpt_dir, device
        self.rng = np.random.default_rng(seed)
        self.models, self.lock = {}, threading.RLock()
        self.arm, self.ghosts, self.knob_sliders = None, [], []
        self.fresh = False
        viz.setup_scene(server)
        server.on_client_connect(self._camera)
        self._gui()
        self.load()

    # ---------------------------------------------------------------- GUI

    def _gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Target"):
            self.g_robot = gui.add_dropdown("Robot", ("franka", "ur5"), initial_value="franka")
            self.g_c = gui.add_dropdown("Constraint", tuple(DISPLAY[c] for c in PAPER_ORDER),
                                        initial_value=DISPLAY["se3"])
            self.g_idx = gui.add_slider("Target index", 0, 29, 1, 0)
            self.g_pw = gui.add_dropdown("Pair width", tuple(str(w) for w in PAIR_WIDTHS), initial_value="100")
            self.g_fresh = gui.add_button("Fresh random target")
            self.g_err = gui.add_markdown("")
        tabs = gui.add_tab_group()
        with tabs.add_tab("Interpolate"):
            self.g_method = gui.add_dropdown("Method", ("linear", "slerp"), initial_value="linear")
            self.g_tbar = gui.add_slider("t_bar", 0, 95, 1, 40)
            self.g_alpha = gui.add_slider("alpha", 0.0, 1.0, 1.0 / (N_FRAMES - 1), 0.0)
            self.g_play = gui.add_checkbox("Play", False)
            self.g_ghost = gui.add_checkbox("Ghost trail", False)
            self.g_trace = gui.add_checkbox("End-effector trace", True)
        with tabs.add_tab("Latent plane"):
            gui.add_markdown("x(u, v) = x_a + u (x_b - x_a) + v (x_c - x_a) at t_bar")
            self.g_u = gui.add_slider("u", -0.5, 1.5, 0.02, 0.0)
            self.g_v = gui.add_slider("v", -0.5, 1.5, 0.02, 0.0)
            self.g_abc = gui.add_checkbox("Show q_a, q_b, q_c", False)
        with tabs.add_tab("Knobs"):
            gui.add_markdown(f"Score-SVD at x_0 = encode(q), t = {T_SCORE}, K = {K_SCORE}. A knob adds "
                             "s v to the DDIM inversion of x_0 at the decode level and decodes from there.")
            self.g_tdec = gui.add_slider("Decode level", 15, 95, 5, 50)
            self.g_score = gui.add_button("Compute Score-SVD at current pose")
            self.g_reset = gui.add_button("Reset knobs")
            self.g_spec = gui.add_markdown("")
            self.g_tan = gui.add_folder("Tangent directions")
            self.g_nor = gui.add_folder("Normal directions", expand_by_default=False)

        for h in (self.g_robot, self.g_c, self.g_idx, self.g_pw):
            h.on_update(lambda _: self._set_fresh(False))
        self.g_fresh.on_click(lambda _: self._set_fresh(True))
        for h in (self.g_method, self.g_tbar):
            h.on_update(lambda _: self.interpolate())
        self.g_alpha.on_update(lambda _: self.show_alpha())
        self.g_ghost.on_update(lambda _: self.draw_ghosts())
        self.g_trace.on_update(lambda _: self.draw_trace())
        for h in (self.g_u, self.g_v):
            h.on_update(lambda _: self.plane())
        self.g_abc.on_update(lambda _: self.draw_abc())
        self.g_score.on_click(lambda _: self.score_svd())
        self.g_reset.on_click(lambda _: self.reset_knobs())
        self.g_tdec.on_update(lambda _: self.knobs())

    def _camera(self, client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (1.2, -1.0, 0.8)
        client.camera.look_at = (0.1, 0.0, 0.3)

    def _set_fresh(self, fresh: bool) -> None:
        self.fresh = fresh
        self.load()

    # ---------------------------------------------------------------- state

    @property
    def constraint(self) -> str:
        return next(c for c in PAPER_ORDER if DISPLAY[c] == self.g_c.value)

    def load(self) -> None:
        with self.lock:
            robot, c, pw = self.g_robot.value, self.constraint, int(self.g_pw.value)
            if robot not in self.models:
                self.models[robot] = load_checkpoint(f"{self.ckpt_dir}/{robot}.pt", self.device)
            self.ik, self.kin = self.models[robot], get_robot(robot)
            has_file = (viz.TARGETS_DIR / f"{robot}_{c}.npz").exists()
            self.g_idx.disabled = not has_file
            if has_file and not self.fresh:
                self.target, *qs = viz.load_pair(robot, c, int(self.g_idx.value), pw)
            else:
                self.target, *qs = fresh_target(self.kin, c, pw, self.rng)
            self.q_abc = [np.asarray(q, dtype=np.float64) for q in qs]
            self.cond = self.ik.cond(self.target)
            self.clear_knobs()
            if self.arm is None or self.arm_robot != robot:
                self.server.scene.remove_by_name("/arm")
                self.arm, self.arm_robot = viz.add_robot(self.server, robot, "/arm"), robot
            self.server.scene.remove_by_name("/target")
            ee = self.kin.fk(np.stack(self.q_abc))[0].mean(0)
            viz.add_target_gizmo(self.server, self.target, "/target", robot_color(robot), center=ee, size=0.5)
            self.draw_abc()
            t = viz.canonical_tbar(robot, c, self.g_method.value)
            t = 40 if t is None else int(t)
            if self.g_tbar.value != t:
                self.g_tbar.value = t          # the slider callback re-interpolates
            else:
                self.interpolate()

    def show(self, q: np.ndarray) -> None:
        """Pose the arm and report the constraint errors of q."""
        self.q = np.asarray(q, dtype=np.float64)
        self.arm.update_cfg(self.q)
        pos_mm, orn_deg = errors(self.kin, self.q, self.target)
        parts = [f"**{DISPLAY[self.constraint]}** ({'fresh' if self.fresh else 'file'} target)"]
        if np.isfinite(pos_mm):
            parts.append(f"position error {float(pos_mm):.1f} mm")
        if np.isfinite(orn_deg):
            parts.append(f"orientation error {float(orn_deg):.2f} deg")
        self.g_err.content = "  \n".join(parts)

    def invert(self, q: np.ndarray, t: int) -> torch.Tensor:
        return self.ik.invert(q, t, self.cond)

    def decode(self, z: torch.Tensor, t: int) -> np.ndarray:
        return self.ik.decode_from(z, t, self.cond).cpu().numpy()

    # ---------------------------------------------------------------- interpolation

    def interpolate(self) -> None:
        with self.lock, torch.no_grad():
            t = int(self.g_tbar.value)
            self.z_abc = [self.invert(q, t) for q in self.q_abc]
            z = INTERPOLANTS[self.g_method.value](self.z_abc[0], self.z_abc[1], N_FRAMES)
            self.path = viz.unwrap(self.decode(z, t))
            self.ee_path = self.kin.fk(self.path)[0]
            self.draw_ghosts()
            self.draw_trace()
            self.show_alpha()

    def show_alpha(self) -> None:
        self.show(self.path[int(round(self.g_alpha.value * (N_FRAMES - 1)))])

    def draw_ghosts(self) -> None:
        self.server.scene.remove_by_name("/ghosts")
        if self.g_ghost.value:
            self.server.scene.add_frame("/ghosts", show_axes=False)
            viz.add_ghost_trail(self.server, self.arm_robot, self.path[::2], "/ghosts",
                                white=0.25, real_last=False)

    def draw_trace(self) -> None:
        self.server.scene.remove_by_name("/ee_trace")
        if self.g_trace.value:
            viz.add_ee_trace(self.server, self.ee_path, "/ee_trace", robot_color(self.arm_robot), 0.004)

    def draw_abc(self) -> None:
        self.server.scene.remove_by_name("/abc")
        if self.g_abc.value:
            self.server.scene.add_frame("/abc", show_axes=False)
            for name, q in zip("abc", self.q_abc):
                viz.add_robot(self.server, self.arm_robot, f"/abc/{name}", viz.SPECS[self.arm_robot].light,
                              0.18, 0.25).update_cfg(q)

    # ---------------------------------------------------------------- latent plane

    def plane(self) -> None:
        with self.lock, torch.no_grad():
            z_a, z_b, z_c = self.z_abc
            z = z_a + self.g_u.value * (z_b - z_a) + self.g_v.value * (z_c - z_a)
            self.show(self.decode(z, int(self.g_tbar.value))[0])

    # ---------------------------------------------------------------- Score-SVD knobs

    def score_svd(self) -> None:
        with self.lock, torch.no_grad():
            self.q0 = self.q.copy()
            noise = torch.randn(K_SCORE, self.ik.dim, generator=torch.Generator().manual_seed(0))
            S = score_matrix(self.ik, self.ik.encode(self.q0), self.cond, T_SCORE, K_SCORE,
                             noise.to(self.ik.device))
            _, sv, self.Vh = torch.linalg.svd(S, full_matrices=False)
            sv = sv.cpu().numpy()
            d_std, d_dag = id_from_spectrum(sv, self.ik.dim)
            k = self.ik.dim - d_dag
            d_true = intrinsic_dim(self.constraint, self.kin.n_joints)
            self.clear_knobs()
            self.g_spec.content = (f"Score-SVD d = {d_std}, Score-SVD† d = **{d_dag}** "
                                   f"(analytic n - k = {d_true})  \nσ: " + ", ".join(f"{s:.1f}" for s in sv))
            for i, s in enumerate(sv):
                with self.g_tan if i >= k else self.g_nor:
                    h = self.server.gui.add_slider(f"{'T' if i >= k else 'N'} v{i + 1} (σ {s:.1f})",
                                                   -2.5, 2.5, 0.05, 0.0)
                h.on_update(lambda _: self.knobs())
                self.knob_sliders.append(h)
            self.knobs()

    def clear_knobs(self) -> None:
        for h in self.knob_sliders:
            h.remove()
        self.knob_sliders = []
        self.g_spec.content = ""

    def reset_knobs(self) -> None:
        for h in self.knob_sliders:
            h.value = 0.0

    def knobs(self) -> None:
        if not self.knob_sliders:
            return
        with self.lock, torch.no_grad():
            t = int(self.g_tdec.value)
            s = torch.tensor([h.value for h in self.knob_sliders], dtype=self.Vh.dtype, device=self.Vh.device)
            q = self.decode(self.invert(self.q0, t) + (s @ self.Vh)[None], t)[0]
            self.show(viz.unwrap(np.stack([self.q0, q]))[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--seed", type=int, default=0, help="seed of the fresh random targets")
    args = p.parse_args()
    server = viser.ViserServer(port=args.port)
    ex = Explorer(server, args.ckpt_dir, args.device, args.seed)
    while True:
        time.sleep(1.0 / 12)
        if ex.g_play.value:
            a = ex.g_alpha.value + 1.0 / (N_FRAMES - 1)
            ex.g_alpha.value = 0.0 if a > 1.0 + 1e-9 else a


if __name__ == "__main__":
    main()
