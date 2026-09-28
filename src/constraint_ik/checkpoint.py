"""Checkpoint schema v1: load (EMA weights) into DiffusionIK, and save in the same schema."""
from __future__ import annotations

import os

import torch

from constraint_ik.constraints import CONSTRAINTS
from constraint_ik.encoding import Normalizer
from constraint_ik.model import CondEndpointDDPM
from constraint_ik.pipeline import DiffusionIK

SCHEMA_VERSION = 1
_ROBOT_BY_DIM = {12: "ur5", 14: "franka"}


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> DiffusionIK:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("model_class") != "CondEndpointDDPM" or ckpt.get("encoding") != "sincos":
        raise ValueError(f"{path}: expected a sincos CondEndpointDDPM checkpoint")
    model = CondEndpointDDPM.from_config(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["ema_state"], strict=True)
    model.eval().requires_grad_(False)
    robot = ckpt.get("robot") or (ckpt.get("args") or {}).get("robot") or _ROBOT_BY_DIM[model.in_dim]
    return DiffusionIK(
        model=model,
        alphas=ckpt["alphas_cumprod"].to(device),
        x_normalizer=Normalizer.from_dict(ckpt["normalizer"]).to(device),
        pos_normalizer=Normalizer.from_dict(ckpt["pos_normalizer"]),
        R6d_normalizer=Normalizer.from_dict(ckpt.get("R6d_normalizer")),
        robot=robot,
        type_names=list(ckpt["type_names"]),
        meta={k: ckpt.get(k) for k in ("schema_version", "step", "n_params", "args")},
    )


def save_checkpoint(path: str, *, model: CondEndpointDDPM, ema_state: dict,
                    x_normalizer: Normalizer, pos_normalizer: Normalizer, alphas: torch.Tensor,
                    robot: str, step: int, args: dict | None = None,
                    type_names: tuple[str, ...] = CONSTRAINTS,
                    R6d_normalizer: Normalizer | None = None) -> None:
    """Write schema v1. `ema_state` holds the weights used at inference; `model` the raw ones."""
    ckpt = {
        "schema_version": SCHEMA_VERSION,
        "model_class": "CondEndpointDDPM",
        "encoding": "sincos",
        "model_cfg": dict(model.config),
        "model_state": model.state_dict(),
        "ema_state": ema_state,
        "T": 1,
        "C": model.in_dim,
        "T_diff": len(alphas),
        "alphas_cumprod": alphas.detach().cpu(),
        "normalizer": x_normalizer.to_dict(),
        "pos_normalizer": pos_normalizer.to_dict(),
        "R6d_normalizer": None if R6d_normalizer is None else R6d_normalizer.to_dict(),
        "type_names": list(type_names),
        "cond_type": "unified_mask_xl",
        "n_params": sum(p.numel() for p in model.parameters()),
        "step": int(step),
        "robot": robot,
        "args": args or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)
