"""Constraint-conditioned diffusion IK and the intrinsic geometry of its latent spaces (CoRL 2026)."""
from constraint_ik.checkpoint import load_checkpoint, save_checkpoint
from constraint_ik.constraints import (
    CONSTRAINTS, LOOSE, PAPER_ORDER, STRICT, ConstraintMap, Target, errors, is_valid,
)
from constraint_ik.pipeline import DiffusionIK
from constraint_ik.robots import Franka, UR5, get_robot

__version__ = "1.0.0"

__all__ = [
    "CONSTRAINTS", "PAPER_ORDER", "LOOSE", "STRICT", "Target", "ConstraintMap", "errors",
    "is_valid", "DiffusionIK", "load_checkpoint", "save_checkpoint", "UR5", "Franka", "get_robot",
]
