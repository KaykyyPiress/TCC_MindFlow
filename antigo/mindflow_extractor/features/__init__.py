"""Blocos de features (A-G) do vetor de 120 dimensões."""
from .geometry import (
    distance,
    distance_2d,
    interpupillary_distance,
    norm_by,
)
from .block_a_facial_aus import compute_block_a, BLOCK_A_NAMES
from .block_b_head_pose import compute_block_b, BLOCK_B_NAMES
from .block_c_gaze import compute_block_c, BLOCK_C_NAMES
from .block_d_eyes import compute_block_d, BLOCK_D_NAMES, eye_aspect_ratio
from .block_e_mouth import compute_block_e, BLOCK_E_NAMES, mouth_aspect_ratio
from .block_f_pose import compute_block_f, BLOCK_F_NAMES
from .block_g_quality import compute_block_g, BLOCK_G_NAMES

__all__ = [
    "distance", "distance_2d", "interpupillary_distance", "norm_by",
    "compute_block_a", "BLOCK_A_NAMES",
    "compute_block_b", "BLOCK_B_NAMES",
    "compute_block_c", "BLOCK_C_NAMES",
    "compute_block_d", "BLOCK_D_NAMES", "eye_aspect_ratio",
    "compute_block_e", "BLOCK_E_NAMES", "mouth_aspect_ratio",
    "compute_block_f", "BLOCK_F_NAMES",
    "compute_block_g", "BLOCK_G_NAMES",
]
