"""
Bloco E — Estado da Boca (8 dimensões).

Implementa o Mouth Aspect Ratio (MAR), análogo do EAR para boca, com
features para detecção de bocejo e dinâmica de abertura.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import landmark_indices as L
from ..config import THRESHOLDS, WINDOWS
from ..temporal_buffer import TemporalBuffer
from .geometry import distance_2d, norm_by, EPS


BLOCK_E_NAMES: Tuple[str, ...] = (
    "E01_mar",                "E02_d_mar",
    "E03_mar_var_5f",         "E04_mouth_open_indicator",
    "E05_mouth_open_duration", "E06_mouth_width",
    "E07_mouth_compression",  "E08_yawn_proxy",
)
assert len(BLOCK_E_NAMES) == 8


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """
    MAR = média(distâncias verticais) / distância horizontal.
    Análogo direto do EAR adaptado para boca.
    """
    vert_sum = 0.0
    for top_idx, bot_idx in L.MAR_VERTICAL_PAIRS:
        vert_sum += distance_2d(landmarks[top_idx, :2], landmarks[bot_idx, :2])
    vert_avg = vert_sum / len(L.MAR_VERTICAL_PAIRS)
    horiz = max(distance_2d(
        landmarks[L.MAR_HORIZONTAL_PAIR[0], :2],
        landmarks[L.MAR_HORIZONTAL_PAIR[1], :2],
    ), EPS)
    return float(vert_avg / horiz)


def compute_block_e(
    landmarks: np.ndarray,
    d_ip: float,
    buf: TemporalBuffer,
    jaw_drop_a23: float,
) -> List[float]:
    """
    Computa as 8 features do Bloco E.

    Args:
        jaw_drop_a23: valor da feature A23 (jaw_drop) já computada pelo
                      Bloco A, reusada na heurística de bocejo (E08).
    """
    e01 = mouth_aspect_ratio(landmarks)
    buf.append("E_mar", e01)

    e02 = buf.delta("E_mar")
    e03 = buf.variance("E_mar", WINDOWS.short)

    e04 = 1.0 if e01 > THRESHOLDS.mouth_open_mar else 0.0
    buf.append("E_mouth_open", e04)
    e05 = float(buf.consecutive_true("E_mouth_open"))

    mouth_w = distance_2d(
        landmarks[L.MOUTH_CORNER_LEFT, :2],
        landmarks[L.MOUTH_CORNER_RIGHT, :2],
    )
    e06 = norm_by(mouth_w, d_ip)

    mouth_h = distance_2d(
        landmarks[L.MOUTH_TOP_INNER, :2],
        landmarks[L.MOUTH_BOTTOM_INNER, :2],
    )
    e07 = norm_by(mouth_w, mouth_h)

    # E08 — yawn_proxy: boca aberta sustentada (>=15f) E jaw_drop alto
    # 15f a 30fps ≈ 0.5s, mais exigente para evitar falso positivo ao falar.
    e08 = 1.0 if (e05 >= 15 and jaw_drop_a23 > 0.6) else 0.0

    return [e01, e02, e03, e04, e05, e06, e07, e08]
