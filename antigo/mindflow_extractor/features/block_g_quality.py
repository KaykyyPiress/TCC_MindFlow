"""
Bloco G — Qualidade do Sinal (4 dimensões).

Meta-features que sinalizam ao LSTM a confiabilidade da detecção no frame.
Permitem graceful degradation em vez de descarte de frames.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import THRESHOLDS


BLOCK_G_NAMES: Tuple[str, ...] = (
    "G01_face_detection_confidence",
    "G02_pose_detection_confidence",
    "G03_brightness_estimate",
    "G04_face_visible_ratio",
)
assert len(BLOCK_G_NAMES) == 4


def compute_block_g(
    face_landmarks: Optional[np.ndarray],
    face_confidence: float,
    pose_confidence: float,
    frame_rgb: np.ndarray,
) -> List[float]:
    """
    Computa as 4 meta-features.

    Args:
        face_landmarks: (478, 3) ou None.
        face_confidence: confidence aproximada do FaceMesh.
        pose_confidence: confidence média de visibility da pose.
        frame_rgb: frame original (H, W, 3) em RGB, usado para luminância.
    """
    g01 = float(face_confidence)
    g02 = float(pose_confidence)

    # G03 — brightness: média de luminância da bounding box do rosto.
    # Conversão RGB → luminância pela fórmula ITU-R BT.601 (Y = 0.299 R + 0.587 G + 0.114 B).
    if face_landmarks is None:
        # sem rosto: luminância média do frame inteiro
        lum = (0.299 * frame_rgb[:, :, 0] +
               0.587 * frame_rgb[:, :, 1] +
               0.114 * frame_rgb[:, :, 2])
        g03 = float(lum.mean() / 255.0)
    else:
        h, w = frame_rgb.shape[:2]
        xs = face_landmarks[:468, 0]
        ys = face_landmarks[:468, 1]
        x_min = int(max(0, xs.min() * w))
        x_max = int(min(w - 1, xs.max() * w))
        y_min = int(max(0, ys.min() * h))
        y_max = int(min(h - 1, ys.max() * h))
        if x_max <= x_min or y_max <= y_min:
            g03 = 0.5
        else:
            roi = frame_rgb[y_min:y_max + 1, x_min:x_max + 1]
            lum = (0.299 * roi[:, :, 0] +
                   0.587 * roi[:, :, 1] +
                   0.114 * roi[:, :, 2])
            g03 = float(lum.mean() / 255.0)

    # G04 — fração de landmarks faciais dentro do frame normalizado [0,1]
    if face_landmarks is None:
        g04 = 0.0
    else:
        xy = face_landmarks[:468, :2]
        in_frame = ((xy >= 0.0) & (xy <= 1.0)).all(axis=1)
        g04 = float(in_frame.mean())

    return [g01, g02, g03, g04]
