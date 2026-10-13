"""
Bloco D — Estado Ocular (16 dimensões).

Implementa o Eye Aspect Ratio (Soukupová & Čech, 2016) e suas dinâmicas
para detecção de piscadas, sonolência e estabilidade de abertura ocular.

A função eye_aspect_ratio é exportada para reuso em outros módulos
(ex.: validação visual no notebook 01).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import landmark_indices as L
from ..config import THRESHOLDS, WINDOWS
from ..temporal_buffer import TemporalBuffer
from .geometry import distance_2d, norm_by, EPS


BLOCK_D_NAMES: Tuple[str, ...] = (
    "D01_ear_L",            "D02_ear_R",          "D03_ear_mean",
    "D04_ear_asymmetry",    "D05_d_ear_L",        "D06_d_ear_R",
    "D07_ear_var_5f",       "D08_blink_indicator",
    "D09_frames_since_blink",
    "D10_eye_openness_score",
    "D11_upper_lid_pos_L",  "D12_upper_lid_pos_R",
    "D13_lower_lid_pos_L",  "D14_lower_lid_pos_R",
    "D15_drowsiness_proxy", "D16_ear_history_var_30f",
)
assert len(BLOCK_D_NAMES) == 16


def eye_aspect_ratio(
    landmarks: np.ndarray,
    indices: Tuple[int, int, int, int, int, int],
) -> float:
    """
    Eye Aspect Ratio segundo Soukupová & Čech (2016).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||)

    onde p1..p6 são os 6 pontos do olho na convenção do paper:
    - p1 = canto externo  - p4 = canto interno
    - p2, p3 = pálpebra superior   - p5, p6 = pálpebra inferior

    A função recebe a tupla de índices na ordem (p1, p2, p3, p4, p5, p6).
    """
    p1, p2, p3, p4, p5, p6 = (landmarks[i, :2] for i in indices)
    vertical_1 = distance_2d(p2, p6)
    vertical_2 = distance_2d(p3, p5)
    horizontal = max(distance_2d(p1, p4), EPS)
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_block_d(
    landmarks: np.ndarray,
    d_ip: float,
    buf: TemporalBuffer,
) -> List[float]:
    """Computa as 16 features do Bloco D."""

    d01 = eye_aspect_ratio(landmarks, L.EYE_LEFT_EAR_INDICES)
    d02 = eye_aspect_ratio(landmarks, L.EYE_RIGHT_EAR_INDICES)
    d03 = (d01 + d02) / 2.0
    d04 = abs(d01 - d02)

    # Deltas (D05-D06) e variância 5f (D07)
    buf.append("D_ear_L", d01)
    buf.append("D_ear_R", d02)
    buf.append("D_ear_mean", d03)
    d05 = buf.delta("D_ear_L")
    d06 = buf.delta("D_ear_R")
    d07 = buf.variance("D_ear_mean", WINDOWS.short)

    # D08 — blink indicator (EAR médio abaixo do threshold)
    d08 = 1.0 if d03 < THRESHOLDS.blink_ear else 0.0
    buf.append("D_blink", d08)

    # D09 — frames desde a última piscada.
    # Implementação: contamos quantos frames consecutivos blink == 0.
    buf.append("D_no_blink", 1.0 - d08)
    d09 = float(buf.consecutive_true("D_no_blink"))

    # D10 — eye_openness_score: razão em relação ao baseline do sujeito.
    # Baseline = mediana do EAR nos últimos frames da janela longa (30f),
    # filtrando piscadas. Em sessão curta, default = d03 (gera score ≈ 1.0).
    d10 = norm_by(d03, _estimate_baseline_ear(buf))

    # D11-D14 — posições verticais das pálpebras vs. iris center, normalizadas
    # Olho esquerdo: pálpebra superior (lm 159) e inferior (lm 145) vs. íris.
    iris_L = landmarks[L.IRIS_LEFT_CENTER, :2]
    iris_R = landmarks[L.IRIS_RIGHT_CENTER, :2]
    upper_L = landmarks[L.EYE_LEFT_TOP_1, :2]
    lower_L = landmarks[L.EYE_LEFT_BOTTOM_1, :2]
    upper_R = landmarks[L.EYE_RIGHT_TOP_1, :2]
    lower_R = landmarks[L.EYE_RIGHT_BOTTOM_1, :2]

    d11 = norm_by(abs(upper_L[1] - iris_L[1]), d_ip)
    d12 = norm_by(abs(upper_R[1] - iris_R[1]), d_ip)
    d13 = norm_by(abs(lower_L[1] - iris_L[1]), d_ip)
    d14 = norm_by(abs(lower_R[1] - iris_R[1]), d_ip)

    # D15 — drowsiness_proxy: fração de frames com olho fechado em janela 30f
    d15 = buf.fraction_true("D_blink", WINDOWS.long)

    # D16 — variância EAR em janela longa
    d16 = buf.variance("D_ear_mean", WINDOWS.long)

    return [d01, d02, d03, d04, d05, d06, d07, d08,
            d09, d10, d11, d12, d13, d14, d15, d16]


def _estimate_baseline_ear(buf: TemporalBuffer) -> float:
    """
    Estima EAR baseline do sujeito a partir do histórico recente.
    Usa o 75º percentil (favorece momentos com olho aberto), removendo
    o viés de piscadas frequentes.
    """
    history = buf._history.get("D_ear_mean")
    if history is None or len(history) < 5:
        return 0.25  # valor canônico de EAR aberto na literatura
    arr = np.array(list(history), dtype=np.float32)
    return float(max(np.percentile(arr, 75), 0.10))
