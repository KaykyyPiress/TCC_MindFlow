"""
Bloco C — Estabilidade do Olhar (24 dimensões).

Estima direção e estabilidade do olhar a partir dos landmarks de íris
(468-477, disponíveis quando refine_landmarks=True) e dos cantos dos
olhos. Não há calibração por usuário — alinhado à proposta in-the-wild
do MindFlow.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .. import landmark_indices as L
from ..config import THRESHOLDS, WINDOWS
from ..temporal_buffer import TemporalBuffer
from .geometry import distance_2d, norm_by, EPS


BLOCK_C_NAMES: Tuple[str, ...] = (
    "C01_gaze_vector_x_L",  "C02_gaze_vector_y_L",
    "C03_gaze_vector_x_R",  "C04_gaze_vector_y_R",
    "C05_gaze_convergence",
    "C06_d_gaze_x_L",       "C07_d_gaze_y_L",
    "C08_d_gaze_x_R",       "C09_d_gaze_y_R",
    "C10_gaze_var_x_L_5f",  "C11_gaze_var_y_L_5f",
    "C12_gaze_var_x_R_5f",  "C13_gaze_var_y_R_5f",
    "C14_saccade_indicator",
    "C15_fixation_duration",
    "C16_gaze_screen_x",    "C17_gaze_screen_y",
    "C18_gaze_asymmetry_x", "C19_gaze_asymmetry_y",
    "C20_iris_size_L",      "C21_iris_size_R",
    "C22_gaze_angular_velocity",
    "C23_gaze_off_screen",
    "C24_gaze_stability_score",
)
assert len(BLOCK_C_NAMES) == 24


def _eye_gaze_vector(
    landmarks: np.ndarray,
    iris_center_idx: int,
    eye_outer_idx: int,
    eye_inner_idx: int,
) -> Tuple[float, float, float]:
    """
    Vetor de olhar normalizado para um olho.

    Returns:
        (gaze_x, gaze_y, eye_width). gaze_x e gaze_y são as componentes
        do vetor (centro_íris - centro_geométrico_do_olho) normalizadas
        pela largura do olho (centrado em zero para olhar reto).
    """
    iris_c = landmarks[iris_center_idx, :2]
    outer  = landmarks[eye_outer_idx,  :2]
    inner  = landmarks[eye_inner_idx,  :2]
    eye_center = (outer + inner) / 2.0
    eye_width  = max(distance_2d(outer, inner), EPS)
    gx = float((iris_c[0] - eye_center[0]) / eye_width)
    gy = float((iris_c[1] - eye_center[1]) / eye_width)
    return gx, gy, eye_width


def _iris_area(landmarks: np.ndarray, peripheral_idxs: Tuple[int, int, int, int]) -> float:
    """Área aproximada da íris pelos 4 landmarks periféricos (norte, leste, sul, oeste)."""
    pts = landmarks[list(peripheral_idxs), :2]
    width  = distance_2d(pts[1], pts[3])
    height = distance_2d(pts[0], pts[2])
    return float(width * height)


def compute_block_c(
    landmarks: np.ndarray,
    d_ip: float,
    buf: TemporalBuffer,
) -> List[float]:
    """Computa as 24 features do Bloco C."""

    # Vetores de olhar por olho (C01-C04)
    gx_l, gy_l, _ = _eye_gaze_vector(
        landmarks, L.IRIS_LEFT_CENTER,
        L.EYE_LEFT_OUTER_CORNER, L.EYE_LEFT_INNER_CORNER,
    )
    gx_r, gy_r, _ = _eye_gaze_vector(
        landmarks, L.IRIS_RIGHT_CENTER,
        L.EYE_RIGHT_OUTER_CORNER, L.EYE_RIGHT_INNER_CORNER,
    )

    # C05 — convergência (distância entre íris normalizada por d_ip)
    iris_dist = distance_2d(
        landmarks[L.IRIS_LEFT_CENTER, :2],
        landmarks[L.IRIS_RIGHT_CENTER, :2],
    )
    c05 = norm_by(iris_dist, d_ip)

    # Alimenta buffer e calcula deltas (C06-C09) e variâncias (C10-C13)
    buf.append("C_gx_L", gx_l)
    buf.append("C_gy_L", gy_l)
    buf.append("C_gx_R", gx_r)
    buf.append("C_gy_R", gy_r)

    c06 = buf.delta("C_gx_L")
    c07 = buf.delta("C_gy_L")
    c08 = buf.delta("C_gx_R")
    c09 = buf.delta("C_gy_R")

    c10 = buf.variance("C_gx_L", WINDOWS.short)
    c11 = buf.variance("C_gy_L", WINDOWS.short)
    c12 = buf.variance("C_gx_R", WINDOWS.short)
    c13 = buf.variance("C_gy_R", WINDOWS.short)

    # C14 — saccade indicator: norma do delta médio acima de threshold
    delta_norm = math.sqrt(
        ((c06 + c08) / 2.0) ** 2 + ((c07 + c09) / 2.0) ** 2
    )
    c14 = 1.0 if delta_norm > THRESHOLDS.saccade_gaze_delta else 0.0

    # C15 — fixation_duration: frames consecutivos SEM sacada.
    # Usamos a inversa do indicator no buffer.
    buf.append("C_no_saccade", 1.0 - c14)
    c15 = float(buf.consecutive_true("C_no_saccade"))

    # C16-C17 — projeção estimada do gaze na tela (mapeamento linear -0.5..0.5)
    gx_mean = (gx_l + gx_r) / 2.0
    gy_mean = (gy_l + gy_r) / 2.0
    c16 = gx_mean + 0.5
    c17 = gy_mean + 0.5

    # C18-C19 — assimetria binocular
    c18 = abs(gx_l - gx_r)
    c19 = abs(gy_l - gy_r)

    # C20-C21 — iris_size (proxy de dilatação pupilar, conhecidamente ruidoso)
    c20 = norm_by(
        _iris_area(landmarks, (L.IRIS_LEFT_TOP, L.IRIS_LEFT_RIGHT,
                               L.IRIS_LEFT_BOTTOM, L.IRIS_LEFT_LEFT)),
        d_ip ** 2,
    )
    c21 = norm_by(
        _iris_area(landmarks, (L.IRIS_RIGHT_TOP, L.IRIS_RIGHT_RIGHT,
                               L.IRIS_RIGHT_BOTTOM, L.IRIS_RIGHT_LEFT)),
        d_ip ** 2,
    )

    # C22 — velocidade angular agregada (em "graus" — convertida via aprox.
    # de 1 unidade normalizada ≈ 30° de campo visual, heurística simples)
    c22 = float(delta_norm * 30.0)

    # C23 — gaze off-screen
    c23 = 1.0 if (c16 < 0.0 or c16 > 1.0 or c17 < 0.0 or c17 > 1.0) else 0.0

    # C24 — score de estabilidade em janela 15f: 1 - σ médio do gaze
    sigma_x = buf.variance("C_gx_L", WINDOWS.medium) + buf.variance("C_gx_R", WINDOWS.medium)
    sigma_y = buf.variance("C_gy_L", WINDOWS.medium) + buf.variance("C_gy_R", WINDOWS.medium)
    sigma_total = (sigma_x + sigma_y) / 4.0  # média dos 4
    c24 = float(max(0.0, 1.0 - sigma_total))

    return [
        gx_l, gy_l, gx_r, gy_r,         # C01-C04
        c05,                            # C05
        c06, c07, c08, c09,             # C06-C09
        c10, c11, c12, c13,             # C10-C13
        c14, c15, c16, c17,             # C14-C17
        c18, c19, c20, c21,             # C18-C21
        c22, c23, c24,                  # C22-C24
    ]
