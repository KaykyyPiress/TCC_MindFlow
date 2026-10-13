"""
Bloco F — Postura Corporal (24 dimensões).

Usa MediaPipe Pose (33 landmarks). Em vídeos de webcam, tipicamente
apenas os landmarks 0..22 são visíveis. Landmarks com visibility baixa
recebem NaN nas features dependentes, conforme política descrita no
documento de especificação.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .. import landmark_indices as L
from ..config import NAN, THRESHOLDS, WINDOWS
from ..temporal_buffer import TemporalBuffer
from .geometry import distance_2d, norm_by


BLOCK_F_NAMES: Tuple[str, ...] = (
    "F01_shoulder_L_x", "F02_shoulder_L_y",
    "F03_shoulder_R_x", "F04_shoulder_R_y",
    "F05_shoulder_tilt",
    "F06_shoulder_width",
    "F07_head_offset_x", "F08_head_offset_y",
    "F09_torso_lean_forward",
    "F10_torso_lean_lateral",
    "F11_neck_angle",
    "F12_slouching_indicator",
    "F13_wrist_L_x", "F14_wrist_L_y",
    "F15_wrist_R_x", "F16_wrist_R_y",
    "F17_hand_to_face_L", "F18_hand_to_face_R",
    "F19_face_occlusion_proxy",
    "F20_d_head_pos_x", "F21_d_head_pos_y",
    "F22_d_shoulder_width",
    "F23_pose_stability_15f",
    "F24_pose_visibility_score",
)
assert len(BLOCK_F_NAMES) == 24


def _xy(landmarks: np.ndarray, idx: int) -> np.ndarray:
    return landmarks[idx, :2]


def _visible(landmarks: np.ndarray, idx: int) -> bool:
    return bool(landmarks[idx, 3] >= THRESHOLDS.landmark_visibility)


def compute_block_f(
    pose_landmarks: Optional[np.ndarray],
    d_ip_face: float,
    buf: TemporalBuffer,
) -> List[float]:
    """
    Computa as 24 features do Bloco F.

    Args:
        pose_landmarks: array (33, 4) com x, y, z, visibility. None se
                        nenhuma pose foi detectada.
        d_ip_face: distância interpupilar calculada do FaceMesh — reusada
                   como referência de escala para manter consistência com
                   os demais blocos.
        buf: buffer temporal.

    Returns:
        Lista de 24 floats. Quando pose ausente ou landmarks invisíveis,
        retorna NaN — substituído por zero apenas na etapa Z-score.
    """
    if pose_landmarks is None:
        return [NAN] * 24

    # F01-F04: posição dos ombros em coords normalizadas do frame
    ok_sL = _visible(pose_landmarks, L.POSE_LEFT_SHOULDER)
    ok_sR = _visible(pose_landmarks, L.POSE_RIGHT_SHOULDER)

    sL = _xy(pose_landmarks, L.POSE_LEFT_SHOULDER) if ok_sL else None
    sR = _xy(pose_landmarks, L.POSE_RIGHT_SHOULDER) if ok_sR else None
    f01 = float(sL[0]) if ok_sL else NAN
    f02 = float(sL[1]) if ok_sL else NAN
    f03 = float(sR[0]) if ok_sR else NAN
    f04 = float(sR[1]) if ok_sR else NAN

    # F05: shoulder_tilt em graus
    if ok_sL and ok_sR:
        dx = float(sR[0] - sL[0])
        dy = float(sR[1] - sL[1])
        f05 = float(math.degrees(math.atan2(dy, dx)))
        f06 = norm_by(distance_2d(sL, sR), d_ip_face)
        mid_shoulders = (sL + sR) / 2.0
    else:
        f05 = NAN
        f06 = NAN
        mid_shoulders = None

    # F07-F08: head_offset (nose vs. midpoint dos ombros)
    ok_nose = _visible(pose_landmarks, L.POSE_NOSE)
    if ok_nose and mid_shoulders is not None:
        nose = _xy(pose_landmarks, L.POSE_NOSE)
        f07 = norm_by(float(nose[0] - mid_shoulders[0]), d_ip_face)
        f08 = norm_by(float(nose[1] - mid_shoulders[1]), d_ip_face)
    else:
        f07 = NAN
        f08 = NAN

    # F09: torso_lean_forward — proxy via razão (largura dos ombros / d_ip).
    # Sujeito inclinado pra frente tende a aumentar visualmente o tamanho da
    # cabeça e diminuir a separação visual dos ombros; razão d_ip/largura cresce.
    if not np.isnan(f06) and f06 > 0:
        f09 = norm_by(1.0, f06)
    else:
        f09 = NAN

    # F10: torso_lean_lateral — F07 (já em unidades d_ip)
    f10 = f07

    # F11: neck_angle — ângulo entre vetor (mid_shoulders → nose) e a vertical
    if ok_nose and mid_shoulders is not None:
        nose = _xy(pose_landmarks, L.POSE_NOSE)
        vec = nose - mid_shoulders
        # vertical (apontando para cima) = (0, -1) em coords de imagem
        # ângulo positivo = pescoço inclinado para um lado
        f11 = float(math.degrees(math.atan2(vec[0], -vec[1])))
    else:
        f11 = NAN

    # F12: slouching_indicator — combinação ponderada (mais alto = mais slouching)
    if not (np.isnan(f09) or np.isnan(f11)):
        f12 = 0.5 * f09 + 0.5 * (abs(f11) / 30.0)  # 30° como normalização
    else:
        f12 = NAN

    # F13-F16: pulsos (frequentemente NaN no DAiSEE)
    if _visible(pose_landmarks, L.POSE_LEFT_WRIST):
        wL = _xy(pose_landmarks, L.POSE_LEFT_WRIST)
        f13, f14 = float(wL[0]), float(wL[1])
    else:
        wL = None
        f13, f14 = NAN, NAN
    if _visible(pose_landmarks, L.POSE_RIGHT_WRIST):
        wR = _xy(pose_landmarks, L.POSE_RIGHT_WRIST)
        f15, f16 = float(wR[0]), float(wR[1])
    else:
        wR = None
        f15, f16 = NAN, NAN

    # F17-F18: distância pulso → nariz
    if wL is not None and ok_nose:
        f17 = norm_by(distance_2d(wL, _xy(pose_landmarks, L.POSE_NOSE)), d_ip_face)
    else:
        f17 = NAN
    if wR is not None and ok_nose:
        f18 = norm_by(distance_2d(wR, _xy(pose_landmarks, L.POSE_NOSE)), d_ip_face)
    else:
        f18 = NAN

    # F19: face_occlusion_proxy — pelo menos uma mão muito próxima ao rosto
    occlusion = False
    if not np.isnan(f17) and f17 < THRESHOLDS.hand_occlusion:
        occlusion = True
    if not np.isnan(f18) and f18 < THRESHOLDS.hand_occlusion:
        occlusion = True
    f19 = 1.0 if occlusion else 0.0

    # F20-F22: dinâmicas
    # Para deltas usamos buffer; valores NaN ficam fora do buffer (não alimentam).
    if not np.isnan(f07):
        buf.append("F_head_x", f07)
        f20 = buf.delta("F_head_x")
    else:
        f20 = NAN
    if not np.isnan(f08):
        buf.append("F_head_y", f08)
        f21 = buf.delta("F_head_y")
    else:
        f21 = NAN
    if not np.isnan(f06):
        buf.append("F_shoulder_w", f06)
        f22 = buf.delta("F_shoulder_w")
    else:
        f22 = NAN

    # F23: estabilidade postural — variância média dos ombros em janela 15f
    if ok_sL and ok_sR:
        buf.append("F_sL_x", float(sL[0]))
        buf.append("F_sR_x", float(sR[0]))
        sigma_lx = buf.variance("F_sL_x", WINDOWS.medium)
        sigma_rx = buf.variance("F_sR_x", WINDOWS.medium)
        f23 = (sigma_lx + sigma_rx) / 2.0
    else:
        f23 = NAN

    # F24: pose_visibility_score — média das visibility dos lm 0..22
    visibilities = pose_landmarks[list(L.POSE_VISIBLE_SUBSET), 3]
    f24 = float(visibilities.mean())

    return [f01, f02, f03, f04, f05, f06, f07, f08,
            f09, f10, f11, f12, f13, f14, f15, f16,
            f17, f18, f19, f20, f21, f22, f23, f24]
