"""
Bloco B — Head Pose (12 dimensões).

Calcula os 3 ângulos de Euler da cabeça (yaw, pitch, roll) via solvePnP
do OpenCV, projetando um modelo 3D canônico de cabeça sobre 6 landmarks
2D do MediaPipe FaceMesh. Adiciona dinâmicas e features de translação.

Referência canônica do método: Mallick (2016), "Head Pose Estimation
using OpenCV and Dlib", LearnOpenCV.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python não está instalado. Rode: pip install opencv-python") from e

from .. import landmark_indices as L
from ..temporal_buffer import TemporalBuffer


BLOCK_B_NAMES: Tuple[str, ...] = (
    "B01_head_yaw",       "B02_head_pitch",     "B03_head_roll",
    "B04_head_yaw_abs",   "B05_head_pitch_abs", "B06_head_roll_abs",
    "B07_d_head_yaw",     "B08_d_head_pitch",   "B09_d_head_roll",
    "B10_head_translation_x", "B11_head_translation_y",
    "B12_face_size_ratio",
)
assert len(BLOCK_B_NAMES) == 12


# Modelo 3D canônico de cabeça (em milímetros, origem na ponta do nariz).
# Coordenadas-padrão amplamente utilizadas na literatura.
_MODEL_3D_POINTS: np.ndarray = np.array([
    (   0.0,    0.0,    0.0),   # nose_tip
    (   0.0, -330.0,  -65.0),   # chin
    (-225.0,  170.0, -135.0),   # left_eye_outer
    ( 225.0,  170.0, -135.0),   # right_eye_outer
    (-150.0, -150.0, -125.0),   # left_mouth_corner
    ( 150.0, -150.0, -125.0),   # right_mouth_corner
], dtype=np.float64)


def _build_camera_matrix(frame_w: int, frame_h: int) -> np.ndarray:
    """Matriz intrínseca aproximada — focal = largura do frame, óptico no centro."""
    focal = float(frame_w)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    return np.array([
        [focal,    0.0, cx],
        [   0.0, focal, cy],
        [   0.0,    0.0, 1.0],
    ], dtype=np.float64)


def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Converte matriz de rotação 3x3 em ângulos de Euler (graus): yaw, pitch, roll.
    Convenção: extrínsica XYZ, alinhada com cv2.RQDecomp3x3.
    """
    # Trick standard com sy = sqrt(R[0,0]^2 + R[1,0]^2)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:  # gimbal lock
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
    return float(yaw), float(pitch), float(roll)


def compute_block_b(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    buf: TemporalBuffer,
) -> List[float]:
    """
    Computa as 12 features do Bloco B.

    Args:
        landmarks: (478, 3) do FaceMesh.
        frame_width, frame_height: dimensões do frame em pixels.
        buf: buffer temporal para deltas B07-B09.

    Returns:
        Lista de 12 floats. Se solvePnP falhar, retorna zeros para os
        ângulos (mas continua calculando B10-B12).
    """
    # Converter landmarks normalizados [0,1] para coordenadas de pixel
    image_points = np.array([
        (landmarks[L.NOSE_TIP, 0]            * frame_width, landmarks[L.NOSE_TIP, 1]            * frame_height),
        (landmarks[L.CHIN_BOTTOM, 0]         * frame_width, landmarks[L.CHIN_BOTTOM, 1]         * frame_height),
        (landmarks[L.EYE_LEFT_OUTER_CORNER, 0]  * frame_width, landmarks[L.EYE_LEFT_OUTER_CORNER, 1]  * frame_height),
        (landmarks[L.EYE_RIGHT_OUTER_CORNER, 0] * frame_width, landmarks[L.EYE_RIGHT_OUTER_CORNER, 1] * frame_height),
        (landmarks[L.MOUTH_CORNER_LEFT, 0]   * frame_width, landmarks[L.MOUTH_CORNER_LEFT, 1]   * frame_height),
        (landmarks[L.MOUTH_CORNER_RIGHT, 0]  * frame_width, landmarks[L.MOUTH_CORNER_RIGHT, 1]  * frame_height),
    ], dtype=np.float64)

    camera_matrix = _build_camera_matrix(frame_width, frame_height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume sem distorção lente

    yaw = pitch = roll = 0.0
    success, rvec, _tvec = cv2.solvePnP(
        _MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if success:
        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = _rotation_matrix_to_euler(R)

    # B01-B06
    b01, b02, b03 = yaw, pitch, roll
    b04, b05, b06 = abs(yaw), abs(pitch), abs(roll)

    # B07-B09 deltas
    buf.append("B_yaw", b01)
    buf.append("B_pitch", b02)
    buf.append("B_roll", b03)
    b07 = buf.delta("B_yaw")
    b08 = buf.delta("B_pitch")
    b09 = buf.delta("B_roll")

    # B10-B11: translação aproximada via deslocamento do nariz vs. centro do frame
    nose_x = float(landmarks[L.NOSE_TIP, 0])
    nose_y = float(landmarks[L.NOSE_TIP, 1])
    b10 = nose_x - 0.5
    b11 = nose_y - 0.5

    # B12: face_size_ratio aproximado pela bounding-box dos landmarks faciais (xy)
    xy = landmarks[:468, :2]
    width  = float(xy[:, 0].max() - xy[:, 0].min())
    height = float(xy[:, 1].max() - xy[:, 1].min())
    b12 = width * height  # área em coords normalizadas [0..1]²

    return [b01, b02, b03, b04, b05, b06, b07, b08, b09, b10, b11, b12]
