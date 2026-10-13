"""
Índices canônicos dos landmarks do MediaPipe utilizados pelas features.

Esta separação evita "magic numbers" espalhados pelo código de features.
Os índices seguem a documentação oficial do MediaPipe FaceMesh (468 lm +
10 lm de íris quando refine_landmarks=True) e MediaPipe Pose (33 lm).

Referências canônicas:
- FaceMesh: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Pose:     https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

IMPORTANTE: estes índices são uma referência inicial baseada em listas
publicadas pela comunidade do MediaPipe. Durante a validação visual via
webcam (notebook 01), eles serão conferidos um a um sobreposicionando os
pontos no rosto. Qualquer ajuste deve ser feito apenas aqui.
"""
from __future__ import annotations

from typing import Tuple


# ===========================================================================
# FACE — 468 landmarks (índices 0..467) + 10 de íris (468..477)
# ===========================================================================

# --- Olhos: contornos para cálculo do EAR (Soukupová & Čech, 2016) ---
# Convenção do EAR: 6 pontos por olho (4 verticais + 2 horizontais)
# Olho ESQUERDO (do ponto de vista do sujeito, à direita na imagem)
EYE_LEFT_OUTER_CORNER: int = 33     # canto externo
EYE_LEFT_INNER_CORNER: int = 133    # canto interno
EYE_LEFT_TOP_1: int = 159           # pálpebra superior (medial)
EYE_LEFT_TOP_2: int = 158           # pálpebra superior (lateral)
EYE_LEFT_BOTTOM_1: int = 145        # pálpebra inferior (medial)
EYE_LEFT_BOTTOM_2: int = 153        # pálpebra inferior (lateral)

EYE_LEFT_EAR_INDICES: Tuple[int, int, int, int, int, int] = (
    EYE_LEFT_OUTER_CORNER,
    EYE_LEFT_TOP_2,
    EYE_LEFT_TOP_1,
    EYE_LEFT_INNER_CORNER,
    EYE_LEFT_BOTTOM_1,
    EYE_LEFT_BOTTOM_2,
)

# Olho DIREITO
EYE_RIGHT_OUTER_CORNER: int = 263
EYE_RIGHT_INNER_CORNER: int = 362
EYE_RIGHT_TOP_1: int = 386
EYE_RIGHT_TOP_2: int = 385
EYE_RIGHT_BOTTOM_1: int = 374
EYE_RIGHT_BOTTOM_2: int = 380

EYE_RIGHT_EAR_INDICES: Tuple[int, int, int, int, int, int] = (
    EYE_RIGHT_OUTER_CORNER,
    EYE_RIGHT_TOP_2,
    EYE_RIGHT_TOP_1,
    EYE_RIGHT_INNER_CORNER,
    EYE_RIGHT_BOTTOM_1,
    EYE_RIGHT_BOTTOM_2,
)

# --- Íris (10 lm adicionais com refine_landmarks=True) ---
# Cada íris tem 5 pontos: 1 centro + 4 periféricos (N, E, S, W)
IRIS_LEFT_CENTER: int = 468
IRIS_LEFT_RIGHT: int = 469
IRIS_LEFT_TOP: int = 470
IRIS_LEFT_LEFT: int = 471
IRIS_LEFT_BOTTOM: int = 472

IRIS_RIGHT_CENTER: int = 473
IRIS_RIGHT_RIGHT: int = 474
IRIS_RIGHT_TOP: int = 475
IRIS_RIGHT_LEFT: int = 476
IRIS_RIGHT_BOTTOM: int = 477

# --- Sobrancelhas ---
BROW_LEFT_INNER: int = 55     # ponta interna sobrancelha esq.
BROW_LEFT_OUTER: int = 70     # ponta externa sobrancelha esq.
BROW_LEFT_MID: int = 107      # meio sobrancelha esq. (para AU4)

BROW_RIGHT_INNER: int = 285
BROW_RIGHT_OUTER: int = 300
BROW_RIGHT_MID: int = 336

# --- Glabela / centro entre sobrancelhas (referência para AU1) ---
GLABELLA: int = 8

# --- Bochechas (para AU6) ---
CHEEK_LEFT: int = 116
CHEEK_RIGHT: int = 345

# --- Boca ---
MOUTH_CORNER_LEFT: int = 61
MOUTH_CORNER_RIGHT: int = 291
MOUTH_TOP_OUTER: int = 0          # topo do lábio superior (vermelhão)
MOUTH_TOP_INNER: int = 13         # topo do lábio superior interno
MOUTH_BOTTOM_INNER: int = 14      # base do lábio inferior interno
MOUTH_BOTTOM_OUTER: int = 17      # base do lábio inferior (queixo-pegado)

# Para o MAR (Mouth Aspect Ratio) — análogo ao EAR
# Verticais (alturas) e horizontais (largura)
MAR_VERTICAL_PAIRS = (
    (13, 14),    # vertical central
    (37, 84),    # vertical esquerda
    (267, 314),  # vertical direita
)
MAR_HORIZONTAL_PAIR = (MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT)

# --- Queixo / mandíbula ---
CHIN_BOTTOM: int = 152
NOSE_TIP: int = 4
NOSE_BRIDGE_TOP: int = 6
NOSE_BRIDGE_BOTTOM: int = 197

# --- Subconjuntos para solvePnP (head pose) ---
# 6 pontos do modelo 3D canônico de cabeça, conforme a convenção clássica
# do OpenCV (Mallick, 2016).
HEAD_POSE_INDICES = {
    "nose_tip": NOSE_TIP,
    "chin": CHIN_BOTTOM,
    "left_eye_outer": EYE_LEFT_OUTER_CORNER,
    "right_eye_outer": EYE_RIGHT_OUTER_CORNER,
    "left_mouth_corner": MOUTH_CORNER_LEFT,
    "right_mouth_corner": MOUTH_CORNER_RIGHT,
}


# ===========================================================================
# POSE — 33 landmarks (índices 0..32). Em webcam só 0..22 costumam aparecer.
# ===========================================================================

POSE_NOSE: int = 0
POSE_LEFT_EYE_INNER: int = 1
POSE_LEFT_EYE: int = 2
POSE_LEFT_EYE_OUTER: int = 3
POSE_RIGHT_EYE_INNER: int = 4
POSE_RIGHT_EYE: int = 5
POSE_RIGHT_EYE_OUTER: int = 6
POSE_LEFT_EAR: int = 7
POSE_RIGHT_EAR: int = 8

POSE_LEFT_SHOULDER: int = 11
POSE_RIGHT_SHOULDER: int = 12
POSE_LEFT_ELBOW: int = 13
POSE_RIGHT_ELBOW: int = 14
POSE_LEFT_WRIST: int = 15
POSE_RIGHT_WRIST: int = 16

# Conjunto utilizado para o pose_visibility_score (F24)
POSE_VISIBLE_SUBSET = tuple(range(0, 23))
