"""
Bloco A — Features Faciais (32 dimensões).

Implementa as aproximações geométricas das Action Units (AUs) do sistema
FACS de Ekman, usando distâncias normalizadas entre subconjuntos de
landmarks do MediaPipe FaceMesh.

Mapeamento ID → significado: ver documento de especificação seção 3.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import landmark_indices as L
from ..temporal_buffer import TemporalBuffer
from .geometry import distance_2d, norm_by

# Nomes das 32 features na ordem do vetor — usado para serialização
# no metadata.json e para testes.
BLOCK_A_NAMES: Tuple[str, ...] = (
    "A01_brow_inner_raise_L",   "A02_brow_inner_raise_R",
    "A03_brow_outer_raise_L",   "A04_brow_outer_raise_R",
    "A05_brow_lower_L",         "A06_brow_lower_R",
    "A07_brow_distance",        "A08_upper_lid_raise_L",
    "A09_upper_lid_raise_R",    "A10_cheek_raise_L",
    "A11_cheek_raise_R",        "A12_lid_tighten_L",
    "A13_lid_tighten_R",        "A14_nose_wrinkle",
    "A15_lip_corner_pull_L",    "A16_lip_corner_pull_R",
    "A17_lip_corner_depress_L", "A18_lip_corner_depress_R",
    "A19_chin_raise",           "A20_lip_pucker",
    "A21_lip_stretch",          "A22_lip_press",
    "A23_jaw_drop",             "A24_face_symmetry_h",
    # Δ (dinâmicas) — calculadas usando o temporal_buffer
    "A25_d_brow_inner_raise",   "A26_d_brow_lower",
    "A27_d_cheek_raise",        "A28_d_lip_corner_pull",
    "A29_d_lip_corner_depress", "A30_d_jaw_drop",
    "A31_d_nose_wrinkle",       "A32_d_lip_press",
)
assert len(BLOCK_A_NAMES) == 32


def compute_block_a(
    landmarks: np.ndarray,
    d_ip: float,
    buf: TemporalBuffer,
) -> List[float]:
    """
    Computa as 32 features do Bloco A para o frame atual.

    Args:
        landmarks: array (478, 3) dos landmarks do FaceMesh.
        d_ip: distância interpupilar (escala de normalização).
        buf: buffer temporal compartilhado para deltas (A25-A32).

    Returns:
        Lista de 32 floats. Funções alimentam o buffer para uso pelos
        deltas em frames subsequentes.
    """
    # ----- Acesso rápido a coordenadas 2D usadas em múltiplas AUs -----
    def p(idx: int) -> np.ndarray:
        return landmarks[idx, :2]

    # ===== Estáticas (A01-A24) =====
    # AU1 — elevação interna das sobrancelhas (Glabela como referência)
    a01 = norm_by(distance_2d(p(L.BROW_LEFT_INNER),  p(L.GLABELLA)), d_ip)
    a02 = norm_by(distance_2d(p(L.BROW_RIGHT_INNER), p(L.GLABELLA)), d_ip)

    # AU2 — elevação externa das sobrancelhas
    a03 = norm_by(distance_2d(p(L.BROW_LEFT_OUTER),  p(L.EYE_LEFT_OUTER_CORNER)),  d_ip)
    a04 = norm_by(distance_2d(p(L.BROW_RIGHT_OUTER), p(L.EYE_RIGHT_OUTER_CORNER)), d_ip)

    # AU4 — abaixamento das sobrancelhas (sinal invertido: maior valor = mais abaixada)
    # Calculado como o NEGATIVO da distância (lm meio da sobrancelha a canto do olho).
    a05 = -norm_by(distance_2d(p(L.BROW_LEFT_MID),  p(L.EYE_LEFT_OUTER_CORNER)),  d_ip)
    a06 = -norm_by(distance_2d(p(L.BROW_RIGHT_MID), p(L.EYE_RIGHT_OUTER_CORNER)), d_ip)
    a07 = norm_by(distance_2d(p(L.BROW_LEFT_INNER), p(L.BROW_RIGHT_INNER)), d_ip)

    # AU5 — elevação da pálpebra superior
    a08 = norm_by(distance_2d(p(L.EYE_LEFT_TOP_1),  p(L.EYE_LEFT_BOTTOM_1)),  d_ip)
    a09 = norm_by(distance_2d(p(L.EYE_RIGHT_TOP_1), p(L.EYE_RIGHT_BOTTOM_1)), d_ip)

    # AU6 — elevação da bochecha (sorriso de Duchenne)
    a10 = norm_by(distance_2d(p(L.CHEEK_LEFT),  p(L.MOUTH_CORNER_LEFT)),  d_ip)
    a11 = norm_by(distance_2d(p(L.CHEEK_RIGHT), p(L.MOUTH_CORNER_RIGHT)), d_ip)

    # AU7 — pálpebras tensionadas: razão da altura do olho sobre largura
    # Usamos EAR específico: (altura)/(largura). Quanto MENOR, mais tensa.
    # Como queremos "maior valor = mais tenso", invertemos: 1 / (EAR + eps)
    eye_l_h = distance_2d(p(L.EYE_LEFT_TOP_1), p(L.EYE_LEFT_BOTTOM_1))
    eye_l_w = distance_2d(p(L.EYE_LEFT_OUTER_CORNER), p(L.EYE_LEFT_INNER_CORNER))
    a12 = norm_by(eye_l_w, eye_l_h)  # razão larg/alt → alta quando pálpebra tensa
    eye_r_h = distance_2d(p(L.EYE_RIGHT_TOP_1), p(L.EYE_RIGHT_BOTTOM_1))
    eye_r_w = distance_2d(p(L.EYE_RIGHT_OUTER_CORNER), p(L.EYE_RIGHT_INNER_CORNER))
    a13 = norm_by(eye_r_w, eye_r_h)

    # AU9 — enrugamento do nariz (encurtamento da ponte do nariz)
    a14 = norm_by(distance_2d(p(L.NOSE_BRIDGE_TOP), p(L.NOSE_BRIDGE_BOTTOM)), d_ip)

    # AU12 — elevação dos cantos da boca (sorriso). Sinal: y cresce para BAIXO
    # em coordenadas de imagem; portanto, "elevação" = canto da boca MENOR y
    # que o topo do lábio. Usamos diferença (lm 13 - lm 61 e 291) que dá
    # positivo quando o canto está acima do meio.
    a15 = norm_by(landmarks[L.MOUTH_TOP_INNER, 1] - landmarks[L.MOUTH_CORNER_LEFT, 1],  d_ip)
    a16 = norm_by(landmarks[L.MOUTH_TOP_INNER, 1] - landmarks[L.MOUTH_CORNER_RIGHT, 1], d_ip)

    # AU15 — depressão dos cantos (cantos abaixo do queixo-interno = lm 17)
    a17 = norm_by(landmarks[L.MOUTH_CORNER_LEFT, 1]  - landmarks[L.MOUTH_BOTTOM_OUTER, 1], d_ip)
    a18 = norm_by(landmarks[L.MOUTH_CORNER_RIGHT, 1] - landmarks[L.MOUTH_BOTTOM_OUTER, 1], d_ip)

    # AU17 — elevação do queixo (lábio inferior pressionado para cima)
    a19 = norm_by(distance_2d(p(L.CHIN_BOTTOM), p(L.MOUTH_BOTTOM_OUTER)), d_ip)

    # AU18 — lábios "biquinho" (largura horizontal da boca diminuída → inverter)
    mouth_w = distance_2d(p(L.MOUTH_CORNER_LEFT), p(L.MOUTH_CORNER_RIGHT))
    a20 = norm_by(d_ip, mouth_w)  # cresce quando a boca afina

    # AU20 — lábios "esticados" (largura horizontal aumentada)
    a21 = norm_by(mouth_w, d_ip)

    # AU24 — lábios pressionados (espaço interno da boca reduzido → inverter)
    mouth_inner = distance_2d(p(L.MOUTH_TOP_INNER), p(L.MOUTH_BOTTOM_INNER))
    a22 = norm_by(d_ip, mouth_inner)  # cresce quando os lábios juntam

    # AU26 — abertura da mandíbula
    a23 = norm_by(distance_2d(p(L.CHIN_BOTTOM), p(L.MOUTH_TOP_INNER)), d_ip)

    # A24 — simetria horizontal agregada: razão entre média esq./média dir.
    # Quando o valor está próximo de 1.0, rosto é simétrico; afastamento sugere
    # contração unilateral (ex.: meio-sorriso) ou rotação não compensada.
    left_avg  = (a01 + a03 + abs(a05) + a08 + a10 + a15 + a17) / 7.0
    right_avg = (a02 + a04 + abs(a06) + a09 + a11 + a16 + a18) / 7.0
    a24 = norm_by(left_avg, right_avg) if right_avg != 0 else 1.0

    statics = [a01, a02, a03, a04, a05, a06, a07, a08,
               a09, a10, a11, a12, a13, a14, a15, a16,
               a17, a18, a19, a20, a21, a22, a23, a24]

    # ===== Dinâmicas (A25-A32) — deltas vs. frame anterior =====
    # As fontes das dinâmicas são as 8 médias de pares L/R mais relevantes
    # para microexpressões.
    sources = {
        "A_brow_inner":         (a01 + a02) / 2.0,
        "A_brow_lower":         (a05 + a06) / 2.0,
        "A_cheek_raise":        (a10 + a11) / 2.0,
        "A_lip_corner_pull":    (a15 + a16) / 2.0,
        "A_lip_corner_depress": (a17 + a18) / 2.0,
        "A_jaw_drop":           a23,
        "A_nose_wrinkle":       a14,
        "A_lip_press":          a22,
    }
    deltas: List[float] = []
    for name, value in sources.items():
        buf.append(name, value)
        deltas.append(buf.delta(name))

    return statics + deltas
