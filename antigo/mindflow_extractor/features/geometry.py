"""
Utilitários geométricos compartilhados pelos blocos de features.

Centraliza:
- Cálculo de distâncias 2D/3D.
- Distância interpupilar (d_ip) — referência canônica de normalização
  intra-frame (seção 2 do documento de especificação).
- Funções de normalização e segurança numérica.
"""
from __future__ import annotations

import numpy as np

from ..landmark_indices import IRIS_LEFT_CENTER, IRIS_RIGHT_CENTER

# Epsilon para evitar divisão por zero
EPS: float = 1e-8


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distância euclidiana entre dois pontos (suporta 2D ou 3D)."""
    return float(np.linalg.norm(p1 - p2))


def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distância euclidiana usando apenas (x, y), ignorando z."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def interpupillary_distance(landmarks: np.ndarray) -> float:
    """
    Calcula a distância interpupilar (d_ip) em coordenadas 2D.

    Convenção: usamos os centros das íris (IRIS_LEFT_CENTER / IRIS_RIGHT_CENTER)
    quando refine_landmarks=True. Esta é a referência métrica canônica para
    normalização intra-frame de todas as features de distância.

    Justificativa (seção 2 do documento):
    A distância interpupilar é uma constante anatômica relativamente estável
    em adultos (~63mm), tornando-a a referência ideal para cancelar o efeito
    da distância sujeito-câmera no DAiSEE in-the-wild.

    Returns:
        d_ip em unidades normalizadas (mesmo sistema dos landmarks).
        Retorna EPS se for inferior a EPS (caso degenerado).
    """
    left = landmarks[IRIS_LEFT_CENTER, :2]
    right = landmarks[IRIS_RIGHT_CENTER, :2]
    d = float(np.linalg.norm(left - right))
    return max(d, EPS)


def norm_by(value: float, denominator: float) -> float:
    """
    Divisão com proteção contra zero.

    Args:
        value: numerador.
        denominator: denominador (será limitado a EPS se < EPS).

    Returns:
        value / max(denominator, EPS), como float Python.
    """
    return float(value / max(denominator, EPS))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Ângulo (em graus) entre dois vetores 2D, usando atan2 para preservar sinal.
    Útil para inclinações (ex.: F05 shoulder_tilt).
    """
    # atan2(y2-y1 cross, dot) entre v1 e v2 — sinal preservado
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = float(np.dot(v1, v2))
    return float(np.degrees(np.arctan2(cross, dot)))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Ponto médio entre p1 e p2 (preserva dimensão de entrada)."""
    return (p1 + p2) / 2.0
