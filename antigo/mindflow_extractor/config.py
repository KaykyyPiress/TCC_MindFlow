"""
MindFlow Extractor — Configurações globais.

Centraliza constantes utilizadas em todo o pipeline de extração:
- Dimensionalidade do vetor final (120d)
- Tamanhos de janelas temporais para dinâmicas
- Thresholds de eventos discretos (piscadas, sacadas, etc.)
- Configurações de captura

A separação destas constantes em um único módulo facilita ajustes
durante a fase de tuning empírico, prevista na metodologia do TCC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Dimensionalidade alvo do vetor unificado (Early Fusion)
# ---------------------------------------------------------------------------
BLOCK_DIMENSIONS: Dict[str, int] = {
    "A_facial_aus": 32,   # Action Units aproximadas
    "B_head_pose": 12,    # Yaw / Pitch / Roll + dinâmicas
    "C_gaze": 24,         # Estabilidade do olhar
    "D_eyes": 16,         # EAR + dinâmica ocular
    "E_mouth": 8,         # MAR + dinâmica da boca
    "F_pose": 24,         # Postura corporal
    "G_quality": 4,       # Meta-features de qualidade
}
VECTOR_DIM: int = sum(BLOCK_DIMENSIONS.values())  # = 120
assert VECTOR_DIM == 120, f"Vetor esperado=120d, obtido={VECTOR_DIM}d"


# ---------------------------------------------------------------------------
# Janelas temporais (em frames) para features dinâmicas
# Estas constantes refletem decisões explicadas na seção 2 do documento
# de especificação: múltiplas resoluções temporais entregues ao LSTM.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalWindows:
    short: int = 5    # variâncias locais (~165ms a 30fps)
    medium: int = 15  # estabilidade postural
    long: int = 30    # frequência de piscadas / sonolência (1s a 30fps)


WINDOWS = TemporalWindows()


# ---------------------------------------------------------------------------
# Thresholds para features binárias
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    # Soukupová & Čech (2016): EAR < 0.20 indica olho fechado
    # Subido para 0.22 para capturar pálpebras levemente caídas (sonolência)
    blink_ear: float = 0.22
    # Norma do delta do gaze acima da qual classifica-se sacada
    saccade_gaze_delta: float = 0.10
    # MAR acima do qual a boca é considerada aberta para bocejo
    # 0.65 exige abertura maior — evita falso positivo ao falar ou respirar
    mouth_open_mar: float = 0.65
    # Dist mão-rosto normalizada por d_ip; abaixo disso = oclusão
    hand_occlusion: float = 0.5
    # Visibility mínima do MediaPipe para considerar landmark confiável
    landmark_visibility: float = 0.5


THRESHOLDS = Thresholds()


# ---------------------------------------------------------------------------
# Configuração da captura
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureConfig:
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    # Numpy dtype usado na persistência do .npy final
    dtype: str = "float32"


CAPTURE = CaptureConfig()


# ---------------------------------------------------------------------------
# Sentinela para feature ausente
# ---------------------------------------------------------------------------
# Usamos NaN para sinalizar feature não calculável neste frame (ex.: pulso
# invisível na pose corporal). A substituição por zero ocorre apenas APÓS
# a Z-score, na etapa de pré-processamento prevista na metodologia.
import numpy as np
NAN: float = float(np.nan)
