"""
Pipeline orquestrador: frame → vetor de 120 dimensões.

Conecta os wrappers do MediaPipe (face + pose) aos 7 blocos de features
e produz, a cada frame, um único np.ndarray de shape (120,) — pronto para
Early Fusion + Normalização Z-score + SMOTE + LSTM.

Este é o coração do módulo. Ele NÃO faz I/O de vídeo; apenas processa
frames recebidos. Os I/O específicos (webcam vs. arquivo) ficam nos
runners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import VECTOR_DIM, NAN
from .extractors import FaceMeshExtractor, PoseExtractor
from .features import (
    interpupillary_distance,
    compute_block_a, BLOCK_A_NAMES,
    compute_block_b, BLOCK_B_NAMES,
    compute_block_c, BLOCK_C_NAMES,
    compute_block_d, BLOCK_D_NAMES,
    compute_block_e, BLOCK_E_NAMES,
    compute_block_f, BLOCK_F_NAMES,
    compute_block_g, BLOCK_G_NAMES,
)
from .temporal_buffer import TemporalBuffer


# Ordem canônica das 120 features no vetor — fonte única da verdade
FEATURE_NAMES: Tuple[str, ...] = (
    BLOCK_A_NAMES +
    BLOCK_B_NAMES +
    BLOCK_C_NAMES +
    BLOCK_D_NAMES +
    BLOCK_E_NAMES +
    BLOCK_F_NAMES +
    BLOCK_G_NAMES
)
assert len(FEATURE_NAMES) == VECTOR_DIM


@dataclass
class FrameResult:
    """Saída por frame do pipeline."""
    vector: np.ndarray            # shape (120,), dtype float32
    frame_idx: int                # índice na sessão (0-based)
    timestamp_ms: float           # timestamp do frame (ms desde o início)
    face_detected: bool
    pose_detected: bool
    mp_face_landmarks: object = None   # raw MediaPipe FaceMesh landmarks (para vis.)
    mp_pose_landmarks: object = None   # raw MediaPipe Pose landmarks (para vis.)


class MindFlowPipeline:
    """
    Pipeline de extração frame → vetor 120d.

    Uso típico:
        with MindFlowPipeline() as pipe:
            for frame_idx, frame_rgb in enumerate(frames):
                result = pipe.process(frame_rgb, frame_idx, timestamp_ms)
                vectors.append(result.vector)
    """

    def __init__(
        self,
        face_min_detection: float = 0.5,
        pose_min_detection: float = 0.5,
        temporal_window: int = 30,
    ) -> None:
        self._face = FaceMeshExtractor(
            min_detection_confidence=face_min_detection,
            min_tracking_confidence=face_min_detection,
        )
        self._pose = PoseExtractor(
            min_detection_confidence=pose_min_detection,
            min_tracking_confidence=pose_min_detection,
        )
        self._buffer = TemporalBuffer(max_window=temporal_window)

    # ---------------------- API principal ----------------------
    def process(
        self,
        frame_rgb: np.ndarray,
        frame_idx: int = 0,
        timestamp_ms: float = 0.0,
    ) -> FrameResult:
        """Processa um único frame e retorna o vetor 120d."""

        face_result = self._face.process(frame_rgb)
        pose_result = self._pose.process(frame_rgb)
        h, w = frame_rgb.shape[:2]

        if face_result.detected and face_result.landmarks is not None:
            face_lm = face_result.landmarks
            d_ip = interpupillary_distance(face_lm)

            block_a = compute_block_a(face_lm, d_ip, self._buffer)
            block_b = compute_block_b(face_lm, w, h, self._buffer)
            block_c = compute_block_c(face_lm, d_ip, self._buffer)
            block_d = compute_block_d(face_lm, d_ip, self._buffer)
            # E08 (yawn_proxy) depende de A23 (já calculado)
            block_e = compute_block_e(face_lm, d_ip, self._buffer, jaw_drop_a23=block_a[22])
        else:
            # Rosto não detectado: blocos A-E preenchidos com NaN
            face_lm = None
            d_ip = 1.0
            block_a = [NAN] * 32
            block_b = [NAN] * 12
            block_c = [NAN] * 24
            block_d = [NAN] * 16
            block_e = [NAN] * 8

        # Bloco F — só depende do Pose, mas usa d_ip do face para escala
        block_f = compute_block_f(pose_result.landmarks, d_ip, self._buffer)

        # Bloco G — meta-features (sempre computáveis)
        block_g = compute_block_g(
            face_lm,
            face_result.confidence,
            pose_result.confidence,
            frame_rgb,
        )

        vector = np.array(
            block_a + block_b + block_c + block_d + block_e + block_f + block_g,
            dtype=np.float32,
        )
        assert vector.shape == (VECTOR_DIM,), f"Esperado (120,), obtido {vector.shape}"

        return FrameResult(
            vector=vector,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            face_detected=face_result.detected,
            pose_detected=pose_result.detected,
            mp_face_landmarks=face_result.mp_face_landmarks,
            mp_pose_landmarks=pose_result.mp_pose_landmarks,
        )

    def reset_temporal_buffer(self) -> None:
        """Limpa histórico temporal — chamado entre vídeos diferentes."""
        self._buffer.reset()

    # ---------------------- gerenciamento de recursos ----------------------
    def close(self) -> None:
        self._face.close()
        self._pose.close()

    def __enter__(self) -> "MindFlowPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
