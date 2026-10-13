"""
Wrapper do MediaPipe Pose.

O Pose retorna 33 landmarks corporais. Em vídeos do DAiSEE (webcam de
estudante sentado), tipicamente apenas os índices 0..22 (cabeça, tronco
superior, braços) estarão visíveis. Os landmarks fornecem .visibility,
usado tanto na feature F24 (pose_visibility_score) quanto para mascarar
features inválidas com NaN no Bloco F.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mediapipe não está instalado. Rode: pip install mediapipe"
    ) from e


@dataclass
class PoseResult:
    """Resultado encapsulado de uma extração de Pose em um frame."""
    landmarks: Optional[np.ndarray]   # shape (33, 4) — x,y,z,visibility
    detected: bool
    confidence: float                 # média de visibility dos lm 0..22
    mp_pose_landmarks: object = None  # objeto raw do MediaPipe (para desenho)


class PoseExtractor:
    """
    Wrapper do mp.solutions.pose.Pose.

    model_complexity=1 é o equilíbrio padrão entre acurácia e velocidade,
    adequado ao requisito da metodologia de operar sem GPU. Para o DAiSEE
    offline, pode-se subir para 2 se o tempo de extração total permitir.
    """

    NUM_LANDMARKS: int = 33

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            enable_segmentation=False,  # não usado pelo pipeline
        )

    def process(self, frame_rgb: np.ndarray) -> PoseResult:
        """
        Extrai landmarks corporais de um frame.

        Args:
            frame_rgb: numpy array (H, W, 3) em RGB.

        Returns:
            PoseResult — landmarks com 4 colunas (x, y, z, visibility) ou
            None se não detectou pose.
        """
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.pose_landmarks:
            return PoseResult(landmarks=None, detected=False, confidence=0.0)

        lm_list = results.pose_landmarks.landmark
        coords = np.array(
            [(lm.x, lm.y, lm.z, lm.visibility) for lm in lm_list],
            dtype=np.float32,
        )

        # Confidence agregada: média da visibility dos landmarks superiores
        # (subconjunto 0..22), pois é o que está disponível em webcam.
        upper = coords[:23, 3]
        confidence = float(upper.mean())

        return PoseResult(
            landmarks=coords,
            detected=True,
            confidence=confidence,
            mp_pose_landmarks=results.pose_landmarks,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
