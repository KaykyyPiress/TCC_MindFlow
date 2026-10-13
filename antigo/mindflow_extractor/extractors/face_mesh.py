"""
Wrapper do MediaPipe FaceMesh com refine_landmarks=True.

A flag refine_landmarks=True é essencial: ativa os 10 landmarks adicionais
de íris (468..477), eliminando a necessidade de um segundo modelo dedicado.
Isso preserva o princípio da metodologia de manter o pipeline enxuto e
executável sem GPU.

Saída padronizada: numpy.ndarray (N, 3) com coordenadas normalizadas em
[0, 1] — N = 478 quando há rosto detectado, None caso contrário. Também
expõe a confidence aproximada do detector como a média de visibilities (o
FaceMesh não retorna confidence por landmark como o Pose, então usamos a
presença ou ausência como sinal binário e adicionamos um "score" estimado).
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
class FaceMeshResult:
    """Resultado encapsulado de uma extração de FaceMesh em um frame."""
    landmarks: Optional[np.ndarray]  # shape (478, 3) ou None
    detected: bool                   # True se rosto foi detectado
    confidence: float                # estimativa de qualidade [0, 1]
    mp_face_landmarks: object = None  # objeto raw do MediaPipe (para desenho)


class FaceMeshExtractor:
    """
    Wrapper minimalista do mp.solutions.face_mesh.FaceMesh.

    Parâmetros padrão escolhidos para o caso de uso do MindFlow:
    - max_num_faces=1: foco em um participante por vez (cliente local).
    - refine_landmarks=True: ativa íris (necessário para o Bloco C).
    - min_detection_confidence=0.5: padrão; ajustável para condições
      adversas no DAiSEE in-the-wild.
    - min_tracking_confidence=0.5: idem.
    """

    NUM_LANDMARKS_WITH_IRIS: int = 478  # 468 face + 10 íris

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
        )

    def process(self, frame_rgb: np.ndarray) -> FaceMeshResult:
        """
        Extrai landmarks de um frame.

        Args:
            frame_rgb: numpy array (H, W, 3) em RGB (não BGR!). Quem chama é
                       responsável por converter de BGR (OpenCV padrão) para
                       RGB antes de passar aqui.

        Returns:
            FaceMeshResult com landmarks ou None se não detectou rosto.
        """
        # Otimização recomendada pelo MediaPipe: marcar imagem como
        # não-writable evita cópia interna.
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return FaceMeshResult(landmarks=None, detected=False, confidence=0.0)

        face_lm = results.multi_face_landmarks[0]
        coords = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_lm.landmark],
            dtype=np.float32,
        )

        # MediaPipe FaceMesh não expõe confidence por landmark.
        # Aproximamos a "confidence" pela razão de landmarks dentro do
        # frame normalizado [0,1] em xy — uma heurística simples mas útil
        # como sinal para a feature G01.
        xy = coords[:, :2]
        in_frame = ((xy >= 0.0) & (xy <= 1.0)).all(axis=1)
        confidence = float(in_frame.mean())

        return FaceMeshResult(
            landmarks=coords,
            detected=True,
            confidence=confidence,
            mp_face_landmarks=face_lm,
        )

    def close(self) -> None:
        """Libera os recursos do MediaPipe."""
        self._face_mesh.close()

    def __enter__(self) -> "FaceMeshExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
