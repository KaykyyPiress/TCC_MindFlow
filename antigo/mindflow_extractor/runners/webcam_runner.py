"""
Runner de webcam — validação visual do pipeline em tempo real.

Exibe overlay com:
- Landmarks faciais (FaceMesh) — toggle [L]
- Esqueleto corporal (Pose)   — toggle [P]
- Valores numéricos das features-chave em tempo real
- Indicadores de piscada, bocejo e sacada
- FPS de processamento

Objetivo principal: permitir que o desenvolvedor valide visualmente
se os cálculos batem com o que aparece no rosto (sanity-check ANTES
de rodar o DAiSEE). Não é produção — só diagnóstico.

Atalhos durante execução:
  Q / ESC  — encerrar
  S        — salvar screenshot do frame atual em outputs/
  R        — resetar buffer temporal
  L        — toggle overlay de landmarks faciais (FaceMesh)
  P        — toggle overlay de esqueleto corporal (Pose)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..pipeline import MindFlowPipeline, FEATURE_NAMES

try:
    import mediapipe as mp
    _mp_drawing = mp.solutions.drawing_utils
    _mp_drawing_styles = mp.solutions.drawing_styles
    _mp_face_mesh = mp.solutions.face_mesh
    _mp_pose = mp.solutions.pose
    _FACE_CONNECTIONS = _mp_face_mesh.FACEMESH_TESSELATION
    _FACE_CONTOURS = _mp_face_mesh.FACEMESH_CONTOURS
    _FACE_IRISES = _mp_face_mesh.FACEMESH_IRISES
    _POSE_CONNECTIONS = _mp_pose.POSE_CONNECTIONS
except ImportError:
    _mp_drawing = None
    _mp_drawing_styles = None
    _FACE_CONNECTIONS = None
    _FACE_CONTOURS = None
    _POSE_CONNECTIONS = None


# Indices de features que aparecem no overlay — escolhidos por diagnostico
_OVERLAY_FEATURES = {
    "Olho(EAR)": FEATURE_NAMES.index("D03_ear_mean"),
    "Boca(MAR)": FEATURE_NAMES.index("E01_mar"),
    "Rotacao H": FEATURE_NAMES.index("B01_head_yaw"),
    "Inclin. H": FEATURE_NAMES.index("B02_head_pitch"),
    "Giro   H":  FEATURE_NAMES.index("B03_head_roll"),
    "Olhar X":   FEATURE_NAMES.index("C01_gaze_vector_x_L"),
    "Olhar Y":   FEATURE_NAMES.index("C02_gaze_vector_y_L"),
    "Piscada":   FEATURE_NAMES.index("D08_blink_indicator"),
    "Sonolenc":  FEATURE_NAMES.index("D15_drowsiness_proxy"),
    "Bocejo":    FEATURE_NAMES.index("E08_yawn_proxy"),
    "Sacada":    FEATURE_NAMES.index("C14_saccade_indicator"),
    "Estabil":   FEATURE_NAMES.index("C24_gaze_stability_score"),
    "Luminosa":  FEATURE_NAMES.index("G03_brightness_estimate"),
    "Conf.Face": FEATURE_NAMES.index("G01_face_detection_confidence"),
}

_COLORS = {
    "green":  (0, 200, 80),
    "yellow": (0, 200, 200),
    "red":    (0, 60, 220),
    "blue":   (200, 100, 0),
    "white":  (230, 230, 230),
    "gray":   (140, 140, 140),
}


# ── Estilos de desenho ──────────────────────────────────────────────────────

def _face_landmark_spec():
    return _mp_drawing.DrawingSpec(color=(0, 220, 100), thickness=1, circle_radius=1)

def _face_contour_spec():
    return _mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1)

def _face_tesselation_spec():
    return _mp_drawing.DrawingSpec(color=(0, 70, 35), thickness=1)

def _pose_landmark_spec():
    return _mp_drawing.DrawingSpec(color=(255, 180, 0), thickness=4, circle_radius=4)

def _pose_connection_spec():
    return _mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2)


# ── Funções de desenho ──────────────────────────────────────────────────────

def _draw_face_landmarks(frame_bgr: np.ndarray, mp_face_landmarks) -> None:
    """Desenha malha FaceMesh (tessellation + contornos + iris)."""
    if mp_face_landmarks is None or _mp_drawing is None:
        return
    # Tesselacao interna (malha fina)
    _mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=mp_face_landmarks,
        connections=_FACE_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=_face_tesselation_spec(),
    )
    # Contornos externos + olhos/boca (linha mais visivel)
    _mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=mp_face_landmarks,
        connections=_FACE_CONTOURS,
        landmark_drawing_spec=_face_landmark_spec(),
        connection_drawing_spec=_face_contour_spec(),
    )
    # Iris — circulo cyan brilhante sobre cada olho
    _mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=mp_face_landmarks,
        connections=_FACE_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=_mp_drawing.DrawingSpec(
            color=(255, 220, 0), thickness=2  # cyan brilhante (BGR)
        ),
    )


def _draw_pose_skeleton(frame_bgr: np.ndarray, mp_pose_landmarks) -> None:
    """Desenha esqueleto corporal do Pose (33 pontos + conexões)."""
    if mp_pose_landmarks is None or _mp_drawing is None:
        return
    _mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=mp_pose_landmarks,
        connections=_POSE_CONNECTIONS,
        landmark_drawing_spec=_pose_landmark_spec(),
        connection_drawing_spec=_pose_connection_spec(),
    )


def _draw_toggle_hud(
    frame_bgr: np.ndarray,
    show_face: bool,
    show_pose: bool,
) -> None:
    """HUD no canto superior direito mostrando estado dos toggles L e P."""
    h, w = frame_bgr.shape[:2]
    box_w = 178
    x0 = w - box_w - 4

    # Fundo semi-transparente
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0 - 2, 4), (w - 4, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)

    def _row(label: str, key: str, active: bool, y: int):
        dot_col = _COLORS["green"] if active else _COLORS["gray"]
        txt_col = _COLORS["white"] if active else _COLORS["gray"]
        status = "ON " if active else "OFF"
        cv2.circle(frame_bgr, (x0 + 8, y - 5), 5, dot_col, -1)
        cv2.putText(
            frame_bgr, f"[{key}] {label}: {status}",
            (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1,
        )

    _row("FaceMesh", "L", show_face, 22)
    _row("Pose    ", "P", show_pose, 46)


def _draw_overlay(
    frame_bgr: np.ndarray,
    vector: np.ndarray,
    fps: float,
    face_detected: bool,
    pose_detected: bool,
) -> None:
    """Painel lateral esquerdo com features numéricas."""
    h, w = frame_bgr.shape[:2]

    # Fundo semi-transparente
    overlay = frame_bgr.copy()
    panel_w = 260
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    # Título
    cv2.putText(frame_bgr, "MindFlow AI", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COLORS["green"], 2)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, _COLORS["white"], 1)

    # Status de detecção
    face_col = _COLORS["green"] if face_detected else _COLORS["red"]
    pose_col = _COLORS["green"] if pose_detected else _COLORS["red"]
    cv2.putText(frame_bgr, f"Face: {'OK' if face_detected else 'ND'}", (8, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_col, 1)
    cv2.putText(frame_bgr, f"Pose: {'OK' if pose_detected else 'ND'}", (8, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, pose_col, 1)

    # Features chave
    y = 112
    for label, idx in _OVERLAY_FEATURES.items():
        val = float(vector[idx])
        nan_str = "NaN" if np.isnan(val) else f"{val:+.3f}"

        if label in ("Piscada", "Bocejo", "Sacada"):
            color = _COLORS["red"] if val >= 0.5 else _COLORS["gray"]
        elif np.isnan(val):
            color = _COLORS["yellow"]
        else:
            color = _COLORS["white"]

        cv2.putText(frame_bgr, f"{label:<10} {nan_str}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        y += 18

    # Rodapé
    cv2.putText(frame_bgr, "Q=quit  S=save  R=reset", (8, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _COLORS["gray"], 1)
    cv2.putText(frame_bgr, "L=facemesh  P=pose", (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _COLORS["gray"], 1)

    # Borda colorida: verde=engajado / vermelho=desengajado
    ear = float(vector[FEATURE_NAMES.index("D03_ear_mean")])
    drowsy = float(vector[FEATURE_NAMES.index("D15_drowsiness_proxy")])
    border_col = _COLORS["red"] if (ear < 0.18 or drowsy > 0.3) else _COLORS["green"]
    cv2.rectangle(frame_bgr, (panel_w, 0), (w - 1, h - 1), border_col, 3)


# ── Runner principal ────────────────────────────────────────────────────────

def run_webcam(
    camera_index: int = 0,
    output_dir: Optional[Path] = None,
    show_window: bool = True,
) -> None:
    """
    Inicia captura por webcam e exibe painel de diagnóstico em tempo real.

    Args:
        camera_index: índice da câmera OpenCV (0 = câmera padrão).
        output_dir: diretório para screenshots (default: outputs/ do projeto).
        show_window: False desabilita a janela (útil em testes headless).
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a câmera índice {camera_index}. "
            "Verifique se ela está conectada e não está em uso."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "MindFlow AI — Validacao Webcam"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    print("=" * 58)
    print("  MindFlow AI — Runner de Validação por Webcam")
    print("=" * 58)
    print("  [Q]/[ESC] encerrar  |  [S] screenshot  |  [R] reset")
    print("  [L] toggle FaceMesh |  [P] toggle Pose skeleton")
    print("-" * 58)

    frame_count = 0
    t_prev = time.perf_counter()

    # Estado dos toggles
    show_face_landmarks: bool = True
    show_pose_skeleton: bool = True

    with MindFlowPipeline() as pipe:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[AVISO] Frame não recebido. Encerrando.")
                break

            # Espelhar horizontalmente (modo espelho — mais intuitivo)
            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t_now = time.perf_counter()
            timestamp_ms = t_now * 1000.0

            result = pipe.process(frame_rgb, frame_idx=frame_count, timestamp_ms=timestamp_ms)

            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # ── Desenha esqueleto/landmarks (antes do painel de texto) ──
            if show_face_landmarks:
                _draw_face_landmarks(frame_bgr, result.mp_face_landmarks)
            if show_pose_skeleton:
                _draw_pose_skeleton(frame_bgr, result.mp_pose_landmarks)

            # ── Painel lateral de features ──
            _draw_overlay(
                frame_bgr,
                result.vector,
                fps,
                result.face_detected,
                result.pose_detected,
            )

            # ── HUD de toggles (canto superior direito) ──
            _draw_toggle_hud(frame_bgr, show_face_landmarks, show_pose_skeleton)

            if show_window:
                cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):        # Q / ESC → sair
                break
            elif key in (ord("s"), ord("S")):           # S → screenshot
                path = output_dir / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(path), frame_bgr)
                print(f"  Screenshot salvo: {path}")
            elif key in (ord("r"), ord("R")):           # R → reset buffer
                pipe.reset_temporal_buffer()
                print("  Buffer temporal resetado.")
            elif key in (ord("l"), ord("L")):           # L → toggle FaceMesh
                show_face_landmarks = not show_face_landmarks
                print(f"  FaceMesh: {'ON' if show_face_landmarks else 'OFF'}")
            elif key in (ord("p"), ord("P")):           # P → toggle Pose
                show_pose_skeleton = not show_pose_skeleton
                print(f"  Pose skeleton: {'ON' if show_pose_skeleton else 'OFF'}")

            frame_count += 1

            # Log periódico no terminal (a cada ~5s a 30fps)
            if frame_count % 150 == 1:
                ear = float(result.vector[FEATURE_NAMES.index("D03_ear_mean")])
                yaw = float(result.vector[FEATURE_NAMES.index("B01_head_yaw")])
                stab = float(result.vector[FEATURE_NAMES.index("C24_gaze_stability_score")])
                print(f"  frame={frame_count:5d} | fps={fps:.1f} | "
                      f"EAR={ear:.3f} | yaw={yaw:+5.1f}° | stability={stab:.3f} | "
                      f"face={'Y' if result.face_detected else 'N'} "
                      f"pose={'Y' if result.pose_detected else 'N'}")

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    print(f"\n  Encerrado após {frame_count} frames processados.")


if __name__ == "__main__":
    run_webcam()
