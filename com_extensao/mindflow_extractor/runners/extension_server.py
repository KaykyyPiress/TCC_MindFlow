"""Servidor local: frame JPEG da extensão -> MediaPipe -> 120d -> LSTM."""
from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass, field
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
import time
from typing import Any, Dict, Optional
from urllib.parse import unquote

import cv2
import numpy as np

from ..config import CAPTURE
from ..inference import MindFlowModelBundle, TASK_NAMES
from ..pipeline import MindFlowPipeline
from ..supabase_history import SupabaseHistoryWriter


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_FRAMES = 30
INFER_EVERY_FRAMES = 15
MAX_PARTICIPANTS = 5
PARTICIPANT_TTL_SECONDS = 300.0


@dataclass
class ParticipantState:
    pipeline: MindFlowPipeline
    buffer: deque = field(default_factory=lambda: deque(maxlen=120))
    processed_at: deque = field(default_factory=lambda: deque(maxlen=90))
    face_quality: deque = field(default_factory=lambda: deque(maxlen=300))
    pose_quality: deque = field(default_factory=lambda: deque(maxlen=300))
    frame_count: int = 0
    last_inference_frame: int = -10_000
    last_seen: float = field(default_factory=time.monotonic)
    last_prediction: Optional[Dict[str, Any]] = None
    lock: Lock = field(default_factory=Lock)


class ParticipantManager:
    def __init__(
        self,
        model: Optional[MindFlowModelBundle],
        history: Optional[SupabaseHistoryWriter],
        persist_every_frames: int,
    ) -> None:
        self.model = model
        self.history = history
        self.persist_every_frames = max(1, persist_every_frames)
        self._states: Dict[str, ParticipantState] = {}
        self._lock = Lock()

    @property
    def participant_count(self) -> int:
        with self._lock:
            return len(self._states)

    def _cleanup(self) -> None:
        cutoff = time.monotonic() - PARTICIPANT_TTL_SECONDS
        expired = []
        with self._lock:
            for participant_id, state in self._states.items():
                if state.last_seen < cutoff:
                    expired.append((participant_id, state))
            for participant_id, _ in expired:
                self._states.pop(participant_id, None)
        for participant_id, state in expired:
            state.pipeline.close()
            if self.history:
                self.history.reset_participant(participant_id)

    def _state_for(self, participant_id: str) -> ParticipantState:
        self._cleanup()
        with self._lock:
            state = self._states.get(participant_id)
            if state:
                return state
            if len(self._states) >= MAX_PARTICIPANTS:
                raise RuntimeError("Limite de 5 participantes ativos atingido.")
            state = ParticipantState(
                pipeline=MindFlowPipeline(temporal_window=CAPTURE.target_fps)
            )
            self._states[participant_id] = state
            return state

    def reset(self, participant_id: str) -> bool:
        with self._lock:
            state = self._states.pop(participant_id, None)
        if not state:
            return False
        state.pipeline.close()
        if self.history:
            self.history.reset_participant(participant_id)
        return True

    def close(self) -> None:
        with self._lock:
            states = list(self._states.values())
            self._states.clear()
        for state in states:
            state.pipeline.close()

    def configure_history_session(self, session_code: str, title: str = "") -> Dict[str, Any]:
        if not self.history:
            raise RuntimeError(
                "Histórico indisponível. Configure URL, chave anon e versão do consentimento no Python."
            )
        self.history.configure_session(session_code, title)
        return self.history.status()

    def process(
        self,
        participant_id: str,
        frame_bgr: np.ndarray,
        captured_at_unix_ms: int,
        session_timestamp_ms: float,
    ) -> Dict[str, Any]:
        state = self._state_for(participant_id)
        started = time.perf_counter()

        with state.lock:
            state.last_seen = time.monotonic()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = state.pipeline.process(
                frame_rgb,
                frame_idx=state.frame_count,
                timestamp_ms=session_timestamp_ms,
            )
            state.buffer.append(result.vector.copy())
            state.processed_at.append(time.monotonic())
            state.face_quality.append(1.0 if result.face_detected else 0.0)
            state.pose_quality.append(1.0 if result.pose_detected else 0.0)
            state.frame_count += 1

            processing_fps = 0.0
            if len(state.processed_at) >= 2:
                elapsed = state.processed_at[-1] - state.processed_at[0]
                if elapsed > 0:
                    processing_fps = (len(state.processed_at) - 1) / elapsed

            prediction_payload = state.last_prediction
            should_infer = (
                self.model is not None
                and len(state.buffer) >= SEQUENCE_FRAMES
                and state.frame_count - state.last_inference_frame >= INFER_EVERY_FRAMES
            )
            if should_infer:
                sequence = np.asarray(list(state.buffer)[-SEQUENCE_FRAMES:], dtype=np.float32)
                prediction = self.model.predict(sequence)
                states = {}
                for task in TASK_NAMES:
                    states[task] = {
                        "probabilities": prediction.probabilities[task],
                        "level": prediction.levels[task],
                        "scorePercent": round(prediction.scores_percent[task], 2),
                        "confidencePercent": round(prediction.confidence_percent[task], 2),
                    }
                prediction_payload = {
                    "ready": True,
                    "sequenceFrames": SEQUENCE_FRAMES,
                    "updatedAtFrame": state.frame_count - 1,
                    "states": states,
                }
                state.last_prediction = prediction_payload
                state.last_inference_frame = state.frame_count

            latency_ms = int(round((time.perf_counter() - started) * 1000.0))
            if (
                prediction_payload
                and self.history
                and state.frame_count % self.persist_every_frames == 0
            ):
                self.history.submit(
                    local_participant_id=participant_id,
                    probabilities={
                        task: prediction_payload["states"][task]["probabilities"]
                        for task in TASK_NAMES
                    },
                    captured_at_unix_ms=captured_at_unix_ms,
                    face_ratio=float(np.mean(list(state.face_quality)[-self.persist_every_frames:])),
                    pose_ratio=float(np.mean(list(state.pose_quality)[-self.persist_every_frames:])),
                    latency_ms=latency_ms,
                )

            return {
                "ok": True,
                "participantId": participant_id,
                "frameIdx": state.frame_count - 1,
                "targetFps": CAPTURE.target_fps,
                "processingFps": round(processing_fps, 2),
                "vectorDim": int(result.vector.shape[0]),
                "finiteCount": int(np.isfinite(result.vector).sum()),
                "nanCount": int(np.isnan(result.vector).sum()),
                "faceDetected": result.face_detected,
                "poseDetected": result.pose_detected,
                "latencyMs": latency_ms,
                "inference": prediction_payload or {
                    "ready": False,
                    "framesCollected": len(state.buffer),
                    "sequenceFrames": SEQUENCE_FRAMES,
                    "states": {},
                },
            }


def decode_frame(packet: Dict[str, Any]) -> np.ndarray:
    image_data_url = str(packet.get("imageDataUrl") or "")
    if not image_data_url.startswith("data:image/") or "," not in image_data_url:
        raise ValueError("imageDataUrl ausente ou inválido.")
    encoded = image_data_url.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 2_000_000:
        raise ValueError("Frame excede 2 MB.")
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV não conseguiu decodificar o frame.")
    return frame


def build_handler(manager: ParticipantManager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MindFlowLocal/0.3"

        def send_json(self, status: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "https://meet.google.com")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self) -> None:
            self.send_json(204, {})

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"ok": False, "error": "endpoint_not_found"})
                return
            self.send_json(200, {
                "ok": True,
                "service": "MindFlow Python MediaPipe bridge",
                "targetFps": CAPTURE.target_fps,
                "frameSize": [CAPTURE.frame_width, CAPTURE.frame_height],
                "maxParticipants": MAX_PARTICIPANTS,
                "participantsInMemory": manager.participant_count,
                "sequenceFrames": SEQUENCE_FRAMES,
                "modelLoaded": manager.model is not None,
                "history": manager.history.status() if manager.history else {
                    "available": False,
                    "enabled": False,
                    "configured": False,
                },
            })

        def do_POST(self) -> None:
            if self.path.startswith("/reset/"):
                participant_id = unquote(self.path[len("/reset/"):]).strip()
                self.send_json(200, {
                    "ok": True,
                    "participantId": participant_id,
                    "reset": manager.reset(participant_id),
                })
                return
            if self.path == "/history/session":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 16_384:
                        raise ValueError("Tamanho da configuração de histórico inválido.")
                    packet = json.loads(self.rfile.read(length).decode("utf-8"))
                    status = manager.configure_history_session(
                        str(packet.get("sessionCode") or ""),
                        str(packet.get("title") or ""),
                    )
                    self.send_json(200, {"ok": True, "history": status})
                except Exception as exc:
                    self.send_json(409, {
                        "ok": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    })
                return
            if self.path != "/frame":
                self.send_json(404, {"ok": False, "error": "endpoint_not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 3_000_000:
                    raise ValueError("Tamanho do pacote inválido.")
                packet = json.loads(self.rfile.read(length).decode("utf-8"))
                participant_id = str(packet.get("participantId") or "").strip()
                if not participant_id:
                    raise ValueError("participantId ausente.")
                frame = decode_frame(packet)
                response = manager.process(
                    participant_id=participant_id,
                    frame_bgr=frame,
                    captured_at_unix_ms=int(packet.get("capturedAtUnixMs") or time.time() * 1000),
                    session_timestamp_ms=float(packet.get("sessionTimestampMs") or 0.0),
                )
                self.send_json(200, response)
            except Exception as exc:
                self.send_json(400, {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                })

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor local MindFlow para a extensão")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "apresentacao_TCC")
    parser.add_argument("--persist-every-frames", type=int, default=30)
    parser.add_argument("--supabase-url", default=os.getenv("MINDFLOW_SUPABASE_URL", ""))
    parser.add_argument("--supabase-anon-key", default=os.getenv("MINDFLOW_SUPABASE_ANON_KEY", ""))
    parser.add_argument("--session-code", default=os.getenv("MINDFLOW_SESSION_CODE", ""))
    parser.add_argument("--consent-version", default=os.getenv("MINDFLOW_CONSENT_VERSION", ""))
    args = parser.parse_args()

    model = None
    try:
        model = MindFlowModelBundle(args.model_dir)
        print(f"Modelo carregado: {args.model_dir}")
    except Exception as exc:
        print(f"AVISO: inferência desativada: {exc}")

    history = None
    if args.supabase_url and args.supabase_anon_key and args.consent_version:
        history = SupabaseHistoryWriter(
            args.supabase_url,
            args.supabase_anon_key,
            args.session_code,
            args.consent_version,
            model_version=args.model_dir.name,
        )
        if args.session_code:
            print(f"Histórico Supabase ativado para a sessão {args.session_code}.")
        else:
            print("Histórico Supabase pronto; aguardando a extensão informar a reunião.")
    else:
        print(
            "Histórico Supabase desativado: configure URL, anon key e a versão "
            "do termo consentido. O código da reunião virá da extensão."
        )

    manager = ParticipantManager(model, history, args.persist_every_frames)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(manager))
    print(f"MindFlow local em http://{args.host}:{args.port} | alvo={CAPTURE.target_fps} FPS")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Encerrando...")
    finally:
        server.server_close()
        manager.close()
        if history:
            history.close()


if __name__ == "__main__":
    main()

