"""Persistência assíncrona do histórico da extensão via RPC do Supabase."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ParticipantCredential:
    participant_id: str
    token: str
    session_id: str


class SupabaseHistoryWriter:
    """Envia inferências sem bloquear a resposta ao vivo da extensão."""

    def __init__(
        self,
        url: str,
        anon_key: str,
        session_code: str,
        consent_version: str,
        model_version: str = "apresentacao_TCC",
        session_title: str = "",
    ) -> None:
        self.url = url.rstrip("/")
        self.anon_key = anon_key.strip()
        self.session_code = session_code.strip().upper()
        self.consent_version = consent_version.strip()
        self.model_version = model_version
        self.session_title = session_title.strip()
        self._credentials: Dict[Tuple[str, str], ParticipantCredential] = {}
        self._credentials_lock = Lock()
        self._queue: Queue = Queue(maxsize=256)
        self._stop = Event()
        self.last_error: Optional[str] = None
        self.saved_count = 0
        self.dropped_count = 0
        self._thread = Thread(target=self._run, name="mindflow-supabase", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.anon_key and self.session_code and self.consent_version)

    def configure_session(self, session_code: str, title: str = "") -> None:
        """Seleciona a reunião sem reiniciar o servidor local.

        Itens já enfileirados carregam o código anterior, portanto uma troca de
        reunião não mistura inferências entre sessões.
        """
        normalized = session_code.strip().upper()
        if not normalized:
            raise ValueError("Código da sessão ausente.")
        self.session_code = normalized
        self.session_title = title.strip()[:140]
        self.last_error = None

    def _rpc(self, function: str, payload: Dict[str, Any]) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.url}/rest/v1/rpc/{function}",
            data=body,
            method="POST",
            headers={
                "apikey": self.anon_key,
                "Authorization": f"Bearer {self.anon_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=8) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Supabase HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Supabase indisponível: {exc.reason}") from exc
        return json.loads(raw.decode("utf-8")) if raw else None

    def _credential_for(
        self,
        local_participant_id: str,
        session_code: Optional[str] = None,
    ) -> ParticipantCredential:
        selected_session = (session_code or self.session_code).strip().upper()
        if not selected_session:
            raise RuntimeError("A extensão ainda não configurou a reunião no histórico.")
        credential_key = (selected_session, local_participant_id)
        with self._credentials_lock:
            current = self._credentials.get(credential_key)
        if current:
            return current

        joined = self._rpc(
            "entrar_sessao",
            {
                "p_codigo": selected_session,
                "p_versao_termo": self.consent_version,
                "p_cliente": {
                    "origem": "chrome_extension_presenter",
                    "local_participant_id": local_participant_id,
                },
            },
        )
        credential = ParticipantCredential(
            participant_id=str(joined["participante_id"]),
            token=str(joined["token"]),
            session_id=str(joined["sessao_id"]),
        )
        with self._credentials_lock:
            self._credentials[credential_key] = credential
        return credential

    def submit(
        self,
        local_participant_id: str,
        probabilities: Dict[str, list],
        captured_at_unix_ms: int,
        face_ratio: float,
        pose_ratio: float,
        latency_ms: int,
    ) -> None:
        if not self.enabled:
            return
        item = {
            "session_code": self.session_code,
            "local_participant_id": local_participant_id,
            "probabilities": probabilities,
            "captured_at_unix_ms": captured_at_unix_ms,
            "face_ratio": face_ratio,
            "pose_ratio": pose_ratio,
            "latency_ms": latency_ms,
        }
        try:
            self._queue.put_nowait(item)
        except Full:
            self.dropped_count += 1

    def _save(self, item: Dict[str, Any]) -> None:
        credential = self._credential_for(
            item["local_participant_id"],
            item.get("session_code"),
        )
        timestamp = datetime.fromtimestamp(
            item["captured_at_unix_ms"] / 1000.0,
            tz=timezone.utc,
        ).isoformat()
        self._rpc(
            "registrar_inferencia",
            {
                "p_participante": credential.participant_id,
                "p_token": credential.token,
                "p_ts": timestamp,
                "p_dimensoes": item["probabilities"],
                "p_versao_modelo": self.model_version,
                "p_face_ratio": item["face_ratio"],
                "p_pose_ratio": item["pose_ratio"],
                "p_latencia_ms": item["latency_ms"],
            },
        )
        self.saved_count += 1
        self.last_error = None

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._save(item)
            except Exception as exc:  # histórico não pode derrubar a inferência
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def close(self, timeout: float = 10.0) -> None:
        """Pede o esvaziamento da fila sem travar indefinidamente o servidor."""
        self._stop.set()
        self._thread.join(timeout=timeout)

    def reset_participant(self, local_participant_id: str) -> None:
        with self._credentials_lock:
            matching_keys = [
                key for key in self._credentials
                if key[1] == local_participant_id
            ]
            credentials = [self._credentials.pop(key) for key in matching_keys]
        for credential in credentials:
            try:
                self._rpc(
                    "sair_sessao",
                    {"p_participante": credential.participant_id, "p_token": credential.token},
                )
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> Dict[str, Any]:
        return {
            # O writer existir significa que URL, chave anon e consentimento
            # foram fornecidos ao iniciar o Python. A sessão pode ainda não ter
            # sido informada pela extensão, portanto "available" e "enabled"
            # representam estados diferentes.
            "available": bool(self.url and self.anon_key and self.consent_version),
            "enabled": self.enabled,
            "configured": bool(self.session_code),
            "sessionCode": self.session_code or None,
            "sessionTitle": self.session_title or None,
            "queued": self._queue.qsize(),
            "saved": self.saved_count,
            "dropped": self.dropped_count,
            "lastError": self.last_error,
        }

