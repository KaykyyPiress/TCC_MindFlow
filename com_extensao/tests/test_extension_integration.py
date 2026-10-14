from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mindflow_extractor.inference import MindFlowModelBundle
from mindflow_extractor.supabase_history import ParticipantCredential, SupabaseHistoryWriter


ROOT = Path(__file__).resolve().parents[1]


def test_beta_artifacts_accept_the_extractor_contract() -> None:
    model_dir = ROOT / "models" / "apresentacao_TCC"
    if not model_dir.exists():
        pytest.skip("Artefatos beta não disponíveis neste checkout.")

    bundle = MindFlowModelBundle(model_dir)
    prediction = bundle.predict(np.zeros((30, 120), dtype=np.float32))

    assert bundle.model_feature_count in (120, 121)
    assert set(prediction.probabilities) == {
        "engagement", "boredom", "confusion", "frustration",
    }
    for probabilities in prediction.probabilities.values():
        assert len(probabilities) == 4
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)


def test_supabase_payload_matches_registrar_inferencia_rpc() -> None:
    writer = object.__new__(SupabaseHistoryWriter)
    writer.model_version = "apresentacao_TCC"
    writer.saved_count = 0
    writer.last_error = "erro anterior"
    calls = []
    writer._credential_for = lambda _local_id, _session_code=None: ParticipantCredential(
        participant_id="00000000-0000-0000-0000-000000000001",
        token="token-temporario",
        session_id="00000000-0000-0000-0000-000000000002",
    )
    writer._rpc = lambda function, payload: calls.append((function, payload))

    probabilities = {
        task: [0.1, 0.2, 0.3, 0.4]
        for task in ("engagement", "boredom", "confusion", "frustration")
    }
    writer._save({
        "session_code": "MF-TESTE",
        "local_participant_id": "participant_01",
        "probabilities": probabilities,
        "captured_at_unix_ms": 1_700_000_000_000,
        "face_ratio": 1.0,
        "pose_ratio": 0.5,
        "latency_ms": 25,
    })

    function, payload = calls[0]
    assert function == "registrar_inferencia"
    assert payload["p_dimensoes"] == probabilities
    assert payload["p_versao_modelo"] == "apresentacao_TCC"
    assert payload["p_face_ratio"] == 1.0
    assert payload["p_pose_ratio"] == 0.5
    assert payload["p_latencia_ms"] == 25
    assert writer.saved_count == 1
    assert writer.last_error is None


def test_history_status_distinguishes_ready_writer_from_active_session() -> None:
    writer = object.__new__(SupabaseHistoryWriter)
    writer.url = "https://example.supabase.co"
    writer.anon_key = "anon-public-key"
    writer.consent_version = "v1"
    writer.session_code = ""
    writer.session_title = ""
    writer.saved_count = 0
    writer.dropped_count = 0
    writer.last_error = None
    writer._queue = type("QueueStub", (), {"qsize": lambda self: 0})()

    waiting = writer.status()
    assert waiting["available"] is True
    assert waiting["configured"] is False
    assert waiting["enabled"] is False

    writer.session_code = "MF-TESTE"
    active = writer.status()
    assert active["available"] is True
    assert active["configured"] is True
    assert active["enabled"] is True
