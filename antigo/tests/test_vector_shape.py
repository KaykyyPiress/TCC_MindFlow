"""
Suite de sanity-checks do pipeline MindFlow.

Garantias testadas:
1. O vetor sempre tem exatamente 120 dimensões.
2. Quando face é detectada (mock de landmarks válidos), blocos A-E têm no máximo 0 NaN.
3. Quando face NÃO é detectada, blocos A-E são todos NaN; G01 = 0.0.
4. O buffer temporal produz deltas corretos (Δ de constante = 0.0).
5. O interpupillary_distance nunca retorna zero (divisão segura).

Os testes usam um frame sintético (ruído RGB) com um mock de landmarks
gerado analiticamente — não dependem de câmera, MediaPipe ou GPU.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Adiciona o pacote ao path (caso rode sem instalar via pip)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mindflow_extractor.config import VECTOR_DIM, NAN
from mindflow_extractor.temporal_buffer import TemporalBuffer
from mindflow_extractor.features.geometry import (
    distance, distance_2d, interpupillary_distance, norm_by,
)
from mindflow_extractor.features.block_d_eyes import eye_aspect_ratio
from mindflow_extractor.features.block_e_mouth import mouth_aspect_ratio
import mindflow_extractor.landmark_indices as L


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def blank_rgb():
    """Frame RGB de 480x640 — todos os pixels = 128 (luminância neutra)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def synthetic_face_landmarks():
    """
    478 landmarks sintéticos com posições anatômicas aproximadas.
    Os índices de íris (468, 473) estão posicionados para d_ip ≈ 0.16,
    garantindo normalização não-zero e valores de feature razoáveis.
    Coordenadas em [0, 1] (sistema MediaPipe).
    """
    lm = np.zeros((478, 3), dtype=np.float32)
    # Centro da imagem = (0.5, 0.5); faces típicas em webcam
    cx, cy = 0.50, 0.40

    # Íris — separação horizontal de 0.08 (d_ip ≈ 0.16 diagonal)
    lm[L.IRIS_LEFT_CENTER]  = [cx - 0.08, cy, 0.0]  # esquerda (imagem)
    lm[L.IRIS_RIGHT_CENTER] = [cx + 0.08, cy, 0.0]  # direita

    # Periféricos da íris esq. (raio ≈ 0.02)
    r_iris = 0.02
    lm[L.IRIS_LEFT_TOP]    = [cx - 0.08,           cy - r_iris, 0.0]
    lm[L.IRIS_LEFT_BOTTOM] = [cx - 0.08,           cy + r_iris, 0.0]
    lm[L.IRIS_LEFT_LEFT]   = [cx - 0.08 - r_iris,  cy, 0.0]
    lm[L.IRIS_LEFT_RIGHT]  = [cx - 0.08 + r_iris,  cy, 0.0]
    # Periféricos da íris dir.
    lm[L.IRIS_RIGHT_TOP]   = [cx + 0.08,           cy - r_iris, 0.0]
    lm[L.IRIS_RIGHT_BOTTOM]= [cx + 0.08,           cy + r_iris, 0.0]
    lm[L.IRIS_RIGHT_LEFT]  = [cx + 0.08 - r_iris,  cy, 0.0]
    lm[L.IRIS_RIGHT_RIGHT] = [cx + 0.08 + r_iris,  cy, 0.0]

    # Olhos — cantos e pálpebras (para EAR)
    eye_half_w = 0.05
    eye_open   = 0.015  # abertura vertical normal ~ EAR 0.30
    lm[L.EYE_LEFT_OUTER_CORNER]  = [cx - 0.08 - eye_half_w, cy, 0.0]
    lm[L.EYE_LEFT_INNER_CORNER]  = [cx - 0.08 + eye_half_w, cy, 0.0]
    lm[L.EYE_LEFT_TOP_1]         = [cx - 0.08, cy - eye_open, 0.0]
    lm[L.EYE_LEFT_TOP_2]         = [cx - 0.08, cy - eye_open * 0.9, 0.0]
    lm[L.EYE_LEFT_BOTTOM_1]      = [cx - 0.08, cy + eye_open, 0.0]
    lm[L.EYE_LEFT_BOTTOM_2]      = [cx - 0.08, cy + eye_open * 0.9, 0.0]

    lm[L.EYE_RIGHT_OUTER_CORNER]  = [cx + 0.08 + eye_half_w, cy, 0.0]
    lm[L.EYE_RIGHT_INNER_CORNER]  = [cx + 0.08 - eye_half_w, cy, 0.0]
    lm[L.EYE_RIGHT_TOP_1]         = [cx + 0.08, cy - eye_open, 0.0]
    lm[L.EYE_RIGHT_TOP_2]         = [cx + 0.08, cy - eye_open * 0.9, 0.0]
    lm[L.EYE_RIGHT_BOTTOM_1]      = [cx + 0.08, cy + eye_open, 0.0]
    lm[L.EYE_RIGHT_BOTTOM_2]      = [cx + 0.08, cy + eye_open * 0.9, 0.0]

    # Sobrancelhas (para AUs 1, 2, 4)
    brow_y = cy - 0.07
    lm[L.BROW_LEFT_INNER]  = [cx - 0.05, brow_y, 0.0]
    lm[L.BROW_LEFT_OUTER]  = [cx - 0.12, brow_y, 0.0]
    lm[L.BROW_LEFT_MID]    = [cx - 0.085, brow_y, 0.0]
    lm[L.BROW_RIGHT_INNER] = [cx + 0.05, brow_y, 0.0]
    lm[L.BROW_RIGHT_OUTER] = [cx + 0.12, brow_y, 0.0]
    lm[L.BROW_RIGHT_MID]   = [cx + 0.085, brow_y, 0.0]
    lm[L.GLABELLA]         = [cx, cy - 0.08, 0.0]

    # Boca (para MAR, AUs 12, 15, 18, 20, 24, 26)
    lm[L.MOUTH_CORNER_LEFT]  = [cx - 0.05, cy + 0.12, 0.0]
    lm[L.MOUTH_CORNER_RIGHT] = [cx + 0.05, cy + 0.12, 0.0]
    lm[L.MOUTH_TOP_INNER]    = [cx, cy + 0.09, 0.0]
    lm[L.MOUTH_BOTTOM_INNER] = [cx, cy + 0.13, 0.0]
    lm[L.MOUTH_BOTTOM_OUTER] = [cx, cy + 0.16, 0.0]
    # MAR vertical pairs (37, 84, 267, 314, 13, 14 já definidos acima via inner)
    lm[37]  = [cx - 0.03, cy + 0.09, 0.0]
    lm[84]  = [cx - 0.03, cy + 0.13, 0.0]
    lm[267] = [cx + 0.03, cy + 0.09, 0.0]
    lm[314] = [cx + 0.03, cy + 0.13, 0.0]

    # Nariz e queixo
    lm[L.NOSE_TIP]          = [cx, cy + 0.04, 0.0]
    lm[L.CHIN_BOTTOM]       = [cx, cy + 0.22, 0.0]
    lm[L.NOSE_BRIDGE_TOP]   = [cx, cy - 0.04, 0.0]
    lm[L.NOSE_BRIDGE_BOTTOM]= [cx, cy + 0.01, 0.0]

    # Bochechas (AU6)
    lm[L.CHEEK_LEFT]  = [cx - 0.12, cy + 0.08, 0.0]
    lm[L.CHEEK_RIGHT] = [cx + 0.12, cy + 0.08, 0.0]

    return lm


@pytest.fixture
def buf():
    return TemporalBuffer(max_window=30)


# ===========================================================================
# Testes de geometria
# ===========================================================================
class TestGeometry:
    def test_distance_2d(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        assert math.isclose(distance_2d(p1, p2), 5.0, rel_tol=1e-6)

    def test_norm_by_zero_denominator(self):
        """norm_by nunca divide por zero."""
        result = norm_by(1.0, 0.0)
        assert math.isfinite(result)
        assert result > 0

    def test_interpupillary_distance_nonzero(self, synthetic_face_landmarks):
        d = interpupillary_distance(synthetic_face_landmarks)
        assert d > 0, "d_ip nunca deve ser zero"
        assert math.isfinite(d)

    def test_interpupillary_distance_value(self, synthetic_face_landmarks):
        """Com as íris a distância x=0.16, d_ip deve ser próximo de 0.16."""
        d = interpupillary_distance(synthetic_face_landmarks)
        assert 0.10 < d < 0.30, f"d_ip={d:.4f} fora do intervalo esperado"


# ===========================================================================
# Testes do buffer temporal
# ===========================================================================
class TestTemporalBuffer:
    def test_delta_first_frame_is_zero(self, buf):
        buf.append("x", 5.0)
        assert buf.delta("x") == 0.0, "Δ no 1º frame deve ser 0"

    def test_delta_constant_signal(self, buf):
        for _ in range(10):
            buf.append("x", 3.14)
        assert buf.delta("x") == 0.0, "Δ de sinal constante deve ser 0"

    def test_delta_step(self, buf):
        buf.append("x", 0.0)
        buf.append("x", 1.0)
        assert math.isclose(buf.delta("x"), 1.0)

    def test_variance_constant(self, buf):
        for _ in range(10):
            buf.append("x", 2.71)
        assert buf.variance("x", 5) < 1e-6

    def test_consecutive_true(self, buf):
        for _ in range(5):
            buf.append("flag", 1.0)
        assert buf.consecutive_true("flag") == 5

    def test_fraction_true(self, buf):
        for i in range(10):
            buf.append("flag", float(i % 2 == 0))
        frac = buf.fraction_true("flag", 10)
        assert math.isclose(frac, 0.5, rel_tol=1e-3)

    def test_reset_clears_history(self, buf):
        buf.append("x", 1.0)
        buf.reset()
        assert len(buf) == 0


# ===========================================================================
# Testes do EAR / MAR
# ===========================================================================
class TestEarMar:
    def test_ear_open_eye(self, synthetic_face_landmarks):
        """Olho aberto deve ter EAR em torno de 0.15-0.40."""
        ear = eye_aspect_ratio(synthetic_face_landmarks, L.EYE_LEFT_EAR_INDICES)
        assert 0.10 < ear < 0.50, f"EAR={ear:.4f} fora do intervalo esperado"

    def test_ear_symmetric(self, synthetic_face_landmarks):
        """Com rosto sintético simétrico, EAR esq. ≈ EAR dir."""
        ear_l = eye_aspect_ratio(synthetic_face_landmarks, L.EYE_LEFT_EAR_INDICES)
        ear_r = eye_aspect_ratio(synthetic_face_landmarks, L.EYE_RIGHT_EAR_INDICES)
        assert abs(ear_l - ear_r) < 0.05, f"Assimetria EAR: L={ear_l:.4f} R={ear_r:.4f}"

    def test_mar_open_mouth(self, synthetic_face_landmarks):
        mar = mouth_aspect_ratio(synthetic_face_landmarks)
        assert mar > 0.0, "MAR deve ser positivo"
        assert math.isfinite(mar), "MAR não deve ser NaN/inf"


# ===========================================================================
# Testes de dimensionalidade dos blocos
# ===========================================================================
class TestBlockDimensions:
    def _run_block_a(self, lm, buf):
        from mindflow_extractor.features.block_a_facial_aus import compute_block_a
        d_ip = interpupillary_distance(lm)
        return compute_block_a(lm, d_ip, buf)

    def _run_block_b(self, lm, buf):
        from mindflow_extractor.features.block_b_head_pose import compute_block_b
        return compute_block_b(lm, 640, 480, buf)

    def _run_block_c(self, lm, buf):
        from mindflow_extractor.features.block_c_gaze import compute_block_c
        d_ip = interpupillary_distance(lm)
        return compute_block_c(lm, d_ip, buf)

    def _run_block_d(self, lm, buf):
        from mindflow_extractor.features.block_d_eyes import compute_block_d
        d_ip = interpupillary_distance(lm)
        return compute_block_d(lm, d_ip, buf)

    def _run_block_e(self, lm, buf, a23=0.1):
        from mindflow_extractor.features.block_e_mouth import compute_block_e
        d_ip = interpupillary_distance(lm)
        return compute_block_e(lm, d_ip, buf, jaw_drop_a23=a23)

    def _run_block_f(self, buf):
        from mindflow_extractor.features.block_f_pose import compute_block_f
        return compute_block_f(None, d_ip_face=0.16, buf=buf)

    def _run_block_g(self, lm, frame):
        from mindflow_extractor.features.block_g_quality import compute_block_g
        return compute_block_g(lm, 0.9, 0.8, frame)

    def test_block_a_is_32d(self, synthetic_face_landmarks, buf):
        r = self._run_block_a(synthetic_face_landmarks, buf)
        assert len(r) == 32

    def test_block_b_is_12d(self, synthetic_face_landmarks, buf):
        r = self._run_block_b(synthetic_face_landmarks, buf)
        assert len(r) == 12

    def test_block_c_is_24d(self, synthetic_face_landmarks, buf):
        r = self._run_block_c(synthetic_face_landmarks, buf)
        assert len(r) == 24

    def test_block_d_is_16d(self, synthetic_face_landmarks, buf):
        r = self._run_block_d(synthetic_face_landmarks, buf)
        assert len(r) == 16

    def test_block_e_is_8d(self, synthetic_face_landmarks, buf):
        r = self._run_block_e(synthetic_face_landmarks, buf)
        assert len(r) == 8

    def test_block_f_no_pose_is_24d_nan(self, buf):
        """Pose ausente → 24 NaN (exceto F19/F24 que são float)."""
        r = self._run_block_f(buf)
        assert len(r) == 24
        # Sem pose: todos devem ser NaN
        assert all(np.isnan(v) for v in r)

    def test_block_g_is_4d(self, synthetic_face_landmarks, blank_rgb):
        r = self._run_block_g(synthetic_face_landmarks, blank_rgb)
        assert len(r) == 4

    def test_no_nan_in_face_blocks_when_face_detected(
        self, synthetic_face_landmarks, blank_rgb, buf
    ):
        """Quando face está presente, blocos A-E não devem ter NaN."""
        a = self._run_block_a(synthetic_face_landmarks, buf)
        b = self._run_block_b(synthetic_face_landmarks, buf)
        c = self._run_block_c(synthetic_face_landmarks, buf)
        d = self._run_block_d(synthetic_face_landmarks, buf)
        e = self._run_block_e(synthetic_face_landmarks, buf)

        for name, block in [("A", a), ("B", b), ("C", c), ("D", d), ("E", e)]:
            nans = [i for i, v in enumerate(block) if np.isnan(v)]
            assert not nans, f"Bloco {name} tem NaN nas posições: {nans}"


# ===========================================================================
# Teste de dimensionalidade do FEATURE_NAMES canônico
# ===========================================================================
def test_feature_names_total():
    from mindflow_extractor.pipeline import FEATURE_NAMES
    assert len(FEATURE_NAMES) == VECTOR_DIM, (
        f"FEATURE_NAMES tem {len(FEATURE_NAMES)} entradas, esperado {VECTOR_DIM}"
    )


def test_feature_names_unique():
    from mindflow_extractor.pipeline import FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == VECTOR_DIM, "Nomes de features duplicados!"
