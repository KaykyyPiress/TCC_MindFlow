"""Inferência LSTM multi-output para o servidor local da extensão."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np


TASK_NAMES = ("engagement", "boredom", "confusion", "frustration")
LEVEL_NAMES = ("very_low", "low", "high", "very_high")
RAW_FEATURE_COUNT = 120
SEQUENCE_LENGTH = 30
EAR_MEAN_INDEX = 70
PERCLOS_WINDOW_FRAMES = 6
PERCLOS_EAR_THRESHOLD = 0.22


@dataclass(frozen=True)
class Prediction:
    probabilities: Dict[str, List[float]]
    levels: Dict[str, int]
    scores_percent: Dict[str, float]
    confidence_percent: Dict[str, float]


class MindFlowModelBundle:
    """Carrega scaler, PCA e a LSTM de quatro cabeças."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        import torch.nn as nn

        model_dir = Path(model_dir)
        scaler_path = model_dir / "scaler.joblib"
        pca_path = model_dir / "pca.joblib"
        model_path = model_dir / "lstm_mindflow.pt"

        missing = [p.name for p in (scaler_path, pca_path, model_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Artefatos ausentes em {model_dir}: {', '.join(missing)}"
            )

        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        input_dim = int(self.pca.n_components_)
        self.sequence_length = SEQUENCE_LENGTH
        self.feature_count = RAW_FEATURE_COUNT
        self.model_feature_count = int(self.scaler.n_features_in_)
        if self.model_feature_count not in (RAW_FEATURE_COUNT, RAW_FEATURE_COUNT + 1):
            raise ValueError(
                "O scaler espera "
                f"{self.model_feature_count} features; o servidor suporta 120 ou 121 (com PERCLOS)."
            )

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        hidden_1 = int(state["lstm1.weight_hh_l0"].shape[1])
        hidden_2 = int(state["lstm2.weight_hh_l0"].shape[1])
        head_indexes = sorted({
            int(key.split(".")[1])
            for key in state
            if key.startswith("heads.") and key.endswith(".weight")
        })
        if len(head_indexes) != len(TASK_NAMES):
            raise ValueError(
                f"O checkpoint possui {len(head_indexes)} cabeças; eram esperadas {len(TASK_NAMES)}."
            )
        output_levels = int(state[f"heads.{head_indexes[0]}.weight"].shape[0])
        if output_levels != len(LEVEL_NAMES):
            raise ValueError(
                f"O checkpoint possui {output_levels} níveis; eram esperados {len(LEVEL_NAMES)}."
            )

        class _MindFlowLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # As dimensões são lidas do checkpoint. Assim, uma nova execução
                # de treino não exige editar manualmente este servidor.
                self.lstm1 = nn.LSTM(input_dim, hidden_1, batch_first=True)
                self.drop = nn.Dropout(p=0.0)
                self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
                self.heads = nn.ModuleList([
                    nn.Linear(hidden_2, output_levels) for _ in head_indexes
                ])

            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.drop(out)
                out, _ = self.lstm2(out)
                hidden = out[:, -1, :]
                return [head(hidden) for head in self.heads]

        self._torch = torch
        self.model = _MindFlowLSTM()
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _perclos(sequence: np.ndarray) -> np.ndarray:
        """PERCLOS causal igual ao usado pelo treino quando o scaler espera 121d."""
        ear = sequence[:, EAR_MEAN_INDEX]
        valid = np.isfinite(ear)
        closed = np.where(valid, ear < PERCLOS_EAR_THRESHOLD, False)
        values = np.full(len(ear), np.nan, dtype=np.float32)
        for index in range(len(ear)):
            start = max(0, index - PERCLOS_WINDOW_FRAMES + 1)
            valid_count = int(valid[start:index + 1].sum())
            if valid_count:
                values[index] = float(closed[start:index + 1].sum() / valid_count)
        return values

    def predict(self, sequence: np.ndarray) -> Prediction:
        """Executa scaler -> PCA -> LSTM para uma sequência (30, 120)."""
        if sequence.shape != (self.sequence_length, self.feature_count):
            raise ValueError(
                f"Sequência esperada ({self.sequence_length}, {self.feature_count}), "
                f"recebida {sequence.shape}"
            )

        # Mantém o mesmo contrato do runner de inferência que gerou o modelo beta.
        prepared = sequence.astype(np.float32)
        if self.model_feature_count == RAW_FEATURE_COUNT + 1:
            prepared = np.concatenate(
                [prepared, self._perclos(prepared).reshape(-1, 1)],
                axis=1,
            )
        clean = np.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self.scaler.transform(clean)
        reduced = self.pca.transform(scaled)
        tensor = self._torch.from_numpy(reduced[None, ...]).float()

        with self._torch.no_grad():
            logits = self.model(tensor)
            arrays = [
                self._torch.softmax(item, dim=-1).squeeze(0).cpu().numpy()
                for item in logits
            ]

        probabilities: Dict[str, List[float]] = {}
        levels: Dict[str, int] = {}
        scores: Dict[str, float] = {}
        confidences: Dict[str, float] = {}

        level_axis = np.arange(4, dtype=np.float32)
        for task, probs in zip(TASK_NAMES, arrays):
            probs = np.asarray(probs, dtype=np.float32)
            level = int(np.argmax(probs))
            probabilities[task] = [float(value) for value in probs]
            levels[task] = level
            scores[task] = float(np.dot(probs, level_axis) / 3.0 * 100.0)
            confidences[task] = float(probs[level] * 100.0)

        return Prediction(probabilities, levels, scores, confidences)

