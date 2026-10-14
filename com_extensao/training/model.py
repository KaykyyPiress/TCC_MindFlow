"""
training/model.py
=================
LSTM multi-output do MindFlow AI — prediz 4 dimensões simultaneamente.

Arquitetura:
    Input(30, D) → LSTM(128) → Dropout → LSTM(64) → 4 cabeças independentes

Cada cabeça:
    Dense(64 → 4)  → Softmax   (4 níveis: 0=muito baixo … 3=muito alto)

Dimensões de saída:
    head 0: Engajamento   (engagement)
    head 1: Tédio         (boredom)
    head 2: Confusão      (confusion)
    head 3: Frustração    (frustration)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from training.config import MindFlowTrainingConfig, DEFAULT_CFG
from training.dataset import TASK_NAMES

logger = logging.getLogger(__name__)

N_TASKS   = 4   # engagement, boredom, confusion, frustration
N_LEVELS  = 4   # 0 … 3


class CoralHead(nn.Module):
    """
    Cabeça ordinal CORAL (Cao, Mirjalili & Raschka, 2020).

    Em vez de n_levels neurônios independentes com softmax, usa 1 único
    vetor de pesos compartilhado (garante que os n_levels-1 logits sejam
    projeções da mesma direção no espaço de features) somado a n_levels-1
    thresholds monotonicamente decrescentes. Cada logit_k responde à
    pergunta cumulativa "o nível é maior que k?" — sigmoid(logit_k) é
    P(nível > k). Rank-consistency (P(nível>0) >= P(nível>1) >= ...) é
    garantida por construção, não por regularização da loss.
    """

    def __init__(self, in_features: int, n_levels: int = N_LEVELS):
        super().__init__()
        self.n_levels = n_levels
        n_thresholds  = n_levels - 1
        self.shared   = nn.Linear(in_features, 1, bias=False)
        self.bias_0   = nn.Parameter(torch.zeros(1))
        # gaps entre thresholds consecutivos — softplus garante > 0,
        # logo os thresholds só podem decrescer, nunca inverter a ordem.
        self.raw_gaps = nn.Parameter(torch.zeros(n_thresholds - 1)) if n_thresholds > 1 else None

    def thresholds(self) -> torch.Tensor:
        if self.raw_gaps is None:
            return self.bias_0
        gaps = torch.nn.functional.softplus(self.raw_gaps)
        rest = self.bias_0 - torch.cumsum(gaps, dim=0)
        return torch.cat([self.bias_0, rest])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shared_logit = self.shared(h)          # (batch, 1)
        return shared_logit + self.thresholds()  # broadcast → (batch, n_levels-1)


class MindFlowLSTM(nn.Module):
    """
    LSTM multi-output para classificação de estados cognitivos.

    Parâmetros
    ----------
    input_dim : dimensão do vetor por frame (n_components do PCA)
    hidden_1  : unidades da 1ª camada LSTM (padrão 128)
    hidden_2  : unidades da 2ª camada LSTM (padrão 64)
    n_tasks   : número de dimensões a prever (padrão 4)
    n_levels  : número de classes por dimensão (padrão 4)
    dropout   : taxa de dropout entre camadas (padrão 0.3)
    ordinal   : usa cabeça CORAL (ordinal) em vez de Linear+softmax.
                Só suportado com n_tasks=1 (modo single-task).
    """

    def __init__(
        self,
        input_dim:  int   = 120,
        hidden_1:   int   = 128,
        hidden_2:   int   = 64,
        n_tasks:    int   = N_TASKS,
        n_levels:   int   = N_LEVELS,
        dropout:    float = 0.3,
        task_names: Optional[List[str]] = None,
        ordinal:    bool  = False,
        # Aliases de compatibilidade com código antigo
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        if ordinal and n_tasks != 1:
            raise ValueError("ordinal=True só é suportado com n_tasks=1 (modo single-task).")

        self.input_dim  = input_dim
        self.hidden_1   = hidden_1
        self.hidden_2   = hidden_2
        self.n_tasks    = n_tasks
        self.n_levels   = n_levels
        self.task_names = task_names or TASK_NAMES
        self.ordinal    = ordinal

        self.lstm1 = nn.LSTM(input_dim,  hidden_1, batch_first=True)
        self.drop  = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(hidden_1,   hidden_2, batch_first=True)

        if ordinal:
            self.heads = nn.ModuleList([CoralHead(hidden_2, n_levels)])
        else:
            # cabeças independentes — uma por dimensão DAiSEE
            self.heads = nn.ModuleList([
                nn.Linear(hidden_2, n_levels) for _ in range(n_tasks)
            ])

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1)  # forget gate bias = 1

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x       : (batch, seq_len, input_dim)
        retorna : lista de n_tasks tensors, cada (batch, n_levels) — logits sem softmax
        """
        out1, _ = self.lstm1(x)
        out1    = self.drop(out1)
        out2, _ = self.lstm2(out1)
        h       = out2[:, -1, :]             # último hidden state: (batch, hidden_2)
        return [head(h) for head in self.heads]

    def predict_proba(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Lista de probabilidades por dimensão — para inferência.
        Nominal: softmax, shape (batch, n_levels).
        Ordinal: sigmoid cumulativo, shape (batch, n_levels-1) — P(nível > k).
        """
        with torch.no_grad():
            logits_list = self.forward(x)
            if self.ordinal:
                return [torch.sigmoid(lg) for lg in logits_list]
            return [torch.softmax(lg, dim=-1) for lg in logits_list]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Tensor (batch, n_tasks) com a classe predita por dimensão."""
        probs = self.predict_proba(x)
        if self.ordinal:
            # nível predito = quantos thresholds cumulativos foram ultrapassados
            return torch.stack([(p > 0.5).sum(dim=-1) for p in probs], dim=1)
        return torch.stack([p.argmax(dim=-1) for p in probs], dim=1)

    def summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        heads_str = "  ".join(f"head_{i}({self.task_names[i]})" for i in range(self.n_tasks))
        model_type = "single-output" if self.n_tasks == 1 else "multi-output"
        if self.ordinal:
            head_desc = f"CORAL({self.hidden_2} → {self.n_levels - 1} thresholds cumulativos)"
        else:
            head_desc = f"Linear({self.hidden_2} → {self.n_levels})"
        return "\n".join([
            f"MindFlowLSTM ({model_type}{', ordinal' if self.ordinal else ''})",
            f"  Input       : (batch, seq, {self.input_dim})",
            f"  LSTM 1      : {self.input_dim} → {self.hidden_1}",
            f"  Dropout",
            f"  LSTM 2      : {self.hidden_1} → {self.hidden_2}",
            f"  Heads       : {heads_str}",
            f"  Cada cabeça : {head_desc}",
            f"  Parâmetros  : {total:,} total | {trainable:,} treináveis",
        ])


def build_model(
    cfg: MindFlowTrainingConfig = DEFAULT_CFG,
    input_dim: Optional[int] = None,
) -> MindFlowLSTM:
    dim = input_dim if input_dim is not None else cfg.features.vector_dim

    n_tasks = N_TASKS
    task_names = list(TASK_NAMES)
    target_task = getattr(cfg.train, "target_task", None)
    ordinal = getattr(cfg.train, "ordinal", False)

    if target_task is not None:
        n_tasks = 1
        task_names = [target_task]
    elif ordinal:
        raise ValueError("cfg.train.ordinal=True requer cfg.train.target_task definido (modo single-task).")

    model = MindFlowLSTM(
        input_dim=dim,
        hidden_1=cfg.model.lstm_hidden_1,
        hidden_2=cfg.model.lstm_hidden_2,
        n_tasks=n_tasks,
        n_levels=N_LEVELS,
        dropout=cfg.model.dropout,
        task_names=task_names,
        ordinal=ordinal,
    )
    logger.info("Modelo construído:\n%s", model.summary())
    return model


def load_model(
    path: Path,
    cfg: MindFlowTrainingConfig = DEFAULT_CFG,
    input_dim: Optional[int] = None,
    device: str = "cpu",
) -> MindFlowLSTM:
    model = build_model(cfg, input_dim=input_dim)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
