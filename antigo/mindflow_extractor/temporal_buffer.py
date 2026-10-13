"""
Buffer temporal circular para suporte às features dinâmicas (Δ e variâncias).

Justificativa de design:
- Várias features (A25-A32, B07-B09, C06-C13, C24, D05-D07, D16, E02-E03,
  F20-F23) dependem do histórico de features escalares dos últimos N frames.
- Manter um histórico genérico no nível do pipeline evita que cada bloco
  de features reimplemente sua própria janela.
- O buffer é simples (deque) e leve, alinhado ao princípio "sem GPU + baixo
  custo computacional" da metodologia.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np


class TemporalBuffer:
    """
    Mantém histórico de features escalares por nome, com janela máxima fixa.

    Uso:
        buf = TemporalBuffer(max_window=30)
        buf.append("ear_mean", 0.27)
        delta = buf.delta("ear_mean")          # f(t) - f(t-1)
        sigma_5 = buf.variance("ear_mean", 5)  # desvio-padrão em 5 frames
    """

    def __init__(self, max_window: int = 30) -> None:
        self.max_window: int = max_window
        self._history: Dict[str, Deque[float]] = {}

    # ---------------------- escrita ----------------------
    def append(self, name: str, value: float) -> None:
        """Adiciona valor ao histórico de uma feature nomeada."""
        if name not in self._history:
            self._history[name] = deque(maxlen=self.max_window)
        self._history[name].append(float(value))

    def append_many(self, values: Dict[str, float]) -> None:
        """Atalho para inserir várias features de uma vez."""
        for name, value in values.items():
            self.append(name, value)

    # ---------------------- leitura ----------------------
    def get(self, name: str) -> Optional[float]:
        """Valor mais recente, ou None se inexistente."""
        h = self._history.get(name)
        return h[-1] if h else None

    def delta(self, name: str) -> float:
        """
        Diferença f(t) - f(t-1). Retorna 0.0 se ainda não há histórico
        suficiente (primeiro frame de uma sessão).
        """
        h = self._history.get(name)
        if h is None or len(h) < 2:
            return 0.0
        return float(h[-1] - h[-2])

    def variance(self, name: str, window: int) -> float:
        """
        Desvio-padrão amostral em janela <window> dos últimos valores.
        Retorna 0.0 se histórico insuficiente.

        Observação: usamos desvio-padrão (σ) em vez de variância (σ²)
        para manter as features na mesma escala das features originais,
        facilitando interpretação e normalização Z-score posterior.
        """
        h = self._history.get(name)
        if h is None or len(h) < 2:
            return 0.0
        window = min(window, len(h))
        recent = list(h)[-window:]
        return float(np.std(recent, ddof=0))

    def consecutive_true(self, name: str) -> int:
        """
        Conta quantos frames consecutivos (a partir do mais recente) o valor
        foi 1.0. Útil para features como D09 (frames_since_blink) e
        E05 (mouth_open_duration).

        Convenção: 1.0 = evento ativo; 0.0 = evento inativo.
        Quando se quer "frames desde o último evento", inverter na chamada.
        """
        h = self._history.get(name)
        if h is None:
            return 0
        count = 0
        for v in reversed(h):
            if v >= 0.5:
                count += 1
            else:
                break
        return count

    def fraction_true(self, name: str, window: int) -> float:
        """Fração de frames em janela com valor binário == 1."""
        h = self._history.get(name)
        if h is None:
            return 0.0
        window = min(window, len(h))
        recent = list(h)[-window:]
        return float(sum(v >= 0.5 for v in recent) / window)

    # ---------------------- utilitários ----------------------
    def reset(self) -> None:
        """Limpa todo o histórico — chamado entre sessões/vídeos."""
        self._history.clear()

    def __len__(self) -> int:
        """Número de features rastreadas."""
        return len(self._history)
