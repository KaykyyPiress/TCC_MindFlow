"""
training/feature_engineering.py
================================
Features derivadas, computadas em cima dos vetores já extraídos pelo
mindflow_extractor (outputs/features/*), sem precisar reprocessar os
vídeos do DAiSEE.

PERCLOS (PERcentage of eye CLOSure)
------------------------------------
Métrica clássica de sonolência/desengajamento (Wierwille & Ellsworth,
1994): fração do tempo, numa janela recente, em que os olhos estão
fechados. O bloco D do extrator já calcula o EAR (Eye Aspect Ratio,
Soukupová & Čech, 2016) por frame — D03_ear_mean, índice 70 do vetor de
120 features — então o PERCLOS pode ser derivado diretamente dos dados
já extraídos, sem tocar no extrator nem reprocessar os vídeos.

Isso é deliberadamente diferente do D15_drowsiness_proxy que já existe
no extrator: D15 usa WINDOWS.long=30 frames, uma constante calibrada
para 30fps (comentário original: "1s a 30fps") só que a extração
efetiva roda a ~6fps (skip_frames=4), então na prática D15 é uma janela
de ~5s, não ~1s. Aqui recalibramos para o fps real.
"""
from __future__ import annotations

import numpy as np

# Índice global do EAR médio (D03_ear_mean) no vetor de 120 features:
# bloco D começa no índice 68, D03 é a 3ª feature do bloco (0-indexed: +2).
EAR_MEAN_INDEX = 68 + 2

# Threshold de olho fechado — mesmo valor usado pelo extrator para o
# indicador de piscada (THRESHOLDS.blink_ear em mindflow_extractor/config.py),
# repetido aqui para não criar uma dependência de import entre os pacotes
# por uma única constante.
DEFAULT_EAR_THRESHOLD = 0.22


def compute_perclos(
    ear_mean:      np.ndarray,
    threshold:     float = DEFAULT_EAR_THRESHOLD,
    window_frames: int   = 6,
) -> np.ndarray:
    """
    PERCLOS causal por frame: fração de frames com olho fechado
    (EAR < threshold) nos `window_frames` frames anteriores, incluindo
    o frame atual.

    Frames com EAR ausente (NaN — rosto não detectado) são excluídos do
    denominador, não contados como olho aberto nem fechado. Se todos os
    frames da janela forem inválidos, o resultado é NaN nesse ponto
    (tratado depois pela mesma etapa de nan_to_num usada no resto do
    pipeline).

    window_frames=6 ≈ 1 segundo, calibrado para o fps efetivo real de
    extração (~6fps, skip_frames=4), não para os 30fps assumidos nos
    comentários originais do extrator.
    """
    n = len(ear_mean)
    valid  = ~np.isnan(ear_mean)
    closed = np.where(valid, ear_mean < threshold, False)

    perclos = np.full(n, np.nan, dtype=np.float32)
    for t in range(n):
        lo = max(0, t - window_frames + 1)
        n_valid = valid[lo: t + 1].sum()
        if n_valid == 0:
            continue
        perclos[t] = closed[lo: t + 1].sum() / n_valid

    return perclos


def add_perclos_column(
    arr:           np.ndarray,
    threshold:     float = DEFAULT_EAR_THRESHOLD,
    window_frames: int   = 6,
) -> np.ndarray:
    """
    Recebe arr (n_frames, 120) — o vetor bruto extraído, ainda com NaN
    não tratado — e retorna (n_frames, 121) com o PERCLOS causal
    concatenado como última coluna.

    Precisa ser chamado ANTES de qualquer nan_to_num em arr, porque o
    cálculo do PERCLOS depende de distinguir "olho fechado" de "rosto
    não detectado" (NaN), e nan_to_num zeraria o EAR, fazendo-o parecer
    sempre fechado.
    """
    ear_mean = arr[:, EAR_MEAN_INDEX]
    perclos  = compute_perclos(ear_mean, threshold=threshold, window_frames=window_frames)
    return np.concatenate([arr, perclos.reshape(-1, 1)], axis=1)
