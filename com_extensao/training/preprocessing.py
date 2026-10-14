"""
training/preprocessing.py
==========================
Pré-processamento do pipeline de treinamento do MindFlow AI.

Etapas (na ordem correta conforme metodologia):
  1. Normalização Z-score      ← fit apenas no treino
  2. BorderlineSMOTE           ← apenas no treino, pós-normalização
  3. PCA                       ← obrigatório, fit apenas no treino

Ordem é crítica:
  - Splits já vêm separados (divisão oficial DAiSEE Train/Val/Test)
  - Normalização fit SOMENTE no treino para evitar data leakage
  - SMOTE apenas no treino: val e test permanecem com distribuição real
  - PCA APÓS SMOTE: componentes calculados sobre espaço já balanceado

Por que BorderlineSMOTE?
  O DAiSEE tem viés severo: ~70-80% das amostras são "high engagement"
  (labels 2-3). O SMOTE básico gera sintéticos em regiões fáceis do
  centro da nuvem. O BorderlineSMOTE (Han et al., 2005) foca apenas
  nos exemplos FRONTEIRIÇOS — aqueles com vizinhos de classes diferentes
  — melhorando especificamente o F1 nas classes minoritárias sem inflar
  regiões que o modelo já classifica bem.
  kind="borderline-1": usa apenas amostras "borderline" reais como semente.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from training.config import MindFlowTrainingConfig, DEFAULT_CFG

logger = logging.getLogger(__name__)

SplitData = Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
]


# ---------------------------------------------------------------------------
# 1. Normalização Z-score
# ---------------------------------------------------------------------------

def fit_scaler(
    X_train: np.ndarray,
    scaler_type: str = "standard",
    save_path: Optional[Path] = None,
) -> object:
    """
    Ajusta o scaler APENAS nos dados de treino.

    X_train : (N_train, seq_len, 121)
    """
    scaler_map = {
        "standard": StandardScaler,
        "minmax":   MinMaxScaler,
        "robust":   RobustScaler,
    }
    if scaler_type not in scaler_map:
        raise ValueError(f"scaler_type inválido: {scaler_type}. Opções: {list(scaler_map)}")

    scaler = scaler_map[scaler_type]()

    # Reshape para 2D: (N*seq_len, 121) → fit → reshape de volta
    n, seq, feat = X_train.shape
    X_flat = X_train.reshape(-1, feat)
    scaler.fit(X_flat)
    logger.info("Scaler '%s' ajustado em %d amostras × %d features", scaler_type, *X_flat.shape)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info("Scaler salvo em: %s", save_path)

    return scaler


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    """Aplica o scaler a qualquer partição (train/val/test)."""
    n, seq, feat = X.shape
    X_flat = X.reshape(-1, feat)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(n, seq, feat)


# ---------------------------------------------------------------------------
# 2. BorderlineSMOTE
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanceia o conjunto de treino com BorderlineSMOTE.

    y_train pode ser:
      - (N,)   — single-label (engagement)
      - (N, 4) — multi-label [engagement, boredom, confusion, frustration]

    O SMOTE usa apenas engagement (dim 0) como critério de balanceamento.
    Os rótulos das outras dimensões são propagados usando os mesmos índices.

    X_train : (N, seq_len, 121) — já normalizado
    y_train : (N,) ou (N, 4)
    """
    multi_label = (y_train.ndim == 2)
    y_engagement = y_train[:, 0] if multi_label else y_train

    n, seq, feat = X_train.shape
    X_flat = X_train.reshape(n, seq * feat)

    counts = np.bincount(y_engagement)
    min_class_count = counts[counts > 0].min()
    k = min(k_neighbors, min_class_count - 1)
    if k < k_neighbors:
        logger.warning(
            "⚠️  k_neighbors reduzido de %d para %d (menor classe tem %d amostras)",
            k_neighbors, k, min_class_count,
        )
    if k < 1:
        logger.warning("⚠️  Classe muito pequena para SMOTE — pulando balanceamento.")
        return X_train, y_train

    smote = BorderlineSMOTE(
        k_neighbors=k,
        kind="borderline-1",
        random_state=seed,
    )
    X_resampled, y_eng_resampled = smote.fit_resample(X_flat, y_engagement)

    n_new = len(X_resampled)
    X_resampled = X_resampled.reshape(n_new, seq, feat)

    if multi_label:
        # Para amostras sintéticas, copia o label do sample original mais próximo
        # (SMOTE garante que amostras sintéticas herdam a classe da semente)
        # Reconstrói y completo usando os rótulos de engagement reamostrado
        n_orig = n
        y_resampled_full = np.zeros((n_new, y_train.shape[1]), dtype=np.int64)
        y_resampled_full[:n_orig] = y_train   # originais — preserva todos os labels
        for i in range(n_orig, n_new):
            # Sintético: usa o engagement que o SMOTE gerou; outras dims = 0 (conservador)
            y_resampled_full[i, 0] = y_eng_resampled[i]
        y_out = y_resampled_full
    else:
        y_out = y_eng_resampled

    unique, cnts = np.unique(y_eng_resampled, return_counts=True)
    logger.info(
        "BorderlineSMOTE: %d → %d amostras | nova dist engajamento: %s",
        n, n_new,
        dict(zip(unique.tolist(), cnts.tolist())),
    )
    return X_resampled, y_out


def apply_smote_multioutput(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    seed: int = 42,
    max_ratio: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica BorderlineSMOTE calibrado por dimensão no modo multi-output.

    Estratégia:
      - Roda o SMOTE para cada uma das 4 dimensões separadamente.
      - Limita sintéticos via max_ratio: cada classe minoritária é ampliada
        até no máximo max_ratio × n_majority daquela dimensão.
      - Labels das outras dimensões propagados do vizinho real mais próximo.

    Com max_ratio=0.25 e dataset completo (~16k janelas reais):
      - Adiciona ~2-3k sintéticos por dimensão = ~10k total
      - Final: ~26k amostras (37% sintético) — controlado e eficaz.

    Parâmetros
    ----------
    max_ratio : float
        Fração da classe majoritária até onde as minoritárias são ampliadas.
        0.25 = classes minoritárias chegam a 25% do tamanho da maior.
    """
    from sklearn.neighbors import NearestNeighbors

    n, seq, feat = X_train.shape
    n_dims = y_train.shape[1]
    X_flat = X_train.reshape(n, seq * feat)

    X_extra: list = []
    y_extra: list = []

    # Classes com menos de 5% da majoritária são "muito raras":
    # limitamos sintéticos a +100 para evitar interpolações de interpolações.
    VERY_RARE_RATIO  = 0.05   # limiar: < 5% da classe majoritária
    VERY_RARE_EXTRA  = 100    # máximo de sintéticos para classes muito raras

    for dim_idx in range(n_dims):
        y_dim = y_train[:, dim_idx]
        counts = np.bincount(y_dim)
        n_majority = int(counts.max())
        ratio_target = max(1, int(n_majority * max_ratio))

        # Define sampling_strategy: amplia só classes menores que a maioria
        strategy = {}
        for cls, cnt in enumerate(counts):
            if cnt == 0:
                continue
            if cnt < n_majority:
                # Classe muito rara: adiciona apenas VERY_RARE_EXTRA sintéticos
                if cnt < n_majority * VERY_RARE_RATIO:
                    t = cnt + VERY_RARE_EXTRA
                    logger.info(
                        "  Dim %d cls %d: muito rara (%d amostras) → +%d sintéticos (cap aplicado)",
                        dim_idx, cls, cnt, VERY_RARE_EXTRA,
                    )
                else:
                    t = ratio_target
                if t > cnt:
                    strategy[cls] = t

        if not strategy:
            logger.info("  Dim %d: já balanceada — pulando SMOTE.", dim_idx)
            continue

        min_count = min(counts[cls] for cls in strategy)
        k = min(k_neighbors, min_count - 1)
        if k < 1:
            logger.warning("  ⚠️  Dim %d: classe muito pequena (k<1) — pulando.", dim_idx)
            continue

        smote = BorderlineSMOTE(
            sampling_strategy=strategy,
            k_neighbors=k,
            kind="borderline-1",
            random_state=seed + dim_idx,
        )

        try:
            X_res, y_res_dim = smote.fit_resample(X_flat, y_dim)
        except Exception as e:
            logger.warning("  ⚠️  Dim %d: SMOTE falhou (%s) — pulando.", dim_idx, e)
            continue

        n_synthetic = len(X_res) - n
        if n_synthetic == 0:
            continue

        # Propaga labels das outras dimensões via vizinho mais próximo
        X_synthetic_flat = X_res[n:]
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_flat)
        _, seed_indices = nbrs.kneighbors(X_synthetic_flat)
        seed_indices = seed_indices.flatten()

        y_synthetic = np.zeros((n_synthetic, n_dims), dtype=np.int64)
        for i, sidx in enumerate(seed_indices):
            y_synthetic[i] = y_train[sidx]       # labels reais do vizinho
        y_synthetic[:, dim_idx] = y_res_dim[n:]  # sobrescreve dim alvo

        X_extra.append(X_synthetic_flat.reshape(n_synthetic, seq, feat))
        y_extra.append(y_synthetic)

        unique, cnts = np.unique(y_res_dim, return_counts=True)
        logger.info(
            "  Dim %d: +%d sintéticos (ratio=%.0f%%) | dist: %s",
            dim_idx, n_synthetic, max_ratio * 100,
            dict(zip(unique.tolist(), cnts.tolist())),
        )

    if X_extra:
        X_out = np.concatenate([X_train] + X_extra, axis=0)
        y_out = np.concatenate([y_train] + y_extra, axis=0)
        n_total_synthetic = len(X_out) - n
        logger.info(
            "  BorderlineSMOTE multi-output: %d → %d amostras (%.1f%% sintético)",
            n, len(X_out), 100 * n_total_synthetic / len(X_out),
        )
    else:
        logger.warning("⚠️  SMOTE multi-output não gerou nenhum sintético.")
        X_out, y_out = X_train, y_train

    return X_out, y_out



# ---------------------------------------------------------------------------
# 3. PCA (obrigatório)
# ---------------------------------------------------------------------------

def fit_pca(
    X_train: np.ndarray,
    variance_threshold: float = 0.95,
    save_path: Optional[Path] = None,
) -> PCA:
    """
    Ajusta PCA no treino. Reduz a dimensão de features retendo
    `variance_threshold` da variância explicada.

    PCA é aplicado por frame (2D), não por sequência.
    """
    n, seq, feat = X_train.shape
    X_flat = X_train.reshape(-1, feat)

    pca = PCA(n_components=variance_threshold, svd_solver="full", random_state=42)
    pca.fit(X_flat)

    logger.info(
        "PCA ajustado: %d → %d componentes (%.1f%% variância retida)",
        feat, pca.n_components_, 100 * variance_threshold,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, save_path)
        logger.info("PCA salvo em: %s", save_path)

    return pca


def apply_pca(X: np.ndarray, pca: PCA) -> np.ndarray:
    """Aplica PCA a qualquer partição."""
    n, seq, feat = X.shape
    X_flat = X.reshape(-1, feat)
    X_reduced = pca.transform(X_flat)
    n_components = X_reduced.shape[1]
    return X_reduced.reshape(n, seq, n_components)


# ---------------------------------------------------------------------------
# Função orquestradora (recebe splits já separados)
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cfg: MindFlowTrainingConfig = DEFAULT_CFG,
    save_artifacts: bool = True,
) -> Tuple[SplitData, Dict]:
    """
    Executa o pipeline completo de pré-processamento na ordem correta.

    Recebe os 3 splits já separados (divisão oficial do DAiSEE).

    Retorna
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test), artifacts_dict
    """
    p = cfg.preprocess

    multi_output = (y_train.ndim == 2 and y_train.shape[1] > 1)

    logger.info("=" * 55)
    logger.info("PIPELINE DE PRÉ-PROCESSAMENTO")
    logger.info("=" * 55)
    logger.info("  Train : %s  Val: %s  Test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    if multi_output:
        logger.info("  Modo: multi-output (%d dimensões)", y_train.shape[1])

    # 1. Normalização (fit SOMENTE no treino)
    logger.info("[1/3] Normalização Z-score (fit no treino)")
    scaler = fit_scaler(
        X_train,
        scaler_type=p.scaler,
        save_path=cfg.scaler_path if save_artifacts else None,
    )
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    # 2. BorderlineSMOTE calibrado (somente no treino)
    #    - Multi-output: 1 rodada usando Engajamento como target + propagação
    #      dos outros labels via vizinho mais próximo. Ratio máximo: 25%.
    #    - Single-output: SMOTE direto no Engajamento.
    logger.info("[2/3] BorderlineSMOTE calibrado (somente no treino)")
    smote_ratio = getattr(p, "smote_max_ratio", 0.25)
    if multi_output:
        X_train, y_train = apply_smote_multioutput(
            X_train, y_train,
            k_neighbors=p.smote_k_neighbors,
            seed=p.smote_random_state,
            max_ratio=smote_ratio,
        )
    else:
        X_train, y_train = apply_smote(
            X_train, y_train,
            k_neighbors=p.smote_k_neighbors,
            seed=p.smote_random_state,
        )


    # 3. PCA (obrigatório)
    logger.info("[3/3] PCA (fit no treino pós-SMOTE, aplicado em todos os splits)")
    pca = fit_pca(
        X_train,
        variance_threshold=p.pca_variance_threshold,
        save_path=cfg.pca_path if save_artifacts else None,
    )
    X_train = apply_pca(X_train, pca)
    X_val   = apply_pca(X_val,   pca)
    X_test  = apply_pca(X_test,  pca)

    logger.info("=" * 55)
    logger.info("Pré-processamento concluído.")
    logger.info(
        "  X_train: %s  X_val: %s  X_test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    logger.info("  Dimensão após PCA: %d componentes", X_train.shape[2])
    logger.info("=" * 55)

    artifacts = {"scaler": scaler, "pca": pca}
    return (X_train, y_train, X_val, y_val, X_test, y_test), artifacts
