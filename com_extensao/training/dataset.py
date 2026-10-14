"""
training/dataset.py
===================
Carregamento do dataset DAiSEE — 4 dimensões simultâneas.

Cada vídeo gera:
    <clip_id>.npy              — shape (N_frames, 120), float32
    <clip_id>_metadata.json    — labels das 4 dimensões DAiSEE

Labels (cada uma independente, 0–3):
    engagement   — nível de engajamento
    boredom      — nível de tédio
    confusion    — nível de confusão
    frustration  — nível de frustração
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from training.config import (
    MindFlowTrainingConfig,
    DEFAULT_CFG,
    FEATURES_DIR_TRAIN,
    FEATURES_DIR_VAL,
    FEATURES_DIR_TEST,
)
from training.feature_engineering import add_perclos_column

logger = logging.getLogger(__name__)

SPLIT_DIRS: Dict[str, Path] = {
    "train":      FEATURES_DIR_TRAIN,
    "validation": FEATURES_DIR_VAL,
    "test":       FEATURES_DIR_TEST,
}

# Nomes das 4 dimensões DAiSEE — ordem fixa, usada em todo o projeto
TASK_NAMES = ["engagement", "boredom", "confusion", "frustration"]

# Nomes amigáveis para exibição
TASK_DISPLAY_NAMES = ["Engajamento", "Tédio", "Confusão", "Frustração"]

# Nomes de nível por dimensão (0-3)
LEVEL_NAMES = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]


def load_split(
    features_dir: Path,
    sequence_len: int = 30,
    stride: int = 15,
    min_face_ratio: float = 0.3,
    max_videos: Optional[int] = None,
    perclos_window_frames: int = 6,
    perclos_ear_threshold: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega .npy + _metadata.json de um diretório.

    Parâmetros
    ----------
    max_videos : int ou None
        Limite de vídeos a carregar. Quando definido, usa seleção estratificada:
        inclui TODOS os vídeos de classes minoritárias de engajamento primeiro
        (garantindo que nenhuma classe fique com 0 amostras), depois preenche
        o restante em ordem alfabética até atingir max_videos.

    Retorna
    -------
    X         : (N, seq_len, 121)
    y         : (N, 4)  — [engagement, boredom, confusion, frustration] cada 0-3
    subjects  : (N,)    — subject_id para auditoria
    clip_ids  : (N,)    — id do clipe (nome do arquivo .npy) de origem de cada janela.
                Várias janelas podem compartilhar o mesmo clip_id (janelas
                sobrepostas do mesmo vídeo) — usado para agregar previsões
                por clipe em vez de por janela na avaliação.
    """
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Diretório de features não encontrado: {features_dir}\n"
            "Execute primeiro o video_runner."
        )

    npy_files = sorted(f for f in features_dir.glob("*.npy")
                       if not f.stem.endswith("_metadata"))

    if not npy_files:
        raise RuntimeError(f"Nenhum .npy em: {features_dir}")

    # --- Seleção estratificada (quando max_videos está definido) ---
    # Garante que todas as classes de engajamento tenham ao menos 1 amostra.
    # Passo 1: escaneia apenas os metadados (rápido, sem carregar .npy)
    # Passo 2: separa por classe de engajamento
    # Passo 3: inclui TODOS os vídeos de classes minoritárias
    # Passo 4: preenche o restante em ordem alfabética
    if max_videos is not None:
        from collections import defaultdict
        # Coleta válidos com label
        valid_files: List[Path] = []
        eng_class_of: Dict[str, int] = {}
        for npy_path in npy_files:
            meta_path = features_dir / f"{npy_path.stem}_metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                continue
            eng = meta.get("label")
            if eng is None:
                continue
            face_ratio = meta.get("quality", {}).get("face_detected_ratio", 1.0)
            if face_ratio < min_face_ratio:
                continue
            valid_files.append(npy_path)
            eng_class_of[npy_path.stem] = int(eng)

        # Agrupa por classe de engajamento
        by_class: Dict[int, List[Path]] = defaultdict(list)
        for p in valid_files:
            by_class[eng_class_of[p.stem]].append(p)

        # Define quota por classe (distribuição alvo igual)
        n_classes = len(by_class)
        quota = max_videos // n_classes if n_classes else max_videos

        # Força inclusão de classes com poucos vídeos (< quota)
        forced: List[Path] = []
        forced_stems: set = set()
        for cls in sorted(by_class):
            if len(by_class[cls]) < quota:
                forced.extend(by_class[cls])
                forced_stems.update(p.stem for p in by_class[cls])
                logger.info(
                    "  [estratificado] Classe %d: apenas %d vídeos — todos incluídos.",
                    cls, len(by_class[cls]),
                )

        # Preenche o restante em ordem alfabética
        remaining_needed = max_videos - len(forced)
        filler: List[Path] = []
        for p in valid_files:
            if remaining_needed <= 0:
                break
            if p.stem not in forced_stems:
                filler.append(p)
                remaining_needed -= 1

        npy_files = forced + filler
        logger.info(
            "  [estratificado] %d vídeos selecionados (%d forçados por classe rara + %d fill)",
            len(npy_files), len(forced), len(filler),
        )

    logger.info("Carregando %d arquivos de %s ...", len(npy_files), features_dir.name)

    all_X:        List[np.ndarray] = []
    all_y:        List[List[int]]  = []
    all_subjects: List[str]        = []
    all_clip_ids: List[str]        = []

    skipped = 0

    for npy_path in npy_files:
        meta_path = features_dir / f"{npy_path.stem}_metadata.json"
        if not meta_path.exists():
            skipped += 1
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Lê label principal (engagement) — obrigatório
        engagement = meta.get("label")
        if engagement is None:
            skipped += 1
            continue
        engagement = int(engagement)

        # Lê as outras 3 dimensões do metadata
        # Compatível tanto com metadata antigo (só engagement) quanto novo (4 labels)
        labels_4 = meta.get("labels_4d")
        if labels_4 is not None:
            row = [
                int(labels_4.get("engagement",  engagement)),
                int(labels_4.get("boredom",      0)),
                int(labels_4.get("confusion",    0)),
                int(labels_4.get("frustration",  0)),
            ]
        else:
            # Metadata antigo: só tem engagement; outras dimensões como 0
            row = [engagement, 0, 0, 0]

        # Filtra vídeos com rosto quase sempre ausente
        face_ratio = meta.get("quality", {}).get("face_detected_ratio", 1.0)
        if face_ratio < min_face_ratio:
            skipped += 1
            continue

        subject_id = str(meta.get("subject_id", "unknown"))

        try:
            arr = np.load(str(npy_path), allow_pickle=False)
        except Exception as e:
            logger.warning("Erro ao carregar %s: %s", npy_path.name, e)
            continue

        if arr.ndim != 2 or arr.shape[1] != 120:
            continue

        n_frames = arr.shape[0]
        if n_frames < sequence_len:
            skipped += 1
            continue

        # PERCLOS precisa do EAR bruto (NaN = rosto não detectado) antes
        # de qualquer nan_to_num — ver training/feature_engineering.py.
        arr = add_perclos_column(
            arr, threshold=perclos_ear_threshold, window_frames=perclos_window_frames,
        )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        clip_id = npy_path.stem
        start = 0
        while start + sequence_len <= n_frames:
            all_X.append(arr[start: start + sequence_len])
            all_y.append(row)
            all_subjects.append(subject_id)
            all_clip_ids.append(clip_id)
            start += stride

    if not all_X:
        raise RuntimeError(f"Nenhuma janela válida em {features_dir}.")

    X        = np.array(all_X,        dtype=np.float32)   # (N, seq, 121)
    y        = np.array(all_y,        dtype=np.int64)      # (N, 4)
    subjects = np.array(all_subjects, dtype=object)
    clip_ids = np.array(all_clip_ids, dtype=object)

    # Log da distribuição de cada dimensão
    n_valid = len(npy_files) - skipped
    for i, task in enumerate(TASK_NAMES):
        unique, counts = np.unique(y[:, i], return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info("  [%s] dist: %s", task, dist)

    logger.info(
        "  Carregado: %d janelas | %d vídeos úteis",
        len(X), n_valid,
    )
    return X, y, subjects, clip_ids


def load_dataset(
    cfg: MindFlowTrainingConfig = DEFAULT_CFG,
    splits: Tuple[str, ...] = ("train", "validation", "test"),
    max_videos_per_split: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Carrega os splits do DAiSEE.

    max_videos_per_split : dict opcional, ex: {"train": 1200, "validation": 400, "test": 400}
        Quando definido, usa seleção estratificada em cada split para garantir
        que nenhuma classe de engajamento fique sem amostras.
    """
    seq_len = cfg.features.sequence_len
    stride  = cfg.preprocess.sequence_stride

    result = {}
    for split in splits:
        if split not in SPLIT_DIRS:
            raise ValueError(f"Split inválido: '{split}'")
        max_v = max_videos_per_split.get(split) if max_videos_per_split else None
        logger.info("=== Carregando split: %s (max=%s) ===", split, max_v or "todos")
        result[split] = load_split(
            SPLIT_DIRS[split],
            sequence_len=seq_len,
            stride=stride,
            max_videos=max_v,
            perclos_window_frames=cfg.features.perclos_window_frames,
            perclos_ear_threshold=cfg.features.perclos_ear_threshold,
        )
    return result


class MindFlowDataset(Dataset):
    """
    Dataset PyTorch multi-output.

    X : (N, seq_len, 121)
    y : (N, 4)   — [engagement, boredom, confusion, frustration]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()   # (N, 4)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def compute_class_weights(self, task_idx: int = 0) -> torch.Tensor:
        """Pesos inversamente proporcionais à freq. de uma dimensão."""
        y_np = self.y[:, task_idx].numpy()
        classes, counts = np.unique(y_np, return_counts=True)
        n_total   = len(y_np)
        n_classes = 4
        weights = np.ones(n_classes, dtype=np.float32)
        for cls, cnt in zip(classes, counts):
            weights[int(cls)] = n_total / (n_classes * cnt)
        return torch.from_numpy(weights).float()
