"""
training/evaluate.py
====================
Avaliação multi-output do modelo MindFlow AI no conjunto de teste.

Reporta métricas para cada uma das 4 dimensões DAiSEE:
  engagement, boredom, confusion, frustration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.config import MindFlowTrainingConfig, DEFAULT_CFG
from training.dataset import MindFlowDataset, TASK_NAMES, TASK_DISPLAY_NAMES, LEVEL_NAMES
from training.model import MindFlowLSTM, build_model, load_model

logger = logging.getLogger(__name__)

# Nomes de nível para exibição (0-3)
_LEVEL_LABELS = ["very_low", "low", "high", "very_high"]


def run_inference(
    model:      MindFlowLSTM,
    X:          np.ndarray,
    device:     torch.device,
    batch_size: int = 128,
    target_task: Optional[str] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Roda inferência (pode ser multi-output ou single-output).

    Retorna
    -------
    preds : (N, n_tasks)  — classe predita por dimensão
    probs : lista de n_tasks arrays — softmax (N, 4) no modo nominal,
            ou sigmoid cumulativo (N, 3) no modo ordinal (CORAL)
    """
    model = model.to(device)
    model.eval()

    n_tasks = model.n_tasks
    ordinal = getattr(model, "ordinal", False)
    all_preds: List[List[int]] = []
    all_probs: List[List[np.ndarray]] = [[] for _ in range(n_tasks)]

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_bat = torch.from_numpy(X[start: start + batch_size]).float().to(device)
            logits_list = model(x_bat)   # lista de n_tasks tensors — (B, 4) nominal ou (B, 3) ordinal

            batch_preds = []
            for t, logits in enumerate(logits_list):
                if ordinal:
                    # sigmoid cumulativo: P(nível > k). Classe predita = nº de thresholds ultrapassados.
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).sum(dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    preds = probs.argmax(dim=-1)
                batch_preds.append(preds.cpu().numpy())
                all_probs[t].extend(probs.cpu().numpy().tolist())

            # combina em (B, n_tasks)
            batch_preds_np = np.stack(batch_preds, axis=1)
            all_preds.extend(batch_preds_np.tolist())

    preds = np.array(all_preds, dtype=np.int64)         # (N, n_tasks)
    probs = [np.array(p) for p in all_probs]             # n_tasks × (N, 4)
    return preds, probs


def compute_metrics_per_task(
    y_true: np.ndarray,   # (N, 4)
    y_pred: np.ndarray,   # (N, n_tasks)
    target_task: Optional[str] = None,
) -> Dict:
    """Calcula métricas para cada dimensão separadamente."""
    results = {}

    tasks_to_eval = [target_task] if target_task is not None else TASK_NAMES

    for idx, task in enumerate(tasks_to_eval):
        t = TASK_NAMES.index(task)
        yt = y_true[:, t]
        # Se for single-task, predição tem apenas 1 coluna (idx 0). Caso contrário, pega a coluna t.
        yp = y_pred[:, 0] if target_task is not None else y_pred[:, t]
        display = TASK_DISPLAY_NAMES[t]

        acc      = accuracy_score(yt, yp)
        f1_mac   = f1_score(yt, yp, average="macro",    zero_division=0)
        f1_wei   = f1_score(yt, yp, average="weighted", zero_division=0)
        prec_mac = precision_score(yt, yp, average="macro", zero_division=0)
        rec_mac  = recall_score(yt, yp,    average="macro", zero_division=0)
        cm       = confusion_matrix(yt, yp, labels=[0, 1, 2, 3]).tolist()

        per_level = {}
        for i, lbl in enumerate(_LEVEL_LABELS):
            per_level[lbl] = {
                "support":   int((yt == i).sum()),
                "precision": float(precision_score(yt, yp, labels=[i], average="macro", zero_division=0)),
                "recall":    float(recall_score(yt, yp,    labels=[i], average="macro", zero_division=0)),
                "f1":        float(f1_score(yt, yp,        labels=[i], average="macro", zero_division=0)),
            }

        report_str = classification_report(
            yt, yp,
            labels=list(range(4)),
            target_names=_LEVEL_LABELS,
            zero_division=0,
        )

        results[task] = {
            "display_name":          display,
            "accuracy":              round(float(acc),      4),
            "f1_macro":              round(float(f1_mac),   4),
            "f1_weighted":           round(float(f1_wei),   4),
            "precision_macro":       round(float(prec_mac), 4),
            "recall_macro":          round(float(rec_mac),  4),
            "confusion_matrix":      cm,
            "per_level":             per_level,
            "classification_report": report_str,
        }

    return results


def aggregate_by_clip(
    y_true:   np.ndarray,   # (N, 4)
    y_pred:   np.ndarray,   # (N, n_tasks)
    clip_ids: np.ndarray,   # (N,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Agrega janelas do mesmo clipe em uma única previsão por clipe.

    O DAiSEE rotula o vídeo inteiro (~10s), mas o pipeline gera várias
    janelas sobrepostas por vídeo (30 frames, stride 15). Avaliar janela
    a janela mede um problema mais granular do que o rótulo realmente
    descreve. Aqui cada clipe vira 1 amostra: o rótulo verdadeiro (igual
    em todas as suas janelas) e a predição por voto majoritário entre
    as janelas daquele clipe, por tarefa.
    """
    unique_clips = np.unique(clip_ids)
    agg_true: List[List[int]] = []
    agg_pred: List[List[int]] = []

    for clip in unique_clips:
        mask = clip_ids == clip
        agg_true.append(y_true[mask][0].tolist())

        preds_clip = y_pred[mask]   # (n_janelas_do_clipe, n_tasks)
        modes = []
        for t in range(preds_clip.shape[1]):
            vals, counts = np.unique(preds_clip[:, t], return_counts=True)
            modes.append(int(vals[np.argmax(counts)]))
        agg_pred.append(modes)

    return np.array(agg_true, dtype=np.int64), np.array(agg_pred, dtype=np.int64)


def print_metrics(task_metrics: Dict, target_task: Optional[str] = None, level: str = "janela") -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    model_type = f"single-output: {target_task}" if target_task is not None else "multi-output"
    print(f"  RESULTADOS NO CONJUNTO DE TESTE — MindFlow AI ({model_type}) — por {level}")
    print(sep)

    for task, m in task_metrics.items():
        print(f"\n  [{m['display_name']}]")
        print(f"    Accuracy       : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
        print(f"    F1-macro       : {m['f1_macro']:.4f}")
        print(f"    F1-weighted    : {m['f1_weighted']:.4f}")
        print(f"    Precision-macro: {m['precision_macro']:.4f}")
        print(f"    Recall-macro   : {m['recall_macro']:.4f}")
        print(f"    Por nível:")
        for lvl, stats in m["per_level"].items():
            print(f"      {lvl:10s}: prec={stats['precision']:.3f}  "
                  f"rec={stats['recall']:.3f}  f1={stats['f1']:.3f}  n={stats['support']}")

    print(f"\n{sep}\n")


def evaluate_model(
    model:     Optional[MindFlowLSTM] = None,
    X_test:    Optional[np.ndarray]   = None,
    y_test:    Optional[np.ndarray]   = None,
    cfg:       MindFlowTrainingConfig  = DEFAULT_CFG,
    input_dim: Optional[int]           = None,
    save:      bool = True,
    clip_ids:  Optional[np.ndarray]    = None,
) -> Dict:
    device = torch.device("cpu")
    target_task = cfg.train.target_task if hasattr(cfg.train, "target_task") else None

    if model is not None:
        model = model.to(device)

    if model is None:
        if not cfg.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {cfg.model_path}.")
        model = load_model(cfg.model_path, cfg=cfg, input_dim=input_dim, device="cpu")

    if X_test is None or y_test is None:
        raise ValueError("X_test e y_test devem ser fornecidos.")

    logger.info("Rodando inferência em %d amostras de teste...", len(X_test))
    y_pred, _ = run_inference(model, X_test, device, batch_size=cfg.train.batch_size, target_task=target_task)

    # y_test pode ser (N,) do dataset antigo ou (N, 4) do novo
    if y_test.ndim == 1:
        # compatibilidade: expande para (N, 4) com engagement na col 0 e 0 nas demais
        y_test_4d = np.zeros((len(y_test), 4), dtype=np.int64)
        y_test_4d[:, 0] = y_test
        y_test = y_test_4d

    task_metrics = compute_metrics_per_task(y_true=y_test, y_pred=y_pred, target_task=target_task)
    print_metrics(task_metrics, target_task=target_task, level="janela")

    clip_metrics: Optional[Dict] = None
    if clip_ids is not None:
        y_true_clip, y_pred_clip = aggregate_by_clip(y_test, y_pred, clip_ids)
        logger.info("Agregação por clipe: %d janelas → %d clipes únicos.",
                    len(y_test), len(y_true_clip))
        clip_metrics = compute_metrics_per_task(y_true=y_true_clip, y_pred=y_pred_clip, target_task=target_task)
        print_metrics(clip_metrics, target_task=target_task, level="clipe")

    if save:
        metrics_path = cfg.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_to_save = {
            task: {k: v for k, v in m.items() if k != "classification_report"}
            for task, m in task_metrics.items()
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info("Métricas salvas em: %s", metrics_path)

        report_path = cfg.report_path
        with open(report_path, "w") as f:
            for task, m in task_metrics.items():
                f.write(f"\n=== {m['display_name']} ===\n")
                f.write(m["classification_report"])
        logger.info("Relatório salvo em: %s", report_path)

        if clip_metrics is not None:
            clip_metrics_path = cfg.output_dir / "metrics_by_clip.json"
            clip_metrics_to_save = {
                task: {k: v for k, v in m.items() if k != "classification_report"}
                for task, m in clip_metrics.items()
            }
            with open(clip_metrics_path, "w") as f:
                json.dump(clip_metrics_to_save, f, indent=2)
            logger.info("Métricas por clipe salvas em: %s", clip_metrics_path)

            clip_report_path = cfg.output_dir / "classification_report_by_clip.txt"
            with open(clip_report_path, "w") as f:
                for task, m in clip_metrics.items():
                    f.write(f"\n=== {m['display_name']} (por clipe) ===\n")
                    f.write(m["classification_report"])
            logger.info("Relatório por clipe salvo em: %s", clip_report_path)

    if clip_metrics is not None:
        return {"window": task_metrics, "clip": clip_metrics}
    return task_metrics
