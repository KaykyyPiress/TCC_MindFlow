"""
training/train.py
=================
Loop de treinamento multi-output do MindFlow AI.

Loss total = soma das 4 FocalLoss (uma por dimensão DAiSEE).
Focal Loss (Lin et al., 2017) foca o gradiente nos exemplos difíceis
(classes minoritárias) e ignora os fáceis (classe majoritária).
Early stopping e checkpoint monitoram o val_f1_macro.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from training.config import MindFlowTrainingConfig, DEFAULT_CFG
from training.dataset import MindFlowDataset, load_dataset, TASK_NAMES
from training.model import MindFlowLSTM, build_model
from training.preprocessing import preprocess_pipeline


# ---------------------------------------------------------------------------
# Focal Loss — estado da arte para classificação com imbalance severo
# Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
# gamma=2: penaliza 4x mais um exemplo com p_t=0.5 do que p_t=0.9
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss com class weights (alpha) e fator de foco (gamma).

    - gamma=0 → equivale ao CrossEntropyLoss ponderado
    - gamma=2 → foco forte nos exemplos difíceis (padrão RetinaNet)
    """
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.weight = weight          # class weights (alpha_t)
        self.gamma  = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log_softmax estável numericamente
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()

        # Coleta p_t e log_p_t da classe correta
        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        # alpha_t: peso da classe correta
        alpha_t = self.weight[targets]

        # Focal term: (1 - p_t)^gamma
        focal   = (1.0 - pt) ** self.gamma

        loss = -alpha_t * focal * log_pt
        return loss.mean()


# ---------------------------------------------------------------------------
# CORAL Loss — regressão ordinal via classificadores binários cumulativos
# Cao, Mirjalili & Raschka, "Rank consistent ordinal regression for neural
# networks with application to age estimation", Pattern Recognition Letters 2020
# ---------------------------------------------------------------------------
class CoralLoss(nn.Module):
    """
    Loss para a cabeça CoralHead (ver training/model.py).

    logits  : (B, n_levels-1) — thresholds cumulativos, logit_k = P(nível > k)
    targets : (B,) — classe inteira 0..n_levels-1

    Converte o rótulo inteiro em n_levels-1 alvos binários cumulativos
    (nível > 0?, nível > 1?, nível > 2?) e aplica BCE em cada um. Como os
    thresholds compartilham o mesmo vetor de pesos (garantido pela
    CoralHead), errar "por 1 nível" custa menos loss do que errar "por 3".
    """
    def __init__(self, n_levels: int = 4):
        super().__init__()
        self.n_levels = n_levels

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        thresholds  = torch.arange(self.n_levels - 1, device=targets.device)
        targets_bin = (targets.unsqueeze(1) > thresholds).float()
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_bin)


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)
    return device


def _run_epoch(
    model:      MindFlowLSTM,
    loader:     DataLoader,
    criterions: List[nn.Module],
    optimizer:  Optional[torch.optim.Optimizer],
    device:     torch.device,
    phase:      str,
    target_task: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Executa uma época (pode ser multi-output ou single-output dependendo de target_task).
    Retorna (loss_mean, accuracy, f1_macro). No modo multi-output, accuracy
    e F1 são a média das quatro tarefas, para o checkpoint não favorecer
    somente Engajamento.
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    all_preds:   List = []
    all_labels:  List = []

    task_idx = TASK_NAMES.index(target_task) if target_task is not None else None
    ordinal  = getattr(model, "ordinal", False)
    context  = torch.enable_grad if is_train else torch.no_grad

    with context():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)         # (B, seq, dim)
            y_batch = y_batch.to(device)         # (B, 4)

            logits_list = model(X_batch)          # logits

            if task_idx is not None:
                # Single-output: logits_list contém 1 tensor
                loss = criterions[0](logits_list[0], y_batch[:, task_idx])
            else:
                # Multi-output: logits_list contém 4 tensors
                loss = sum(
                    criterions[t](logits_list[t], y_batch[:, t])
                    for t in range(len(TASK_NAMES))
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)

            if task_idx is not None:
                logits_to_eval = logits_list[0]
                if ordinal:
                    # nível predito = quantos thresholds cumulativos foram ultrapassados
                    preds = (torch.sigmoid(logits_to_eval) > 0.5).sum(dim=-1)
                else:
                    preds = logits_to_eval.argmax(dim=-1)
                labels = y_batch[:, task_idx]
            else:
                # Uma coluna por tarefa, para medir as quatro de forma justa.
                preds = torch.stack(
                    [task_logits.argmax(dim=-1) for task_logits in logits_list],
                    dim=1,
                )
                labels = y_batch[:, :len(TASK_NAMES)]

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels)
    preds_np  = np.asarray(all_preds)
    n         = len(labels_np)
    loss_mean = total_loss / n

    if labels_np.ndim == 1:
        accuracy = float((preds_np == labels_np).mean())
        f1_macro = f1_score(
            labels_np, preds_np, labels=list(range(4)), average="macro", zero_division=0,
        )
    else:
        task_accuracies = [
            float((preds_np[:, task_idx] == labels_np[:, task_idx]).mean())
            for task_idx in range(labels_np.shape[1])
        ]
        task_f1 = [
            f1_score(
                labels_np[:, task_idx], preds_np[:, task_idx],
                labels=list(range(4)), average="macro", zero_division=0,
            )
            for task_idx in range(labels_np.shape[1])
        ]
        accuracy = float(np.mean(task_accuracies))
        f1_macro = float(np.mean(task_f1))

    return loss_mean, accuracy, f1_macro


def train_model(
    cfg:        MindFlowTrainingConfig = DEFAULT_CFG,
    save_model: bool = True,
) -> Dict:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    # 1. Dados — dataset completo DAiSEE (5.482 treino / 1.720 val / 1.866 teste)
    # Seleção estratificada garante todas as classes de engajamento presentes
    # em todos os splits (classes raras são força-incluídas).
    logger.info("Carregando splits completos do DAiSEE...")
    data = load_dataset(
        cfg=cfg,
        splits=("train", "validation", "test"),
        max_videos_per_split=None,  # sem limite — usa todos os videos extraidos
    )
    X_train, y_train, _, _          = data["train"]
    X_val,   y_val,   _, _          = data["validation"]
    X_test,  y_test,  _, clip_ids_test = data["test"]

    # 2. Pré-processamento (usa engagement=y[:,0] como alvo do SMOTE)
    logger.info("Pré-processando...")
    (X_train, y_train, X_val, y_val, X_test, y_test), artifacts = preprocess_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test,
        cfg=cfg,
        save_artifacts=save_model,
    )

    # 3. DataLoaders
    ds_train = MindFlowDataset(X_train, y_train)
    ds_val   = MindFlowDataset(X_val,   y_val)

    loader_train = DataLoader(ds_train, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    loader_val   = DataLoader(ds_val,   batch_size=cfg.train.batch_size,
                              shuffle=False, num_workers=0)

    # 4. Modelo
    input_dim = X_train.shape[2]
    model = build_model(cfg, input_dim=input_dim).to(device)

    # 5. FocalLoss por dimensão com gamma específico por tarefa
    MAX_WEIGHT = 8.0
    TASK_GAMMAS = {
        "engagement":  1.0,   # classes difíceis entre si
        "boredom":     2.0,   # classe 0 muito fácil (95% dos dados)
        "confusion":   2.0,
        "frustration": 2.0,
    }
    criterions = []
    target_task = getattr(cfg.train, "target_task", None)
    ordinal     = getattr(cfg.train, "ordinal", False)

    if ordinal:
        criterions.append(CoralLoss(n_levels=model.n_levels))
        logger.info("Modo SINGLE-TASK ORDINAL (CORAL) [%s]", target_task)
    elif target_task is not None:
        t = TASK_NAMES.index(target_task)
        w     = ds_train.compute_class_weights(task_idx=t).to(device)
        w     = torch.clamp(w, max=MAX_WEIGHT)
        gamma = TASK_GAMMAS.get(target_task, 2.0)
        criterions.append(FocalLoss(weight=w, gamma=gamma))
        logger.info("Modo SINGLE-TASK [%s] | Pesos gamma=%.1f (max=%.0f): %s",
                    target_task, gamma, MAX_WEIGHT, w.cpu().tolist())
    else:
        for t, task_name in enumerate(TASK_NAMES):
            w     = ds_train.compute_class_weights(task_idx=t).to(device)
            w     = torch.clamp(w, max=MAX_WEIGHT)
            gamma = TASK_GAMMAS.get(task_name, 2.0)
            criterions.append(FocalLoss(weight=w, gamma=gamma))
            logger.info("Pesos [%s] gamma=%.1f (max=%.0f): %s",
                        task_name, gamma, MAX_WEIGHT, w.cpu().tolist())

    # 6. Optimizer e scheduler
    optimizer = Adam(model.parameters(),
                     lr=cfg.train.learning_rate,
                     weight_decay=cfg.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  patience=cfg.train.lr_patience,
                                  factor=cfg.train.lr_factor)

    # 7. Loop
    history: Dict = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss":   [], "val_acc":   [], "val_f1":   [],
        "lr": [],
    }

    best_val_f1        = -1.0
    epochs_no_improve  = 0
    best_epoch         = 0

    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO — %d épocas máx. | device=%s", cfg.train.max_epochs, device)
    if target_task is not None:
        logger.info("Modo: SINGLE-TASK (Tarefa Alvo: %s) | Input dim (pós-PCA): %d", target_task, input_dim)
    else:
        logger.info("Modo: MULTI-TASK | Input dim (pós-PCA): %d | Tarefas: %d × %d classes", input_dim, len(TASK_NAMES), 4)
    logger.info("=" * 60)

    for epoch in range(1, cfg.train.max_epochs + 1):
        t_loss, t_acc, t_f1 = _run_epoch(model, loader_train, criterions, optimizer, device, "train", target_task=target_task)
        v_loss, v_acc, v_f1 = _run_epoch(model, loader_val,   criterions, None,      device, "val",   target_task=target_task)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(v_f1)   # maximiza val_f1

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        history["lr"].append(current_lr)

        logger.info(
            "Época %3d/%d | train loss=%.4f acc=%.3f f1=%.3f | "
            "val loss=%.4f acc=%.3f f1=%.3f | lr=%.2e",
            epoch, cfg.train.max_epochs,
            t_loss, t_acc, t_f1, v_loss, v_acc, v_f1, current_lr,
        )

        if v_f1 > best_val_f1:
            best_val_f1   = v_f1
            best_epoch    = epoch
            epochs_no_improve = 0
            if save_model:
                torch.save(model.state_dict(), cfg.model_path)
                logger.info("  💾 Checkpoint salvo (val_f1=%.4f)", best_val_f1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.train.early_stop_patience:
            logger.info("⏹  Early stopping na época %d (melhor: época %d)", epoch, best_epoch)
            break

    logger.info("=" * 60)
    logger.info("Treinamento concluído. Melhor época: %d", best_epoch)

    # Mantém no JSON os mesmos critérios usados para salvar o checkpoint.
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1

    # A avaliação deve usar exatamente o checkpoint que venceu na validação,
    # e não os pesos da última época executada após possível overfitting.
    if save_model and cfg.model_path.exists():
        best_state = torch.load(cfg.model_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        model.eval()
        logger.info("Checkpoint da melhor época recarregado para avaliação.")

    if save_model and cfg.eval.save_history:
        history_path = cfg.history_path
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Histórico salvo em: %s", history_path)

    history["_test_data"]  = (X_test, y_test, clip_ids_test)
    history["_model"]      = model
    history["_artifacts"]  = artifacts
    history["_best_epoch"] = best_epoch
    history["_input_dim"]  = input_dim

    return history
