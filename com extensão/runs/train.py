"""
runs/train.py
=============
Script de treino do MindFlow AI.

Uso:
    python runs/train.py
    python runs/train.py --target-task engagement
    python runs/train.py --target-task engagement --ordinal
"""
import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from training.config import DEFAULT_CFG
from training.dataset import TASK_NAMES
from training.train import train_model
from training.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindFlow AI — Treinamento")
    parser.add_argument(
        "--target-task", choices=TASK_NAMES, default=None,
        help="Treina só essa dimensão isolada (padrão: multi-output com as 4 dimensões).",
    )
    parser.add_argument(
        "--ordinal", action="store_true",
        help="Usa cabeça ordinal CORAL em vez de Linear+softmax nominal. Requer --target-task.",
    )
    args = parser.parse_args()

    if args.ordinal and args.target_task is None:
        parser.error("--ordinal requer --target-task (CORAL só é suportado em modo single-task).")

    cfg = DEFAULT_CFG
    if args.target_task is not None:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(
                cfg.train, target_task=args.target_task, ordinal=args.ordinal,
            )
        )

    print("=" * 60)
    print("  MindFlow AI — Treinamento")
    if cfg.train.ordinal:
        print(f"  Modo: SINGLE-TASK ORDINAL / CORAL ({cfg.train.target_task})")
    elif cfg.train.target_task is not None:
        print(f"  Modo: SINGLE-TASK ({cfg.train.target_task})")
    print("=" * 60)

    history = train_model(cfg=cfg, save_model=True)

    model                         = history["_model"]
    X_test, y_test, clip_ids_test = history["_test_data"]
    input_dim                     = history["_input_dim"]

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
        input_dim=input_dim,
        save=True,
        clip_ids=clip_ids_test,
    )

    print(f"\n✅ Treino concluído. Artefatos salvos em: {cfg.output_dir}")
    print("   Rode agora: python runs/plot.py   para gerar os gráficos.")
