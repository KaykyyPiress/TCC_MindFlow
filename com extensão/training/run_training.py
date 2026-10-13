"""
training/run_training.py
========================
Script de entrada único para o pipeline completo de treinamento.

Uso:
    # Treino completo (requer features extraídas em outputs/features/)
    python training/run_training.py

    # Apenas avaliação (requer modelo já treinado em outputs/training/)
    python training/run_training.py --eval-only

Saídas geradas em outputs/training/:
    lstm_mindflow.pt          — modelo treinado (pesos)
    scaler.joblib             — StandardScaler serializado
    pca.joblib                — PCA serializado
    metrics.json              — métricas do conjunto de teste
    training_history.json     — curvas de treino/val por época
    classification_report.txt — relatório por classe
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Garante que o pacote training seja encontrado ao rodar direto do terminal
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.config import MindFlowTrainingConfig, DEFAULT_CFG, OUTPUT_DIR
from training.train import train_model
from training.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:

    if args.eval_only:
        logger.info("Modo: avaliação do modelo salvo em disco.")
        logger.warning(
            "Para --eval-only, você precisa fornecer X_test/y_test. "
            "Esta opção é útil apenas se você salvou os dados de teste separadamente."
        )
        return

    # ----------------------------------------------------------------
    # Modo padrão: treino completo
    # ----------------------------------------------------------------
    logger.info("Modo: treinamento completo.")
    cfg = DEFAULT_CFG

    history = train_model(cfg=cfg, save_model=True)

    model             = history["_model"]
    X_test, y_test, _ = history["_test_data"]
    input_dim         = history["_input_dim"]

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
        input_dim=input_dim,
        save=True,
    )

    logger.info("Pipeline concluído.")
    logger.info("  Modelo salvo  : %s", OUTPUT_DIR / "lstm_mindflow.pt")
    for task, m in metrics.items():
        logger.info("  [%s] F1-macro=%.4f  Acc=%.4f",
                    m["display_name"], m["f1_macro"], m["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MindFlow AI — Pipeline de Treinamento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Apenas avaliar modelo já treinado (não retreina)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
