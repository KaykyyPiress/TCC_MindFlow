"""
training/plot_results.py
========================
Gera todos os gráficos dos resultados de treinamento do MindFlow AI.

Gráficos gerados em outputs/training/plots/:
  1. confusion_matrix.png      — Matriz de confusão normalizada
  2. training_curves.png       — Loss e F1 por época (treino vs val)
  3. per_class_metrics.png     — Precision / Recall / F1 por classe
  4. class_distribution.png    — Distribuição das classes no test set
  5. accuracy_summary.png      — Painel resumo das métricas finais

Uso:
    python training/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sem display (funciona em servidor/headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs" / "training"
PLOT_DIR   = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de cores consistente
COLORS = {
    "primary":   "#4F86F7",
    "secondary": "#F76F4F",
    "success":   "#4FBF7F",
    "warning":   "#F7C84F",
    "muted":     "#A0A0A0",
    "bg":        "#1A1A2E",
    "card":      "#16213E",
    "text":      "#E0E0E0",
}

CLASS_NAMES_SHORT = ["very_low", "low", "high", "very_high"]
CLASS_NAMES_FULL  = [
    "Very Low\nEngagement",
    "Low\nEngagement",
    "High\nEngagement",
    "Very High\nEngagement",
]
CLASS_COLORS = ["#E74C3C", "#E67E22", "#2ECC71", "#3498DB"]


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["card"],
        "axes.edgecolor":    "#444466",
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "text.color":        COLORS["text"],
        "grid.color":        "#333355",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
    })


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Matriz de Confusão
# ---------------------------------------------------------------------------

def plot_confusion_matrix(metrics: dict):
    cm = np.array(metrics["confusion_matrix"])   # (4, 4)
    n_classes = cm.shape[0]

    # Normaliza por linha (recall por classe)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Matriz de Confusão — MindFlow AI", fontsize=15, fontweight="bold", y=1.02)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Contagem absoluta", "Normalizada (por classe real)"],
        [".0f", ".2f"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_title(title, pad=12)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(CLASS_NAMES_SHORT, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES_SHORT, fontsize=9)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

        thresh = data.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                val = data[i, j]
                color = "white" if val < thresh else "#1A1A2E"
                text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = PLOT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# 2. Curvas de Treinamento
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Curvas de Treinamento — MindFlow AI", fontsize=15, fontweight="bold")

    pairs = [
        ("train_loss", "val_loss",  "Loss (CrossEntropy)",  "Loss"),
        ("train_acc",  "val_acc",   "Accuracy",             "Accuracy"),
        ("train_f1",   "val_f1",    "F1-Macro",             "F1-Macro"),
    ]

    for ax, (tr_key, val_key, title, ylabel) in zip(axes, pairs):
        ax.plot(epochs, history[tr_key],  color=COLORS["primary"],
                linewidth=2, label="Treino", marker="o", markersize=3)
        ax.plot(epochs, history[val_key], color=COLORS["secondary"],
                linewidth=2, label="Validação", marker="s", markersize=3, linestyle="--")

        # Marca melhor val
        best_idx = int(np.argmin(history["val_loss"]))
        ax.axvline(x=best_idx + 1, color=COLORS["success"], linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(best_idx + 1.2, ax.get_ylim()[0] * 1.05 if ax.get_ylim()[0] > 0 else 0.02,
                f"Melhor\n(época {best_idx + 1})", color=COLORS["success"], fontsize=8)

        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# 3. Métricas por Classe
# ---------------------------------------------------------------------------

def plot_per_class_metrics(metrics: dict):
    per_class = metrics["per_class"]
    classes   = list(per_class.keys())
    n         = len(classes)

    prec = [per_class[c]["precision"] for c in classes]
    rec  = [per_class[c]["recall"]    for c in classes]
    f1   = [per_class[c]["f1"]        for c in classes]
    sup  = [per_class[c]["support"]   for c in classes]

    x = np.arange(n)
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle("Métricas por Classe — MindFlow AI", fontsize=15, fontweight="bold")

    # Barras de precision/recall/f1
    bars_p = ax1.bar(x - w, prec, w, label="Precision", color=COLORS["primary"],   alpha=0.85)
    bars_r = ax1.bar(x,      rec,  w, label="Recall",    color=COLORS["secondary"], alpha=0.85)
    bars_f = ax1.bar(x + w, f1,   w, label="F1-Score",   color=COLORS["success"],   alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1 por classe")
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    # Support (quantidade de amostras por classe no test)
    colors_bar = [CLASS_COLORS[i % 4] for i in range(n)]
    bars_s = ax2.bar(x, sup, color=colors_bar, alpha=0.85, width=0.5)
    for bar, s in zip(bars_s, sup):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(s), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
    ax2.set_ylabel("Amostras no Test")
    ax2.set_title("Distribuição de classes no conjunto de teste")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "per_class_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# 4. Painel Resumo
# ---------------------------------------------------------------------------

def plot_summary(metrics: dict, history: dict):
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Resumo de Resultados — MindFlow AI (subset parcial)",
                 fontsize=14, fontweight="bold", y=1.02)

    gs = fig.add_gridspec(1, 3, wspace=0.4)

    # --- Gauge de Accuracy ---
    ax1 = fig.add_subplot(gs[0])
    acc = metrics["accuracy"]
    baseline = 0.62
    theta = np.linspace(np.pi, 0, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color="#333355", linewidth=10, solid_capstyle="round")

    # Fill até accuracy
    theta_fill = np.linspace(np.pi, np.pi - acc * np.pi, 200)
    color_gauge = COLORS["success"] if acc >= baseline else COLORS["secondary"]
    ax1.plot(np.cos(theta_fill), np.sin(theta_fill),
             color=color_gauge, linewidth=10, solid_capstyle="round")

    ax1.text(0, -0.15, f"{acc*100:.1f}%", ha="center", va="center",
             fontsize=22, fontweight="bold", color=color_gauge)
    ax1.text(0, -0.45, "Accuracy", ha="center", fontsize=10)
    ax1.text(0, -0.65, f"Baseline: {baseline*100:.0f}%", ha="center",
             fontsize=9, color=COLORS["muted"])
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-0.8, 1.2)
    ax1.axis("off")
    ax1.set_title("Accuracy (Test)")

    # --- Métricas globais ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    metric_names  = ["F1-Macro", "F1-Weighted", "Precision-Macro", "Recall-Macro"]
    metric_values = [
        metrics["f1_macro"], metrics["f1_weighted"],
        metrics["precision_macro"], metrics["recall_macro"],
    ]
    colors_m = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]

    for i, (name, val, col) in enumerate(zip(metric_names, metric_values, colors_m)):
        y = 0.85 - i * 0.22
        ax2.add_patch(plt.Rectangle((0, y - 0.07), 1, 0.16,
                                     transform=ax2.transAxes,
                                     color=COLORS["card"], clip_on=False,
                                     linewidth=1, edgecolor=col))
        ax2.text(0.08, y + 0.01, name, transform=ax2.transAxes,
                 fontsize=9, color=COLORS["text"])
        ax2.text(0.92, y + 0.01, f"{val:.4f}", transform=ax2.transAxes,
                 fontsize=11, fontweight="bold", ha="right", color=col)

    ax2.set_title("Métricas Globais (Test)")

    # --- LR decay ---
    ax3 = fig.add_subplot(gs[2])
    epochs = list(range(1, len(history["lr"]) + 1))
    ax3.plot(epochs, history["lr"], color=COLORS["warning"], linewidth=2, marker="o", markersize=4)
    ax3.set_title("Learning Rate por Época")
    ax3.set_xlabel("Época")
    ax3.set_ylabel("LR")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "accuracy_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_dark_style()

    metrics_path = OUTPUT_DIR / "metrics.json"
    history_path = OUTPUT_DIR / "training_history.json"

    if not metrics_path.exists():
        print("❌ metrics.json não encontrado. Execute python training/run_training.py primeiro.")
        return
    if not history_path.exists():
        print("❌ training_history.json não encontrado.")
        return

    metrics = load_json(metrics_path)
    history = load_json(history_path)

    print(f"\n📊 Gerando gráficos em: {PLOT_DIR}\n")

    plot_confusion_matrix(metrics)
    plot_training_curves(history)
    plot_per_class_metrics(metrics)
    plot_summary(metrics, history)

    print(f"\n✅ Todos os gráficos salvos em: {PLOT_DIR}")
    print("   Abra a pasta para visualizá-los.")


if __name__ == "__main__":
    main()
