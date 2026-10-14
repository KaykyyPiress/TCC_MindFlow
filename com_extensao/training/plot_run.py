"""
training/plot_run.py
====================
Gera todos os gráficos de um run específico de treinamento MindFlow.

Uso:
    python training/plot_run.py
    python training/plot_run.py --run outputs/training/runs/2026-06-09_22-15-59

Gráficos salvos em: <run_dir>/plots/
  1. training_curves.png  — Loss, Accuracy e F1 por época
  2. confusion_matrix.png — Matriz de confusão por target (engagement, boredom, etc.)
  3. per_class_metrics.png — Precision / Recall / F1 por nível e target
  4. summary.png          — Painel geral de métricas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paleta e estilo
# ---------------------------------------------------------------------------
COLORS = {
    "primary":   "#4F86F7",
    "secondary": "#F76F4F",
    "success":   "#4FBF7F",
    "warning":   "#F7C84F",
    "muted":     "#A0A0A0",
    "bg":        "#12121F",
    "card":      "#1A1A2E",
    "border":    "#2A2A4A",
    "text":      "#E0E0E0",
}

TARGET_COLORS = {
    "engagement":  "#4F86F7",
    "boredom":     "#F76F4F",
    "confusion":   "#F7C84F",
    "frustration": "#4FBF7F",
}

CLASS_LEVELS = ["very_low", "low", "high", "very_high"]
CLASS_LABELS = ["Very Low", "Low", "High", "Very High"]
CLASS_COLORS = ["#E74C3C", "#E67E22", "#2ECC71", "#3498DB"]


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["card"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "text.color":        COLORS["text"],
        "grid.color":        COLORS["border"],
        "grid.alpha":        0.6,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "legend.facecolor":  COLORS["card"],
        "legend.edgecolor":  COLORS["border"],
    })


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Curvas de Treinamento
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, plot_dir: Path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Curvas de Treinamento — MindFlow AI", fontsize=16, fontweight="bold", y=1.02)

    pairs = [
        ("train_loss",  "val_loss",  "Loss (CrossEntropy)", "Loss"),
        ("train_acc",   "val_acc",   "Accuracy",            "Accuracy"),
        ("train_f1",    "val_f1",    "F1-Macro",            "F1-Macro"),
    ]

    # O treino seleciona checkpoint por F1 de validação, não pela menor loss.
    best_epoch = int(history.get("best_epoch", np.argmax(history["val_f1"]) + 1))

    for ax, (tr_key, val_key, title, ylabel) in zip(axes, pairs):
        tr_data  = history[tr_key]
        val_data = history[val_key]

        ax.plot(epochs, tr_data,  color=COLORS["primary"],   linewidth=2.5,
                label="Treino", marker="o", markersize=4, alpha=0.9)
        ax.plot(epochs, val_data, color=COLORS["secondary"],  linewidth=2.5,
                label="Validação", marker="s", markersize=4, linestyle="--", alpha=0.9)

        # Sombra suave
        ax.fill_between(epochs, tr_data,  alpha=0.08, color=COLORS["primary"])
        ax.fill_between(epochs, val_data, alpha=0.08, color=COLORS["secondary"])

        # Linha da melhor época (menor val_loss)
        ax.axvline(x=best_epoch, color=COLORS["success"], linestyle=":", linewidth=2, alpha=0.8)
        ymin, ymax = ax.get_ylim()
        ax.text(best_epoch + 0.3, ymin + (ymax - ymin) * 0.05,
                f"Melhor\n(ép. {best_epoch})", color=COLORS["success"], fontsize=8, alpha=0.9)

        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = plot_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# 2. Matrizes de Confusão (um painel 2×2)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(metrics: dict, plot_dir: Path):
    targets = list(metrics.keys())
    n = len(targets)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14 * cols // 2, 6 * rows))
    axes = np.array(axes).flatten()

    fig.suptitle("Matrizes de Confusão — MindFlow AI", fontsize=16, fontweight="bold", y=1.01)

    for idx, target in enumerate(targets):
        ax = axes[idx]
        tm = metrics[target]
        cm = np.array(tm["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

        color = TARGET_COLORS.get(target, COLORS["primary"])
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("custom", [COLORS["card"], color])

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{tm['display_name']} — Acc: {tm['accuracy']:.1%}", pad=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_LABELS, fontsize=8)
        ax.set_xlabel("Predito", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)

        for i in range(4):
            for j in range(4):
                val = cm_norm[i, j]
                raw = cm[i, j]
                txt_color = "white" if val < 0.5 else COLORS["bg"]
                ax.text(j, i, f"{val:.2f}\n({raw})", ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Esconde eixos extras
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = plot_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# 3. Métricas por Classe e Target
# ---------------------------------------------------------------------------
def plot_per_class_metrics(metrics: dict, plot_dir: Path):
    targets = list(metrics.keys())
    n_targets = len(targets)

    fig, axes = plt.subplots(n_targets, 1, figsize=(13, 5 * n_targets))
    if n_targets == 1:
        axes = [axes]

    fig.suptitle("Precision / Recall / F1 por Nível e Target — MindFlow AI",
                 fontsize=15, fontweight="bold", y=1.01)

    x = np.arange(4)
    w = 0.25

    for ax, target in zip(axes, targets):
        tm     = metrics[target]
        levels = tm["per_level"]
        color  = TARGET_COLORS.get(target, COLORS["primary"])

        prec = [levels.get(lvl, {}).get("precision", 0) for lvl in CLASS_LEVELS]
        rec  = [levels.get(lvl, {}).get("recall",    0) for lvl in CLASS_LEVELS]
        f1   = [levels.get(lvl, {}).get("f1",        0) for lvl in CLASS_LEVELS]
        sup  = [levels.get(lvl, {}).get("support",   0) for lvl in CLASS_LEVELS]

        bars_p = ax.bar(x - w, prec, w, label="Precision", color=COLORS["primary"],   alpha=0.85)
        bars_r = ax.bar(x,     rec,  w, label="Recall",    color=COLORS["secondary"], alpha=0.85)
        bars_f = ax.bar(x + w, f1,   w, label="F1-Score",  color=COLORS["success"],   alpha=0.85)

        for bars in [bars_p, bars_r, bars_f]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.03:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        # Suporte como texto sob o label
        for i, (lvl, s) in enumerate(zip(CLASS_LABELS, sup)):
            ax.text(x[i], -0.08, f"n={s}", ha="center", fontsize=7.5,
                    color=COLORS["muted"], transform=ax.get_xaxis_transform())

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS, fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score")
        acc_str = f"Acc: {tm['accuracy']:.1%}  |  F1-Macro: {tm['f1_macro']:.4f}  |  F1-Weighted: {tm['f1_weighted']:.4f}"
        ax.set_title(f"{tm['display_name']} — {acc_str}", color=color)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = plot_dir / "per_class_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# 4. Painel Resumo
# ---------------------------------------------------------------------------
def plot_summary(metrics: dict, history: dict, plot_dir: Path):
    targets = list(metrics.keys())

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Painel de Resultados — MindFlow AI  (Run: {plot_dir.parent.name})",
                 fontsize=16, fontweight="bold", y=0.98)

    # Grid: linha de cima = gauges de accuracy; linha de baixo = métricas detalhadas
    outer = gridspec.GridSpec(2, 1, hspace=0.55, figure=fig)
    top   = gridspec.GridSpecFromSubplotSpec(1, len(targets), subplot_spec=outer[0], wspace=0.4)
    bot   = gridspec.GridSpecFromSubplotSpec(1, 2,            subplot_spec=outer[1], wspace=0.5)

    # --- Gauges de Accuracy por target ---
    for idx, target in enumerate(targets):
        ax = fig.add_subplot(top[idx])
        tm  = metrics[target]
        acc = tm["accuracy"]
        color = TARGET_COLORS.get(target, COLORS["primary"])

        theta_bg   = np.linspace(np.pi, 0, 200)
        theta_fill = np.linspace(np.pi, np.pi - acc * np.pi, 200)

        ax.plot(np.cos(theta_bg),   np.sin(theta_bg),   color=COLORS["border"], linewidth=12, solid_capstyle="round")
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color,            linewidth=12, solid_capstyle="round")

        ax.text(0, -0.05, f"{acc:.1%}", ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)
        ax.text(0, -0.45, "Accuracy", ha="center", fontsize=9, color=COLORS["muted"])
        ax.text(0, -0.7,  tm["display_name"], ha="center", fontsize=11,
                fontweight="bold", color=COLORS["text"])

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.85, 1.15)
        ax.axis("off")

    # --- Curva val_loss (bottom left) ---
    ax_loss = fig.add_subplot(bot[0])
    epochs  = list(range(1, len(history["train_loss"]) + 1))
    ax_loss.plot(epochs, history["train_loss"], color=COLORS["primary"],   linewidth=2.5,
                 label="Train Loss", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"],   color=COLORS["secondary"], linewidth=2.5,
                 label="Val Loss",   marker="s", markersize=3, linestyle="--")
    ax_loss.fill_between(epochs, history["train_loss"], alpha=0.1, color=COLORS["primary"])
    ax_loss.fill_between(epochs, history["val_loss"],   alpha=0.1, color=COLORS["secondary"])

    best_ep = int(history.get("best_epoch", np.argmax(history["val_f1"]) + 1))
    ax_loss.axvline(x=best_ep, color=COLORS["success"], linestyle=":", linewidth=2, alpha=0.8,
                    label=f"Melhor época ({best_ep})")
    ax_loss.set_title("Loss por Época")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.4)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    # --- Tabela de métricas globais (bottom right) ---
    ax_tbl = fig.add_subplot(bot[1])
    ax_tbl.axis("off")
    ax_tbl.set_title("Métricas Globais por Target", pad=14)

    col_labels = ["Target", "Accuracy", "F1-Macro", "F1-Weighted"]
    table_data = []
    row_colors = []

    for target in targets:
        tm = metrics[target]
        table_data.append([
            tm["display_name"],
            f"{tm['accuracy']:.1%}",
            f"{tm['f1_macro']:.4f}",
            f"{tm['f1_weighted']:.4f}",
        ])
        row_colors.append([TARGET_COLORS.get(target, "#888") + "33"] * 4)

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(COLORS["card"] if r > 0 else COLORS["border"])
        cell.set_edgecolor(COLORS["border"])
        cell.set_text_props(color=COLORS["text"], fontweight="bold" if r == 0 else "normal")
        if r > 0:
            cell.set_facecolor(row_colors[r - 1][c])

    plt.tight_layout()
    path = plot_dir / "summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✅  {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plota resultados de um run MindFlow.")
    parser.add_argument(
        "--run",
        default=str(ROOT / "outputs" / "training" / "runs" / "2026-06-09_22-15-59"),
        help="Caminho para a pasta do run (contém metrics.json e training_history.json)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.json"

    if not metrics_path.exists():
        print(f"❌ metrics.json não encontrado em: {run_dir}")
        sys.exit(1)
    if not history_path.exists():
        print(f"❌ training_history.json não encontrado em: {run_dir}")
        sys.exit(1)

    metrics = load_json(metrics_path)
    history = load_json(history_path)

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    set_dark_style()
    print(f"\n📊 Gerando gráficos em: {plot_dir}\n")

    plot_training_curves(history, plot_dir)
    plot_confusion_matrix(metrics, plot_dir)
    plot_per_class_metrics(metrics, plot_dir)
    plot_summary(metrics, history, plot_dir)

    print(f"\n✅ Todos os gráficos salvos em: {plot_dir}")


if __name__ == "__main__":
    main()
