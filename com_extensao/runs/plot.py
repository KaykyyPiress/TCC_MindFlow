"""
runs/plot.py
============
Gera gráficos dos resultados de treinamento do MindFlow AI.

Lê outputs/training/metrics.json  (multi-output: 4 dimensões)
     outputs/training/training_history.json

Salva em outputs/training/plots/

Uso:
    python runs/plot.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_BASE_DIR = ROOT / "outputs" / "training" / "runs"

def get_run_dir() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.is_absolute():
            p = ROOT / sys.argv[1]
        return p
    runs = sorted(OUTPUT_BASE_DIR.glob("*"))
    if not runs:
        raise FileNotFoundError(f"Nenhum treino encontrado em {OUTPUT_BASE_DIR}")
    return runs[-1]

RUN_DIR = get_run_dir()
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_DISPLAY = ["Engajamento", "Tédio", "Confusão", "Frustração"]
TASK_KEYS    = ["engagement", "boredom", "confusion", "frustration"]
TASK_COLORS  = ["#00E676", "#FF9800", "#CE93D8", "#F44336"]
LEVEL_LABELS = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]

DARK = "#0D1117"
CARD = "#161B22"
BORDER = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "legend.facecolor": CARD,
    "legend.edgecolor": BORDER,
})


def load_metrics():
    path = RUN_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"metrics.json não encontrado em {RUN_DIR}. Rode runs/train.py primeiro.")
    with open(path) as f:
        return json.load(f)


def load_history():
    path = RUN_DIR / "training_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Plot 1: Curvas de treino ─────────────────────────────────────────────────

def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Curvas de Treinamento — MindFlow AI", fontsize=14, color=TEXT, weight="bold", y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)
    specs = [
        ("Loss Total (4 dimensões)", "train_loss", "val_loss", "#58A6FF"),
        ("Accuracy (Engajamento)",   "train_acc",  "val_acc",  "#00E676"),
        ("F1-macro (Engajamento)",   "train_f1",   "val_f1",   "#FF9800"),
    ]

    for ax, (title, tr_key, val_key, color) in zip(axes, specs):
        ax.plot(epochs, history[tr_key],  color=color,   lw=2, label="Treino",    alpha=0.9)
        ax.plot(epochs, history[val_key], color="#F44336", lw=2, label="Validação", ls="--", alpha=0.9)
        ax.set_title(title, fontsize=11, color=TEXT)
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {out.name}")


# ─── Plot 2: Métricas por dimensão ───────────────────────────────────────────

def plot_metrics_per_task(metrics: dict):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Métricas por Dimensão DAiSEE", fontsize=14, color=TEXT, weight="bold")

    metric_keys   = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1-macro", "F1-weighted", "Precision", "Recall"]
    x = np.arange(len(metric_keys))
    width = 0.6

    for ax, task_key, display, color in zip(axes, TASK_KEYS, TASK_DISPLAY, TASK_COLORS):
        if task_key not in metrics:
            ax.set_visible(False)
            continue
        m = metrics[task_key]
        vals = [m.get(k, 0) for k in metric_keys]
        bars = ax.bar(x, vals, width, color=color, alpha=0.85, zorder=3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)

        ax.set_title(display, fontsize=12, color=color, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.4, zorder=0)

    plt.tight_layout()
    out = PLOTS_DIR / "metrics_per_task.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {out.name}")


# ─── Plot 3: Matrizes de confusão (2×2 grid) ─────────────────────────────────

def plot_confusion_matrices(metrics: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Matrizes de Confusão — 4 Dimensões DAiSEE", fontsize=14, color=TEXT, weight="bold")

    for ax, task_key, display, color in zip(axes.flat, TASK_KEYS, TASK_DISPLAY, TASK_COLORS):
        if task_key not in metrics or "confusion_matrix" not in metrics[task_key]:
            ax.set_visible(False)
            continue

        cm = np.array(metrics[task_key]["confusion_matrix"])
        # Normaliza por linha
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)

        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

        # Anotações
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = cm_norm[i, j]
                txt_color = "black" if pct > 0.5 else TEXT
                ax.text(j, i, f"{val}\n({pct:.0%})",
                        ha="center", va="center", fontsize=8, color=txt_color)

        ax.set_title(display, fontsize=12, color=color, weight="bold")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["M.Baixo", "Baixo", "Alto", "M.Alto"], fontsize=8)
        ax.set_yticklabels(["M.Baixo", "Baixo", "Alto", "M.Alto"], fontsize=8)
        ax.set_xlabel("Predito", fontsize=9)
        ax.set_ylabel("Real",    fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {out.name}")


# ─── Plot 4: F1 por nível (barras agrupadas) ──────────────────────────────────

def plot_per_level_f1(metrics: dict):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("F1 por Nível — Cada Dimensão DAiSEE", fontsize=14, color=TEXT, weight="bold")

    level_colors = ["#58A6FF", "#3FB950", "#FF9800", "#F44336"]

    for ax, task_key, display, task_color in zip(axes, TASK_KEYS, TASK_DISPLAY, TASK_COLORS):
        if task_key not in metrics:
            ax.set_visible(False)
            continue

        per_level = metrics[task_key].get("per_level", {})
        level_order = ["very_low", "low", "high", "very_high"]
        labels_br   = ["Muito\nBaixo", "Baixo", "Alto", "Muito\nAlto"]

        f1_vals = [per_level.get(k, {}).get("f1", 0) for k in level_order]
        supports = [per_level.get(k, {}).get("support", 0) for k in level_order]

        bars = ax.bar(range(4), f1_vals, color=level_colors, alpha=0.85, zorder=3)
        for bar, f1, sup in zip(bars, f1_vals, supports):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"F1={f1:.2f}\nn={sup}", ha="center", va="bottom", fontsize=7.5, color=TEXT)

        ax.set_title(display, fontsize=12, color=task_color, weight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_br, fontsize=8)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("F1-score", fontsize=9)
        ax.grid(True, axis="y", alpha=0.4, zorder=0)

    plt.tight_layout()
    out = PLOTS_DIR / "f1_per_level.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {out.name}")


# ─── Plot 5: Painel Resumo ────────────────────────────────────────────────────

def plot_summary_panel(metrics: dict, history: dict | None):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("MindFlow AI — Painel Resumo", fontsize=16, color=TEXT, weight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # Radar / Barras de métricas por dimensão (linha superior)
    metric_keys  = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_short = ["Acc", "F1", "Prec", "Rec"]

    for col, (task_key, display, color) in enumerate(zip(TASK_KEYS, TASK_DISPLAY, TASK_COLORS)):
        ax = fig.add_subplot(gs[0, col])
        if task_key not in metrics:
            continue
        m = metrics[task_key]
        vals = [m.get(k, 0) for k in metric_keys]

        theta = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False)
        vals_plot = vals + [vals[0]]
        theta_plot = list(theta) + [theta[0]]

        ax2 = fig.add_subplot(gs[0, col], polar=True)
        ax2.set_facecolor(CARD)
        ax2.plot(theta_plot, vals_plot, color=color, lw=2)
        ax2.fill(theta_plot, vals_plot, color=color, alpha=0.25)
        ax2.set_xticks(theta)
        ax2.set_xticklabels(metric_short, fontsize=8, color=TEXT)
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color=MUTED)
        ax2.set_ylim(0, 1)
        ax2.set_title(display, fontsize=10, color=color, weight="bold", pad=10)
        ax2.tick_params(colors=MUTED)
        ax2.spines["polar"].set_color(BORDER)
        ax2.grid(color=BORDER, linewidth=0.5)

    # Curvas de treino (linha inferior, abrange todas as colunas)
    ax_curves = fig.add_subplot(gs[1, :])
    if history:
        epochs = range(1, len(history["train_loss"]) + 1)
        ax_curves.plot(epochs, history["val_loss"],  color="#58A6FF", lw=2, label="Val Loss")
        ax_curves.plot(epochs, history["train_loss"], color="#3FB950", lw=2, label="Train Loss", ls="--", alpha=0.7)
        ax_curves.set_title("Curva de Loss durante Treinamento", fontsize=11, color=TEXT)
        ax_curves.set_xlabel("Época")
        ax_curves.set_ylabel("Loss")
        ax_curves.grid(True, alpha=0.4)
        ax_curves.legend(fontsize=9)
    else:
        ax_curves.text(0.5, 0.5, "Histórico não disponível",
                       ha="center", va="center", color=MUTED, transform=ax_curves.transAxes)

    out = PLOTS_DIR / "summary_panel.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {out.name}")


def plot_accuracy_summary(metrics: dict):
    """4 gauges em arco (velocímetro) — um por dimensão DAiSEE."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("Acurácia Geral por Dimensão", fontsize=15, color=TEXT,
                 weight="bold", y=1.01)

    def _draw_gauge(ax, value: float, title: str, color: str, f1: float,
                    acc_w: float, prec: float, rec: float):
        """Desenha um gauge estilo velocímetro com brilho, inspirado na imagem do usuário."""
        ax.set_facecolor(DARK)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.8, 1.2)
        ax.axis("off")

        # Ângulos: da ponta esquerda (abaixo de 180) até ponta direita (abaixo de 0)
        start_angle = np.pi * 1.2  # ~216 graus
        end_angle = np.pi * -0.2   # ~-36 graus
        span = start_angle - end_angle

        # ── Arco de Fundo (Grosso e escuro) ──
        theta_bg = np.linspace(start_angle, end_angle, 200)
        ax.plot(np.cos(theta_bg), np.sin(theta_bg), 
                color="#21262D", linewidth=35, solid_capstyle="butt", zorder=1)

        # ── Arco Colorido (Valor) ──
        if value > 0:
            val_angle = start_angle - (value * span)
            theta_val = np.linspace(start_angle, val_angle, 200)
            ax.plot(np.cos(theta_val), np.sin(theta_val), 
                    color=color, linewidth=35, solid_capstyle="butt", zorder=2)
            
            # ── Agulha (Gradiente com brilho na ponta) ──
            num_segments = 50
            needle_len = 0.85
            r_vals = np.linspace(0, needle_len, num_segments)
            for i in range(num_segments - 1):
                alpha = (i / num_segments)**2
                ax.plot([r_vals[i]*np.cos(val_angle), r_vals[i+1]*np.cos(val_angle)],
                        [r_vals[i]*np.sin(val_angle), r_vals[i+1]*np.sin(val_angle)],
                        color="#ffffff", linewidth=6, alpha=alpha, zorder=10)

            # Ponta final da agulha (branca)
            ax.plot([r_vals[-2]*np.cos(val_angle), r_vals[-1]*np.cos(val_angle)],
                    [r_vals[-2]*np.sin(val_angle), r_vals[-1]*np.sin(val_angle)],
                    color="#ffffff", linewidth=6, solid_capstyle="round", zorder=11)

        # ── Ticks e Rótulos (0 a 100) ──
        for pct in [0, 10, 50, 100]:
            ang = start_angle - (pct / 100) * span
            # Coloca o texto no interior do arco
            tx = 0.65 * np.cos(ang)
            ty = 0.65 * np.sin(ang)
            ax.text(tx, ty, f"{pct}", ha="center", va="center", 
                    fontsize=10, color=TEXT, weight="bold", zorder=5)

        # ── Percentual Central ──
        ax.text(0, -0.3, f"{value*100:.1f}%", ha="center", va="center",
                fontsize=30, color=TEXT, weight="bold", zorder=11)

        # ── Título com borda colorida ──
        ax.text(0, -0.65, title, ha="center", va="center",
                fontsize=16, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                          edgecolor=color, linewidth=1.5))

    for ax, task_key, display, color in zip(axes, TASK_KEYS, TASK_DISPLAY, TASK_COLORS):
        if task_key not in metrics:
            ax.text(0.5, 0.5, "N/D", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            continue
        m = metrics[task_key]
        _draw_gauge(
            ax,
            value  = m.get("accuracy",         0),
            title  = display,
            color  = color,
            f1     = m.get("f1_macro",          0),
            acc_w  = m.get("f1_weighted",        0),
            prec   = m.get("precision_macro",    0),
            rec    = m.get("recall_macro",       0),
        )

    plt.tight_layout()
    out = PLOTS_DIR / "accuracy_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MindFlow AI — Geração de Gráficos")
    print("=" * 55)

    try:
        metrics = load_metrics()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    history = load_history()
    if history is None:
        print("  ⚠️  training_history.json não encontrado — curvas de treino serão omitidas.")

    print(f"\n  Gerando gráficos para a execução: {RUN_DIR.name}")
    print(f"  Salvando gráficos em: {PLOTS_DIR}\n")

    if history:
        plot_training_curves(history)

    plot_accuracy_summary(metrics)
    plot_metrics_per_task(metrics)
    plot_confusion_matrices(metrics)
    plot_per_level_f1(metrics)
    plot_summary_panel(metrics, history)

    print(f"\n✅ Todos os gráficos salvos em: {PLOTS_DIR}")
