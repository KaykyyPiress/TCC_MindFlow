"""Avalia exatamente o artefato usado pela extensão no split test do DAiSEE."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindflow_extractor.inference import (  # noqa: E402
    LEVEL_NAMES,
    MindFlowModelBundle,
    TASK_NAMES,
)


def iter_windows(
    features_dir: Path,
    sequence_length: int,
    stride: int,
    min_face_ratio: float,
) -> Iterable[Tuple[np.ndarray, np.ndarray, str]]:
    for feature_path in sorted(features_dir.glob("*.npy")):
        metadata_path = features_dir / f"{feature_path.stem}_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if float(metadata.get("quality", {}).get("face_detected_ratio", 1.0)) < min_face_ratio:
            continue
        labels = metadata.get("labels_4d")
        if not labels:
            continue
        target = np.asarray([int(labels[task]) for task in TASK_NAMES], dtype=np.int64)
        features = np.load(feature_path, allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != 120:
            continue
        for start in range(0, features.shape[0] - sequence_length + 1, stride):
            yield features[start:start + sequence_length], target, feature_path.stem


def predict_batches(
    bundle: MindFlowModelBundle,
    sequences: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    torch = bundle._torch
    for start in range(0, len(sequences), batch_size):
        prepared = sequences[start:start + batch_size].astype(np.float32)
        if bundle.model_feature_count == bundle.feature_count + 1:
            perclos = np.stack([bundle._perclos(sequence) for sequence in prepared])
            prepared = np.concatenate([prepared, perclos[..., None]], axis=2)
        clean = np.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
        flat = clean.reshape(-1, clean.shape[-1])
        scaled = bundle.scaler.transform(flat)
        reduced = bundle.pca.transform(scaled).reshape(
            len(prepared), bundle.sequence_length, -1
        )
        tensor = torch.from_numpy(reduced).float()
        with torch.no_grad():
            logits = bundle.model(tensor)
        predictions.append(np.stack([
            output.argmax(dim=-1).cpu().numpy() for output in logits
        ], axis=1))
    return np.concatenate(predictions, axis=0)


def aggregate_by_clip(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clip_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    indexes: Dict[str, list] = defaultdict(list)
    for index, clip_id in enumerate(clip_ids):
        indexes[str(clip_id)].append(index)
    true_by_clip = []
    pred_by_clip = []
    for clip_id in sorted(indexes):
        selected = indexes[clip_id]
        true_by_clip.append(y_true[selected[0]])
        pred_by_clip.append([
            int(np.bincount(y_pred[selected, task], minlength=4).argmax())
            for task in range(len(TASK_NAMES))
        ])
    return np.asarray(true_by_clip), np.asarray(pred_by_clip)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for task_index, task in enumerate(TASK_NAMES):
        truth = y_true[:, task_index]
        predicted = y_pred[:, task_index]
        majority_class, majority_count = Counter(truth.tolist()).most_common(1)[0]
        results[task] = {
            "samples": int(len(truth)),
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(truth, predicted, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(truth, predicted, average="macro", zero_division=0)),
            "majority_baseline_accuracy": float(majority_count / len(truth)),
            "majority_class": LEVEL_NAMES[majority_class],
            "support": np.bincount(truth, minlength=4).astype(int).tolist(),
            "confusion_matrix": confusion_matrix(
                truth, predicted, labels=list(range(4))
            ).astype(int).tolist(),
            "classification_report": classification_report(
                truth,
                predicted,
                labels=list(range(4)),
                target_names=LEVEL_NAMES,
                zero_division=0,
            ),
        }
    return results


def save_confusion_plot(metrics: Dict[str, dict], output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for axis, task in zip(axes.flat, TASK_NAMES):
        matrix = np.asarray(metrics[task]["confusion_matrix"], dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
        image = axis.imshow(normalized, cmap="YlOrRd", vmin=0, vmax=1)
        axis.set_title(task.capitalize())
        axis.set_xticks(range(4), LEVEL_NAMES, rotation=25, ha="right")
        axis.set_yticks(range(4), LEVEL_NAMES)
        axis.set_xlabel("Predito")
        axis.set_ylabel("Real")
        for row in range(4):
            for column in range(4):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}\n{normalized[row, column]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[row, column] > 0.55 else "black",
                )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia o modelo implantado no split de teste extraído do DAiSEE."
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models" / "apresentacao_TCC"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "daisee_labeled_test"
    )
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--min-face-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    bundle = MindFlowModelBundle(args.model_dir)
    records = list(iter_windows(
        args.features_dir,
        sequence_length=bundle.sequence_length,
        stride=args.stride,
        min_face_ratio=args.min_face_ratio,
    ))
    if not records:
        raise RuntimeError("Nenhuma janela válida encontrada.")

    sequences = np.stack([record[0] for record in records]).astype(np.float32)
    truth = np.stack([record[1] for record in records])
    clip_ids = np.asarray([record[2] for record in records], dtype=object)
    predicted = predict_batches(bundle, sequences, args.batch_size)
    clip_truth, clip_predicted = aggregate_by_clip(truth, predicted, clip_ids)

    window_metrics = calculate_metrics(truth, predicted)
    clip_metrics = calculate_metrics(clip_truth, clip_predicted)
    report = {
        "model_dir": str(args.model_dir.resolve()),
        "features_dir": str(args.features_dir.resolve()),
        "model_feature_count": bundle.model_feature_count,
        "sequence_frames": bundle.sequence_length,
        "stride_frames": args.stride,
        "effective_fps_from_metadata": 6.0,
        "windows": int(len(truth)),
        "clips": int(len(clip_truth)),
        "by_window": window_metrics,
        "by_clip": clip_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "classification_report.txt").open("w", encoding="utf-8") as handle:
        for level_name, metrics in (("JANELA", window_metrics), ("CLIPE", clip_metrics)):
            handle.write(f"\n{'=' * 72}\n{level_name}\n{'=' * 72}\n")
            for task in TASK_NAMES:
                handle.write(f"\n--- {task.upper()} ---\n")
                handle.write(metrics[task]["classification_report"])
    save_confusion_plot(
        window_metrics,
        args.output_dir / "confusion_matrices_by_window.png",
        "DAiSEE test — modelo implantado — avaliação por janela",
    )
    save_confusion_plot(
        clip_metrics,
        args.output_dir / "confusion_matrices_by_clip.png",
        "DAiSEE test — modelo implantado — avaliação por clipe",
    )

    print(f"Clipes: {len(clip_truth)} | janelas: {len(truth)}")
    for task in TASK_NAMES:
        metrics = clip_metrics[task]
        print(
            f"{task:12s} acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
    print(f"Relatório: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
