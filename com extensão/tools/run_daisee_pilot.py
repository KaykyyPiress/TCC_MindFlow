"""Executa um treino-piloto pequeno do MindFlow no DAiSEE oficial.

O piloto usa os splits oficiais, processa todos os frames dos vídeos de 30 FPS,
preserva cinco segundos de contexto por janela e escolhe clipes que cubram as
quatro classes de cada uma das quatro tarefas. Os artefatos ficam separados do
modelo usado atualmente pela extensão.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindflow_extractor.pipeline import MindFlowPipeline
from mindflow_extractor.runners.video_runner import process_video
from training import dataset as dataset_module
from training.config import MindFlowTrainingConfig
from training.dataset import TASK_NAMES
from training.evaluate import evaluate_model
from training.train import train_model


LOGGER = logging.getLogger("mindflow.pilot")
SPLIT_SETTINGS = {
    "train": ("Train", "TrainLabels.csv"),
    "validation": ("Validation", "ValidationLabels.csv"),
    "test": ("Test", "TestLabels.csv"),
}


def read_candidates(daisee_root: Path, split_name: str, labels_file: str) -> List[Dict]:
    """Cruza os vídeos realmente presentes com as linhas do CSV oficial."""
    labels_path = daisee_root / "Labels" / labels_file
    videos_root = daisee_root / "DataSet" / split_name
    if not labels_path.exists():
        raise FileNotFoundError(f"CSV de rótulos não encontrado: {labels_path}")
    if not videos_root.exists():
        raise FileNotFoundError(f"Split do DAiSEE não encontrado: {videos_root}")

    video_by_clip = {
        path.stem: path
        for pattern in ("*.avi", "*.mp4")
        for path in videos_root.rglob(pattern)
    }

    candidates: List[Dict] = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = {str(key).strip(): value for key, value in raw_row.items()}
            clip_id = Path(row.get("ClipID", row.get("VideoID", ""))).stem
            video_path = video_by_clip.get(clip_id)
            if video_path is None:
                continue
            labels = {
                "engagement": int(row["Engagement"]),
                "boredom": int(row["Boredom"]),
                "confusion": int(row["Confusion"]),
                "frustration": int(row["Frustration"]),
            }
            candidates.append(
                {
                    "clip_id": clip_id,
                    "video_path": video_path,
                    "subject_id": video_path.parent.parent.name,
                    "labels": labels,
                }
            )
    return candidates


def _coverage_counts(rows: Iterable[Dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for task in TASK_NAMES:
            counts[(task, row["labels"][task])] += 1
    return counts


def select_balanced(candidates: List[Dict], limit: int, quota: int, seed: int) -> List[Dict]:
    """Seleciona primeiro classes raras e depois completa com diversidade."""
    if limit > len(candidates):
        raise ValueError(f"Pedido de {limit} vídeos, mas só há {len(candidates)} disponíveis.")

    availability = _coverage_counts(candidates)
    missing = [(task, level) for task in TASK_NAMES for level in range(4)
               if availability[(task, level)] == 0]
    if missing:
        raise RuntimeError(f"O split não contém todas as combinações tarefa/classe: {missing}")

    needed = {(task, level): quota for task in TASK_NAMES for level in range(4)}
    chosen: List[Dict] = []
    chosen_ids = set()
    chosen_subjects = set()

    while any(value > 0 for value in needed.values()):
        remaining_targets = [target for target, value in needed.items() if value > 0]
        # Classes com menos exemplos disponíveis têm prioridade.
        target = min(remaining_targets, key=lambda item: (availability[item], item))
        eligible = [row for row in candidates
                    if row["clip_id"] not in chosen_ids
                    and row["labels"][target[0]] == target[1]]
        if not eligible:
            raise RuntimeError(f"Não foi possível cumprir a cobertura mínima de {target}.")

        def score(row: Dict) -> Tuple[float, int, str]:
            coverage = 0.0
            for task in TASK_NAMES:
                combo = (task, row["labels"][task])
                if needed[combo] > 0:
                    coverage += 1.0 + (1.0 / availability[combo])
            new_subject = int(row["subject_id"] not in chosen_subjects)
            # O clip_id invertido não é necessário: max usa cobertura e sujeito;
            # a ordenação anterior torna o desempate determinístico.
            return coverage, new_subject, row["clip_id"]

        selected = max(sorted(eligible, key=lambda row: row["clip_id"]), key=score)
        chosen.append(selected)
        chosen_ids.add(selected["clip_id"])
        chosen_subjects.add(selected["subject_id"])
        for task in TASK_NAMES:
            combo = (task, selected["labels"][task])
            needed[combo] = max(0, needed[combo] - 1)

    if len(chosen) > limit:
        raise RuntimeError(
            f"A cobertura solicitada precisa de {len(chosen)} vídeos, acima do limite {limit}."
        )

    rng = random.Random(seed)
    remaining = [row for row in candidates if row["clip_id"] not in chosen_ids]
    rng.shuffle(remaining)
    # Ao completar, privilegia pessoas ainda não escolhidas.
    remaining.sort(key=lambda row: row["subject_id"] in chosen_subjects)
    chosen.extend(remaining[: limit - len(chosen)])
    return sorted(chosen, key=lambda row: row["clip_id"])


def serializable_row(row: Dict) -> Dict:
    return {
        "clip_id": row["clip_id"],
        "video_path": str(row["video_path"]),
        "subject_id": row["subject_id"],
        "labels": row["labels"],
    }


def validate_coverage(rows: List[Dict], split: str) -> Dict[str, Dict[str, int]]:
    distribution: Dict[str, Dict[str, int]] = {}
    for task in TASK_NAMES:
        counts = Counter(row["labels"][task] for row in rows)
        distribution[task] = {str(level): counts[level] for level in range(4)}
        absent = [level for level in range(4) if counts[level] == 0]
        if absent:
            raise RuntimeError(f"O subconjunto {split} ficou sem {task} nas classes {absent}.")
    return distribution


def _summarize_extraction(clip_id: str, metadata: Dict) -> Dict:
    if not metadata:
        raise RuntimeError(f"Nenhum frame foi extraído de {clip_id}.")
    return {
        "clip_id": clip_id,
        "source_fps": metadata.get("source_fps"),
        "processed_frames": metadata.get("processed_frames"),
        "face_detected_ratio": metadata.get("quality", {}).get("face_detected_ratio"),
        "pose_detected_ratio": metadata.get("quality", {}).get("pose_detected_ratio"),
        "processing_time_s": metadata.get("processing_time_s"),
    }


def _extract_video_worker(row: Dict, output_dir: str) -> Dict:
    """Processa um vídeo dentro de um processo independente."""
    metadata = process_video(
        video_path=row["video_path"],
        output_dir=output_dir,
        label=row["labels"]["engagement"],
        labels_4d=row["labels"],
        subject_id=row["subject_id"],
        skip_frames=0,
        show_progress=False,
    )
    return _summarize_extraction(row["clip_id"], metadata)


def extract_split(rows: List[Dict], output_dir: Path, workers: int = 1) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_clip: Dict[str, Dict] = {}
    pending: List[Tuple[int, Dict]] = []

    # Primeiro identifica o que já está em cache. Isso também evita iniciar o
    # MediaPipe quando todos os vídeos do split já foram extraídos.
    for index, row in enumerate(rows, start=1):
        clip_id = row["clip_id"]
        npy_path = output_dir / f"{clip_id}.npy"
        meta_path = output_dir / f"{clip_id}_metadata.json"
        if npy_path.exists() and meta_path.exists():
            LOGGER.info("[%d/%d] %s já extraído; reutilizando.", index, len(rows), clip_id)
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            results_by_clip[clip_id] = _summarize_extraction(clip_id, metadata)
        else:
            pending.append((index, row))

    if workers == 1 and pending:
        # Modo sequencial: uma única instância do MediaPipe é compartilhada.
        with MindFlowPipeline() as pipeline:
            for index, row in pending:
                clip_id = row["clip_id"]
                LOGGER.info("[%d/%d] Extraindo %s a 30 FPS...", index, len(rows), clip_id)
                metadata = process_video(
                    video_path=row["video_path"],
                    output_dir=output_dir,
                    label=row["labels"]["engagement"],
                    labels_4d=row["labels"],
                    subject_id=row["subject_id"],
                    skip_frames=0,
                    show_progress=True,
                    pipeline_instance=pipeline,
                )
                results_by_clip[clip_id] = _summarize_extraction(clip_id, metadata)

    elif workers > 1 and pending:
        LOGGER.info(
            "Extraindo %d vídeos com %d processos paralelos...",
            len(pending), workers,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _extract_video_worker,
                    serializable_row(row),
                    str(output_dir),
                ): (index, row)
                for index, row in pending
            }
            completed = 0
            for future in as_completed(futures):
                index, row = futures[future]
                clip_id = row["clip_id"]
                try:
                    results_by_clip[clip_id] = future.result()
                except Exception as exc:
                    for other in futures:
                        other.cancel()
                    raise RuntimeError(f"Falha ao extrair {clip_id}: {exc}") from exc
                completed += 1
                LOGGER.info(
                    "[%d/%d novos | posição %d/%d] %s concluído.",
                    completed, len(pending), index, len(rows), clip_id,
                )

    return [results_by_clip[row["clip_id"]] for row in rows]


def build_config(run_id: str) -> MindFlowTrainingConfig:
    base = MindFlowTrainingConfig()
    return dataclasses.replace(
        base,
        run_id=run_id,
        features=dataclasses.replace(
            base.features,
            sequence_len=150,          # 5 segundos a 30 FPS
            perclos_window_frames=30, # 1 segundo a 30 FPS
        ),
        preprocess=dataclasses.replace(
            base.preprocess,
            sequence_stride=75,       # 50% de sobreposição
        ),
        train=dataclasses.replace(
            base.train,
            max_epochs=20,
            batch_size=16,
            early_stop_patience=5,
            lr_patience=2,
            device="cpu",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino-piloto MindFlow/DAiSEE a 30 FPS")
    parser.add_argument("--daisee-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Vídeos extraídos simultaneamente (recomendado: 2).",
    )
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers deve ser pelo menos 1.")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    daisee_root = args.daisee_root.resolve()
    pilot_root = ROOT / "outputs" / "pilot_30fps"
    feature_root = pilot_root / "features"
    pilot_root.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    quotas = {"train": 2, "validation": 1, "test": 1}
    selected_by_split: Dict[str, List[Dict]] = {}
    distributions: Dict[str, Dict] = {}

    for offset, (split_key, (split_name, labels_file)) in enumerate(SPLIT_SETTINGS.items()):
        candidates = read_candidates(daisee_root, split_name, labels_file)
        selected = select_balanced(
            candidates,
            limit=limits[split_key],
            quota=quotas[split_key],
            seed=args.seed + offset,
        )
        selected_by_split[split_key] = selected
        distributions[split_key] = validate_coverage(selected, split_key)
        LOGGER.info(
            "%s: %d de %d vídeos selecionados | %d pessoas",
            split_key,
            len(selected),
            len(candidates),
            len({row["subject_id"] for row in selected}),
        )

    manifest = {
        "purpose": "pipeline_smoke_test_not_production_validation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "capture": {
            "source_target_fps": 30,
            "skip_frames": 0,
            "sequence_frames": 150,
            "sequence_seconds": 5,
            "stride_frames": 75,
            "perclos_window_frames": 30,
        },
        "counts": limits,
        "extraction_workers": args.workers,
        "class_distribution_by_clip": distributions,
        "splits": {
            split: [serializable_row(row) for row in rows]
            for split, rows in selected_by_split.items()
        },
    }
    manifest_path = pilot_root / "pilot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Manifesto salvo em %s", manifest_path)

    extraction_summary = {}
    for split, rows in selected_by_split.items():
        LOGGER.info("=== Extração do split %s ===", split)
        extraction_summary[split] = extract_split(
            rows, feature_root / split, workers=args.workers,
        )

    summary_path = pilot_root / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(extraction_summary, handle, indent=2, ensure_ascii=False)

    if args.extract_only:
        LOGGER.info("Extração concluída; treino omitido por --extract-only.")
        return 0

    # Redireciona o carregador somente para as features deste piloto.
    dataset_module.SPLIT_DIRS = {
        "train": feature_root / "train",
        "validation": feature_root / "validation",
        "test": feature_root / "test",
    }

    run_id = f"pilot_30fps_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    cfg = build_config(run_id)
    with (cfg.output_dir / "pilot_config.json").open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(cfg), handle, indent=2, ensure_ascii=False)
    with (cfg.output_dir / "pilot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    LOGGER.info("=== Treinamento-piloto ===")
    history = train_model(cfg=cfg, save_model=True)
    model = history["_model"]
    X_test, y_test, clip_ids_test = history["_test_data"]
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
        input_dim=history["_input_dim"],
        save=True,
        clip_ids=clip_ids_test,
    )

    result = {
        "status": "completed",
        "run_id": run_id,
        "run_dir": str(cfg.output_dir),
        "best_epoch": history["_best_epoch"],
        "metrics": metrics,
        "warning": "Pilot with few videos; do not use as final accuracy estimate.",
    }
    result_path = pilot_root / "latest_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"\nPILOT_RESULT={result_path}")
    print(f"PILOT_RUN_DIR={cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
