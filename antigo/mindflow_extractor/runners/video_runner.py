"""
Runner de vídeo — processamento batch (DAiSEE-ready).

Contrato de saída por vídeo:
  <output_dir>/<video_stem>.npy           — shape (N_frames, 120), float32
  <output_dir>/<video_stem>_metadata.json — metadados da sessão

O formato garante que uma sessão inteira vire um único par de arquivos,
conforme decisão metodológica do TCC: "Sessão inteira → 1 .npy + 1
metadata.json".

Uso via linha de comando:
    python -m mindflow_extractor.runners.video_runner \\
        --video path/to/video.mp4 \\
        --output path/to/output/dir \\
        [--label 0]  # label DAiSEE opcional (0-3)

Uso programático:
    from mindflow_extractor.runners import process_video
    process_video(video_path="clip.mp4", output_dir="outputs/", label=1)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np

from ..pipeline import MindFlowPipeline, FEATURE_NAMES
from ..config import VECTOR_DIM


def process_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    label: Optional[int] = None,
    subject_id: Optional[str] = None,
    skip_frames: int = 0,
    show_progress: bool = True,
) -> Dict:
    """
    Processa um vídeo completo e salva o par .npy + _metadata.json.

    Args:
        video_path: caminho para o vídeo (.mp4, .avi, etc.).
        output_dir: diretório onde o par de arquivos será gravado.
        label: rótulo DAiSEE (0=very_low, 1=low, 2=high, 3=very_high)
               ou None se não disponível.
        subject_id: ID do sujeito DAiSEE (ex.: "110001").
        skip_frames: quantos frames pular entre leituras (0 = nenhum).
                     Útil para debug rápido (skip_frames=3 processa 1/4).
        show_progress: imprime barra de progresso no terminal.

    Returns:
        dict com metadados da sessão (mesmo conteúdo do .json).

    Raises:
        FileNotFoundError: se o vídeo não existir.
        RuntimeError: se o vídeo não puder ser aberto pelo OpenCV.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não conseguiu abrir: {video_path}")

    # Metadados da fonte
    source_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = video_path.stem
    npy_path  = output_dir / f"{stem}.npy"
    meta_path = output_dir / f"{stem}_metadata.json"

    vectors: list = []
    frame_meta: list = []
    t_start = time.perf_counter()

    with MindFlowPipeline() as pipe:
        frame_idx = 0
        read_idx  = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            read_idx += 1
            # skip_frames: lê todos, processa apenas a cada N+1
            if skip_frames > 0 and (read_idx - 1) % (skip_frames + 1) != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = (read_idx - 1) / source_fps * 1000.0

            result = pipe.process(frame_rgb, frame_idx=frame_idx, timestamp_ms=timestamp_ms)

            vectors.append(result.vector)
            frame_meta.append({
                "frame_idx":     frame_idx,
                "source_frame":  read_idx - 1,
                "timestamp_ms":  round(timestamp_ms, 2),
                "face_detected": result.face_detected,
                "pose_detected": result.pose_detected,
            })

            frame_idx += 1

            if show_progress and frame_idx % 100 == 0:
                elapsed = time.perf_counter() - t_start
                pct = (read_idx / max(source_frames, 1)) * 100
                fps_proc = frame_idx / max(elapsed, 1e-6)
                print(f"\r  [{stem}] {pct:5.1f}% | "
                      f"frames={frame_idx} | {fps_proc:.1f} fps proc", end="", flush=True)

    cap.release()
    if show_progress:
        print()  # newline após o progresso

    if not vectors:
        print(f"[AVISO] Nenhum frame processado para {video_path.name}")
        return {}

    # Array final (N, 120)
    array = np.array(vectors, dtype=np.float32)

    # Estatísticas de qualidade da sessão
    nan_counts = np.isnan(array).sum(axis=0)  # NaN por feature
    face_ok_frames = sum(1 for m in frame_meta if m["face_detected"])
    pose_ok_frames = sum(1 for m in frame_meta if m["pose_detected"])
    total_frames = len(frame_meta)

    metadata = {
        "version":          "1.0",
        "video_file":       video_path.name,
        "subject_id":       subject_id,
        "label":            label,
        "label_meaning": {
            0: "very_low_engagement",
            1: "low_engagement",
            2: "high_engagement",
            3: "very_high_engagement",
        }.get(label, "unknown") if label is not None else None,
        "source_fps":       source_fps,
        "source_resolution":[source_w, source_h],
        "source_frames":    source_frames,
        "processed_frames": total_frames,
        "skip_frames":      skip_frames,
        "array_shape":      list(array.shape),
        "vector_dim":       VECTOR_DIM,
        "feature_names":    list(FEATURE_NAMES),
        "quality": {
            "face_detected_ratio": round(face_ok_frames / max(total_frames, 1), 4),
            "pose_detected_ratio": round(pose_ok_frames / max(total_frames, 1), 4),
            "nan_per_feature":     nan_counts.tolist(),
        },
        "processing_time_s": round(time.perf_counter() - t_start, 2),
        "frames":           frame_meta,
    }

    # Persistência
    np.save(str(npy_path), array)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if show_progress:
        print(f"  Salvo: {npy_path.name}  shape={array.shape}")
        print(f"         {meta_path.name}")
        print(f"  Face OK: {face_ok_frames}/{total_frames} frames "
              f"({100*face_ok_frames/max(total_frames,1):.1f}%)")

    return metadata


def process_daisee_batch(
    daisee_root: Union[str, Path],
    output_dir: Union[str, Path],
    split: str = "Train",
    max_videos: Optional[int] = None,
) -> None:
    """
    Processa em lote vídeos do DAiSEE com estrutura padrão:
    <daisee_root>/<split>/<subject_id>/<session_id>/<video>.avi

    Lê os labels do CSV padrão do DAiSEE
    (<daisee_root>/Labels/<split>Labels.csv).

    Args:
        daisee_root: caminho raiz do dataset DAiSEE descompactado.
        output_dir: diretório de saída para os pares .npy + .json.
        split: "Train", "Validation" ou "Test".
        max_videos: limita o número de vídeos (útil para debug).
    """
    import csv

    daisee_root = Path(daisee_root)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carrega labels
    labels_csv = daisee_root / "Labels" / f"{split}Labels.csv"
    label_map: Dict[str, Dict] = {}
    if labels_csv.exists():
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip = Path(row.get("ClipID", row.get("VideoID", ""))).stem
                label_map[clip] = {
                    "engagement":  int(row.get("Engagement",  0)),
                    "boredom":     int(row.get("Boredom",     0)),
                    "confusion":   int(row.get("Confusion",   0)),
                    "frustration": int(row.get("Frustration", 0)),
                }
    else:
        print(f"[AVISO] Arquivo de labels não encontrado: {labels_csv}")

    # Encontra todos os vídeos no split
    video_paths = sorted((daisee_root / split).rglob("*.avi"))
    video_paths += sorted((daisee_root / split).rglob("*.mp4"))

    if max_videos:
        video_paths = video_paths[:max_videos]

    print(f"\n  DAiSEE batch: {split} | {len(video_paths)} vídeos")
    print(f"  Labels carregados: {len(label_map)}")
    print("-" * 55)

    ok_count = err_count = 0
    for i, vp in enumerate(video_paths, 1):
        stem = vp.stem
        label_data = label_map.get(stem, {})
        engagement_label = label_data.get("engagement", None)
        subject_id = vp.parent.parent.name  # convenção de pastas DAiSEE

        # Evita reprocessar se já existe
        if (output_dir / f"{stem}.npy").exists():
            print(f"  [{i:4d}/{len(video_paths)}] SKIP (já existe): {stem}")
            continue

        print(f"  [{i:4d}/{len(video_paths)}] {stem}", end=" ")
        try:
            process_video(
                video_path=vp,
                output_dir=output_dir,
                label=engagement_label,
                subject_id=subject_id,
                show_progress=False,
            )
            print("OK")
            ok_count += 1
        except Exception as exc:
            print(f"ERRO: {exc}")
            err_count += 1

    print(f"\n  Concluído: {ok_count} OK | {err_count} erros")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MindFlow — extração de features para vídeo")
    sub = parser.add_subparsers(dest="cmd")

    # subcomando: processar um único vídeo
    p_single = sub.add_parser("single", help="Processa um único vídeo")
    p_single.add_argument("--video",   required=True, help="Caminho do vídeo")
    p_single.add_argument("--output",  required=True, help="Diretório de saída")
    p_single.add_argument("--label",   type=int, default=None, help="Label de engajamento (0-3)")
    p_single.add_argument("--subject", default=None, help="ID do sujeito")
    p_single.add_argument("--skip",    type=int, default=0, help="Frames a pular")

    # subcomando: batch do DAiSEE
    p_batch = sub.add_parser("daisee", help="Processa batch do dataset DAiSEE")
    p_batch.add_argument("--root",   required=True, help="Raiz do DAiSEE descompactado")
    p_batch.add_argument("--output", required=True, help="Diretório de saída")
    p_batch.add_argument("--split",  default="Train", choices=["Train", "Validation", "Test"])
    p_batch.add_argument("--max",    type=int, default=None, help="Máximo de vídeos a processar")

    args = parser.parse_args()

    if args.cmd == "single":
        process_video(
            video_path=args.video,
            output_dir=args.output,
            label=args.label,
            subject_id=args.subject,
            skip_frames=args.skip,
        )
    elif args.cmd == "daisee":
        process_daisee_batch(
            daisee_root=args.root,
            output_dir=args.output,
            split=args.split,
            max_videos=args.max,
        )
    else:
        parser.print_help()
