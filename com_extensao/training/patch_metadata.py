"""
training/patch_metadata.py
==========================
Atualiza os _metadata.json já extraídos para incluir os 4 labels DAiSEE.

NÃO re-processa nenhum vídeo — só lê o CSV de labels e atualiza os JSONs.
Leva ~5 segundos para todos os splits.

Uso:
    python training/patch_metadata.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DAISEE    = ROOT / "DAiSEE"
LABELS_CSV = DAISEE / "Labels" / "AllLabels.csv"

SPLITS = {
    "train":      ROOT / "outputs" / "features" / "train",
    "validation": ROOT / "outputs" / "features" / "validation",
    "test":       ROOT / "outputs" / "features" / "test",
}


def main():
    # Carrega o CSV completo de labels
    df = pd.read_csv(LABELS_CSV)
    df.columns = df.columns.str.strip()

    # Cria dicionário: clip_id (sem extensão) → {engagement, boredom, confusion, frustration}
    label_map = {}
    for _, row in df.iterrows():
        clip_id = Path(row["ClipID"]).stem   # "1100011002.avi" → "1100011002"
        label_map[clip_id] = {
            "engagement":  int(row["Engagement"]),
            "boredom":     int(row["Boredom"]),
            "confusion":   int(row["Confusion"]),
            "frustration": int(row["Frustration"]),
        }

    print(f"Labels carregados: {len(label_map)} vídeos\n")

    total_updated = 0
    total_missing = 0

    for split, features_dir in SPLITS.items():
        if not features_dir.exists():
            print(f"  [{split}] diretório não encontrado — pulando")
            continue

        meta_files = sorted(features_dir.glob("*_metadata.json"))
        updated = 0
        missing = 0

        for meta_path in meta_files:
            # Extrai clip_id do nome do arquivo: "1100011002_metadata.json" → "1100011002"
            clip_id = meta_path.stem.replace("_metadata", "")

            if clip_id not in label_map:
                missing += 1
                continue

            labels = label_map[clip_id]

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Adiciona/atualiza labels_4d
            meta["labels_4d"] = labels
            # Mantém label principal como engagement (retrocompatibilidade)
            meta["label"] = labels["engagement"]

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            updated += 1

        print(f"  [{split}] {updated} JSONs atualizados | {missing} não encontrados no CSV")
        total_updated += updated
        total_missing += missing

    print(f"\n✅ Total: {total_updated} JSONs atualizados, {total_missing} não encontrados")
    print("   Agora rode: python training/run_training.py")


if __name__ == "__main__":
    main()
