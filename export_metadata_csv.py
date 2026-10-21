#!/usr/bin/env python
"""Exporta blocos de metadados do corpus para arquivos CSV.
Blocos gerados (exceto cache global e dados de execução):
- grain_identity.csv   → id, arquivo, posição, escala, onset, classe
- grain_descriptors.csv → descritores acústicos por grão
- files.csv            → estatísticas por arquivo de amostra
- selection_summary.csv → métricas principais de seleção (key/value)
"""
import csv
import sys
from pathlib import Path

# garantir que o src/ está no path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.engine import Corpus

corpus = Corpus.load()

out_dir = Path("metadata_csv")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. grain_identity.csv
# ------------------------------------------------------------
with open(out_dir / "grain_identity.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "grain_id", "file_name", "start_sample", "start_seconds",
        "duration_seconds", "scale_ms", "onset", "class"
    ])
    for i, g in enumerate(corpus.grains):
        start_s = g.get("start") / corpus.sr if corpus.sr else ""
        dur_s = len(g.get("audio", [])) / corpus.sr if corpus.sr else ""
        w.writerow([
            i,
            g.get("file", ""),
            g.get("start", ""),
            round(start_s, 4) if start_s else "",
            round(dur_s, 4) if dur_s else "",
            round(g.get("scale", 0) * 1000, 1) if g.get("scale") else "",
            "Y" if g.get("onset") else "N",
            g.get("class", "")
        ])

# ------------------------------------------------------------
# 2. grain_descriptors.csv
# ------------------------------------------------------------
with open(out_dir / "grain_descriptors.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["grain_id", "zcr", "centroid", "flatness", "hf_ratio",
                "rms", "f0", "voicing", "noisy"])
    for i, g in enumerate(corpus.grains):
        w.writerow([
            i,
            g.get("zcr", ""),
            g.get("centroid", ""),
            g.get("flatness", ""),
            g.get("hf_ratio", ""),
            g.get("rms", ""),
            g.get("f0", ""),
            g.get("voicing", ""),
            g.get("noisy", "")
        ])

# ------------------------------------------------------------
# 3. files.csv
# ------------------------------------------------------------
with open(out_dir / "files.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "duration_seconds", "non_silent_ratio", "candidates", "selected"])
    for name, stats in corpus.file_stats.items():
        dur = stats.get("duration", 0.0)
        non_s = stats.get("non_silent", 0.0)
        cand = stats.get("candidates", 0)
        sel = stats.get("selected", 0)
        w.writerow([name, round(dur, 3), round(non_s, 3), cand, sel])

# ------------------------------------------------------------
# 4. selection_summary.csv  (key / value)
# ------------------------------------------------------------
with open(out_dir / "selection_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    sel = corpus.selection_info
    for k, v in sel.items():
        if isinstance(v, (int, float)):
            w.writerow([k, v])
        elif isinstance(v, dict):
            for subk, subv in v.items():
                w.writerow([f"{k}.{subk}", subv])
        else:
            w.writerow([k, str(v)])

print(f"Arquivos CSV gravados em {out_dir.resolve()}")
for p in sorted(out_dir.iterdir()):
    print(f"  - {p.name}  ({p.stat().st_size} bytes)")