# -*- coding: utf-8 -*-
"""
HPSEC Suite - Reset per etapa amb cascade
==========================================

Mòdul core reutilitzable (sense dependències GUI).
Esborra JSONs i outputs downstream des d'una etapa concreta.

Cascade:
- Reset Import (0)   → esborra 0+1+2+3
- Reset Calibrar (1)  → esborra 1+2+3
- Reset Analitzar (2) → esborra 2+3
- Reset Revisar (3)   → esborra 3 (RESULTATS/ + SUMMARY)

user_notes.json es MANTÉ sempre (no s'esborra mai).
"""

import os
import shutil
from pathlib import Path

# Fitxers associats a cada etapa
STAGE_FILES = {
    0: {"json": "import_manifest.json",     "plots": "PLOTS_Import_*"},
    1: {"json": "calibration_result.json",   "plots": "PLOTS_Calibration_*"},
    2: {"json": "analysis_result.json",      "plots": "PLOTS_Analysis_*"},
    3: {"json": "review_result.json",        "extra": ["RESULTATS/", "CHECK/SUMMARY.xlsx"]},
}

STAGE_NAMES = {0: "Importar", 1: "Calibrar", 2: "Analitzar", 3: "Revisar"}


def reset_stage(seq_path, from_stage, dry_run=False):
    """Reset des de from_stage fins a 3 (cascade).

    Args:
        seq_path: Path de la seqüència (ex: .../283_SEQ)
        from_stage: Etapa des de la qual resetejar (0-3)
        dry_run: Si True, només llista què s'esborraria

    Returns:
        {"deleted": [...], "errors": [...], "dry_run": bool}
    """
    seq_path = Path(seq_path)
    data_path = seq_path / "CHECK" / "data"
    check_path = seq_path / "CHECK"

    deleted = []
    errors = []

    for stage in range(from_stage, 4):
        stage_info = STAGE_FILES.get(stage, {})

        # Esborrar JSON de CHECK/data/
        json_name = stage_info.get("json")
        if json_name:
            json_file = data_path / json_name
            if json_file.exists():
                if dry_run:
                    deleted.append(str(json_file))
                else:
                    try:
                        json_file.unlink()
                        deleted.append(str(json_file))
                    except Exception as e:
                        errors.append(f"{json_file}: {e}")

        # Esborrar PDFs/plots de CHECK/ (glob pattern)
        plots_pattern = stage_info.get("plots")
        if plots_pattern and check_path.exists():
            for f in check_path.glob(plots_pattern):
                if dry_run:
                    deleted.append(str(f))
                else:
                    try:
                        if f.is_dir():
                            shutil.rmtree(f)
                        else:
                            f.unlink()
                        deleted.append(str(f))
                    except Exception as e:
                        errors.append(f"{f}: {e}")

        # Extras (stage 3: RESULTATS/, SUMMARY)
        for extra in stage_info.get("extra", []):
            extra_path = seq_path / extra
            if extra_path.exists():
                if dry_run:
                    deleted.append(str(extra_path))
                else:
                    try:
                        if extra_path.is_dir():
                            shutil.rmtree(extra_path)
                        else:
                            extra_path.unlink()
                        deleted.append(str(extra_path))
                    except Exception as e:
                        errors.append(f"{extra_path}: {e}")

    return {"deleted": deleted, "errors": errors, "dry_run": dry_run}


def reset_batch(seq_paths, from_stage, progress_cb=None, dry_run=False):
    """Reset per múltiples seqüències amb callback de progrés.

    Args:
        seq_paths: Llista de paths de seqüències
        from_stage: Etapa des de la qual resetejar (0-3)
        progress_cb: Callable(current, total, seq_name) o None
        dry_run: Si True, només llista què s'esborraria

    Returns:
        {"total": N, "ok": N, "fail": N, "details": {seq_name: result}}
    """
    total = len(seq_paths)
    ok_count = 0
    fail_count = 0
    details = {}

    for i, sp in enumerate(seq_paths):
        seq_name = os.path.basename(sp)
        if progress_cb:
            progress_cb(i + 1, total, seq_name)

        result = reset_stage(sp, from_stage, dry_run=dry_run)
        details[seq_name] = result

        if result["errors"]:
            fail_count += 1
        else:
            ok_count += 1

    return {"total": total, "ok": ok_count, "fail": fail_count, "details": details}
