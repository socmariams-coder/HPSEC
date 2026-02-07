# -*- coding: utf-8 -*-
"""
Batch processing amb DEV NOTES activades.

Fa un reset, reimporta i analitza les SEQs per capturar
problemes de detecció (batman, etc.) a Development_Notes.json

Ús:
    python batch_dev_test.py [--seqs 283,284,285]
"""

import os
import sys
from pathlib import Path

# ACTIVAR DEV NOTES
os.environ["HPSEC_DEV_NOTES"] = "1"

from hpsec_dev_notes import start_session, end_session, add_note, print_summary
from hpsec_import import import_sequence
from hpsec_calibrate import calibrate_from_import
from hpsec_analyze import analyze_sequence


def process_sequence(seq_path: str) -> dict:
    """Processa una seqüència completa."""
    seq_name = Path(seq_path).name
    result = {"seq_name": seq_name, "success": False, "stages": {}}

    # 1. IMPORT
    print(f"\n[1/3] Importar {seq_name}...")
    try:
        import_result = import_sequence(seq_path)
        result["stages"]["import"] = "OK" if import_result.get("success") else "FAIL"
        if not import_result.get("success"):
            add_note(seq_name, "import", f"Import fallit: {import_result.get('error', '?')}", "error")
            return result
    except Exception as e:
        result["stages"]["import"] = f"ERROR: {e}"
        add_note(seq_name, "import", f"Exception: {e}", "error")
        return result

    # 2. CALIBRAR
    print(f"[2/3] Calibrar {seq_name}...")
    try:
        cal_result = calibrate_from_import(import_result)
        result["stages"]["calibrate"] = "OK" if cal_result.get("success") else "FAIL"

        # Capturar info calibració
        if cal_result.get("has_batman"):
            add_note(seq_name, "calibrate", "KHP amb batman detectat", "warning",
                    {"khp_name": cal_result.get("khp_name", "?")})

    except Exception as e:
        result["stages"]["calibrate"] = f"ERROR: {e}"
        add_note(seq_name, "calibrate", f"Exception: {e}", "error")
        # Continuar amb calibració per defecte
        cal_result = {"success": True, "shift_direct": 0, "shift_uib": 0}

    # 3. ANALITZAR
    print(f"[3/3] Analitzar {seq_name}...")
    try:
        analysis_result = analyze_sequence(
            imported_data=import_result,
            calibration_data=cal_result,
            save_json=True
        )
        result["stages"]["analyze"] = "OK" if analysis_result.get("success") else "FAIL"

        # Resum anomalies
        samples = analysis_result.get("samples_analyzed", {})
        n_batman = sum(1 for s in samples.values()
                      if s.get("batman_direct") or s.get("batman_uib"))
        if n_batman > 0:
            add_note(seq_name, "analyze", f"{n_batman} mostres amb batman detectat", "warning")

        result["success"] = True

    except Exception as e:
        result["stages"]["analyze"] = f"ERROR: {e}"
        add_note(seq_name, "analyze", f"Exception: {e}", "error")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch dev test amb notes")
    parser.add_argument("--seqs", type=str, help="Llista SEQs (ex: 283,284,285)")
    parser.add_argument("--folder", type=str, default="C:/Users/Lequia/Desktop/Dades3",
                       help="Carpeta amb SEQs")
    parser.add_argument("--pattern", type=str, default="*_SEQ*",
                       help="Patró per cercar SEQs")
    args = parser.parse_args()

    data_folder = Path(args.folder)

    # Determinar SEQs a processar
    if args.seqs:
        seq_names = [s.strip() for s in args.seqs.split(",")]
        seq_paths = [data_folder / name for name in seq_names if (data_folder / name).exists()]
    else:
        seq_paths = sorted([p for p in data_folder.glob(args.pattern) if p.is_dir()])

    if not seq_paths:
        print("No s'han trobat SEQs a processar")
        return

    print(f"\n{'='*60}")
    print(f"BATCH DEV TEST - {len(seq_paths)} SEQs")
    print(f"{'='*60}")

    # Iniciar sessió dev notes
    start_session(f"Batch test {len(seq_paths)} SEQs")

    results = []
    for seq_path in seq_paths:
        try:
            result = process_sequence(str(seq_path))
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {seq_path.name}: {e}")
            add_note(seq_path.name, "batch", f"Exception global: {e}", "error")

    # Resum
    print(f"\n{'='*60}")
    print("RESUM BATCH")
    print(f"{'='*60}")

    ok_count = sum(1 for r in results if r.get("success"))
    print(f"Processades: {len(results)} | OK: {ok_count} | FAIL: {len(results)-ok_count}")

    for r in results:
        status = "✓" if r.get("success") else "✗"
        stages = " → ".join(f"{k}:{v}" for k, v in r.get("stages", {}).items())
        print(f"  {status} {r['seq_name']}: {stages}")

    # Finalitzar sessió i mostrar notes
    end_session(f"Processades {len(results)} SEQs")

    print("\n")
    print_summary()


if __name__ == "__main__":
    main()
