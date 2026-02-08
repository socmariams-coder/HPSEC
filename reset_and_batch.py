# -*- coding: utf-8 -*-
"""
Reset i Batch Import per desenvolupament.

Opcions:
  --reset-cal     Reset calibracions (calibration_result.json)
  --reset-ana     Reset anàlisi (analysis_result.json)
  --reset-all     Reset tot (CHECK/data/)
  --reset-history Reset històric global (REGISTRY/KHP_History.json)
  --import        Fer batch import
  --folder        Carpeta amb SEQs (defecte: Dades3)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime


def reset_calibrations(data_folder: str, include_history: bool = False):
    """Esborra calibration_result.json de totes les SEQs."""
    data_path = Path(data_folder)
    count = 0

    for seq_dir in data_path.glob("*_SEQ*"):
        if not seq_dir.is_dir():
            continue

        cal_file = seq_dir / "CHECK" / "data" / "calibration_result.json"
        if cal_file.exists():
            try:
                cal_file.unlink()
                print(f"  [DEL] {seq_dir.name}/calibration_result.json")
                count += 1
            except Exception as e:
                print(f"  [ERR] {seq_dir.name}: {e}")

    print(f"\nCalibrations reset: {count} fitxers esborrats")

    if include_history:
        reset_history(data_folder)


def reset_analysis(data_folder: str):
    """Esborra analysis_result.json de totes les SEQs."""
    data_path = Path(data_folder)
    count = 0

    for seq_dir in data_path.glob("*_SEQ*"):
        if not seq_dir.is_dir():
            continue

        ana_file = seq_dir / "CHECK" / "data" / "analysis_result.json"
        if ana_file.exists():
            try:
                ana_file.unlink()
                print(f"  [DEL] {seq_dir.name}/analysis_result.json")
                count += 1
            except Exception as e:
                print(f"  [ERR] {seq_dir.name}: {e}")

    print(f"\nAnalysis reset: {count} fitxers esborrats")


def reset_all(data_folder: str):
    """Esborra tot CHECK/data/ de totes les SEQs."""
    data_path = Path(data_folder)
    count = 0

    for seq_dir in data_path.glob("*_SEQ*"):
        if not seq_dir.is_dir():
            continue

        check_data = seq_dir / "CHECK" / "data"
        if check_data.exists():
            try:
                shutil.rmtree(check_data)
                print(f"  [DEL] {seq_dir.name}/CHECK/data/")
                count += 1
            except Exception as e:
                print(f"  [ERR] {seq_dir.name}: {e}")

    print(f"\nFull reset: {count} carpetes esborrades")


def reset_history(data_folder: str):
    """Esborra l'històric global de calibracions."""
    data_path = Path(data_folder)
    registry = data_path / "REGISTRY"

    history_files = [
        "KHP_History.json",
        "Calibration_History.json",
        "QC_History.json",
    ]

    for hf in history_files:
        hf_path = registry / hf
        if hf_path.exists():
            # Fer backup abans d'esborrar
            backup = registry / f"{hf}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.copy(hf_path, backup)
                hf_path.unlink()
                print(f"  [DEL] REGISTRY/{hf} (backup: {backup.name})")
            except Exception as e:
                print(f"  [ERR] {hf}: {e}")

    print("\nHistory reset complete")


def batch_import(data_folder: str, dev_notes: bool = True):
    """Fa import de totes les SEQs."""
    # Activar dev notes si cal
    if dev_notes:
        os.environ["HPSEC_DEV_NOTES"] = "1"
        from hpsec_dev_notes import start_session, end_session, add_note
        start_session("Batch import")

    from hpsec_import import import_sequence, save_import_manifest
    from hpsec_reports import generate_import_plots

    data_path = Path(data_folder)
    seq_dirs = sorted([d for d in data_path.glob("*_SEQ*") if d.is_dir()])

    print(f"\nBatch import: {len(seq_dirs)} SEQs")
    print("=" * 50)

    results = {"ok": 0, "fail": 0, "skip": 0}

    for seq_dir in seq_dirs:
        # Saltar si ja té import
        manifest = seq_dir / "CHECK" / "data" / "import_manifest.json"
        if manifest.exists():
            print(f"  [SKIP] {seq_dir.name} (ja importat)")
            results["skip"] += 1
            continue

        try:
            print(f"  [IMP] {seq_dir.name}...", end=" ", flush=True)
            result = import_sequence(str(seq_dir))
            if result.get("success"):
                # IMPORTANT: Guardar manifest!
                save_import_manifest(result)
                # Generar grafics
                try:
                    generate_import_plots(str(seq_dir), result)
                except Exception:
                    pass
                print("OK")
                results["ok"] += 1
            else:
                # Extreure missatge d'error (errors és una llista)
                errors = result.get('errors', [])
                error_msg = errors[0] if errors else "Error desconegut"
                print(f"FAIL: {error_msg}")
                results["fail"] += 1
                if dev_notes:
                    add_note(seq_dir.name, "import", f"Import failed: {error_msg}", "error")
        except Exception as e:
            print(f"ERROR: {e}")
            results["fail"] += 1

    print("=" * 50)
    print(f"OK: {results['ok']} | FAIL: {results['fail']} | SKIP: {results['skip']}")

    if dev_notes:
        end_session(f"Import: {results['ok']} OK, {results['fail']} fail")


def batch_calibrate(data_folder: str, dev_notes: bool = True, force: bool = False):
    """Fa calibracio de totes les SEQs importades."""
    if dev_notes:
        os.environ["HPSEC_DEV_NOTES"] = "1"
        from hpsec_dev_notes import start_session, end_session, add_note
        start_session("Batch calibrate")

    from hpsec_import import import_from_manifest
    from hpsec_calibrate import calibrate_from_import
    from hpsec_reports import generate_calibration_plots

    data_path = Path(data_folder)
    seq_dirs = sorted([d for d in data_path.glob("*_SEQ*") if d.is_dir()])

    print(f"\nBatch calibrate: {len(seq_dirs)} SEQs")
    print("=" * 50)

    results = {"ok": 0, "fail": 0, "skip": 0, "no_import": 0}

    for seq_dir in seq_dirs:
        manifest_path = seq_dir / "CHECK" / "data" / "import_manifest.json"
        cal_path = seq_dir / "CHECK" / "data" / "calibration_result.json"

        # Verificar que hi ha import
        if not manifest_path.exists():
            print(f"  [SKIP] {seq_dir.name} (no importat)")
            results["no_import"] += 1
            continue

        # Saltar si ja calibrat (excepte force)
        if cal_path.exists() and not force:
            print(f"  [SKIP] {seq_dir.name} (ja calibrat)")
            results["skip"] += 1
            continue

        try:
            print(f"  [CAL] {seq_dir.name}...", end=" ", flush=True)

            # Carregar dades des de manifest
            imported = import_from_manifest(str(seq_dir))
            if not imported or not imported.get("success"):
                errors = imported.get("errors", []) if imported else []
                error_msg = errors[0] if errors else "manifest invalid"
                print(f"FAIL: {error_msg}")
                results["fail"] += 1
                continue

            # Calibrar
            result = calibrate_from_import(imported)

            if result and result.get("success"):
                # Generar grafics
                try:
                    generate_calibration_plots(str(seq_dir), result, imported)
                except Exception:
                    pass
                source = result.get("khp_source", "?")
                mode = result.get("mode", "?")
                print(f"OK ({source}, {mode})")
                results["ok"] += 1
            else:
                errors = result.get("errors", []) if result else []
                error_msg = errors[0] if errors else "Error desconegut"
                print(f"FAIL: {error_msg}")
                results["fail"] += 1
                if dev_notes:
                    add_note(seq_dir.name, "calibrate", f"Calibration failed: {error_msg}", "error")

        except Exception as e:
            print(f"ERROR: {e}")
            results["fail"] += 1

    print("=" * 50)
    print(f"OK: {results['ok']} | FAIL: {results['fail']} | SKIP: {results['skip']} | NO_IMP: {results['no_import']}")

    if dev_notes:
        end_session(f"Calibrate: {results['ok']} OK, {results['fail']} fail")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reset i Batch per desenvolupament")
    parser.add_argument("--folder", type=str, default="C:/Users/Lequia/Desktop/Dades3",
                       help="Carpeta amb SEQs")
    parser.add_argument("--reset-cal", action="store_true", help="Reset calibracions")
    parser.add_argument("--reset-ana", action="store_true", help="Reset anàlisi")
    parser.add_argument("--reset-all", action="store_true", help="Reset tot CHECK/data/")
    parser.add_argument("--reset-history", action="store_true", help="Reset històric global")
    parser.add_argument("--import", dest="do_import", action="store_true", help="Batch import")
    parser.add_argument("--calibrate", action="store_true", help="Batch calibrate")
    parser.add_argument("--force", action="store_true", help="Forcar recalibrar/reimportar")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"RESET & BATCH - {args.folder}")
    print(f"{'='*60}")

    if args.reset_all:
        print("\n[1] RESET COMPLET (CHECK/data/)")
        confirm = input("    Confirmes? (s/N): ")
        if confirm.lower() == 's':
            reset_all(args.folder)

    if args.reset_cal:
        print("\n[2] RESET CALIBRACIONS")
        reset_calibrations(args.folder, include_history=args.reset_history)

    if args.reset_ana:
        print("\n[3] RESET ANÀLISI")
        reset_analysis(args.folder)

    if args.reset_history and not args.reset_cal:
        print("\n[4] RESET HISTÒRIC")
        reset_history(args.folder)

    if args.do_import:
        print("\n[5] BATCH IMPORT")
        batch_import(args.folder)

    if args.calibrate:
        print("\n[6] BATCH CALIBRATE")
        batch_calibrate(args.folder, force=args.force)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
