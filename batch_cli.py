# -*- coding: utf-8 -*-
"""
HPSEC Suite - CLI unificat per batch processing i reset
========================================================

Ús:
    python batch_cli.py status
    python batch_cli.py reset --from import
    python batch_cli.py reset --from calibrate --seqs 283,284
    python batch_cli.py reset --from analyze --all --dry-run
    python batch_cli.py process --stage import
    python batch_cli.py process --stage calibrate --seqs 283,284
    python batch_cli.py process --pipeline --all
"""

import argparse
import sys
import os
from pathlib import Path

# Assegurar que el directori arrel estigui al path
sys.path.insert(0, str(Path(__file__).parent))

from hpsec_config import get_config
from gui.models.sequence_state import get_all_sequences, Phase
from hpsec_reset import reset_stage, reset_batch, STAGE_NAMES

# Mapeig nom d'etapa → número
STAGE_MAP = {
    "import": 0, "importar": 0,
    "calibrate": 1, "calibrar": 1,
    "analyze": 2, "analitzar": 2, "analyse": 2,
    "review": 3, "revisar": 3,
}


def _get_data_folder():
    cfg = get_config()
    return cfg.get("paths", "data_folder")


def _get_sequences(data_folder):
    return get_all_sequences(data_folder)


def _filter_by_nums(sequences, seq_nums):
    """Filtra seqüències per números (ex: 283,284)."""
    nums = set()
    for s in seq_nums.split(","):
        s = s.strip()
        if s:
            nums.add(s)

    filtered = []
    for seq in sequences:
        # Extreure número de seq_name (ex: 283_SEQ → 283, 283B_SEQ → 283)
        name = seq.seq_name.upper().replace("_SEQ", "").replace("_BP", "")
        # Treure sufixos de lletra (283B → 283)
        base_num = name.rstrip("ABCDEFGH")
        if base_num in nums or name in nums or seq.seq_name in nums:
            filtered.append(seq)
    return filtered


def cmd_status(args):
    """Mostra l'estat de totes les seqüències."""
    data_folder = _get_data_folder()
    sequences = _get_sequences(data_folder)

    if not sequences:
        print(f"Cap seqüència trobada a {data_folder}")
        return

    print(f"{'#':<4} {'Sequencia':<20} {'Mode':<6} {'Tipus':<7} {'Imp':>4} {'Cal':>4} {'Ana':>4} {'Rev':>4} {'%':>4}")
    print("-" * 75)

    for i, seq in enumerate(sequences, 1):
        imp = "OK" if seq.import_status.completed else "--"
        cal = "OK" if seq.calibrate_status.completed else "--"
        ana = "OK" if seq.analyze_status.completed else "--"
        rev = "OK" if seq.review_status.completed else "--"
        pct = seq.progress_pct

        print(f"{i:<4} {seq.seq_name:<20} {seq.data_mode or '-':<6} {seq.method or '-':<7} {imp:>4} {cal:>4} {ana:>4} {rev:>4} {pct:>3}%")

    # Resum
    total = len(sequences)
    imported = sum(1 for s in sequences if s.import_status.completed)
    calibrated = sum(1 for s in sequences if s.calibrate_status.completed)
    analyzed = sum(1 for s in sequences if s.analyze_status.completed)
    print(f"\nTotal: {total} | Imp: {imported} | Cal: {calibrated} | Ana: {analyzed}")


def cmd_reset(args):
    """Reseteja seqüències des d'una etapa."""
    from_stage = STAGE_MAP.get(args.stage.lower())
    if from_stage is None:
        print(f"Etapa no vàlida: {args.stage}")
        print(f"Opcions: {', '.join(STAGE_MAP.keys())}")
        sys.exit(1)

    data_folder = _get_data_folder()
    sequences = _get_sequences(data_folder)

    if args.seqs:
        sequences = _filter_by_nums(sequences, args.seqs)

    if not sequences:
        print("Cap seqüència seleccionada.")
        return

    # Mostrar cascade
    stages_affected = [STAGE_NAMES[s] for s in range(from_stage, 4)]
    print(f"Reset des de '{STAGE_NAMES[from_stage]}' (cascade: {' > '.join(stages_affected)})")
    print(f"Seqüències: {len(sequences)}")
    if args.dry_run:
        print("[DRY-RUN] No s'esborrarà res.")

    seq_paths = [seq.seq_path for seq in sequences]

    def progress(current, total, name):
        print(f"  [{current}/{total}] {name}")

    result = reset_batch(seq_paths, from_stage, progress_cb=progress, dry_run=args.dry_run)

    print(f"\nResultat: {result['ok']} OK, {result['fail']} errors")

    if args.dry_run:
        for name, detail in result["details"].items():
            if detail["deleted"]:
                print(f"\n  {name}:")
                for f in detail["deleted"]:
                    print(f"    - {os.path.basename(f)}")


def cmd_process(args):
    """Processa seqüències per etapa."""
    from gui.widgets.dashboard_panel import run_import, run_calibrate, run_analyze

    data_folder = _get_data_folder()
    sequences = _get_sequences(data_folder)

    if args.seqs:
        sequences = _filter_by_nums(sequences, args.seqs)

    if not sequences:
        print("Cap seqüència seleccionada.")
        return

    # Determinar fases
    if args.pipeline:
        phases = [Phase.IMPORT, Phase.CALIBRATE, Phase.ANALYZE]
        phase_label = "Pipeline complet"
    elif args.stage:
        stage = STAGE_MAP.get(args.stage.lower())
        if stage is None:
            print(f"Etapa no vàlida: {args.stage}")
            sys.exit(1)
        phases = [
            [Phase.IMPORT, Phase.CALIBRATE, Phase.ANALYZE][stage]
        ] if stage < 3 else []
        phase_label = STAGE_NAMES.get(stage, args.stage)
    else:
        print("Cal especificar --stage o --pipeline")
        sys.exit(1)

    print(f"{phase_label}: {len(sequences)} seqüències")
    print("-" * 50)

    ok_count, fail_count = 0, 0

    for phase in phases:
        phase_name = phase.value
        for i, seq in enumerate(sequences, 1):
            print(f"  [{i}/{len(sequences)}] {seq.seq_name}: {phase_name}...", end=" ", flush=True)

            if phase == Phase.IMPORT:
                siblings = seq.siblings if hasattr(seq, 'siblings') else []
                success, msg, _ = run_import(seq.seq_path, siblings=siblings)
            elif phase == Phase.CALIBRATE:
                success, msg, _ = run_calibrate(seq.seq_path)
            elif phase == Phase.ANALYZE:
                success, msg, _ = run_analyze(seq.seq_path)
            else:
                success, msg = False, "Fase no implementada"

            if success:
                ok_count += 1
                print("OK")
            else:
                fail_count += 1
                print(f"ERROR: {msg}")

    print(f"\nResultat: {ok_count} OK, {fail_count} errors")


def main():
    parser = argparse.ArgumentParser(description="HPSEC Suite - Batch CLI")
    subparsers = parser.add_subparsers(dest="command", help="Comanda")

    # status
    sp_status = subparsers.add_parser("status", help="Mostra estat de totes les seqüències")
    sp_status.set_defaults(func=cmd_status)

    # reset
    sp_reset = subparsers.add_parser("reset", help="Reseteja seqüències des d'una etapa")
    sp_reset.add_argument("--from", dest="stage", required=True,
                          help="Etapa: import, calibrate, analyze, review")
    sp_reset.add_argument("--seqs", help="Números de seqüència separats per comes (ex: 283,284)")
    sp_reset.add_argument("--all", action="store_true", help="Totes les seqüències")
    sp_reset.add_argument("--dry-run", action="store_true", help="Només mostra què s'esborraria")
    sp_reset.set_defaults(func=cmd_reset)

    # process
    sp_process = subparsers.add_parser("process", help="Processa seqüències")
    sp_process.add_argument("--stage", help="Etapa: import, calibrate, analyze")
    sp_process.add_argument("--pipeline", action="store_true", help="Pipeline complet")
    sp_process.add_argument("--seqs", help="Números de seqüència separats per comes")
    sp_process.add_argument("--all", action="store_true", help="Totes les seqüències")
    sp_process.set_defaults(func=cmd_process)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
