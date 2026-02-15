# -*- coding: utf-8 -*-
"""
Batch: genera MasterFiles per totes les SEQs de Dades3 que no en tinguin.

Ús:
    python batch_masterfiles_dades3.py           # Dry run (llistar estat)
    python batch_masterfiles_dades3.py --run     # Generar els que falten
    python batch_masterfiles_dades3.py --force   # Regenerar TOTS (sobreescriu)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hpsec_migrate_master import migrate_single

DATA_FOLDER = "C:/Users/Lequia/Desktop/Dades3"


def find_sequences(data_folder):
    seqs = []
    for item in sorted(Path(data_folder).iterdir()):
        if item.is_dir() and "_SEQ" in item.name.upper():
            seqs.append(item)
    return seqs


def has_masterfile(seq_path):
    for f in seq_path.glob("*MasterFile*.xlsx"):
        if "backup" not in f.name.lower():
            return True
    return False


def main():
    run = "--run" in sys.argv
    force = "--force" in sys.argv

    sequences = find_sequences(DATA_FOLDER)
    with_mf = [s for s in sequences if has_masterfile(s)]
    without_mf = [s for s in sequences if not has_masterfile(s)]

    print("=" * 65)
    print(f"MASTERFILES — {DATA_FOLDER}")
    print("=" * 65)
    print(f"  Total SEQs:       {len(sequences)}")
    print(f"  Amb MasterFile:   {len(with_mf)}")
    print(f"  Sense MasterFile: {len(without_mf)}")

    if not run and not force:
        print(f"\n  Sense MasterFile:")
        for s in without_mf:
            print(f"    [--] {s.name}")
        print(f"\n  Amb MasterFile:")
        for s in with_mf:
            print(f"    [OK] {s.name}")
        print(f"\n  Ús:")
        print(f"    python {Path(__file__).name} --run     # Generar els que falten")
        print(f"    python {Path(__file__).name} --force   # Regenerar TOTS")
        return

    to_process = sequences if force else without_mf
    print(f"\n  Processant {len(to_process)} SEQs {'(FORCE)' if force else ''}...")
    print("=" * 65)

    success = 0
    skipped = 0
    errors = []

    for i, seq in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {seq.name}...", end=" ", flush=True)

        try:
            result = migrate_single(str(seq), force=force)
            status = result.get("status", "error")

            if status == "ok":
                print(f"OK — {result.get('file', '?')}")
                success += 1
            elif status == "exists":
                print("SKIP (ja existeix)")
                skipped += 1
            elif status == "need_input":
                print("NEED INPUT (rawdata ambigua)")
                errors.append((seq.name, "need_input"))
            else:
                msg = result.get("message", "Error desconegut")
                print(f"ERR — {msg}")
                errors.append((seq.name, msg))
        except Exception as e:
            print(f"EXCEPTION — {e}")
            errors.append((seq.name, str(e)))

    print("\n" + "=" * 65)
    print(f"RESULTAT: {success} generats, {skipped} skipped, {len(errors)} errors")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    print("=" * 65)


if __name__ == "__main__":
    main()
