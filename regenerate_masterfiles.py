# -*- coding: utf-8 -*-
"""
Regenera tots els MasterFiles amb la nova columna Sample_Rep.

Ús:
    python regenerate_masterfiles.py          # Llista SEQs disponibles
    python regenerate_masterfiles.py --all    # Regenera TOTS
    python regenerate_masterfiles.py 283_SEQ  # Regenera una SEQ específica
"""

import os
import sys
from pathlib import Path

# Afegir path del projecte
sys.path.insert(0, str(Path(__file__).parent))

from hpsec_config import get_config
from hpsec_migrate_master import migrate_single


def find_sequences(data_folder):
    """Troba totes les carpetes SEQ."""
    sequences = []
    for item in sorted(Path(data_folder).iterdir()):
        if item.is_dir() and "_SEQ" in item.name.upper():
            sequences.append(item)
    return sequences


def regenerate_masterfile(seq_path, force=True):
    """Regenera el MasterFile d'una SEQ."""
    result = migrate_single(str(seq_path), force=force)
    return result


def main():
    cfg = get_config()
    data_folder = cfg.get("paths", "data_folder")

    sequences = find_sequences(data_folder)

    if len(sys.argv) < 2:
        print("=" * 60)
        print("REGENERAR MASTERFILES")
        print("=" * 60)
        print(f"\nCarpeta: {data_folder}")
        print(f"SEQs trobades: {len(sequences)}")
        print()
        for seq in sequences:
            mf = list(seq.glob("*MasterFile*.xlsx"))
            status = "[OK]" if mf else "[--]"
            print(f"  {status} {seq.name}")
        print()
        print("Us:")
        print("  python regenerate_masterfiles.py --all     # Regenera TOTS")
        print("  python regenerate_masterfiles.py 283_SEQ   # Regenera una SEQ")
        return

    # Determinar quines SEQs processar
    if "--all" in sys.argv:
        to_process = sequences
    else:
        # Buscar SEQ específica
        seq_name = sys.argv[1]
        to_process = [s for s in sequences if seq_name in s.name]
        if not to_process:
            print(f"ERROR: No s'ha trobat SEQ '{seq_name}'")
            return

    print("=" * 60)
    print(f"REGENERANT {len(to_process)} MASTERFILES")
    print("=" * 60)

    success = 0
    errors = []

    for i, seq in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {seq.name}...")

        result = regenerate_masterfile(seq)
        status = result.get("status", "error")

        if status == "ok":
            print(f"  [OK] {result.get('file', '?')}")
            success += 1
        elif status == "need_input":
            print(f"  [!!] Necessita rawdata manual")
            errors.append((seq.name, "need_input"))
        else:
            msg = result.get("message", "Error desconegut")
            print(f"  [ERR] {msg}")
            errors.append((seq.name, msg))

    print()
    print("=" * 60)
    print(f"RESULTAT: {success}/{len(to_process)} correctes")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
