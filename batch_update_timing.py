# -*- coding: utf-8 -*-
"""
batch_update_timing.py - Actualitza timing stats a tots els consolidation.json
==============================================================================

Script ràpid per afegir/actualitzar la secció "timing" als JSONs existents
sense necessitat de reconsolidar tot.

Ús:
    python batch_update_timing.py [base_folder]

Autor: HPSEC Suite
"""

import os
import sys
import glob
import json
from datetime import datetime

# Importar funcions necessàries
from hpsec_consolidate import (
    llegir_masterfile_nou,
    calculate_timing_stats,
    detect_master_format,
)


def find_masterfile(seq_path):
    """Troba el MasterFile d'una SEQ."""
    patterns = ['*MasterFile*.xlsx', '*_MasterFile.xlsx', '*MasterFile.xlsx']
    for pattern in patterns:
        matches = glob.glob(os.path.join(seq_path, pattern))
        matches = [m for m in matches if '~$' not in m]
        if matches:
            return matches[0]
    return None


def update_seq_timing(seq_path, verbose=True):
    """
    Actualitza el timing d'una SEQ.

    Returns:
        dict amb resultat: {"status": "updated"|"created"|"skipped"|"error", "message": str}
    """
    seq_name = os.path.basename(seq_path)

    # Trobar MasterFile
    masterfile = find_masterfile(seq_path)
    if not masterfile:
        return {"status": "skipped", "message": "No MasterFile"}

    # Verificar format
    fmt = detect_master_format(masterfile)
    if fmt != "NEW":
        return {"status": "skipped", "message": f"Format {fmt} (no NEW)"}

    # Llegir MasterFile
    try:
        master_data = llegir_masterfile_nou(masterfile)
        if "error" in master_data:
            return {"status": "error", "message": master_data["error"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # Calcular timing stats
    timing_stats = calculate_timing_stats(master_data)
    if not timing_stats:
        return {"status": "skipped", "message": "No timing data available"}

    # Path al JSON
    check_folder = os.path.join(seq_path, "CHECK")
    json_path = os.path.join(check_folder, "consolidation.json")

    # Llegir JSON existent o crear nou
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            action = "updated"
        except:
            data = {}
            action = "created"
    else:
        # Crear carpeta CHECK si no existeix
        os.makedirs(check_folder, exist_ok=True)
        data = {
            "meta": {
                "seq": seq_name.replace("_SEQ", "").replace("_BP", ""),
                "generated_at": datetime.now().isoformat(),
                "note": "Minimal JSON - only timing stats"
            }
        }
        action = "created"

    # Actualitzar timing
    data["timing"] = timing_stats
    data["meta"] = data.get("meta", {})
    data["meta"]["timing_updated_at"] = datetime.now().isoformat()

    # Guardar
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return {"status": action, "message": f"OK - timing added"}
    except Exception as e:
        return {"status": "error", "message": f"Save error: {e}"}


def main():
    # Base folder
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        base_folder = "C:/Users/Lequia/Desktop/Dades2"

    print("=" * 70)
    print("BATCH UPDATE TIMING STATS")
    print("=" * 70)
    print(f"Base folder: {base_folder}")
    print()

    # Trobar totes les SEQs
    seqs = sorted(glob.glob(os.path.join(base_folder, "*_SEQ*")))
    print(f"SEQs trobades: {len(seqs)}")
    print("-" * 70)

    # Processar
    stats = {"updated": 0, "created": 0, "skipped": 0, "error": 0}

    for seq_path in seqs:
        seq_name = os.path.basename(seq_path)
        result = update_seq_timing(seq_path)

        status = result["status"]
        stats[status] = stats.get(status, 0) + 1

        # Output (ASCII safe)
        icon = {"updated": "OK", "created": "NEW", "skipped": "SKIP", "error": "ERR"}.get(status, "?")
        print(f"  [{icon:<4}] {seq_name:<20} {result['message']}")

    # Resum
    print("-" * 70)
    print(f"RESUM: {stats['updated']} updated, {stats['created']} created, "
          f"{stats['skipped']} skipped, {stats['error']} errors")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    main()
