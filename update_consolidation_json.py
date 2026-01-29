#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update_consolidation_json.py - Actualitza JSON amb info dels Excel consolidats

Llegeix els Excel consolidats existents i afegeix els camps nous al JSON
sense necessitat de reconsolidar:
  - file_dad, file_uib
  - row_start, row_end
  - npts
  - match_confidence (default 100% per consolidats antics)

Ús:
    python update_consolidation_json.py <seq_path>
    python update_consolidation_json.py --all  # Totes les SEQ a Dades2
"""

import os
import sys
import json
import glob
import pandas as pd
from datetime import datetime


def update_json_from_excel(seq_path, verbose=True):
    """
    Actualitza consolidation.json amb info dels Excel consolidats.

    Args:
        seq_path: Ruta de la seqüència
        verbose: Mostrar progrés

    Returns:
        dict amb resultat: {"success": bool, "updated": int, "message": str}
    """
    check_folder = os.path.join(seq_path, "CHECK")
    json_path = os.path.join(check_folder, "consolidation.json")
    con_folder = os.path.join(seq_path, "Resultats_Consolidats")

    # Verificar que existeix el JSON
    if not os.path.exists(json_path):
        return {"success": False, "updated": 0, "message": "No existeix consolidation.json"}

    # Verificar que existeix carpeta consolidats
    if not os.path.isdir(con_folder):
        return {"success": False, "updated": 0, "message": "No existeix Resultats_Consolidats"}

    # Llegir JSON existent
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        return {"success": False, "updated": 0, "message": f"Error llegint JSON: {e}"}

    # Indexar info dels Excel
    excel_info = {}
    excel_files = glob.glob(os.path.join(con_folder, "*.xlsx"))
    excel_files = [f for f in excel_files if "~$" not in f]

    for f in excel_files:
        try:
            df_id = pd.read_excel(f, sheet_name='ID')
            id_dict = dict(zip(df_id['Field'], df_id['Value']))

            sample = str(id_dict.get('Sample', ''))
            replica = str(id_dict.get('Replica', ''))
            name = f"{sample}_R{replica}" if replica else sample

            excel_info[name] = {
                'file_dad': str(id_dict.get('File_DAD', '')),
                'file_uib': str(id_dict.get('File_DOC_UIB', '')),
                'row_start': int(id_dict.get('DOC_Row_Start', 0) or 0),
                'row_end': int(id_dict.get('DOC_Row_End', 0) or 0),
                'npts': int(id_dict.get('DOC_N_Points', id_dict.get('DAD_N_Points', 0)) or 0),
            }
        except Exception as e:
            if verbose:
                print(f"  Warning: Error llegint {os.path.basename(f)}: {e}")

    # Actualitzar samples al JSON
    samples = summary.get('samples', [])
    updated_count = 0

    for s in samples:
        name = s.get('name', '')
        if name in excel_info:
            info = excel_info[name]
            # Afegir camps nous (només si no existeixen o estan buits)
            if not s.get('file_dad'):
                s['file_dad'] = info['file_dad']
            if not s.get('file_uib'):
                s['file_uib'] = info['file_uib']
            if not s.get('row_start'):
                s['row_start'] = info['row_start']
            if not s.get('row_end'):
                s['row_end'] = info['row_end']
            if not s.get('npts'):
                s['npts'] = info['npts']
            # Confiança per defecte (consolidats antics = 100%)
            if 'match_confidence' not in s:
                s['match_confidence'] = 100.0
                s['match_type'] = 'legacy'
            updated_count += 1

    # Actualitzar metadata
    if 'meta' not in summary:
        summary['meta'] = {}
    summary['meta']['json_updated_at'] = datetime.now().isoformat()
    summary['meta']['json_update_script'] = 'update_consolidation_json.py'

    # === QUALITY CHECK: Detectar files duplicades i mostres duplicades ===
    row_usage = {}  # {(row_start, row_end): [sample_names]}
    sample_count = {}  # {sample_name: count}
    duplicate_rows = []
    duplicate_samples = []

    for s in samples:
        row_start = s.get('row_start') or 0
        row_end = s.get('row_end') or 0
        sample_name = s.get('name', s.get('mostra', 'unknown'))

        # Comptar mostres duplicades
        sample_count[sample_name] = sample_count.get(sample_name, 0) + 1

        # Files duplicades (només si tenim files vàlides)
        if row_start and row_end and row_start > 0 and row_end > 0 and row_start != row_end:
            key = (row_start, row_end)
            if key not in row_usage:
                row_usage[key] = []
            row_usage[key].append(sample_name)

    # Detectar mostres duplicades
    for sample_name, count in sample_count.items():
        if count > 1:
            duplicate_samples.append({
                "sample": sample_name,
                "count": count
            })

    # Detectar files compartides entre mostres DIFERENTS
    for (row_start, row_end), sample_names in row_usage.items():
        # Només és un problema si hi ha mostres DIFERENTS
        unique_names = set(sample_names)
        if len(unique_names) > 1:
            duplicate_rows.append({
                "rows": f"{row_start}-{row_end}",
                "samples": list(unique_names),
                "count": len(unique_names)
            })

    duplicate_rows.sort(key=lambda x: x["count"], reverse=True)
    duplicate_samples.sort(key=lambda x: x["count"], reverse=True)

    # Actualitzar quality section
    if 'quality' not in summary:
        summary['quality'] = {}
    summary['quality']['duplicate_rows'] = duplicate_rows
    summary['quality']['duplicate_row_count'] = len(duplicate_rows)
    summary['quality']['duplicate_samples'] = duplicate_samples
    summary['quality']['duplicate_sample_count'] = len(duplicate_samples)

    # Quality issues array
    quality_issues = []

    # Mostres duplicades (WARNING - cal revisar blocs B1/B2)
    for dup in duplicate_samples:
        quality_issues.append({
            "type": "DUPLICATE_SAMPLE",
            "severity": "WARNING",
            "message": f"Mostra '{dup['sample']}' apareix {dup['count']} cops (revisar blocs B1/B2)",
            "sample": dup["sample"],
            "count": dup["count"]
        })

    # Files duplicades (ERROR - problema de MasterFile)
    for dup in duplicate_rows:
        quality_issues.append({
            "type": "DUPLICATE_ROWS",
            "severity": "ERROR",
            "message": f"Files {dup['rows']} usades per mostres diferents: {', '.join(dup['samples'])}",
            "samples": dup["samples"],
            "rows": dup["rows"]
        })

    summary['quality_issues'] = quality_issues

    # Afegir errors/warnings
    if 'errors' not in summary:
        summary['errors'] = []
    if 'warnings' not in summary:
        summary['warnings'] = []

    # Eliminar errors/warnings de duplicats anteriors
    summary['errors'] = [e for e in summary['errors'] if 'DUPLICAT' not in e]
    summary['warnings'] = [w for w in summary['warnings'] if 'DUPLICAT' not in w]

    # Afegir nous errors/warnings
    for dup in duplicate_samples:
        summary['warnings'].append(
            f"AVÍS DUPLICAT: Mostra '{dup['sample']}' apareix {dup['count']} cops (revisar blocs B1/B2)"
        )
    for dup in duplicate_rows:
        summary['errors'].append(
            f"ERROR DUPLICAT: Files {dup['rows']} usades per {len(dup['samples'])} mostres: {', '.join(dup['samples'])}"
        )

    # Guardar JSON actualitzat
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return {"success": False, "updated": 0, "message": f"Error guardant JSON: {e}"}

    return {
        "success": True,
        "updated": updated_count,
        "message": f"Actualitzats {updated_count}/{len(samples)} samples"
    }


def find_all_sequences(base_path=None):
    """Troba totes les seqüències amb consolidation.json."""
    if base_path is None:
        # Buscar a Dades2
        base_path = r"C:\Users\Lequia\Desktop\Dades2"

    sequences = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and "_SEQ" in item:
            json_path = os.path.join(item_path, "CHECK", "consolidation.json")
            if os.path.exists(json_path):
                sequences.append(item_path)

    return sorted(sequences)


def main():
    if len(sys.argv) < 2:
        print("Ús: python update_consolidation_json.py <seq_path>")
        print("     python update_consolidation_json.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        # Actualitzar totes les seqüències
        sequences = find_all_sequences()
        print(f"Trobades {len(sequences)} seqüències amb consolidation.json\n")

        total_updated = 0
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            print(f"Processant {seq_name}...", end=" ")
            result = update_json_from_excel(seq_path, verbose=False)
            if result["success"]:
                print(f"OK ({result['updated']} samples)")
                total_updated += result['updated']
            else:
                print(f"ERROR: {result['message']}")

        print(f"\nTotal: {total_updated} samples actualitzats")
    else:
        # Actualitzar una seqüència específica
        seq_path = sys.argv[1]
        if not os.path.isdir(seq_path):
            print(f"Error: No existeix {seq_path}")
            sys.exit(1)

        print(f"Actualitzant {os.path.basename(seq_path)}...")
        result = update_json_from_excel(seq_path)

        if result["success"]:
            print(f"OK: {result['message']}")
        else:
            print(f"ERROR: {result['message']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
