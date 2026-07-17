# -*- coding: utf-8 -*-
"""Neteja de carpetes PER_SAMPLE/ òrfenes (duplicats sense propietari).

Diagnòstic 2026-07-17: cap codi de la Suite actual escriu ni llegeix PER_SAMPLE/
(ni a l'arrel de la SEQ ni dins RESULTATS/). Els fitxers porten un esquema
("csv_companion", "$schema_version") que ja no existeix al codi: són restes d'un
export per-mostra antic, substituït pel paquet FAIR v2 (traces/ + results_SEC.csv
+ datapackage.json). Duplicaven ~1 MB per SEQ sense que res els consumís.

Seguretat: només s'esborra una carpeta PER_SAMPLE si TOT el seu contingut són
fitxers .json/.csv (cap subcarpeta, cap altra extensió). Si hi ha res més, es
salta i s'avisa.

Ús:
    python -X utf8 neteja_per_sample.py            # simulacre (informe, no esborra)
    python -X utf8 neteja_per_sample.py --aplicar  # esborra de veritat
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpsec_config import get_data_folders


def _es_nomes_json_csv(folder):
    """True si la carpeta només conté fitxers .json/.csv (sense subcarpetes)."""
    for entry in os.scandir(folder):
        if entry.is_dir():
            return False
        if not entry.name.lower().endswith((".json", ".csv")):
            return False
    return True


def _mida_kb(folder):
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024.0


def main():
    aplicar = "--aplicar" in sys.argv
    candidates = []

    for base in get_data_folders():
        if not os.path.isdir(base):
            continue
        for seq in sorted(os.listdir(base)):
            seq_path = os.path.join(base, seq)
            if not os.path.isdir(seq_path):
                continue
            for rel in ("PER_SAMPLE", os.path.join("RESULTATS", "PER_SAMPLE")):
                p = os.path.join(seq_path, rel)
                if os.path.isdir(p):
                    candidates.append((seq, rel, p))

    if not candidates:
        print("Cap carpeta PER_SAMPLE trobada. Res a fer.")
        return

    total_kb = 0.0
    esborrades = 0
    for seq, rel, p in candidates:
        kb = _mida_kb(p)
        n = sum(len(files) for _r, _d, files in os.walk(p))
        if not _es_nomes_json_csv(p):
            print(f"  SALTADA {seq}/{rel} — conté coses que no són .json/.csv, revisar a mà")
            continue
        total_kb += kb
        if aplicar:
            shutil.rmtree(p)
            esborrades += 1
            print(f"  ESBORRADA {seq}/{rel} ({n} fitxers, {kb:.0f} KB)")
        else:
            print(f"  candidata {seq}/{rel} ({n} fitxers, {kb:.0f} KB)")

    print()
    if aplicar:
        print(f"Fet: {esborrades} carpetes esborrades, {total_kb/1024:.1f} MB alliberats.")
    else:
        print(f"SIMULACRE: {len(candidates)} carpetes, {total_kb/1024:.1f} MB en total.")
        print("Per esborrar de veritat:  python -X utf8 neteja_per_sample.py --aplicar")


if __name__ == "__main__":
    main()
