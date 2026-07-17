# -*- coding: utf-8 -*-
"""Migració: seq_date de KHP_History = data d'ADQUISICIÓ, no la de processament.

Fins al 2026-07-17, `register_calibration()` no rebia mai `seq_date` al khp_data i
queia a `datetime.now()` en silenci → totes les entrades de l'historial portaven la
data en què es van PROCESSAR, no la que es van córrer a l'instrument. Amb això,
qualsevol selecció per data (calibració vigent per règim, "les N més recents" de les
comparatives, evolució temporal) operava sobre una data sense significat.

Aquest script recupera la data real del manifest de cada SEQ
(`import_manifest.json` → `sequence.date`) i reescriu `seq_date` i `date`.
La data de processament es conserva a `date_processed` (i s'hi copia la vella si
l'entrada no en tenia, per no perdre-la).

Ús:
    python -X utf8 migra_seq_date_adquisicio.py            # simulacre (no escriu)
    python -X utf8 migra_seq_date_adquisicio.py --aplicar  # escriu (fa còpia .bak)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpsec_calibrate import get_seq_acquisition_date, get_history_path
from hpsec_config import get_data_folders


def troba_seq(seq_name):
    """Localitza la carpeta d'una SEQ a qualsevol de les carpetes de dades."""
    for base in get_data_folders():
        p = os.path.join(base, seq_name)
        if os.path.isdir(p):
            return p
    return None


def main():
    aplicar = "--aplicar" in sys.argv

    seq_ref = None
    for base in get_data_folders():
        if os.path.isdir(base):
            for d in os.listdir(base):
                if d.endswith("_SEQ") or "_SEQ" in d:
                    seq_ref = os.path.join(base, d)
                    break
        if seq_ref:
            break
    hist_path = get_history_path(seq_ref) if seq_ref else None
    if not hist_path or not os.path.exists(hist_path):
        print("No s'ha trobat KHP_History.json"); return 1

    print("Historial:", hist_path)
    with open(hist_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("calibrations") or []
    print(f"Entrades: {len(entries)}\n")

    hdr = f"{'seq_name':<20} {'seq_date ara':<12} {'adquisicio':<12} {'accio':<28}"
    print(hdr); print("-" * len(hdr))

    cache, n_fix, n_ok, n_sense = {}, 0, 0, 0
    for e in entries:
        name = e.get("seq_name") or ""
        if name not in cache:
            p = troba_seq(name)
            cache[name] = get_seq_acquisition_date(p) if p else None
        real = cache[name]
        ara = str(e.get("seq_date") or "")[:10]

        if not real:
            accio = "SENSE MANIFEST — es deixa igual"
            n_sense += 1
        elif ara == real:
            accio = "ja correcta"
            n_ok += 1
        else:
            accio = f"corregeix {ara} -> {real}"
            n_fix += 1
            if aplicar:
                # Conservar la data de processament abans de sobreescriure
                if not e.get("date_processed") and ara:
                    e["date_processed"] = e.get("seq_date")
                e["seq_date"] = real
                e["date"] = real
        print(f"{name[:20]:<20} {ara:<12} {str(real or '—'):<12} {accio:<28}")

    print(f"\nResum: {n_fix} a corregir · {n_ok} ja correctes · {n_sense} sense manifest")

    if not aplicar:
        print("\nSIMULACRE — no s'ha escrit res. Torna-hi amb --aplicar per desar.")
        return 0

    bak = hist_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(hist_path, bak)
    data["updated"] = datetime.now().isoformat()
    tmp = hist_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, hist_path)
    print(f"\nDesat. Còpia de seguretat: {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
