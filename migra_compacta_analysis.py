# -*- coding: utf-8 -*-
"""Migració: aprima analysis_result.json antics (duplicats + indentació).

Fins al 2026-07-17, el Quantificar reescrivia analysis_result.json amb el dict
cru en memòria, re-inflant el fitxer (~23 MB a la 303): la llista plana
'samples' tornava a portar els arrays de senyal que ja són a samples_grouped
(el writer canònic els treia des del febrer, a593d2e, però el Quantificar se'l
saltava). A més, tots els writers desaven amb indent=2 (+40%).

Aquest script reescriu cada analysis_result.json:
  - treu els camps d'array de la llista plana 'samples'
    (ANALYSIS_FLAT_ARRAY_KEYS de hpsec_analyze — font única)
  - desa compacte (sense indentació), atòmic, amb còpia .bak

SENSE pèrdua d'informació: els senyals segueixen sencers a samples_grouped,
i totes les decisions manuals (selecció de rèpliques, reparacions,
quantificació) queden intactes.

Ús:
    python -X utf8 migra_compacta_analysis.py            # simulacre (no escriu)
    python -X utf8 migra_compacta_analysis.py --aplicar  # escriu (fa còpia .bak)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpsec_analyze import ANALYSIS_FLAT_ARRAY_KEYS, strip_flat_sample_arrays
from hpsec_config import get_data_folders
from hpsec_utils import _atomic_write_json


def main():
    aplicar = "--aplicar" in sys.argv
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_abans = total_despres = 0
    n = 0

    for base in get_data_folders():
        if not os.path.isdir(base):
            continue
        for seq in sorted(os.listdir(base)):
            p = os.path.join(base, seq, "CHECK", "data", "analysis_result.json")
            if not os.path.isfile(p):
                continue
            abans = os.path.getsize(p)
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  SALTADA {seq}: JSON illegible ({e})")
                continue

            te_arrays = any(
                isinstance(s, dict) and any(k in s for k in ANALYSIS_FLAT_ARRAY_KEYS)
                for s in data.get("samples", [])
            )
            slim = strip_flat_sample_arrays(data)
            despres_estimat = len(json.dumps(slim, ensure_ascii=False).encode("utf-8"))

            total_abans += abans
            total_despres += despres_estimat
            n += 1
            marca = "arrays duplicats + indent" if te_arrays else "només indent"
            print(f"  {seq}: {abans/1e6:.1f} -> {despres_estimat/1e6:.1f} MB ({marca})")

            if aplicar:
                shutil.copy2(p, p + f".bak_{stamp}")
                _atomic_write_json(p, slim, indent=None, ensure_ascii=False)

    print()
    if n == 0:
        print("Cap analysis_result.json trobat.")
        return
    estalvi = (total_abans - total_despres) / 1e6
    if aplicar:
        print(f"Fet: {n} fitxers, {total_abans/1e6:.0f} -> {total_despres/1e6:.0f} MB "
              f"({estalvi:.0f} MB alliberats). Còpies .bak_{stamp} al costat de cada fitxer.")
        print("Quan hagis comprovat que el dashboard i Analitzar carreguen bé, "
              "esborra els .bak.")
    else:
        print(f"SIMULACRE: {n} fitxers, {total_abans/1e6:.0f} -> {total_despres/1e6:.0f} MB "
              f"(estalvi {estalvi:.0f} MB).")
        print("Per aplicar:  python -X utf8 migra_compacta_analysis.py --aplicar")


if __name__ == "__main__":
    main()
