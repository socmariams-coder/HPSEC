"""
Anàlisi àrees MQ (Milli-Q) per COLUMN i BP.
MQ és aigua pura → DOC ≈ 0. Qualsevol àrea és fons/soroll.
"""

import json
import os
import re
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"

RF_COL = 752.90; RF_BP = 646.82
IC_COL = 18.11; IC_BP = 2.87
VOL_COL = 400.0; VOL_BP = 100.0


def seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    mq_col = []
    mq_bp = []

    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        method = data.get("method", "?")
        if method not in ("COLUMN", "BP"):
            continue

        sn = seq_num(d)

        for s in data.get("samples", []):
            name = s.get("name", "")
            if not name.upper().startswith("MQ"):
                continue

            area_total = s.get("areas", {}).get("DOC", {}).get("total", 0)
            pi_area = s.get("peak_info", {}).get("area", 0)
            bl = s.get("peak_info", {}).get("baseline_level", 0)
            snr = s.get("snr_info", {}).get("snr_direct", 0)

            fracs = {}
            if method == "COLUMN":
                for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
                    fracs[fn] = s.get("areas", {}).get("DOC", {}).get(fn, 0)

            vol = VOL_COL if method == "COLUMN" else VOL_BP
            ppm = (area_total - (IC_COL if method == "COLUMN" else IC_BP)) * 1000 / (
                (RF_COL if method == "COLUMN" else RF_BP) * vol)

            entry = {
                "seq": d, "name": name, "method": method,
                "area_total": area_total, "pi_area": pi_area,
                "baseline": bl, "snr": snr, "ppm": ppm,
                "fracs": fracs, "seq_num": sn,
            }

            if method == "COLUMN":
                mq_col.append(entry)
            else:
                mq_bp.append(entry)

    print("=" * 70)
    print("ÀREES MQ (MILLI-Q) — COLUMN vs BP")
    print("=" * 70)

    print(f"\n  MQ COLUMN: {len(mq_col)} entrades")
    print(f"  MQ BP:     {len(mq_bp)} entrades")

    # COLUMN
    if mq_col:
        areas = [e["area_total"] for e in mq_col]
        ppms = [e["ppm"] for e in mq_col]
        print(f"\n  COLUMN MQ:")
        print(f"    Àrea total:  mediana={np.median(areas):.1f}  "
              f"mitjana={np.mean(areas):.1f} ± {np.std(areas):.1f}")
        print(f"    ppm:         mediana={np.median(ppms):.3f}  "
              f"mitjana={np.mean(ppms):.3f} ± {np.std(ppms):.3f}")
        print(f"    Rang àrea:   [{min(areas):.1f}, {max(areas):.1f}]")
        print(f"    Rang ppm:    [{min(ppms):.3f}, {max(ppms):.3f}]")

        # Per fracció
        print(f"\n    Per fracció (mediana àrea):")
        for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
            vals = [e["fracs"].get(fn, 0) for e in mq_col]
            ppm_f = np.median(vals) * 1000 / (RF_COL * VOL_COL)
            print(f"      {fn:5s}: àrea={np.median(vals):7.1f}  (≈{ppm_f:.3f} ppm)")

    # BP
    if mq_bp:
        areas = [e["area_total"] for e in mq_bp]
        ppms = [e["ppm"] for e in mq_bp]
        print(f"\n  BP MQ:")
        print(f"    Àrea total:  mediana={np.median(areas):.1f}  "
              f"mitjana={np.mean(areas):.1f} ± {np.std(areas):.1f}")
        print(f"    ppm:         mediana={np.median(ppms):.3f}  "
              f"mitjana={np.mean(ppms):.3f} ± {np.std(ppms):.3f}")
        print(f"    Rang àrea:   [{min(areas):.1f}, {max(areas):.1f}]")
        print(f"    Rang ppm:    [{min(ppms):.3f}, {max(ppms):.3f}]")

    # Comparació directa
    if mq_col and mq_bp:
        med_col = np.median([e["area_total"] for e in mq_col])
        med_bp = np.median([e["area_total"] for e in mq_bp])
        ppm_col = np.median([e["ppm"] for e in mq_col])
        ppm_bp = np.median([e["ppm"] for e in mq_bp])
        print(f"\n  COMPARACIÓ:")
        print(f"    Àrea MQ COLUMN: {med_col:.1f}  ({ppm_col:.3f} ppm)")
        print(f"    Àrea MQ BP:     {med_bp:.1f}  ({ppm_bp:.3f} ppm)")
        print(f"    Ratio BP/COL àrea: {med_bp / med_col:.2f}" if med_col > 0 else "")
        print(f"    Δ ppm BP-COL:   {ppm_bp - ppm_col:+.3f} ppm")

    # Detall per SEQ
    print(f"\n{'='*70}")
    print(f"DETALL PER SEQÜÈNCIA")
    print(f"{'='*70}")

    print(f"\n  COLUMN MQ (ordenat per àrea):")
    for e in sorted(mq_col, key=lambda x: x["area_total"]):
        print(f"    {e['seq']:20s}  {e['name']:10s}  àrea={e['area_total']:7.1f}  "
              f"ppm={e['ppm']:6.3f}  bl={e['baseline']:6.1f}")

    print(f"\n  BP MQ (ordenat per àrea):")
    for e in sorted(mq_bp, key=lambda x: x["area_total"]):
        print(f"    {e['seq']:20s}  {e['name']:10s}  àrea={e['area_total']:7.1f}  "
              f"ppm={e['ppm']:6.3f}  bl={e['baseline']:6.1f}")


if __name__ == "__main__":
    main()
