"""
Verificar si els KHP de les SEQs problemàtiques donen l'àrea esperada per 400µL.

Si RF_COL=752.9, un KHP de X ppm a 400µL hauria de donar:
  area_esperada = RF * vol * ppm / 1000 + IC = 752.9 * 400 * X / 1000 + 18.2

Si l'àrea real és la meitat, el volum real és ~200µL (no 400).
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

RF_COL = 752.90; IC_COL = 18.11
RF_BP = 646.82; IC_BP = 2.87


def seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    # Carregar totes les SEQs amb KHP
    print("=" * 90)
    print("KHP: ÀREA REAL vs ÀREA ESPERADA (400µL)")
    print("=" * 90)

    seqs_interest = [
        # Problemàtiques (ratio ~2x)
        "272_SEQ", "274_SEQ",
        # OK (veïnes)
        "275_SEQ", "276B_SEQ", "288_SEQ",
        # Antigues problemàtiques
        "073_SEQ", "074_SEQ", "075_SEQ",
    ]

    # També buscar totes les COLUMN amb KHP per context
    all_khp = []

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
            if not name.upper().startswith("KHP"):
                continue

            area = s.get("areas", {}).get("DOC", {}).get("total", 0)
            pi_area = s.get("peak_info", {}).get("area", 0)
            vol = s.get("injection_volume")

            # Extreure concentració
            conc = 0
            m = re.search(r"(\d+\.?\d*)", name)
            if m:
                conc = float(m.group(1))

            if conc <= 0 or area <= 0:
                continue

            # Àrea esperada a 400µL (COLUMN) o 100µL (BP)
            if method == "COLUMN":
                vol_assumed = 400
                area_exp = RF_COL * vol_assumed * conc / 1000 + IC_COL
                rf = RF_COL
                ic = IC_COL
            else:
                vol_assumed = 100
                area_exp = RF_BP * vol_assumed * conc / 1000 + IC_BP
                rf = RF_BP
                ic = IC_BP

            # Volum estimat des de l'àrea real
            vol_estimated = (area - ic) * 1000 / (rf * conc) if conc > 0 and rf > 0 else 0

            ratio = area / area_exp if area_exp > 0 else 0

            entry = {
                "seq": d, "name": name, "method": method,
                "conc": conc, "area": area, "pi_area": pi_area,
                "area_exp": area_exp, "ratio": ratio,
                "vol_assumed": vol_assumed, "vol_estimated": vol_estimated,
                "vol_manifest": vol,
                "seq_num": sn,
                "is_interest": d in seqs_interest,
            }
            all_khp.append(entry)

    print(f"\nTotal KHP trobats: {len(all_khp)}")

    # ================================================================
    # 1. SEQs d'interès
    # ================================================================
    print(f"\n{'='*90}")
    print(f"1. SEQs D'INTERÈS — KHP àrea real vs esperada")
    print(f"{'='*90}")

    for seq_name in seqs_interest:
        entries = [e for e in all_khp if e["seq"] == seq_name]
        if not entries:
            print(f"\n  {seq_name}: CAP KHP")
            continue

        print(f"\n  {seq_name} [{entries[0]['method']}]:")
        print(f"    {'KHP':>12s}  {'conc':>5s}  {'àrea':>7s}  {'esperada':>8s}  "
              f"{'ratio':>6s}  {'vol_est':>7s}  {'vol_manif':>9s}")
        for e in sorted(entries, key=lambda x: x["conc"]):
            vm = str(e["vol_manifest"]) if e["vol_manifest"] else "?"
            print(f"    {e['name']:>12s}  {e['conc']:5.1f}  {e['area']:7.1f}  "
                  f"{e['area_exp']:8.1f}  {e['ratio']:6.3f}  {e['vol_estimated']:7.0f}µL  "
                  f"{vm:>9s}")

    # ================================================================
    # 2. TOTES LES COLUMN — distribució vol_estimat
    # ================================================================
    print(f"\n{'='*90}")
    print(f"2. TOTES LES COLUMN — VOLUM ESTIMAT DES DE KHP")
    print(f"{'='*90}")

    col_khp = [e for e in all_khp if e["method"] == "COLUMN" and e["vol_estimated"] > 0]
    if col_khp:
        vols = [e["vol_estimated"] for e in col_khp]
        print(f"\n  N = {len(col_khp)} KHP COLUMN")
        print(f"  Vol estimat mediana: {np.median(vols):.0f} µL")
        print(f"  Vol estimat mitjana: {np.mean(vols):.0f} ± {np.std(vols):.0f} µL")

        # Agrupar per volum estimat
        v_groups = {"~100µL": [], "~200µL": [], "~400µL": [], "Altre": []}
        for e in col_khp:
            v = e["vol_estimated"]
            if 50 < v < 150:
                v_groups["~100µL"].append(e)
            elif 150 < v < 300:
                v_groups["~200µL"].append(e)
            elif 300 < v < 600:
                v_groups["~400µL"].append(e)
            else:
                v_groups["Altre"].append(e)

        print(f"\n  Distribució:")
        for label, entries in v_groups.items():
            if entries:
                seqs = sorted(set(e["seq"] for e in entries))
                print(f"    {label}: {len(entries)} KHP de {len(seqs)} SEQs")
                if label != "~400µL":
                    for seq in seqs:
                        seq_entries = [e for e in entries if e["seq"] == seq]
                        vols_seq = [e["vol_estimated"] for e in seq_entries]
                        print(f"      {seq:20s}  vol_est={np.median(vols_seq):.0f}µL  "
                              f"(n={len(seq_entries)})")

    # ================================================================
    # 3. BP — distribució vol_estimat
    # ================================================================
    print(f"\n{'='*90}")
    print(f"3. TOTES LES BP — VOLUM ESTIMAT DES DE KHP")
    print(f"{'='*90}")

    bp_khp = [e for e in all_khp if e["method"] == "BP" and e["vol_estimated"] > 0]
    if bp_khp:
        vols = [e["vol_estimated"] for e in bp_khp]
        print(f"\n  N = {len(bp_khp)} KHP BP")
        print(f"  Vol estimat mediana: {np.median(vols):.0f} µL")
        print(f"  Vol estimat mitjana: {np.mean(vols):.0f} ± {np.std(vols):.0f} µL")

        v_groups = {"~50µL": [], "~100µL": [], "~200µL": [], "Altre": []}
        for e in bp_khp:
            v = e["vol_estimated"]
            if 30 < v < 75:
                v_groups["~50µL"].append(e)
            elif 75 < v < 150:
                v_groups["~100µL"].append(e)
            elif 150 < v < 300:
                v_groups["~200µL"].append(e)
            else:
                v_groups["Altre"].append(e)

        print(f"\n  Distribució:")
        for label, entries in v_groups.items():
            if entries:
                seqs = sorted(set(e["seq"] for e in entries))
                print(f"    {label}: {len(entries)} KHP de {len(seqs)} SEQs")
                if label != "~100µL":
                    for seq in seqs:
                        seq_entries = [e for e in entries if e["seq"] == seq]
                        vols_seq = [e["vol_estimated"] for e in seq_entries]
                        print(f"      {seq:20s}  vol_est={np.median(vols_seq):.0f}µL  "
                              f"(n={len(seq_entries)})")


if __name__ == "__main__":
    main()
