"""
Diagnòstic detallat dels parells 272/271 i 274/273.
Ratio ~2x a baixa concentració (1-2 ppm).

Mirem:
- Volums d'injecció
- Àrees crues vs ppm
- Fraccions COLUMN
- Cromatogrames MQ d'aquelles SEQs
- Comparar amb SEQs veïnes que funcionen bé (275/273, 276B/277)
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
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "diag_272_274")

RF_COL = 752.90; RF_BP = 646.82
IC_COL = 18.11; IC_BP = 2.87
VOL_COL = 400.0; VOL_BP = 100.0

SEQS_INTEREST = [
    # Problemàtiques
    "272_SEQ", "271_SEQ_BP",
    "274_SEQ", "273_SEQ_BP",
    # OK (veïnes per comparar)
    "275_SEQ", "276B_SEQ", "276_SEQ", "277_SEQ_BP",
    "288_SEQ", "287_SEQ_BP",
]


def load_seq_detail(seq_name):
    """Carregar totes les dades d'anàlisi d'una SEQ."""
    f = os.path.join(DATA_DIR, seq_name, "CHECK", "data", "analysis_result.json")
    if not os.path.isfile(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(seq_name):
    """Carregar manifest per veure volums."""
    f = os.path.join(DATA_DIR, seq_name, "CHECK", "data", "import_manifest.json")
    if not os.path.isfile(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ================================================================
    # 1. INFO BÀSICA DE CADA SEQ
    # ================================================================
    print("=" * 80)
    print("1. INFO BÀSICA DE LES SEQÜÈNCIES D'INTERÈS")
    print("=" * 80)

    for seq_name in SEQS_INTEREST:
        data = load_seq_detail(seq_name)
        manifest = load_manifest(seq_name)
        if not data:
            print(f"\n  {seq_name}: NO TROBAT")
            continue

        method = data.get("method", "?")
        n_samples = len([s for s in data.get("samples", [])
                         if not s.get("name", "").upper().startswith(("MQ", "NAOH", "BUFFER", "KHP"))])
        n_mq = len([s for s in data.get("samples", [])
                     if s.get("name", "").upper().startswith("MQ")])

        # Volums del manifest
        vol_info = "?"
        if manifest:
            samples_m = manifest.get("samples", [])
            vols = set()
            if isinstance(samples_m, list):
                for sm in samples_m:
                    v = sm.get("injection_volume")
                    if v:
                        vols.add(v)
            elif isinstance(samples_m, dict):
                for k, sm in samples_m.items():
                    v = sm.get("injection_volume") if isinstance(sm, dict) else None
                    if v:
                        vols.add(v)
            vol_info = ", ".join(str(v) for v in sorted(vols)) if vols else "NO DEFINIT"

        print(f"\n  {seq_name:20s}  [{method}]  {n_samples} mostres, {n_mq} MQ  vol={vol_info}")

        # MQ àrees
        for s in data.get("samples", []):
            if s.get("name", "").upper().startswith("MQ"):
                area = s.get("areas", {}).get("DOC", {}).get("total", 0)
                vol = s.get("injection_volume", "?")
                print(f"    MQ: {s['name']:10s}  àrea={area:7.1f}  vol={vol}")

    # ================================================================
    # 2. DETALL MOSTRES COMPARTIDES 272 vs 271_BP
    # ================================================================
    for col_seq, bp_seq in [("272_SEQ", "271_SEQ_BP"),
                             ("274_SEQ", "273_SEQ_BP"),
                             ("275_SEQ", "273_SEQ_BP"),
                             ("288_SEQ", "287_SEQ_BP")]:
        print(f"\n{'='*80}")
        print(f"2. DETALL MOSTRES: {col_seq} vs {bp_seq}")
        print(f"{'='*80}")

        col_data = load_seq_detail(col_seq)
        bp_data = load_seq_detail(bp_seq)
        if not col_data or not bp_data:
            print(f"  FALTEN DADES")
            continue

        # Indexar per nom
        col_by = {}
        for s in col_data.get("samples", []):
            n = s.get("name", "")
            if not n.upper().startswith(("MQ", "NAOH", "BUFFER", "KHP")):
                col_by.setdefault(n, []).append(s)
        bp_by = {}
        for s in bp_data.get("samples", []):
            n = s.get("name", "")
            if not n.upper().startswith(("MQ", "NAOH", "BUFFER", "KHP")):
                bp_by.setdefault(n, []).append(s)

        shared = sorted(set(col_by) & set(bp_by))
        print(f"  Mostres compartides: {len(shared)}")

        if not shared:
            continue

        print(f"\n  {'Mostra':>12s}  {'vol_C':>6s}  {'area_C':>7s}  {'ppm_C':>6s}  "
              f"{'vol_B':>6s}  {'area_B':>7s}  {'ppm_B':>6s}  {'ratio':>6s}  "
              f"{'BioP':>5s}  {'HS':>5s}  {'BB':>5s}  {'SB':>5s}  {'LMW':>5s}")
        print("  " + "-" * 120)

        for name in shared:
            cr = col_by[name]
            br = bp_by[name]

            # Mitjana rèpliques
            def avg(reps, *keys):
                vals = []
                for s in reps:
                    v = s
                    for k in keys:
                        v = v.get(k, {}) if isinstance(v, dict) else {}
                    if isinstance(v, (int, float)):
                        vals.append(v)
                return np.mean(vals) if vals else 0

            c_area = avg(cr, "areas", "DOC", "total")
            b_area = avg(br, "areas", "DOC", "total")
            c_vol = cr[0].get("injection_volume", VOL_COL)
            b_vol = br[0].get("injection_volume", VOL_BP)

            c_ppm = (c_area - IC_COL) * 1000 / (RF_COL * (c_vol or VOL_COL))
            b_ppm = (b_area - IC_BP) * 1000 / (RF_BP * (b_vol or VOL_BP))

            ratio = b_ppm / c_ppm if c_ppm > 0 else 0

            # Fraccions
            fracs = {}
            for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
                fracs[fn] = avg(cr, "areas", "DOC", fn)

            print(f"  {name:>12s}  {c_vol:6.0f}  {c_area:7.1f}  {c_ppm:6.2f}  "
                  f"{b_vol:6.0f}  {b_area:7.1f}  {b_ppm:6.2f}  {ratio:6.2f}  "
                  + "  ".join(f"{fracs[fn]:5.0f}" for fn in ["BioP", "HS", "BB", "SB", "LMW"]))

        # Comparació àrees normalitzades per volum
        print(f"\n  Comparació àrees/vol:")
        for name in shared:
            cr = col_by[name]
            br = bp_by[name]

            def avg(reps, *keys):
                vals = []
                for s in reps:
                    v = s
                    for k in keys:
                        v = v.get(k, {}) if isinstance(v, dict) else {}
                    if isinstance(v, (int, float)):
                        vals.append(v)
                return np.mean(vals) if vals else 0

            c_area = avg(cr, "areas", "DOC", "total")
            b_area = avg(br, "areas", "DOC", "total")
            c_vol = cr[0].get("injection_volume", VOL_COL)
            b_vol = br[0].get("injection_volume", VOL_BP)

            c_apv = c_area / (c_vol or VOL_COL)
            b_apv = b_area / (b_vol or VOL_BP)
            ratio_apv = b_apv / c_apv if c_apv > 0 else 0

            print(f"    {name:>12s}  COL={c_apv:.3f}/µL  BP={b_apv:.3f}/µL  "
                  f"ratio={ratio_apv:.3f}")

    # ================================================================
    # 3. ANOMALIES REPORTADES
    # ================================================================
    print(f"\n{'='*80}")
    print(f"3. ANOMALIES PER SEQ")
    print(f"{'='*80}")

    for seq_name in ["272_SEQ", "271_SEQ_BP", "274_SEQ", "273_SEQ_BP"]:
        data = load_seq_detail(seq_name)
        if not data:
            continue
        print(f"\n  {seq_name}:")
        for s in data.get("samples", []):
            name = s.get("name", "")
            anoms = s.get("anomalies", [])
            if anoms:
                codes = [a.get("code", "?") for a in anoms]
                print(f"    {name:>12s}: {', '.join(codes)}")

    # ================================================================
    # 4. COMPARAR VOLUMS MANIFEST vs ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print(f"4. VOLUMS: MANIFEST vs ANALYSIS_RESULT")
    print(f"{'='*80}")

    for seq_name in ["272_SEQ", "271_SEQ_BP", "274_SEQ", "273_SEQ_BP",
                      "275_SEQ", "288_SEQ", "287_SEQ_BP"]:
        data = load_seq_detail(seq_name)
        manifest = load_manifest(seq_name)
        if not data:
            continue

        method = data.get("method", "?")
        print(f"\n  {seq_name} [{method}]:")

        # Volums de l'analysis_result
        vols_ar = set()
        for s in data.get("samples", []):
            v = s.get("injection_volume")
            if v:
                vols_ar.add(v)

        # Volums del manifest
        vols_m = set()
        vol_source = "?"
        if manifest:
            for sm in manifest.get("samples", []):
                if isinstance(sm, dict):
                    v = sm.get("injection_volume")
                    vs = sm.get("inj_volume_source", "?")
                    if v:
                        vols_m.add(v)
                    if vs != "?":
                        vol_source = vs

        print(f"    Volums analysis: {sorted(vols_ar)}")
        print(f"    Volums manifest: {sorted(vols_m)}")
        print(f"    Font volum:      {vol_source}")


if __name__ == "__main__":
    main()
