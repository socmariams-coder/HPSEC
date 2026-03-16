"""
Batch analysis de parells COLUMN/BP per comparació.

Usa exactament el mateix codi que la Suite:
  import_from_manifest → ensure_data_loaded → calibrate_from_import → analyze_sequence → save_analysis_result

Només analitza SEQs que:
  1. Tenen import_manifest.json (ja importades)
  2. NO tenen analysis_result.json (no re-analitza)
  3. Formen part d'un parell COLUMN/BP amb ≥3 mostres compartides
"""

import json
import os
import sys
import time
import traceback

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Afegir root al path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hpsec_import import import_from_manifest, ensure_data_loaded
from hpsec_calibrate import calibrate_from_import
from hpsec_analyze import analyze_sequence, save_analysis_result
from hpsec_config import load_config

DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"
MIN_SHARED = 3  # mínim mostres compartides per considerar un parell


def find_all_seqs():
    """Trobar totes les SEQs importades amb els seus noms de mostra."""
    seqs = {}
    for d in sorted(os.listdir(DATA_DIR)):
        manifest_path = os.path.join(DATA_DIR, d, "CHECK", "data", "import_manifest.json")
        if not os.path.isfile(manifest_path):
            continue

        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)

        method = m.get("sequence", {}).get("method", "?")
        has_analysis = os.path.isfile(
            os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        )

        # Extreure noms de mostra (excloure blancs/KHP)
        samples_raw = m.get("samples", [])
        names = set()
        if isinstance(samples_raw, list):
            for s in samples_raw:
                n = s.get("name", "") if isinstance(s, dict) else ""
                if n and not n.upper().startswith(("MQ", "NAOH", "BUFFER", "KHP")):
                    names.add(n)
        elif isinstance(samples_raw, dict):
            for k, v in samples_raw.items():
                n = v.get("name", k) if isinstance(v, dict) else k
                if n and not n.upper().startswith(("MQ", "NAOH", "BUFFER", "KHP")):
                    names.add(n)

        seqs[d] = {
            "method": method,
            "names": names,
            "has_analysis": has_analysis,
            "path": os.path.join(DATA_DIR, d),
        }
    return seqs


def find_pairs(seqs):
    """Trobar parells COLUMN/BP amb ≥MIN_SHARED mostres compartides."""
    col = {k: v for k, v in seqs.items()
           if v["method"] == "COLUMN" and "_CAL" not in k.upper()}
    bp = {k: v for k, v in seqs.items()
          if v["method"] == "BP" and "_CAL" not in k.upper()}

    pairs = []
    for ck in sorted(col.keys()):
        cv = col[ck]
        for bk in sorted(bp.keys()):
            bv = bp[bk]
            shared = cv["names"] & bv["names"]
            if len(shared) >= MIN_SHARED:
                pairs.append((ck, bk, sorted(shared)))
    return pairs


def analyze_seq(seq_name, seq_path, config):
    """Pipeline completa: import → calibrate → analyze → save."""
    t0 = time.time()

    # 1. Import from manifest
    manifest_path = os.path.join(seq_path, "CHECK", "data", "import_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    imported = import_from_manifest(seq_path, manifest=manifest, config=config, load_data=False)
    if not imported.get("success"):
        return False, f"Import failed: {imported.get('errors', [])}"

    # 2. Ensure data loaded
    if imported.get("data_deferred"):
        ensure_data_loaded(imported, config=config)

    # 3. Calibrate (KHP verification)
    cal_data = calibrate_from_import(imported, config=config)
    # Cal data pot fallar si no hi ha KHP — no és blocker per l'anàlisi

    # 4. Analyze
    analysis = analyze_sequence(imported, cal_data, config=config)
    if not analysis.get("success"):
        return False, f"Analysis failed: {analysis.get('errors', [])}"

    # 5. Save
    save_analysis_result(analysis, output_path=None)

    dt = time.time() - t0
    n_samples = analysis.get("summary", {}).get("total_samples", 0)
    method = analysis.get("method", "?")
    return True, f"OK ({method}, {n_samples} mostres, {dt:.1f}s)"


def main():
    config = load_config()
    seqs = find_all_seqs()
    pairs = find_pairs(seqs)

    # Recopilar SEQs que necessiten anàlisi
    to_analyze = set()
    for ck, bk, shared in pairs:
        if not seqs[ck]["has_analysis"]:
            to_analyze.add(ck)
        if not seqs[bk]["has_analysis"]:
            to_analyze.add(bk)

    print(f"Parells COLUMN/BP trobats: {len(pairs)}")
    print(f"SEQs a analitzar: {len(to_analyze)} (de {len(seqs)} importades)")

    if not to_analyze:
        print("Totes les SEQs ja estan analitzades!")
        return

    # Llistar
    print(f"\nSEQs pendents d'anàlisi:")
    for s in sorted(to_analyze):
        method = seqs[s]["method"]
        print(f"  {s:25s} [{method}]")

    print(f"\n{'='*60}")
    print(f"INICIANT BATCH ANALYSIS")
    print(f"{'='*60}\n")

    ok_count = 0
    fail_count = 0
    for i, seq_name in enumerate(sorted(to_analyze), 1):
        seq_path = seqs[seq_name]["path"]
        print(f"[{i}/{len(to_analyze)}] {seq_name}...", end=" ", flush=True)

        try:
            success, msg = analyze_seq(seq_name, seq_path, config)
            if success:
                print(msg)
                ok_count += 1
            else:
                print(f"FAIL: {msg}")
                fail_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETAT: {ok_count} OK, {fail_count} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
