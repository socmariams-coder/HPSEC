"""
Diagnostic script: COL vs BP comparison
========================================
Compara totes les parelles COL/BP que tenen analisi completada.
Genera CSV + scatter plots (DOC ppm + A254).

Usage:
    python diag_col_bp.py                   # totes les parelles
    python diag_col_bp.py --range 3         # proximitat ±3 (default)
    python diag_col_bp.py --seq 288         # una SEQ concreta
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path

# Project imports
from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_analysis(seq_path):
    """Load analysis_result.json for a SEQ, return dict or None."""
    json_path = os.path.join(seq_path, "CHECK", "data", "analysis_result.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_manifest_volume(seq_path):
    """Load typical injection volume from import_manifest.json."""
    manifest_path = os.path.join(seq_path, "CHECK", "data", "import_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Try sequence-level volume
        seq_info = manifest.get("sequence", {})
        vol = seq_info.get("injection_volume")
        if vol:
            return vol
        # Try from samples
        samples = manifest.get("samples", [])
        if isinstance(samples, list):
            for s in samples:
                reps = s.get("replicas", [])
                if isinstance(reps, list):
                    for r in reps:
                        inj = r.get("injection", {})
                        v = inj.get("inj_volume")
                        if v:
                            return v
                elif isinstance(reps, dict):
                    for rk, rv in reps.items():
                        inj = rv.get("injection", {})
                        v = inj.get("inj_volume")
                        if v:
                            return v
        elif isinstance(samples, dict):
            for name, sdata in samples.items():
                reps = sdata.get("replicas", {})
                for rk, rv in reps.items():
                    if isinstance(rv, dict):
                        inj = rv.get("injection", {})
                        v = inj.get("inj_volume")
                        if v:
                            return v
    except Exception:
        pass
    return None


def get_sample_metrics(sample_data):
    """Extract key metrics from a sample_data dict."""
    replicas = sample_data.get("replicas", {})
    selected = sample_data.get("selected", {})
    quant = sample_data.get("quantification", {})
    comparison = sample_data.get("comparison", {})

    doc_sel = selected.get("doc", "1")
    dad_sel = selected.get("dad", doc_sel)

    # Resolve "comp" -> actual replica
    doc_rep = replicas.get(doc_sel, {})
    if doc_sel == "comp":
        for rk, rv in replicas.items():
            if isinstance(rv, dict) and rv.get("timeout_composition"):
                doc_rep = rv
                break

    dad_rep = replicas.get(dad_sel, {})
    if not isinstance(doc_rep, dict):
        doc_rep = {}
    if not isinstance(dad_rep, dict):
        dad_rep = {}

    # DOC areas
    areas_doc = (doc_rep.get("areas") or {}).get("DOC", {})
    areas_uib = doc_rep.get("areas_uib") or {}

    # A254 area (from DAD replica)
    areas_a254 = (dad_rep.get("areas") or {}).get("A254", {})

    # SNR
    snr_info = doc_rep.get("snr_info") or {}
    snr_dad = dad_rep.get("snr_info_dad") or {}
    snr_a254 = (snr_dad.get("A254") or {}).get("snr", 0) if isinstance(snr_dad.get("A254"), dict) else 0

    # Timeout info
    timeout_info = doc_rep.get("timeout_info") or {}

    # Anomalies
    anomalies = []
    for rk, rd in replicas.items():
        if not isinstance(rd, dict):
            continue
        for a in rd.get("anomalies", []):
            code = a.get("code") if isinstance(a, dict) else str(a)
            if code not in anomalies:
                anomalies.append(code)

    # Peak info
    peak_info = doc_rep.get("peak_info") or {}

    # Volume — try multiple sources
    inj_vol = None
    for rk, rd in replicas.items():
        if isinstance(rd, dict):
            # Source 1: injection dict
            inj = rd.get("injection") or {}
            v = inj.get("inj_volume")
            if v:
                inj_vol = v
                break
            # Source 2: direct field
            v = rd.get("injection_volume") or rd.get("inj_volume")
            if v:
                inj_vol = v
                break

    # Fractions
    fracs = {}
    for frac_name in ["BioP", "HS", "BB", "SB", "LMW"]:
        fracs[f"DOC_{frac_name}"] = areas_doc.get(frac_name, 0) or 0
        fracs[f"A254_{frac_name}"] = areas_a254.get(frac_name, 0) or 0

    return {
        "ppm_direct": quant.get("concentration_ppm_direct") or quant.get("concentration_ppm"),
        "ppm_uib": quant.get("concentration_ppm_uib"),
        "area_doc_total": areas_doc.get("total", 0),
        "area_uib_total": areas_uib.get("total", 0),
        "area_a254_total": areas_a254.get("total", 0),
        "snr_direct": snr_info.get("snr_direct", 0),
        "snr_uib": snr_info.get("snr_uib", 0),
        "snr_a254": snr_a254,
        "r2_doc": comparison.get("doc", {}).get("pearson", 0) if comparison else 0,
        "r2_dad": (comparison.get("dad", {}) or {}).get("pearson_min", 0) if comparison else 0,
        "n_timeouts": timeout_info.get("n_timeouts", 0),
        "timeout_severity": timeout_info.get("severity", "OK"),
        "baseline_level": peak_info.get("baseline_level", 0),
        "t_max": peak_info.get("t_max", 0),
        "injection_volume": inj_vol,
        "anomalies": "; ".join(anomalies) if anomalies else "",
        "sample_valid": sample_data.get("sample_valid", True),
        "skip_quant": sample_data.get("skip_quantification", False),
        "hci": quant.get("hci"),
        **fracs,
    }


def find_all_pairs(data_folders, search_range=3):
    """Find all COL/BP pairs by number proximity."""
    seqs = {}  # num -> {type: (name, path)}
    for folder in data_folders:
        if not os.path.isdir(folder):
            continue
        for d in os.listdir(folder):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            num = extract_seq_number(d)
            typ = detect_seq_type(d)
            if num is None or typ not in ("COLUMN", "BP"):
                continue
            if num not in seqs:
                seqs[num] = {}
            seqs[num][typ] = (d, full)

    col_nums = sorted(n for n, v in seqs.items() if "COLUMN" in v)
    bp_nums = sorted(n for n, v in seqs.items() if "BP" in v)

    pairs = []
    used_bp = set()
    for cn in col_nums:
        # Find closest BP within range
        best_bp = None
        best_dist = search_range + 1
        for bn in bp_nums:
            if bn in used_bp:
                continue
            dist = abs(bn - cn)
            if dist <= search_range and dist < best_dist:
                best_dist = dist
                best_bp = bn

        if best_bp is not None:
            used_bp.add(best_bp)
            col_name, col_path = seqs[cn]["COLUMN"]
            bp_name, bp_path = seqs[best_bp]["BP"]
            pairs.append({
                "col_num": cn, "bp_num": best_bp,
                "col_name": col_name, "bp_name": bp_name,
                "col_path": col_path, "bp_path": bp_path,
            })

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnostic COL vs BP comparison")
    parser.add_argument("--range", type=int, default=3, help="Search range for pairing (default ±3)")
    parser.add_argument("--seq", type=int, help="Specific SEQ number to compare")
    parser.add_argument("--out", type=str, default="_results/diag_col_bp", help="Output folder")
    args = parser.parse_args()

    data_folders = get_data_folders()
    print(f"Data folders: {data_folders}")

    pairs = find_all_pairs(data_folders, args.range)
    if args.seq:
        pairs = [p for p in pairs if p["col_num"] == args.seq or p["bp_num"] == args.seq]

    print(f"\n{len(pairs)} parells COL/BP trobats (range ±{args.range})")

    # Load analysis and compare
    rows = []
    skipped = []

    for pair in pairs:
        col_analysis = load_analysis(pair["col_path"])
        bp_analysis = load_analysis(pair["bp_path"])

        if not col_analysis or not col_analysis.get("success"):
            skipped.append((pair["col_name"], "sense analisi COL"))
            continue
        if not bp_analysis or not bp_analysis.get("success"):
            skipped.append((pair["bp_name"], "sense analisi BP"))
            continue

        # Load volumes from manifest as fallback
        vol_col_manifest = load_manifest_volume(pair["col_path"])
        vol_bp_manifest = load_manifest_volume(pair["bp_path"])

        col_grouped = col_analysis.get("samples_grouped", {})
        bp_grouped = bp_analysis.get("samples_grouped", {})

        # Find common sample names
        common = set(col_grouped.keys()) & set(bp_grouped.keys())
        common = {n for n in common
                  if col_grouped[n].get("sample_type", "SAMPLE") == "SAMPLE"
                  and bp_grouped[n].get("sample_type", "SAMPLE") == "SAMPLE"}

        if not common:
            skipped.append((f"{pair['col_name']}+{pair['bp_name']}", "0 mostres comunes"))
            continue

        for name in sorted(common):
            col_m = get_sample_metrics(col_grouped[name])
            bp_m = get_sample_metrics(bp_grouped[name])

            ppm_col = col_m["ppm_direct"]
            ppm_bp = bp_m["ppm_direct"]

            if ppm_col is None or ppm_bp is None:
                continue
            if ppm_col <= 0 and ppm_bp <= 0:
                continue

            ratio_doc = ppm_bp / ppm_col if ppm_col > 0 else None
            ratio_a254 = (bp_m["area_a254_total"] / col_m["area_a254_total"]
                         if col_m["area_a254_total"] and col_m["area_a254_total"] > 0
                         and bp_m["area_a254_total"] and bp_m["area_a254_total"] > 0
                         else None)

            # Diagnostic flags
            flags = []
            if col_m["n_timeouts"] > 0:
                flags.append(f"TO_COL({col_m['n_timeouts']})")
            if bp_m["n_timeouts"] > 0:
                flags.append(f"TO_BP({bp_m['n_timeouts']})")
            if col_m["anomalies"]:
                flags.append(f"ANOM_COL:{col_m['anomalies']}")
            if bp_m["anomalies"]:
                flags.append(f"ANOM_BP:{bp_m['anomalies']}")
            if not col_m["sample_valid"]:
                flags.append("INVALID_COL")
            if not bp_m["sample_valid"]:
                flags.append("INVALID_BP")
            if col_m["skip_quant"]:
                flags.append("SKIP_COL")
            if bp_m["skip_quant"]:
                flags.append("SKIP_BP")

            # Volume (from analysis or manifest fallback)
            vol_col = col_m["injection_volume"] or vol_col_manifest
            vol_bp = bp_m["injection_volume"] or vol_bp_manifest
            if vol_col and vol_bp and abs(vol_col - vol_bp) > 10:
                flags.append(f"VOL_DIFF({vol_col}/{vol_bp})")

            # Ratio outlier
            if ratio_doc and (ratio_doc < 0.75 or ratio_doc > 1.25):
                flags.append("RATIO_OUT")

            rows.append({
                "sample": name,
                "col_seq": pair["col_name"],
                "bp_seq": pair["bp_name"],
                "ppm_col": ppm_col,
                "ppm_bp": ppm_bp,
                "ratio_doc": ratio_doc,
                "ppm_uib_col": col_m["ppm_uib"],
                "ppm_uib_bp": bp_m["ppm_uib"],
                "area_doc_col": col_m["area_doc_total"],
                "area_doc_bp": bp_m["area_doc_total"],
                "area_uib_col": col_m["area_uib_total"],
                "area_uib_bp": bp_m["area_uib_total"],
                "area_a254_col": col_m["area_a254_total"],
                "area_a254_bp": bp_m["area_a254_total"],
                "ratio_a254": ratio_a254,
                "snr_col": col_m["snr_direct"],
                "snr_bp": bp_m["snr_direct"],
                "snr_a254_col": col_m["snr_a254"],
                "snr_a254_bp": bp_m["snr_a254"],
                "r2_doc_col": col_m["r2_doc"],
                "r2_doc_bp": bp_m["r2_doc"],
                "r2_dad_col": col_m["r2_dad"],
                "r2_dad_bp": bp_m["r2_dad"],
                "baseline_col": col_m["baseline_level"],
                "baseline_bp": bp_m["baseline_level"],
                "t_max_col": col_m["t_max"],
                "t_max_bp": bp_m["t_max"],
                "vol_col": vol_col,
                "vol_bp": vol_bp,
                "to_col": col_m["n_timeouts"],
                "to_bp": bp_m["n_timeouts"],
                "hci_col": col_m["hci"],
                "hci_bp": bp_m["hci"],
                # Fractions DOC
                "DOC_BioP_col": col_m["DOC_BioP"], "DOC_BioP_bp": bp_m["DOC_BioP"],
                "DOC_HS_col": col_m["DOC_HS"], "DOC_HS_bp": bp_m["DOC_HS"],
                "DOC_BB_col": col_m["DOC_BB"], "DOC_BB_bp": bp_m["DOC_BB"],
                "DOC_SB_col": col_m["DOC_SB"], "DOC_SB_bp": bp_m["DOC_SB"],
                "DOC_LMW_col": col_m["DOC_LMW"], "DOC_LMW_bp": bp_m["DOC_LMW"],
                # Fractions A254
                "A254_BioP_col": col_m["A254_BioP"], "A254_BioP_bp": bp_m["A254_BioP"],
                "A254_HS_col": col_m["A254_HS"], "A254_HS_bp": bp_m["A254_HS"],
                "A254_BB_col": col_m["A254_BB"], "A254_BB_bp": bp_m["A254_BB"],
                "A254_SB_col": col_m["A254_SB"], "A254_SB_bp": bp_m["A254_SB"],
                "A254_LMW_col": col_m["A254_LMW"], "A254_LMW_bp": bp_m["A254_LMW"],
                "flags": "; ".join(flags) if flags else "",
            })

    # --- Output ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skipped
    if skipped:
        print(f"\nSaltades ({len(skipped)}):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    if not rows:
        print("\nCap parella amb dades per comparar!")
        return

    # CSV
    csv_path = out_dir / "col_bp_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path} ({len(rows)} files)")

    # Stats
    valid = [r for r in rows if r["ratio_doc"] is not None]
    ratios = np.array([r["ratio_doc"] for r in valid])
    outliers = [r for r in valid if r["ratio_doc"] < 0.75 or r["ratio_doc"] > 1.25]
    flagged = [r for r in rows if r["flags"]]

    print(f"\n=== RESUM ===")
    print(f"Mostres comparades: {len(valid)}")
    print(f"Ratio DOC mediana: {np.median(ratios):.3f}")
    print(f"Ratio DOC mitjana: {np.mean(ratios):.3f} (SD={np.std(ratios):.3f})")
    print(f"Outliers (ratio <0.75 o >1.25): {len(outliers)}")
    print(f"Mostres amb flags: {len(flagged)}")

    ratios_a254 = [r["ratio_a254"] for r in valid if r["ratio_a254"] is not None]
    if ratios_a254:
        ra = np.array(ratios_a254)
        print(f"\nRatio A254 mediana: {np.median(ra):.3f}")
        print(f"Ratio A254 mitjana: {np.mean(ra):.3f} (SD={np.std(ra):.3f})")

        # Comparar amb ratio teoric de volums
        vol_ratios = []
        for r in valid:
            if r["vol_col"] and r["vol_bp"] and r["vol_col"] > 0:
                vol_ratios.append(r["vol_bp"] / r["vol_col"])
        if vol_ratios:
            vol_ratio_med = np.median(vol_ratios)
            print(f"\nRatio volums BP/COL teoric: {vol_ratio_med:.3f} "
                  f"(vol_bp={valid[0]['vol_bp']}uL / vol_col={valid[0]['vol_col']}uL)")
            print(f"Ratio A254 observat / teoric: {np.median(ra) / vol_ratio_med:.3f} "
                  f"(1.0 = perfecte, <1 = perdua senyal BP, >1 = guany)")
            print(f"Ratio DOC observat / teoric: {np.median(ratios):.3f} "
                  f"(no depèn de volum — DOC es quantifica per calibracio)")
        else:
            print(f"\n(Sense info de volums per comparar amb ratio teoric)")

    # Top outliers
    if outliers:
        print(f"\n=== OUTLIERS DOC (ratio fora 0.75-1.25) ===")
        for r in sorted(outliers, key=lambda x: abs(x["ratio_doc"] - 1.0), reverse=True)[:15]:
            print(f"  {r['sample']:20s} {r['col_seq']:15s}/{r['bp_seq']:15s}  "
                  f"ratio={r['ratio_doc']:.3f}  ppm_col={r['ppm_col']:.2f}  ppm_bp={r['ppm_bp']:.2f}  "
                  f"vol={r['vol_col']}/{r['vol_bp']}  {r['flags']}")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"COL vs BP Diagnostic ({len(valid)} mostres)", fontsize=14, fontweight="bold")

        # 1. Scatter DOC ppm
        ax = axes[0, 0]
        col_ppms = np.array([r["ppm_col"] for r in valid])
        bp_ppms = np.array([r["ppm_bp"] for r in valid])
        out_mask = np.array([r["ratio_doc"] < 0.75 or r["ratio_doc"] > 1.25 for r in valid])
        ax.scatter(col_ppms[~out_mask], bp_ppms[~out_mask], c="#228be6", s=15, alpha=0.6)
        ax.scatter(col_ppms[out_mask], bp_ppms[out_mask], c="#e03131", s=25, marker="x", alpha=0.8)
        mx = max(col_ppms.max(), bp_ppms.max()) * 1.1
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        if len(col_ppms) >= 3:
            z = np.polyfit(col_ppms, bp_ppms, 1)
            xf = np.linspace(0, mx, 50)
            ax.plot(xf, np.polyval(z, xf), "r-", alpha=0.5, lw=1)
            ax.set_title(f"DOC ppm (slope={z[0]:.3f}, R²={np.corrcoef(col_ppms, bp_ppms)[0,1]**2:.4f})")
        else:
            ax.set_title("DOC ppm")
        ax.set_xlabel("ppm COLUMN")
        ax.set_ylabel("ppm BP")

        # 2. Scatter A254
        ax = axes[0, 1]
        a254_valid = [r for r in valid if r["area_a254_col"] and r["area_a254_bp"]
                      and r["area_a254_col"] > 0 and r["area_a254_bp"] > 0]
        if a254_valid:
            ca = np.array([r["area_a254_col"] for r in a254_valid])
            ba = np.array([r["area_a254_bp"] for r in a254_valid])
            ra_out = np.array([r.get("ratio_a254", 1) is not None
                              and (r["ratio_a254"] < 0.75 or r["ratio_a254"] > 1.25)
                              for r in a254_valid])
            ax.scatter(ca[~ra_out], ba[~ra_out], c="#40c057", s=15, alpha=0.6)
            ax.scatter(ca[ra_out], ba[ra_out], c="#e03131", s=25, marker="x", alpha=0.8)
            mx254 = max(ca.max(), ba.max()) * 1.1
            ax.plot([0, mx254], [0, mx254], "k--", alpha=0.3)
            if len(ca) >= 3:
                z254 = np.polyfit(ca, ba, 1)
                xf254 = np.linspace(0, mx254, 50)
                ax.plot(xf254, np.polyval(z254, xf254), "r-", alpha=0.5, lw=1)
                ax.set_title(f"A254 (slope={z254[0]:.3f}, R²={np.corrcoef(ca, ba)[0,1]**2:.4f})")
            else:
                ax.set_title("A254")
            ax.set_xlabel("A254 COLUMN")
            ax.set_ylabel("A254 BP")
        else:
            ax.text(0.5, 0.5, "Sense dades A254", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("A254")

        # 3. Bland-Altman DOC
        ax = axes[0, 2]
        means = (col_ppms + bp_ppms) / 2
        diffs = bp_ppms - col_ppms
        ax.scatter(means[~out_mask], diffs[~out_mask], c="#228be6", s=15, alpha=0.6)
        ax.scatter(means[out_mask], diffs[out_mask], c="#e03131", s=25, marker="x", alpha=0.8)
        md = np.mean(diffs)
        sd = np.std(diffs)
        ax.axhline(md, color="blue", ls="-", alpha=0.4)
        ax.axhline(md + 1.96 * sd, color="red", ls="--", alpha=0.3)
        ax.axhline(md - 1.96 * sd, color="red", ls="--", alpha=0.3)
        ax.axhline(0, color="black", ls="-", alpha=0.2)
        ax.set_title(f"Bland-Altman DOC (bias={md:.3f})")
        ax.set_xlabel("Mitjana (COL+BP)/2")
        ax.set_ylabel("Diferencia (BP-COL)")

        # 4. Histogram ratios DOC + A254
        ax = axes[1, 0]
        ax.hist(ratios, bins=min(20, len(ratios)), color="#228be6", alpha=0.7, edgecolor="white", label="DOC")
        ax.axvline(np.median(ratios), color="red", ls="-", alpha=0.6, label=f"Med={np.median(ratios):.3f}")
        ax.axvline(1.0, color="black", ls="--", alpha=0.3)
        if ratios_a254:
            ax.hist(ratios_a254, bins=min(20, len(ratios_a254)), color="#40c057",
                    alpha=0.4, edgecolor="white", label="A254")
            ax.axvline(np.median(ratios_a254), color="#2b8a3e", ls="-", alpha=0.6,
                       label=f"Med A254={np.median(ratios_a254):.3f}")
        ax.legend(fontsize=7)
        ax.set_title("Distribucio ratio BP/COL")
        ax.set_xlabel("Ratio")

        # 5. DOC ratio vs concentration
        ax = axes[1, 1]
        ax.scatter(col_ppms, ratios, c="#228be6", s=15, alpha=0.6)
        ax.axhline(1.0, color="black", ls="--", alpha=0.3)
        ax.axhline(0.75, color="red", ls="--", alpha=0.2)
        ax.axhline(1.25, color="red", ls="--", alpha=0.2)
        ax.set_title("Ratio DOC vs concentracio")
        ax.set_xlabel("ppm COLUMN")
        ax.set_ylabel("Ratio BP/COL")
        ax.set_ylim(0, max(2.0, ratios.max() * 1.1))

        # 6. DOC ratio per SEQ pair (temporal)
        ax = axes[1, 2]
        seq_pairs_unique = sorted(set((r["col_seq"], r["bp_seq"]) for r in valid))
        x_labels = []
        medians = []
        stds = []
        for col_s, bp_s in seq_pairs_unique:
            pair_ratios = [r["ratio_doc"] for r in valid
                          if r["col_seq"] == col_s and r["bp_seq"] == bp_s]
            if pair_ratios:
                x_labels.append(f"{extract_seq_number(col_s)}/{extract_seq_number(bp_s)}")
                medians.append(np.median(pair_ratios))
                stds.append(np.std(pair_ratios))

        x = np.arange(len(x_labels))
        ax.errorbar(x, medians, yerr=stds, fmt="o-", color="#228be6", ms=5, capsize=3)
        ax.axhline(1.0, color="black", ls="--", alpha=0.3)
        ax.axhline(0.75, color="red", ls="--", alpha=0.2)
        ax.axhline(1.25, color="red", ls="--", alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title("Ratio DOC per parell SEQ")
        ax.set_ylabel("Ratio BP/COL (mediana ± SD)")

        for a in axes.flat:
            a.tick_params(labelsize=7)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        fig.tight_layout()
        plot_path = out_dir / "col_bp_diagnostic.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        print(f"\nPlot: {plot_path}")
        plt.close(fig)

        # --- Per-fraction scatter ---
        frac_names = ["BioP", "HS", "BB", "SB", "LMW"]
        fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
        fig2.suptitle("Fraccions COL vs BP", fontsize=14, fontweight="bold")

        for i, frac in enumerate(frac_names):
            # DOC fraction
            ax = axes2[0, i]
            cv = np.array([r[f"DOC_{frac}_col"] for r in valid])
            bv = np.array([r[f"DOC_{frac}_bp"] for r in valid])
            mask = (cv > 0) & (bv > 0)
            if mask.any():
                ax.scatter(cv[mask], bv[mask], c="#228be6", s=10, alpha=0.5)
                mx_f = max(cv[mask].max(), bv[mask].max()) * 1.1
                ax.plot([0, mx_f], [0, mx_f], "k--", alpha=0.3, lw=0.8)
                if mask.sum() >= 3:
                    try:
                        zf = np.polyfit(cv[mask], bv[mask], 1)
                        r2f = np.corrcoef(cv[mask], bv[mask])[0, 1] ** 2
                        ax.set_title(f"DOC {frac}\nslope={zf[0]:.2f} R²={r2f:.3f}", fontsize=8)
                    except Exception:
                        ax.set_title(f"DOC {frac}", fontsize=8)
                else:
                    ax.set_title(f"DOC {frac}", fontsize=8)
            else:
                ax.set_title(f"DOC {frac} (no data)", fontsize=8)
            ax.tick_params(labelsize=6)

            # A254 fraction
            ax = axes2[1, i]
            cv = np.array([r[f"A254_{frac}_col"] for r in valid])
            bv = np.array([r[f"A254_{frac}_bp"] for r in valid])
            mask = (cv > 0) & (bv > 0)
            if mask.any():
                ax.scatter(cv[mask], bv[mask], c="#40c057", s=10, alpha=0.5)
                mx_f = max(cv[mask].max(), bv[mask].max()) * 1.1
                ax.plot([0, mx_f], [0, mx_f], "k--", alpha=0.3, lw=0.8)
                if mask.sum() >= 3:
                    try:
                        zf = np.polyfit(cv[mask], bv[mask], 1)
                        r2f = np.corrcoef(cv[mask], bv[mask])[0, 1] ** 2
                        ax.set_title(f"A254 {frac}\nslope={zf[0]:.2f} R²={r2f:.3f}", fontsize=8)
                    except Exception:
                        ax.set_title(f"A254 {frac}", fontsize=8)
                else:
                    ax.set_title(f"A254 {frac}", fontsize=8)
            else:
                ax.set_title(f"A254 {frac} (no data)", fontsize=8)
            ax.tick_params(labelsize=6)

        for a in axes2.flat:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        fig2.tight_layout()
        plot2_path = out_dir / "col_bp_fractions.png"
        fig2.savefig(str(plot2_path), dpi=150, bbox_inches="tight")
        print(f"Plot fraccions: {plot2_path}")
        plt.close(fig2)

    except ImportError:
        print("\nmatplotlib no disponible — nomes CSV generat")

    print("\nFet!")


if __name__ == "__main__":
    main()
