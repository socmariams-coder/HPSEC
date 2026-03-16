"""
Diagnostic script: COL vs BP comparison (v2)
=============================================
Compara TOTES les mostres (SAMPLE + KHP) entre SEQs COL i BP.
Matching per nom de mostra, NO per proximitat numèrica.

Usage:
    python diag_col_bp.py                   # totes les parelles
    python diag_col_bp.py --seq 288         # una SEQ concreta
    python diag_col_bp.py --type KHP        # nomes KHP
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_analysis(seq_path):
    """Load analysis_result.json, return dict or None."""
    jp = os.path.join(seq_path, "CHECK", "data", "analysis_result.json")
    if not os.path.exists(jp):
        return None
    try:
        with open(jp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_manifest_volume(seq_path):
    """Load typical injection volume from manifest."""
    mp = os.path.join(seq_path, "CHECK", "data", "import_manifest.json")
    if not os.path.exists(mp):
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
        v = (m.get("sequence") or {}).get("injection_volume")
        if v:
            return v
        samples = m.get("samples", [])
        if isinstance(samples, list):
            for s in samples:
                for r in (s.get("replicas") or []):
                    v = (r.get("injection") or {}).get("inj_volume")
                    if v:
                        return v
        elif isinstance(samples, dict):
            for sdata in samples.values():
                for rv in (sdata.get("replicas") or {}).values():
                    if isinstance(rv, dict):
                        v = (rv.get("injection") or {}).get("inj_volume")
                        if v:
                            return v
    except Exception:
        pass
    return None


def get_sample_metrics(sample_data):
    """Extract metrics from sample_data dict."""
    replicas = sample_data.get("replicas", {})
    selected = sample_data.get("selected", {})
    quant = sample_data.get("quantification") or {}
    comparison = sample_data.get("comparison") or {}

    doc_sel = selected.get("doc", "1")
    dad_sel = selected.get("dad", doc_sel)

    doc_rep = replicas.get(doc_sel, {})
    if doc_sel == "comp":
        for rv in replicas.values():
            if isinstance(rv, dict) and rv.get("timeout_composition"):
                doc_rep = rv
                break
    dad_rep = replicas.get(dad_sel, {})
    if not isinstance(doc_rep, dict):
        doc_rep = {}
    if not isinstance(dad_rep, dict):
        dad_rep = {}

    areas_doc = (doc_rep.get("areas") or {}).get("DOC", {})
    areas_uib = doc_rep.get("areas_uib") or {}
    areas_a254 = (dad_rep.get("areas") or {}).get("A254", {})
    snr_info = doc_rep.get("snr_info") or {}
    snr_dad = dad_rep.get("snr_info_dad") or {}
    snr_a254 = (snr_dad.get("A254") or {}).get("snr", 0) if isinstance(snr_dad.get("A254"), dict) else 0
    timeout_info = doc_rep.get("timeout_info") or {}
    peak_info = doc_rep.get("peak_info") or {}

    anomalies = []
    for rd in replicas.values():
        if not isinstance(rd, dict):
            continue
        for a in rd.get("anomalies", []):
            code = a.get("code") if isinstance(a, dict) else str(a)
            if code not in anomalies:
                anomalies.append(code)

    inj_vol = None
    for rd in replicas.values():
        if isinstance(rd, dict):
            v = (rd.get("injection") or {}).get("inj_volume") or rd.get("injection_volume")
            if v:
                inj_vol = v
                break

    fracs = {}
    for f in ["BioP", "HS", "BB", "SB", "LMW"]:
        fracs[f"DOC_{f}"] = areas_doc.get(f, 0) or 0
        fracs[f"A254_{f}"] = areas_a254.get(f, 0) or 0

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
        "r2_dad": (comparison.get("dad") or {}).get("pearson_min", 0) if comparison else 0,
        "n_timeouts": timeout_info.get("n_timeouts", 0),
        "timeout_severity": timeout_info.get("severity", "OK"),
        "baseline_level": peak_info.get("baseline_level", 0),
        "t_max": peak_info.get("t_max", 0),
        "injection_volume": inj_vol,
        "anomalies": "; ".join(anomalies) if anomalies else "",
        "sample_valid": sample_data.get("sample_valid", True),
        "sample_type": sample_data.get("sample_type", "SAMPLE"),
        "hci": quant.get("hci"),
        **fracs,
    }


# ---------------------------------------------------------------------------
# Matching by sample name
# ---------------------------------------------------------------------------

def scan_all_seqs(data_folders):
    """Scan all SEQs with analysis."""
    seqs = []
    for folder in data_folders:
        if not os.path.isdir(folder):
            continue
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            jp = os.path.join(full, "CHECK", "data", "analysis_result.json")
            if not os.path.exists(jp):
                continue
            method = detect_seq_type(d)
            seqs.append({"name": d, "path": full, "method": method})
    return seqs


def build_inventory(seqs, include_types=None):
    """Build global sample inventory: {name: {col_seq, bp_seq, col_data, bp_data}}."""
    if include_types is None:
        include_types = {"SAMPLE", "KHP"}

    inventory = defaultdict(dict)

    for seq_info in seqs:
        data = load_analysis(seq_info["path"])
        if not data or not data.get("success"):
            continue
        seq_info["_data"] = data
        sg = data.get("samples_grouped", {})
        vol = load_manifest_volume(seq_info["path"])

        for name, sdata in sg.items():
            st = sdata.get("sample_type", "SAMPLE")
            if st not in include_types:
                continue
            method = seq_info["method"]
            if method == "BP":
                inventory[name]["bp_seq"] = seq_info["name"]
                inventory[name]["bp_path"] = seq_info["path"]
                inventory[name]["bp_data"] = sdata
                inventory[name]["bp_vol"] = vol
                inventory[name]["sample_type"] = st
            else:
                inventory[name]["col_seq"] = seq_info["name"]
                inventory[name]["col_path"] = seq_info["path"]
                inventory[name]["col_data"] = sdata
                inventory[name]["col_vol"] = vol
                inventory[name]["sample_type"] = st

    return dict(inventory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnostic COL vs BP comparison v2")
    parser.add_argument("--seq", type=int, help="Specific SEQ number to filter")
    parser.add_argument("--type", type=str, default="ALL",
                        help="Sample type filter: ALL, SAMPLE, KHP (default: ALL)")
    parser.add_argument("--out", type=str, default="_results/diag_col_bp",
                        help="Output folder")
    args = parser.parse_args()

    data_folders = get_data_folders()
    print(f"Data folders: {data_folders}")

    # Scan
    seqs = scan_all_seqs(data_folders)
    if args.seq:
        seqs = [s for s in seqs if extract_seq_number(s["name"]) == args.seq
                or (extract_seq_number(s["name"]) and
                    abs(extract_seq_number(s["name"]) - args.seq) <= 5)]

    col_seqs = [s for s in seqs if s["method"] == "COLUMN"]
    bp_seqs = [s for s in seqs if s["method"] == "BP"]
    print(f"\nSEQs amb analisi: {len(col_seqs)} COLUMN, {len(bp_seqs)} BP")

    # Build inventory
    include = {"SAMPLE", "KHP"} if args.type == "ALL" else {args.type}
    inventory = build_inventory(seqs, include_types=include)
    print(f"Mostres a l'inventari: {len(inventory)}")

    # Separate: both modes, COL only, BP only
    both = {n: v for n, v in inventory.items() if "col_data" in v and "bp_data" in v}
    col_only = {n: v for n, v in inventory.items() if "col_data" in v and "bp_data" not in v}
    bp_only = {n: v for n, v in inventory.items() if "bp_data" in v and "col_data" not in v}

    print(f"  Amb ambdos modes (COL+BP): {len(both)}")
    print(f"  Nomes COLUMN: {len(col_only)}")
    print(f"  Nomes BP: {len(bp_only)}")

    # Build comparison rows
    rows = []
    for name in sorted(both.keys()):
        inv = both[name]
        col_m = get_sample_metrics(inv["col_data"])
        bp_m = get_sample_metrics(inv["bp_data"])

        ppm_col = col_m["ppm_direct"]
        ppm_bp = bp_m["ppm_direct"]

        if ppm_col is None and ppm_bp is None:
            continue

        ratio_doc = (ppm_bp / ppm_col if ppm_col and ppm_col > 0 and ppm_bp else None)
        ratio_a254 = (bp_m["area_a254_total"] / col_m["area_a254_total"]
                      if col_m["area_a254_total"] and col_m["area_a254_total"] > 0
                      and bp_m["area_a254_total"] and bp_m["area_a254_total"] > 0
                      else None)

        vol_col = inv.get("col_vol")
        vol_bp = inv.get("bp_vol")
        ratio_vol = (vol_bp / vol_col if vol_col and vol_bp and vol_col > 0 else None)

        # Flags
        flags = []
        if col_m["n_timeouts"] > 0:
            flags.append(f"TO_COL({col_m['n_timeouts']})")
        if bp_m["n_timeouts"] > 0:
            flags.append(f"TO_BP({bp_m['n_timeouts']})")
        if col_m["anomalies"]:
            flags.append(f"ANOM_COL:{col_m['anomalies']}")
        if bp_m["anomalies"]:
            flags.append(f"ANOM_BP:{bp_m['anomalies']}")
        if ratio_doc and (ratio_doc < 0.75 or ratio_doc > 1.25):
            flags.append("RATIO_DOC_OUT")
        if ratio_a254 and ratio_vol and ratio_vol > 0:
            norm_a254 = ratio_a254 / ratio_vol
            if norm_a254 < 0.5 or norm_a254 > 1.5:
                flags.append(f"A254_NORM={norm_a254:.2f}")

        rows.append({
            "sample": name,
            "sample_type": inv.get("sample_type", "SAMPLE"),
            "col_seq": inv.get("col_seq", ""),
            "bp_seq": inv.get("bp_seq", ""),
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
            "ratio_a254_norm": (ratio_a254 / ratio_vol if ratio_a254 and ratio_vol else None),
            "vol_col": vol_col,
            "vol_bp": vol_bp,
            "ratio_vol": ratio_vol,
            "snr_col": col_m["snr_direct"],
            "snr_bp": bp_m["snr_direct"],
            "snr_a254_col": col_m["snr_a254"],
            "snr_a254_bp": bp_m["snr_a254"],
            "r2_doc_col": col_m["r2_doc"],
            "r2_doc_bp": bp_m["r2_doc"],
            "baseline_col": col_m["baseline_level"],
            "baseline_bp": bp_m["baseline_level"],
            "t_max_col": col_m["t_max"],
            "t_max_bp": bp_m["t_max"],
            "to_col": col_m["n_timeouts"],
            "to_bp": bp_m["n_timeouts"],
            "hci_col": col_m["hci"],
            "hci_bp": bp_m["hci"],
            **{f"DOC_{f}_col": col_m[f"DOC_{f}"] for f in ["BioP","HS","BB","SB","LMW"]},
            **{f"DOC_{f}_bp": bp_m[f"DOC_{f}"] for f in ["BioP","HS","BB","SB","LMW"]},
            **{f"A254_{f}_col": col_m[f"A254_{f}"] for f in ["BioP","HS","BB","SB","LMW"]},
            **{f"A254_{f}_bp": bp_m[f"A254_{f}"] for f in ["BioP","HS","BB","SB","LMW"]},
            "flags": "; ".join(flags) if flags else "",
        })

    # --- Output ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("\nCap mostra amb ambdos modes per comparar!")
        return

    # CSV
    csv_path = out_dir / "col_bp_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path} ({len(rows)} files)")

    # Split by type
    samples = [r for r in rows if r["sample_type"] == "SAMPLE"]
    khps = [r for r in rows if r["sample_type"] == "KHP"]
    print(f"  SAMPLE: {len(samples)}, KHP: {len(khps)}")

    # Stats
    valid = [r for r in rows if r["ratio_doc"] is not None]
    if valid:
        ratios = np.array([r["ratio_doc"] for r in valid])
        outliers = [r for r in valid if r["ratio_doc"] < 0.75 or r["ratio_doc"] > 1.25]

        print(f"\n=== RESUM GLOBAL ({len(valid)} mostres) ===")
        print(f"Ratio DOC: mediana={np.median(ratios):.3f}, "
              f"mitjana={np.mean(ratios):.3f}, SD={np.std(ratios):.3f}")
        print(f"Outliers DOC: {len(outliers)}/{len(valid)}")

        ratios_a254 = [r["ratio_a254"] for r in valid if r["ratio_a254"] is not None]
        if ratios_a254:
            ra = np.array(ratios_a254)
            print(f"Ratio A254: mediana={np.median(ra):.3f}, "
                  f"mitjana={np.mean(ra):.3f}, SD={np.std(ra):.3f}")

        # Normalized A254 (dividing by volume ratio)
        norm_a254 = [r["ratio_a254_norm"] for r in valid if r["ratio_a254_norm"] is not None]
        if norm_a254:
            na = np.array(norm_a254)
            print(f"Ratio A254 normalitzat (÷ vol): mediana={np.median(na):.3f}, "
                  f"mitjana={np.mean(na):.3f}, SD={np.std(na):.3f}")

        # Per sample type
        for st, label in [("SAMPLE", "MOSTRES"), ("KHP", "KHP")]:
            subset = [r for r in valid if r["sample_type"] == st]
            if not subset:
                continue
            sr = np.array([r["ratio_doc"] for r in subset])
            so = [r for r in subset if r["ratio_doc"] < 0.75 or r["ratio_doc"] > 1.25]
            print(f"\n  {label} ({len(subset)}):")
            print(f"    DOC ratio: med={np.median(sr):.3f}, mean={np.mean(sr):.3f}, SD={np.std(sr):.3f}")
            print(f"    Outliers: {len(so)}/{len(subset)}")

        # Per SEQ pair
        print(f"\n=== PER PARELL SEQ ===")
        pairs = defaultdict(list)
        for r in valid:
            pairs[(r["col_seq"], r["bp_seq"])].append(r)
        for (cs, bs), pr in sorted(pairs.items()):
            pr_ratios = np.array([r["ratio_doc"] for r in pr])
            n_out = sum(1 for r in pr if r["ratio_doc"] < 0.75 or r["ratio_doc"] > 1.25)
            types = set(r["sample_type"] for r in pr)
            type_str = "+".join(sorted(types))
            print(f"  {cs:15s} / {bs:15s}: n={len(pr):2d} ({type_str:>10s})  "
                  f"ratio={np.median(pr_ratios):.3f}±{np.std(pr_ratios):.3f}  "
                  f"outliers={n_out}")

        # Outlier details
        if outliers:
            print(f"\n=== OUTLIERS DOC ({len(outliers)}) ===")
            for r in sorted(outliers, key=lambda x: abs(x["ratio_doc"] - 1.0), reverse=True)[:20]:
                print(f"  {r['sample']:20s} [{r['sample_type']:6s}] "
                      f"{r['col_seq']:15s}/{r['bp_seq']:15s}  "
                      f"ratio={r['ratio_doc']:.3f}  "
                      f"ppm={r['ppm_col']:.2f}/{r['ppm_bp']:.2f}  "
                      f"vol={r['vol_col']}/{r['vol_bp']}  "
                      f"{r['flags']}")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f"COL vs BP — {len(valid)} mostres ({len(samples)} SAMPLE + {len(khps)} KHP)",
                     fontsize=14, fontweight="bold")

        # Colors per type
        def _colors(data_list):
            return ['#E74C3C' if r['sample_type'] == 'KHP' else '#228be6' for r in data_list]

        # 1. Scatter DOC
        ax = axes[0, 0]
        if valid:
            col_ppms = np.array([r["ppm_col"] or 0 for r in valid])
            bp_ppms = np.array([r["ppm_bp"] or 0 for r in valid])
            ax.scatter(col_ppms, bp_ppms, c=_colors(valid), s=20, alpha=0.6, edgecolors='none')
            mx = max(col_ppms.max(), bp_ppms.max()) * 1.1
            ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
            if len(col_ppms) >= 3:
                z = np.polyfit(col_ppms, bp_ppms, 1)
                r2 = np.corrcoef(col_ppms, bp_ppms)[0, 1] ** 2
                xf = np.linspace(0, mx, 50)
                ax.plot(xf, np.polyval(z, xf), "r-", alpha=0.5, lw=1)
                ax.set_title(f"DOC ppm (slope={z[0]:.3f}, R²={r2:.4f})", fontsize=9)
        ax.set_xlabel("ppm COLUMN"); ax.set_ylabel("ppm BP")

        # 2. Scatter A254
        ax = axes[0, 1]
        a254_valid = [r for r in valid if r["area_a254_col"] and r["area_a254_bp"]
                      and r["area_a254_col"] > 0 and r["area_a254_bp"] > 0]
        if a254_valid:
            ca = np.array([r["area_a254_col"] for r in a254_valid])
            ba = np.array([r["area_a254_bp"] for r in a254_valid])
            ax.scatter(ca, ba, c=_colors(a254_valid), s=20, alpha=0.6, edgecolors='none')
            mx = max(ca.max(), ba.max()) * 1.1
            ax.plot([0, mx], [0, mx * (0.25 if mx > 100 else 1)], "g--", alpha=0.3, label="vol ratio")
            if len(ca) >= 3:
                z = np.polyfit(ca, ba, 1)
                r2 = np.corrcoef(ca, ba)[0, 1] ** 2
                xf = np.linspace(0, mx, 50)
                ax.plot(xf, np.polyval(z, xf), "r-", alpha=0.5, lw=1)
                ax.set_title(f"A254 (slope={z[0]:.3f}, R²={r2:.4f})", fontsize=9)
            ax.legend(fontsize=7)
        ax.set_xlabel("A254 COLUMN"); ax.set_ylabel("A254 BP")

        # 3. Bland-Altman DOC
        ax = axes[0, 2]
        if valid:
            means = (col_ppms + bp_ppms) / 2
            diffs = bp_ppms - col_ppms
            ax.scatter(means, diffs, c=_colors(valid), s=20, alpha=0.6, edgecolors='none')
            md, sd = np.mean(diffs), np.std(diffs)
            ax.axhline(md, color="blue", ls="-", alpha=0.4)
            ax.axhline(md + 1.96 * sd, color="red", ls="--", alpha=0.3)
            ax.axhline(md - 1.96 * sd, color="red", ls="--", alpha=0.3)
            ax.axhline(0, color="black", ls="-", alpha=0.2)
            ax.set_title(f"Bland-Altman DOC (bias={md:.3f})", fontsize=9)
        ax.set_xlabel("Mitjana"); ax.set_ylabel("Dif (BP-COL)")

        # 4. Histogram ratios
        ax = axes[1, 0]
        if valid:
            ax.hist(ratios, bins=min(25, len(ratios)), color="#228be6", alpha=0.7, edgecolor="white", label="DOC")
            ax.axvline(np.median(ratios), color="red", ls="-", alpha=0.6, label=f"Med={np.median(ratios):.3f}")
            ax.axvline(1.0, color="black", ls="--", alpha=0.3)
            if ratios_a254:
                ax.hist(ratios_a254, bins=min(25, len(ratios_a254)), color="#40c057",
                        alpha=0.4, edgecolor="white", label="A254")
            # KHP ratios highlighted
            khp_ratios = [r["ratio_doc"] for r in valid if r["sample_type"] == "KHP" and r["ratio_doc"]]
            if khp_ratios:
                for kr in khp_ratios:
                    ax.axvline(kr, color="#E74C3C", ls=":", alpha=0.5, lw=0.8)
            ax.legend(fontsize=7)
        ax.set_title("Distribucio ratio BP/COL"); ax.set_xlabel("Ratio")

        # 5. Ratio vs concentration
        ax = axes[1, 1]
        if valid:
            ax.scatter(col_ppms, ratios, c=_colors(valid), s=20, alpha=0.6, edgecolors='none')
            ax.axhline(1.0, color="black", ls="--", alpha=0.3)
            ax.axhline(0.75, color="red", ls="--", alpha=0.2)
            ax.axhline(1.25, color="red", ls="--", alpha=0.2)
            ax.set_ylim(0, max(2.0, ratios.max() * 1.1))
        ax.set_title("Ratio DOC vs concentracio"); ax.set_xlabel("ppm COL"); ax.set_ylabel("Ratio")

        # 6. Ratio A254 normalitzat per volum
        ax = axes[1, 2]
        if norm_a254:
            na = np.array(norm_a254)
            ax.hist(na, bins=min(20, len(na)), color="#40c057", alpha=0.7, edgecolor="white")
            ax.axvline(np.median(na), color="red", ls="-", alpha=0.6,
                       label=f"Med={np.median(na):.3f}")
            ax.axvline(1.0, color="black", ls="--", alpha=0.3, label="Teoric (1.0)")
            ax.legend(fontsize=7)
        ax.set_title("A254 BP/COL normalitzat per volum"); ax.set_xlabel("Ratio norm")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#228be6', ms=8, label='SAMPLE'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', ms=8, label='KHP'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9)

        for a in axes.flat:
            a.tick_params(labelsize=7)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = out_dir / "col_bp_diagnostic.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        print(f"\nPlot: {plot_path}")
        plt.close(fig)

        # Per-fraction scatter
        frac_names = ["BioP", "HS", "BB", "SB", "LMW"]
        fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
        fig2.suptitle(f"Fraccions COL vs BP ({len(valid)} mostres)", fontsize=14, fontweight="bold")

        for i, frac in enumerate(frac_names):
            for row_idx, (prefix, color) in enumerate([("DOC", "#228be6"), ("A254", "#40c057")]):
                ax = axes2[row_idx, i]
                cv = np.array([r[f"{prefix}_{frac}_col"] for r in valid])
                bv = np.array([r[f"{prefix}_{frac}_bp"] for r in valid])
                mask = (cv > 0) & (bv > 0)
                if mask.any():
                    cc = [('#E74C3C' if valid[j]['sample_type'] == 'KHP' else color)
                          for j in range(len(valid))]
                    cc_masked = [c for c, m in zip(cc, mask) if m]
                    ax.scatter(cv[mask], bv[mask], c=cc_masked, s=10, alpha=0.5)
                    mx = max(cv[mask].max(), bv[mask].max()) * 1.1
                    ax.plot([0, mx], [0, mx], "k--", alpha=0.3, lw=0.8)
                    if mask.sum() >= 3:
                        try:
                            zf = np.polyfit(cv[mask], bv[mask], 1)
                            r2f = np.corrcoef(cv[mask], bv[mask])[0, 1] ** 2
                            ax.set_title(f"{prefix} {frac}\nsl={zf[0]:.2f} R²={r2f:.3f}", fontsize=8)
                        except Exception:
                            ax.set_title(f"{prefix} {frac}", fontsize=8)
                else:
                    ax.set_title(f"{prefix} {frac}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        fig2.tight_layout()
        plot2_path = out_dir / "col_bp_fractions.png"
        fig2.savefig(str(plot2_path), dpi=150, bbox_inches="tight")
        print(f"Plot fraccions: {plot2_path}")
        plt.close(fig2)

    except ImportError:
        print("\nmatplotlib no disponible")

    print("\nFet!")


if __name__ == "__main__":
    main()
