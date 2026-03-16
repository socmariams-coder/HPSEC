"""
Deep diagnostic: COL vs BP — per-SEQ analysis including KHP, calibration, delays.
==================================================================================
For each SEQ pair, extracts:
- KHP data (area, RF, concentration, quality)
- Calibration applied (RF, intercept, fingerprint)
- Delay info
- Volume info
- Sample ratios with anomaly context
"""
import os, json, csv
import numpy as np
from collections import defaultdict
from pathlib import Path
from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_seq_info(seq_path):
    """Extract comprehensive info from a SEQ."""
    name = os.path.basename(seq_path)
    data_dir = os.path.join(seq_path, "CHECK", "data")

    analysis = load_json(os.path.join(data_dir, "analysis_result.json"))
    calibration = load_json(os.path.join(data_dir, "calibration_result.json"))
    manifest = load_json(os.path.join(data_dir, "import_manifest.json"))

    info = {
        "name": name,
        "path": seq_path,
        "method": detect_seq_type(name),
        "seq_num": extract_seq_number(name),
    }

    # --- Manifest info ---
    if manifest:
        seq_meta = manifest.get("sequence") or {}
        info["manifest_method"] = seq_meta.get("method", "?")
        info["manifest_volume"] = seq_meta.get("injection_volume")
        info["manifest_n_samples"] = len(manifest.get("samples", []))
        info["manifest_date"] = seq_meta.get("date", "?")

        # Delay from 0-INFO
        master_info = manifest.get("master_file") or {}
        info["masterfile"] = master_info.get("path", "?")
        zero_info = manifest.get("zero_info") or {}
        info["net_delay"] = zero_info.get("net_delay_min")

        # Volume from samples
        samples = manifest.get("samples", [])
        volumes = set()
        if isinstance(samples, list):
            for s in samples:
                for r in (s.get("replicas") or []):
                    v = (r.get("injection") or {}).get("inj_volume")
                    if v:
                        volumes.add(v)
        elif isinstance(samples, dict):
            for sdata in samples.values():
                reps = sdata.get("replicas") or {}
                for rv in (reps.values() if isinstance(reps, dict) else reps):
                    if isinstance(rv, dict):
                        v = (rv.get("injection") or {}).get("inj_volume")
                        if v:
                            volumes.add(v)
        info["volumes_found"] = sorted(volumes) if volumes else []
        info["volume_unique"] = volumes.pop() if len(volumes) == 1 else None

    # --- Calibration info ---
    if calibration:
        cal_data = calibration if isinstance(calibration, dict) else {}
        info["cal_success"] = cal_data.get("success", False)

        # KHP data
        khp_direct = cal_data.get("khp_data_direct") or {}
        khp_uib = cal_data.get("khp_data_uib") or {}

        # Extract KHP entries
        for label, khp in [("direct", khp_direct), ("uib", khp_uib)]:
            entries = khp.get("entries") or khp.get("khp_entries") or []
            if not entries and isinstance(khp, list):
                entries = khp

            khp_info = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                khp_info.append({
                    "conc_ppm": e.get("conc_ppm") or e.get("concentration_ppm", 0),
                    "area": e.get("area") or e.get("area_total", 0),
                    "rf": e.get("rf") or e.get("rf_mass_cal", 0),
                    "volume": e.get("volume") or e.get("inj_volume_uL", 0),
                    "valid": e.get("valid_for_calibration", True),
                    "quality_score": e.get("quality_score", 0),
                    "anomalies": [a.get("code") if isinstance(a, dict) else str(a)
                                  for a in (e.get("calibration_anomalies") or
                                           e.get("anomalies") or [])],
                })
            info[f"khp_{label}"] = khp_info

        # Calibration params used
        info["cal_rf_direct"] = cal_data.get("rf_mass_cal_direct") or cal_data.get("rf_mass_cal")
        info["cal_intercept_direct"] = cal_data.get("intercept_direct") or cal_data.get("intercept")
        info["cal_rf_uib"] = cal_data.get("rf_mass_cal_uib")
        info["cal_intercept_uib"] = cal_data.get("intercept_uib")
        info["cal_fingerprint"] = cal_data.get("calibration_fingerprint", "?")

        # Warnings
        info["cal_warnings"] = cal_data.get("warnings_structured") or []

    # --- Analysis info ---
    if analysis and analysis.get("success"):
        info["analysis_success"] = True
        info["analysis_method"] = analysis.get("method", "?")
        info["analysis_date"] = analysis.get("timestamp", "?")
        info["cal_fingerprint_analysis"] = analysis.get("calibration_fingerprint", "?")

        sg = analysis.get("samples_grouped") or {}
        info["n_samples"] = sum(1 for s in sg.values()
                               if s.get("sample_type") == "SAMPLE")
        info["n_khp"] = sum(1 for s in sg.values()
                           if s.get("sample_type") == "KHP")
        info["n_blank"] = sum(1 for s in sg.values()
                             if s.get("sample_type") == "BLANK")

        # Sample ppm stats
        ppms = []
        for sname, sdata in sg.items():
            if sdata.get("sample_type") != "SAMPLE":
                continue
            quant = sdata.get("quantification") or {}
            ppm = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
            if ppm and ppm > 0:
                ppms.append(ppm)
        info["ppm_median"] = float(np.median(ppms)) if ppms else None
        info["ppm_mean"] = float(np.mean(ppms)) if ppms else None
        info["ppm_range"] = (float(np.min(ppms)), float(np.max(ppms))) if ppms else None

        # Anomaly summary
        n_timeout = 0
        n_irreg = 0
        for sname, sdata in sg.items():
            for rd in (sdata.get("replicas") or {}).values():
                if not isinstance(rd, dict):
                    continue
                ti = rd.get("timeout_info") or {}
                if ti.get("n_timeouts", 0) > 0:
                    n_timeout += 1
                for a in rd.get("anomalies", []):
                    code = a.get("code") if isinstance(a, dict) else str(a)
                    if "IRREGULAR_TOP" in code:
                        n_irreg += 1
        info["n_timeout_reps"] = n_timeout
        info["n_irreg_reps"] = n_irreg

        info["_sg"] = sg  # keep for sample-level comparison
    else:
        info["analysis_success"] = False

    return info


def compare_pair(col_info, bp_info):
    """Compare a COL/BP pair in depth."""
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"PARELL: {col_info['name']} (COLUMN)  <->  {bp_info['name']} (BP)")
    report.append(f"{'='*80}")

    # --- Basic info ---
    report.append(f"\n--- INFO BASICA ---")
    for label, info in [("COL", col_info), ("BP", bp_info)]:
        report.append(f"  {label}: {info['name']}")
        report.append(f"    Volum manifest: {info.get('manifest_volume')} uL")
        report.append(f"    Volums trobats: {info.get('volumes_found', [])}")
        report.append(f"    Net delay: {info.get('net_delay')} min")
        report.append(f"    Mostres: {info.get('n_samples', '?')} SAMPLE, "
                      f"{info.get('n_khp', '?')} KHP, {info.get('n_blank', '?')} BLANK")
        report.append(f"    Timeouts: {info.get('n_timeout_reps', 0)} reps, "
                      f"Irregulars: {info.get('n_irreg_reps', 0)} reps")

    # --- Calibration ---
    report.append(f"\n--- CALIBRACIO ---")
    for label, info in [("COL", col_info), ("BP", bp_info)]:
        rf_d = info.get("cal_rf_direct")
        int_d = info.get("cal_intercept_direct")
        rf_u = info.get("cal_rf_uib")
        int_u = info.get("cal_intercept_uib")
        fp = info.get("cal_fingerprint", "?")
        fp_a = info.get("cal_fingerprint_analysis", "?")
        report.append(f"  {label}: RF_direct={rf_d}  intercept_direct={int_d}")
        report.append(f"       RF_uib={rf_u}  intercept_uib={int_u}")
        report.append(f"       fingerprint: cal={fp}  analysis={fp_a}")
        stale = "SI" if fp != fp_a and fp != "?" and fp_a != "?" else "no"
        report.append(f"       stale: {stale}")

    # --- KHP ---
    report.append(f"\n--- KHP ---")
    for label, info in [("COL", col_info), ("BP", bp_info)]:
        for signal in ["direct", "uib"]:
            khps = info.get(f"khp_{signal}", [])
            if not khps:
                report.append(f"  {label} {signal}: (sense KHP)")
                continue
            report.append(f"  {label} {signal}: {len(khps)} entrades")
            for k in khps:
                anoms = ", ".join(k["anomalies"]) if k["anomalies"] else "OK"
                valid = "V" if k["valid"] else "X"
                report.append(
                    f"    [{valid}] conc={k['conc_ppm']}ppm  area={k['area']:.0f}  "
                    f"RF={k['rf']:.0f}  vol={k['volume']}uL  Q={k['quality_score']}  {anoms}")

            # KHP stats
            valid_khps = [k for k in khps if k["valid"] and k["area"] > 0]
            if valid_khps:
                rfs = [k["rf"] for k in valid_khps if k["rf"] > 0]
                areas = [k["area"] for k in valid_khps]
                if rfs:
                    report.append(f"    RF: mean={np.mean(rfs):.0f}  SD={np.std(rfs):.0f}  "
                                  f"CV={np.std(rfs)/np.mean(rfs)*100:.1f}%")
                report.append(f"    Area: mean={np.mean(areas):.0f}  SD={np.std(areas):.0f}")

    # --- Sample-level comparison ---
    col_sg = col_info.get("_sg") or {}
    bp_sg = bp_info.get("_sg") or {}

    # Find common samples
    col_samples = {n: s for n, s in col_sg.items() if s.get("sample_type") == "SAMPLE"}
    bp_samples = {n: s for n, s in bp_sg.items() if s.get("sample_type") == "SAMPLE"}
    common = sorted(set(col_samples.keys()) & set(bp_samples.keys()))

    report.append(f"\n--- MOSTRES COMUNES: {len(common)} ---")
    if not common:
        report.append("  (cap mostra comuna)")
        return report

    ratios = []
    ratios_a254 = []
    rows_detail = []

    for name in common:
        cs = col_samples[name]
        bs = bp_samples[name]

        def _get_ppm(sdata):
            q = sdata.get("quantification") or {}
            return q.get("concentration_ppm_direct") or q.get("concentration_ppm")

        def _get_area_doc(sdata):
            sel = (sdata.get("selected") or {}).get("doc", "1")
            rep = (sdata.get("replicas") or {}).get(sel, {})
            if sel == "comp":
                for rv in (sdata.get("replicas") or {}).values():
                    if isinstance(rv, dict) and rv.get("timeout_composition"):
                        rep = rv
                        break
            if not isinstance(rep, dict):
                return 0, 0, {}
            areas = (rep.get("areas") or {}).get("DOC") or {}
            ti = rep.get("timeout_info") or {}
            return areas.get("total", 0), ti.get("n_timeouts", 0), areas

        def _get_area_a254(sdata):
            sel = (sdata.get("selected") or {}).get("dad", (sdata.get("selected") or {}).get("doc", "1"))
            rep = (sdata.get("replicas") or {}).get(sel, {})
            if not isinstance(rep, dict):
                return 0
            return ((rep.get("areas") or {}).get("A254") or {}).get("total", 0)

        def _get_anomalies(sdata):
            anoms = set()
            for rd in (sdata.get("replicas") or {}).values():
                if not isinstance(rd, dict):
                    continue
                for a in rd.get("anomalies", []):
                    code = a.get("code") if isinstance(a, dict) else str(a)
                    anoms.add(code)
            return anoms

        ppm_col = _get_ppm(cs)
        ppm_bp = _get_ppm(bs)
        area_col, to_col, fracs_col = _get_area_doc(cs)
        area_bp, to_bp, fracs_bp = _get_area_doc(bs)
        a254_col = _get_area_a254(cs)
        a254_bp = _get_area_a254(bs)
        anom_col = _get_anomalies(cs)
        anom_bp = _get_anomalies(bs)

        ratio_ppm = ppm_bp / ppm_col if ppm_col and ppm_bp and ppm_col > 0 else None
        ratio_area = area_bp / area_col if area_col and area_bp and area_col > 0 else None

        vol_col = col_info.get("volume_unique") or col_info.get("manifest_volume")
        vol_bp = bp_info.get("volume_unique") or bp_info.get("manifest_volume")
        vol_ratio = vol_bp / vol_col if vol_col and vol_bp and vol_col > 0 else None

        ratio_a254 = a254_bp / a254_col if a254_col and a254_bp and a254_col > 0 else None
        ratio_a254_norm = ratio_a254 / vol_ratio if ratio_a254 and vol_ratio else None

        if ratio_ppm:
            ratios.append(ratio_ppm)
        if ratio_a254_norm:
            ratios_a254.append(ratio_a254_norm)

        flags = []
        if ratio_ppm and (ratio_ppm < 0.85 or ratio_ppm > 1.15):
            flags.append("PPM_OUT")
        if to_col > 0:
            flags.append(f"TO_C({to_col})")
        if to_bp > 0:
            flags.append(f"TO_B({to_bp})")
        irreg_col = any("IRREGULAR_TOP" in a for a in anom_col)
        irreg_bp = any("IRREGULAR_TOP" in a for a in anom_bp)
        if irreg_col:
            flags.append("IRR_C")
        if irreg_bp:
            flags.append("IRR_B")

        rows_detail.append({
            "name": name, "ppm_col": ppm_col, "ppm_bp": ppm_bp,
            "ratio_ppm": ratio_ppm,
            "area_col": area_col, "area_bp": area_bp, "ratio_area": ratio_area,
            "a254_col": a254_col, "a254_bp": a254_bp,
            "ratio_a254_norm": ratio_a254_norm,
            "to_col": to_col, "to_bp": to_bp,
            "flags": " ".join(flags),
        })

    # Summary
    if ratios:
        ra = np.array(ratios)
        n_out = sum(1 for r in ratios if r < 0.85 or r > 1.15)
        report.append(f"\n  Ratio ppm: med={np.median(ra):.3f}  mean={np.mean(ra):.3f}  "
                      f"SD={np.std(ra):.3f}  outliers={n_out}/{len(ratios)}")

    if ratios_a254:
        ra254 = np.array(ratios_a254)
        report.append(f"  A254 norm: med={np.median(ra254):.3f}  mean={np.mean(ra254):.3f}  "
                      f"SD={np.std(ra254):.3f}")

    # Detail table
    report.append(f"\n  {'Mostra':20s} {'ppm_C':>7s} {'ppm_B':>7s} {'ratio':>6s} "
                  f"{'area_C':>7s} {'area_B':>7s} {'A254n':>6s} {'flags'}")
    report.append(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*20}")
    for r in rows_detail:
        ppm_c = f"{r['ppm_col']:.2f}" if r['ppm_col'] else "  -  "
        ppm_b = f"{r['ppm_bp']:.2f}" if r['ppm_bp'] else "  -  "
        ratio = f"{r['ratio_ppm']:.3f}" if r['ratio_ppm'] else "  -  "
        area_c = f"{r['area_col']:.0f}" if r['area_col'] else "  -  "
        area_b = f"{r['area_bp']:.0f}" if r['area_bp'] else "  -  "
        a254n = f"{r['ratio_a254_norm']:.3f}" if r['ratio_a254_norm'] else "  -  "
        report.append(f"  {r['name']:20s} {ppm_c:>7s} {ppm_b:>7s} {ratio:>6s} "
                      f"{area_c:>7s} {area_b:>7s} {a254n:>6s} {r['flags']}")

    # Diagnosis
    report.append(f"\n--- DIAGNOSI ---")
    if ratios:
        med = np.median(ratios)
        if 0.90 <= med <= 1.10:
            report.append(f"  Concordanca: BONA (med={med:.3f})")
        elif 0.80 <= med <= 1.20:
            biaix = "BP dona mes" if med > 1 else "BP dona menys"
            report.append(f"  Concordanca: ACCEPTABLE amb biaix ({biaix}, med={med:.3f})")
        else:
            biaix = "BP dona mes" if med > 1 else "BP dona menys"
            report.append(f"  Concordanca: DEFICIENT ({biaix}, med={med:.3f})")

        # Check if bias is systematic
        sd = np.std(ratios)
        if sd < 0.10 and abs(med - 1.0) > 0.10:
            report.append(f"  Biaix SISTEMATIC (SD={sd:.3f} < 0.10, |biaix|={abs(med-1):.3f} > 0.10)")
            report.append(f"  -> Probable causa: calibracio, volum, o delay diferent")
        elif sd > 0.20:
            report.append(f"  Alta VARIABILITAT (SD={sd:.3f} > 0.20)")
            report.append(f"  -> Probable causa: timeouts/irregulars afecten algunes mostres")

        # KHP cross-check
        col_khps = col_info.get("khp_direct", [])
        bp_khps = bp_info.get("khp_direct", [])
        col_valid = [k for k in col_khps if k["valid"] and k["rf"] > 0]
        bp_valid = [k for k in bp_khps if k["valid"] and k["rf"] > 0]
        if col_valid and bp_valid:
            rf_col = np.mean([k["rf"] for k in col_valid])
            rf_bp = np.mean([k["rf"] for k in bp_valid])
            rf_ratio = rf_bp / rf_col if rf_col > 0 else None
            if rf_ratio:
                report.append(f"  KHP RF: COL={rf_col:.0f}  BP={rf_bp:.0f}  ratio={rf_ratio:.3f}")
                if abs(rf_ratio - 1.0) > 0.15:
                    report.append(f"  -> KHP confirma biaix en la calibracio!")

    return report


def main():
    folders = get_data_folders()
    print(f"Data folders: {folders}")

    # Scan all SEQs
    all_seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            jp = os.path.join(full, "CHECK", "data", "analysis_result.json")
            if os.path.exists(jp):
                all_seqs.append(full)

    print(f"SEQs amb analisi: {len(all_seqs)}")

    # Load all SEQ info
    seq_infos = {}
    for sp in all_seqs:
        info = get_seq_info(sp)
        seq_infos[info["name"]] = info

    col_infos = {n: i for n, i in seq_infos.items() if i["method"] == "COLUMN"}
    bp_infos = {n: i for n, i in seq_infos.items() if i["method"] == "BP"}
    print(f"COLUMN: {len(col_infos)}, BP: {len(bp_infos)}")

    # Match pairs by sample name overlap
    pairs = []
    used_bp = set()
    for col_name, col_info in sorted(col_infos.items()):
        col_sg = col_info.get("_sg") or {}
        col_samples = {n for n, s in col_sg.items() if s.get("sample_type") == "SAMPLE"}
        if not col_samples:
            continue

        best_bp = None
        best_overlap = 0
        for bp_name, bp_info in bp_infos.items():
            if bp_name in used_bp:
                continue
            bp_sg = bp_info.get("_sg") or {}
            bp_samples = {n for n, s in bp_sg.items() if s.get("sample_type") == "SAMPLE"}
            overlap = len(col_samples & bp_samples)
            if overlap > best_overlap:
                best_overlap = overlap
                best_bp = bp_name

        if best_bp and best_overlap > 0:
            used_bp.add(best_bp)
            pairs.append((col_name, best_bp, best_overlap))

    print(f"\nParells detectats (per nom de mostra): {len(pairs)}")
    for cn, bn, ov in pairs:
        print(f"  {cn:15s} <-> {bn:15s}  ({ov} mostres comunes)")

    # Deep comparison per pair
    full_report = []
    full_report.append(f"DIAGNOSTIC PROFUND COL vs BP")
    full_report.append(f"Data: {len(all_seqs)} SEQs, {len(pairs)} parells")
    full_report.append(f"")

    for col_name, bp_name, _ in pairs:
        pair_report = compare_pair(seq_infos[col_name], seq_infos[bp_name])
        full_report.extend(pair_report)

    # Unmatched SEQs
    matched_col = {cn for cn, _, _ in pairs}
    matched_bp = {bn for _, bn, _ in pairs}
    unmatched_col = set(col_infos.keys()) - matched_col
    unmatched_bp = set(bp_infos.keys()) - matched_bp

    if unmatched_col or unmatched_bp:
        full_report.append(f"\n{'='*80}")
        full_report.append(f"SEQS SENSE PARELLA")
        full_report.append(f"{'='*80}")
        if unmatched_col:
            full_report.append(f"  COLUMN sense BP: {sorted(unmatched_col)}")
        if unmatched_bp:
            full_report.append(f"  BP sense COLUMN: {sorted(unmatched_bp)}")

    # Save report
    out = Path("_results/diag_col_bp")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "deep_diagnostic.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_report))

    # Also print
    for line in full_report:
        print(line)

    print(f"\nReport: {report_path}")
    print("Fet!")


if __name__ == "__main__":
    main()
