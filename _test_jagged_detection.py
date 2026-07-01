"""
Script de test per investigar la detecció de pics jagged (irregular_top)
i els límits d'integració a analizar_khp_data().

Carrega les dades KHP d'una SEQ_CAL i compara:
1. Bounds de detect_main_peak (amb pre-repair intern)
2. Bounds de all_peaks (scipy left_bases/right_bases)
3. Bounds de find_peak_boundaries directe
4. Detecció irregular_top sobre cada segment

Objectiu: entendre per què el bucle all_peaks sobreescriu bounds correctes
i per què detect_irregular_top no detecta el pic jagged d'1 ppm.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Afegir HPSEC al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpsec_core import (
    detect_main_peak, detect_all_peaks, find_peak_boundaries,
    detect_irregular_top, detect_peak_anomaly, calc_top_smoothness,
    get_baseline_stats,
)
from hpsec_calibrate import DEFAULT_CONFIG
from hpsec_import import import_from_manifest, load_manifest, extract_doc_from_masterfile


def load_khp_data(seq_path):
    """Carrega les dades KHP d'una SEQ."""
    print(f"Carregant {seq_path}...")
    imported = import_from_manifest(seq_path, load_data=True)
    if not imported or not imported.get("success"):
        print(f"ERROR: no s'ha pogut carregar {seq_path}")
        return None

    method = imported.get("method", "COLUMN")
    samples = imported.get("samples", {})
    khp_list = imported.get("khp_samples", [])

    results = []
    for sample_name in khp_list:
        sample = samples.get(sample_name, {})
        replicas = sample.get("replicas", {})
        for rep_id, rep_data in replicas.items():
            for signal_type in ["direct", "uib"]:
                sig = rep_data.get(signal_type)
                if sig is None:
                    continue
                t = sig.get("t")
                y_net = sig.get("y_net")
                if t is None or y_net is None:
                    continue

                # Extreure concentració del nom
                from hpsec_calibrate import extract_khp_conc
                conc = extract_khp_conc(sample_name)

                results.append({
                    "name": sample_name,
                    "replica": rep_id,
                    "signal": signal_type,
                    "conc": conc,
                    "t": np.asarray(t),
                    "y_net": np.asarray(y_net),
                    "method": method,
                })

    print(f"  {len(results)} senyals KHP carregats")
    return results


def analyze_peak_detection(entry):
    """Analitza la detecció de pic per una entrada KHP."""
    t = entry["t"]
    y = entry["y_net"]
    method = entry["method"]
    is_bp = method == "BP"
    label = f"{entry['name']}_R{entry['replica']}_{entry['signal']}"
    conc = entry["conc"]

    config = DEFAULT_CONFIG.copy()

    # 1. detect_main_peak (amb pre-repair intern)
    peak_info = detect_main_peak(t, y, config["peak_min_prominence_pct"], is_bp=is_bp)
    if not peak_info.get("valid"):
        print(f"  {label} conc={conc}: NO PEAK DETECTED")
        return

    dmp_left = peak_info["left_idx"]
    dmp_right = peak_info["right_idx"]
    dmp_width = t[dmp_right] - t[dmp_left]
    dmp_area = peak_info["area"]
    peak_idx = peak_info["peak_idx"]

    # 2. all_peaks (scipy bounds)
    all_peaks = detect_all_peaks(t, y, config["peak_min_prominence_pct"])
    ap_left, ap_right = dmp_left, dmp_right  # default si no es troba
    ap_match = False
    for pk in all_peaks:
        if pk['idx'] == peak_idx or abs(pk['t'] - peak_info['t_max']) < 0.1:
            ap_left = pk['left_idx']
            ap_right = pk['right_idx']
            ap_match = True
            break
    ap_width = t[ap_right] - t[ap_left]
    ap_area = float(np.trapezoid(y[ap_left:ap_right+1], t[ap_left:ap_right+1]))

    # 3. find_peak_boundaries directe (sense pre-repair)
    bl_stats = get_baseline_stats(t, y, mode=method)
    bl_level = bl_stats.get("mean", 0)
    fpb_left, fpb_right = find_peak_boundaries(t, y, peak_idx, bl_level, is_bp=is_bp)
    fpb_width = t[fpb_right] - t[fpb_left]
    fpb_area = float(np.trapezoid(y[fpb_left:fpb_right+1], t[fpb_left:fpb_right+1]))

    # 4. Detecció irregular_top sobre cada segment
    # a) Segment detect_main_peak
    seg_dmp = y[dmp_left:dmp_right+1]
    t_seg_dmp = t[dmp_left:dmp_right+1]
    anomaly_dmp = detect_peak_anomaly(t_seg_dmp, seg_dmp)

    # b) Segment all_peaks
    seg_ap = y[ap_left:ap_right+1]
    t_seg_ap = t[ap_left:ap_right+1]
    anomaly_ap = detect_peak_anomaly(t_seg_ap, seg_ap)

    # c) Segment find_peak_boundaries
    seg_fpb = y[fpb_left:fpb_right+1]
    t_seg_fpb = t[fpb_left:fpb_right+1]
    anomaly_fpb = detect_peak_anomaly(t_seg_fpb, seg_fpb)

    # d) Finestra ±5 min al voltant del pic (com fa detect_main_peak internament)
    half_w = 3.0 if is_bp else 5.0
    win_mask = (t >= t[peak_idx] - half_w) & (t <= t[peak_idx] + half_w)
    t_win = t[win_mask]
    y_win = y[win_mask]
    anomaly_win = detect_peak_anomaly(t_win, y_win)
    irr_win = detect_irregular_top(t_win, y_win)

    # Imprimir resultats
    print(f"\n{'='*70}")
    print(f"  {label}  conc={conc}ppm  t_peak={t[peak_idx]:.2f} min")
    print(f"{'='*70}")
    print(f"  {'Mètode':<25} {'left':>6} {'right':>6} {'width':>6} {'area':>8} {'seg_len':>7} "
          f"{'irr_top':>7} {'n_val':>5} {'depth%':>6} {'smooth':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*6}")

    def _row(name, li, ri, area, anomaly):
        irr = anomaly.get('irregular_top_info', {})
        print(f"  {name:<25} {t[li]:>6.2f} {t[ri]:>6.2f} {t[ri]-t[li]:>6.2f} {area:>8.1f} "
              f"{ri-li+1:>7} {anomaly.get('is_irregular_top', False)!s:>7} "
              f"{irr.get('n_valleys', 0):>5} {irr.get('max_depth', 0)*100:>5.1f}% "
              f"{anomaly.get('smoothness', 0):>6.1f}")

    _row("detect_main_peak", dmp_left, dmp_right, dmp_area, anomaly_dmp)
    _row("all_peaks (scipy)", ap_left, ap_right, ap_area, anomaly_ap)
    _row("find_peak_boundaries", fpb_left, fpb_right, fpb_area, anomaly_fpb)

    # Finestra àmplia: imprimir amb índexs de la finestra
    irr_w = anomaly_win.get('irregular_top_info', {})
    print(f"  {'finestra ±5min':<25} {t_win[0]:>6.2f} {t_win[-1]:>6.2f} "
          f"{t_win[-1]-t_win[0]:>6.2f} {'---':>8} {len(y_win):>7} "
          f"{anomaly_win.get('is_irregular_top', False)!s:>7} "
          f"{irr_w.get('n_valleys', 0):>5} {irr_w.get('max_depth', 0)*100:>5.1f}% "
          f"{anomaly_win.get('smoothness', 0):>6.1f}")

    # Si detect_main_peak va reparar internament
    if peak_info.get("area_repaired") is not None:
        print(f"  ** detect_main_peak va reparar internament: area_orig={dmp_area:.1f}, "
              f"area_repaired={peak_info['area_repaired']:.1f}")

    return {
        "label": label,
        "conc": conc,
        "dmp_width": dmp_width,
        "ap_width": ap_width,
        "fpb_width": fpb_width,
        "anomaly_dmp": anomaly_dmp,
        "anomaly_ap": anomaly_ap,
        "anomaly_win": anomaly_win,
    }


def main():
    # Buscar SEQ_CAL al REGISTRY
    registry = Path(r"C:\Users\Lequia\Desktop\Dades3")
    seq_cal_dirs = sorted([d for d in registry.iterdir()
                           if d.is_dir() and "SEQ" in d.name.upper()])

    # Prioritzar SEQ_CAL
    cal_dirs = [d for d in seq_cal_dirs if "CAL" in d.name.upper()]
    if cal_dirs:
        seq_path = str(cal_dirs[-1])  # Última SEQ_CAL
    elif seq_cal_dirs:
        seq_path = str(seq_cal_dirs[-1])
    else:
        print("No s'han trobat SEQs al REGISTRY")
        return

    # Permetre sobreescriure per argument
    if len(sys.argv) > 1:
        seq_path = sys.argv[1]

    print(f"SEQ: {seq_path}")
    khp_data = load_khp_data(seq_path)
    if not khp_data:
        return

    # Filtrar només Direct (el que ens interessa principalment)
    direct_data = [e for e in khp_data if e["signal"] == "direct"]
    uib_data = [e for e in khp_data if e["signal"] == "uib"]

    print(f"\n{'#'*70}")
    print(f"  ANÀLISI DOC DIRECT ({len(direct_data)} senyals)")
    print(f"{'#'*70}")

    results = []
    for entry in sorted(direct_data, key=lambda e: (e["conc"], e["replica"])):
        r = analyze_peak_detection(entry)
        if r:
            results.append(r)

    # Resum comparatiu
    if results:
        print(f"\n{'#'*70}")
        print(f"  RESUM: diferències boundaries")
        print(f"{'#'*70}")
        print(f"  {'Label':<30} {'conc':>5} {'DMP':>6} {'AP':>6} {'FPB':>6} "
              f"{'AP/DMP':>6} {'irr_win':>7}")
        for r in results:
            ratio = r["ap_width"] / r["dmp_width"] if r["dmp_width"] > 0 else 0
            irr_win = r["anomaly_win"].get("is_irregular_top", False)
            print(f"  {r['label']:<30} {r['conc']:>5.2f} "
                  f"{r['dmp_width']:>6.2f} {r['ap_width']:>6.2f} {r['fpb_width']:>6.2f} "
                  f"{ratio:>6.2f} {irr_win!s:>7}")


if __name__ == "__main__":
    main()
