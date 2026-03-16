#!/usr/bin/env python3
"""
KHP Re-integration: UIB signal.

Equivalent a khp_reintegrate_doc.py pero per senyal UIB.
Llegeix CSVs UIB directament (no MasterFile 2-TOC).

1. Llegeix manifest JSON (baseline, volum, CSV path UIB)
2. Llegeix CSV UIB (2 columnes: time, DOC)
3. Downsample a cadencia DOC (dt=0.0667 min)
4. Aplica baseline del manifest (y_net = y - baseline)
5. Integra amb detect_main_peak() sobre y_net
6. Opcionalment guia la seleccio de pic amb DAD 254nm
7. Genera plots i scatter area vs ug_DOC

Output: REGISTRY/review_uib/
"""

import os
import sys
import json
import csv
import logging
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hpsec_import import extract_khp_conc
from hpsec_core import (
    detect_main_peak, detect_all_peaks, find_peak_boundaries,
    downsample_to_cadence, DOC_TARGET_DT_MIN, get_baseline_value,
)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hpsec_config.json")


def extract_conc_from_calname(name):
    """Extreu concentracio (ppm) d'un nom de mostra en SEQ _CAL."""
    match = re.search(r'(\d+\.?\d*)\s*(ppm|ppb)', str(name), re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "PPB":
        value /= 1000.0
    return value


def extract_volume_from_calname(name):
    """Extreu volum (uL) d'un nom de mostra si present."""
    match = re.search(r'(\d+)\s*ul', str(name), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_data_folder():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("paths", {}).get("data_folder", "")
    except Exception:
        return ""


def parse_uib_sensitivity(raw):
    """Parseja sensibilitat UIB del manifest (format '1-700' o numeric)."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        m = re.search(r'(\d+)\s*$', str(raw))
        return float(m.group(1)) if m else None


def load_uib_csv(csv_path):
    """Llegeix CSV UIB (2 columnes: time, DOC). Retorna t, y arrays.
    Els CSVs UIB solen ser UTF-16 LE amb BOM, separats per tab."""
    try:
        # Intentar encodings habituals dels CSVs UIB (Sievers M9e)
        for enc in ['utf-16', 'utf-16-le', 'utf-8', 'latin-1']:
            try:
                df = pd.read_csv(csv_path, header=None, encoding=enc, sep='\t')
                if df.shape[1] >= 2:
                    break
                # Si nomes 1 col amb tab, provar comma
                df = pd.read_csv(csv_path, header=None, encoding=enc, sep=',')
                if df.shape[1] >= 2:
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            logger.warning(f"No encoding valid per {csv_path}")
            return None, None

        if df.shape[1] < 2:
            return None, None
        t = df.iloc[:, 0].values.astype(float)
        y = df.iloc[:, 1].values.astype(float)
        # Netejar NaN
        mask = np.isfinite(t) & np.isfinite(y)
        t, y = t[mask], y[mask]
        if len(t) < 10:
            return None, None
        return t, y
    except Exception as e:
        logger.warning(f"Error llegint UIB CSV {csv_path}: {e}")
        return None, None


def load_dad_254(dad_path):
    """Llegeix Export3D CSV i retorna t, y_254 (columna 254nm)."""
    try:
        # Export3D: comma-separated, UTF-16 o UTF-8
        df = None
        for enc in ['utf-16', 'utf-8']:
            try:
                df = pd.read_csv(dad_path, sep=',', encoding=enc, engine='python')
                if df.shape[1] >= 2:
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if df is None or df.shape[1] < 2:
            return None, None
        time_col = [c for c in df.columns if 'time' in str(c).lower()]
        if time_col:
            t = df[time_col[0]].values
        else:
            # Primer columna es temps (Unnamed: 0 o similar)
            t = df.iloc[:, 0].values
        # Buscar columna 254nm
        col_254 = None
        for c in df.columns:
            cs = str(c).strip()
            if cs == '254' or cs == '254.0':
                col_254 = c
                break
            try:
                if abs(float(cs) - 254) < 2:
                    col_254 = c
                    break
            except ValueError:
                continue
        if col_254 is None:
            return None, None
        y = df[col_254].values
        return t, y
    except Exception as e:
        logger.warning(f"Error llegint DAD {dad_path}: {e}")
        return None, None


def plot_chromatogram(t, y_raw, y_net, peak, baseline, t_254_max,
                      seq_name, khp_name, conc, mode, volume, flags, out_path):
    """Genera plot del cromatograma UIB amb baseline, limits i referencia 254nm."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, y_raw, 'b-', linewidth=0.5, alpha=0.4, label='UIB raw')
    ax.plot(t, y_net, 'b-', linewidth=0.8, label='UIB net (y-baseline)')

    if peak.get("valid", False):
        t_max = peak["t_max"]
        left_idx = peak["left_idx"]
        right_idx = peak["right_idx"]
        area = peak["area"]

        ax.fill_between(t[left_idx:right_idx+1], 0,
                        y_net[left_idx:right_idx+1], alpha=0.3, color='green',
                        label=f'Area={area:.1f}')

        ax.axvline(t[left_idx], color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(t[right_idx], color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(t_max, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.plot(t_max, y_net[peak["peak_idx"]], 'rv', markersize=8)

    if t_254_max is not None and t_254_max > 0:
        ax.axvline(t_254_max, color='purple', linestyle=':', linewidth=1.2, alpha=0.7,
                   label=f'254nm={t_254_max:.2f}')

    ax.axhline(0, color='grey', linestyle='-', linewidth=0.3)

    flag_str = ", ".join(flags) if flags else "OK"
    color = 'red' if any(f.startswith("!") for f in flags) else ('orange' if flags else 'green')
    ax.set_title(f"{seq_name} | {khp_name} | {conc:g} ppm | {mode} | v={volume}uL | "
                 f"BL={baseline:.0f} | [{flag_str}]",
                 fontsize=9, color=color)
    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("DOC UIB net (ppb)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    data_folder = get_data_folder()
    if not data_folder:
        print("ERROR: data_folder no configurat")
        return

    registry_dir = os.path.join(data_folder, "REGISTRY")
    review_dir = os.path.join(registry_dir, "review_uib")
    chromo_dir = os.path.join(review_dir, "chromatograms")
    os.makedirs(chromo_dir, exist_ok=True)

    # Trobar totes les SEQs amb manifest
    seq_dirs = []
    for item in os.listdir(data_folder):
        item_path = os.path.join(data_folder, item)
        if os.path.isdir(item_path) and ("_SEQ" in item):
            manifest_path = os.path.join(item_path, "CHECK", "data", "import_manifest.json")
            if os.path.exists(manifest_path):
                seq_dirs.append((item, item_path, manifest_path))

    print(f"Trobades {len(seq_dirs)} SEQs amb manifest")

    results = []
    errors = []

    for seq_name, seq_path, manifest_path in sorted(seq_dirs):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            errors.append(f"{seq_name}: error manifest: {e}")
            continue

        method = (manifest.get("sequence", {}).get("method", "") or "COLUMN").upper()
        if not method or method == "?":
            method = "COLUMN"

        uib_sensitivity = parse_uib_sensitivity(
            manifest.get("sequence", {}).get("uib_sensitivity"))

        samples = manifest.get("samples", [])
        khp_samples = [s for s in samples if s.get("type", "").upper() == "KHP"]

        if not khp_samples:
            continue

        mode = method
        is_bp = (mode == "BP")

        for sample in khp_samples:
            sample_name = sample.get("name", "KHP")
            conc = extract_khp_conc(sample_name)
            if conc <= 0:
                conc = extract_conc_from_calname(sample_name)
            if conc <= 0:
                conc = sample.get("conc_ppm", 0) or 0
            if conc <= 0:
                continue

            name_volume = extract_volume_from_calname(sample_name)

            reps = sample.get("replicas", [])
            if not reps:
                continue

            for rep_idx, rep in enumerate(reps):
                rep_label = f"R{rep_idx+1}"

                # UIB CSV
                uib_info = rep.get("uib", {})
                if not isinstance(uib_info, dict) or not uib_info.get("file"):
                    continue  # Sense UIB

                uib_file = uib_info["file"]
                # Buscar CSV al directori de la SEQ (pot estar a rawdata/)
                uib_csv = None
                for candidate in [
                    os.path.join(seq_path, uib_file),
                    os.path.join(seq_path, "rawdata", uib_file),
                ]:
                    if os.path.exists(candidate):
                        uib_csv = candidate
                        break

                if not uib_csv:
                    # Buscar recursivament
                    for root, dirs, files in os.walk(seq_path):
                        if uib_file in files:
                            uib_csv = os.path.join(root, uib_file)
                            break

                if not uib_csv:
                    errors.append(f"{seq_name}/{sample_name}/{rep_label}: UIB CSV {uib_file} not found")
                    continue

                # Volum
                if name_volume:
                    volume = name_volume
                else:
                    volume = rep.get("injection", {}).get("inj_volume", 0)
                    if not volume or volume <= 0:
                        volume = 100 if is_bp else 400

                # Llegir UIB
                t_raw, y_raw = load_uib_csv(uib_csv)
                if t_raw is None:
                    errors.append(f"{seq_name}/{sample_name}/{rep_label}: error UIB CSV")
                    continue

                # Downsample
                dt_raw = float(np.median(np.diff(t_raw)))
                if dt_raw < DOC_TARGET_DT_MIN * 0.8:
                    t_ds, y_ds = downsample_to_cadence(t_raw, y_raw, DOC_TARGET_DT_MIN)
                else:
                    t_ds, y_ds = t_raw, y_raw

                # Baseline
                manifest_baseline = uib_info.get("baseline")
                if manifest_baseline is not None and manifest_baseline > 0:
                    baseline = float(manifest_baseline)
                else:
                    baseline = get_baseline_value(t_ds, y_ds, mode=mode)

                y_net = np.maximum(y_ds - baseline, 0)

                # DAD 254nm (per guiar seleccio)
                t_254_max = None
                dad_info = rep.get("dad", {})
                if isinstance(dad_info, dict) and dad_info.get("file"):
                    dad_file = dad_info["file"]
                    dad_csv = None
                    for candidate in [
                        os.path.join(seq_path, dad_file),
                        os.path.join(seq_path, "rawdata", dad_file),
                    ]:
                        if os.path.exists(candidate):
                            dad_csv = candidate
                            break
                    if not dad_csv:
                        for root, dirs, files in os.walk(seq_path):
                            if dad_file in files:
                                dad_csv = os.path.join(root, dad_file)
                                break

                    if dad_csv:
                        t_dad, y_254 = load_dad_254(dad_csv)
                        if t_dad is not None and len(t_dad) > 0:
                            idx_max = int(np.argmax(y_254))
                            t_254_max = float(t_dad[idx_max])

                # Integrar
                peak = detect_main_peak(t_ds, y_net, is_bp=is_bp)

                if not peak.get("valid", False):
                    errors.append(f"{seq_name}/{sample_name}/{rep_label}: no peak")
                    continue

                peak_idx = peak["peak_idx"]
                t_max = peak["t_max"]
                left_idx = peak["left_idx"]
                right_idx = peak["right_idx"]
                area = peak["area"]
                fallback = peak.get("fallback", False)

                # CR local (+-5 min)
                cr_window = 5.0 if not is_bp else 3.0
                cr_mask = (t_ds >= t_max - cr_window) & (t_ds <= t_max + cr_window)
                local_area = float(np.trapz(y_net[cr_mask], t_ds[cr_mask])) if cr_mask.sum() > 1 else 0
                cr = area / local_area if local_area > 0 else 0

                # Shift UIB vs 254nm
                shift_254 = (t_max - t_254_max) if t_254_max else None

                # Saturacio UIB (per forma Gaussiana)
                from hpsec_core import detect_peak_clipping
                _clip = detect_peak_clipping(t_ds, y_net)
                saturated = _clip["is_saturated"]

                # Flags
                flags_error = []
                flags_warn = []

                if saturated:
                    flags_error.append("!SAT")
                if is_bp and t_max > 12.0:
                    flags_error.append(f"!T_RET({t_max:.1f})")
                elif not is_bp and (t_max < 15 or t_max > 30):
                    flags_error.append(f"!T_RET({t_max:.1f})")
                if area <= 0:
                    flags_error.append("!AREA_ZERO")
                if fallback:
                    flags_warn.append("FALLBACK")
                if cr < 0.50:
                    flags_error.append(f"!MULTI_PK({cr:.0%})")
                elif cr < 0.70:
                    flags_warn.append(f"MULTI_PK({cr:.0%})")
                elif cr < 0.90:
                    flags_warn.append(f"MULTI_PK_M({cr:.0%})")
                if t_254_max and shift_254 is not None and abs(shift_254) > 0.5:
                    flags_warn.append(f"SHIFT254({shift_254:.2f})")

                all_flags = flags_error + flags_warn

                ug_doc = conc * volume / 1000.0
                rf_mass = area / ug_doc if ug_doc > 0 else 0

                t_start = float(t_ds[left_idx])
                t_end = float(t_ds[right_idx])

                row = {
                    "seq": seq_name,
                    "sample": sample_name,
                    "replica": rep_label,
                    "mode": mode,
                    "conc_ppm": conc,
                    "volume_uL": volume,
                    "ug_doc": round(ug_doc, 4),
                    "area": round(area, 1),
                    "t_max": round(t_max, 3),
                    "t_start": round(t_start, 3),
                    "t_end": round(t_end, 3),
                    "width": round(t_end - t_start, 3),
                    "baseline": round(baseline, 1),
                    "rf_mass": round(rf_mass, 1),
                    "cr": round(cr, 3),
                    "t_254": round(t_254_max, 3) if t_254_max else "",
                    "shift_254": round(shift_254, 3) if shift_254 is not None else "",
                    "sensitivity": uib_sensitivity or "",
                    "saturated": "Y" if saturated else "N",
                    "n_raw": len(t_raw),
                    "n_ds": len(t_ds),
                    "dt_raw": round(dt_raw, 5),
                    "fallback": "Y" if fallback else "N",
                    "flags": "|".join(all_flags) if all_flags else "",
                    "ok": "N" if flags_error else "Y",
                }
                results.append(row)

                # Plot
                plot_name = f"{seq_name}_{sample_name}_{rep_label}_UIB.png"
                plot_path = os.path.join(chromo_dir, plot_name)
                plot_chromatogram(t_ds, y_ds, y_net, peak, baseline, t_254_max,
                                  seq_name, sample_name, conc, mode, volume,
                                  all_flags, plot_path)

                status = "OK" if not flags_error else f"ERR:{','.join(flags_error)}"
                if flags_warn:
                    status += f" W:{','.join(flags_warn)}"
                print(f"  {seq_name}/{sample_name}/{rep_label} {mode} "
                      f"{conc:g}ppm v={volume}uL BL={baseline:.0f}: "
                      f"area={area:.1f} t={t_max:.3f} "
                      f"RF={rf_mass:.0f} CR={cr:.2f} [{status}]")

    # === CSV ===
    output_csv = os.path.join(review_dir, "khp_uib_reintegration.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV: {output_csv} ({len(results)} entrades)")
    else:
        print("0 resultats!")
        return

    # === Estadistiques ===
    df = pd.DataFrame(results)
    n_ok = len(df[df["ok"] == "Y"])
    n_flags = len(df[df["ok"] == "N"])
    print(f"\nResultats: {len(df)} total, {n_ok} OK ({n_ok/len(df)*100:.0f}%), "
          f"{n_flags} amb flags")
    print(f"  COLUMN: {len(df[df['mode']=='COLUMN'])}, "
          f"BP: {len(df[df['mode']=='BP'])}")

    # Flags summary
    all_flags_list = []
    for f_str in df["flags"].dropna():
        if f_str:
            all_flags_list.extend(f_str.split("|"))
    if all_flags_list:
        from collections import Counter
        flag_counts = Counter(f.split("(")[0] for f in all_flags_list)
        print("\nFlags:")
        for flag, count in flag_counts.most_common():
            print(f"  {flag}: {count}")

    # === SCATTER: area vs ug_doc ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (mode_filter, ax) in enumerate(zip(["COLUMN", "BP"], axes)):
        df_mode = df[df["mode"] == mode_filter]
        if df_mode.empty:
            ax.set_title(f"{mode_filter}: sense dades UIB")
            continue

        df_ok = df_mode[df_mode["ok"] == "Y"]
        df_bad = df_mode[df_mode["ok"] == "N"]

        if not df_bad.empty:
            ax.scatter(df_bad["ug_doc"], df_bad["area"], c='lightgrey',
                       marker='x', s=40, alpha=0.6, label=f'Flags (n={len(df_bad)})')
        if not df_ok.empty:
            ax.scatter(df_ok["ug_doc"], df_ok["area"], c='dodgerblue',
                       marker='o', s=50, alpha=0.8, label=f'OK (n={len(df_ok)})')

            from scipy import stats as sp_stats
            x = df_ok["ug_doc"].values
            y_vals = df_ok["area"].values
            if len(x) >= 3:
                slope, intercept, r, p, se = sp_stats.linregress(x, y_vals)
                x_line = np.linspace(0, x.max() * 1.1, 100)
                ax.plot(x_line, slope * x_line + intercept, 'b--', linewidth=1.5,
                        label=f'R2={r**2:.4f}, slope={slope:.0f}, int={intercept:.1f}')

        ax.set_xlabel("ug DOC injectat")
        ax.set_ylabel("Area UIB")
        ax.set_title(f"{mode_filter} UIB (n={len(df_mode)})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    scatter_path = os.path.join(review_dir, "scatter_uib_clean.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"\nScatter: {scatter_path}")

    # === RF_mass summary ===
    print("\n=== RF_mass summary UIB (nomes OK) ===")
    df_ok_all = df[df["ok"] == "Y"]
    for mode_filter in ["COLUMN", "BP"]:
        df_m = df_ok_all[df_ok_all["mode"] == mode_filter]
        if df_m.empty:
            print(f"  {mode_filter}: cap entrada OK")
            continue
        rf = df_m["rf_mass"]
        print(f"  {mode_filter} (n={len(df_m)}): RF_mass median={rf.median():.0f}, "
              f"mean={rf.mean():.0f}, std={rf.std():.0f}, "
              f"CV={rf.std()/rf.mean()*100:.0f}%")
        for conc_val in sorted(df_m["conc_ppm"].unique()):
            df_c = df_m[df_m["conc_ppm"] == conc_val]
            rf_c = df_c["rf_mass"]
            std_str = f"+/- {rf_c.std():.0f}" if len(df_c) > 1 else ""
            print(f"    {conc_val:g} ppm (n={len(df_c)}): "
                  f"RF={rf_c.median():.0f} {std_str}")

    # === Comparacio repliques per concentracio ===
    print("\n=== Comparacio repliques (mateixa mostra, >1 replica) ===")
    grouped = df_ok_all.groupby(["seq", "sample"])
    for (seq, sample), grp in sorted(grouped):
        if len(grp) < 2:
            continue
        areas = grp["area"].values
        t_maxs = grp["t_max"].values
        a_mean = np.mean(areas)
        rsd = np.std(areas) / a_mean * 100 if a_mean > 0 else 0
        dt = np.max(t_maxs) - np.min(t_maxs)
        if rsd > 15 or dt > 0.3:
            flag = " *** DIVERGENT ***"
        else:
            flag = ""
        print(f"  {seq}/{sample}: areas={[f'{a:.1f}' for a in areas]}, "
              f"RSD={rsd:.1f}%, dt={dt:.3f} min{flag}")

    print(f"\n{len(errors)} errors:")
    for e in errors[:20]:
        print(f"  {e}")
    if len(errors) > 20:
        print(f"  ... i {len(errors)-20} mes")

    # === PDF per SEQ ===
    generate_pdf_per_seq(df, chromo_dir, review_dir)


def generate_pdf_per_seq(df, chromo_dir, review_dir):
    """Genera PDF amb cromatogrames UIB agrupats per SEQ."""
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(review_dir, "khp_uib_chromatograms_per_seq.pdf")
    seqs = df.groupby("seq")

    with PdfPages(pdf_path) as pdf:
        for seq_name, df_seq in sorted(seqs, key=lambda x: x[0]):
            n = len(df_seq)
            ncols = min(4, n)
            nrows = int(np.ceil(n / ncols))
            fig_h = max(4, nrows * 3.2)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, fig_h),
                                     squeeze=False)

            mode = df_seq["mode"].iloc[0]
            n_ok = len(df_seq[df_seq["ok"] == "Y"])
            fig.suptitle(f"{seq_name} ({mode}) UIB - {n} injeccions, {n_ok} OK",
                         fontsize=14, fontweight='bold', y=0.99)

            for i, (_, row) in enumerate(df_seq.iterrows()):
                r = i // ncols
                c = i % ncols
                ax = axes[r][c]

                png_name = f"{row['seq']}_{row['sample']}_{row['replica']}_UIB.png"
                png_path = os.path.join(chromo_dir, png_name)

                if os.path.exists(png_path):
                    img = plt.imread(png_path)
                    ax.imshow(img)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.text(0.5, 0.5, "PNG\nnot found", ha='center', va='center',
                            fontsize=10, color='grey', transform=ax.transAxes)

                ok_str = "OK" if row["ok"] == "Y" else "FAIL"
                color = 'green' if row["ok"] == "Y" else 'red'
                if row["flags"] and row["ok"] == "Y":
                    color = 'orange'

                title = (f"{row['sample']} {row['replica']} "
                         f"({row['conc_ppm']:g}ppm, {row['volume_uL']:.0f}uL)\n"
                         f"A={row['area']:.0f} RF={row['rf_mass']:.0f} "
                         f"CR={row['cr']:.2f} [{ok_str}]")
                ax.set_title(title, fontsize=7, color=color, pad=2)

                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)

            for i in range(n, nrows * ncols):
                r = i // ncols
                c = i % ncols
                axes[r][c].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

    print(f"\nPDF per SEQ: {pdf_path}")


if __name__ == "__main__":
    main()
