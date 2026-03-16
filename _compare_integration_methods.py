"""
Comparació de mètodes d'integració per KHP: COLUMN vs BP.

Carrega cromatogrames crus (t_doc, y_doc_net) dels analysis_result.json,
aplica múltiples mètodes d'integració, i compara:
  - RF per concentració (hauria de ser constant)
  - Ratio BP/COL (hauria de ser consistent)

Mètodes:
  1. tangent  — Projecció tangent al punt d'inflexió (Agilent, mètode actual)
  2. thr1     — Threshold 1% alçada pic
  3. thr5     — Threshold 5% alçada pic
  4. thr10    — Threshold 10% alçada pic
  5. trapez   — Tot sobre baseline (integració completa)
  6. gauss    — Fit gaussià, àrea analítica = h * sigma * sqrt(2*pi)
"""

import json
import os
import sys
import re
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "integration_methods")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Mètodes d'integració ───

def baseline_from_edges(t, y, is_bp=False):
    """Baseline des dels extrems del cromatograma."""
    n = len(y)
    if is_bp:
        # BP: últim 20%
        edge = max(10, n // 5)
        return float(np.median(y[-edge:]))
    else:
        # COLUMN: mínim entre primer 10% i últim 10%
        edge = max(10, n // 10)
        bl_start = np.median(y[:edge])
        bl_end = np.median(y[-edge:])
        return float(min(bl_start, bl_end))


def find_main_peak_idx(t, y):
    """Trobar el pic principal (màxim global del senyal net)."""
    return int(np.argmax(y))


def integrate_tangent(t, y, baseline, peak_idx):
    """Mètode tangent (Agilent) — projecció al punt d'inflexió."""
    n = len(y)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    dt = float(np.median(diffs))
    if dt <= 0:
        return None

    # Savitzky-Golay (finestra 0.7 min)
    sg_window = int(0.7 / dt)
    if sg_window % 2 == 0:
        sg_window += 1
    sg_window = max(7, min(sg_window, n if n % 2 == 1 else n - 1))

    y_smooth = savgol_filter(y, sg_window, 3)

    # Derivades
    dt_arr = np.diff(t)
    dt_arr[dt_arr == 0] = 1e-10
    d1 = np.diff(y_smooth) / dt_arr
    d2 = np.diff(d1) / dt_arr[:-1]

    # Punts d'inflexió
    left_infl = None
    for i in range(peak_idx - 1, 0, -1):
        if i < len(d2) - 1 and d2[i] > 0 and d2[i + 1] <= 0:
            left_infl = i + 1
            break

    right_infl = None
    for i in range(min(peak_idx, len(d2) - 1), len(d2) - 1):
        if d2[i] < 0 and d2[i + 1] >= 0:
            right_infl = i + 1
            break

    if left_infl is None or right_infl is None:
        return None  # fallback needed

    # Projecció tangent a baseline
    # Esquerra
    slope_left = d1[left_infl] if left_infl < len(d1) else 0
    if slope_left > 0:
        t_left = t[left_infl] + (baseline - y_smooth[left_infl]) / slope_left
    else:
        t_left = t[max(0, peak_idx - n // 4)]

    # Dreta
    slope_right = d1[right_infl] if right_infl < len(d1) else 0
    if slope_right < 0:
        t_right = t[right_infl] + (baseline - y_smooth[right_infl]) / slope_right
    else:
        t_right = t[min(n - 1, peak_idx + n // 4)]

    # Trobar índexs
    left_idx = max(0, np.searchsorted(t, t_left))
    right_idx = min(n - 1, np.searchsorted(t, t_right))

    if right_idx <= left_idx:
        return None

    y_net = np.maximum(y[left_idx:right_idx + 1] - baseline, 0)
    area = float(np.trapezoid(y_net, t[left_idx:right_idx + 1]))
    return area


def integrate_threshold(t, y, baseline, peak_idx, threshold_pct):
    """Integració per threshold: baseline + X% de l'alçada del pic."""
    n = len(y)
    peak_height = y[peak_idx] - baseline
    if peak_height <= 0:
        return 0.0

    threshold = baseline + (threshold_pct / 100.0) * peak_height

    # Buscar límit esquerre
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= threshold:
            left_idx = i
            break
    else:
        left_idx = 0

    # Buscar límit dret
    right_idx = peak_idx
    for i in range(peak_idx + 1, n):
        if y[i] <= threshold:
            right_idx = i
            break
    else:
        right_idx = n - 1

    if right_idx <= left_idx:
        return 0.0

    y_net = np.maximum(y[left_idx:right_idx + 1] - baseline, 0)
    area = float(np.trapezoid(y_net, t[left_idx:right_idx + 1]))
    return area


def integrate_trapezoid_full(t, y, baseline):
    """Integració completa: tot per sobre de la baseline."""
    y_net = np.maximum(y - baseline, 0)
    return float(np.trapezoid(y_net, t))


def gaussian(x, amp, mu, sigma):
    """Funció gaussiana."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def integrate_gaussian_fit(t, y, baseline, peak_idx):
    """Fit gaussià: àrea analítica = amp * sigma * sqrt(2*pi)."""
    y_net = np.maximum(y - baseline, 0)
    amp0 = y_net[peak_idx]
    mu0 = t[peak_idx]

    # Estimar sigma des de FWHM
    half_max = amp0 / 2
    left_hm = peak_idx
    for i in range(peak_idx, -1, -1):
        if y_net[i] <= half_max:
            left_hm = i
            break
    right_hm = peak_idx
    for i in range(peak_idx, len(y_net)):
        if y_net[i] <= half_max:
            right_hm = i
            break
    fwhm = t[right_hm] - t[left_hm]
    sigma0 = max(fwhm / 2.355, 0.1)

    # Fit
    try:
        # Zona al voltant del pic (±5 sigma)
        margin = max(int(5 * sigma0 / np.median(np.diff(t))), 20)
        i_start = max(0, peak_idx - margin)
        i_end = min(len(t), peak_idx + margin)

        popt, _ = curve_fit(
            gaussian, t[i_start:i_end], y_net[i_start:i_end],
            p0=[amp0, mu0, sigma0],
            bounds=([0, t[i_start], 0.01], [amp0 * 3, t[i_end], fwhm * 2]),
            maxfev=5000
        )
        amp_fit, mu_fit, sigma_fit = popt
        area = amp_fit * sigma_fit * np.sqrt(2 * np.pi)
        return float(area)
    except Exception:
        return None


# ─── Extracció de dades ───

def extract_conc(name):
    """Extreure concentració del nom KHP."""
    name_up = name.upper().replace(' ', '')
    if 'PPB' in name_up:
        m = re.search(r'KHP[_]?(\d+\.?\d*)', name_up)
        return float(m.group(1)) / 1000.0 if m else None
    m = re.search(r'KHP[_]?(\d+\.?\d*)', name_up)
    if not m:
        return None
    v = float(m.group(1))
    if v >= 10 and name_up.startswith('KHP0'):
        return v / 100.0
    return v if v <= 10 else None


def load_khp_chromatograms():
    """Carregar tots els KHP amb cromatogrames crus."""
    entries = []
    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        method = data.get("method", "?")

        for s in data.get("khp_samples", []):
            if "t_doc" not in s or "y_doc_net" not in s:
                continue
            conc = extract_conc(s.get("name", ""))
            if conc is None or conc <= 0:
                continue

            t = np.array(s["t_doc"])
            y = np.array(s["y_doc_net"])
            if len(t) < 10 or len(t) != len(y):
                continue

            vol = float(s.get("inj_volume") or (400 if method == "COLUMN" else 100))
            is_bp = method == "BP"

            entries.append({
                "seq": d,
                "method": method,
                "name": s.get("name", "?"),
                "replica": s.get("replica", "?"),
                "conc": conc,
                "vol": vol,
                "is_bp": is_bp,
                "t": t,
                "y": y,
            })
    return entries


# ─── Main ───

def main():
    entries = load_khp_chromatograms()
    print(f"KHP amb cromatogrames: {len(entries)}")

    methods = ["tangent", "thr1", "thr5", "thr10", "trapez", "gauss"]

    results = []
    for e in entries:
        t, y = e["t"], e["y"]
        baseline = baseline_from_edges(t, y, e["is_bp"])
        peak_idx = find_main_peak_idx(t, y)

        areas = {}

        # 1. Tangent (Agilent)
        a = integrate_tangent(t, y, baseline, peak_idx)
        areas["tangent"] = a if a and a > 0 else None

        # 2-4. Threshold
        for pct, key in [(1, "thr1"), (5, "thr5"), (10, "thr10")]:
            areas[key] = integrate_threshold(t, y, baseline, peak_idx, pct)

        # 5. Trapezoid total
        areas["trapez"] = integrate_trapezoid_full(t, y, baseline)

        # 6. Gaussian fit
        a = integrate_gaussian_fit(t, y, baseline, peak_idx)
        areas["gauss"] = a if a and a > 0 else None

        row = {
            "seq": e["seq"], "method": e["method"], "name": e["name"],
            "replica": e["replica"], "conc": e["conc"], "vol": e["vol"],
            "baseline": baseline, "peak_height": y[peak_idx] - baseline,
        }
        for m in methods:
            a = areas.get(m)
            if a and a > 0:
                row[f"area_{m}"] = a
                row[f"ppm_{m}"] = a * 1000 / (e["vol"] * e["conc"])  # RF efectiu
                # Calcular ppm real: no tenim RF per cada metode,
                # calculem RF = area / (conc * vol / 1000) per veure consistencia
                row[f"rf_{m}"] = a * 1000 / (e["conc"] * e["vol"])
            else:
                row[f"area_{m}"] = None
                row[f"ppm_{m}"] = None
                row[f"rf_{m}"] = None

        results.append(row)

    # ─── Taula resum ───
    print(f"\n{'='*120}")
    print("ÀREES PER MÈTODE D'INTEGRACIÓ (KHP)")
    print(f"{'='*120}")

    header = f"{'SEQ':22s} {'M':3s} {'Name':12s} {'Conc':>5s} {'Vol':>4s}"
    for m in methods:
        header += f" {m:>8s}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x["method"], x["conc"], x["seq"])):
        line = f"{r['seq']:22s} {r['method'][:3]:3s} {r['name']:12s} {r['conc']:5.2f} {r['vol']:4.0f}"
        for m in methods:
            a = r.get(f"area_{m}")
            if a:
                line += f" {a:8.1f}"
            else:
                line += f" {'---':>8s}"
        print(line)

    # ─── RF per mètode, concentració i mode ───
    print(f"\n{'='*120}")
    print("RF PER MÈTODE (RF = Area * 1000 / (conc * vol)) — RF CONSTANT = MÈTODE BO")
    print(f"{'='*120}")

    for mode in ["COLUMN", "BP"]:
        sub = [r for r in results if r["method"] == mode]
        if not sub:
            continue
        print(f"\n--- {mode} ---")
        concs = sorted(set(r["conc"] for r in sub))

        header = f"{'Conc':>6s} {'n':>3s}"
        for m in methods:
            header += f" {'RF_'+m:>12s}"
        print(header)
        print("-" * len(header))

        for c in concs:
            entries_c = [r for r in sub if abs(r["conc"] - c) < max(0.005, c * 0.15)]
            n = len(entries_c)
            line = f"{c:6.3f} {n:3d}"
            for m in methods:
                rfs = [r[f"rf_{m}"] for r in entries_c if r.get(f"rf_{m}")]
                if rfs:
                    line += f" {np.mean(rfs):6.0f}+/-{np.std(rfs):4.0f}"
                else:
                    line += f" {'---':>12s}"
            print(line)

        # CV total per mètode
        line = f"{'CV%':>6s} {'':>3s}"
        for m in methods:
            rfs = [r[f"rf_{m}"] for r in sub if r.get(f"rf_{m}")]
            if rfs and np.mean(rfs) > 0:
                cv = np.std(rfs) / np.mean(rfs) * 100
                line += f" {cv:11.1f}%"
            else:
                line += f" {'---':>12s}"
        print(line)

    # ─── Comparació directa COLUMN vs BP ───
    print(f"\n{'='*120}")
    print("COMPARACIÓ COLUMN vs BP: Ratio àrea/µL (BP/COL) per mètode i concentració")
    print("Ratio estable ≈ recuperació columna. Ratio variable = problema integració.")
    print(f"{'='*120}")

    col = [r for r in results if r["method"] == "COLUMN"]
    bp = [r for r in results if r["method"] == "BP"]
    col_concs = sorted(set(round(r["conc"], 3) for r in col))
    bp_concs = sorted(set(round(r["conc"], 3) for r in bp))
    shared_concs = sorted(set(round(c, 2) for c in col_concs) & set(round(c, 2) for c in bp_concs))

    if shared_concs:
        header = f"{'Conc':>6s}"
        for m in methods:
            header += f" {'R_'+m:>8s}"
        print(header)
        print("-" * len(header))

        for c in shared_concs:
            ce = [r for r in col if abs(r["conc"] - c) < max(0.005, c * 0.15)]
            be = [r for r in bp if abs(r["conc"] - c) < max(0.005, c * 0.15)]
            line = f"{c:6.3f}"
            for m in methods:
                col_an = [r.get(f"area_{m}", 0) / r["vol"] for r in ce if r.get(f"area_{m}")]
                bp_an = [r.get(f"area_{m}", 0) / r["vol"] for r in be if r.get(f"area_{m}")]
                if col_an and bp_an:
                    ratio = np.mean(bp_an) / np.mean(col_an)
                    line += f" {ratio:8.3f}"
                else:
                    line += f" {'---':>8s}"
            print(line)

    # ─── Gràfics ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("RF per mètode d'integració — COLUMN vs BP", fontsize=14, fontweight="bold")

        for idx, m in enumerate(methods):
            ax = axes[idx // 3, idx % 3]

            for mode, color, marker in [("COLUMN", "#2196F3", "o"), ("BP", "#FF9800", "s")]:
                sub = [r for r in results if r["method"] == mode and r.get(f"rf_{m}")]
                if not sub:
                    continue
                concs = [r["conc"] for r in sub]
                rfs = [r[f"rf_{m}"] for r in sub]
                ax.scatter(concs, rfs, c=color, marker=marker, s=40, alpha=0.7,
                          edgecolors="k", linewidths=0.5, label=mode, zorder=5)

            ax.set_xlabel("Concentració KHP (ppm)")
            ax.set_ylabel("RF (area*1000/(conc*vol))")
            ax.set_title(m)
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

            # Línia horitzontal RF ideal (mitjana dels punts raonables)
            all_rfs = [r[f"rf_{m}"] for r in results if r.get(f"rf_{m}") and r[f"rf_{m}"] < 3000]
            if all_rfs:
                ax.axhline(np.median(all_rfs), color="red", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "rf_per_method.png"), dpi=150, bbox_inches="tight")
        print(f"\nGràfic guardat: {OUT_DIR}/rf_per_method.png")
        plt.close()

        # Pàgina 2: Exemples de cromatogrames amb límits de cada mètode
        # Seleccionar 4 exemples: COLUMN 1ppm, 5ppm, BP 1ppm, 5ppm
        examples = []
        for mode, conc_target in [("COLUMN", 1.0), ("COLUMN", 5.0), ("BP", 1.0), ("BP", 2.0)]:
            candidates = [e for e in entries if e["method"] == mode
                         and abs(e["conc"] - conc_target) < max(0.05, conc_target * 0.15)]
            if candidates:
                examples.append(candidates[0])

        if examples:
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
            fig2.suptitle("Cromatogrames KHP — Límits d'integració per mètode", fontsize=13, fontweight="bold")

            for idx, e in enumerate(examples):
                ax = axes2[idx // 2, idx % 2]
                t, y = e["t"], e["y"]
                baseline = baseline_from_edges(t, y, e["is_bp"])
                peak_idx = find_main_peak_idx(t, y)

                # Zoom al pic
                t_peak = t[peak_idx]
                if e["is_bp"]:
                    t_min, t_max_plot = max(0, t_peak - 3), min(t[-1], t_peak + 5)
                else:
                    t_min, t_max_plot = max(0, t_peak - 10), min(t[-1], t_peak + 15)
                mask = (t >= t_min) & (t <= t_max_plot)

                ax.plot(t[mask], y[mask], "k-", linewidth=1, label="Senyal")
                ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                # Mostrar límits per cada mètode amb colors
                colors_m = {"thr1": "#4CAF50", "thr5": "#FF9800", "thr10": "#F44336",
                           "trapez": "#9C27B0", "gauss": "#2196F3"}

                peak_h = y[peak_idx] - baseline
                for pct, key, col in [(1, "thr1", "#4CAF50"), (5, "thr5", "#FF9800"), (10, "thr10", "#F44336")]:
                    thr = baseline + pct / 100.0 * peak_h
                    ax.axhline(thr, color=col, linestyle=":", linewidth=0.8, alpha=0.6, label=f"thr{pct}%")

                ax.set_title(f"{e['method']} {e['name']} ({e['conc']:.1f} ppm, {e['vol']:.0f}µL)")
                ax.set_xlabel("Temps (min)")
                ax.set_ylabel("DOC net (ppb)")
                ax.legend(fontsize=7, loc="upper right")

            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "chromatogram_examples.png"), dpi=150, bbox_inches="tight")
            print(f"Cromatogrames guardat: {OUT_DIR}/chromatogram_examples.png")
            plt.close()

    except ImportError:
        print("[WARN] matplotlib no disponible")

    # ─── CSV ───
    csv_path = os.path.join(OUT_DIR, "integration_methods_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["seq", "method", "name", "replica", "conc", "vol", "baseline", "peak_height"]
        for m in methods:
            cols.extend([f"area_{m}", f"rf_{m}"])
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [str(r.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")
    print(f"CSV guardat: {csv_path}")


if __name__ == "__main__":
    main()
