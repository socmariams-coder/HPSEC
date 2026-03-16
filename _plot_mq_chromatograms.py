"""
Visualitzar cromatogrames DOC dels MQ per entendre d'on ve l'àrea.

Selecciona MQ representatius (baixa/mitjana/alta àrea) de COLUMN i BP,
carrega les dades crues i pinta el cromatograma amb les finestres de fracció.
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

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "mq_chromatograms")

from hpsec_import import import_from_manifest, ensure_data_loaded
from hpsec_config import get_config
from hpsec_core import get_baseline_value
from scipy.integrate import trapezoid

FRAC_WINDOWS = {
    "BioP": (10.8, 18.0),
    "HS":   (18.0, 23.0),
    "BB":   (23.0, 26.0),
    "SB":   (26.0, 32.0),
    "LMW":  (32.0, 70.0),
}
FRAC_COLORS = {
    "BioP": '#2196F3', "HS": '#4CAF50', "BB": '#FF9800',
    "SB": '#9C27B0', "LMW": '#795548'
}


def seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def find_mq_targets():
    """Trobar MQ representatius: baix/mig/alt àrea per COLUMN i BP."""
    entries = []
    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        method = data.get("method", "?")
        if method not in ("COLUMN", "BP"):
            continue
        for s in data.get("samples", []):
            name = s.get("name", "")
            if not name.upper().startswith("MQ"):
                continue
            area = s.get("areas", {}).get("DOC", {}).get("total", 0)
            entries.append({
                "seq": d, "name": name, "method": method,
                "area": area, "seq_num": seq_num(d),
            })

    # Seleccionar per mode: percentils 10, 25, 50, 75, 90
    targets = []
    for mode in ["COLUMN", "BP"]:
        mode_entries = sorted([e for e in entries if e["method"] == mode],
                              key=lambda x: x["area"])
        if len(mode_entries) < 5:
            continue
        n = len(mode_entries)
        for pct_label, idx in [("P10", n // 10), ("P25", n // 4),
                                ("P50", n // 2), ("P75", 3 * n // 4),
                                ("P90", 9 * n // 10)]:
            e = mode_entries[min(idx, n - 1)]
            targets.append({**e, "pct": pct_label})

    return targets


def load_mq_chromatogram(seq_name, sample_name, config):
    """Carregar cromatograma DOC d'un MQ."""
    seq_path = os.path.join(DATA_DIR, seq_name)
    manifest_path = os.path.join(seq_path, "CHECK", "data", "import_manifest.json")
    if not os.path.isfile(manifest_path):
        return None, None, None

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    imported = import_from_manifest(seq_path, manifest=manifest, config=config, load_data=False)
    if imported.get("data_deferred"):
        ensure_data_loaded(imported, config=config)

    samples = imported.get("samples", {})
    if sample_name not in samples:
        return None, None, None

    s = samples[sample_name]
    method = imported.get("method", "COLUMN")
    reps = s.get("replicas", {})
    for rep_key, rep_data in reps.items():
        direct = rep_data.get("direct")
        if not direct:
            continue
        t = direct.get("t")
        y = direct.get("y")
        if t is not None and y is not None:
            return np.array(t), np.array(y), method
    return None, None, None


def main():
    config = get_config()
    os.makedirs(OUT_DIR, exist_ok=True)

    targets = find_mq_targets()
    print(f"Targets seleccionats: {len(targets)}")

    # Separar per mode
    col_targets = [t for t in targets if t["method"] == "COLUMN"]
    bp_targets = [t for t in targets if t["method"] == "BP"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ================================================================
    # PLOT 1: COLUMN MQ (5 percentils)
    # ================================================================
    n_col = len(col_targets)
    if n_col > 0:
        fig, axes = plt.subplots(n_col, 2, figsize=(20, 4 * n_col))
        if n_col == 1:
            axes = axes.reshape(1, -1)

        for i, target in enumerate(col_targets):
            t, y, method = load_mq_chromatogram(target["seq"], target["name"], config)
            if t is None:
                print(f"  SKIP {target['seq']}/{target['name']}")
                continue

            bl = get_baseline_value(t, y, mode="COLUMN", config=config)
            y_net = np.maximum(y - bl, 0)

            # Calcular àrees per fracció
            frac_areas = {}
            for fn, (t0, t1) in FRAC_WINDOWS.items():
                mask = (t >= t0) & (t < t1)
                if mask.sum() > 1:
                    frac_areas[fn] = float(trapezoid(y_net[mask], t[mask]))
                else:
                    frac_areas[fn] = 0.0
            total = sum(frac_areas.values())

            label = (f"{target['pct']} — {target['seq']}/{target['name']}  "
                     f"(àrea={target['area']:.0f})")
            print(f"  COLUMN {label}")

            # Cromatograma cru + baseline + finestres
            ax = axes[i, 0]
            ax.plot(t, y, 'k-', lw=0.6, alpha=0.8)
            ax.axhline(bl, color='red', ls='--', lw=1, alpha=0.7,
                       label=f'baseline={bl:.1f}')

            for fn, (t0, t1) in FRAC_WINDOWS.items():
                ax.axvspan(t0, t1, alpha=0.12, color=FRAC_COLORS[fn])
                mid = (t0 + t1) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else bl + 10,
                        fn, ha='center', fontsize=7, color=FRAC_COLORS[fn],
                        fontweight='bold', va='top')

            ax.set_xlim(5, 55)
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Senyal DOC (ppb)")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=7)

            # Senyal net (y - baseline, clipped a 0) amb àrees ombrejades
            ax = axes[i, 1]
            ax.plot(t, y_net, 'k-', lw=0.6, alpha=0.8)
            ax.axhline(0, color='grey', ls=':', lw=0.5)

            for fn, (t0, t1) in FRAC_WINDOWS.items():
                mask = (t >= t0) & (t < t1)
                if mask.sum() > 1:
                    ax.fill_between(t[mask], 0, y_net[mask],
                                    alpha=0.3, color=FRAC_COLORS[fn],
                                    label=f'{fn}={frac_areas[fn]:.0f}')

            ax.set_xlim(5, 55)
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Senyal net (ppb)")
            ax.set_title(f"Net — total={total:.0f}", fontsize=9)
            ax.legend(fontsize=7, ncol=5, loc='upper right')

        fig.suptitle("MQ COLUMN — Cromatogrames per percentil d'àrea",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "mq_column_chromatograms.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  Gràfic COLUMN: {path}")

    # ================================================================
    # PLOT 2: BP MQ (5 percentils)
    # ================================================================
    n_bp = len(bp_targets)
    if n_bp > 0:
        fig, axes = plt.subplots(n_bp, 2, figsize=(20, 4 * n_bp))
        if n_bp == 1:
            axes = axes.reshape(1, -1)

        for i, target in enumerate(bp_targets):
            t, y, method = load_mq_chromatogram(target["seq"], target["name"], config)
            if t is None:
                print(f"  SKIP {target['seq']}/{target['name']}")
                continue

            bl = get_baseline_value(t, y, mode="BP", config=config)
            y_net = np.maximum(y - bl, 0)
            total = float(trapezoid(y_net, t)) if len(t) > 1 else 0

            label = (f"{target['pct']} — {target['seq']}/{target['name']}  "
                     f"(àrea={target['area']:.0f})")
            print(f"  BP {label}")

            # Cromatograma cru + baseline
            ax = axes[i, 0]
            ax.plot(t, y, 'k-', lw=0.6, alpha=0.8)
            ax.axhline(bl, color='red', ls='--', lw=1, alpha=0.7,
                       label=f'baseline={bl:.1f}')
            ax.set_xlim(0, 15)
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Senyal DOC (ppb)")
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=7)

            # Senyal net
            ax = axes[i, 1]
            ax.plot(t, y_net, 'k-', lw=0.6, alpha=0.8)
            ax.fill_between(t, 0, y_net, alpha=0.3, color='steelblue')
            ax.axhline(0, color='grey', ls=':', lw=0.5)
            ax.set_xlim(0, 15)
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Senyal net (ppb)")
            ax.set_title(f"Net — total={total:.0f}", fontsize=9)

        fig.suptitle("MQ BP — Cromatogrames per percentil d'àrea",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "mq_bp_chromatograms.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  Gràfic BP: {path}")


if __name__ == "__main__":
    main()
