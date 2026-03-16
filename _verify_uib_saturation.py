"""
Verificació saturació UIB — 293_SEQ_CAL

Compara cromatogrames UIB de totes les concentracions KHP per determinar
si el "plateau" detectat a 5ppm és saturació real o artefacte (timeout, etc).

Genera: REGISTRY/review/uib_saturation_293.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from hpsec_core import detect_peak_clipping, downsample_to_cadence, DOC_TARGET_DT_MIN

SEQ_PATH = Path(r"C:\Users\Lequia\Desktop\Dades3\293_SEQ_CAL")
CSV_DIR = SEQ_PATH / "CSV"
MANIFEST = SEQ_PATH / "CHECK" / "data" / "import_manifest.json"
OUTPUT_DIR = Path(r"C:\Users\Lequia\Desktop\Dades3\REGISTRY\review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Concentracions KHP (extretes del nom)
CONC_MAP = {
    "KHP01": 0.1,
    "KHP025": 0.25,
    "KHP05": 0.5,
    "KHP1": 1.0,
    "KHP2": 2.0,
    "KHP5": 5.0,
}


def read_uib_csv(path):
    """Llegeix CSV UIB (UTF-16, tab-separated, sense header)."""
    df = pd.read_csv(path, sep='\t', header=None, encoding='utf-16',
                     names=['t', 'y'])
    return df['t'].values, df['y'].values


def main():
    # Carregar manifest per timeouts i baselines
    with open(MANIFEST, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    samples = manifest["samples"]

    # Recollir dades
    entries = []
    for sample in samples:
        name = sample["name"]
        conc = CONC_MAP.get(name)
        if conc is None:
            continue
        for rep_info in sample["replicas"]:
            rep = rep_info["replica"]
            uib_file = rep_info.get("uib", {}).get("file")
            baseline = rep_info.get("uib", {}).get("baseline", 0)
            direct_info = rep_info.get("direct", {})
            timeout_ranges = direct_info.get("timeout_ranges", [])

            if not uib_file:
                continue

            uib_path = CSV_DIR / uib_file
            if not uib_path.exists():
                print(f"SKIP {uib_file}: no existeix")
                continue

            t_raw, y_raw = read_uib_csv(uib_path)

            # Downsample a cadència DOC (com fa la Suite)
            t_ds, y_ds = downsample_to_cadence(t_raw, y_raw, DOC_TARGET_DT_MIN)

            # Net = raw - baseline
            y_net = y_ds - baseline

            # Detecció saturació
            clip = detect_peak_clipping(t_ds, y_net)

            # Timeout UIB estimat (transferir de Direct)
            uib_timeouts = []
            for tr in timeout_ranges:
                uib_timeouts.append({
                    "t_start": tr["t_start_min"],
                    "t_end": tr["t_end_min"],
                    "zone": tr["zone"],
                })

            entries.append({
                "name": name,
                "conc": conc,
                "rep": rep,
                "t": t_ds,
                "y_raw": y_ds,
                "y_net": y_net,
                "baseline": baseline,
                "clip": clip,
                "timeouts_direct": uib_timeouts,
                "t_raw_full": t_raw,
                "y_raw_full": y_raw,
            })

    # Ordenar per concentració
    entries.sort(key=lambda e: (e["conc"], e["rep"]))

    # =========================================================================
    # FIGURA 1: Panell complet — un subplot per concentració
    # =========================================================================
    concs = sorted(set(e["conc"] for e in entries))
    fig, axes = plt.subplots(len(concs), 2, figsize=(16, 3 * len(concs)),
                             gridspec_kw={"width_ratios": [3, 1]})
    if len(concs) == 1:
        axes = axes.reshape(1, -1)

    for row, conc in enumerate(concs):
        ax_full = axes[row, 0]
        ax_zoom = axes[row, 1]

        conc_entries = [e for e in entries if e["conc"] == conc]
        colors = ['#2E86AB', '#E74C3C']

        for j, e in enumerate(conc_entries):
            t, y = e["t"], e["y_net"]
            label = f"R{e['rep']}"
            c = colors[j % 2]

            # Gràfic complet
            ax_full.plot(t, y, color=c, linewidth=0.8, alpha=0.8, label=label)

            # Marcar timeouts Direct (zona equivalent UIB)
            for to in e["timeouts_direct"]:
                ax_full.axvspan(to["t_start"] - 0.2, to["t_end"] + 1.8,
                                alpha=0.15, color='orange',
                                label=f"Timeout {to['zone']}" if j == 0 else None)

        # Zoom al pic (±5 min del màxim)
        # Trobar t_max del pic més alt
        all_y = np.concatenate([e["y_net"] for e in conc_entries])
        all_t = np.concatenate([e["t"] for e in conc_entries])
        peak_idx = np.argmax(all_y)
        t_peak = all_t[peak_idx]

        for j, e in enumerate(conc_entries):
            t, y = e["t"], e["y_net"]
            c = colors[j % 2]
            clip = e["clip"]

            ax_zoom.plot(t, y, color=c, linewidth=1.0, alpha=0.9,
                         label=f"R{e['rep']}")

            # Línia 98% del màxim (llindar plateau)
            y_max = clip["y_max_observed"]
            if y_max > 0:
                ax_zoom.axhline(y_max * 0.98, color=c, linestyle='--',
                                linewidth=0.5, alpha=0.5)

        ax_zoom.set_xlim(t_peak - 5, t_peak + 5)

        # Info saturació
        clip_info = conc_entries[0]["clip"]
        sat_text = (f"plateau_ratio={clip_info['plateau_ratio']:.3f}\n"
                    f"plateau={clip_info['plateau_width_pts']} pts\n"
                    f"FWHM={clip_info['fwhm_pts']} pts\n"
                    f"y_max={clip_info['y_max_observed']:.1f}")
        if clip_info["is_saturated"]:
            sat_text += "\n⚠ SATURAT"
            ax_zoom.set_facecolor('#fff5f5')
        else:
            sat_text += "\n✓ OK"
            ax_zoom.set_facecolor('#f5fff5')

        ax_zoom.text(0.98, 0.95, sat_text, transform=ax_zoom.transAxes,
                     fontsize=7, va='top', ha='right', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_full.set_title(f"KHP {conc} ppm — cromatograma UIB complet",
                          fontsize=10, fontweight='bold')
        ax_zoom.set_title(f"Zoom pic ({t_peak - 5:.0f}–{t_peak + 5:.0f} min)",
                          fontsize=9)
        ax_full.set_ylabel("DOC UIB net (ppb)", fontsize=8)
        ax_full.legend(fontsize=7, loc='upper right')
        ax_zoom.set_ylabel("")
        ax_full.tick_params(labelsize=7)
        ax_zoom.tick_params(labelsize=7)

        if row == len(concs) - 1:
            ax_full.set_xlabel("Temps (min)", fontsize=8)
            ax_zoom.set_xlabel("Temps (min)", fontsize=8)

    fig.suptitle("293_SEQ_CAL — Verificació saturació UIB\n"
                 "Sensibilitat: 700 ppb | Mètode: plateau/FWHM ratio (threshold=0.40)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out1 = OUTPUT_DIR / "uib_saturation_293.png"
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Guardat: {out1}")
    plt.close(fig)

    # =========================================================================
    # FIGURA 2: Comparació directa UIB vs DOC Direct (pic normalitzat)
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    axes2 = axes2.flatten()

    for i, conc in enumerate(concs):
        ax = axes2[i]
        e = [x for x in entries if x["conc"] == conc][0]  # R1

        t, y_net = e["t"], e["y_net"]
        y_max = np.max(y_net)
        if y_max > 0:
            y_norm = y_net / y_max
        else:
            y_norm = y_net

        # Centrar al pic
        peak_idx = np.argmax(y_net)
        t_centered = t - t[peak_idx]

        ax.plot(t_centered, y_norm, color='#2E86AB', linewidth=1.2,
                label=f'UIB (y_max={y_max:.0f})')

        # Gaussiana ideal per referència
        # Estimar sigma des del FWHM
        clip = e["clip"]
        if clip["fwhm_pts"] > 3:
            dt = np.median(np.diff(t))
            fwhm_min = clip["fwhm_pts"] * dt
            sigma = fwhm_min / 2.355
            t_gauss = np.linspace(-5, 5, 200)
            y_gauss = np.exp(-0.5 * (t_gauss / sigma) ** 2)
            ax.plot(t_gauss, y_gauss, color='#999', linewidth=0.8,
                    linestyle='--', label=f'Gaussiana (σ={sigma:.2f} min)')

        # Línia 98%
        ax.axhline(0.98, color='red', linewidth=0.5, linestyle=':',
                   alpha=0.5, label='98% (llindar plateau)')

        # Línia sensibilitat 700
        if y_max > 0:
            sens_norm = 700 / y_max
            if 0 < sens_norm < 1.5:
                ax.axhline(sens_norm, color='orange', linewidth=0.8,
                           linestyle='-.', alpha=0.7,
                           label=f'Sensibilitat 700 ({sens_norm:.2f})')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"{conc} ppm — ratio={clip['plateau_ratio']:.3f}"
                     f" {'⚠SAT' if clip['is_saturated'] else '✓OK'}",
                     fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='upper left')
        ax.set_xlabel("Temps relatiu (min)", fontsize=7)
        ax.set_ylabel("Senyal normalitzat", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    fig2.suptitle("293_SEQ_CAL — Forma del pic UIB normalitzat vs Gaussiana ideal\n"
                  "Si el pic és realment saturat, el cim serà pla (no seguirà la Gaussiana)",
                  fontsize=11, fontweight='bold')
    fig2.tight_layout()
    out2 = OUTPUT_DIR / "uib_saturation_293_normalized.png"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Guardat: {out2}")
    plt.close(fig2)

    # =========================================================================
    # RESUM CONSOLA
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESUM SATURACIÓ UIB — 293_SEQ_CAL")
    print("=" * 70)
    print(f"{'Conc':>6}  {'Rep':>3}  {'y_max':>8}  {'plateau':>8}  {'FWHM':>6}  "
          f"{'ratio':>7}  {'SAT?':>5}  {'Timeout prop pic?':>18}")
    print("-" * 70)

    for e in entries:
        clip = e["clip"]
        # Comprovar si algun timeout Direct cau prop del pic UIB
        peak_idx = np.argmax(e["y_net"])
        t_peak = e["t"][peak_idx]
        timeout_near = False
        for to in e["timeouts_direct"]:
            # Timeout Direct transferit a UIB: ±2 min del pic
            if abs(to["t_start"] - t_peak) < 5 or abs(to["t_end"] - t_peak) < 5:
                timeout_near = True
                break

        sat_flag = "!! SI" if clip['is_saturated'] else "   NO"
        to_flag = "!! SI" if timeout_near else "   NO"
        print(f"{e['conc']:6.2f}  R{e['rep']:>2}  {clip['y_max_observed']:8.1f}  "
              f"{clip['plateau_width_pts']:8d}  {clip['fwhm_pts']:6d}  "
              f"{clip['plateau_ratio']:7.3f}  {sat_flag:>5}  {to_flag:>18}")

    # Check: hi ha algun indicador temporal de saturació?
    # Si és saturació real, el plateau hauria de ser a y constant (derivada ≈ 0)
    # Si és timeout, hi hauria una pertorbació (pic espuri o gap)
    print("\n--- Anàlisi derivada al cim ---")
    for e in entries:
        if e["conc"] < 2:
            continue
        clip = e["clip"]
        peak_idx = np.argmax(e["y_net"])
        y = e["y_net"]
        # Derivada al voltant del pic (±10 punts)
        margin = min(15, peak_idx, len(y) - peak_idx - 1)
        y_peak_region = y[peak_idx - margin:peak_idx + margin + 1]
        dy = np.diff(y_peak_region)
        # Desviació estàndard de la derivada prop del cim
        # Saturació real: dy ≈ 0 (constant). Timeout: dy fluctua.
        near_top = y_peak_region >= np.max(y_peak_region) * 0.95
        if np.sum(near_top) > 3:
            dy_at_top = np.diff(y_peak_region[near_top])
            dy_std = np.std(dy_at_top)
            dy_mean = np.mean(np.abs(dy_at_top))
            print(f"  {e['name']} R{e['rep']} ({e['conc']} ppm): "
                  f"std(dy) al cim = {dy_std:.3f}, mean|dy| = {dy_mean:.3f} ppb/pt "
                  f"{'-> PLATAFORMA NETA (saturacio)' if dy_std < 1.0 else '-> IRREGULARITAT (no saturacio)'}")
        else:
            print(f"  {e['name']} R{e['rep']} ({e['conc']} ppm): "
                  f"pocs punts al cim ({np.sum(near_top)})")


if __name__ == "__main__":
    main()
