# -*- coding: utf-8 -*-
"""
Representació VWD amb longituds d'ona conegudes (VWD1A=220 nm, VWD1B=254 nm).
EXPLORACIÓ EXTERNA — no toca la Suite.

Clau: el DAD (en línia, AIGÜES AMUNT) també mesura 220 i 254. El VWD (nou) mesura
els MATEIXOS λ però AIGÜES AVALL del detector DOC. Representem cada λ als dos punts
del flux per localitzar on apareix l'oscil·lació.

Per mostra (2 files, magnitud real en mAU):
  dalt  — 220 nm: DAD 220 (amunt) vs VWD1A/220 (avall)
  baix  — 254 nm: DAD 254 (amunt) vs VWD1B/254 (avall)
  + DOC normalitzat superposat (referència de retenció) i finestra del pic DOC.

Ús:  python vwd_representa.py 308_SEQ_BP 307_SEQ_BP
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HPSEC = r"C:\Users\maria\Proyectos\HPSEC"
DADES = r"C:\Users\maria\Proyectos\Dades3"
sys.path.insert(0, HPSEC)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import hpsec_import as I
import hpsec_analyze as A
from vwd_vs_actual import _norm, build_vwd_index   # reutilitza helpers

HERE = os.path.dirname(os.path.abspath(__file__))
VWD_WL = {"VWD1A": "220", "VWD1B": "254"}          # assignació confirmada per la usuària
PAIR_COLOR = {"220": "#c0392b", "254": "#e67e22"}


def _twin_doc(ax, t_doc, y_doc, t_start, t_end):
    if t_start is not None and t_end is not None:
        ax.axvspan(t_start, t_end, color="#1f3b73", alpha=0.08, zorder=0)
    axd = ax.twinx()
    if t_doc is not None and len(t_doc):
        axd.plot(t_doc, _norm(y_doc), color="#1f3b73", lw=1.8, alpha=0.55,
                 label="DOC (norm.)")
        axd.set_ylim(-0.05, 1.3)
    axd.set_yticks([])
    return axd


def plot_sample(outdir, seq, grp, rep, t_doc, y_doc, t_dad, dad, vwd,
                t_start, t_end):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, wl, ch in zip(axes, ("220", "254"), ("VWD1A", "VWD1B")):
        # DAD (amunt) en mAU
        if t_dad is not None and wl in dad:
            ax.plot(t_dad, dad[wl], color="#5a5a5a", lw=1.3,
                    label=f"DAD {wl} nm (en línia, amunt)")
        # VWD (avall) en mAU
        if ch in vwd:
            tt, yy = vwd[ch]
            ax.plot(tt, yy, color=PAIR_COLOR[wl], lw=1.4,
                    label=f"VWD {wl} nm (nou, avall del DOC)")
        _twin_doc(ax, t_doc, y_doc, t_start, t_end)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_ylabel(f"{wl} nm  (mAU)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    axes[0].set_title(f"{seq}  ·  {grp} rep{rep}   —   mateix λ, dos punts del flux "
                      f"(DAD amunt vs VWD avall del DOC) · ombra = pic DOC")
    axes[1].set_xlabel("temps (min)")
    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in f"{grp}_rep{rep}")
    fig.savefig(os.path.join(outdir, f"{safe}_matched.png"), dpi=120)
    plt.close(fig)


def process_seq(seq):
    wdir = os.path.join(DADES, seq)
    if not os.path.isdir(wdir):
        print(f"  !! no existeix {wdir}"); return
    outdir = os.path.join(HERE, f"out_{seq}", "matched")
    os.makedirs(outdir, exist_ok=True)
    vwd_idx = build_vwd_index(os.path.join(wdir, "CSV"))
    imp = I.import_sequence(wdir); I.ensure_data_loaded(imp)
    res = A.analyze_sequence(imp, {}, do_quantify=False)
    sg = res.get("samples_grouped") or {}
    n = 0
    for grp in sorted(sg.keys()):
        for rep, r in sorted((sg[grp].get("replicas") or {}).items()):
            if not isinstance(r, dict):
                continue
            key = (I.normalize_key(grp), I.normalize_rep(rep))
            vwdf = vwd_idx.get(key, {})
            if not vwdf:
                continue
            t_doc = np.asarray(r.get("t_doc"), float)
            y_doc = np.asarray(r.get("y_doc_net"), float)
            pk = r.get("peak_info") or {}
            t_start, t_end = pk.get("t_start"), pk.get("t_end")
            df_dad = r.get("df_dad")
            t_dad, dad = None, {}
            if df_dad is not None:
                t_dad = df_dad["time (min)"].to_numpy(float)
                for wl in ("220", "254"):
                    if wl in getattr(df_dad, "columns", []):
                        dad[wl] = df_dad[wl].to_numpy(float)
            vwd = {}
            for ch in ("VWD1A", "VWD1B"):
                if ch in vwdf:
                    dfc, st = I.llegir_dad_1a(vwdf[ch], ch)
                    if not dfc.empty:
                        vwd[ch] = (dfc["time (min)"].to_numpy(float),
                                   dfc[ch].to_numpy(float))
            plot_sample(outdir, seq, grp, rep, t_doc, y_doc, t_dad, dad, vwd,
                        t_start, t_end)
            n += 1
    print(f"  {seq}: {n} figures -> {outdir}")


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["308_SEQ_BP"]
    print("Representació VWD (220/254) — mateix λ, dos punts del flux")
    for s in seqs:
        process_seq(s)
