# -*- coding: utf-8 -*-
"""Rectes de regressio amb TOTES les injeccions ben integrades.

Cada injeccio s'integra amb la finestra estreta ancorada al pic de 254 nm
(+-2*FWHM, base local), de manera que la finestra s'ajusta al pic i no s'escapa.
Es mostren els punts individuals de cada replica, la recta ajustada, R2, RF_mass
i l'ordenada. Les injeccions sense pic real (avortades) es marquen i s'exclouen.

Genera:
  reg_rectes.png       — recta per seqüencia, punts individuals + ajust
  reg_punts.png        — area integrada per injeccio (dispersio entre repliques)

Us:  python -X utf8 fes_figures_regressio.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compara_integracions import recull, VOL_UL

SCR = os.path.dirname(os.path.abspath(__file__))
SEQS = ["293_SEQ_CAL", "305_SEQ_CAL", "306_SEQ_CAL"]
COLORS = {"293_SEQ_CAL": "#7F7F7F", "305_SEQ_CAL": "#1F9BD8", "306_SEQ_CAL": "#D62728"}

# Injeccio avortada (sense pic de KHP): 306, 0,25 ppm R2. Es detecta sola pel
# criteri d'alcada, pero la marquem explicitament per traçabilitat.
SNR_MIN = 20.0     # per sota d'aixo, no hi ha pic utilitzable


def punts(seq):
    """[(conc, rep, ug, area, ok)] amb integracio estreta; ok=False si avortada."""
    files, _, _ = recull(seq)
    out = []
    for f in files:
        e = f["est"]
        if e is None:
            continue
        ug = f["conc"] * f.get("vol", VOL_UL) / 1000.0
        # avortada: alçada neta molt petita respecte del que s'espera a la seva conc
        h = e.get("h_net", 0)
        ok = h > 30 and e["area"] > 0        # 30 ppb: llindar de pic real
        out.append(dict(conc=f["conc"], rep=f["rep"], ug=ug,
                        area=e["area"], h=h, ok=ok))
    return out


def ajusta(pts):
    v = [p for p in pts if p["ok"]]
    if len(v) < 3:
        return None
    ug = np.array([p["ug"] for p in v])
    ar = np.array([p["area"] for p in v])
    S, b = np.polyfit(ug, ar, 1)
    pred = S * ug + b
    ss_res = float(np.sum((ar - pred) ** 2))
    ss_tot = float(np.sum((ar - np.mean(ar)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return dict(S=float(S), b=float(b), r2=r2, n=len(v))


def fig_rectes():
    dades = {s: punts(s) for s in SEQS}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, seq in zip(axes, SEQS):
        pts = dades[seq]
        fit = ajusta(pts)
        cl = COLORS[seq]
        # punts vàlids (cada rèplica per separat)
        vx = [p["ug"] for p in pts if p["ok"]]
        vy = [p["area"] for p in pts if p["ok"]]
        ax.scatter(vx, vy, s=55, color=cl, edgecolor="white", zorder=3,
                   label="injeccions integrades")
        # punts avortats
        ax_bad = [(p["ug"], p["area"], p) for p in pts if not p["ok"]]
        for ug, ar, p in ax_bad:
            ax.scatter([ug], [ar], s=90, marker="x", color="#000", zorder=4,
                       label=f"avortada ({p['conc']:g} ppm R{p['rep']})")
        if fit:
            xx = np.linspace(0, max(vx) * 1.05, 50)
            ax.plot(xx, fit["S"] * xx + fit["b"], "-", color=cl, lw=1.6, alpha=0.8)
            eq = (f"àrea = {fit['S']:.0f}·µg {fit['b']:+.0f}\n"
                  f"R² = {fit['r2']:.4f}   (n = {fit['n']})")
            ax.text(0.04, 0.96, eq, transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="#F4F8FB", ec=cl, alpha=0.9))
        ax.set_title(seq, fontsize=12, loc="left")
        ax.set_xlabel("µg DOC injectats", fontsize=10)
        ax.set_ylabel("àrea integrada (finestra estreta)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Rectes de calibratge amb TOTES les injeccions ben integrades "
                 "(finestra estreta ancorada al 254 nm)", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "reg_rectes.png"), dpi=110)
    plt.close(fig)
    print("   reg_rectes.png")
    return dades


def fig_punts(dades):
    """Area per concentracio: les dues repliques i la seva concordança."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seq in SEQS:
        pts = [p for p in dades[seq] if p["ok"]]
        cl = COLORS[seq]
        per = {}
        for p in pts:
            per.setdefault(p["conc"], []).append(p["area"] / p["ug"])  # RF_mass per inj
        cs = sorted(per)
        for c in cs:
            for v in per[c]:
                ax.scatter([c], [v], s=40, color=cl, alpha=0.7)
        mitj = [np.mean(per[c]) for c in cs]
        ax.plot(cs, mitj, "-", color=cl, lw=1.3, label=seq[:3])
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0,1", "0,25", "0,5", "1", "2", "3", "5"])
    ax.set_xlabel("concentració (ppm)", fontsize=10)
    ax.set_ylabel("RF_mass per injecció (àrea / µg)", fontsize=10)
    ax.set_title("Factor de resposta per injecció, integració estreta "
                 "(pla = recta ideal)", fontsize=11, loc="left")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "reg_punts.png"), dpi=110)
    plt.close(fig)
    print("   reg_punts.png")


def informe():
    print("=" * 92)
    print("RECTES AMB TOTES LES INJECCIONS BEN INTEGRADES (finestra estreta ancorada)")
    print("=" * 92)
    dades = {s: punts(s) for s in SEQS}
    for seq in SEQS:
        pts = dades[seq]
        fit = ajusta(pts)
        print(f"\n{seq}")
        print(f"{'ppm':>6}{'rep':>5}{'µg DOC':>9}{'àrea':>11}{'h_net':>9}{'estat':>10}")
        print("-" * 52)
        for p in sorted(pts, key=lambda x: (x["conc"], x["rep"])):
            estat = "OK" if p["ok"] else "AVORTADA"
            print(f"{p['conc']:>6g}{p['rep']:>5}{p['ug']:>9.2f}{p['area']:>11.1f}"
                  f"{p['h']:>9.1f}{estat:>10}")
        if fit:
            print(f"  → recta: RF_mass = {fit['S']:.0f}   ordenada = {fit['b']:+.1f}   "
                  f"R² = {fit['r2']:.4f}   (n = {fit['n']} injeccions)")
    return dades


if __name__ == "__main__":
    dades = informe()
    print("\nFigures:")
    fig_rectes()
    fig_punts(dades)
