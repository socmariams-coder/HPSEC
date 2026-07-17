# -*- coding: utf-8 -*-
"""Figures de l'informe d'integracio (305 vs 306).

Mateix estil que les figures de l'informe comparatiu de calibracions.
Genera:
  int_doc_continu.png    — DOC cru continu, un panell per seqüencia
  int_finestres.png      — cromatograma amb finestra de la Suite vs finestra estreta
  int_sobreintegracio.png— factor de sobre-integracio i amplada/FWHM per concentracio
  int_rectes.png         — recta de calibratge amb cada metode d'integracio

Us:  python -X utf8 fes_figures_integracio.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compara_integracions import recull, recta, VOL_UL
from fes_figures import doc_cru_per_injeccio, color_de

SCR = os.path.dirname(os.path.abspath(__file__))
# Totes les seqüencies de COLUMN a TOTES les figures, perque siguin comparables
# panell a panell. La 293 aporta la referencia de base plana.
SEQS = ["293_SEQ_CAL", "305_SEQ_CAL", "306_SEQ_CAL"]
# Unio de concentracions: la 293 te 0,1 i 2 ppm; la 305/306 tenen 3 ppm.
CONCS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
C_SUITE, C_EST = "#E74C3C", "#27AE60"


def fig_doc_continu():
    """Mateix estil que raw_doc_continu.png, nomes 305 i 306."""
    fig, axes = plt.subplots(len(SEQS), 1, figsize=(16, 4.2 * len(SEQS)))
    for ax, seq in zip(axes, SEQS):
        t0 = 0.0
        for sample, conc, rep, t, y in doc_cru_per_injeccio(seq):
            tt = t - t[0] + t0
            c = color_de(conc)
            ax.plot(tt, y, color=c, lw=0.7)
            i = int(np.argmax(y))
            ax.annotate(f"{conc:g}·R{rep}", xy=(tt[i], y[i]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5, color=c)
            t0 = tt[-1]
        ax.set_title(f"{seq} — DOC cru continu (2-TOC), injeccions KHP en ordre d'adquisició",
                     fontsize=12)
        ax.set_ylabel("DOC cru (ppb)", fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("temps concatenat (min)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "int_doc_continu.png"), dpi=110)
    plt.close(fig)
    print("   int_doc_continu.png")


def fig_finestres(dades):
    fig, axes = plt.subplots(len(CONCS), len(SEQS), figsize=(7.5 * len(SEQS), 2.6 * len(CONCS)))
    for i, conc in enumerate(CONCS):
        for j, seq in enumerate(SEQS):
            ax = axes[i][j]
            files, _, _ = dades[seq]
            f = next((x for x in files if x["conc"] == conc and x["rep"] == 1), None)
            if f is None or f["est"] is None:
                # Concentracio no present en aquesta seq (0,1 i 2 ppm nomes a la 293;
                # 3 ppm nomes a la 305/306): panell buit i retolat, no amagat, per
                # mantenir la graella comparable.
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#DDD")
                ax.text(0.5, 0.5, f"{seq[:3]} — {conc:g} ppm\n(no present)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8.5, color="#AAA")
                continue
            e = f["est"]
            ax.plot(f["t"], f["y"], color="#2563EB", lw=0.8)
            ax.axvspan(f["suite_lo"], f["suite_hi"], color=C_SUITE, alpha=0.12,
                       label=f"Suite {f['suite_ampl']:.0f} min ({f['suite_ampl']/e['fwhm']:.0f}×FWHM) → A={f['suite_area']:.0f}")
            ax.axvspan(e["t_lo"], e["t_hi"], color=C_EST, alpha=0.30,
                       label=f"estreta {e['t_hi']-e['t_lo']:.1f} min (4×FWHM) → A={e['area']:.0f}")
            if f["t254"] and f["fiable254"]:
                ax.axvline(f["t254"], color="#8E44AD", ls=":", lw=1.1,
                           label="pic 254 nm (àncora)")
            ax.set_title(f"{seq[:3]} — {conc:g} ppm R1 (FWHM {e['fwhm']:.2f} min)",
                         fontsize=9.5, loc="left")
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
            ax.grid(alpha=0.22)
            ax.set_ylabel("DOC (ppb)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (min)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "int_finestres.png"), dpi=110)
    plt.close(fig)
    print("   int_finestres.png")


def fig_sobreintegracio(dades):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    marca = {"293_SEQ_CAL": ("o", "#7F7F7F"), "305_SEQ_CAL": ("s", "#1F9BD8"),
             "306_SEQ_CAL": ("^", "#D62728")}
    for seq in SEQS:
        files, _, _ = dades[seq]
        per = {}
        for f in files:
            if not f["est"] or f["est"]["area"] <= 0:
                continue
            per.setdefault(f["conc"], []).append(
                (f["suite_area"] / f["est"]["area"], f["suite_ampl"] / f["est"]["fwhm"]))
        cs = sorted(per)
        r = [float(np.mean([v[0] for v in per[c]])) for c in cs]
        w = [float(np.mean([v[1] for v in per[c]])) for c in cs]
        m, col = marca[seq]
        axes[0].plot(cs, r, m + "-", color=col, label=seq[:3], ms=6)
        axes[1].plot(cs, w, m + "-", color=col, label=seq[:3], ms=6)
    axes[0].axhline(1.0, color="#333", ls="--", lw=0.8, label="sense sobre-integració")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("àrea Suite / àrea estreta", fontsize=10)
    axes[0].set_title("Factor de sobre-integració", fontsize=11, loc="left")
    axes[1].axhline(4.0, color="#333", ls="--", lw=0.8, label="finestra correcta (4×FWHM)")
    axes[1].set_ylabel("amplada de la finestra / FWHM", fontsize=10)
    axes[1].set_title("Amplada de la finestra de la Suite", fontsize=11, loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("concentració (ppm)", fontsize=10)
        ax.set_xticks(CONCS + [0.1, 2.0])
        ax.set_xticklabels([f"{c:g}" for c in CONCS + [0.1, 2.0]])
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "int_sobreintegracio.png"), dpi=110)
    plt.close(fig)
    print("   int_sobreintegracio.png")


def fig_rectes(dades):
    fig, axes = plt.subplots(1, len(SEQS), figsize=(6.5 * len(SEQS), 4.8))
    for ax, seq in zip(axes, SEQS):
        files, _, _ = dades[seq]
        ps, pe = {}, {}
        for f in files:
            ps.setdefault(f["conc"], []).append(f["suite_area"])
            if f["est"]:
                pe.setdefault(f["conc"], []).append(f["est"]["area"])
        for nom, dat, col, mk in (("Suite (vigent)", ps, C_SUITE, "o"),
                                  ("estreta ancorada", pe, C_EST, "s")):
            cs = sorted(dat)
            ug = np.array([c * VOL_UL / 1000 for c in cs])
            ar = np.array([float(np.mean(dat[c])) for c in cs])
            r = recta(list(zip(ug, ar)))
            ax.plot(ug, ar, mk, color=col, ms=7,
                    label=f"{nom}: RF={r[0]:.0f}, R²={r[2]:.4f}")
            xx = np.linspace(0, ug.max() * 1.05, 50)
            ax.plot(xx, r[0] * xx + r[1], "-", color=col, lw=1.2, alpha=0.7)
        ax.set_title(f"{seq}", fontsize=11, loc="left")
        ax.set_xlabel("µg DOC injectats", fontsize=10)
        ax.set_ylabel("àrea integrada", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "int_rectes.png"), dpi=110)
    plt.close(fig)
    print("   int_rectes.png")


def main():
    print("Figures de l'informe d'integració:")
    dades = {s: recull(s) for s in SEQS}
    fig_doc_continu()
    fig_finestres(dades)
    fig_sobreintegracio(dades)
    fig_rectes(dades)


if __name__ == "__main__":
    main()
