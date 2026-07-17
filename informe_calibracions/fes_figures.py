# -*- coding: utf-8 -*-
"""Generador de les figures de senyals crus de l'informe comparatiu de calibracions.

Produeix els PNG que consumeix `gen_docx.py`:
  COLUMN (un panell per SEQ):  raw_doc_continu.png · raw_254_overlay.png · raw_uib_overlay.png
  BP     (un panell per SEQ):  bp_doc_continu.png  · bp_254_overlay.png  · bp_uib_overlay.png

Fonts de dades:
  - DOC cru continu: MasterFile, full 2-TOC (senyal cru) + full 4-TOC_CALC (mapa
    fila TOC -> injeccio i temps relatiu). Es el senyal SENSE corregir de base.
  - 254 nm i UIB per injeccio: CHECK/data/calibration_result.json (replicas_info
    -> t_dad/y_dad_254; replicas_info_uib -> t_doc/y_doc).

Us:
    python -X utf8 fes_figures.py           # nomes COLUMN (per defecte)
    python -X utf8 fes_figures.py --bp      # tambe les de BP
    python -X utf8 fes_figures.py --tot
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = os.path.dirname(os.path.abspath(__file__))
DADES = r"C:\Users\maria\Proyectos\Dades3"

SEQS_COLUMN = ["293_SEQ_CAL", "305_SEQ_CAL", "306_SEQ_CAL"]
SEQS_BP = ["292_SEQ_CAL_BP", "304_SEQ_CAL_BP"]

# Color estable per concentracio: les mateixes ppm tenen el mateix color a totes
# les seqs, de manera que els panells es poden comparar d'un cop d'ull.
COLOR_CONC = {
    0.1: "#7F7F7F", 0.25: "#1F9BD8", 0.5: "#2CA02C", 1.0: "#FF9E1B",
    2.0: "#9467BD", 3.0: "#E8710A", 5.0: "#D62728",
}
UIB_FULL_SCALE = 999.63


def color_de(conc):
    return COLOR_CONC.get(float(conc), "#555555")


# ---------------------------------------------------------------- MasterFile
def _master_path(seq):
    hits = glob.glob(os.path.join(DADES, seq, "*MasterFile*.xlsx"))
    if not hits:
        raise FileNotFoundError(f"Sense MasterFile a {seq}")
    return hits[0]


def _find_header(path, sheet, marker, nrows=30):
    probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=nrows)
    for i in range(len(probe)):
        if any(marker in str(v) for v in probe.iloc[i].tolist()):
            return i
    raise ValueError(f"No s'ha trobat '{marker}' al full {sheet} de {path}")


def doc_cru_per_injeccio(seq):
    """[(sample, conc, rep, t_rel, y_raw), ...] en ordre d'adquisicio, nomes KHP."""
    path = _master_path(seq)
    h = _find_header(path, "2-TOC", "Result ID")
    toc = pd.read_excel(path, sheet_name="2-TOC", header=h)
    toc.columns = [str(c).strip() for c in toc.columns]
    toc = toc[pd.to_numeric(toc["TOC(ppb)"], errors="coerce").notna()].reset_index(drop=True)
    y_all = pd.to_numeric(toc["TOC(ppb)"], errors="coerce").to_numpy(dtype=float)

    calc = pd.read_excel(path, sheet_name="4-TOC_CALC", header=0)
    calc.columns = [str(c).strip() for c in calc.columns]
    calc = calc.dropna(subset=["Sample"])
    calc["Inj_Index"] = pd.to_numeric(calc["Inj_Index"], errors="coerce")
    calc["TOC_Row"] = pd.to_numeric(calc["TOC_Row"], errors="coerce")
    calc["Temps_Relatiu (min)"] = pd.to_numeric(calc["Temps_Relatiu (min)"], errors="coerce")

    # TOC_Row es 1-based sobre el full sencer; l'offset fins a la 1a fila de dades
    # es dedueix de la fila de capcalera trobada.
    offset = h + 2

    out = []
    for inj, g in calc.groupby("Inj_Index", sort=True):
        sample = str(g["Sample"].iloc[0])
        parsed = _parse_khp(sample)
        if parsed is None:
            continue  # nomes patrons KHP (fora MQ, NaOH, ...)
        conc, rep = parsed
        rows = (g["TOC_Row"].to_numpy(dtype=float) - offset).astype(int)
        t_rel = g["Temps_Relatiu (min)"].to_numpy(dtype=float)
        ok = (rows >= 0) & (rows < len(y_all)) & np.isfinite(t_rel)
        if ok.sum() < 10:
            continue
        out.append((sample, conc, rep, t_rel[ok], y_all[rows[ok]]))
    return out


# Codi del nom -> concentracio en ppm (els noms son KHP{codi}_R{replica})
_TAULA_CONC = {"01": 0.1, "025": 0.25, "05": 0.5, "1": 1.0, "2": 2.0, "3": 3.0, "5": 5.0}


def _parse_khp(sample):
    """'KHP025_R1' -> (0.25, 1). Retorna None si no es un patro KHP."""
    m = re.match(r"KHP(\d+)_R(\d+)$", sample, re.I)
    if not m:
        return None
    codi, rep = m.group(1), int(m.group(2))
    if codi not in _TAULA_CONC:
        raise ValueError(f"Codi de concentracio desconegut a '{sample}': KHP{codi}")
    return _TAULA_CONC[codi], rep


# ------------------------------------------------------- calibration_result
def cal_json(seq):
    p = os.path.join(DADES, seq, "CHECK", "data", "calibration_result.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def senyals_254(seq):
    """(conc, rep, t_dad, y_254) per injeccio. El y_dad_254 del JSON es el senyal CRU."""
    d = cal_json(seq)
    out = []
    for cal in d.get("calibrations") or []:
        conc = cal.get("conc_ppm")
        for rep in cal.get("replicas_info") or []:
            t, y = rep.get("t_dad"), rep.get("y_dad_254")
            if t is None or y is None or len(t) < 10:
                continue
            out.append((conc, rep.get("replica_num"),
                        np.asarray(t, float), np.asarray(y, float)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def senyals_uib(seq):
    """(conc, rep, t, y_uib) llegint els CSV CRUS de la carpeta CSV/.

    El `replicas_info_uib` del JSON porta el senyal ja corregit de base i NO
    serveix per a aquesta figura: aqui interessa el nivell absolut (fons i
    saturacio al fons d'escala del detector, 999,63). Els CSV son UTF-16,
    separats per tabulador i sense capcalera: temps(min) <TAB> valor.
    """
    out = []
    for path in glob.glob(os.path.join(DADES, seq, "CSV", "*UIB*.CSV")):
        m = re.match(r"(KHP\d+)_(\d+)_UIB", os.path.basename(path), re.I)
        if not m:
            continue
        parsed = _parse_khp(f"{m.group(1)}_R{m.group(2)}")
        if parsed is None:
            continue
        conc, rep = parsed
        df = pd.read_csv(path, sep="\t", header=None, names=["t", "y"], encoding="utf-16")
        t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        out.append((conc, rep, t[ok], y[ok]))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def senyals_per_injeccio(seq, quin):
    return senyals_254(seq) if quin == "254" else senyals_uib(seq)


# ------------------------------------------------------------------ figures
def fig_doc_continu(seqs, out_png, titol_mode):
    fig, axes = plt.subplots(len(seqs), 1, figsize=(16, 4.2 * len(seqs)))
    if len(seqs) == 1:
        axes = [axes]
    for ax, seq in zip(axes, seqs):
        injs = doc_cru_per_injeccio(seq)
        t0 = 0.0
        for sample, conc, rep, t, y in injs:
            tt = t - t[0] + t0
            c = color_de(conc)
            ax.plot(tt, y, color=c, lw=0.7)
            i = int(np.argmax(y))
            ax.annotate(f"{conc:g}·R{rep}", xy=(tt[i], y[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=c)
            t0 = tt[-1]
        ax.set_title(f"{seq} — DOC cru continu (2-TOC), injeccions KHP en ordre d'adquisició",
                     fontsize=12)
        ax.set_ylabel("DOC cru (ppb)", fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("temps concatenat (min)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, out_png), dpi=110)
    plt.close(fig)
    print("  ", out_png)


def fig_overlay(seqs, quin, out_png, ylabel, titol):
    fig, axes = plt.subplots(1, len(seqs), figsize=(6.2 * len(seqs), 5.2))
    if len(seqs) == 1:
        axes = [axes]
    for ax, seq in zip(axes, seqs):
        sers = senyals_per_injeccio(seq, quin)
        ymax_glob = -np.inf
        for conc, rep, t, y in sers:
            ax.plot(t - t[0], y, color=color_de(conc), lw=0.9,
                    label=f"{conc:g} ppm R{rep}")
            ymax_glob = max(ymax_glob, float(np.nanmax(y)))
        if quin == "254":
            ax.axhline(0, color="#888", ls="--", lw=0.7)
        else:
            ax.annotate(f"y_max={UIB_FULL_SCALE:g}", xy=(0.98, 0.96),
                        xycoords="axes fraction", ha="right", fontsize=8, color="#777")
        ax.set_title(f"{seq} — {titol}", fontsize=12)
        ax.set_xlabel("temps des de l'inici de la finestra (min)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, out_png), dpi=110)
    plt.close(fig)
    print("  ", out_png)


def main():
    fer_bp = "--bp" in sys.argv or "--tot" in sys.argv
    print("Figures COLUMN:")
    fig_doc_continu(SEQS_COLUMN, "raw_doc_continu.png", "COLUMN")
    fig_overlay(SEQS_COLUMN, "254", "raw_254_overlay.png",
                "254 nm (mAU)", "254 nm (DAD) cru per injecció (superposat)")
    fig_overlay(SEQS_COLUMN, "uib", "raw_uib_overlay.png",
                "UIB cru", "UIB cru per injecció (superposat)")
    if fer_bp:
        print("Figures BP:")
        fig_doc_continu(SEQS_BP, "bp_doc_continu.png", "BP")
        fig_overlay(SEQS_BP, "254", "bp_254_overlay.png",
                    "254 nm (mAU)", "254 nm (DAD) cru per injecció (superposat)")
        fig_overlay(SEQS_BP, "uib", "bp_uib_overlay.png",
                    "UIB cru", "UIB cru per injecció (superposat)")
    else:
        print("Figures BP NO regenerades (les publicades es mantenen; --bp per refer-les).")


if __name__ == "__main__":
    main()
