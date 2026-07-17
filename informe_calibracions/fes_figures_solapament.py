# -*- coding: utf-8 -*-
"""Solapament dels pics injeccio a injeccio, canals DOC i 254 nm.

Prova de coincidencia: totes les injeccions d'una seqüencia han d'eluir al mateix
temps i, un cop normalitzades a l'alcada, han de tenir la MATEIXA forma. Si no
coincideixen, la seqüencia te un problema (deriva de retencio, sobrecarrega,
degradacio del llit o integracio inconsistent).

Genera:
  sol_doc_solapat.png   — DOC: cru i normalitzat, un panell per seqüencia
  sol_254_solapat.png   — 254 nm: cru i normalitzat, un panell per seqüencia
  sol_coincidencia.png  — metriques de coincidencia (t_ret i FWHM per concentracio)

Us:  python -X utf8 fes_figures_solapament.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fes_figures import color_de

SCR = os.path.dirname(os.path.abspath(__file__))
DADES = r"C:\Users\maria\Proyectos\Dades3"
SEQS = ["293_SEQ_CAL", "305_SEQ_CAL", "306_SEQ_CAL"]


def cal(seq):
    p = os.path.join(DADES, seq, "CHECK", "data", "calibration_result.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _base_local(t, y, t_lo, t_hi, pad=2.0):
    m = ((t >= t_lo - pad) & (t < t_lo)) | ((t > t_hi) & (t <= t_hi + pad))
    if m.sum() < 4:
        return np.full_like(t, float(np.median(y)))
    return np.polyval(np.polyfit(t[m], y[m], 1), t)


def pics(seq, canal, hw=4.0):
    """[(conc, rep, t_rel_al_maxim, y_net)] retallats al voltant del pic.

    canal='doc' -> replicas_info (t_doc/y_doc) · canal='254' -> t_dad/y_dad_254.
    El senyal es retorna net de base local, sense centrar: aixi el solapament
    mostra si els pics COINCIDEIXEN en temps.
    """
    out = []
    d = cal(seq)
    for c in d.get("calibrations") or []:
        conc = c.get("conc_ppm")
        for rep in c.get("replicas_info") or []:
            if canal == "doc":
                t, y = rep.get("t_doc"), rep.get("y_doc")
                jan = (18, 30)
            else:
                t, y = rep.get("t_dad"), rep.get("y_dad_254")
                jan = (18, 28)
            if t is None or y is None or len(t) < 30:
                continue
            t = np.asarray(t, float); y = np.asarray(y, float)
            m = (t >= jan[0]) & (t <= jan[1])
            if m.sum() < 5:
                continue
            i = np.where(m)[0][int(np.argmax(y[m]))]
            t_max = float(t[i])
            w = (t >= t_max - hw) & (t <= t_max + hw)
            bl = _base_local(t, y, t_max - 1.2, t_max + 1.2)
            out.append((conc, rep.get("replica_num"), t_max, t[w], y[w] - bl[w]))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def fig_solapat(canal, out_png, ylabel, titol):
    fig, axes = plt.subplots(2, len(SEQS), figsize=(6.2 * len(SEQS), 8.4))
    for j, seq in enumerate(SEQS):
        ps = pics(seq, canal)
        # --- fila 0: cru (mostra si coincideixen en temps i com escalen) ---
        ax = axes[0][j]
        for conc, rep, t_max, t, y in ps:
            ax.plot(t, y, color=color_de(conc), lw=1.0,
                    label=f"{conc:g} ppm R{rep}")
        ax.set_title(f"{seq} — {titol} (cru)", fontsize=11, loc="left")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2, framealpha=0.85)
        # --- fila 1: normalitzat a l'alcada (mostra si tenen la MATEIXA forma) ---
        ax = axes[1][j]
        t_refs = [tm for _, _, tm, _, _ in ps]
        t_ref = float(np.median(t_refs)) if t_refs else 0.0
        for conc, rep, t_max, t, y in ps:
            h = float(np.max(y)) if y.size else 0
            if h <= 0:
                continue
            ax.plot(t - t_ref, y / h, color=color_de(conc), lw=1.0, alpha=0.85)
        ax.axvline(0, color="#888", ls=":", lw=0.9)
        ax.axhline(0.5, color="#BBB", ls="--", lw=0.7)
        ax.set_title(f"{seq} — normalitzat a l'alçada", fontsize=11, loc="left")
        ax.set_ylabel("senyal / alçada", fontsize=9)
        ax.set_xlabel(f"t − {t_ref:.2f} min", fontsize=9)
        ax.set_xlim(-3, 3)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Solapament de pics injecció a injecció — {titol}", fontsize=13, y=0.998)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, out_png), dpi=110)
    plt.close(fig)
    print("  ", out_png)


def metriques(seq, canal):
    """[(conc, rep, t_max, fwhm, alcada)] per injeccio."""
    res = []
    for conc, rep, t_max, t, y in pics(seq, canal):
        if y.size < 5 or np.max(y) <= 0:
            continue
        h = float(np.max(y))
        half = h / 2
        i = int(np.argmax(y))
        li = ri = i
        while li > 0 and y[li] > half:
            li -= 1
        while ri < len(y) - 1 and y[ri] > half:
            ri += 1
        res.append((conc, rep, t_max, float(t[ri] - t[li]), h))
    return res


def fig_coincidencia():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    marca = {"293_SEQ_CAL": ("o", "#7F7F7F"), "305_SEQ_CAL": ("s", "#1F9BD8"),
             "306_SEQ_CAL": ("^", "#D62728")}
    for col, canal in enumerate(["doc", "254"]):
        nom = "DOC" if canal == "doc" else "254 nm"
        for seq in SEQS:
            M = metriques(seq, canal)
            if not M:
                continue
            mk, cl = marca[seq]
            cs = [m[0] for m in M]
            tr = [m[2] for m in M]
            fw = [m[3] for m in M]
            axes[0][col].plot(cs, tr, mk, color=cl, ms=6, alpha=0.8, label=seq[:3])
            axes[1][col].plot(cs, fw, mk, color=cl, ms=6, alpha=0.8, label=seq[:3])
        axes[0][col].set_title(f"{nom} — temps de retenció per injecció", fontsize=11, loc="left")
        axes[0][col].set_ylabel("t del màxim (min)", fontsize=9)
        axes[1][col].set_title(f"{nom} — FWHM per injecció", fontsize=11, loc="left")
        axes[1][col].set_ylabel("FWHM (min)", fontsize=9)
        for r in (0, 1):
            axes[r][col].set_xscale("log")
            axes[r][col].set_xlabel("concentració (ppm)", fontsize=9)
            axes[r][col].set_xticks([0.1, 0.25, 0.5, 1, 2, 3, 5])
            axes[r][col].set_xticklabels(["0,1", "0,25", "0,5", "1", "2", "3", "5"])
            axes[r][col].grid(alpha=0.25, which="both")
            axes[r][col].legend(fontsize=8)
    fig.suptitle("Coincidència entre injeccions: si la columna i la integració són estables, "
                 "els punts han de ser plans", fontsize=12, y=0.998)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "sol_coincidencia.png"), dpi=110)
    plt.close(fig)
    print("   sol_coincidencia.png")


def informe_text():
    print("\n" + "=" * 104)
    print("COINCIDENCIA DELS PICS DINS DE CADA SEQÜENCIA")
    print("=" * 104)
    for canal in ["doc", "254"]:
        nom = "DOC" if canal == "doc" else "254 nm"
        print(f"\n--- Canal {nom} ---")
        print(f"{'seq':<16}{'t_ret mitja':>13}{'dispersio':>12}{'rang':>9} | "
              f"{'FWHM mitja':>12}{'dispersio':>12}{'rang FWHM':>11}")
        print("-" * 92)
        for seq in SEQS:
            M = metriques(seq, canal)
            if not M:
                continue
            tr = np.array([m[2] for m in M])
            fw = np.array([m[3] for m in M])
            print(f"{seq:<16}{tr.mean():>13.3f}{tr.std():>12.3f}"
                  f"{tr.max()-tr.min():>9.3f} | {fw.mean():>12.3f}{fw.std():>12.3f}"
                  f"{fw.max()-fw.min():>11.3f}")

    print("\n" + "=" * 104)
    print("FWHM PER CONCENTRACIO (si creix amb la concentracio -> sobrecarrega de columna)")
    print("=" * 104)
    for canal in ["doc", "254"]:
        nom = "DOC" if canal == "doc" else "254 nm"
        print(f"\n--- Canal {nom} ---")
        concs = [0.1, 0.25, 0.5, 1, 2, 3, 5]
        print(f"{'seq':<16}" + "".join(f"{f'{c:g}':>9}" for c in concs) + f"{'5/baix':>10}")
        print("-" * 82)
        for seq in SEQS:
            M = metriques(seq, canal)
            per = {}
            for c, r, tm, fw, h in M:
                per.setdefault(c, []).append(fw)
            linia = f"{seq:<16}"
            vals = []
            for c in concs:
                if c in per:
                    v = float(np.mean(per[c])); vals.append((c, v))
                    linia += f"{v:>9.3f}"
                else:
                    linia += f"{'—':>9}"
            if len(vals) >= 2:
                linia += f"{vals[-1][1]/vals[0][1]:>10.2f}"
            print(linia)


def main():
    print("Figures de solapament:")
    fig_solapat("doc", "sol_doc_solapat.png", "DOC net (ppb)", "canal DOC")
    fig_solapat("254", "sol_254_solapat.png", "254 nm net (mAU)", "canal 254 nm")
    fig_coincidencia()
    informe_text()


if __name__ == "__main__":
    main()
