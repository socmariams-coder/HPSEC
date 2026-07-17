# -*- coding: utf-8 -*-
"""LOD i LOQ de les seqüencies de calibratge, per tres criteris independents.

Motiu: el senyal del DOC s'ha multiplicat per ~8, pero el soroll i el fons tambe.
La pregunta rellevant no es quant ha pujat el senyal sino qui guanya la cursa
entre senyal i soroll, que es el que fixa el limit de deteccio.

Criteris:
  A. Soroll (alcada)     LOD = 3·s/S_h   LOQ = 10·s/S_h
     s  = soroll instrumental estimat de forma immune a la deriva:
          s = SD(diff(y))/sqrt(2) sobre una finestra sense pic.
     S_h= pendent de la recta alcada neta (ppb) vs concentracio (ppm).
  B. Regressio (ICH Q2)  LOD = 3,3·Sy/S_a  LOQ = 10·Sy/S_a
     Sy = error estandard dels residus de la recta area vs ppm.
     S_a= pendent de la recta area vs ppm.
  C. Blanc (MQ)          LOD = 3·SD_blanc/S_a  LOQ = 10·SD_blanc/S_a
     SD_blanc = dispersio de l'area integrada a la finestra del pic en injeccions
                d'aigua MQ (sense KHP).

A mes es calcula el LOQ EMPIRIC: la concentracio mes baixa on les repliques
concorden (RSD <= 10%) i la recuperacio es raonable. Es el limit real d'us.

Us:  python -X utf8 calcula_lod_loq.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

DADES = r"C:\Users\maria\Proyectos\Dades3"
SEQS = ["293_SEQ_CAL", "305_SEQ_CAL", "306_SEQ_CAL"]
VOL_UL = 400.0
TAULA = {"01": 0.1, "025": 0.25, "05": 0.5, "1": 1.0, "2": 2.0, "3": 3.0, "5": 5.0}
FIN_SORROLL = (5.0, 18.0)   # finestra sense pic (el KHP surt a 21-24 min)
N_FWHM = 2.0


def injeccions(seq):
    """[(nom, conc|None, rep, t, y_raw)] de totes les injeccions de la seq."""
    p = glob.glob(os.path.join(DADES, seq, "*MasterFile*.xlsx"))[0]
    probe = pd.read_excel(p, sheet_name="2-TOC", header=None, nrows=30)
    h = next(i for i in range(len(probe))
             if any("Result ID" in str(v) for v in probe.iloc[i].tolist()))
    toc = pd.read_excel(p, sheet_name="2-TOC", header=h)
    toc.columns = [str(c).strip() for c in toc.columns]
    toc = toc[pd.to_numeric(toc["TOC(ppb)"], errors="coerce").notna()].reset_index(drop=True)
    y_all = pd.to_numeric(toc["TOC(ppb)"], errors="coerce").to_numpy(dtype=float)
    calc = pd.read_excel(p, sheet_name="4-TOC_CALC", header=0)
    calc.columns = [str(c).strip() for c in calc.columns]
    calc = calc.dropna(subset=["Sample"])
    off = h + 2
    out = []
    for _, g in calc.groupby("Inj_Index", sort=True):
        s = str(g["Sample"].iloc[0])
        rows = (pd.to_numeric(g["TOC_Row"], errors="coerce").to_numpy(float) - off).astype(int)
        t = pd.to_numeric(g["Temps_Relatiu (min)"], errors="coerce").to_numpy(float)
        ok = (rows >= 0) & (rows < len(y_all)) & np.isfinite(t)
        if ok.sum() < 50:
            continue
        m = re.match(r"KHP(\d+)_R(\d+)$", s, re.I)
        if m and m.group(1) in TAULA:
            out.append((s, TAULA[m.group(1)], int(m.group(2)), t[ok], y_all[rows[ok]]))
        elif re.match(r"MQ", s, re.I):
            out.append((s, None, 0, t[ok], y_all[rows[ok]]))
    return out


def soroll(t, y):
    """Soroll instrumental sobre una finestra sense pic.

    Es lleva la deriva amb un ajust lineal i es retalla el 5% extrem de cada
    costat, que correspon a les punxes de recarrega de xeringues del TOC
    (timeout) i no es soroll de mesura.
    """
    m = (t >= FIN_SORROLL[0]) & (t <= FIN_SORROLL[1])
    if m.sum() < 20:
        return np.nan
    tt, yy = t[m], y[m]
    resid = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
    lo, hi = np.percentile(resid, [5, 95])
    r = resid[(resid >= lo) & (resid <= hi)]
    return float(np.std(r, ddof=1)) if len(r) > 10 else np.nan


def pas_quantitzacio(t, y):
    """Resolucio de lectura del TOC: menor diferencia no nul·la entre mesures
    consecutives. El TOC reporta ~3 xifres significatives, de manera que el pas
    creix amb el nivell de senyal i acaba fixant un terra de soroll propi."""
    m = (t >= FIN_SORROLL[0]) & (t <= FIN_SORROLL[1])
    d = np.abs(np.diff(y[m]))
    d = d[d > 0]
    return float(np.min(d)) if d.size else np.nan


def pic_i_area(t, y, t_ancora=None):
    """Alcada neta i area del pic principal (finestra ±2·FWHM, base local)."""
    m = (t >= 18) & (t <= 30)
    if m.sum() < 10:
        return np.nan, np.nan
    idx = np.where(m)[0]
    i = idx[int(np.argmax(y[m]))]
    bl0 = float(np.median(y[(t >= 5) & (t <= 18)]))
    half = (y[i] + bl0) / 2
    li = ri = i
    while li > idx[0] and y[li] > half:
        li -= 1
    while ri < idx[-1] and y[ri] > half:
        ri += 1
    fwhm = max(float(t[ri] - t[li]), 0.5)
    lo, hi = float(t[i]) - N_FWHM * fwhm, float(t[i]) + N_FWHM * fwhm
    w = (t >= lo) & (t <= hi)
    fl = ((t >= lo - 1.5) & (t < lo)) | ((t > hi) & (t <= hi + 1.5))
    bl = (np.polyval(np.polyfit(t[fl], y[fl], 1), t) if fl.sum() >= 4
          else np.full_like(t, bl0))
    area = float(trapezoid(np.maximum(y[w] - bl[w], 0), t[w]))
    return float(y[i] - bl[i]), area


def area_finestra_blanc(t, y, t_pic, fwhm=1.4):
    """Area integrada a la finestra on sortiria el pic, en una injeccio de MQ."""
    lo, hi = t_pic - N_FWHM * fwhm, t_pic + N_FWHM * fwhm
    w = (t >= lo) & (t <= hi)
    fl = ((t >= lo - 1.5) & (t < lo)) | ((t > hi) & (t <= hi + 1.5))
    if w.sum() < 5 or fl.sum() < 4:
        return np.nan
    bl = np.polyval(np.polyfit(t[fl], y[fl], 1), t)
    return float(trapezoid(np.maximum(y[w] - bl[w], 0), t[w]))


def regressio(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    S, b = np.polyfit(x, y, 1)
    pred = S * x + b
    resid = y - pred
    # error estandard dels residus (n-2 graus de llibertat)
    Sy = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 2)))
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return {"S": float(S), "b": float(b), "Sy": Sy, "r2": float(r2), "n": len(x)}


print("=" * 116)
print("LOD i LOQ — mode COLUMN, injeccio de 400 µL")
print("=" * 116)

RES = {}
for seq in SEQS:
    injs = injeccions(seq)
    khp = [(c, r, t, y) for _, c, r, t, y in injs if c is not None]
    mq = [(t, y) for nom, c, r, t, y in injs if c is None]

    # Soroll instrumental i resolucio de lectura
    sor = [soroll(t, y) for _, _, t, y in khp]
    s_med = float(np.nanmedian(sor))
    q_med = float(np.nanmedian([pas_quantitzacio(t, y) for _, _, t, y in khp]))

    # Alcades i arees per injeccio
    dades = {}
    t_pics = []
    for c, r, t, y in khp:
        h, a = pic_i_area(t, y)
        dades.setdefault(c, []).append((h, a))
        m = (t >= 18) & (t <= 30)
        t_pics.append(float(t[m][int(np.argmax(y[m]))]))
    t_pic_med = float(np.median(t_pics))

    concs = sorted(dades)
    h_mit = [float(np.nanmean([v[0] for v in dades[c]])) for c in concs]
    a_mit = [float(np.nanmean([v[1] for v in dades[c]])) for c in concs]

    reg_h = regressio(concs, h_mit)
    reg_a = regressio(concs, a_mit)

    # Blancs
    ab = [area_finestra_blanc(t, y, t_pic_med) for t, y in mq]
    ab = [x for x in ab if np.isfinite(x)]
    sd_blanc = float(np.std(ab, ddof=1)) if len(ab) > 1 else np.nan
    mitj_blanc = float(np.mean(ab)) if ab else np.nan

    RES[seq] = dict(s=s_med, q=q_med, reg_h=reg_h, reg_a=reg_a, sd_blanc=sd_blanc,
                    mitj_blanc=mitj_blanc, concs=concs, dades=dades,
                    n_blancs=len(ab), base=float(np.median(
                        [np.median(y[(t >= FIN_SORROLL[0]) & (t <= FIN_SORROLL[1])])
                         for _, _, t, y in khp])))

    print(f"\n{'─'*116}\n{seq}\n{'─'*116}")
    print(f"  Línia de base a la finestra {FIN_SORROLL[0]:g}–{FIN_SORROLL[1]:g} min: "
          f"{RES[seq]['base']:.1f} ppb")
    print(f"  Soroll instrumental (SD, deriva llevada, extrems retallats): {s_med:.3f} ppb"
          f"   → soroll relatiu {s_med/RES[seq]['base']*100:.3f}% de la base")
    print(f"  Resolució de lectura del TOC (pas mínim): {q_med:.2f} ppb")
    print(f"  Recta ALÇADA: {reg_h['S']:.1f} ppb/ppm · ordenada {reg_h['b']:+.1f} · R² {reg_h['r2']:.4f}")
    print(f"  Recta ÀREA:   {reg_a['S']:.1f} /ppm · ordenada {reg_a['b']:+.1f} · "
          f"R² {reg_a['r2']:.4f} · Sy {reg_a['Sy']:.1f}")
    print(f"  Blancs MQ (n={len(ab)}): àrea mitjana {mitj_blanc:.1f} · SD {sd_blanc:.1f}")

print("\n" + "=" * 116)
print("RESULTATS — LOD i LOQ en ppm de la mostra injectada")
print("=" * 116)
print(f"{'seq':<16}{'criteri':<26}{'LOD (ppm)':>12}{'LOQ (ppm)':>12}   vs 293")
print("-" * 90)
base = {}
for seq in SEQS:
    R = RES[seq]
    crits = [
        ("A · soroll (alçada)", 3 * R["s"] / R["reg_h"]["S"], 10 * R["s"] / R["reg_h"]["S"]),
        ("B · regressió (ICH Q2)", 3.3 * R["reg_a"]["Sy"] / R["reg_a"]["S"],
         10 * R["reg_a"]["Sy"] / R["reg_a"]["S"]),
    ]
    if np.isfinite(R["sd_blanc"]):
        crits.append(("C · blanc MQ", 3 * R["sd_blanc"] / R["reg_a"]["S"],
                      10 * R["sd_blanc"] / R["reg_a"]["S"]))
    for nom, lod, loq in crits:
        if seq == "293_SEQ_CAL":
            base[nom] = (lod, loq)
            extra = "(referència)"
        else:
            b = base.get(nom)
            extra = (f"LOD ×{lod/b[0]:.1f} · LOQ ×{loq/b[1]:.1f}"
                     if b and b[0] > 0 and b[1] > 0 else "—")
        print(f"{seq:<16}{nom:<26}{lod:>12.4f}{loq:>12.4f}   {extra}")
    print()

print("=" * 116)
print("LOQ EMPÍRIC — concordança entre rèpliques (RSD de l'alçada neta) per concentració")
print("(el LOQ teòric no captura la irreproductibilitat; aquest sí. Criteri: RSD ≤ 10%)")
print("=" * 116)
print(f"{'seq':<16}" + "".join(f"{f'{c:g} ppm':>12}" for c in [0.1, 0.25, 0.5, 1, 2, 3, 5]))
print("-" * 100)
for seq in SEQS:
    R = RES[seq]
    linia = f"{seq:<16}"
    for c in [0.1, 0.25, 0.5, 1, 2, 3, 5]:
        if c not in R["dades"]:
            linia += f"{'—':>12}"
            continue
        hs = [v[0] for v in R["dades"][c] if np.isfinite(v[0])]
        if len(hs) < 2:
            linia += f"{'—':>12}"
            continue
        rsd = float(np.std(hs, ddof=1) / np.mean(hs) * 100)
        marca = "" if rsd <= 10 else " ✗"
        linia += f"{rsd:>10.1f}%{marca:<1}"
    print(linia)
print("\n  ✗ = rèpliques que no concorden (RSD > 10%) → concentració no quantificable a la pràctica")
