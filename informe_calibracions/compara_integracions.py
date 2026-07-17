# -*- coding: utf-8 -*-
"""Comparacio fina de la integracio del DOC entre les seqüencies 305 i 306.

Les dues comparteixen detector, mode, volum i metode i nomes difereixen en la
columna cromatografica. Aquest script contrasta, injeccio a injeccio:

  - on situa el pic l'ancora de 254 nm i on el situa la integracio del DOC,
  - quina finestra obre la Suite i quina amplada te respecte del FWHM del pic,
  - quina area en surt i quina en sortiria amb una finestra estreta ancorada,
  - com queda la recta de calibratge amb cada metode.

Objectiu: separar el que es efecte de la COLUMNA del que es efecte de la
INTEGRACIO, i comprovar si totes dues seqüencies convergeixen al mateix factor
de resposta quan s'integren igual.

Us:
    python -X utf8 compara_integracions.py            # taules a consola
    python -X utf8 compara_integracions.py --figura   # + PNG comparatiu
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.integrate import trapezoid

SCR = os.path.dirname(os.path.abspath(__file__))
DADES = r"C:\Users\maria\Proyectos\Dades3"
SEQS = ["293_SEQ_CAL", "305_SEQ_CAL", "306_SEQ_CAL"]
VOL_UL = 400.0
N_FWHM = 2.0        # semi-amplada de la finestra estreta, en multiples del FWHM
HW_CERCA = 3.0      # semi-finestra de cerca del maxim al voltant de l'ancora


def cal_json(seq):
    p = os.path.join(DADES, seq, "CHECK", "data", "calibration_result.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pic_254(t, y):
    """Maxim del 254 corregit de deriva (a la 306 la base cau fins a -7 mAU)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if len(t) < 50:
        return None
    n = len(t)
    edge = np.r_[np.arange(0, n // 6), np.arange(5 * n // 6, n)]
    y_det = y - np.polyval(np.polyfit(t[edge], y[edge], 1), t)
    return float(t[int(np.argmax(y_det))])


def base_local(t, y, t_lo, t_hi, pad=1.5):
    """Base lineal ajustada als dos flancs de la finestra del pic."""
    m = ((t >= t_lo - pad) & (t < t_lo)) | ((t > t_hi) & (t <= t_hi + pad))
    if m.sum() < 4:
        return np.full_like(t, float(np.median(y)))
    return np.polyval(np.polyfit(t[m], y[m], 1), t)


def integra_estreta(t, y, t_ancora):
    """Maxim prop de l'ancora, FWHM per creuament, i integracio a +-N_FWHM*FWHM
    sobre base local. Retorna dict amb area, t_max, fwhm, limits i alcada neta."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = (t >= t_ancora - HW_CERCA) & (t <= t_ancora + HW_CERCA)
    if m.sum() < 5:
        return None
    idx = np.where(m)[0]
    i = idx[int(np.argmax(y[m]))]
    bl0 = float(np.median(np.r_[y[m][:3], y[m][-3:]]))
    half = (y[i] + bl0) / 2.0
    li = ri = i
    while li > idx[0] and y[li] > half:
        li -= 1
    while ri < idx[-1] and y[ri] > half:
        ri += 1
    fwhm = float(t[ri] - t[li])
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = 1.5
    t_lo, t_hi = float(t[i]) - N_FWHM * fwhm, float(t[i]) + N_FWHM * fwhm
    w = (t >= t_lo) & (t <= t_hi)
    if w.sum() < 5:
        return None
    bl = base_local(t, y, t_lo, t_hi)
    return {
        "area": float(trapezoid(np.maximum(y[w] - bl[w], 0), t[w])),
        "t_max": float(t[i]), "fwhm": fwhm, "t_lo": t_lo, "t_hi": t_hi,
        "h_net": float(y[i] - bl[i]),
        "bl_pend": float((bl[-1] - bl[0]) / (t[-1] - t[0])),  # deriva de base, ppb/min
    }


def recull(seq):
    """Una entrada per injeccio amb tot el que cal per comparar."""
    d = cal_json(seq)
    files = []
    t254_tots = []
    for cal in d.get("calibrations") or []:
        for rep in cal.get("replicas_info") or []:
            t254_tots.append(pic_254(rep.get("t_dad") or [], rep.get("y_dad_254") or []))
    # Ancora de consens: mediana dels 254 fiables (a la 306 algun es va a 37 min)
    bons = [x for x in t254_tots if x is not None and 15 <= x <= 28]
    t254_ref = float(np.median(bons)) if bons else None

    # Desfas DOC-254 empiric de la seq (mediana), per ancorar la cerca del maxim
    desfas = []
    for cal in d.get("calibrations") or []:
        for rep in cal.get("replicas_info") or []:
            tm = rep.get("t_max")
            if tm is not None and t254_ref is not None and abs(tm - t254_ref) < 6:
                desfas.append(tm - t254_ref)
    shift = float(np.median(desfas)) if desfas else 2.0

    for cal in d.get("calibrations") or []:
        conc = cal.get("conc_ppm")
        for rep in cal.get("replicas_info") or []:
            t = np.asarray(rep.get("t_doc") or [], float)
            y = np.asarray(rep.get("y_doc") or [], float)
            if len(t) < 20:
                continue
            t254 = pic_254(rep.get("t_dad") or [], rep.get("y_dad_254") or [])
            fiable254 = t254 is not None and 15 <= t254 <= 28
            ancora = (t254 if fiable254 else t254_ref) + shift
            est = integra_estreta(t, y, ancora)
            ts, te = rep.get("t_start"), rep.get("t_end")
            files.append({
                "seq": seq, "conc": conc, "rep": rep.get("replica_num"),
                "t254": t254, "fiable254": fiable254,
                "suite_area": rep.get("area") or 0.0,
                "suite_tmax": rep.get("t_max"),
                "suite_lo": ts, "suite_hi": te,
                "suite_ampl": (te - ts) if (ts is not None and te is not None) else np.nan,
                "suite_fwhm": rep.get("fwhm_doc"),
                "est": est, "t": t, "y": y,
            })
    return files, t254_ref, shift


def recta(parells):
    """parells = [(ug, area)] -> (RF_mass, intercept, R2)"""
    ug = np.array([p[0] for p in parells], float)
    ar = np.array([p[1] for p in parells], float)
    ok = np.isfinite(ug) & np.isfinite(ar)
    if ok.sum() < 3:
        return None
    sl, ic = np.polyfit(ug[ok], ar[ok], 1)
    pred = sl * ug[ok] + ic
    ss_res = float(np.sum((ar[ok] - pred) ** 2))
    ss_tot = float(np.sum((ar[ok] - np.mean(ar[ok])) ** 2))
    return sl, ic, (1 - ss_res / ss_tot if ss_tot else np.nan)


def main():
    dades = {}
    for seq in SEQS:
        dades[seq] = recull(seq)

    # ---------------- Detall injeccio a injeccio ----------------
    for seq in SEQS:
        files, t254_ref, shift = dades[seq]
        print("\n" + "=" * 122)
        print(f"{seq} — ancora 254 de consens = {t254_ref:.2f} min · desfas DOC−254 = {shift:+.2f} min")
        print("=" * 122)
        hdr = (f"{'ppm':>5} {'rep':>4} | {'t254':>6} {'t_DOC':>6} {'FWHM':>5} | "
               f"{'finestra Suite':>17} {'ampl':>6} {'x FWHM':>7} | "
               f"{'area Suite':>11} {'area estreta':>12} {'sobre-int':>10} | {'h_net':>9}")
        print(hdr); print("-" * len(hdr))
        for f in sorted(files, key=lambda x: (x["conc"], x["rep"])):
            e = f["est"]
            if e is None:
                print(f"{f['conc']:>5g} {f['rep']:>4} | (sense integracio estreta)")
                continue
            xf = f["suite_ampl"] / e["fwhm"] if e["fwhm"] else np.nan
            ratio = f["suite_area"] / e["area"] if e["area"] > 0 else np.nan
            marca = "" if f["fiable254"] else " *"
            print(f"{f['conc']:>5g} {f['rep']:>4} | {(f['t254'] or np.nan):>5.2f}{marca:<1} "
                  f"{e['t_max']:>6.2f} {e['fwhm']:>5.2f} | "
                  f"{f['suite_lo']:>7.2f}–{f['suite_hi']:<7.2f} {f['suite_ampl']:>6.2f} "
                  f"{xf:>7.1f} | {f['suite_area']:>11.1f} {e['area']:>12.1f} "
                  f"{ratio:>9.2f}x | {e['h_net']:>9.1f}")
        print("  * = pic de 254 no fiable en aquesta injeccio (deriva); s'usa l'ancora de consens")

    # ---------------- Amplada de finestra: resum ----------------
    print("\n" + "=" * 122)
    print("AMPLADA DE LA FINESTRA DE LA SUITE, EN MULTIPLES DEL FWHM DEL PIC")
    print("(una integracio correcta d'un pic gaussia son ~4 FWHM; per sobre, s'hi cola senyal que no es el pic)")
    print("=" * 122)
    print(f"{'seq':<16} {'min':>6} {'mediana':>9} {'max':>7} | {'injeccions > 6 FWHM':>20}")
    print("-" * 62)
    for seq in SEQS:
        files, _, _ = dades[seq]
        xs = [f["suite_ampl"] / f["est"]["fwhm"] for f in files
              if f["est"] and f["est"]["fwhm"] and np.isfinite(f["suite_ampl"])]
        n_mal = sum(1 for x in xs if x > 6)
        print(f"{seq:<16} {min(xs):>6.1f} {np.median(xs):>9.1f} {max(xs):>7.1f} | "
              f"{n_mal:>10} de {len(xs)}")

    # ---------------- Rectes ----------------
    print("\n" + "=" * 122)
    print("RECTA DE CALIBRATGE amb cada metode d'integracio")
    print("=" * 122)
    print(f"{'seq':<16} {'metode':<22} {'RF_mass':>9} {'intercept':>11} {'R2':>8}")
    print("-" * 70)
    resum = {}
    for seq in SEQS:
        files, _, _ = dades[seq]
        per_conc_s, per_conc_e = {}, {}
        for f in files:
            per_conc_s.setdefault(f["conc"], []).append(f["suite_area"])
            if f["est"]:
                per_conc_e.setdefault(f["conc"], []).append(f["est"]["area"])
        ps = [(c * VOL_UL / 1000, float(np.mean(v))) for c, v in sorted(per_conc_s.items())]
        pe = [(c * VOL_UL / 1000, float(np.mean(v))) for c, v in sorted(per_conc_e.items())]
        for nom, p in (("Suite (vigent)", ps), ("estreta ancorada", pe)):
            r = recta(p)
            if r:
                print(f"{seq:<16} {nom:<22} {r[0]:>9.0f} {r[1]:>11.1f} {r[2]:>8.4f}")
                resum[(seq, nom)] = r
        print()

    # ---------------- La pregunta clau ----------------
    print("=" * 122)
    print("CONVERGEIXEN 305 I 306 QUAN S'INTEGREN IGUAL?")
    print("=" * 122)
    for nom in ("Suite (vigent)", "estreta ancorada"):
        a = resum.get(("305_SEQ_CAL", nom))
        b = resum.get(("306_SEQ_CAL", nom))
        c = resum.get(("293_SEQ_CAL", nom))
        if a and b:
            print(f"{nom:<22} RF 305={a[0]:>8.0f}  RF 306={b[0]:>8.0f}  "
                  f"-> 306/305 = {b[0]/a[0]:.3f}")
        if a and c:
            print(f"{'':<22} guany vs 293: 305 x{a[0]/c[0]:.1f}  306 x{b[0]/c[0]:.1f}")

    if "--figura" in sys.argv:
        fes_figura(dades)


def fes_figura(dades):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concs = [0.25, 0.5, 1.0, 3.0, 5.0]
    seqs = ["305_SEQ_CAL", "306_SEQ_CAL"]
    fig, axes = plt.subplots(len(concs), 2, figsize=(15, 2.6 * len(concs)))
    for i, conc in enumerate(concs):
        for j, seq in enumerate(seqs):
            ax = axes[i][j]
            files, _, _ = dades[seq]
            f = next((x for x in files if x["conc"] == conc and x["rep"] == 1), None)
            if f is None or f["est"] is None:
                ax.set_visible(False)
                continue
            e = f["est"]
            t, y = f["t"], f["y"]
            ax.plot(t, y, color="#2563EB", lw=0.8)
            ax.axvspan(f["suite_lo"], f["suite_hi"], color="#E74C3C", alpha=0.12,
                       label=f"Suite {f['suite_ampl']:.0f} min → A={f['suite_area']:.0f}")
            ax.axvspan(e["t_lo"], e["t_hi"], color="#27AE60", alpha=0.30,
                       label=f"estreta {e['t_hi']-e['t_lo']:.1f} min → A={e['area']:.0f}")
            if f["t254"]:
                ax.axvline(f["t254"], color="#8E44AD", ls=":", lw=1.1)
            ax.set_title(f"{seq[:3]} — {conc:g} ppm R1 (FWHM {e['fwhm']:.2f} min)",
                         fontsize=9.5, loc="left")
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
            ax.grid(alpha=0.22)
            ax.set_ylabel("DOC (ppb)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (min)", fontsize=9)
    fig.suptitle("Integració del DOC: 305 (columna anterior) vs 306 (columna substituïda) — "
                 "finestra de la Suite vs finestra estreta ancorada al 254 nm", fontsize=12, y=0.999)
    fig.tight_layout()
    out = os.path.join(SCR, "integracio_305_vs_306.png")
    fig.savefig(out, dpi=110)
    print("\nFigura desada:", out)


if __name__ == "__main__":
    main()
