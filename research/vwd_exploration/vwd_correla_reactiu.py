# -*- coding: utf-8 -*-
"""
Correlació de l'oscil·lació VWD amb el cicle d'injecció de reactiu del DOC.
EXPLORACIÓ EXTERNA — no toca la Suite.

Hipòtesi (usuària): l'equip de DOC injecta OXIDANT (persulfat) i ÀCID amb bomba
cíclica; cada cicle afegeix reactiu que absorbeix UV (persulfat fort a 220, feble a
254) i canvia l'índex de refracció → la cel·la del VWD (aigües avall) veu el cicle
com una oscil·lació periòdica.

Tests (si es compleixen, l'oscil·lació ÉS el cicle de reactiu, no química de mostra):
  (1) PERÍODE únic i constant a totes les injeccions i als dos λ.
  (2) ENGANXAT AL RELLOTGE de l'equip: mateixos temps absoluts independent de la
      mostra -> en promitjar moltes injeccions l'oscil·lació NO s'esborra
      (índex de fase-lock ~1); si fos aleatòria, s'esborraria (~0).
  (3) SIGNATURA ESPECTRAL: amplitud osc 220 >> 254 (coherent amb persulfat).
  (4) 220 i 254 EN FASE (mateixa font): xcorr lag ~0.

Ús:  python vwd_correla_reactiu.py 308_SEQ_BP 307_SEQ_BP
Sortides a out_<SEQ>/reactiu/: _PERIODOGRAMA.png, _FASE_LOCK.png, _220vs254.png, _CORRELA.csv
"""
import sys, os, csv
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
from vwd_vs_actual import build_vwd_index, _resample

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = (0.5, 10.0)       # finestra d'anàlisi (exclou front d'injecció; dins rang VWD)
DT = 0.01               # min


def _prep(t, y, grid):
    """Remostreja a grid, treu deriva lineal, retorna detrended (NaN->0)."""
    yy = _resample(t, y, grid)
    m = np.isfinite(yy)
    if m.sum() < 32:
        return None
    g = grid[m]; v = yy[m]
    v = v - np.polyval(np.polyfit(g, v, 1), g)
    out = np.full_like(grid, np.nan)
    out[m] = v
    return out


def _autocorr_period(y, dt=DT, pmin=0.8, pmax=4.0):
    y = y[np.isfinite(y)]
    if len(y) < 64:
        return np.nan
    y = y - y.mean()
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    if ac[0] == 0:
        return np.nan
    ac = ac / ac[0]
    lo, hi = int(pmin / dt), int(pmax / dt)
    hi = min(hi, len(ac) - 1)
    if hi <= lo + 2:
        return np.nan
    k = lo + int(np.argmax(ac[lo:hi]))
    return k * dt


def _osc_amp(y):
    y = y[np.isfinite(y)]
    return (np.percentile(y, 97.5) - np.percentile(y, 2.5)) if len(y) else np.nan


def process_seq(seq):
    wdir = os.path.join(DADES, seq)
    if not os.path.isdir(wdir):
        print(f"  !! no existeix {wdir}"); return
    outdir = os.path.join(HERE, f"out_{seq}", "reactiu")
    os.makedirs(outdir, exist_ok=True)
    vwd_idx = build_vwd_index(os.path.join(wdir, "CSV"))
    imp = I.import_sequence(wdir); I.ensure_data_loaded(imp)
    res = A.analyze_sequence(imp, {}, do_quantify=False)
    sg = res.get("samples_grouped") or {}

    grid = np.arange(WIN[0], WIN[1], DT)
    traces220, traces254, labels = [], [], []
    rows = []
    for grp in sorted(sg.keys()):
        for rep, r in sorted((sg[grp].get("replicas") or {}).items()):
            if not isinstance(r, dict):
                continue
            key = (I.normalize_key(grp), I.normalize_rep(rep))
            vwdf = vwd_idx.get(key, {})
            ch_t = {}
            for ch in ("VWD1A", "VWD1B"):
                if ch in vwdf:
                    dfc, st = I.llegir_dad_1a(vwdf[ch], ch)
                    if not dfc.empty:
                        ch_t[ch] = _prep(dfc["time (min)"].to_numpy(float),
                                         dfc[ch].to_numpy(float), grid)
            d220 = ch_t.get("VWD1A"); d254 = ch_t.get("VWD1B")
            if d220 is None:
                continue
            traces220.append(d220); labels.append(f"{grp}_{rep}")
            if d254 is not None:
                traces254.append(d254)
            row = {"sample": grp, "rep": rep}
            row["period220_min"] = _autocorr_period(d220)
            row["amp220"] = _osc_amp(d220)
            if d254 is not None:
                row["period254_min"] = _autocorr_period(d254)
                row["amp254"] = _osc_amp(d254)
                row["ratio_amp_220_254"] = (row["amp220"] / row["amp254"]
                                            if row["amp254"] else np.nan)
                # fase 220 vs 254 (xcorr lag, min)
                a = np.nan_to_num(d220); b = np.nan_to_num(d254)
                if a.std() > 0 and b.std() > 0:
                    n = len(grid); ms = int(2.0 / DT); best = (0, -2)
                    aa = (a - a.mean()); bb = (b - b.mean())
                    for s in range(-ms, ms + 1):
                        x = aa[s:] if s >= 0 else aa[:s]
                        z = bb[:n - s] if s >= 0 else bb[-s:]
                        if len(x) < 50:
                            continue
                        dn = np.linalg.norm(x) * np.linalg.norm(z)
                        if dn == 0:
                            continue
                        c = float(np.dot(x, z) / dn)
                        if c > best[1]:
                            best = (s, c)
                    row["lag220_254_min"] = best[0] * DT
                    row["corr220_254"] = best[1]
            rows.append(row)

    if not traces220:
        print(f"  {seq}: sense traces VWD"); return

    T = np.vstack(traces220)
    # normalitza cada traça per la seva amplitud (per fase-lock just, no per magnitud)
    Tn = T / (np.nanstd(T, axis=1, keepdims=True) + 1e-9)
    mean_trace = np.nanmean(Tn, axis=0)
    # índex fase-lock = amplitud del promig / amplitud mitjana individual
    amp_mean = np.nanstd(mean_trace)
    amp_indiv = np.nanmean(np.nanstd(Tn, axis=1))
    phase_lock = amp_mean / amp_indiv if amp_indiv else np.nan

    _fig_periodogram(outdir, seq, T, rows)
    _fig_phase_lock(outdir, seq, grid, Tn, mean_trace, phase_lock, len(traces220))
    _fig_220_254(outdir, seq, grid, traces220, traces254, labels)

    # CSV
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(outdir, f"_CORRELA_{seq}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})

    # resum numèric
    p = np.array([r.get("period220_min") for r in rows
                  if isinstance(r.get("period220_min"), float)
                  and np.isfinite(r.get("period220_min"))])
    ratios = np.array([r.get("ratio_amp_220_254") for r in rows
                       if isinstance(r.get("ratio_amp_220_254"), float)
                       and np.isfinite(r.get("ratio_amp_220_254"))])
    lags = np.array([r.get("lag220_254_min") for r in rows
                     if isinstance(r.get("lag220_254_min"), float)
                     and np.isfinite(r.get("lag220_254_min"))])
    print(f"  {seq}: n={len(traces220)}  "
          f"PERÍODE 220 = {np.median(p):.2f} min (±{p.std():.2f}, cv={p.std()/p.mean()*100:.0f}%)  |  "
          f"FASE-LOCK = {phase_lock:.2f}  |  "
          f"ràtio amp 220/254 = {np.median(ratios):.0f}  |  "
          f"lag 220-254 = {np.median(lags):+.2f} min -> {outdir}")


def _fig_periodogram(outdir, seq, T, rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    freqs = np.fft.rfftfreq(T.shape[1], d=DT)
    for row_t in T:
        P = np.abs(np.fft.rfft(np.nan_to_num(row_t - np.nanmean(row_t))))
        a.plot(freqs, P, color="#c0392b", alpha=0.25, lw=0.8)
    Pm = np.nanmean([np.abs(np.fft.rfft(np.nan_to_num(rt - np.nanmean(rt)))) for rt in T], axis=0)
    a.plot(freqs, Pm, color="k", lw=1.8, label="mitjana")
    a.set_xlim(0, 2); a.set_xlabel("freqüència (cicles/min)")
    a.set_ylabel("|FFT|"); a.set_title(f"{seq}: espectre VWD 220 (totes les injeccions)")
    kk = 1 + int(np.argmax(Pm[1:np.searchsorted(freqs, 2)]))
    if freqs[kk] > 0:
        a.axvline(freqs[kk], color="#1f3b73", ls="--",
                  label=f"{freqs[kk]:.2f}/min  (T={1/freqs[kk]:.2f} min)")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    per = [r.get("period220_min") for r in rows
           if isinstance(r.get("period220_min"), float) and np.isfinite(r.get("period220_min"))]
    if per:
        b.hist(per, bins=14, color="#c0392b", alpha=0.8)
        b.axvline(np.median(per), color="k", ls="--", label=f"mediana {np.median(per):.2f} min")
        b.legend()
    b.set_title(f"{seq}: període per injecció (autocorrelació)")
    b.set_xlabel("període (min)"); b.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"_PERIODOGRAMA_{seq}.png"), dpi=120)
    plt.close(fig)


def _fig_phase_lock(outdir, seq, grid, Tn, mean_trace, phase_lock, n):
    fig, ax = plt.subplots(figsize=(11, 5))
    for rt in Tn:
        ax.plot(grid, rt, color="#c0392b", alpha=0.18, lw=0.7)
    ax.plot(grid, mean_trace, color="k", lw=2.4,
            label=f"PROMIG de {n} injeccions")
    ax.set_xlabel("temps d'equip (min)"); ax.set_ylabel("VWD 220 detrended (norm.)")
    ax.set_title(f"{seq}: fase-lock al rellotge de l'equip  —  índex = {phase_lock:.2f}  "
                 f"(prop d'1 = enganxat; prop de 0 = aleatori)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"_FASE_LOCK_{seq}.png"), dpi=120)
    plt.close(fig)


def _fig_220_254(outdir, seq, grid, t220, t254, labels):
    if not t254:
        return
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # una injecció representativa (la 1a amb dades netes)
    i = 0
    a = t220[i]; b = t254[i]
    ax.plot(grid, a / (np.nanstd(a) + 1e-9), color="#c0392b", lw=1.4, label="VWD 220 (norm.)")
    ax.plot(grid, b / (np.nanstd(b) + 1e-9), color="#e67e22", lw=1.4, label="VWD 254 (norm.)")
    ax.set_title(f"{seq}: 220 i 254 en fase?  ({labels[i]})")
    ax.set_xlabel("temps d'equip (min)"); ax.set_ylabel("detrended / σ")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"_220vs254_{seq}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["308_SEQ_BP"]
    print("Correlació oscil·lació VWD <-> cicle de reactiu del DOC")
    for s in seqs:
        process_seq(s)
