# -*- coding: utf-8 -*-
"""
VWD vs senyal actual — EXPLORACIÓ EXTERNA (NO forma part de la Suite)
=====================================================================

Mètode en desenvolupament: la sortida del detector DOC (OCD/TOC) s'ha connectat
a un detector UV de longitud d'ona variable (VWD) que mesura a DUES longituds
d'ona (canals VWD1A i VWD1B). La Suite NO llegeix aquest detector i NO s'ha de
tocar. Aquest script tracta el VWD "a banda" però s'integra amb el que la Suite
JA processa: agafa el DOC processat (t_doc, y_doc_net) i el DAD 254 tal com els
calcula el pipeline real, i hi posa el VWD al costat per analitzar-lo a fons
CONTRA el senyal actual.

Comparacions:
  - DOC (carboni, OCD/TOC)   = "l'actual" al qual el VWD va connectat aigües avall
  - DAD 254 (UV, en línia HPLC, aigües amunt)
  - VWD1A, VWD1B (UV, detector nou, aigües avall del DOC)

Mètriques per mostra:
  - temps del pic (t a senyal màxim) de cada senyal
  - retard (lag) VWD↔DOC i VWD↔DAD per correlació creuada (volum mort / registre
    de bases de temps)
  - correlació de forma un cop alineat (Pearson)
  - raó espectral VWD1A/VWD1B al llarg del pic (informació de pendent espectral)
  - cobertura temporal: finestra del VWD vs finestra del DOC

Sortides a  research/vwd_exploration/out_<SEQ>/ :
  - <mostra>_overlay.png  (superposició + raó espectral)
  - _SUMMARY_<SEQ>.csv     (mètriques per mostra)
  - _PANORAMA_<SEQ>.png    (distribució de lags i correlacions)

Ús:
  python vwd_vs_actual.py 308_SEQ_BP
  python vwd_vs_actual.py 307_SEQ_BP 308_SEQ_BP
"""
import sys, os, glob, csv
import numpy as np

HPSEC = r"C:\Users\maria\Proyectos\HPSEC"
DADES = r"C:\Users\maria\Proyectos\Dades3"
sys.path.insert(0, HPSEC)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hpsec_import as I
import hpsec_analyze as A

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- utilitats
def _norm(y):
    """Normalitza a [0,1] sobre el rang (robust a baseline/offset)."""
    y = np.asarray(y, float)
    lo, hi = np.nanmin(y), np.nanmax(y)
    if not np.isfinite(hi - lo) or (hi - lo) <= 0:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def _resample(t, y, grid):
    """Interpola (t,y) sobre grid; fora de rang -> NaN."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 5:
        return np.full_like(grid, np.nan, float)
    out = np.interp(grid, t[m], y[m], left=np.nan, right=np.nan)
    return out


def _xcorr_lag(t_ref, y_ref, t_mov, y_mov, dt=0.01, max_lag_min=3.0, w=None):
    """
    Retard de y_mov respecte y_ref per correlació creuada, en minuts.
    lag>0  => y_mov surt MÉS TARD que y_ref (desplaçar y_mov cap enrere per alinear).
    w = (t0,t1) restringeix l'anàlisi a la finestra d'elució (exclou front d'injecció).
    Retorna (lag_min, corr_max_alineat).
    """
    lo = w[0] if w else -1e9
    hi = w[1] if w else 1e9
    t0 = max(np.nanmin(t_ref), np.nanmin(t_mov), lo)
    t1 = min(np.nanmax(t_ref), np.nanmax(t_mov), hi)
    if not np.isfinite(t1 - t0) or (t1 - t0) < 1.0:
        return np.nan, np.nan
    grid = np.arange(t0, t1, dt)
    a = _norm(_resample(t_ref, y_ref, grid))
    b = _norm(_resample(t_mov, y_mov, grid))
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.where(mask, a, 0.0); b = np.where(mask, b, 0.0)
    a = a - a.mean(); b = b - b.mean()
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan
    n = len(grid)
    max_shift = int(round(max_lag_min / dt))
    best_lag, best_corr = 0, -2.0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            x = a[s:]; z = b[:n - s]
        else:
            x = a[:n + s]; z = b[-s:]
        if len(x) < 20:
            continue
        denom = (np.linalg.norm(x) * np.linalg.norm(z))
        if denom == 0:
            continue
        c = float(np.dot(x, z) / denom)
        if c > best_corr:
            best_corr, best_lag = c, s
    # s>0 a la convenció de dalt desplaça a (ref) endavant -> mov surt més tard: lag = +s*dt
    return best_lag * dt, best_corr


def _win_mask(t, w):
    t = np.asarray(t, float)
    if w is None:
        return np.isfinite(t)
    return np.isfinite(t) & (t >= w[0]) & (t <= w[1])


def _peak_rt(t, y, w=None):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = _win_mask(t, w) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    return float(t[m][int(np.nanargmax(y[m]))])


def _osc_metrics(t, y, w):
    """Caracteritza l'oscil·lació de la línia de base dins la finestra w (post-pic):
    període dominant (min) i amplitud pic-a-pic relativa al rang global."""
    m = _win_mask(t, w) & np.isfinite(y)
    if m.sum() < 32:
        return np.nan, np.nan
    tt = np.asarray(t, float)[m]; yy = np.asarray(y, float)[m]
    dt = np.median(np.diff(tt))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan
    yd = yy - np.polyval(np.polyfit(tt, yy, 1), tt)  # treu deriva lineal
    # període dominant via FFT
    f = np.fft.rfftfreq(len(yd), d=dt)
    P = np.abs(np.fft.rfft(yd - yd.mean()))
    if len(f) > 2:
        k = 1 + int(np.argmax(P[1:]))
        period = (1.0 / f[k]) if f[k] > 0 else np.nan
    else:
        period = np.nan
    yrange = np.nanmax(y) - np.nanmin(y)
    amp_rel = (np.nanmax(yd) - np.nanmin(yd)) / yrange if yrange > 0 else np.nan
    return period, amp_rel


# ---------------------------------------------------- indexació fitxers VWD
def build_vwd_index(csv_dir):
    """{(normkey, rep): {'VWD1A': path, 'VWD1B': path}}"""
    idx = {}
    for ch in ("VWD1A", "VWD1B"):
        for p in glob.glob(os.path.join(csv_dir, f"*{ch}*.CSV")) + \
                 glob.glob(os.path.join(csv_dir, f"*{ch}*.csv")):
            base = os.path.basename(p)
            stem = os.path.splitext(base)[0]
            import re
            stem2 = re.sub(rf"_?{ch}\d*", "", stem, flags=re.IGNORECASE)
            # reparteix mostra/rep com el UIB
            mostra, rep = I.netejar_nom_uib(stem2 + ".csv")
            key = (I.normalize_key(mostra), I.normalize_rep(rep))
            idx.setdefault(key, {})[ch] = p
    return idx


# --------------------------------------------------------------- processa 1 seq
def process_seq(seq):
    wdir = os.path.join(DADES, seq)
    if not os.path.isdir(wdir):
        print(f"  !! no existeix {wdir}"); return
    outdir = os.path.join(HERE, f"out_{seq}")
    os.makedirs(outdir, exist_ok=True)
    csv_dir = os.path.join(wdir, "CSV")
    vwd_idx = build_vwd_index(csv_dir)

    imp = I.import_sequence(wdir); I.ensure_data_loaded(imp)
    res = A.analyze_sequence(imp, {}, do_quantify=False)
    sg = res.get("samples_grouped") or {}

    rows = []
    for grp in sorted(sg.keys()):
        for rep, r in sorted((sg[grp].get("replicas") or {}).items()):
            if not isinstance(r, dict):
                continue
            key = (I.normalize_key(grp), I.normalize_rep(rep))
            vwd = vwd_idx.get(key, {})
            if "VWD1A" not in vwd and "VWD1B" not in vwd:
                continue  # sense VWD per aquesta rèplica

            # --- senyals que JA processa la Suite ---
            t_doc = np.asarray(r.get("t_doc"), float)
            y_doc = np.asarray(r.get("y_doc_net"), float)
            pk = r.get("peak_info") or {}
            t_start, t_end, t_max_doc = pk.get("t_start"), pk.get("t_end"), pk.get("t_max")

            df_dad = r.get("df_dad")
            t_dad = y_dad = None
            if df_dad is not None and "254" in getattr(df_dad, "columns", []):
                t_dad = df_dad["time (min)"].to_numpy(float)
                y_dad = df_dad["254"].to_numpy(float)

            # --- VWD cru (detector nou) ---
            sig = {}
            for ch in ("VWD1A", "VWD1B"):
                if ch in vwd:
                    df, st = I.llegir_dad_1a(vwd[ch], ch)
                    if not df.empty:
                        sig[ch] = (df["time (min)"].to_numpy(float),
                                   df[ch].to_numpy(float))

            # finestra d'elució (exclou front d'injecció) i finestra post-pic (base)
            win = None; win_post = None
            if t_start is not None and t_end is not None:
                win = (max(t_start - 1.0, 0.5), t_end + 1.0)
                win_post = (t_end + 0.5, t_end + 5.0)

            # --- mètriques ---
            row = {"sample": grp, "rep": rep}
            row["rt_doc"] = _peak_rt(t_doc, y_doc)
            row["rt_dad254"] = _peak_rt(t_dad, y_dad) if y_dad is not None else np.nan
            row["lag_DAD_vs_DOC_min"] = (_xcorr_lag(t_doc, y_doc, t_dad, y_dad, w=win)[0]
                                         if y_dad is not None else np.nan)
            for ch in ("VWD1A", "VWD1B"):
                if ch in sig:
                    tt, yy = sig[ch]
                    # pic DINS la finestra d'elució (no el front)
                    row[f"rt_{ch}_win"] = _peak_rt(tt, yy, win)
                    row[f"rt_{ch}_global"] = _peak_rt(tt, yy)
                    lag_doc, corr_doc = _xcorr_lag(t_doc, y_doc, tt, yy, w=win)
                    row[f"lag_{ch}_vs_DOC_min"] = lag_doc
                    row[f"corr_{ch}_DOC_win"] = corr_doc
                    if y_dad is not None:
                        lag_dad, corr_dad = _xcorr_lag(t_dad, y_dad, tt, yy, w=win)
                        row[f"lag_{ch}_vs_DAD_min"] = lag_dad
                        row[f"corr_{ch}_DAD_win"] = corr_dad
                    # oscil·lació de base post-pic (diagnòstic instrumental)
                    per, amp = _osc_metrics(tt, yy, win_post)
                    row[f"osc_period_{ch}_min"] = per
                    row[f"osc_amp_rel_{ch}"] = amp
                    # SNR analit: amplitud del senyal a la finestra / amplitud oscil·lació base
                    mw = _win_mask(tt, win) & np.isfinite(yy)
                    mp = _win_mask(tt, win_post) & np.isfinite(yy)
                    if mw.sum() > 5 and mp.sum() > 5:
                        sig_amp = np.nanmax(yy[mw]) - np.nanmin(yy[mw])
                        base_amp = np.nanmax(yy[mp]) - np.nanmin(yy[mp])
                        row[f"snr_win_over_base_{ch}"] = (sig_amp / base_amp
                                                          if base_amp > 0 else np.nan)
                    row[f"tmin_{ch}"] = float(np.nanmin(tt))
                    row[f"tmax_win_{ch}"] = float(np.nanmax(tt))
            row["doc_tmax_min"] = float(np.nanmax(t_doc)) if t_doc.size else np.nan
            if "VWD1A" in sig and t_end is not None:
                row["vwd_cobreix_pic"] = bool(np.nanmax(sig["VWD1A"][0]) >= t_end)
            rows.append(row)

            # --- figura per mostra ---
            _plot_sample(outdir, seq, grp, rep, t_doc, y_doc, t_dad, y_dad,
                         sig, t_start, t_end, t_max_doc)

    # --- CSV resum ---
    if rows:
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        csvp = os.path.join(outdir, f"_SUMMARY_{seq}.csv")
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in keys})
        _plot_panorama(outdir, seq, rows)
        print(f"  {seq}: {len(rows)} mostres amb VWD -> {outdir}")
    else:
        print(f"  {seq}: cap mostra amb VWD aparellada")
    return rows


# ------------------------------------------------------------------- figures
def _plot_sample(outdir, seq, grp, rep, t_doc, y_doc, t_dad, y_dad,
                 sig, t_start, t_end, t_max_doc):
    fig, (ax1, axr, ax2) = plt.subplots(3, 1, figsize=(11, 10),
                                        height_ratios=[3, 2, 1.4])

    # superposició normalitzada
    if t_doc.size:
        ax1.plot(t_doc, _norm(y_doc), color="#1f3b73", lw=2.2, label="DOC (actual, net)")
    if y_dad is not None:
        ax1.plot(t_dad, _norm(y_dad), color="#8a8a8a", lw=1.2, alpha=0.8,
                 label="DAD 254 (UV en línia)")
    colors = {"VWD1A": "#c0392b", "VWD1B": "#e67e22"}
    for ch in ("VWD1A", "VWD1B"):
        if ch in sig:
            tt, yy = sig[ch]
            ax1.plot(tt, _norm(yy), color=colors[ch], lw=1.6, label=f"{ch} (UV nou)")
    if t_start is not None and t_end is not None:
        ax1.axvspan(t_start, t_end, color="#1f3b73", alpha=0.07, label="finestra pic DOC")
    if t_max_doc is not None:
        ax1.axvline(t_max_doc, color="#1f3b73", ls=":", lw=1, alpha=0.6)
    ax1.set_xlabel("temps (min)"); ax1.set_ylabel("senyal normalitzat [0-1]")
    ax1.set_title(f"{seq}  ·  {grp} rep{rep}   —   VWD (nou) vs senyal actual")
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.grid(alpha=0.25)

    # magnitud REAL del VWD (unitats natives) + finestra pic DOC
    for ch in ("VWD1A", "VWD1B"):
        if ch in sig:
            tt, yy = sig[ch]
            axr.plot(tt, yy, color=colors[ch], lw=1.3, label=ch)
    if t_start is not None and t_end is not None:
        axr.axvspan(t_start, t_end, color="#1f3b73", alpha=0.10,
                    label="finestra pic DOC")
    axr.set_ylabel("VWD (mAU, cru)"); axr.set_xlabel("temps (min)")
    axr.legend(fontsize=8, loc="upper right"); axr.grid(alpha=0.25)
    axr.set_title("Magnitud real del VWD — el pic d'analit domina o ho fa la base?",
                  fontsize=9)

    # raó espectral VWD1A/VWD1B
    if "VWD1A" in sig and "VWD1B" in sig:
        ta, ya = sig["VWD1A"]; tb, yb = sig["VWD1B"]
        grid = np.arange(max(ta.min(), tb.min()), min(ta.max(), tb.max()), 0.01)
        ya_i = _resample(ta, ya, grid); yb_i = _resample(tb, yb, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(yb_i) > 0.05 * np.nanmax(np.abs(yb_i)),
                             ya_i / yb_i, np.nan)
        ax2.plot(grid, ratio, color="#7d3c98", lw=1.4)
        ax2.set_ylabel("VWD1A / VWD1B")
        if t_start is not None and t_end is not None:
            ax2.axvspan(t_start, t_end, color="#1f3b73", alpha=0.07)
            ax2.set_xlim(max(grid.min(), (t_start or grid.min()) - 2),
                         min(grid.max(), (t_end or grid.max()) + 2))
        ax2.grid(alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "només un canal VWD disponible",
                 ha="center", va="center", transform=ax2.transAxes, color="#888")
    ax2.set_xlabel("temps (min)")

    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in f"{grp}_rep{rep}")
    fig.savefig(os.path.join(outdir, f"{safe}_overlay.png"), dpi=120)
    plt.close(fig)


def _plot_panorama(outdir, seq, rows):
    lags = [r.get("lag_VWD1A_vs_DOC_min") for r in rows
            if isinstance(r.get("lag_VWD1A_vs_DOC_min"), float)
            and np.isfinite(r.get("lag_VWD1A_vs_DOC_min"))]
    corrs = [r.get("corr_VWD1A_DOC_win") for r in rows
             if isinstance(r.get("corr_VWD1A_DOC_win"), float)
             and np.isfinite(r.get("corr_VWD1A_DOC_win"))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    if lags:
        a.hist(lags, bins=12, color="#c0392b", alpha=0.8)
        a.axvline(np.median(lags), color="k", ls="--",
                  label=f"mediana {np.median(lags):.2f} min")
        a.legend()
    a.set_title(f"{seq}: retard VWD1A vs DOC (min)")
    a.set_xlabel("lag (min)  [+ = VWD surt més tard]"); a.grid(alpha=0.25)
    if corrs:
        b.hist(corrs, bins=12, color="#1f3b73", alpha=0.8)
        b.axvline(np.median(corrs), color="k", ls="--",
                  label=f"mediana {np.median(corrs):.3f}")
        b.legend()
    b.set_title(f"{seq}: correlació de forma VWD1A↔DOC (alineat)")
    b.set_xlabel("Pearson"); b.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"_PANORAMA_{seq}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["308_SEQ_BP"]
    print("VWD vs actual — exploració externa")
    for s in seqs:
        process_seq(s)
