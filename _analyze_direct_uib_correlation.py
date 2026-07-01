"""
Analisi correlacio DOC Direct <-> UIB per la 288_SEQ.

Objectiu: determinar si es pot ESTIMAR el cromatograma DOC Direct
punt a punt a partir del cromatograma UIB, per fer mostres sense
Direct comparables amb l'historic.

Genera:
  1. Factor de conversio local f(t) = y_direct(t) / y_uib(t) per zona temporal
  2. Scatter punt-a-punt (y_direct vs y_uib) amb regressio
  3. Overlay cromatogrames: Direct real vs Direct estimat des d'UIB
  4. Residuals per zona (BioP, HS, BB, SB, LMW)
  5. Model + estadistiques

Output: _results/direct_uib_correlation/ (PNGs + CSV + JSON)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

# -- Config --
SEQ_PATH = Path(r"C:\Users\Lequia\Desktop\HPSEC\Dades3\288_SEQ")
ANALYSIS_JSON = SEQ_PATH / "CHECK" / "data" / "analysis_result.json"
OUT_DIR = Path(__file__).parent / "_results" / "direct_uib_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fraccions COLUMN amb limits temporals (min) - de hpsec_config.json
FRACTIONS = {
    "BioP": (11.5, 16.5),
    "HS":   (16.5, 26.5),
    "BB":   (26.5, 33.0),
    "SB":   (33.0, 41.0),
    "LMW":  (41.0, 55.0),
}

# -- Load --
print("Carregant analysis_result.json...")
with open(ANALYSIS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

samples_grouped = data.get("samples_grouped", {})
method = data.get("method", "COLUMN")
print(f"Metode: {method}, {len(samples_grouped)} mostres agrupades")

# -- Extreure parells de cromatogrames --
pairs = []  # (sample, rep, t, y_direct, y_uib)
for sample_name, sample_data in samples_grouped.items():
    # Saltar blancs/controls per model
    if any(tag in sample_name.upper() for tag in ["MQ", "NAOH", "BUFFER"]):
        continue
    replicas = sample_data.get("replicas", {})
    for rep_key, rep in replicas.items():
        t = rep.get("t_doc")
        y_direct = rep.get("y_doc_direct_net")
        y_uib = rep.get("y_doc_uib_net")

        if t is None or y_direct is None or y_uib is None:
            continue
        t = np.array(t)
        y_direct = np.array(y_direct)
        y_uib = np.array(y_uib)

        if len(t) < 100 or len(y_direct) != len(t) or len(y_uib) != len(t):
            continue

        # Filtre: si areas identiques (bug antic is_uib_only), saltar
        if np.allclose(y_direct, y_uib, atol=0.01):
            continue

        pairs.append({
            "sample": sample_name,
            "rep": rep_key,
            "t": t,
            "y_direct": y_direct,
            "y_uib": y_uib,
        })

print(f"\n{len(pairs)} parells de cromatogrames (Direct+UIB)")
if len(pairs) < 3:
    print("AVORTAT: massa poques dades")
    raise SystemExit(1)

# -- 0. Interpolar al time base comu --
# Agafar el time base mes dens com a referencia
lens = [len(p["t"]) for p in pairs]
ref_idx = np.argmax(lens)
t_ref = pairs[ref_idx]["t"]
n_pts = len(t_ref)
print(f"Time base referencia: {n_pts} punts ({t_ref[0]:.2f} - {t_ref[-1]:.2f} min)")

for p in pairs:
    if len(p["t"]) != n_pts or not np.allclose(p["t"], t_ref, atol=0.01):
        p["y_direct"] = np.interp(t_ref, p["t"], p["y_direct"])
        p["y_uib"] = np.interp(t_ref, p["t"], p["y_uib"])
        p["t"] = t_ref

# Matriu: cada fila es un parell, cada columna un punt temporal
mat_direct = np.array([p["y_direct"] for p in pairs])
mat_uib = np.array([p["y_uib"] for p in pairs])

# Factor punt a punt (evitant div/0: nomes on UIB > threshold)
NOISE_FLOOR = 5.0  # ppb minim per considerar senyal real
factor_matrix = np.full_like(mat_direct, np.nan)
for i in range(len(pairs)):
    mask_signal = mat_uib[i] > NOISE_FLOOR
    factor_matrix[i, mask_signal] = mat_direct[i, mask_signal] / mat_uib[i, mask_signal]

# Estadistiques del factor per punt temporal
factor_median = np.nanmedian(factor_matrix, axis=0)
factor_mean = np.nanmean(factor_matrix, axis=0)
factor_std = np.nanstd(factor_matrix, axis=0)
factor_count = np.sum(~np.isnan(factor_matrix), axis=0)

# On no hi ha prou dades, usar factor global
MIN_COUNT = 5
sparse = factor_count < MIN_COUNT
global_factor = np.nanmedian(factor_matrix[~np.isnan(factor_matrix)])
factor_median[sparse] = global_factor

print(f"\nFactor global mediana: {global_factor:.4f}")

# Suavitzar el factor (Savitzky-Golay)
from scipy.signal import savgol_filter
factor_smooth = factor_median.copy()
valid = ~np.isnan(factor_smooth)
if np.sum(valid) > 31:
    factor_smooth[valid] = savgol_filter(factor_smooth[valid], 31, 2)

# -- 2. Regressio punt-a-punt global --
# Aplanar tots els punts on ambdos senyals tenen senyal
all_d = []
all_u = []
for p in pairs:
    mask = (p["y_uib"] > NOISE_FLOOR) & (p["y_direct"] > NOISE_FLOOR)
    all_d.extend(p["y_direct"][mask])
    all_u.extend(p["y_uib"][mask])
all_d = np.array(all_d)
all_u = np.array(all_u)

slope_pt, intercept_pt, r_pt, p_pt, se_pt = sp_stats.linregress(all_u, all_d)
r2_pt = r_pt**2

# Model per origen
slope_origin = np.sum(all_u * all_d) / np.sum(all_u**2)
pred_o = slope_origin * all_u
r2_origin = 1 - np.sum((all_d - pred_o)**2) / np.sum((all_d - np.mean(all_d))**2)

print(f"\nRegressio punt-a-punt ({len(all_d)} punts):")
print(f"  Direct = {slope_pt:.4f} x UIB + {intercept_pt:.2f}  (R2={r2_pt:.6f})")
print(f"  Origen: Direct = {slope_origin:.4f} x UIB  (R2={r2_origin:.6f})")

# -- 3. Per-fraccio: factor mediana per zona --
print(f"\nFactor per fraccio:")
frac_factors = {}
for frac, (t_ini, t_fi) in FRACTIONS.items():
    mask_t = (t_ref >= t_ini) & (t_ref < t_fi)
    if not np.any(mask_t):
        continue
    frac_vals = factor_matrix[:, mask_t]
    frac_med = np.nanmedian(frac_vals)
    frac_std = np.nanstd(frac_vals)
    frac_cv = frac_std / frac_med * 100 if frac_med > 0 else 0
    frac_factors[frac] = {"median": float(frac_med), "std": float(frac_std),
                           "cv_pct": float(frac_cv)}
    print(f"  {frac:5s} [{t_ini:.1f}-{t_fi:.1f} min]:  factor={frac_med:.4f}  "
          f"std={frac_std:.4f}  CV={frac_cv:.1f}%")

# -- 4. Validacio: estimar Direct i comparar amb real --
print(f"\n{'='*60}")
print("VALIDACIO: Direct real vs estimat (3 models)")
print(f"{'='*60}")

for model_name, estimate_fn in [
    ("Factor global",  lambda y_u: y_u * global_factor),
    ("Regressio",      lambda y_u: slope_pt * y_u + intercept_pt),
    ("Factor local",   lambda y_u: y_u * factor_smooth),
]:
    errors_area = []  # % error en area total
    errors_shape = []  # Pearson cromatograma
    for p in pairs:
        y_est = estimate_fn(p["y_uib"])
        # Area total (trapezoid)
        area_real = np.trapezoid(np.maximum(p["y_direct"], 0), p["t"])
        area_est = np.trapezoid(np.maximum(y_est, 0), p["t"])
        if area_real > 0:
            errors_area.append((area_est - area_real) / area_real * 100)
        # Forma (Pearson)
        mask = (p["y_direct"] > NOISE_FLOOR) | (y_est > NOISE_FLOOR)
        if np.sum(mask) > 10:
            r, _ = sp_stats.pearsonr(p["y_direct"][mask], y_est[mask])
            errors_shape.append(r)

    ea = np.array(errors_area)
    es = np.array(errors_shape)
    print(f"\n  {model_name}:")
    print(f"    Area: bias={np.mean(ea):.2f}%  std={np.std(ea):.2f}%  "
          f"max_abs={np.max(np.abs(ea)):.2f}%")
    print(f"    Forma: Pearson mediana={np.median(es):.4f}  "
          f"min={np.min(es):.4f}  mean={np.mean(es):.4f}")

# -- 5. Guardar model JSON --
model_out = {
    "seq": "288_SEQ",
    "method": method,
    "n_pairs": len(pairs),
    "n_points_per_chromatogram": n_pts,
    "global_factor": float(global_factor),
    "regression_pointwise": {
        "slope": float(slope_pt), "intercept": float(intercept_pt),
        "r2": float(r2_pt), "n_points": len(all_d),
    },
    "regression_origin": {"slope": float(slope_origin), "r2": float(r2_origin)},
    "factor_per_fraction": frac_factors,
    "factor_local": {
        "t": t_ref.tolist(),
        "factor_smooth": factor_smooth.tolist(),
        "factor_median": [float(x) if not np.isnan(x) else None for x in factor_median],
    },
    "usage": "y_direct_estimated = y_uib * factor_smooth (interpolat al time base)",
}
model_path = OUT_DIR / "model_direct_uib.json"
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_out, f, indent=2, ensure_ascii=False)
print(f"\nModel: {model_path}")

# -- 6. CSV parells --
csv_path = OUT_DIR / "direct_uib_pairs.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("sample,rep,area_direct,area_uib,ratio,pearson_shape\n")
    for p in pairs:
        ad = float(np.trapezoid(np.maximum(p["y_direct"], 0), p["t"]))
        au = float(np.trapezoid(np.maximum(p["y_uib"], 0), p["t"]))
        ratio = au / ad if ad > 0 else 0
        r_val, _ = sp_stats.pearsonr(p["y_direct"], p["y_uib"])
        f.write(f"{p['sample']},{p['rep']},{ad:.1f},{au:.1f},{ratio:.4f},{r_val:.4f}\n")
print(f"CSV: {csv_path}")

# -- 7. Plots --
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 14), dpi=120)
    fig.suptitle("288_SEQ - Correlacio cromatograma Direct <-> UIB", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.3)

    # -- P1: Factor local f(t) --
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.fill_between(t_ref, factor_median - factor_std, factor_median + factor_std,
                     alpha=0.15, color="steelblue", label="+/- 1 std")
    ax1.plot(t_ref, factor_median, "b-", lw=0.5, alpha=0.4, label="Mediana crua")
    ax1.plot(t_ref, factor_smooth, "r-", lw=1.5, label="Factor suavitzat")
    ax1.axhline(global_factor, color="green", ls="--", lw=1,
                label=f"Global = {global_factor:.3f}")
    # Zones fraccions
    colors_frac = {"BioP": "#E8D5B7", "HS": "#B7D5E8", "BB": "#D5E8B7",
                   "SB": "#E8B7D5", "LMW": "#D5B7E8"}
    for frac, (ti, tf) in FRACTIONS.items():
        ax1.axvspan(ti, tf, alpha=0.08, color=colors_frac.get(frac, "gray"))
        ax1.text((ti+tf)/2, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0.5,
                 frac, ha="center", fontsize=7, alpha=0.6)
    ax1.set_xlabel("Temps (min)")
    ax1.set_ylabel("Factor Direct/UIB")
    ax1.set_title("Factor de conversio local f(t)")
    ax1.set_ylim(0.3, 1.2)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)
    # Re-annotate fractions at correct y
    for frac, (ti, tf) in FRACTIONS.items():
        ax1.text((ti+tf)/2, 0.35, frac, ha="center", fontsize=7, alpha=0.6)

    # -- P2: Scatter punt-a-punt (subsampled) --
    ax2 = fig.add_subplot(gs[0, 2])
    # Subsample per visualitzacio
    n_sub = min(50000, len(all_d))
    idx = np.random.default_rng(42).choice(len(all_d), n_sub, replace=False)
    ax2.scatter(all_u[idx], all_d[idx], s=1, alpha=0.1, zorder=2)
    x_fit = np.linspace(0, np.percentile(all_u, 99.5), 100)
    ax2.plot(x_fit, slope_pt * x_fit + intercept_pt, "r-", lw=1.5,
             label=f"y={slope_pt:.3f}x+{intercept_pt:.1f}\nR2={r2_pt:.4f}")
    ax2.plot(x_fit, x_fit, "k:", lw=0.5, alpha=0.3, label="1:1")
    ax2.set_xlabel("UIB (ppb)")
    ax2.set_ylabel("Direct (ppb)")
    ax2.set_title(f"Punt-a-punt ({n_sub//1000}k pts)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # -- P3-P5: Exemples overlay (3 mostres) --
    example_indices = [0, len(pairs)//2, len(pairs)-1]
    for plot_i, pi in enumerate(example_indices):
        ax = fig.add_subplot(gs[1, plot_i])
        p = pairs[pi]
        y_est = p["y_uib"] * factor_smooth
        ax.plot(p["t"], p["y_direct"], "b-", lw=1, label="Direct (real)", alpha=0.8)
        ax.plot(p["t"], p["y_uib"], "orange", lw=0.7, label="UIB (raw)", alpha=0.5)
        ax.plot(p["t"], y_est, "r--", lw=1, label="Direct (estimat)", alpha=0.8)
        # Zones fraccions
        for frac, (ti, tf) in FRACTIONS.items():
            ax.axvspan(ti, tf, alpha=0.05, color=colors_frac.get(frac, "gray"))
        ax.set_xlabel("Temps (min)")
        ax.set_ylabel("ppb")
        ax.set_title(f"{p['sample']} R{p['rep']}")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(8, 60)

    # -- P6: Residual per fraccio (boxplot) --
    ax6 = fig.add_subplot(gs[2, 0])
    frac_residuals = {}
    for frac, (ti, tf) in FRACTIONS.items():
        mask_t = (t_ref >= ti) & (t_ref < tf)
        errs = []
        for p in pairs:
            y_est = p["y_uib"][mask_t] * factor_smooth[mask_t]
            area_real = np.trapezoid(np.maximum(p["y_direct"][mask_t], 0), p["t"][mask_t])
            area_est = np.trapezoid(np.maximum(y_est, 0), p["t"][mask_t])
            if area_real > 10:
                errs.append((area_est - area_real) / area_real * 100)
        frac_residuals[frac] = errs

    bp_data = [frac_residuals[f] for f in FRACTIONS]
    bp = ax6.boxplot(bp_data, labels=list(FRACTIONS.keys()), patch_artist=True)
    for patch, frac in zip(bp['boxes'], FRACTIONS):
        patch.set_facecolor(colors_frac.get(frac, "lightgray"))
    ax6.axhline(0, color="black", lw=0.5)
    ax6.set_ylabel("Error area (%)")
    ax6.set_title("Residual per fraccio (factor local)")
    ax6.grid(True, alpha=0.3, axis="y")

    # -- P7: Pearson cromatograma per mostra --
    ax7 = fig.add_subplot(gs[2, 1])
    pearsons = []
    labels_p = []
    for p in pairs:
        y_est = p["y_uib"] * factor_smooth
        mask = (p["y_direct"] > NOISE_FLOOR) | (y_est > NOISE_FLOOR)
        if np.sum(mask) > 10:
            r, _ = sp_stats.pearsonr(p["y_direct"][mask], y_est[mask])
            pearsons.append(r)
            labels_p.append(f"{p['sample'][:8]}")
    ax7.barh(range(len(pearsons)), pearsons, color="steelblue", alpha=0.7)
    ax7.set_yticks(range(len(pearsons)))
    ax7.set_yticklabels(labels_p, fontsize=5)
    ax7.set_xlabel("Pearson (Direct real vs estimat)")
    ax7.set_title(f"Qualitat per mostra (mediana={np.median(pearsons):.4f})")
    ax7.axvline(0.99, color="green", ls="--", lw=0.8, alpha=0.5)
    ax7.set_xlim(min(min(pearsons)-0.01, 0.95), 1.001)
    ax7.grid(True, alpha=0.3, axis="x")

    # -- P8: Comparacio 3 models --
    ax8 = fig.add_subplot(gs[2, 2])
    models_compare = {
        "Factor global": lambda y_u: y_u * global_factor,
        "Regressio": lambda y_u: slope_pt * y_u + intercept_pt,
        "Factor local": lambda y_u: y_u * factor_smooth,
    }
    x_pos = np.arange(len(models_compare))
    area_biases = []
    area_stds = []
    shape_meds = []
    for mname, mfn in models_compare.items():
        ea = []
        es = []
        for p in pairs:
            y_est = mfn(p["y_uib"])
            ar = np.trapezoid(np.maximum(p["y_direct"], 0), p["t"])
            ae = np.trapezoid(np.maximum(y_est, 0), p["t"])
            if ar > 0:
                ea.append((ae - ar) / ar * 100)
            mask = (p["y_direct"] > NOISE_FLOOR) | (y_est > NOISE_FLOOR)
            if np.sum(mask) > 10:
                r, _ = sp_stats.pearsonr(p["y_direct"][mask], y_est[mask])
                es.append(r)
        area_biases.append(np.mean(ea))
        area_stds.append(np.std(ea))
        shape_meds.append(np.median(es))

    bars = ax8.bar(x_pos - 0.2, shape_meds, 0.35, label="Pearson forma", color="steelblue")
    ax8.bar(x_pos + 0.2, [100-abs(b) for b in area_biases], 0.35,
            label="100-|bias area|%", color="coral", alpha=0.7)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(list(models_compare.keys()), fontsize=8)
    ax8.set_ylabel("Score")
    ax8.set_title("Comparacio models")
    ax8.legend(fontsize=7)
    ax8.set_ylim(0.9, 1.01)
    ax8.grid(True, alpha=0.3, axis="y")
    # Anotar valors
    for i, (b, s, sm) in enumerate(zip(area_biases, area_stds, shape_meds)):
        ax8.text(i, 0.905, f"bias={b:.1f}%\nstd={s:.1f}%\nr={sm:.4f}",
                 ha="center", fontsize=6, va="top")

    fig.savefig(OUT_DIR / "chromatogram_correlation.png", bbox_inches="tight")
    print(f"\nPlot: {OUT_DIR / 'chromatogram_correlation.png'}")
    plt.close()

    # -- Plot extra: tots els overlays (miniatures 4x4) --
    n_per_page = 16
    n_pages = (len(pairs) + n_per_page - 1) // n_per_page
    for page in range(n_pages):
        fig_ov, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=100)
        fig_ov.suptitle(f"288_SEQ - Overlay Direct vs Estimat ({page+1}/{n_pages})",
                        fontweight="bold")
        for idx_in_page in range(n_per_page):
            idx_global = page * n_per_page + idx_in_page
            r_ax, c_ax = divmod(idx_in_page, 4)
            ax = axes[r_ax][c_ax]
            if idx_global >= len(pairs):
                ax.set_visible(False)
                continue
            p = pairs[idx_global]
            y_est = p["y_uib"] * factor_smooth
            ax.plot(p["t"], p["y_direct"], "b-", lw=0.8, alpha=0.8)
            ax.plot(p["t"], y_est, "r--", lw=0.8, alpha=0.7)
            # Pearson
            mask = (p["y_direct"] > NOISE_FLOOR) | (y_est > NOISE_FLOOR)
            r_val = sp_stats.pearsonr(p["y_direct"][mask], y_est[mask])[0] if np.sum(mask) > 10 else 0
            # Area error
            ar = np.trapezoid(np.maximum(p["y_direct"], 0), p["t"])
            ae = np.trapezoid(np.maximum(y_est, 0), p["t"])
            err = (ae - ar) / ar * 100 if ar > 0 else 0
            ax.set_title(f"{p['sample']} R{p['rep']}\nr={r_val:.3f} err={err:.1f}%",
                         fontsize=7)
            ax.set_xlim(8, 60)
            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.2)
        fig_ov.tight_layout()
        fig_ov.savefig(OUT_DIR / f"overlays_page{page+1}.png", bbox_inches="tight")
        plt.close()
    print(f"Overlays: {n_pages} pagines")

except ImportError as e:
    print(f"\n[matplotlib no disponible: {e}]")

print(f"\n{'='*60}")
print("RESUM MODEL")
print(f"{'='*60}")
print(f"  Factor global:    Direct = {global_factor:.4f} x UIB")
print(f"  Regressio:        Direct = {slope_pt:.4f} x UIB + {intercept_pt:.2f}")
print(f"  Factor local:     vector f(t) suavitzat ({n_pts} punts)")
print(f"  Recomanat:        Factor local (millor preservacio de forma)")
print(f"\n  Aplicacio:  y_direct_est[i] = y_uib[i] * factor_smooth[i]")
print(f"              (interpolar factor_smooth al time base de la mostra)")
print("\nFet.")
