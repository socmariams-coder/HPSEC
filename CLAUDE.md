# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.
It is read automatically at the start of every session.

## Project

HPSEC Suite — analytical chemistry system for HPSEC chromatogram processing (DAD-DOC).
Python + PyQt6 GUI. No test suite; validation via DOCtor_validation.py.

## Key commands

```bash
python HPSEC_Suite.py          # Main GUI
python batch_process.py        # Batch processing
pip install -r requirements.txt
```

## Architecture summary

- **hpsec_core.py** — shared math (bi-Gaussian, Batman, SNR, peak repair)
- **hpsec_calibrate.py** — calibration engine (rf_mass_cal, intercept, QC)
- **hpsec_consolidate.py** — .D file consolidation to Excel
- **gui/** — PyQt6 GUI panels (consolidate, calibrate, process/analyze, dashboard)
- **Calibration_Reference.json** — active calibration data (rf_mass_cal, intercept, thresholds)

## Calibration model

- `rf_mass_cal`: per signal/mode dict `{"direct": {"column": X, "bp": Y}, "uib": {...}}`
- `intercept`: same nested structure, or scalar 0 for backwards compat
- Formula: `ppm = (Area - intercept) * 1000 / (rf_mass_cal * volume_uL)`
- When intercept=0 (origin model): simplifies to `ppm = Area * 1000 / (rf_mass_cal * volume)`

## Feature status

Track real implementation state. Update this section after each feature commit.
Mark features as DONE only when code is fully functional end-to-end, not when placeholders exist.

- [x] Consolidation panel (read .D, create Excel) — DONE
- [x] Calibration: rf_mass_cal per-mode (direct/uib x column/bp) — DONE
- [x] Calibration: intercept per-mode (nested dict, quantify, plot) — DONE (4f2d245)
- [x] Calibration: RF per-mode applied (COLUMN rf=628+intercept=81, BP rf=915 origin) — DONE (JSON updated)
- [x] Calibration: QC history + KHP validation — DONE
- [x] Calibration: plot recta with tolerance bands — DONE
- [x] Process wizard: batch anomaly detection — DONE
- [x] Analyze panel: chromatogram view + anomaly flags — DONE
- [x] Analyze backend: areas_uib in BP DUAL mode — DONE (v1.6.0)
- [x] Analyze backend: quantify_sample with rf_direct/rf_uib separate — DONE (v1.6.0)
- [x] Analyze backend: R² DAD per 6 wavelengths (pearson_per_wavelength) — DONE (v1.6.0)
- [x] Dashboard panel — DONE
- [x] Analyze panel: taula unificada DOC+DAD (14 cols: Sel DOC, Sel DAD, SNR, SNR_254, R²_DOC, R²_DAD, HCI) — DONE (851f0b1)
- [x] Analyze panel: dropdowns rèplica independents DOC vs DAD + opció "Cap" — DONE
- [x] Analyze panel: detail dialog grid layout (DOC|UIB + DAD pairs + fraction table) — DONE (851f0b1)
- [x] Analyze panel: Batman repair button in detail dialog (repair_with_parabola) — DONE
- [x] Analyze panel: sample_valid flag (ambdues rèpliques amb anomalies no reparables) — DONE
- [x] Analyze panel: HCI (Humic Character Index) column with PCA+LDA model — DONE
- [x] Analyze backend: HCI compute_hci() from Export3D (hpsec_humic.py, no sklearn) — DONE
- [x] Export: HCI + HCI_Character columns in SUMMARY.xlsx — DONE
- [ ] Analyze panel: mostrar bigaussian (R², asym, quality) per BP — PENDING
- [ ] Analyze panel: mostrar timeouts amb icones/tooltip — PENDING
- [ ] Analyze backend: detecció deriva baseline DAD per replica selection — PENDING (TODO a hpsec_analyze.py L1210)
- [x] Calibration: flux renovació calibració global (UI panel + regression) — DONE (GlobalCalibrationPanel refactored: 2 vistes CAL+QC)
- [x] Calibration: auto-fit rf_mass_cal + intercept from KHP history (regression) — DONE (fit_calibration_from_history)
- [x] Calibration: separació SEQ_CAL vs producció KHP al panell global — DONE
- [x] Calibration: QC Levey-Jennings chart (desviació % vs recta vigent) — DONE
- [x] Calibration: requantify_analysis_json() per recalibració retroactiva — DONE
- [x] Calibration: patch_excel_calibration() per patch Excels existents — DONE
- [x] Calibration: calibration_fingerprint per detectar canvis (is_cal_stale) — DONE
- [x] Dashboard: indicador ⟳ calibració obsoleta a columna Analitzar — DONE
- [x] Calibration: KHP DAD 254nm fallback from MasterFile 3-DAD_KHP in manifest loading — DONE
- [x] Calibration: clean_khp_history() to remove invalid entries (conc=0, area=0) — DONE
- [x] Export: PDF analysis report (generate_analysis_report.py) — DONE
- [x] Export: SUMMARY.xlsx ampliat (A_UIB, ppm_UIB, A_254, SNR_254, R²_DOC, R²_DAD, Anomalies) — DONE
- [x] Export: ID sheet with full traceability (RF, intercept, anomalies, batman repair, timeouts) — DONE
- [x] Export: skip invalid samples (sample_valid=False or "Cap") — DONE
- [x] Export: BP integration in COLUMN mode (ID_BP sheet, BP rows in RESULTS, BP cols in SUMMARY) — DONE
- [x] Export: fractions loaded from hpsec_config.json (not hardcoded) — DONE
- [x] Export: timeout zone_summary fix (dict format) — DONE
- [x] Wizard: Step 4 "Revisar" replaces "Exportar" (ReviewSummaryPanel with stats, charts, generate button) — DONE
- [x] Wizard: "Generar Resultats" exports to SEQ/RESULTATS/ (Excels) + SEQ/CHECK/ (SUMMARY) — DONE
- [x] Analyze backend: light analysis for BLANK/CONTROL (area_total + SNR only, no fractions/quantification) — DONE
- [x] Analyze panel: BLANC/CONTROL rows at bottom with separator + grey background — DONE
- [x] Export: lightweight Excel for BLANK/CONTROL (ID + DOC sheets only) — DONE
- [x] Export: SUMMARY.xlsx Type column (SAMPLE/PR/BLANK/CONTROL) — DONE
- [x] Architecture refactor: unify anomaly system (ANOMALY_CATALOG in hpsec_warnings.py) — DONE (0d0b960)
- [ ] **Import: GUI pregunta volum quan no trobat al v11** — PENDING (IMPORTANT)
  - Quan `inj_volume_source='default'` o `inj_volume=None`, diàleg confirmació a l'usuari
  - Validació creuada col13 vs 0-INFO quan ambdós existeixen i discrepan
  - Icona/tooltip a calibració si volum és estimat (no del manifest)
- [ ] Architecture refactor: unify detection functions in hpsec_core.py — PENDING
- [x] Integration: derivative-based peak boundaries (Agilent tangent projection) — DONE (find_peak_boundaries in hpsec_core.py)
- [x] Integration: re-process all KHP data with new derivative method — DONE (batch re-calibrate 137 SEQs, KHP_History regenerated with 96 entries)
- [x] Config panel: 3 tabs per impacte (Anàlisi/Seqüència/Sistema), tots params editables — DONE
- [x] Config panel: badges d'impacte (retroactiu/futur), diàleg de reprocessament — DONE
- [x] Config panel: TimeFractionsEditor, TimeoutZonesEditor, WavelengthSelector, PatternListEditor — DONE
- [x] Config backend: config fingerprint (SHA-256 16 chars) per detectar obsolescència — DONE
- [x] Config backend: migració batman_max_sep → batman_max_sep_min — DONE
- [x] Config backend: REPROCESS_SECTIONS/FUTURE_SECTIONS/IMMEDIATE_SECTIONS constants — DONE
- [x] Wizard: Rename step 2 "Calibrar" → "Verificar" (TAB_NAMES + tab_names) — DONE
- [x] Wizard: Delay diagnostic tool at step 2 (shift indicator, slider, impact preview, reimport) — DONE
- [x] Wizard: SEQ_CAL detection at step 2 (CalibratePanel) + regression moved to step 3 (AnalyzePanel) — DONE
- [ ] Wizard: Apply calibration at step 4 (Revisar) + retroactive requantification + SEQ list — PENDING

## Research / Exploration (not integrated into Suite)

All exploratory scripts and results live in `research/` — NOT part of the production Suite.

### Humic exploration (`research/humic_exploration/`)
- `rf_humic_lda.py` — PCA+LDA model HA/FA discrimination (LOO 93.5%, 31 refs, 242 features)
- `rf_humic_analysis.py` — RF classifier + spectral analysis of HS zone
- `rf_humic_index.py` — Humic index variants exploration
- `rf_spectral_exploration.py` — General DAD spectral exploration
- `rf_humic_std_report.py` — Standards inventory report
- Output: `rf_humic_analysis/` (10 plots + CSVs)

### BB exploration (`research/bb_exploration/`)
- `rf_bb_analysis.py` — RF (100% CV) + PCA+LDA (94.4% LOO) on BB zone, 249 features, 11 classes
- `rf_bb_fingerprint.py` — Organic/inorganic separation at BB (A254=organic, excess A210=inorganic)
  - Key finding: A210 at BB is >90% inorganic for water samples
  - PTLL/PTT ratio: 3.0x inorganic load difference
  - Organic reference ratio A210/A254 at BB: 13.1 (geometric mean HA=6.7, FA=25.8)
- `rf_bb_temporal.py` — Temporal sub-structure within BB (early/mid/late decomposition)
  - Key finding: Inorganic elution timing differs between water types (FR peaks mid-BB, LAB/LAB_T late-BB)
  - CAVEAT: Temporal alignment between chromatograms not verified — sub-window metrics may be unreliable
- Output: `rf_bb_analysis/`, `rf_bb_fingerprint/`, `rf_bb_temporal/`

### Integration status
- [x] HCI (Humic Character Index) integrated into Suite (hpsec_humic.py + model JSON) — DONE
- [ ] BB fingerprint integration — BLOCKED (temporal alignment not verified, no reference standards for BB)
- [ ] BB organic/inorganic metrics as single-sample report — PENDING (needs alignment verification first)

## Known bugs (pendents de fix)

- **Import re-llegeix masterfile quan ja existeix manifest JSON**: Quan una SEQ ja té import_manifest.json,
  el panell d'importar hauria de carregar directament del JSON sense tornar a parsejar el masterfile.
  Actualment ho re-llegeix sempre.
- **072_SEQ: dades DAD KHP no detectades al carregar, però sí al refer**: FIXED — El manifest de seqs
  antigues (pre-FIX F2.2) no tenia `dad` info per KHP. Afegit fallback a 3-DAD_KHP del MasterFile quan
  el manifest no conté DAD per KHP samples.
- **Import auto-detecta columna volum erròniament en seqs BP**: FIXED — Eliminat BP guard que bloquejava
  lectura de volums per BP. Ara l'heurístic index-13 s'aplica a TOTS els modes. Si cap font té volum
  (ni capçalera, ni 0-INFO, ni index-13), warning emès i volum = None (no se suposa res).
  `get_injection_volume()` ara accepta `manifest_volume` amb prioritat absoluta. `register_calibration()`
  rebutja entrades sense volum en lloc de suposar 100µL.
- **KHP_History.json: blancs (conc=0) marcats valid_for_calibration=True**: FIXED — Guards afegits a
  `register_calibration()` per rebutjar conc=0 i area=0. Funció `clean_khp_history()` disponible
  per netejar entrades antigues invàlides.

## Design decisions

- **Analyze table**: ONE unified table with all DOC+DAD columns. NO DOC/DAD toggle selector (not practical).
  PLAN_TAULA_ANALITZAR.md proposed a selector but it was rejected.

## Instructions for Claude

1. **Read this file at session start** to understand current feature state.
2. **After implementing a feature**, update the Feature Status section above.
3. **Never mark a feature DONE** if the code is a placeholder or partial stub.
4. **When user asks about a feature**, check this list first to give accurate status.
5. Comments and variable names: Catalan/English mix is normal.
6. See `PROVES/CLAUDE.md` for detailed technical reference (thresholds, TOC params, etc).
7. **Working Notes**: Actualitzar la secció Working Notes cada vegada que es toca un tema nou
   rellevant (NO només al final de sessió). Incloure context, decisions, dades trobades.
   Això és CRÍTIC — l'usuari no ha de re-explicar context entre sessions.

## Working notes

> Last updated: 2026-02-21

### Revisió calibració KHP (EN CURS)

**Bug crític trobat i fixat — finestra Savitzky-Golay a `find_peak_boundaries()`:**
- `hpsec_core.py` L1112: finestra SG era `n // 20 * 2 + 1` → proporcional al cromatograma (117 pt = 7.9 min)
- Amb finestra massa gran, el suavitzat aplanava el pic i les pendents al punt d'inflexió
  eren massa petites → la projecció tangent donava límits 14x massa amples (27σ vs 4σ esperats)
- **FIX**: finestra SG basada en temps (`sg_target_min = 0.7` min ≈ FWHM típic pic HPLC-SEC)
  → `sg_window = int(0.7 / dt_median)`, amb clamp a [7, n]
- Verificat: amb SG=11pt (0.73 min), projecció tangent dóna 4.1-4.4σ = correcte per gaussiana

**Resultats calibració COLUMN DOC (amb fix SG):**
- 36 entrades OK (de 123 totals), R²=0.977, slope=749, RF mediana=730, CV=16%
- Per concentració: 1ppm RF=724±109, 2ppm RF=726±27, 3ppm RF=828±3, 5ppm RF=749±77
- SEQs 256-274 COLUMN tenien v=100µL al manifest però àrees idèntiques a v=400µL
  → detecció automàtica VOL_SUSPECT: si COLUMN amb v=100 i àrea>400, corregir a v=400

**Calibració BP DOC — PENDENT, alineació temporal en curs:**
- R²=0.019, RF CV=165% — **NO ACCEPTABLE** (amb dades sense alinear)
- **Causa arrel**: el 2-TOC del masterfile és un flux continu de mesures TOC (una cada 4s).
  La Suite assigna files TOC a cada injecció HPLC via un desfase temporal (0-INFO "Net delay").
  Si el desfase és erroni o absent (nan), les files assignades no corresponen al cromatograma
  real → baselines de ~100 ppb (= nivell aigua MQ del TOC), pics desplaçats, àrees incorrectes.

**Delay HPLC↔TOC:**
- El delay (Net delay) és el paràmetre clau per assignar files TOC a cada injecció HPLC.
  Es llegeix del v11 original (full 0-CHECK) i s'aplica al MasterFile (full 0-INFO).
- `hpsec_migrate_master.py` calcula el delay a partir de les hores rellotge HPLC i TOC.
  **IMPORTANT**: aquest càlcul automàtic pot ser erroni (demostrat amb la 156: delay automàtic
  -3.01 vs delay real 7.67 del v11). Sempre verificar amb el v11 original.
- **156_SEQ_BP**: delay corregit de -3.01 a 7.67 min (del v11). Amb delay erroni, les files
  estaven desplaçades -175 rows (~11.7 min) i cada R1 agafava el pic de la injecció anterior.

**Flags de qualitat (khp_reintegrate_doc.py):**
- MULTI_PEAK: CR local (±5 min) < 0.70
- MULTI_PEAK_MILD: CR local 0.70-0.90
- T_RET_ANOMAL: COLUMN fora 18-28 min, BP fora 0-12 min
- VOL_SUSPECT: COLUMN v=100µL amb àrea>400 (probablement v=400µL erroni)
- FALLBACK_MAX: `find_peak_boundaries` ha caigut al fallback threshold

**Script diagnòstic: `khp_reintegrate_doc.py`:**
- Llegeix manifest JSON per baseline i volum (mateixa metodologia que la Suite)
- Extreu DOC del MasterFile 2-TOC amb `extract_doc_from_masterfile()`
- Integra amb `detect_main_peak()` sobre y_net = max(y_raw - baseline, 0)
- Genera CSV (226 entrades) + scatter plots a `REGISTRY/review/`

**Fitxers a REGISTRY/review/:**
- `khp_doc_reintegration.csv`: 226 entrades amb àrea, RF, flags, vol_corrected
- `scatter_doc_clean.png`: Area vs ug_DOC per mode amb recta regressió
- `153_SEQ_diagnostic.png`: Diagnòstic visual pics solapats KHP1/KHP500
- `peak_limits_comparison.png`: Comparació 5 mètodes de límits (tangent, inflexió, ±2σ, ±3σ, 5%)
- `chromatograms/*.png`: 226 cromatogrames individuals

**Regeneració MasterFiles BP:**
- `regenerate_bp_masterfiles.py`: pipeline complet (rawdata → MasterFile → delay → 4-TOC_CALC)
- Dry-run: 45 OK, 5 REVISAR, 3 sense HPLC
- Pendent: usuari verificant delays de les 5 REVISAR (111, 169, 221, 225, 277)
- **111_SEQ_BP és clau** (l'usuari ho marca explícitament)

**Detecció KHP per nom de mostra:**
- SEQs amb concentracions al nom sense "KHP": 111, 113, 114 (KHP pur), 148 (mostra+buff), 225 (HA/FA)
- Solució: convenció `_CAL` al nom de carpeta. Si la SEQ conté "CAL", tota injecció no-exclosa
  (MQ/NaOH/BUFFER/etc.) es tracta com a KHP. Implementat a `_extract_khp_from_masterfile()`.
- SEQs a renombrar per l'usuari: `111→111_CAL`, `113→113_CAL`, `114→114_CAL`
- La 148 i 225 NO són KHP pur (buffer i HA/FA respectivament) → no renombrar.

**Sistemàtica preparació KHP — IMPORTANT:**
- A partir de la **111_SEQ_BP** (inclosa), els KHP es preparen amb **pipetes Pasteur**.
- Les SEQs anteriors a la 111 (072B–109B) tenen sistemàtica desconeguda.
- Pot haver-hi **diferències sistemàtiques** en la preparació entre pre-111 i post-111.
- Considerar separar les rectes de calibració pre/post-111 o verificar si hi ha salt en RF.

**Resultats regressió BP (amb SEQs _CAL):**
- 281 entrades totals (123 COLUMN + 158 BP), 137 OK
- **114_SEQ_BP_CAL: R²=0.9954**, slope=681, intercept=-3.9 → RF_BP=681 (referència!)
  - Test linealitat volum: 5ppm × 6 volums (50-200µL). Resultat perfecte.
  - t_max coherent: 2.4-2.7 min. RF consistent per tots els volums (654-708).
  - RF_BP=681 és ~7% menys que RF_COLUMN=730 — diferència plausible camí hidràulic.
- **113_SEQ_BP_CAL: R²=0.858**, slope=1224, intercept=232
  - L'intercept alt (232) indica àrea de fons significativa a baixa concentració.
  - Àrees inflades per offset: a 0.05ppm, àrea=210 ≈ tot offset, no KHP real.
  - Slope corregit (1224) no és directament comparable, cal model amb intercept.
- **111_SEQ_BP_CAL: PROBLEMÀTICA** — R²=0.025
  - Àrees molt baixes (6-132) per TOTES les conc (incloent 5ppm, que a 114 dóna 335).
  - 16/29 entrades amb MULTI_PEAK flags.
  - Probable causa: delay no prou precís o 4-TOC_CALC no captura el pic complet.
  - Cal revisar: la 111 té 3 blocs de 16 inj (48 total), potser cada bloc necessita
    delay diferent (tèrmica del sistema canvia en 8h de seqüència).
- Totes BP (sense outlier 271): R²=0.31, slope=488, intercept=159
- Post-111 only: R²=0.44, slope=600, intercept=101

**Regressió combinada (millor estimació RF_BP):**
- De la 114 (R²=0.9954): **RF_BP_direct = 681 ± 20** (intercept ≈ 0)
- Coherent amb la tendència de les altres SEQs a concentracions altes (5ppm RF~683)

**Anàlisi integració BP (analyze_integration_bp.py) — COMPLETAT:**
- 6 mètodes comparats: thr1%, thr5%, tangent Agilent, bigauss 3s, bigauss analytic, trapezoid net
- **Tangent (Agilent) confirmat com a millor**: R²=0.978 combinat, 0.999 per 152 sola
- **152_SEQ_BP com a referència primària**: slope=817, intercept=11 (~0), R²=0.9992, n=10
- **156_SEQ_BP exclosa**: intercept=58 (possible contaminació DOC preparació)
- Diferència RF: COLUMN=628 vs BP=817 és efecte d'integració, no del detector
  (el detector TOC és el mateix; calculant àrea efectiva per µg: COLUMN=830, BP=817 = 1.6% diferència)

**Calibration_Reference.json ACTUALITZAT (2026-02-20):**
- `rf_mass_cal.direct.bp`: 915 → **817** (de 152_SEQ_BP tangent)
- `r2.bp`: 0.8213 → **0.9992**
- `n_points.bp`: 7 → **10**
- `intercept.direct.bp`: 0 (mantingut — intercept 152=11 ≈ negligible)

**Fix volums d'injecció (CRÍTIC):**
- `hpsec_migrate_master.py`: volum llegit de col N (Unnamed:13) del v11, NO hardcoded.
  Si no existeix → 0-INFO = "DESCONEGUT" + warning. Detecta volums variables.
- Eliminat BP guard a `hpsec_import.py`: ara l'heurístic index-13 s'aplica a TOTS els modes
- `get_injection_volume()` a `hpsec_calibrate.py`: nou param `manifest_volume` amb prioritat absoluta
- `register_calibration()`: rebutja entrades sense volum (en lloc de suposar 100µL)
- `quantify_sample()` a `hpsec_analyze.py`: warning si volum no ve del manifest
- Propagació `method` a `master_data` per futur ús
- **PENDENT**: els MasterFiles BP existents (generats amb l'antic hardcode BP=100) poden tenir
  volums incorrectes. Cal regenerar els afectats (especialment 107, i qualsevol BP amb v≠100).

### Comparació calibracions BP: 292 vs 152 (2026-02-21)

**292_SEQ_CAL_BP** (referència nova, 6 conc ben distribuïdes):
- slope=647, intercept=2.8, R²=0.9987, n=6 (0.1, 0.25, 0.5, 1, 2, 5 ppm)
- RF_mass molt consistent, intercept ≈ 0

**152_SEQ_BP** (referència antiga):
- slope=812, intercept=14.2, R²=0.9994, n=4 (0.25, 1, 3, 5 ppm)
- RF_mass inconsistent per concentració: 1334 a 0.25ppm vs 834 a 5ppm → suggereix offset de fons

**Comparació directa**: 292 dóna -20.3% menys RF que 152
- A 0.25ppm: 152 àrea=230 vs 292 àrea=174 → 152 inflada per possible offset DOC
- La 292 (6 punts, R²=0.999, bona distribució) és més fiable que la 152 (4 punts, offset)

**Evolució temporal RF:**
- BP: tendència -1.5 RF/SEQ, alta variabilitat (CV ~30%)
- COLUMN: pràcticament estable (tendència ~0/SEQ)

**PENDENT**: Actualitzar `Calibration_Reference.json` BP RF de 817 a ~650 (ref 292)
- L'usuari ha dit "de moment prenem nota, queda pendent d'actualitzar"

### Anàlisi 293_SEQ_CAL COLUMN (2026-02-21)

**Històric COLUMN (pre-293):**
- Global clean (30 entrades, 400µL, 0.25-5ppm): slope=785, intercept=3, R²=0.980
- Recent (>250): slope=751, intercept=40, R²=0.967
- 2ppm only (més estable): RF=797±42

**293_SEQ_CAL COLUMN:**
- 3 entrades (0.1, 0.25, 0.5 ppm a 400µL) — només concentracions baixes
- slope=620, intercept=41.3, R²=0.9982
- **vs referència actual (RF=628, intercept=81): -1.3%** — pràcticament idèntic!
- vs històric global (RF=785): -21.1%
- vs històric recent (RF=751): -17.5%
- **Conclusió**: la referència actual (RF=628+intercept=81) és correcta per COLUMN

### Bugs fixats sessió 2026-02-21

**Concentracions decimals (0.1, 0.25, 0.5 ppm):**
- `get_condition_key()`: `int(conc_ppm)` truncava 0.1→0, 0.25→0, 0.5→0
  → Fix: format decimal amb trailing zero stripping
- 8+ llocs GUI amb `:.0f` o `int(conc)` → tot canviat a `:g`
- Filtres concentració: tolerància absoluta ±1 ppm → relativa 10% (`max(0.01, conc*0.1)`)
- KHP_History.json: netejat entrada antiga amb conc=25.0, fixats 19 condition_keys truncats

**Timeout en KHP:**
- `validate_khp_quality()`: timeout WARNING ara → issues +100 (era: warnings +50)
- Simplificat: INFO=OK, WARNING/CRITICAL=outlier (no cal UI addicional)

**Gràfic històric:**
- Filtrat `qc_history` per concentració i volum abans de passar a `plot_calibration()`

### Refactor GlobalCalibrationPanel — Recta CAL + QC Monitor (2026-02-21)

**Motivació**: El panell barrejava totes les 250+ entrades KHP per fer regressió. Ara separa:
- SEQ_CAL (13 entrades) → Tab "Recta de Calibració" per construir/actualitzar calibració
- Producció (114 entrades) → Tab "Control de Qualitat" amb gràfic Levey-Jennings

**Fitxers modificats:**
- `hpsec_calibrate.py`: + `compute_calibration_fingerprint()` (SHA-256[:16]), + `requantify_analysis_json()` (recalcula ppm sense reprocessar)
- `hpsec_analyze.py`: estampa `calibration_fingerprint` al JSON d'anàlisi
- `hpsec_export.py`: + `patch_excel_calibration()` per patchejar Excels existents
- `gui/widgets/global_calibration_panel.py`: reescriptura completa (2 vistes: CalibrationLineView + QCMonitorView)
- `gui/models/sequence_state.py`: + camp `calibration_fingerprint`, + propietat `is_cal_stale`
- `gui/main_window.py`: connexió senyal `calibration_updated` → dashboard refresh
- `gui/widgets/dashboard_panel.py`: indicador ✔⟳ (taronja) quan calibració obsoleta

**Funcionalitats noves:**
- **CalibrationLineView**: selector SEQ_CAL amb checkboxes, regressió, scatter+residuals, stats per conc, comparació amb vigent, aplicar amb opció retroactiva + requantificació automàtica
- **QCMonitorView**: gràfic Levey-Jennings (desviació % vs recta), línies ±10%/±20%, tendència, indicador EN CONTROL/ATENCIÓ/FORA DE CONTROL
- **requantify_analysis_json()**: modifica NOMÉS ppm/fraccions als JSONs existents des de les àrees (que no canvien). Verificat: RF+10% → ppm -9.1%, àrees intactes
- **calibration_fingerprint**: patró idèntic a config_fingerprint — detecta canvi de calibració al dashboard

**Separació _CAL**: per convenció de nom, `"_CAL" in seq_name.upper()`. 13 entrades de 3 SEQs (111_CAL, 113_CAL, 114_CAL, 292_SEQ_CAL, 293_SEQ_CAL).

### Rename batman → irregular_top + fix integració pics irregulars (2026-02-21)

- Commit `9b12b28`: rename detect_batman→detect_irregular_top a 16 fitxers (168 ocurrències)
- Pre-repair a detect_main_peak: detectar pic irregular → reparar amb paràbola → find_peak_boundaries sobre senyal reparat → integrar sobre senyal original
- Verificat: KHP1 (1ppm) àrea 96.8→304.7, desviació -71%→-8.3%

### Canvis sessió 2026-02-21
- `hpsec_calibrate.py`: fix `get_condition_key()` decimals, fix timeout severity, fix concentration filter tolerance
- `gui/models/sequence_state.py`: `:g` format per concentracions
- `gui/widgets/calibrate_panel/panel.py`: `:g` formats (5 llocs), tolerància relativa (2 llocs), filtre qc_history
- `gui/widgets/analyze_panel/panel.py`: `:g` format KHP display
- `gui/widgets/history_panel.py`: `concs.add(conc)` float, `:g` formats (3 llocs), tolerància relativa
- `gen_cal_analysis.py`: script diagnòstic 4 pàgines PDF (292 vs 152 BP + temporal RF)
- Commits: `faa98f2` (decimal display), `2afdc14` (timeout + history filter)

### Canvis sessió anterior (2026-02-20)
- `Calibration_Reference.json`: BP rf=817, R²=0.999, n=10 (ref: 152_SEQ_BP tangent)
- `hpsec_migrate_master.py`: volum de col N del v11 (no hardcoded), warning si absent
- `hpsec_import.py`: eliminat BP guard volum, warning si cap injecció té volum, propagació method
- `hpsec_calibrate.py`: `get_injection_volume(manifest_volume=)`, `register_calibration()` rebutja vol=None
- `hpsec_analyze.py`: warning si volum no al manifest (heurístic com a fallback)
- `analyze_integration_bp.py`: script comparació 6 mètodes integració BP

### Sessions anteriors (resum)
- Carpetes renombrades: `111→111_CAL`, `113→113_CAL`, `114→114_CAL`
- `regenerate_bp_masterfiles.py`: regeneració MasterFiles + delay + 4-TOC_CALC
- `fix_masterfile_delay.py`: 45/53 SEQs amb delay mesurat aplicat
- `khp_reintegrate_doc.py`: lectura directa MasterFile, mode _CAL
- `hpsec_config.py`, `hpsec_import.py`, `hpsec_migrate_master.py`: pre-margin 1.5 min
- Pipeline 254→DOC: `analizar_khp_data()` reescrit, recalibrate_all_khp.py, calibration_review.py
- validate_khp_quality: 5 nous checks (bigaussian, t_ret, mismatch, no_dad)
- fit_calibration_from_history: mode="ALL", signal="254"
- Dashboard: columna Inj amb detecció importació incompleta
- Derivative integration: find_peak_boundaries amb projecció tangent Agilent
- KHP DAD 254nm: fallback robust des de MasterFile 3-DAD_KHP
- Startup optimization: 24s → <1s (lazy tabs, metadata-only JSON)
- Config panel: 3 tabs, badges impacte, fingerprint SHA-256
