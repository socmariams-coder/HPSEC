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

- **hpsec_core.py** — shared math (bi-Gaussian, irregular top, SNR, peak repair, saturation detection)
- **hpsec_calibrate.py** — calibration engine (rf_mass_cal, intercept, QC)
- **hpsec_consolidate.py** — .D file consolidation to Excel
- **gui/** — PyQt6 GUI panels (consolidate, calibrate, process/analyze, dashboard)
- **Calibration_Reference.json** — active calibration data (rf_mass_cal, intercept, thresholds)

## Calibration model (v3.0)

- **Calibracions independents per senyal/sensibilitat**: cada entrada a `calibrations[]` cobreix UN sol àmbit (`signal_scope` = 'direct'/'uib', `uib_sensitivity` = 700/1000/null)
- `rf_mass_cal`: dict planer `{"column": X, "bp": Y}` (el senyal ja va definit per `signal_scope`)
- `intercept`: idem planer `{"column": X, "bp": Y}`
- `active_calibration_ids`: dict `{"direct": "CAL_...", "uib": "CAL_...", "uib_700": "CAL_..."}` — una activa per àmbit
- Formula: `ppm = (Area - intercept) * 1000 / (rf_mass_cal * volume_uL)`
- When intercept=0 (origin model): simplifies to `ppm = Area * 1000 / (rf_mass_cal * volume)`
- **Migració automàtica**: `load_calibration_reference()` auto-migra v2.0 (nested) a v3.0 (planer) al primer accés

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
- [x] Analyze backend: UIB timeout estimation from DOC Direct pattern — DONE (estimate_timeout_for_uib + _estimate_uib_timeouts_from_sequence)
- [ ] Analyze backend: detecció deriva baseline DAD per replica selection — PENDING (TODO a hpsec_analyze.py L1210)
- [x] Calibration: flux renovació calibració global (UI panel + regression) — DONE (GlobalCalibrationPanel → consulta-only)
- [x] Calibration: auto-fit rf_mass_cal + intercept from KHP history (regression) — DONE (fit_calibration_from_history)
- [x] Calibration: separació SEQ_CAL vs producció KHP al panell global — DONE
- [x] Calibration: QC Levey-Jennings chart (desviació % vs recta vigent) — DONE
- [x] Calibration: requantify_analysis_json() per recalibració retroactiva — DONE
- [x] Calibration: patch_excel_calibration() per patch Excels existents — DONE
- [x] Calibration: calibration_fingerprint per detectar canvis (is_cal_stale) — DONE
- [x] Dashboard: indicador ⟳ calibració obsoleta a columna Analitzar — DONE
- [x] Wizard SEQ_CAL: detecció automàtica (≥3 KHP, ≥2 conc) + nom _CAL — DONE
- [x] Wizard SEQ_CAL: regressió al pas 3 (AnalyzePanel) amb taula punts, scatter, comparació vigent — DONE
- [x] Wizard SEQ_CAL: botó "Aplicar com a Nova Calibració" al pas 4 (ReviewSummaryPanel) — DONE
- [x] Wizard SEQ_CAL: validació ppm_obs vs ppm_teòric al pas 3 (AnalyzePanel) — DONE
- [x] Wizard SEQ_CAL: resum regressió al pas 4 (ReviewSummaryPanel) — DONE
- [x] GlobalCalibrationPanel: convertit a consulta-only (sense aplicar/requantificar) — DONE
- [x] Calibration: v3.0 independent per signal_scope/uib_sensitivity — DONE (migració automàtica v2→v3)
- [x] Calibration: UIB intercept independent a quantify_sample — DONE
- [x] GlobalCalibrationPanel: vista resum sense SEQ_CAL (taula params, scatter, historial) — DONE
- [x] Export: KHP chromatogram PNGs a CHECK/data/khp_plots/ — DONE (save_all_khp_chromatograms)
- [x] Export: PDF calibration report amb pàgines cromatogrames KHP — DONE
- [x] Wizard: Rename step 2 "Calibrar" → "Verificar" (TAB_NAMES + tab_names) — DONE
- [x] Wizard: Delay diagnostic tool at step 2 (shift indicator, slider, impact preview, reimport) — DONE
- [x] Wizard: Apply calibration at step 4 (Revisar) + retroactive requantification + SEQ list — DONE
- [x] Calibration: KHP DAD 254nm fallback from MasterFile 3-DAD_KHP in manifest loading — DONE
- [x] Calibration: clean_khp_history() to remove invalid entries (conc=0, area=0) — DONE
- [x] Export: PDF analysis report (generate_analysis_report.py) — DONE
- [x] Export: PDF calibration report (generate_calibration_report in hpsec_reports.py) — DONE (e0ba6ca)
- [x] Calibration: regression_data persisted in Calibration_Reference.json — DONE (ad93a2f)
- [x] Calibration: PDF report buttons at wizard Step 4 + GlobalCalibrationPanel — DONE (6527509)
- [x] Calibration: shared comparison HTML + prediction band helpers — DONE (e0ba6ca)
- [x] Export: SUMMARY.xlsx ampliat (A_UIB, ppm_UIB, A_254, SNR_254, R²_DOC, R²_DAD, Anomalies) — DONE
- [x] Export: ID sheet with full traceability (RF, intercept, anomalies, batman repair, timeouts) — DONE
- [x] Export: skip invalid samples (sample_valid=False or "Cap") — DONE
- [x] Export: BP integration in COLUMN mode (ID_BP sheet, BP rows in RESULTS, BP cols in SUMMARY) — DONE
- [x] Export: fractions loaded from hpsec_config.json (not hardcoded) — DONE
- [x] Export: timeout zone_summary fix (dict format) — DONE
- [x] Wizard: Step 4 "Exportar" (ExportPanel with BP consolidation, generate button, FAIR) — DONE
- [x] Wizard: "Generar Resultats" exports to SEQ/RESULTATS/ (Excels) + SEQ/CHECK/ (SUMMARY) — DONE
- [x] Analyze panel: Resum Visual collapsible (timeout, DOC stacked, DOC overlay, A254, DAD overlay) — DONE
- [x] Architecture refactor: Fusió Analitzar+Revisar → Analitzar+Exportar (review_summary_panel eliminat) — DONE
- [x] Analyze backend: light analysis for BLANK/CONTROL (area_total + SNR only, no fractions/quantification) — DONE
- [x] Analyze panel: BLANC/CONTROL rows at bottom with separator + grey background — DONE
- [x] Export: lightweight Excel for BLANK/CONTROL (ID + DOC sheets only) — DONE
- [x] Export: SUMMARY.xlsx Type column (SAMPLE/PR/BLANK/CONTROL) — DONE
- [x] Architecture refactor: unify anomaly system (ANOMALY_CATALOG in hpsec_warnings.py) — DONE (0d0b960)
- [x] Architecture refactor: unify warnings — single source ANOMALY_CATALOG + wizard warning bar — DONE
  - create_warning() + WARNING_DEFINITIONS eliminats (~200 línies codi mort)
  - Tots els backends usen create_anomaly() (hpsec_import, hpsec_calibrate, hpsec_analyze)
  - WarningBarWidget al wizard: barra persistent entre header i tabs
  - WarningReviewDialog eliminat (la barra mostra tot directament)
  - _update_validation() simplificada (usa warnings_structured directament)
  - get_max_warning_level() suporta severity (nou) i level (antic) keys
- [ ] **Import: GUI pregunta volum quan no trobat al v11** — PENDING (IMPORTANT)
  - Quan `inj_volume_source='default'` o `inj_volume=None`, diàleg confirmació a l'usuari
  - Validació creuada col13 vs 0-INFO quan ambdós existeixen i discrepan
  - Icona/tooltip a calibració si volum és estimat (no del manifest)
- [x] Memory optimization: DAD filter 6λ, TOC release, cal cache, df cleanup (~112 MB/SEQ) — DONE (f7d03d4)
- [ ] Architecture refactor: unify detection functions in hpsec_core.py — PENDING
- [x] Integration: derivative-based peak boundaries (Agilent tangent projection) — DONE (find_peak_boundaries in hpsec_core.py)
- [x] Integration: re-process all KHP data with new derivative method — DONE (batch re-calibrate 137 SEQs, KHP_History regenerated with 96 entries)
- [x] Config panel: 3 tabs per impacte (Anàlisi/Seqüència/Sistema), ~30 params editables — DONE
- [x] Config panel: badges d'impacte (retroactiu/futur), diàleg de reprocessament — DONE
- [x] Config panel: TimeFractionsEditor, TimeoutZonesEditor, WavelengthSelector, PatternListEditor — DONE
- [x] Config panel: simplificat (22 params interns eliminats del GUI) + contrasenya "LEQUIA" al guardar — DONE (7b275f7)
- [x] Warnings: ANOMALY_CATALOG unificat — 26 codis (16 analyze + 10 KHP calibrate), camp `action` a tots — DONE
- [x] Warnings: quality_issues reemplaçat per create_anomaly() a hpsec_calibrate.py — DONE
- [x] Warnings: dashboard alimenta Verificar/Revisar (calibrate_warnings, review_warnings) — DONE
- [x] Warnings: calibrate_panel badges severitat (icona+color+tooltip acció) en lloc de score numèric — DONE
- [x] Warnings: analyze_panel tooltips amb guia d'acció ("→ Excloure rèplica", etc.) — DONE
- [x] Warnings: unificació font única + barra wizard (ANOMALY_CATALOG, WarningBarWidget) — DONE
- [x] Analyze: BLANK (MQ) → anàlisi completa + quantificació (no light) — DONE
- [x] Analyze: PR exclusió quantificació configurable (no_quantification_patterns) — DONE
- [x] Analyze: menú contextual "Excloure/Incloure quantificació" per mostra — DONE
- [x] **⚡ Consolidació BP+COLUMN al Revisar (Pas 4)** — DONE (Fases 1-5)
  - Pla complet a `~/.claude/plans/synthetic-tinkering-hoare.md` (6 fases)
  - Fix: review_result.json s'escriu al completar generació → dashboard Revisar funcional
  - Secció BP al Revisar: taula mostres vinculades + dropdown override SEQ BP
  - BPDiscoveryWorker + _BPReloadWorker (cerca background + canvi dropdown)
  - find_bp_for_samples() a hpsec_consolidate.py (proximitat + nom)
  - export_sequence() accepta bp_resolved pre-resolt (evita doble cerca)
  - Dashboard: indicador ⟳ quan BP actualitzada post-revisió (is_bp_stale)
  - sequence_state.py: review_bp_name, review_bp_mtime, is_bp_stale property
  - [x] Cleanup: ConsolidatePanel eliminat (488 línies mortes) — Fase 6
  - Fitxers: review_summary_panel.py, hpsec_export.py, sequence_state.py, dashboard_panel.py, hpsec_consolidate.py
- [x] Config backend: config fingerprint (SHA-256 16 chars) per detectar obsolescència — DONE
- [x] Config backend: migració batman_max_sep → batman_max_sep_min — DONE
- [x] Config backend: REPROCESS_SECTIONS/FUTURE_SECTIONS/IMMEDIATE_SECTIONS constants — DONE
- [x] Wizard: Rename step 2 "Calibrar" → "Verificar" (TAB_NAMES + tab_names) — DONE
- [x] Wizard: Delay diagnostic tool at step 2 (shift indicator, slider, impact preview, reimport) — DONE
- [x] Wizard: SEQ_CAL detection at step 2 (CalibratePanel) + regression moved to step 3 (AnalyzePanel) — DONE
- [x] Wizard: Apply calibration at step 4 (Revisar) + retroactive requantification + SEQ list — DONE
- [x] Dashboard: diferenciar SEQ_CAL visualment (fons blau, [CAL] prefix, fases 2-4 = "—") — DONE (347c6b4)
- [x] **Dashboard: redisseny minimalista** — DONE
  - 15→9 columnes: eliminat #, Tipus, Mode, M, PC, PR (integrat a tooltip nom + col INJ compacta)
  - Capçaleres fases abreujades (I V A R) amb tooltip complet
  - `setFixedWidth` eliminat de tots els botons
  - Botó Reset eliminat: opcions al menú Processar + submenu context menu per SEQ individual
  - Stats compactes: `I:45 V:42 A:38 R:12 /120` amb tooltip detallat
  - Filtre Estat: opció "CAL" per filtrar SEQ_CAL
  - Header: `Seqüències (DATA_HPSEC)` amb path complet al tooltip
  - Sort default per data descendent (SEQs recents primer)

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

- **Import re-llegeix masterfile quan ja existeix manifest JSON**: FIXED — `import_from_manifest(load_data=False)`
  carrega metadades del manifest sense llegir MasterFile/CSV/DAD. `ensure_data_loaded()` completa les
  dades crues quan realment es necessiten (anàlisi, calibració, preview cromatograma). Speedup: 10s → 1ms.
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

> Last updated: 2026-02-26

### Unificació avisos — font única + barra wizard (2026-02-26)

**Problema**: 2 sistemes paral·lels d'avisos (`create_warning()` + `WARNING_DEFINITIONS` i
`create_anomaly()` + `ANOMALY_CATALOG`). El wizard amagava avisos darrere un botó status_indicator;
l'usuari no els veia si no clicava. ~200 línies de codi mort (funcions de migració, filtres, etc).

**Solució implementada:**

1. **`hpsec_warnings.py`**: `ANOMALY_CATALOG` com a font ÚNICA. Eliminats: `WARNING_DEFINITIONS`,
   `create_warning()`, `filter_warnings_by_level()`, `filter_warnings_by_stage()`, `has_blockers()`,
   `dismiss_warning()`, `warnings_summary()`, `migrate_legacy_warning()`, `migrate_warnings_list()`,
   `normalize_warnings()`, `create_warnings_from_timeout_info()`, `create_warnings_from_irregular_top_info()`.
   Afegits codis: `IMP_NO_DATA`, `IMP_MISSING_UIB`, `IMP_MISSING_DAD`, `IMP_ORPHAN_FILES`,
   `IMP_INCOMPLETE`, `CAL_NO_KHP`, `CAL_ALL_REPLICAS_INVALID`, `CAL_GLOBAL_ONLY`,
   `CAL_REPLICA_OUTLIER`, `ANA_NO_CALIBRATION`, `ANA_EMPTY_SAMPLES`.
   `get_max_warning_level()` mantingut com a alias backward-compat (suporta `severity` i `level` keys).

2. **Backends migrats**: `_generate_import_warnings()`, `_generate_calibration_warnings()`,
   `_generate_analysis_warnings()` usen `create_anomaly()` en lloc de `create_warning()`.
   `IMP_EMPTY_CSV` i `IMP_FALLBACK_DAD` eliminats (no accionables).

3. **`WarningBarWidget`** (`process_wizard_panel.py`): barra persistent entre header i tabs.
   Mostra avisos de TOTES les fases completades. Desplegada si ≤3, plegada si >3.
   Color fons adaptat a severitat màxima (vermell/groc/blau). Cada avis mostra icona + missatge + acció.

4. **`WarningReviewDialog` eliminat**: la barra ja mostra tot directament. El status_indicator
   marca avisos com a OK amb un clic (sense diàleg). `WarningSkipDialog` mantingut (per "Següent" amb avisos).

5. **`_update_validation()` simplificada** (`calibrate_panel/panel.py`): ja no recolleix anomalies
   duplicades de `khp_data_direct/uib` — usa `warnings_structured` directament del backend.

6. **`_get_warning_level()` simplificat**: sempre calcula des de `warnings_structured` via
   `get_max_warning_level()`, eliminats fallbacks a `warning_level` key i `warnings` strings.

7. **`_on_*_completed()` actualitzats**: calculen `warning_level` des de `warnings_structured`
   (no del backend `data.get('warning_level')`). Criden `_update_warning_bar()` al final.

**Fitxers modificats**: hpsec_warnings.py, hpsec_import.py, hpsec_calibrate.py, hpsec_analyze.py,
gui/widgets/process_wizard_panel.py, gui/widgets/calibrate_panel/panel.py

### Calibracions independents per senyal/sensibilitat — v3.0 (2026-02-26)

**Problema**: El sistema de calibració tractava Direct i UIB dins d'una sola entrada,
sense distingir sensibilitat UIB (700/1000 ppb). Dades UIB a sensibilitat 700 NO es
poden barrejar amb dades a 1000 per calibrar.

**Solució implementada — Calibration_Reference.json v3.0:**

1. **Estructura nova**: cada entrada a `calibrations[]` cobreix UN sol àmbit (`signal_scope`
   + `uib_sensitivity`). `rf_mass_cal` i `intercept` planers: `{"column": X, "bp": Y}`
   (el senyal ja va definit per `signal_scope`).

2. **`active_calibration_ids`**: dict `{"direct": "CAL_...", "uib": "CAL_...", "uib_700": "..."}`
   — una calibració activa per àmbit. Substitueix l'antic `active_calibration_id` únic.

3. **Migració automàtica v2→v3**: `load_calibration_reference()` detecta v2.0, divideix cada
   entrada antiga en DIRECT + UIB, genera `active_calibration_ids`, guarda i re-llegeix.
   4 entrades antigues → 8 noves. Verificat: RF Direct/COLUMN=752.9, RF UIB/COLUMN=628.

4. **Getters actualitzats**: `get_active_global_calibration(signal, sensitivity)`,
   `get_calibration_for_date(date, signal, sensitivity)`, `get_rf_mass_cal(..., sensitivity)`,
   `get_calibration_intercept(..., sensitivity)`. Helpers `_extract_rf_from_cal()` i
   `_extract_intercept_from_cal()` suporten format planer i nested.

5. **`add_calibration()`**: nous params `signal_scope`, `uib_sensitivity`. Tanca NOMÉS
   calibracions del MATEIX àmbit. ID amb suffix: `CAL_20260226_143000_DIRECT`.

6. **`compute_calibration_fingerprint()`**: si `calibration=None`, hasheja TOTS els
   `active_calibration_ids` (no un sol).

7. **`quantify_sample()` a hpsec_analyze.py**: intercept UIB independent (`intercept_uib`)
   en lloc de reutilitzar l'intercept Direct. `uib_sensitivity` llegit de `sample_result`.

8. **GlobalCalibrationPanel**: vista resum (showEvent) amb taula paràmetres actius,
   scatter regressió amb banda predicció 95%, taula historial calibracions, botó PDF.
   Visible quan no hi ha SEQ_CAL carregada (substitueix "Selecciona una SEQ_CAL").

9. **Cromatogrames KHP PNG**: `save_khp_chromatogram_plot()` i `save_all_khp_chromatograms()`
   a `hpsec_reports.py`. Hook a `calibrate_from_import()` guarda PNGs a
   `SEQ/CHECK/data/khp_plots/`. DOC + baseline + límits integració + àrea ombrejada + 254nm.

10. **PDF calibració amb cromatogrames**: pàgines extra (A4 landscape, GridSpec 3x2)
    amb PNGs dels cromatogrames. `_find_khp_chromatogram_pngs()` busca a
    `regression_data.chromatogram_plots_dir` o `source.seq_references`.

**Fitxers modificats**: hpsec_calibrate.py, hpsec_analyze.py, hpsec_reports.py,
gui/widgets/global_calibration_panel.py, gui/widgets/seq_cal_regression_widget.py,
gui/widgets/history_panel.py

**Call sites fixats** (passaven dict com a primer arg posicional):
- `global_calibration_panel.py` L2237-2240: `get_rf_mass_cal(new_cal, signal=...)` → `get_rf_mass_cal(signal=...)`
- `history_panel.py` L1235-1236: idem

### Unificació sistema d'avisos (2026-02-24)

**Problema**: 3 sistemes paral·lels d'avisos que no s'integren:
1. ANOMALY_CATALOG (18 codis) — usat per analyze, ben estructurat
2. quality_issues (strings lliures) — usat per calibrate, no estructurat
3. WARNING_DEFINITIONS (56 codis) — bridge fràgil amb parsing strings

**Símptomes**: Dashboard "Verificar" sempre buit (hardcoded []), quality_score numèric inintel·ligible,
sense context de mostra ni guia d'acció.

**Solució implementada — ANOMALY_CATALOG com a font única:**

1. **hpsec_warnings.py**: +10 codis KHP (KHP_IRREGULAR_TOP, KHP_MULTI_PEAK, KHP_TIMEOUT_PEAK,
   KHP_SNR_LOW, KHP_RSD_HIGH, KHP_FWHM_HIGH, KHP_ASYMMETRY, KHP_CR_LOW, KHP_BASELINE_DRIFT,
   KHP_NO_DAD). Camp `action` afegit a les 26 entrades. `create_anomaly()` retorna `action`.

2. **hpsec_calibrate.py**: `analizar_khp_data()` genera `calibration_anomalies` amb `create_anomaly()`
   en lloc de strings quality_issues. quality_score derivat automàticament. `calibration_anomalies`
   propagat al return dict, a l'agregació de rèpliques, i a `register_calibration()`.
   `_generate_calibration_warnings()` simplificada: recull anomalies ja estructurades.

3. **sequence_state.py**: Nous camps `calibrate_warnings` i `review_warnings`. `_extract_metadata()`
   extreu avisos blocker/warning de `calibration_anomalies`. `calibrate_state` ara retorna 'warning'
   si KHP local té anomalies (era sempre 'ok').

4. **dashboard_panel.py**: `phases_data` alimenta Verificar i Revisar (era hardcoded []). Tooltip
   prioritza avisos concrets sobre "KHP sibling" genèric. Fallback quality_issues per JSONs antics.

5. **calibrate_panel/panel.py**: Col 15-16 substituïdes: score numèric → badge icona+color
   (✔/ℹ/⚠/✘) amb tooltip que inclou label + acció del catàleg. Fallback per dades sense
   calibration_anomalies. `_update_validation()` prioritza anomalies sobre quality_issues strings.

6. **analyze_panel/panel.py**: `_classify_sample_status()` tooltips enriquits amb `action` del catàleg
   (format: "CRÍTIC: label\n   → acció recomanada").

**WARNING_DEFINITIONS marcat com deprecated** (no eliminat per backward compat JSONs antics).

### Config panel simplificat + contrasenya (2026-02-24)

**Eliminats 22 paràmetres del GUI** (segueixen al JSON, però no editables per l'usuari):
- Detecció d'Anomalies (8 params): interns algorisme
- Càlcul Baseline (6 params): 4 interns + method="mode" + min_noise
- Cromatograma (5 params): max_duration duplicat, smoothing calibrats
- Calibració (3 params): quality_max, min_cals_average, use_historical_fallback

**Contrasenya fixa "LEQUIA"** a `_save_config()` i `_reset_defaults()`.

### Optimització memòria (2026-02-23)

**Problema**: La Suite carregava ~115 MB per seqüència quan en necessitava ~3 MB.
Causa principal: Export3D DAD (101 wavelengths quan en calen 6), DataFrame 2-TOC
retingut en memòria, i fitxers de calibració rellegits del disc repetidament.

**4 blocs implementats (commit f7d03d4):**

1. **DAD filtrat a 6λ durant importació (~95 MB estalvi/SEQ)**:
   - `llegir_dad_export3d(path, wavelengths_to_keep=None)`: filtra columnes just després de llegir CSV
   - 6 call sites actualitzats (find_data_for_injection, import_from_manifest ×3, ensure_data_loaded, import_panel GUI)
   - Wavelengths de `hpsec_config.json → wavelengths.selected` (220, 254, 272, 290, 362)
   - **HCI no afectat**: `compute_hci()` a `hpsec_humic.py` rellegeix Export3D original del disc
     via `dad_export3d_path` (path propagat a `_flatten_samples_for_processing`), no usa el DataFrame en memòria

2. **Alliberar master_data["toc"] (~2 MB estalvi/SEQ)**:
   - `master_data["toc"] = None` al final de `import_sequence()` i `ensure_data_loaded()`
   - Si es necessita de nou (reimportació), `ensure_data_loaded()` rellegeix MasterFile des del disc

3. **Cache calibració amb verificació mtime (estalvi I/O)**:
   - `load_calibration_reference()`: cache `_cal_ref_cache` + `_cal_ref_mtime` (5x speedup)
   - `load_khp_history()`: cache `_khp_cache` + `_khp_mtime` + `_khp_cache_path` (94x speedup)
   - `load_local_calibrations()`: cache `_local_cal_cache` + `_local_cal_mtime` + `_local_cal_path` (31x speedup, 10 call sites)
   - Invalidació automàtica als corresponents `save_*()` functions
   - ~25 crides/sessió passen de lectura disc a lectura memòria

4. **Eliminar "df" redundant de rep_data["direct"] (~12 MB estalvi/SEQ)**:
   - `"df": df_doc` eliminat de 3 llocs (find_data_for_injection, import_from_manifest, ensure_data_loaded)
   - Verificat: cap codi extern accedeix a `rep_data["direct"]["df"]`, només `"t"` i `"y"`

**Safety guards mantinguts** a `hpsec_analyze.py`:
- `_flatten_samples_for_processing` L2254: `len(columns) > 8` → no-op (DAD ja filtrat)
- `analyze_sample` L1617: idem

### UIB Downsample + Saturació (2026-02-23)

**Problema**: UIB CSV té dt=0.005 min (14k punts) vs DOC Direct dt=0.067 min (1.1k punts).
Amb 12.6x més punts, el Savitzky-Golay (finestra 131 pts vs 11 pts) i les derivades es
comporten diferent → límits d'integració i àrees no comparables entre UIB i Direct.
A més, el detector UIB (Sievers M9e) satura a la sensibilitat configurada (700/1000 ppb),
retallant pics d'alta concentració.

**Solució implementada:**

1. **`hpsec_core.py`**: Nova funció `downsample_to_cadence(t, y, target_dt)`.
   - Bin-average: bins uniformes de mida target_dt, mitjana dels punts per bin.
   - Preserva àrea integrada i forma del pic.
   - Auto-detect: si dt_median >= target_dt * 0.8, retorna dades originals.
   - Constant `DOC_TARGET_DT_MIN = 0.0667` (4 segons, cadència TOC).

2. **`hpsec_import.py`**: Downsample aplicat a 3 punts d'entrada UIB:
   - L2708: mostres regulars (`find_data_for_injection`)
   - L4314: KHP via manifest (`import_from_manifest`)
   - L4720: KHP via `ensure_data_loaded`

3. **`gui/widgets/import_panel/panel.py`**: 4t punt d'entrada UIB (reassignació manual).

4. **`hpsec_warnings.py`**: Nova anomalia `UIB_SATURATED` (BLOCKER, invalidates=True).
   - Icon: "SAT", description: "Senyal UIB saturat"

5. **Detecció saturació per forma del pic** (refactored 2026-02-25):
   - **`hpsec_core.py`**: `detect_peak_clipping(t, y)` — detecta retall/clipping per
     **plateau/FWHM ratio**. Gaussiana normal ≈ 0.17, threshold > 0.40 = saturat.
     Inclou estimació automàtica de baseline (median bottom 20%).
     **Independent de qualsevol paràmetre de sensibilitat** — basat en forma intrínseca.
   - **`hpsec_calibrate.py`**: `analizar_khp_data()` crida `detect_peak_clipping` quan
     `doc_source == "uib"`. Guard: Direct MAI entra al codi de saturació.
     Enrichment Direct←UIB: `uib_saturated` propagat de l'entrada UIB (L5020).
   - **`hpsec_analyze.py`**: `analyze_sample()` crida `detect_peak_clipping` per UIB.
   - **`_build_entries()`** (2 implementacions: `hpsec_calibrate.py` + `seq_cal_regression_widget.py`):
     llegeixen `uib_saturated` del backend (ja calculat), guard `signal_name == 'uib'`.
   - **Verificat amb 293_SEQ_CAL**: 5ppm ratio=0.65 → SAT, 2ppm ratio=0.17 → ok
     (y_max=828 > sens=700, però forma normal → correcte: sensibilitat és rang recomanat,
     no límit dur de retall).

6. **`gui/widgets/analyze_panel/panel.py`**: Auto-exclusió i UI:
   - Punts UIB saturats auto-exclosos de la regressió (`_seq_cal_excluded`)
   - Columna anomalies mostra "SAT" en vermell
   - Swap senyal Direct/UIB recalcula exclusions

**Verificació**: Downsample 15000→1125 pts (13.3x), dt 0.005→0.0675 min.

### Informe calibració PDF + regression_data al JSON (2026-02-23)

**Motivació**: L'informe PDF de calibració recalculava la regressió des de KHP_History.
L'usuari va demanar que TOTES les dades vinguessin del JSON, sense recalcular res.

**Canvis implementats:**

1. **`hpsec_calibrate.py`**: `add_calibration()` nou param `regression_data`.
   - `_sanitize_regression_data()`: converteix numpy→Python, genera `stats_per_concentration`
   - El JSON ara guarda: punts (ug_doc, area, residual, y_pred), RMS, stats per conc, model info

2. **`gui/widgets/review_summary_panel.py`**: Passa `regression_data` a `add_calibration()`.
   - Botó "📄 Generar Informe Calibració (PDF)" visible després d'aplicar
   - Scatter miniatura: 7×3 (era 6×2.5), amb recta vigent (taronja) i banda predicció 95%
   - Equació en monospace amb fons blau
   - Comptador retro: "X/Y SEQs seleccionades"
   - Comparació via `format_calibration_comparison_html()` compartit

3. **`gui/widgets/global_calibration_panel.py`**: Botó "📄 Generar Informe PDF" al panell global.

4. **`hpsec_reports.py`**: `generate_calibration_report()` — 5 pàgines PDF:
   - P1: Resum executiu (taula params, equació, stats per concentració)
   - P2: Scatter regressió + residuals (banda predicció 95%)
   - P3: Evolució temporal RF des de KHP_History.json
   - P4: QC Levey-Jennings (desviació % vs recta vigent)
   - P5: Historial calibracions
   - **TOT des de JSON** — NO recalcula regressió

5. **`gui/widgets/analyze_panel/_helpers.py`**: Helpers compartits:
   - `format_calibration_comparison_html()`: taula HTML amb capçalera blava, deltes colorats
   - `compute_prediction_band()`: interval predicció 95% via scipy t-distribution

6. **`gui/widgets/analyze_panel/panel.py`** (Step 3):
   - Comparació usa helper compartit (amb fila equació)
   - Scatter: banda predicció 95%, línies RMS als residuals, etiquetes ppm

**Calibracions antigues** (pre-regression_data) no tindran les pàgines de scatter al PDF.
Es mostra un missatge informatiu i es genera igualment les pàgines 1, 3, 4, 5.

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

### Redisseny Wizard — 4 Fases (2026-02-22)

**Implementat en worktree `claude/serene-williamson`, merged a main:**

**Fase 1: Renaming Calibrar → Verificar**
- `process_wizard_panel.py`: TAB_NAMES, tab_names, comments — totes les ocurrències

**Fase 2: Delay diagnostic tool (pas 2 Verificar)**
- `hpsec_delay.py`: backend Net delay (read, estimate impact, update MasterFile)
- `calibrate_panel/panel.py`: secció delay amb indicador shift (colors), slider ±15min,
  spinbox sincronitzat, preview "X files reassignades", botó "Aplicar i Reimportar"
- Cached timestamps per resposta instantània del slider

**Fase 3: Moure regressió SEQ_CAL del pas 2 al pas 3**
- `calibrate_panel/panel.py`: `_detect_seq_cal()` marca `is_seq_cal`, amaga UI normal,
  mostra `_seq_cal_info_group` + delay diagnostic. NO fa regressió (va al pas 3)
- `analyze_panel/panel.py`: secció completa regressió amb taula punts (checkboxes),
  scatter+residuals (matplotlib), RF/intercept/R²/RMS, comparació vigent, recalcular
- NO botó "Aplicar" (va al pas 4)

**Fase 4: Aplicar calibració al pas 4 (Revisar)**
- `review_summary_panel.py`: secció "APLICAR CALIBRACIÓ (SEQ_CAL)" amb:
  - Resum regressió, comparació HTML vigent vs nova, scatter miniatura
  - DateEdit valid_from, checkbox retroactiu, llista SEQs amb checkboxes
  - Botó "Aplicar com a Nova Calibració" → `add_calibration()` + `requantify_analysis_json()`
  - Dashboard refresh automàtic

### Wizard SEQ_CAL — Regressió al wizard (2026-02-21)

**Implementat flux complet per SEQ_CAL al wizard (versió original, ara refactored):**
- Detecció automàtica: `_detect_seq_cal()` (refactored from `_detect_and_run_seq_cal`)
- Regressió moguda de pas 2 a pas 3 (AnalyzePanel)
- Aplicació moguda de pas 2 a pas 4 (ReviewSummaryPanel)
- GlobalCalibrationPanel: consulta-only (sense aplicar/requantificar)

### Redisseny Wizard — Sessió 2026-02-22

**Fase 1: Rename Calibrar → Verificar (COMPLETAT)**
- `process_wizard_panel.py`: TAB_NAMES, tab_names dict
- `dashboard_panel.py`: STAGE_NAMES, columna headers, context menus
- `sequence_state.py`: Phase.CALIBRATE display
- `main_window.py`: comentari
- Commit: `0358c31`

**Fase 2: Delay diagnostic tool (COMPLETAT)**
- `hpsec_delay.py`: NOU — backend per gestió delay HPLC↔TOC
  - `read_current_delay(mf_path)`: llegeix 0-INFO B12
  - `estimate_delay_impact(mf_path, old_delay, new_delay)`: quantes files canvien
  - `update_masterfile_delay(mf_path, net_delay_min)`: actualitza + regenera 4-TOC_CALC + backup
- `gui/widgets/calibrate_panel/panel.py`: secció diagnòstic delay
  - `_build_delay_diagnostic_section()`: UI amb indicador shift, slider, spinbox, impacte, botó
  - `_update_delay_diagnostic(result)`: mostra per BP (sempre) o COLUMN (shift > 2 min)
  - `_on_delay_slider_changed/_on_delay_spin_changed`: sincronitzats bidireccional
  - `_update_delay_impact(new_delay)`: preview en temps real (quantes files canvien)
  - `_delay_apply_and_reimport()`: aplica delay → reimporta → re-verifica
  - Indicador qualitat: verd < 0.5 min, taronja 0.5-2 min, vermell > 2 min
  - Slider: ±15 min, pas 0.1 min
  - Integrat a _on_finished (normal + error path)

**Fase 3: Moure regressió SEQ_CAL al pas 3 — COMPLETAT** (merged)
**Fase 4: Aplicar calibració al pas 4 — COMPLETAT** (merged)

### Sessió 2026-02-22 (continuació) — UIB timeouts + fixes wizard

**Bug: Suite penjada al passar a Verificar (ensure_data_loaded al UI thread):**
- Símptoma: "al reimportar la 288 i passar a verificar cursor ocupat [...] (no respon)"
- Causa: `ensure_data_loaded()` bloquejava el thread principal (carregar MasterFile+CSV+DAD és lent)
- Fix: Mogut `ensure_data_loaded()` dins dels workers (threads):
  - `calibrate_panel/worker.py`: CalibrateWorker.run() crida ensure_data_loaded si data_deferred
  - `analyze_panel/worker.py`: AnalyzeWorker.run() crida ensure_data_loaded si data_deferred
  - `calibrate_panel/panel.py`: eliminat ensure_data_loaded del UI thread (L926-932)
  - `analyze_panel/panel.py`: eliminat ensure_data_loaded del UI thread (L433-440)

**Bug: Preload manifest camps incorrectes (process_wizard_panel.py):**
- `_preload_completed_stages()` creava `imported_data` amb:
  - `manifest.get("method")` → None (hauria de ser `manifest["sequence"]["method"]`)
  - `manifest.get("samples")` → **llista** no dict (crash a `ensure_data_loaded()` que espera `.items()`)
  - `manifest.get("masterfile_path")` → None (hauria de ser `manifest["master_file"]["path"]`)
- Fix: Reescrit per extreure de l'estructura anidada + convertir samples llista→dict
- Afecta a: totes les SEQs al carregar automàticament des del manifest (auto-load)

**UIB timeout estimation (NOU — hpsec_core.py + hpsec_analyze.py):**
- **Context**: El timeout del TOC (recàrrega xeringues Sievers M9e, ~74s cada ~77.2 min)
  afecta tant DOC Direct (gap temporal) com UIB (patró anòmal sense gap temporal).
  UIB mostra un pic espuri (~1.8 min durada, fins +75% sobre baseline) al mateix temps que el timeout Direct.
- **Problema**: No es pot detectar el timeout a UIB per gaps temporals (CSV continu).
  Cal estimar-lo des de DOC Direct o des del model predictiu.
- **Implementació 3 nivells**:
  1. `estimate_timeout_for_uib()` a `hpsec_core.py`: transfereix posicions timeout de DOC Direct a UIB,
     o usa model predictiu (`hpsec_planner.py`) amb T0 i sample_duration
  2. Integrat a `analyze_sample()` a `hpsec_analyze.py`: per cada mostra DUAL, estima UIB timeouts
  3. `_estimate_uib_timeouts_from_sequence()` a `hpsec_analyze.py`: post-processa tota la seqüència
     per extrapolar timeouts a injeccions sense DOC Direct via regressió lineal sobre patró observat
- **Verificació 288_SEQ**: drift consistent -1.4 min/inj (teòric -1.45 per COLUMN 78.65 min)
- **hpsec_planner.py**: mòdul existent (418 línies) amb model complet de predicció timeout, mai usat.
  Constants: TOC_CYCLE_MIN=77.2, TOC_TIMEOUT_SEC=74, SAMPLE_DURATION_CURRENT=78.65
- Zona d'anomalia UIB: t_timeout - 0.2 min a t_timeout + 1.8 min (pre/post marges)

**Investigació 288_SEQ Export3D:**
- Primera importació no reconeixia DAD → reimportació correcta
- Causa: manifest antic sense info DAD; reimportació regenera manifest amb Export3D detectat
- 33 fitxers Export3D correctament detectats al reimportar (dad_source=export3d)

### Canvis sessió 2026-02-21 (continuació)
- **GlobalCalibrationPanel refactor**: 2 vistes (CalibrationLineView + QCMonitorView)
  - `hpsec_calibrate.py`: `compute_calibration_fingerprint()`, `requantify_analysis_json()`
  - `hpsec_export.py`: `patch_excel_calibration()` (openpyxl cell-level patching)
  - `gui/widgets/global_calibration_panel.py`: reescriptura completa (~700 línies)
  - `gui/models/sequence_state.py`: `calibration_fingerprint`, `is_cal_stale`
  - `gui/main_window.py`: `calibration_updated` signal → dashboard refresh
  - `gui/widgets/dashboard_panel.py`: indicador ✔⟳ per SEQs amb calibració obsoleta
  - Commit: `2b43eb0`
- **Fix import re-read MasterFile**: `import_from_manifest(load_data=False)` per auto-load
  - `hpsec_import.py`: param `load_data`, funció `ensure_data_loaded()`
  - `gui/widgets/import_panel/worker.py`: `load_data` param
  - `gui/widgets/import_panel/panel.py`: `_auto_load_from_manifest()` amb `load_data=False`
  - `gui/widgets/analyze_panel/panel.py`: `ensure_data_loaded()` abans d'analitzar
  - `gui/widgets/calibrate_panel/panel.py`: `ensure_data_loaded()` en 3 punts
  - Speedup: 10s → 1ms (factor 10000x) per auto-load SEQs ja importades

### Canvis sessió 2026-02-21 (inici)
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
