# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.
It is read automatically at the start of every session.

## Project

HPSEC Suite — analytical chemistry system for HPSEC chromatogram processing (DAD-DOC).
Python + PySide6 GUI. Test suite: `test_robustesa_audit.py` (runner propi, no pytest).

## Key commands

```bash
python hpsec_suite_qt.py           # Main GUI (wizard 5 passos + panells globals)
python test_robustesa_audit.py     # Tests de robustesa (28 checks)
pip install -r requirements.txt
```

## Architecture summary

- **hpsec_version.py** — centralized version constant (`SUITE_VERSION`, `SUITE_FULL`)
- **hpsec_core.py** — shared math (bi-Gaussian, irregular top, SNR, peak repair, saturation detection)
- **hpsec_calibrate.py** — calibration engine (rf_mass_cal, intercept, QC)
- **hpsec_consolidate.py** — .D file consolidation to Excel
- **gui/** — PySide6 GUI panels (dashboard + wizard 5 passos: Importar, Verificar,
  Analitzar, Quantificar, Exportar; tabs globals: Mostres, QC/KHP, Calibració Global,
  Manteniment, Configuració)
- **archive/** — scripts de diagnòstic d'un sol ús i docs obsolets (fora de git,
  només disc local)
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
- [x] Calibration: sistema de reparació de pics UNIFICAT — el diàleg d'Analitzar (JaggedPeakRepairDialog) a tot arreu (taula mètriques, detall KHP via repair_requested, Cal.Global); overrides existents carregats a les cards, sync només-modificats (no esborra reparacions en tancar), Δ% real via recompute_area_with_repair, navegació ◀▶ entre grups/punts — DONE (2026-07-15, verificat fum GUI offscreen)
- [x] Calibration: UIB intercept independent a quantify_sample — DONE
- [x] **Calibration: règims instrumentals (blocs de comparabilitat)** — DONE (2026-07-17)
  - `regimes[]` + `regime_pending_events[]` a Calibration_Reference.json (font única)
  - Esdeveniment (canvi columna/detector) = CANDIDAT pendent, no parteix bloc; el
    primer KHP posterior el confirma (règim nou, frontera a la data de l'esdeveniment)
    o el descarta (equivalent → el bloc continua)
  - `check_calibration_equivalence` (llindars QC 15/25%), `resolve_regime_on_calibration`,
    `register_calibration_validation`, `filter_history_by_regime` (hpsec_calibrate.py)
  - Aplicar SEQ_CAL (Cal.Global): equivalent→oferta VALIDACIÓ (no obre règim),
    break→règim nou abans d'add_calibration (regime_id estampat a la cal nova)
  - Manteniment: categories "Canvi columna"/"Canvi detector/guany" + checkbox candidat
  - Fronteres SEMPRE amb data d'ADQUISICIÓ (get_seq_acquisition_date)
- [ ] Calibration: cablejar `filter_history_by_regime()` als consumidors de l'historial
  (comparatives "N més recents", fit_calibration_from_history, Levey-Jennings) — PENDING
- [x] Storage: analysis_result.json compacte — DONE (2026-07-17)
  - Regla: cap writer escriu el dict cru; sempre `strip_flat_sample_arrays()` + indent=None
  - La llista plana `samples` no porta arrays a disc (única còpia: samples_grouped)
  - 3 writers unificats (save_analysis_result, QuantifyPanel._persist_result,
    requantify_analysis_json); `migra_compacta_analysis.py` per fitxers antics (52→20 MB)
- [x] Refactor: fórmula de quantificació única — DONE (2026-07-17)
  - `hpsec_core.area_to_ppm()` i `area_to_rf_mass()` són l'ÚNICA implementació
  - Abans: forward copiada a 5 llocs + inversa a 7. NO reescriure la fórmula enlloc més
  - Substituïts 11 call sites (analyze, calibrate, composition_dialog, 2 panells)
  - Verificat: 16/16 ppm reals reproduïts exacte (sense canvi numèric)
- [x] Cleanup: codi mort eliminat (~500 línies) — DONE (2026-07-17)
  - `validate_khp_for_alignment` (calibrate, 272 l., 0 usos) + `_realign_bp_by_dad254`
    (import, 229 l., 0 usos) + import mort quantify_with_global_calibration a analyze
- [ ] Export: número de SEQ als noms de fitxer de RESULTATS ({sample}_{SEQ}_HPSEC_C.xlsx,
  {SEQ}_SUMMARY.xlsx) per evitar col·lisions en carpetes compartides — PENDING
- [ ] Export FAIR: regime_id + fronteres de règim a fulla ID, metadata.json i
  datapackage.json — PENDING
- [x] GlobalCalibrationPanel: vista resum sense SEQ_CAL (taula params, scatter, historial) — DONE
- [x] GlobalCalibrationPanel: SEQ_CAL auto-flow (Direct→UIB→resum) — DONE
- [x] Export: KHP chromatogram PNGs a CHECK/data/khp_plots/ — DONE (save_all_khp_chromatograms)
- [x] Export: PDF calibration report amb pàgines cromatogrames KHP — DONE
- [x] Export: PDF dual-signal calibration report (Direct+UIB combinat) — DONE
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
- [x] Export: noms fitxer sense rèplica — `{sample}_HPSEC_{C|B}.xlsx` amb col·lisions — DONE
- [x] Export: CSV cromatogrames + resultats (FAIR format obert, separador configurable) — DONE
- [x] Export: SUMMARY.csv amb metadades i fingerprints — DONE
- [x] Export: ID sheet reorganitzada 10 seccions + nous camps traçabilitat — DONE
- [x] Export: UI opcions CSV (checkboxes + separador combo) a ExportPanel — DONE
- [x] **Export FAIR v2: RAW/PROCESSED subcarpetes + DAD full λ + ZIP** — DONE
  - RAW/: DOC_Direct_RAW, DOC_UIB_RAW, DAD 101λ downsampled dt=0.04 min
  - PROCESSED/: DOC_net (shift+baseline+smoothing+repair), 6λ DAD, fraccions, ppm
  - BP DAD RAW: 1 fila a t_max amb totes les λ
  - BP+COLUMN junts a les mateixes carpetes RAW/ i PROCESSED/
  - ZIP packaging (checkbox a ExportPanel)
  - metadata.json FAIR amb fingerprints + llista mostres
  - `dad_export3d_path` propagat a `summarize_sample()` (hpsec_analyze.py)
- [x] Wizard: Step 4 "Exportar" (ExportPanel with BP consolidation, generate button, FAIR) — DONE
- [x] Wizard: "Generar Resultats" exports to SEQ/RESULTATS/ (Excels) + SEQ/CHECK/ (SUMMARY) — DONE
- [x] Export panel redissenyat — auto-generació + export addicional — DONE
  - Part 1: results_frame (auto-generació Excels+SUMMARY a SEQ/RESULTATS/ + SEQ/CHECK/)
  - Part 2: consolidació BP (sense canvis)
  - Part 3: "Export addicional" — checkboxes contingut + destí (carpeta o ZIP)
  - Checkboxes: Excels, SUMMARY, RAW CSVs, PROCESSED CSVs, CSV SUMMARY, PDF, metadata.json
  - Destí: carpeta (SharePoint/OneDrive/local) o ZIP (amb temp dir)
  - GenerateWorker: SUMMARY a custom_output_dir quan set
  - Decimals: sempre `.` (punt), dates ISO 8601, sense separador milers
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
- [x] **Dashboard: múltiples carpetes de dades** — DONE
  - `hpsec_config.py`: `data_folders` (llista) substitueix `data_folder` (string)
  - Migració automàtica `data_folder` → `data_folders` a `_migrate_config()`
  - `get_data_folders()` retorna llista, `get_data_folder()` retorna la primera
  - `get_registry_path()` suporta `registry_folder` explícit o derivat de 1a carpeta
  - `SequenceState`: nous camps `source_folder`, `source_path`
  - `get_all_sequences()` accepta llista de carpetes, escaneja totes
  - Dashboard: combo filtre "Carpeta" (visible si >1 carpeta)
  - Config Panel: `QListWidget` per carpetes amb validació anti-duplicats SEQ
  - `global_calibration_panel.py`: cerca SEQ_CAL a TOTES les carpetes
- [x] **Versió centralitzada — hpsec_version.py** — DONE
  - `hpsec_version.py`: `SUITE_VERSION = "2.1.0"`, `SUITE_NAME`, `SUITE_FULL`
  - Tots els JSON outputs (8 fitxers) usen `suite_version` + `*_module` consistentment
  - PDF reports: header "HPSEC Suite v2.1.0", footer amb versió sempre present
  - Excel ID sheet: `Suite_Version` únic (sense duplicats)
  - metadata.json, import_manifest, analysis_result, calibration_result, review_result: uniformitzats
- [x] **Architecture: fusió Dashboard+Processar → tab únic "Processar"** — DONE
  - QStackedWidget: page 0 = DashboardPanel (eager), page 1 = ProcessWizardPanel (lazy)
  - 7 tabs (era 8): Processar, Exportar, Mostres, Històric, Cal.Global, Manteniment, Config
  - `show_dashboard()` / `_show_wizard()` per navegació stacked
  - Tots els índexs de tab actualitzats (Exportar 2→1, Cal.Global 5→4, etc.)
- [ ] **Anàlisi espectral DAD avançada** — PENDING (FUTUR)
  - DAD actual: adquisició fins 400 nm
  - E₂/E₃ (A₂₅₄/A₃₆₅): factible. S₂₇₅₋₂₉₅ i SR: factibles. E₄/E₆: NO (fora rang)
  - HCI ja calculat (hpsec_humic.py) però amagat a la UI — fer visible
  - Idea: tabs separats "Analitzar DOC" i "Analitzar DAD"
  - Correlació DOC↔DAD com a exploració futura
- [ ] **Tab Exportar: redisseny com a empaquetador standalone** — PENDING (FUTUR)
  - Actual: duplicat del wizard pas 4, depèn de `processed_data` (no funciona sol)
  - Objectiu: seleccionar SEQ ja processada → empaquetar fitxers existents → carpeta/ZIP
  - Export "net": mostres + KHP, sense blancs/controls/NaOH
  - COLUMN+BP combinats per mostra (via `find_bp_for_samples`)
  - ZIP: nom automàtic + triar carpeta (no nom fitxer)
  - Pendent definir: multi-SEQ, agrupació per campanya, relació amb inventari mostres

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

## TODO (millores menors)

- [ ] **CalibratePanel.showEvent() guard**: afegir `_initialized` flag per evitar re-executar
  `_check_existing_calibration()` cada cop que es mostra el tab (funciona, pero redundant)
- [ ] **HistoryPanel.showEvent() guard**: afegir flag per evitar recarregar tot l'historial
  cada cop que es canvia de tab (dades globals, no per-SEQ)
- [ ] **Repair review**: revisar sistema de reparació (batman/irregular_top) amb 288_SEQ com a cas de test
- [ ] **UIB saturació vs timeout**: verificar amb la tècnica si clipping a 5ppm és real
  (veure `archive/diagnostic_scripts/_verify_uib_saturation.py`)

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

El diari de sessions viu a `WORKING_NOTES.md` (mateix format que abans:
una secció per tema/sessió, la més recent a dalt). Actualitzar-lo cada
vegada que es toca un tema nou rellevant, com fins ara.
