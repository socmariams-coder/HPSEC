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
- [x] Calibration: flux renovació calibració global (UI panel + regression) — DONE (GlobalCalibrationPanel)
- [x] Calibration: auto-fit rf_mass_cal + intercept from KHP history (regression) — DONE (fit_calibration_from_history)
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
- [ ] Architecture refactor: unify detection functions in hpsec_core.py — PENDING
- [x] Integration: derivative-based peak boundaries (Agilent tangent projection) — DONE (find_peak_boundaries in hpsec_core.py)
- [x] Integration: re-process all KHP data with new derivative method — DONE (batch re-calibrate 137 SEQs, KHP_History regenerated with 96 entries)
- [x] Config panel: 3 tabs per impacte (Anàlisi/Seqüència/Sistema), tots params editables — DONE
- [x] Config panel: badges d'impacte (retroactiu/futur), diàleg de reprocessament — DONE
- [x] Config panel: TimeFractionsEditor, TimeoutZonesEditor, WavelengthSelector, PatternListEditor — DONE
- [x] Config backend: config fingerprint (SHA-256 16 chars) per detectar obsolescència — DONE
- [x] Config backend: migració batman_max_sep → batman_max_sep_min — DONE
- [x] Config backend: REPROCESS_SECTIONS/FUTURE_SECTIONS/IMMEDIATE_SECTIONS constants — DONE

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
- **Import auto-detecta columna volum erròniament en seqs BP**: `hpsec_import.py` (L1681-1690) busca
  una columna numèrica amb valors 50-1000 a l'índex 13 del masterfile. En seqs BP, aquesta columna
  pot contenir dades de pics (no volums) amb valors que passen el filtre (ex: 400). Resultat: BP
  registrat amb volum=400µL en lloc de 100µL → RF calculat x4 massa baix. Fix parcial aplicat a
  `register_calibration()` (correcció forçada BP→100µL). Fix definitiu pendent a hpsec_import.py:
  requerir capçalera explícita "Volume"/"Vol" o prioritzar 0-INFO sobre auto-detecció.
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
