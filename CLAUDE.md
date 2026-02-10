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
- [x] Analyze panel: taula unificada DOC+DAD (12 cols: SNR, SNR_254, R²_DOC, R²_DAD) — DONE
- [ ] Analyze panel: dropdowns rèplica independents DOC vs DAD — PENDING (backend suporta, GUI no)
- [ ] Analyze panel: mostrar bigaussian (R², asym, quality) per BP — PENDING
- [ ] Analyze panel: mostrar timeouts amb icones/tooltip — PENDING
- [ ] Analyze backend: detecció deriva baseline DAD per replica selection — PENDING (TODO a hpsec_analyze.py L1210)
- [x] Calibration: flux renovació calibració global (UI panel + regression) — DONE (GlobalCalibrationPanel)
- [x] Calibration: auto-fit rf_mass_cal + intercept from KHP history (regression) — DONE (fit_calibration_from_history)
- [ ] Calibration: UI to edit intercept values directly from panel — PENDING
- [ ] Export: PDF batch reports — PARTIAL (template exists)
- [ ] Architecture refactor: unify detection functions in hpsec_core.py — PENDING

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
