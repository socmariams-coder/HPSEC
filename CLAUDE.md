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
- [x] Calibration: QC history + KHP validation — DONE
- [x] Calibration: plot recta with tolerance bands — DONE
- [x] Process wizard: batch anomaly detection — DONE
- [x] Analyze panel: chromatogram view + anomaly flags — DONE
- [x] Dashboard panel — DONE
- [ ] Calibration: UI to edit intercept values from panel — PENDING
- [ ] Calibration: auto-fit intercept from KHP data (regression) — PENDING
- [ ] Export: PDF batch reports — PARTIAL (template exists)
- [ ] Architecture refactor: unify detection functions in hpsec_core.py — PENDING

## Instructions for Claude

1. **Read this file at session start** to understand current feature state.
2. **After implementing a feature**, update the Feature Status section above.
3. **Never mark a feature DONE** if the code is a placeholder or partial stub.
4. **When user asks about a feature**, check this list first to give accurate status.
5. Comments and variable names: Catalan/English mix is normal.
6. See `PROVES/CLAUDE.md` for detailed technical reference (thresholds, TOC params, etc).
