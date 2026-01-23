# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HPSEC Suite is a Python-based analytical chemistry system for chromatogram anomaly detection in High-Performance Size Exclusion Chromatography (HPSEC). It analyzes organic matter composition using DAD-DOC data for environmental/water chemistry applications.

## Commands

**Run applications:**
```bash
python HPSEC_Suite.py          # Main unified GUI application
python DOCtor_BP.py            # Bypass sample analysis
python DOCtor_C.py             # Column sample analysis
python DOCtor_Diag.py          # Manual peak labeling/validation GUI
python batch_process.py        # Batch process multiple sequences
python DOCtor_validation.py    # Validate detection thresholds
```

**Build executable:**
```bash
python build_exe.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture

### Core Modules (Shared Libraries)
- **hpsec_core.py** - Shared mathematics: bi-Gaussian fitting, Batman detection, SNR calculation, peak repair algorithms. Changes here affect both BP and Column modules.
- **hpsec_utils.py** - Utilities: baseline statistics, peak detection, GUI dialogs

### Main Applications
- **HPSEC_Suite.py** - Unified GUI entry point (consolidate → calibrate → process → export)
- **DOCtor_BP.py** - Bypass sample anomaly detection (v1.6)
- **DOCtor_C.py** - Column sample anomaly detection (v1.0) - active development
- **DOCtor_Diag.py** - Manual validation and labeling GUI
- **DOCtor_GUI.py** - Interactive peak labeling tool

### Data Flow Pipeline
```
SEQ Folder (*.D files)
    ↓
CONSOLIDATE: Read .D files → Create Excel consolidates
    ↓
CALIBRATE: Find KHP standards → Calculate correction factor
    ↓
PROCESS: Detect anomalies → Select best replicates (highest R²/SNR)
    ↓
EXPORT: Generate PDF reports, Excel results, CSV metrics
```

## Key Technical Concepts

### Anomaly Detection Types
- **BATMAN**: Peak valleys at summit (detector artifacts)
- **AMORPHOUS**: Irregular peaks with angles, steps, or ramping
- **STOP/TIMEOUT**: Flat plateaus indicating detector pauses (CV < 0.1%, duration 0.8-3.5 min)
- **EARS**: Secondary peaks within 0.5 min of main peak (>10% height)
- **PEARSON**: Low correlation between replicates (≤0.997)
- **ASYMMETRY**: Out-of-range sigma ratios (valid range: 0.33-3.0)

### R² Thresholds (bi-Gaussian fit quality)
- R² ≥ 0.987 → VALID (green)
- 0.980 ≤ R² < 0.987 → CHECK (orange)
- R² < 0.980 → INVALID (red)

### Key Configuration Constants
Most detection thresholds are configurable constants at the top of each module:
- `NOISE_THRESHOLD = 20.0 mAU`
- `PEARSON_INSPECT = 0.995`
- `BATMAN_MAX_SEP_MIN = 0.5`
- `REPAIR_FACTOR = 0.85`
- `SNR_THRESHOLD = 10.0`
- Humic zone (HS region): 18-23 min

### Calibration Factors (Historical)
- Directe samples: 0.82
- UIB SEQ < 275: 0.84
- UIB SEQ ≥ 275: 0.87

### Injection Volumes
- BP mode: 100 µL
- Column mode: 400 µL (except SEQ 256-274: 100 µL)

## Documentation References

- **RESUM_APROXIMACIONS.md** - Analysis of amorphous detection approaches (STOP, angles, derivatives)
- **MILLORA_DURACIO_SEGMENTS.md** - Segment duration validation implementation and test results
- **TODO_SUITE.txt** - Development roadmap, known anomalies (e.g., KHP2_284_BP height issue)

## Development Notes

- Documentation and comments are in Catalan/English mix
- Output files are generated in CHECK/ subdirectories
- The codebase uses percentile-based baselines and multi-filter anomaly detection for robustness
- No automated test suite exists; validation is done via DOCtor_validation.py
