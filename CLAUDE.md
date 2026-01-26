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

### TOC Timeout Parameters (Syringe Reload)
The TOC analyzer (Sievers M9e) reloads syringes periodically, causing data gaps:
- **Reload cycle**: 77.2 min (1140 measurements × ~4.06s)
- **Timeout duration**: 74 seconds (gap in data)
- **HPLC injection time**: 78.65 min (current)
- **Drift per sample**: 1.45 min (timeout position shifts backward)

**Affected zone duration** (analyzed from 331 samples):
- PRE-timeout anomaly: ~0.5 min (syringe emptying, unstable measurements)
- Timeout gap: ~1.2 min (74s, no data)
- POST-timeout anomaly: ~1.0 min (stabilization after reload)
- **TOTAL AFFECTED: ~2.5-3 min**

**Impact by zone** (Pearson Direct vs UIB):
| Zone | Pearson | Severity |
|------|---------|----------|
| HS (18-23 min) | 0.983 | CRITICAL |
| BioP (0-18 min) | 0.991 | WARNING |
| BB (23-30 min) | 0.996 | WARNING |
| SB (30-40 min) | 0.998 | WARNING |
| LMW (40-70 min) | 0.999 | INFO |

**Optimization recommendation**: Change HPLC method duration to 77.2 min + wait 3-5 min after flush → 100% timeouts in post-run zone (no impact).

### Calibration Factors (Historical)
- Directe samples: 0.82
- UIB SEQ < 275: 0.84
- UIB SEQ ≥ 275: 0.87

### UIB Sensitivity Settings
- **700 ppb** (SEQ 269-274): Lower noise, better LOD
- **1000 ppb** (SEQ ≥ 275): Higher dynamic range, more noise

**LOD comparison** (analyzed from 156 DUAL samples):
| System | 700 ppb | 1000 ppb |
|--------|---------|----------|
| UIB | 1.85 mAU | 3.47 mAU |
| Direct | 1.24 mAU | 3.47 mAU |

**Recommendation**: Use Direct as primary signal (better SNR). If only UIB available, 700 ppb is preferred for low concentration samples.

### Injection Volumes
- BP mode: 100 µL
- Column mode: 400 µL (except SEQ 256-274: 100 µL)

## Documentation References

- **RESUM_APROXIMACIONS.md** - Analysis of amorphous detection approaches (STOP, angles, derivatives)
- **MILLORA_DURACIO_SEGMENTS.md** - Segment duration validation implementation and test results
- **TODO_SUITE.txt** - Development roadmap, known anomalies (e.g., KHP2_284_BP height issue)

## Architecture Refactoring (Planned)

See **PLAN_REFACTOR_DETECCIO.md** for complete details.

### Current Issue: Code Duplication
Detection functions are duplicated across modules with inconsistent implementations:

| Function | hpsec_core.py | hpsec_consolidate.py | hpsec_calibrate.py |
|----------|---------------|----------------------|--------------------|
| `detect_batman()` | ✅ Robust | ❌ | ⚠️ Simplified |
| `detect_timeout()` | ❌ | ✅ Robust (dt-based) | ⚠️ Simplified (CV) |
| `detect_main_peak()` | ❌ | ❌ | ✅ |

### Target Architecture
All detection functions should be in **hpsec_core.py** (Single Source of Truth):
```
hpsec_core.py
├── detect_timeout(t_min)          # Best method: dt intervals > 60s
├── detect_batman(t, y)            # Strict: peak-valley-peak pattern
├── detect_peak_anomaly(t, y)      # Hybrid: batman + smoothness
├── detect_main_peak(t, y)         # Find primary peak
├── detect_all_peaks(t, y)         # Find all significant peaks
└── integrate_chromatogram(mode)   # mode='full' | 'main_peak'
```

Other modules import from hpsec_core:
- **hpsec_calibrate.py**: Adds KHP-specific validation (multi-peak, outlier stats)
- **hpsec_consolidate.py**: Uses detect_timeout() for TOC gap detection
- **DOCtor_BP.py / DOCtor_C.py**: Uses all detection functions

### KHP Integration Mode
- **Column KHP**: `integrate_chromatogram(mode='main_peak')` - Only integrate the main peak
- **BP KHP / Samples**: `integrate_chromatogram(mode='full')` - Integrate entire chromatogram

## Development Notes

- Documentation and comments are in Catalan/English mix
- Output files are generated in CHECK/ subdirectories
- The codebase uses percentile-based baselines and multi-filter anomaly detection for robustness
- No automated test suite exists; validation is done via DOCtor_validation.py
