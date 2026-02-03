# Calculations per Signal Type (BP vs COLUMN)

Data: 2026-02-03

## Taula de càlculs

| Calculation | DOC Direct | DOC UIB | DAD 254 | Other DAD |
|-------------|:----------:|:-------:|:-------:|:---------:|
| **BP Mode** |
| Baseline | ✓ windowed (5-10 min) | ✓ pròpia (5-10 min) | ✓ percentile | ✓ percentile |
| Peak detection | ✓ | ✓ | ✓ | - |
| Batman detection | ✓ | ✓ | - | - |
| FWHM/Symmetry | ✓ | ✓ | ✓ | - |
| SNR/LOD/LOQ | ✓ | ✓ | ✓ | ✓ |
| Area (total) | ✓ | ✓ | ✓ | ✓ |
| Timeout detection | ✓ (source) | ✓ (propagated) | - | - |
| **COLUMN Mode** |
| Baseline | ✓ windowed (0-10 min, sense peaks) | ✓ pròpia (0-10 min) | ✓ percentile | ✓ percentile |
| Peak detection + tmax | ✓ | ✓ | ✓ | ✓ |
| Batman detection | ✓ | ✓ | - | - |
| FWHM/Symmetry | ✓ | ✓ | - | - |
| SNR/LOD/LOQ | ✓ | ✓ | ✓ | ✓ |
| Fractions (BioP/HS/BB/SB/LMW) | ✓ | ✓ | ✓ | ✓ |
| Pearson Direct vs UIB | ✓ comparison | - | - | - |
| DOC/254 ratio | ✓ numerator | - | ✓ denominator | - |
| SB/HS ratio | ✓ | ✓ | - | - |
| Peaks in HS zone | - | - | ✓ (counted) | - |
| Timeout detection | ✓ (source) | ✓ (propagated) | - | - |

## Notes

### Baseline
- **DOC Direct/UIB**: Finestra temporal (windowed) per evitar zones amb peaks o timeouts
  - BP: 5-10 min (post-peak zone)
  - COLUMN: 0-10 min (pre-peak zone, verificant que no hi ha peaks)
- **UIB**: Baseline pròpia, NO propagada de Direct
- **DAD**: Baseline per percentil (més robust per senyals amb múltiples peaks)

### Symmetry
- Sempre calculada al **50% d'altura del pic** (estàndard cromatogràfic)

### SNR/LOD/LOQ
- Calculat per TOTES les senyals
- LOD = 3 × baseline_noise
- LOQ = 10 × baseline_noise

### Timeout
- Detectat només a DOC (intervals dt > 60s)
- Propagat a UIB per coherència temporal
- No aplica a DAD (adquisició independent)

### Fraccions temporals (només COLUMN)
- BioP: 0-18 min
- HS: 18-23 min
- BB: 23-30 min
- SB: 30-40 min
- LMW: 40-70 min
