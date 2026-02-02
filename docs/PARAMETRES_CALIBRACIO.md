# Paràmetres de Calibració KHP

## Senyals Disponibles

| Codi | Senyal | Descripció |
|------|--------|------------|
| **D** | Direct | DOC mesurat directament |
| **U** | UIB | DOC mesurat via UIB (pot tenir sensibilitat diferent) |
| **254** | DAD 254nm | Absorbància a 254nm (detecta aromàtics) |

## Abreviacions de Paràmetres

### Àrea i Response Factor

| Abrev. | Nom Complet | Unitats | Descripció |
|--------|-------------|---------|------------|
| **Area** | Àrea del pic | mAU·min | Àrea integrada del pic principal |
| **Area_T** | Àrea total | mAU·min | Àrea total del cromatograma |
| **RF** | Response Factor | mAU·min/ppm | Àrea / Concentració. Sensibilitat del detector |
| **RF_V** | RF normalitzat per Volum | (mAU·min/ppm)/µL ×100 | RF / Volum × 100. **CLAU per comparar calibracions amb diferent volum d'injecció** |

### Temps i Forma del Pic

| Abrev. | Nom Complet | Unitats | Descripció |
|--------|-------------|---------|------------|
| **t_max** | Temps de retenció | min | Temps del màxim del pic |
| **FWHM** | Full Width at Half Maximum | min | Amplada del pic a mitja alçada. Indicador de resolució/degradació columna |
| **Sym** | Simetria | - | Ratio d'amplades esquerra/dreta. Ideal ≈ 1.0 |

### Qualitat del Senyal

| Abrev. | Nom Complet | Unitats | Descripció |
|--------|-------------|---------|------------|
| **SNR** | Signal-to-Noise Ratio | - | Alçada pic / Soroll baseline. Mínim acceptable: 10 |
| **AR** | Area Ratio | - | Àrea pic principal / Àrea total. Indica "puresa" del cromatograma |
| **nP** | Nombre de pics | - | Pics detectats. Ideal = 1 per KHP |

### Relació entre Senyals

| Abrev. | Nom Complet | Unitats | Descripció |
|--------|-------------|---------|------------|
| **DOC/254** | Ratio DOC/254nm | - | Àrea DOC / Àrea 254nm. Característic de cada compost |
| **Shift** | Shift temporal | s | Diferència t_max entre DOC i 254nm |

## Paràmetres per Senyal

### Direct (D)
- `area_d` - Àrea pic principal DOC Direct
- `rf_d` - Response Factor Direct
- `rf_v_d` - RF Direct normalitzat per volum
- `t_max_d` - Temps retenció DOC Direct
- `fwhm_d` - FWHM DOC Direct
- `snr_d` - SNR DOC Direct
- `sym_d` - Simetria DOC Direct
- `ar_d` - Area Ratio DOC Direct
- `d254_d` - Ratio DOC/254 (usant DOC Direct)

### UIB (U)
- `area_u` - Àrea pic principal DOC UIB
- `rf_u` - Response Factor UIB
- `rf_v_u` - RF UIB normalitzat per volum
- `t_max_u` - Temps retenció DOC UIB
- `fwhm_u` - FWHM DOC UIB
- `snr_u` - SNR DOC UIB
- `sym_u` - Simetria DOC UIB
- `ar_u` - Area Ratio DOC UIB
- `d254_u` - Ratio DOC/254 (usant DOC UIB)

### 254nm
- `area_254` - Àrea pic principal 254nm
- `rf_254` - Response Factor 254nm
- `rf_v_254` - RF 254nm normalitzat per volum
- `t_max_254` - Temps retenció 254nm
- `fwhm_254` - FWHM 254nm
- `ar_254` - Area Ratio 254nm

## Paràmetres de Condicions

| Camp | Descripció |
|------|------------|
| `conc_ppm` | Concentració KHP (ppm) |
| `volume_uL` | Volum d'injecció (µL) |
| `uib_sensitivity` | Sensibilitat UIB (ppb): 700 o 1000 |
| `mode` | Mode: COLUMN o BP |
| `doc_mode` | Mode DOC: DUAL, DIRECT, UIB |

## Paràmetres de Qualitat i Traçabilitat

| Camp | Descripció |
|------|------------|
| `quality_score` | Puntuació qualitat (0=perfecte, ≥100=invàlid) |
| `n_replicas` | Nombre de rèpliques |
| `rsd` | Desviació estàndard relativa entre rèpliques (%) |
| `selection.method` | Mètode selecció: average, R1, R2, best_quality |
| `selection.is_manual` | Si la selecció és manual |

## Fórmules

```
RF = Area / Concentració
RF_V = RF / Volum × 100
AR = Area_pic_principal / Area_total
D/254 = Area_DOC / Area_254
Shift = t_max_DOC - t_max_254
```

## Valors de Referència (KHP)

| Paràmetre | Rang Normal | Alerta |
|-----------|-------------|--------|
| AR | > 0.9 | < 0.7 |
| nP | 1 | > 1 |
| SNR | > 100 | < 10 |
| FWHM | < 1.2 min | > 1.5 min |
| Sym | 0.8 - 1.5 | < 0.5 o > 2.5 |
| Shift | < 30 s | > 60 s |
