# ANÀLISI CALIBRACIÓ KHP - RESULTATS I TASQUES PENDENTS

Data: 2026-02-07

---

## 1. BIGAUSSIAN FIT COM A DETECTOR D'OUTLIERS

El fit bigaussià (gaussiana asimètrica) ajusta el pic DOC i dóna:
- **R²**: qualitat de l'ajust (>0.987 = VALID, 0.980-0.987 = CHECK, <0.980 = INVALID)
- **asymmetry**: ratio sigma_right/sigma_left (pic ideal ~1.5 per HPSEC, >3 indica deformació)

**BG INVALID vol dir**: el pic KHP no s'ajusta bé a una gaussiana asimètrica.
Causes: pic doble, cua excessiva, solapament amb altres pics, baseline molt desplaçada,
pic retallat/saturat. Un BG INVALID descarta la calibració perquè indica que la
integració d'àrea no és fiable.

### Casos detectats:
- **272_SEQ**: R²=0.974, asymmetry=0.56, sym=2.39, FWHM=1.89 → Pic molt ample i asimètric al revés (cua a l'esquerra). RSD=16.9% entre rèpliques. Àrea inflada ×10.
- **266_SEQ**: R²=0.916, asymmetry=5.66, sym=0.42 → Pic extremadament deformat (cua 5.7× més ampla a la dreta). RSD=41% entre rèpliques (704 vs 1692). Probable solapament o baseline incorrecta.

---

## 2. HIPÒTESI: COLUMN 100µL SÓN REALMENT 400µL

Referència COLUMN 400µL (n=6): **RF_mass = 726 ± 8 (CV=1.1%)**

| SEQ | Conc | Àrea | RF@100 | RF@400 | vs ref | BG R² | Sym | Veredicte |
|-----|------|------|--------|--------|--------|-------|-----|-----------|
| 269B_SEQ | 2ppm | 576.8 | 2883.8 | **720.9** | **-0.8%** | N/D | 0.87 | PERFECTE |
| 262B_SEQ | 5ppm | 1379.8 | 2759.6 | **689.9** | **-5.0%** | 0.999 | 0.74 | OK |
| 265_SEQ | 5ppm | 1366.6 | 2733.3 | **683.3** | **-5.9%** | 0.999 | 0.73 | OK |
| 256_SEQ | 5ppm | 1313.6 | 2627.2 | **656.8** | **-9.6%** | 0.999 | 0.86 | OK (límit) |
| 262_SEQ | 5ppm | 1289.3 | 2578.6 | **644.7** | **-11.3%** | 0.998 | 0.80 | WARNING |
| 258_SEQ | 1ppm | 114.6 | 1145.7 | 286.4 | -60.6% | 0.991 | 0.77 | NO (1ppm!) |
| 274_SEQ | 2ppm | 148.7 | 743.3 | 185.8 | -74.4% | 0.999 | 0.86 | NO |
| 272_SEQ | 2ppm | 1445.1 | 7225.7 | 1806.4 | +149% | 0.974 | 2.39 | REBUTJAR (BG INVALID) |

**5 de 8 encaixen com a 400µL** (269B, 262B, 265, 256, 262).

---

## 3. CASOS A REVISAR MANUALMENT

### 262_SEQ — WARNING (-11.3%)
- BG R²=0.998 (VALID), sym=0.80, RSD=0.54% → Pic perfecte
- Àrea=1289 vs 262B_SEQ=1380 (mateixa mostra, diferent processament?)
- Si és 400µL: RF=645, un 11% per sota de referència 726
- **REVISAR**: confirmar volum d'injecció real al MasterFile/log instrument

### 258_SEQ — NO PASSA ni com 100 ni com 400
- Concentració **1ppm** (no 2ppm ni 5ppm) → cas diferent!
- BG R²=0.991 (VALID), RSD=0.09% → Pic excel·lent
- RF@100=1146 (si fos 100µL esperat ~726 → +58%)
- RF@400=286 (si fos 400µL → -61%)
- **REVISAR**: confirmar concentració real i volum. No encaixa en cap escenari estàndard

### 274_SEQ — NO PASSA ni com 100 ni com 400
- BG R²=0.999 (VALID), sym=0.86, RSD=0.83% → Pic perfecte
- Àrea=149, molt baixa per 2ppm
- RF@100=743 (coincideix amb ref 726 si és 100µL!)
- **HIPÒTESI**: Aquesta SÍ que és realment 100µL. RF_mass=743 ≈ 726 referència
- **REVISAR**: confirmar volum. Si és 100µL real, confirma que RF_mass COLUMN és ~730

### 272_SEQ — REBUTJAR
- BG INVALID (R²=0.974), sym=2.39, FWHM=1.89
- Àrea=1445 (rep1) vs 2034 (rep2) → RSD=17%
- Pic deformat (doble o saturat), no fiable
- **ACCIÓ**: Marcar com a outlier

### 266_SEQ — REBUTJAR
- BG INVALID (R²=0.916), asymmetry=5.66, sym=0.42
- Àrea rep1=705 vs rep2=1692 → RSD=41%
- Pic completament deformat
- **ACCIÓ**: Ja marcat com outlier, correcte

---

## 4. MODELS DE CALIBRACIÓ — REGRESSIÓ LLIURE vs ORIGEN

### Anàlisi estadística

|  | COLUMN | BP |
|--|--------|----|
| **Model recomanat** | **LLIURE** | Origen |
| Equació | Area = 628 × Mass + 81 | Area = 915 × Mass |
| R² | **0.9944** | 0.8213 |
| n punts | 13 | 7 |
| Error abs mig | **2.6%** | 7.5% |
| Intercept | 80.8 ± 18.3 | -8.2 ± 37.5 |
| p-value intercept | **0.0011** (significatiu) | 0.84 (no signif.) |
| AIC (vs origen) | 88.7 vs 100.0 → **lliure millor** | 42.4 vs 40.5 → origen millor |

### Per què COLUMN necessita intercept?

L'intercept de 81 mAU·min és **estadísticament significatiu** (p=0.001).
Causa: la baseline a t_retention ~21min acumula àrea residual que no depèn
de la massa de KHP. Amb model origen, els punts a 0.8µg queden +6% per sobre
i els de 2.0µg queden -2.4% per sota → biaix sistemàtic.

### Per què BP té R² baixa?

No és problema de model: la R² és baixa perquè:
- Només 2 nivells de massa (0.1 i 0.2 µg) → poc rang dinàmic
- Dispersió real a 0.2µg: 155-201 mAU·min (277_SEQ=-15%, 286_SEQ=+10%)
- Calen més punts a masses diferents per millorar

---

## 5. COMPARACIÓ BP vs COLUMN AMB NOUS MODELS

### Predicció de concentració (model antic rf=682 vs nous models)

| SEQ | Mode | Àrea | Conc real | Conc antic | Err antic | Conc nou | Err nou |
|-----|------|------|-----------|------------|-----------|----------|---------|
| 270_SEQ_BP | BP | 195.8 | 2.0 | 2.87 | +43.5% | 2.14 | +7.0% |
| 273_SEQ_BP | BP | 167.5 | 2.0 | 2.46 | +22.8% | 1.83 | -8.4% |
| 277_SEQ_BP | BP | 155.2 | 2.0 | 2.28 | +13.8% | 1.70 | -15.2% |
| 279B_SEQ_BP | BP | 183.4 | 2.0 | 2.69 | +34.5% | 2.00 | +0.2% |
| 281_SEQ_BP | BP | 196.9 | 2.0 | 2.89 | +44.3% | 2.15 | +7.6% |
| 286_SEQ_BP | BP | 201.0 | 2.0 | 2.95 | +47.4% | 2.20 | +9.8% |
| 286_SEQ_BP | BP (50µL) | 87.6 | 2.0 | 2.57 | +28.4% | 1.91 | -4.3% |
| 275_SEQ | COL | 579.1 | 2.0 | 2.12 | +6.1% | 1.98 | -0.9% |
| 276B_SEQ | COL | 584.2 | 2.0 | 2.14 | +7.1% | 2.00 | +0.2% |
| 278_SEQ | COL | 591.4 | 2.0 | 2.17 | +8.4% | 2.03 | +1.6% |
| 282_SEQ | COL | 556.4 | 2.0 | 2.04 | +2.0% | 1.89 | -5.4% |
| 283_SEQ | COL | 630.7 | 2.0 | 2.31 | +15.6% | 2.19 | +9.4% |
| 256_SEQ | COL | 1313.6 | 5.0 | 4.82 | -3.7% | 4.91 | -1.9% |
| 262B_SEQ | COL | 1379.8 | 5.0 | 5.06 | +1.2% | 5.17 | +3.4% |

### Millora amb nous models

| | Model antic (rf=682) | Model nou | Millora |
|--|----------------------|-----------|---------|
| **BP error abs mig** | **33.5%** | **7.5%** | **-26 pp** |
| **COLUMN error abs mig** | 5.6% | 2.6% | -3 pp |

El model antic **sobreestimava BP en +34% de mitjana** perquè usava rf=682 (COLUMN)
per mostres BP que tenen rf=915.

### Àrees esperades per massa (amb el model correcte, ambdós donen la mateixa concentració)

| Mass (µg) | Àrea BP | Àrea COLUMN | Diff àrea |
|-----------|---------|-------------|-----------|
| 0.050 | 45.7 | 112.2 | -59% |
| 0.100 | 91.5 | 143.6 | -36% |
| 0.200 | 183.0 | 206.5 | -11% |
| 0.400 | 366.0 | 332.1 | +10% |
| 0.800 | 731.9 | 583.4 | +25% |
| 2.000 | 1829.8 | 1337.4 | +37% |

La diferència d'àrea **canvia amb la massa** perquè COLUMN té intercept i BP no.
A masses baixes COLUMN dóna més àrea (l'offset de 81 pesa més),
a masses altes BP dóna més àrea (slope 915 > 628).

---

## 6. DATASET FIABLE FINAL

### Inclosos (n=20)

**BP (n=7)**:
| SEQ | Vol | Conc | Mass | Àrea | RF | BG R² | Err model |
|-----|-----|------|------|------|----|-------|-----------|
| 286_SEQ_BP | 50 | 2 | 0.100 | 87.6 | 876 | 0.999 | -4.3% |
| 270_SEQ_BP | 100 | 2 | 0.200 | 195.8 | 979 | 0.999 | +7.0% |
| 273_SEQ_BP | 100 | 2 | 0.200 | 167.5 | 838 | 0.999 | -8.4% |
| 277_SEQ_BP | 100 | 2 | 0.200 | 155.2 | 776 | 0.999 | -15.2% |
| 279B_SEQ_BP | 100 | 2 | 0.200 | 183.4 | 917 | 0.999 | +0.2% |
| 281_SEQ_BP | 100 | 2 | 0.200 | 196.9 | 984 | 0.999 | +7.6% |
| 286_SEQ_BP | 100 | 2 | 0.200 | 201.0 | 1005 | 0.999 | +9.8% |

**COLUMN (n=13)**:
| SEQ | Vol | Rec? | Conc | Mass | Àrea | RF | BG R² | Err model |
|-----|-----|------|------|------|------|----|-------|-----------|
| 269B_SEQ | 400 | Y | 2 | 0.800 | 576.8 | 721 | N/D | -1.3% |
| 275_SEQ | 400 | | 2 | 0.800 | 579.1 | 724 | 0.997 | -0.9% |
| 276B_SEQ | 400 | | 2 | 0.800 | 584.2 | 730 | 0.997 | +0.2% |
| 276_SEQ | 400 | | 2 | 0.800 | 572.3 | 715 | 0.997 | -2.2% |
| 278_SEQ | 400 | | 2 | 0.800 | 591.4 | 739 | 0.997 | +1.6% |
| 282B_SEQ | 400 | | 2 | 0.800 | 584.4 | 730 | 0.997 | +0.2% |
| 282_SEQ | 400 | | 2 | 0.800 | 556.4 | 696 | 0.997 | -5.4% |
| 283_SEQ | 400 | | 2 | 0.800 | 630.7 | 788 | 0.998 | +9.4% |
| 285_SEQ | 400 | | 2 | 0.800 | 575.6 | 719 | 0.997 | -1.6% |
| 256_SEQ | 400 | Y | 5 | 2.000 | 1313.6 | 657 | 0.999 | -1.9% |
| 262B_SEQ | 400 | Y | 5 | 2.000 | 1379.8 | 690 | 0.999 | +3.4% |
| 262_SEQ | 400 | Y | 5 | 2.000 | 1289.3 | 645 | 0.998 | -3.8% |
| 265_SEQ | 400 | Y | 5 | 2.000 | 1366.6 | 683 | 0.999 | +2.3% |

### Exclosos (n=9 SEQs)

| SEQ | Motiu |
|-----|-------|
| 271_SEQ_BP | Outlier flag, àrea=4816 (×25 esperat) |
| 284_SEQ_BP | Outlier flag, àrea=79 (×0.4 esperat) |
| 263_SEQ_BP | IQR outlier, RF=1298 (fora rang) |
| 268_SEQ_BP | IQR outlier, RF=513 (fora rang) |
| 266_SEQ | BG INVALID R²=0.916, RSD=41% |
| 267_SEQ | Outlier flag |
| 272_SEQ | BG INVALID R²=0.974, sym=2.39 |
| 258_SEQ | 1ppm, no encaixa en cap model |
| 274_SEQ | Àrea no encaixa (possible 100µL real, a confirmar) |

---

## 7. ACCIONS PENDENTS

### Prioritat ALTA — Calibració

- [ ] **Implementar model lliure per COLUMN**: Area = 628 × Mass + 81
  - Modificar Calibration_Reference.json per suportar `intercept` (ara només `rf_mass_cal`)
  - Modificar `hpsec_calibrate.py` per usar model `y = slope*x + intercept`
  - El codi actual assumeix `Area = rf × Mass` → cal afegir intercept a la fórmula
- [ ] **Actualitzar rf_mass_cal per BP**: 682 → 915 (model origen)
- [ ] **Ampliar rang dinàmic BP**: afegir calibracions amb 50µL a 5ppm (mass=0.25µg)
  o 100µL a 1ppm/5ppm per millorar R²

### Prioritat ALTA — Dades

- [ ] **Revisar MasterFiles/logs**: confirmar volum real d'injecció per 256, 258, 262, 262B, 265, 269B, 274
- [ ] **Revisar 258_SEQ**: concentració 1ppm és correcta? Volum?
- [ ] **Revisar 274_SEQ**: si és realment 100µL, confirma RF≈730 per COLUMN 100µL
- [ ] **Marcar 272_SEQ com a outlier** (BG INVALID)

### Prioritat MITJA — Codi

- [ ] **Actualitzar KHP_History.json** amb bigaussian per totes les entrades
  (batch de re-calibració fet, però KHP_History potser no s'ha actualitzat)
- [ ] **Flexibilitzar criteri cal in/out** de 5% a 10% (TODO al codi)
- [ ] **269B_SEQ**: falta bigaussian (no s'ha pogut calcular), revisar per què
- [ ] **Toleràncies al report**: adaptar ±15%/±20% als nous models
- [ ] **Report PDF**: adaptar gràfic de recta de calibració als models lliure/origen per mode

### Prioritat BAIXA — Investigació

- [ ] **Entendre l'offset COLUMN de 81 mAU·min**: baseline a t~21min acumula àrea residual?
  Comprovar si depèn de la matriu de la mostra o és fix
- [ ] **277_SEQ_BP**: investigar per què desvia -15% (la pitjor del dataset BP net)
- [ ] **283_SEQ**: investigar per què desvia +9.4% (la pitjor del dataset COLUMN net)

---

## 8. CONCLUSIONS

1. **Bigaussian funciona** com a filtre QC: detecta 272_SEQ i 266_SEQ com a outliers reals
2. **COLUMN 100µL probablement són 400µL** per a 5 de 8 SEQs
3. **Cal model LLIURE per COLUMN** (intercept=81, p=0.001): R² passa de 0.9846 a **0.9944**
4. **Cal RF separat per BP**: 915 vs 682 actual → el model antic sobreestimava BP en **+34%**
5. **BP necessita més rang dinàmic** per millorar R²=0.82 (només 2 nivells de massa)
6. **Amb els nous models, l'error es redueix**: BP de 33.5% a 7.5%, COLUMN de 5.6% a 2.6%
7. **BP i COLUMN donen la mateixa concentració** si s'aplica el model correcte a cadascun
