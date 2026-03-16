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

## 4. DATASET NET ACTUAL

| Grup | n | RF_mass | CV |
|------|---|---------|-----|
| COLUMN 400µL | 6 | 726 ± 8 | 1.1% |
| COLUMN 100µL→400µL (5 que passen) | 5 | 679 ± 29 | 4.3% |
| Combinat COLUMN | 11 | 705 ± 30 | 4.3% |
| BP 100µL (net) | 6 | 917 ± 84 | 9.2% |
| BP 50µL | 1 | 876 | - |

**BP vs COLUMN: +29% sistemàtic** (910 vs 705)

---

## 5. TASQUES PENDENTS

- [ ] **Revisar MasterFiles/logs**: confirmar volum real d'injecció per 256, 258, 262, 262B, 265, 269B, 274
- [ ] **Revisar 258_SEQ**: concentració 1ppm és correcta? Volum?
- [ ] **Revisar 274_SEQ**: si és realment 100µL, confirma RF≈730 per COLUMN 100µL
- [ ] **Marcar 272_SEQ com a outlier** (BG INVALID, pic deformat)
- [ ] **Investigar diferència BP vs COLUMN +29%**: baseline? integració? Necessita mostres processades en ambdós modes
- [ ] **Actualitzar KHP_History.json** amb bigaussian per totes les entrades (batch fet, però History no s'ha actualitzat)
- [ ] **Decidir recta de calibració**: separada per mode o global amb factor de correcció?
- [ ] **Flexibilitzar criteri cal in/out** de 5% a 10% (TODO al codi)
- [ ] **269B_SEQ**: falta bigaussian (no s'ha pogut calcular), revisar per què

---

## 6. CONCLUSIONS PRELIMINARS

1. **Bigaussian funciona** com a filtre QC: detecta 272_SEQ i 266_SEQ com a outliers reals
2. **COLUMN 100µL probablement són 400µL** per a 5 de 8 SEQs (encaixen amb ±10%)
3. **274_SEQ pot ser l'única COLUMN 100µL real** (RF=743 ≈ referència 726)
4. **BP i COLUMN no donen el mateix RF_mass** (+29% BP). Cal investigar
5. **COLUMN 400µL és el grup més estable** (CV=1.1%), millor referència per calibració
