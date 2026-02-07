# Pla de Millora: Taula de Resultats (Analitzar)

**Data:** 2026-02-05
**Estat:** ✅ COMPLETAT (Backend v1.6.0 + Frontend)

---

## Problemes Identificats

### 1. Estructura de Senyals Barrejada
Hi ha 3 tipus de senyals que estan barrejats a la taula:

| Senyal | Descripció | Info necessària |
|--------|------------|-----------------|
| **DOC Direct** | Senyal principal DOC | Àrea, SNR, t_max, ppm (amb rf_direct) |
| **DOC UIB** | Senyal secundari DOC (si DUAL) | Àrea, SNR, t_max, ppm (amb rf_uib) |
| **DAD** | 6 λ (220, 252, 254, 272, 290, 362nm) | SNR per λ, Àrees per λ, R² |

### 2. Columnes de Rèplica
- **Actual:** 1 columna amb ★ (poc visible)
- **Hauria de ser:** 2 dropdowns independents
  - `Rep DOC`: Rèplica per DOC (Direct+UIB comparteixen)
  - `Rep DAD`: Rèplica per DAD (independent)

### 3. Àrees Barrejades
- **Actual:** Una sola columna "Àrea"
- **Hauria de ser:**
  - `A_Direct`: Àrea DOC Direct
  - `A_UIB`: Àrea DOC UIB (si mode DUAL)

### 4. Capçalera [ppm]
- **Actual:** `[ppm]`
- **Correcte:** `[DOC] (ppm)` - clarifica que és concentració calculada des de DOC

### 5. Calibració Direct vs UIB
- **CONFIRMAT:** Cal usar `rf_direct` per calcular ppm_Direct i `rf_uib` per calcular ppm_UIB
- Cal revisar que el codi ho faci correctament

### 6. Info DAD Incompleta
- **Actual:** Només SNR (màx/mín λ)
- **Falta:**
  - Àrees per λ (ES CALCULEN - cal mostrar)
  - Selecció de rèplica independent
  - R² entre rèpliques DAD
  - SNR mínim + quina λ

### 7. Format Visual
- Columna mostra: massa ampla
- Última columna: desquadrada
- Marges: revisar altres panels per consistència

### 8. Mode BP
- **IMPORTANT:** En mode BP SÍ pot haver-hi UIB (mode DUAL)
- No és que BP = sense UIB

---

## Proposta: Selector DOC / DAD

En lloc d'una taula farragosa, **dues vistes** amb selector:

```
[ ● DOC (Direct/UIB) ]  [ ○ DAD (6λ) ]
```

### Vista DOC (Direct/UIB)

| Mostra | Rep | A_Direct | ppm_D | A_UIB | ppm_U | R²_rep | SNR_D | SNR_U | Estat |
|--------|-----|----------|-------|-------|-------|--------|-------|-------|-------|

- **Rep**: Dropdown selecció rèplica DOC
- **A_Direct / A_UIB**: Àrees separades
- **ppm_D / ppm_U**: Calculats amb rf_direct i rf_uib respectivament
- **R²_rep**: Pearson entre rèpliques (Direct)
- **SNR_D / SNR_U**: SNR per cada senyal

#### Mode BP: Info addicional
| R²_bigauss | Asym | Quality |
|------------|------|---------|
- **R²_bigauss**: R² de l'ajust bi-gaussià
- **Asym**: Ratio sigma_right/sigma_left (ideal ~1.0)
- **Quality**: VALID / CHECK / INVALID

#### Columna Estat (totes les mostres)
Indicadors visuals per:
- ⚠️ **Timeout**: Si `timeout_info.n_timeouts > 0` (tooltip amb zona i severitat)
- 🦇 **Batman**: Si `BATMAN_DIRECT` o `BATMAN_UIB` a anomalies
- ⬇️ **Low SNR**: Si `LOW_SNR` a anomalies
- ❌ **No Peak**: Si `NO_PEAK` a anomalies
- ✓ **OK**: Si no hi ha anomalies

### Vista DAD (6 longituds d'ona)

| Mostra | Rep | A_254 | SNR_220 | SNR_252 | SNR_254 | SNR_272 | SNR_290 | SNR_362 | SNR_min | λ_min | R²_rep |
|--------|-----|-------|---------|---------|---------|---------|---------|---------|---------|-------|--------|

- **Rep**: Dropdown selecció rèplica DAD (independent de DOC)
- **A_254**: Àrea a 254nm (o altres λ rellevants)
- **SNR per λ**: Les 6 longituds d'ona
- **SNR_min**: El valor mínim de SNR
- **λ_min**: Quina λ té el mínim SNR
- **R²_rep**: Pearson entre rèpliques DAD

---

## Pla d'Implementació

### Fase 1: Revisió de Dades
1. Llegir `analysis_result.json` per veure quines dades es guarden
2. Verificar que `rf_uib` s'usa correctament a `hpsec_analyze.py`
3. Comprovar càlcul de R² entre rèpliques (DOC i DAD)
4. Identificar dades que falten o no es calculen
5. Revisar `target_wavelengths`: [220, 252, 254, 272, 290, 362]

### Fase 2: Backend (hpsec_analyze.py)
1. Assegurar càlcul ppm amb rf_direct i rf_uib per separat
2. Afegir R² entre rèpliques per DAD si falta
3. Guardar SNR per cada λ del DAD (ja es fa via `snr_info_dad`)
4. Guardar àrees per cada λ del DAD (ja es fa via `dad_wavelengths`)
5. Guardar selecció de rèplica DOC i DAD separadament

### Fase 3: Frontend (analyze_panel.py)
1. Afegir selector DOC/DAD a sobre de la taula
2. Crear dues funcions: `_populate_doc_table()` i `_populate_dad_table()`
3. Dropdowns de rèplica independents per cada vista
4. Format consistent amb altres panels (mides, marges)
5. Columna mostra més estreta

### Fase 4: Integració
1. Quan canvia rèplica DOC → recalcular ppm_D i ppm_U
2. Quan canvia rèplica DAD → actualitzar SNRs i àrees DAD
3. Guardar seleccions al JSON

---

## Dades que es Calculen (Revisió analysis_result.json)

### Estructura General
```json
{
  "samples": [...],           // Llista plana de totes les rèpliques
  "samples_grouped": {...}    // Agrupat per mostra amb comparació rèpliques
}
```

### Per cada rèplica (samples[])

#### DOC
- `peak_info`: area, t_max, height, baseline_level, valid
- `snr_info`:
  - `snr_direct`: SNR DOC Direct
  - `snr_uib`: SNR DOC UIB (si DUAL)
  - `baseline_noise_direct`, `lod_direct`, `loq_direct`
- `areas.DOC`: Àrees per fraccions (BP: BioP | COLUMN: BB, HS, SB, LMW)
- `areas_uib`: **SEMPRE BUIT {}** ⚠️ **PROBLEMA: No es calculen àrees UIB!**

#### Bigaussian (NOMÉS mode BP) ✅
- `bigaussian_doc`: Ajust bi-gaussià per DOC
  - `r2`: R² de l'ajust
  - `amplitude`, `sigma_left`, `sigma_right`
  - `asymmetry`: ratio sigma_right/sigma_left
  - `quality`: VALID (R² ≥ 0.987) / CHECK (≥ 0.980) / INVALID
- `bigaussian_254`: Ajust bi-gaussià per DAD 254nm (mateixa estructura)

#### Timeouts TOC ✅ (CRÍTIC per selecció rèpliques)
- `timeout_info`: Info de timeouts detectats
  - `n_timeouts`: Nombre de timeouts
  - `n_major_timeouts`: Timeouts majors (recàrrega xeringues ~74s)
  - `severity`: OK / INFO / WARNING / CRITICAL
  - `timeouts[]`: Llista amb detalls per cada timeout:
    - `t_start_min`: Posició temporal
    - `duration_sec`: Duració
    - `zone`: Zona afectada (BioP, HS, BB, SB, LMW, POST_RUN)
  - `zone_summary`: Resum zones afectades
  - `total_affected_min`: Duració total afectada

**Severitat per zona:**
| Zona | Rang (min) | Severitat |
|------|------------|-----------|
| HS | 18-23 | **CRITICAL** |
| BioP | 0-18 | WARNING |
| BB | 23-30 | WARNING |
| SB | 30-40 | WARNING |
| LMW | 40-70 | INFO |
| POST_RUN | 70+ | OK |

#### Anomalies (afecten selecció rèplica) ✅
- `anomalies[]`: Llista d'anomalies detectades
  - `BATMAN_DIRECT`: Patró Batman a DOC Direct
  - `BATMAN_UIB`: Patró Batman a DOC UIB
  - `NO_PEAK`: No s'ha detectat pic
  - `TIMEOUT_IN_PEAK`: Timeout afecta el pic principal (CRÍTIC!)
  - `LOW_SNR`: SNR per sota del llindar
  - `UIB_NO_BASELINE`: UIB sense baseline calculat

#### DAD (6 wavelengths: 220, 252, 254, 272, 290, 362)
- `areas.A220`, `areas.A252`, `areas.A254`, `areas.A272`, `areas.A290`, `areas.A362`
  - Cadascun amb fraccions (BioP per BP, BB/HS/SB/LMW per COLUMN)
  - `total`: àrea total de la λ
- `snr_info_dad.A220`, etc.
  - `snr`: SNR per la λ
  - `noise`, `peak_height`, `lod`, `loq`

### Per mostra agrupada (samples_grouped{})

- `replicas`: Dict amb rèpliques {"1": {...}, "2": {...}}
- `comparison`: Comparació R² entre rèpliques
- `selection_reason`: Raó selecció automàtica
- `selected`: **JA EXISTEIX separació DOC/DAD!**
  ```json
  "selected": {
    "dad": "1",
    "doc": "2"
  }
  ```
- `quantification`:
  - `concentration_ppm`: **NOMÉS UN VALOR** ⚠️ (hauria de ser ppm_direct i ppm_uib)
  - `fractions`: {}

---

## Problemes Detectats al Backend (Anàlisi Codi)

### 1. `areas_uib` sempre buit en mode BP

**Localització:** `hpsec_analyze.py`, línies 1551-1565

```python
if not is_bp:  # ⚠️ PROBLEMA: Només calcula areas_uib si NO és BP!
    ...
    if is_dual and "DOC" in areas:
        areas_uib = calcular_fraccions_temps(t_doc, y_doc_uib_net, config)
        result["areas_uib"] = areas_uib
```

**Problema:** El bloc `if not is_bp:` exclou el càlcul d'`areas_uib` per a mode BP.
Però l'usuari ha confirmat que BP pot tenir mode DUAL.

**Fix proposat:** Moure el càlcul d'`areas_uib` fora del bloc `if not is_bp:` o afegir bloc específic per BP DUAL.

### 2. `quantify_sample` només usa un RF

**Localització:** `hpsec_analyze.py`, línies 1108-1182

```python
def quantify_sample(sample_result, calibration_data, mode="COLUMN"):
    ...
    rf = calibration_data.get("rf")  # ⚠️ Només usa "rf", no rf_direct/rf_uib
    ...
    result["concentration_ppm"] = float(area_total / rf)  # Un sol valor
```

**Problema:** Només calcula `concentration_ppm` amb un RF genèric.
No usa `rf_direct` i `rf_uib` per separat.

**Fix proposat:** Modificar per retornar:
- `concentration_ppm_direct` = area_direct / rf_direct
- `concentration_ppm_uib` = area_uib / rf_uib

### 3. SNR UIB SÍ es calcula ✅

**Localització:** `hpsec_analyze.py`, línies 519-524

```python
if noise_uib > 0:
    result["snr_uib"] = height_uib / noise_uib
else:
    result["snr_uib"] = calc_snr(y_doc_uib, height_uib)
```

**Estat:** OK, `snr_uib` SÍ es calcula i es guarda a `snr_info`.

### 4. Calibració guarda rf_direct i rf_uib correctament ✅

**Localització:** `hpsec_calibrate.py`, línies 3095-3100

```python
result = {
    "rf_direct": 0.0,
    ...
    "rf_uib": 0.0,
    ...
}
```

**Estat:** OK, la calibració SÍ guarda `rf_direct` i `rf_uib` per separat.

### 5. R² entre rèpliques DAD NOMÉS per 254nm ⚠️

**Localització:** `hpsec_analyze.py`, línies 966-1015

```python
# COMPARACIÓ DAD (254nm)  ← Només 254!
if '254' in df_dad1.columns and '254' in df_dad2.columns:
    pearson_254, _ = pearsonr(y_254_1, y_254_2)
    result["dad"]["pearson_254"] = float(pearson_254)
```

**Problema:** Només calcula R² per 254nm, no per les altres 5 λ (220, 252, 272, 290, 362).

**Fix proposat:** Calcular R² per totes les λ i afegir:
- `pearson_per_wavelength`: Dict amb R² per cada λ
- `pearson_min`: Valor mínim de R²
- `wavelength_min`: Quina λ té el mínim R²

---

## Accions Requerides al Backend

### ✅ Ja funciona correctament:
1. `snr_uib` es calcula (línies 519-524)
2. `rf_direct` i `rf_uib` es guarden a calibració
3. Àrees DAD per λ es calculen (`areas.A220`, `areas.A254`, etc.)
4. SNR DAD per λ es calcula (`snr_info_dad.A220`, etc.)
5. Selecció independent DOC/DAD ja existeix (`selected.doc`, `selected.dad`)
6. **Bigaussian** per BP es calcula (`bigaussian_doc`, `bigaussian_254`) - cal mostrar!
7. **Timeouts** es detecten i afecten selecció (`timeout_info`, `TIMEOUT_IN_PEAK`)
8. **Anomalies** es detecten i es consideren crítiques per selecció

#### Lògica de selecció de rèplica (`recommend_replica`):
```
Anomalies crítiques (exclouen rèplica):
  - BATMAN_DIRECT, BATMAN_UIB, NO_PEAK, TIMEOUT_IN_PEAK

Ordre de decisió:
  1. Si R1 té anomalies crítiques i R2 no → seleccionar R2
  2. Si R2 té anomalies crítiques i R1 no → seleccionar R1
  3. Si ambdues tenen anomalies → triar per SNR (score baix: 0.3)
  4. Si cap té anomalies → triar per SNR (>10% diferència)
```

### ✅ Corregit (v1.6.0):
1. **`areas_uib` en mode BP**: Mogut fora de `if not is_bp:` ✓
2. **`quantify_sample`**: Ara retorna `ppm_direct` i `ppm_uib` separats ✓
3. **R² DAD per totes les λ**: Afegit `pearson_per_wavelength`, `pearson_min`, `wavelength_min` ✓

### 📝 Canvis concrets:

#### Fix 1: areas_uib per BP DUAL
```python
# Línia ~1560: Moure FORA del if not is_bp
if is_dual and "DOC" in areas:
    areas_uib = calcular_fraccions_temps(t_doc, y_doc_uib_net, config)
    result["areas_uib"] = areas_uib
```

#### Fix 2: quantify_sample amb rf_direct/rf_uib
```python
def quantify_sample(sample_result, calibration_data, mode="COLUMN"):
    ...
    rf_direct = calibration_data.get("rf_direct") or calibration_data.get("rf")
    rf_uib = calibration_data.get("rf_uib", 0)

    # DOC Direct
    area_direct = sample_result.get("areas", {}).get("DOC", {}).get("total", 0)
    if area_direct > 0 and rf_direct > 0:
        result["concentration_ppm_direct"] = float(area_direct / rf_direct)

    # DOC UIB
    area_uib = sample_result.get("areas_uib", {}).get("total", 0)
    if area_uib > 0 and rf_uib > 0:
        result["concentration_ppm_uib"] = float(area_uib / rf_uib)
```

#### Fix 3: R² DAD per totes les λ
```python
# A compare_replicas(), després de la línia 972:
wavelengths = ['220', '252', '254', '272', '290', '362']
pearson_per_wl = {}

for wl in wavelengths:
    if wl in df_dad1.columns and wl in df_dad2.columns:
        y1 = df_dad1[wl].to_numpy()
        y2 = df_dad2[wl].to_numpy()
        # Interpolar si cal...
        pearson_val, _ = pearsonr(y1_interp, y2_interp)
        pearson_per_wl[wl] = float(pearson_val)

result["dad"]["pearson_per_wavelength"] = pearson_per_wl
if pearson_per_wl:
    min_wl = min(pearson_per_wl, key=pearson_per_wl.get)
    result["dad"]["pearson_min"] = pearson_per_wl[min_wl]
    result["dad"]["wavelength_min"] = min_wl
```

---

## Preguntes Pendents

1. ~~Calibració UIB usa rf_uib?~~ → **SÍ, confirmat**
2. ~~Rèpliques DOC i DAD independents?~~ → **SÍ, confirmat**
3. ~~Mode BP té UIB?~~ → **SÍ, pot tenir mode DUAL**
4. Quines àrees DAD mostrar? (totes 6 o només 254?)
5. Cal R² entre mostres a més de R² entre rèpliques?

---

## Resum Canvis Frontend (analyze_panel.py)

### Estructura Nova

```
┌─────────────────────────────────────────────────────────────┐
│ [WARNINGS BAR - si hi ha avisos]                           │
├─────────────────────────────────────────────────────────────┤
│ INFO PANEL: DADES | CALIBRACIÓ | STATUS                    │
├─────────────────────────────────────────────────────────────┤
│ [ ● DOC (Direct/UIB) ]  [ ○ DAD (6λ) ]   ← Selector        │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────────┐│
│ │ TAULA DOC (visible quan selector = DOC)                 ││
│ │ Mostra | Rep▼ | A_Dir | ppm_D | A_UIB | ppm_U | R² | SNR││
│ └──────────────────────────────────────────────────────────┘│
│ ┌──────────────────────────────────────────────────────────┐│
│ │ TAULA DAD (visible quan selector = DAD)                 ││
│ │ Mostra | Rep▼ | A254 | SNR_220..362 | SNR_min | λ_min   ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Mides Proposades

- Columna Mostra: 120px (era massa ampla)
- Columna Rep: 60px (dropdown)
- Columnes numèriques: 80px cadascuna
- Columna Status: 50px

### ✅ Implementat

1. **Backend (v1.6.0)**: `areas_uib`, `quantify_sample`, R² DAD 6λ ✓
2. **Frontend**: Selector DOC/DAD amb botons toggle ✓
3. **Frontend**: `_populate_doc_table()` i `_populate_dad_table()` ✓
4. **Frontend**: Dropdowns rèplica independents per DOC i DAD ✓
5. **Integració**: Seleccions es guarden a `selected.doc` i `selected.dad` ✓

---

## Notes Addicionals

- L'informe detallat es revisarà per separat
- Les àrees DAD ES CALCULEN, només cal mostrar-les
- El selector DOC/DAD fa les taules més clares i menys farragoses
- La barra de warnings ja està implementada (coherent amb altres panels)
