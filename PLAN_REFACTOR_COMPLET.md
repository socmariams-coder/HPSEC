# PLA DE REFACTORITZACIÓ COMPLET - HPSEC Suite

**Data inici:** 2026-01-29
**Última actualització:** 2026-01-30
**Objectiu:** Traçabilitat total + JSON com a font única + Eliminar duplicacions

---

## 0. REGISTRE DE CANVIS

### 2026-01-30 - Sessió 4: Pla Refactor Pipeline 5 Fases
**Consultat:** Sí

**Anàlisi realitzada:**
- Revisat flux actual: pipeline llegeix Excels ja processats (de consolidate antic)
- Identificat que `hpsec_process.py` té funcions però pipeline NO les crida
- Definida arquitectura objectiu: pipeline com a orquestra (no processa)
- Documentat input/output esperat per cada fase

**Pla creat:** Secció 8 - PLA REFACTOR PIPELINE

**Pròxims passos:**
1. Verificar `hpsec_process.process_sequence()`
2. Verificar/crear `hpsec_review.review_sequence()`
3. Verificar/crear `hpsec_reports.export_sequence()`

### 2026-01-30 - Sessió 3: Fix BP mode + Tests
**Consultat:** Sí

**Problemes detectats:**
1. `hpsec_replica.py:293` - Comprovava `fit_result.get("valid")` però `fit_bigaussian()` retorna `status`
2. `hpsec_pipeline.py:510` - Passava config global a `consolidate_sequence()` però aquest espera format diferent

**Correccions:**
- ✅ hpsec_replica.py: Canvi `fit_result.get("valid")` → `fit_result.get("status") in ("VALID", "CHECK")`
- ✅ hpsec_pipeline.py: `config=None` per usar DEFAULT_CONSOLIDATE_CONFIG intern
- ✅ hpsec_replica.py: `_baseline_stats()` ara usa últims punts per BP (pic és al principi)

**Tests:**
- ✅ 283_SEQ (COLUMN) - Passa amb tots els camps traçabilitat
- ✅ 284_SEQ_BP (BP) - Passa amb r2=0.9979, r2_status=VALID

### 2026-01-29 - Sessió 2: Neteja duplicacions
**Consultat:** Sí (després de fer canvis - ERROR, cal consultar ABANS)

**Canvis realitzats:**
- ✅ hpsec_consolidate.py: Eliminades definicions locals duplicades (~85 línies)
  - normalize_key, is_khp, mode_robust, obtenir_seq, seq_tag
  - is_bp_seq, skip_sample_direct, split_sample_rep, clean_sample_name, normalize_rep
  - Ara s'importen de hpsec_import.py
- ✅ hpsec_calibrate.py: Afegits imports desde hpsec_import i hpsec_utils
  - Eliminades definicions locals de is_khp, extract_khp_conc, obtenir_seq, mode_robust, t_at_max
- ✅ hpsec_process.py: Afegit import mode_robust desde hpsec_utils
- ✅ hpsec_reports.py: Afegit import is_khp desde hpsec_import
- ✅ hpsec_utils.py: Eliminades funcions mogudes a hpsec_import

**Tests:**
- ✅ Tots els imports funcionen
- ✅ Funcions retornen valors correctes
- ✅ Test amb seqüència real (283_SEQ) passa

### 2026-01-29 - Sessió 2 (cont.): Ampliar JSON per traçabilitat
**Consultat:** Sí

**Canvis realitzats:**
- ✅ hpsec_replica.py: Afegida funció `compare_replicas_by_fraction()` amb FRACTION_ZONES
- ✅ hpsec_pipeline.py: Ampliat `_create_export_summary()` per incloure:
  - peak_indices (left, right, max)
  - baseline_stats (noise, percentile_10)
  - timeout_info (n_timeouts, severity, intervals[])
  - smoothness, anomaly_score
  - r2, r2_status, asymmetry (per BP)
  - low_signal flag
  - height, t_peak
- ✅ hpsec_pipeline.py: Afegit comparison.by_fraction per COLUMN mode

**Pendent:**
- ⏳ Reprocessar seqüències per generar nous JSONs
- ⏳ Verificar que GUI llegeix correctament els nous camps

---

## 1. ESTAT ACTUAL (Actualitzat)

### 1.1 Mòduls Principals
```
CAPA 0 - CONFIG
├── hpsec_config.py      - Configuració centralitzada

CAPA 1 - CORE (Fonts Canòniques)
├── hpsec_core.py        - Detecció: detect_main_peak, detect_batman, detect_timeout
├── hpsec_utils.py       - Utilitats: baseline_stats, mode_robust, t_at_max
└── hpsec_import.py      - Identificació: is_khp, extract_khp_conc, obtenir_seq, normalize_key

CAPA 2 - PROCESSAMENT
├── hpsec_consolidate.py - Consolidació (re-exporta de CAPA 1 per compatibilitat)
├── hpsec_calibrate.py   - Calibració KHP
├── hpsec_process.py     - Processament senyal
└── hpsec_replica.py     - Avaluació/selecció rèpliques

CAPA 3 - ORQUESTRACIÓ
├── hpsec_pipeline.py    - Pipeline complet
└── hpsec_review.py      - Revisió seleccions

CAPA 4 - OUTPUT
├── hpsec_reports.py     - Generació PDF
└── HPSEC_Suite.py       - GUI
```

### 1.2 Duplicacions - ESTAT ACTUAL

| Funció | Font Canònica | Estat |
|--------|---------------|-------|
| `detect_main_peak()` | hpsec_core.py | ✅ COMPLETAT |
| `detect_batman()` | hpsec_core.py | ✅ COMPLETAT |
| `find_peak_boundaries()` | hpsec_core.py | ✅ COMPLETAT |
| `is_khp()` | hpsec_import.py | ✅ COMPLETAT |
| `extract_khp_conc()` | hpsec_import.py | ✅ COMPLETAT |
| `obtenir_seq()` | hpsec_import.py | ✅ COMPLETAT |
| `normalize_key()` | hpsec_import.py | ✅ COMPLETAT |
| `mode_robust()` | hpsec_utils.py | ✅ COMPLETAT |
| `t_at_max()` | hpsec_utils.py | ✅ COMPLETAT |
| `baseline_stats()` | hpsec_utils.py | ✅ COMPLETAT |

### 1.3 Funcions Òrfenes (Pendents de Revisar)
- `_update_khp_stats()` - hpsec_consolidate
- `_extract_replicas_info()` - hpsec_calibrate
- `_get_reference_area()` - hpsec_calibrate
- `_flatten_samples_for_processing()` - hpsec_process

---

## 2. BUITS DE TRAÇABILITAT ✅ RESOLT (2026-01-30)

### 2.1 Mètriques Calculades - ESTAT ACTUAL

| Mètrica | Calculada a | Propòsit | Estat |
|---------|-------------|----------|-------|
| `peak_indices` (left/right/max) | hpsec_replica | Límits integració | ✅ GUARDAT |
| `baseline_stats` (noise) | hpsec_replica | Soroll baseline | ✅ GUARDAT |
| `smoothness` | hpsec_core | Irregularitat pic | ✅ GUARDAT |
| `r2_status` (VALID/CHECK/INVALID) | hpsec_replica | Qualitat fit | ✅ GUARDAT |
| `anomaly_score` | hpsec_replica | Puntuació combinada | ✅ GUARDAT |
| `timeout_intervals` | hpsec_core | Intervals exactes | ✅ GUARDAT |
| `asymmetry` | hpsec_replica | Sigma ratio | ✅ GUARDAT |
| `comparison_by_fraction` | hpsec_replica | Pearson per zona | ✅ GUARDAT |
| `dad_eval` (A254) | hpsec_replica | Wavelength principal | ✅ GUARDAT |

### 2.2 Nivell Traçabilitat Actual: **95-100%**

**Reconstructible des de JSON:**
- ✅ Concentracions (area × factor)
- ✅ Seleccions amb motius
- ✅ Històric calibració
- ✅ Per què un pic és "batman" (anomaly_score, smoothness)
- ✅ Intervals exactes de timeout (timeout_info.intervals)
- ✅ Soroll baseline (baseline_stats.noise)
- ✅ Qualitat fit (r2, r2_status)
- ✅ Comparació per fracció (comparison.by_fraction)

---

## 3. FLUX DE DADES (JSON-Centric)

```
         ┌────────────────────────────────────┐
         │         FITXERS FONT               │
         │  MasterFile.xlsx, UIB.csv, DAD/    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────────┐
         │  FASE 1: IMPORT + CONSOLIDACIÓ     │
         │  hpsec_import.py + hpsec_consolidate│
         └──────────────┬─────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────────┐
         │  consolidation.json                │
         │  + Resultats_Consolidats/*.xlsx    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────────┐
         │  FASE 2: CALIBRACIÓ                │
         │  hpsec_calibrate.py                │
         └──────────────┬─────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────────┐
         │  CALDATA/calibrations.json         │
         └──────────────┬─────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────────┐
         │  FASE 3: PROCESSAMENT + SELECCIÓ   │
         │  hpsec_process + hpsec_replica     │
         └──────────────┬─────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────────┐
         │  processing_summary.json           │
         │  (FONT ÚNICA PER GUI I REPORTS)    │
         └──────────────┬─────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐
   │  HPSEC_Suite    │    │  hpsec_reports  │
   │  (GUI)          │    │  (PDF)          │
   └─────────────────┘    └─────────────────┘
```

---

## 4. PLA D'EXECUCIÓ

### FASE 1: Eliminar Duplicacions ✅ COMPLETAT

| Tasca | Estat | Data |
|-------|-------|------|
| 1.1 Eliminar detect_batman/detect_main_peak de utils | ✅ | 2026-01-29 |
| 1.2 Centralitzar is_khp/extract_khp_conc a hpsec_import | ✅ | 2026-01-29 |
| 1.3 Centralitzar mode_robust/t_at_max a hpsec_utils | ✅ | 2026-01-29 |
| 1.4 Netejar duplicats a hpsec_consolidate | ✅ | 2026-01-29 |
| 1.5 Netejar duplicats a hpsec_calibrate | ✅ | 2026-01-29 |
| 1.6 Actualitzar imports a hpsec_reports | ✅ | 2026-01-29 |

### FASE 2: Ampliar JSON per Traçabilitat ✅ COMPLETAT

**Tasca 2.1:** Afegir a `processing_summary.json` per cada rèplica:
```json
{
  "replicas": {
    "R1": {
      "valid": true,
      "snr": 1777.8,
      "batman": false,
      "timeout": false,
      "irr": false,
      "area": 1070.76,
      // NOU:
      "peak_indices": {"left": 432, "right": 582, "max": 512},
      "baseline_stats": {"mean": 0.001, "std": 0.015, "p10": -0.02, "p90": 0.03},
      "r2_value": 0.992,
      "r2_status": "VALID",
      "anomaly_score": 0.15,
      "asymmetry_ratio": 1.42,
      "smoothness": 85.3
    }
  }
}
```

**Tasca 2.2:** Afegir timeout_intervals:
```json
{
  "timeout_info": {
    "detected": true,
    "count": 1,
    "severity": "WARNING",
    "intervals": [
      {"start_min": 39.1, "end_min": 40.3, "zone": "SB", "duration_sec": 72}
    ]
  }
}
```

**Tasca 2.3:** Afegir comparison_by_fraction:
```json
{
  "comparison": {
    "pearson": 0.984,
    "area_diff_pct": 4.0,
    "by_fraction": {
      "BioP": {"pearson": 0.998, "area_diff_pct": 2.1},
      "HS": {"pearson": 0.983, "area_diff_pct": 15.3},
      "BB": {"pearson": 0.991, "area_diff_pct": 5.2}
    }
  }
}
```

### FASE 3: Refactor hpsec_replica per Exportar Tot ✅ COMPLETAT

| Tasca | Estat | Data |
|-------|-------|------|
| 3.1 Modificar evaluate_replica() per retornar TOTES les mètriques | ✅ | 2026-01-29 (ja ho feia) |
| 3.2 Afegir compare_replicas_by_fraction() | ✅ | 2026-01-29 |
| 3.3 Actualitzar hpsec_pipeline per guardar totes les mètriques noves | ✅ | 2026-01-29 |

### FASE 4: Validar GUI Llegeix Correctament (PENDENT)

| Tasca | Estat |
|-------|-------|
| 4.1 HPSEC_Suite ha de mostrar les noves mètriques | ⏳ |
| 4.2 Afegir columnes a la taula QC si escau | ⏳ |

### FASE 6: Reparació Batman Supervisada (NOU - PENDENT)

**Problema actual:** `repair_batman=True` s'aplica automàticament a KHP durant consolidació.

**Objectiu:** La reparació ha de ser explícita i supervisada.

| Tasca | Estat |
|-------|-------|
| 6.1 Canviar `repair_batman=False` per defecte a validate_khp_for_alignment | ⏳ |
| 6.2 Afegir opció de reparació a la GUI de revisió (FASE 4) | ⏳ |
| 6.3 Guardar al JSON si s'ha reparat i amb quins paràmetres | ⏳ |
| 6.4 Per mostres no-KHP: reparació només a fase selecció rèpliques | ⏳ |

**Flux desitjat:**
```
Detectar → JSON marca batman=true → Usuari revisa → Decideix reparar → Acció explícita
```

### FASE 5: Tests i Documentació (PARCIAL)

| Tasca | Estat | Data |
|-------|-------|------|
| 5.1 Executar pipeline en 283_SEQ (COLUMN) | ✅ | 2026-01-30 |
| 5.1 Executar pipeline en 284_SEQ_BP (BP + DUAL) | ✅ | 2026-01-30 |
| 5.2 Verificar JSON conté totes les mètriques | ✅ | 2026-01-30 |
| 5.3 Actualitzar CLAUDE.md amb nova estructura | ⏳ |

#### Test 283_SEQ (COLUMN) - 2026-01-30

**Resultat:** ✅ CORRECTE - Tots els camps nous presents

```
Sample: FR2606_283
Comparison:
  pearson: 0.984
  area_diff_pct: 4.0%
  by_fraction:
    BioP: pearson=0.611, diff=32.3%  ⚠️ Correlació baixa
    HS:   pearson=0.953, diff=5.2%
    BB:   pearson=0.999, diff=0.4%
    SB:   pearson=0.998, diff=2.6%
    LMW:  pearson=0.996, diff=1.1%

Replica R1:
  valid: True
  height: 290.8
  t_peak: 20.48 min
  snr: 10466.8
  smoothness: 10.4
  anomaly_score: 30.0
  timeout_info: {n_timeouts: 0, severity: OK}
  peak_indices: {left: 154, max: 307, right: 624}
  baseline_stats: {noise: 0.028}
```

**Observació:** BioP mostra Pearson=0.611 entre rèpliques - possible artefacte a zona inicial que cal investigar.

#### Test 284_SEQ_BP (BP) - 2026-01-30

**Resultat:** ✅ CORRECTE - Tots els camps nous presents

**Correccions necessàries:**
1. `hpsec_replica.py`: Canvi `fit_result.get("valid")` → `fit_result.get("status")` (fit_bigaussian retorna "status" no "valid")
2. `hpsec_pipeline.py`: No passar config global a consolidate_sequence (usa DEFAULT_CONSOLIDATE_CONFIG intern)

```
Sample: FR2606_284_BP
  selected: MIXED
  R1:
    valid: True
    r2: 0.9979
    r2_status: VALID
    height: 357.8
    t_peak: 1.08 min
    snr: 14343.2
    smoothness: 82.67
    anomaly_score: 0.0
    asymmetry: 1.65
    timeout_info: {n_timeouts: 0, severity: OK}
    peak_indices: {left: 3, max: 16, right: 144}
    baseline_stats: {noise: 0.025}
```

**Correcció baseline:** Inicialment baseline_noise=119 mAU (incorrecte). Ara usa últims punts per BP → noise=0.025 mAU, SNR=14343.

---

## 5. MÈTRIQUES D'ÈXIT

- [x] Zero duplicacions de funcions entre mòduls
- [x] JSON conté 100% de les mètriques calculades (verificat 2026-01-30)
- [ ] GUI no fa cap càlcul, només llegeix JSON
- [x] Traçabilitat: Des de concentració final fins a paràmetres de detecció
- [x] Tests passen per COLUMN i BP+DUAL (completat 2026-01-30)

---

## 6. RISCOS

| Risc | Probabilitat | Mitigació |
|------|--------------|-----------|
| Trencar imports existents | ALTA | ✅ Mitigat - tests passen |
| JSON massa gran | MITJANA | Comprimir intervals, guardar només diferències |
| Performance | BAIXA | Les mètriques addicionals són petites |

---

## 7. NOTES IMPORTANTS

**⚠️ REGLA: Sempre consultar aquest document ABANS de fer canvis!**

- hpsec_consolidate.py ÉS DEPRECATED - no usar per nou desenvolupament
- hpsec_import.py conté les funcions d'identificació (is_khp, extract_khp_conc, obtenir_seq)
- hpsec_utils.py conté utilitats matemàtiques (baseline_stats, mode_robust, t_at_max)
- hpsec_core.py conté funcions de detecció (detect_main_peak, detect_batman, detect_timeout)

---

## 8. PLA REFACTOR PIPELINE - Arquitectura 5 Fases (2026-01-30)

### 8.1 Arquitectura Objectiu

```
PIPELINE (orquestra - NO processa)
    │
    ├── Fase 1: hpsec_import.import_sequence()
    │           └── OUTPUT: dades RAW (t, y, DAD, master_info)
    │
    ├── Fase 2: hpsec_calibrate.calibrate_sequence()
    │           └── OUTPUT: shifts, factor, àrea KHP
    │
    ├── Fase 3: hpsec_process.process_sequence()
    │           └── OUTPUT: dades processades (baseline, smooth, àrees)
    │
    ├── Fase 4: hpsec_review.review_sequence()
    │           └── OUTPUT: seleccions, comparacions, warnings
    │
    └── Fase 5: hpsec_reports.export_sequence()
                └── OUTPUT: JSON, Excel, PDF
```

### 8.2 FASE 3: hpsec_process.py

**Estat actual:** ✅ Funcions implementades, ❌ No cridat pel pipeline

**Funcions existents:**
| Funció | Estat | Descripció |
|--------|-------|------------|
| `truncate_chromatogram()` | ✅ | Tallar a temps màxim |
| `get_baseline_correction()` | ✅ | Correcció baseline (BP usa finals) |
| `apply_smoothing()` | ✅ | Suavitzat Savgol |
| `apply_shift()` | ✅ | Aplicar shift temporal |
| `process_dad()` | ✅ | Processar DAD multi-λ |
| `calcular_fraccions_temps()` | ✅ | Àrees per fraccions |
| `calcular_arees_fraccions_complet()` | ✅ | Àrees DOC + DAD |
| `calculate_snr_info()` | ✅ | SNR, LOD, LOQ |
| `process_sample()` | ✅ | Processar 1 mostra |
| `process_sequence()` | ✅ | Processar seqüència completa |

**Input esperat (de calibrate):**
```python
calibration_data = {
    "shift_uib": float,      # Shift per senyal UIB (min)
    "shift_direct": float,   # Shift per senyal Direct (min)
    "factor": float,         # Factor quantificació (ppm/àrea)
    "khp_area": float,       # Àrea KHP referència
    "khp_conc": float,       # Concentració KHP (ppm)
}
```

**Output esperat (per review):**
```python
processed_data = {
    "samples": {
        "FR2606": {
            "R1": {
                "t": array,           # Temps processat
                "y_doc": array,       # DOC net (baseline corregit)
                "y_dad": dict,        # DAD per λ
                "areas": dict,        # Àrees per fracció
                "snr_info": dict,     # SNR, LOD, LOQ
                "peak_info": dict,    # Índexs pic
                "timeout_info": dict, # Timeouts detectats
                "anomalies": list,    # Batman, etc.
            },
            "R2": {...}
        }
    }
}
```

**PENDENT:**
- [ ] Verificar que `process_sequence()` accepta output de `import_sequence()`
- [ ] Verificar mapeig shifts calibrate → process
- [ ] Afegir càlcul concentracions (area × factor)

### 8.3 FASE 4: hpsec_review.py

**Estat actual:** ⚠️ Parcialment implementat

**Funcions existents:**
| Funció | Estat | Descripció |
|--------|-------|------------|
| `review_sequence()` | ❓ Verificar | Funció principal |
| Comparació rèpliques | ⚠️ A hpsec_replica | Cal integrar |
| Selecció millor | ⚠️ A hpsec_replica | Cal integrar |

**Input esperat (de process):**
```python
processed_data  # Output de process_sequence()
```

**Output esperat (per export):**
```python
review_result = {
    "samples": {
        "FR2606": {
            "selected": "R1",           # o "R2", "MIXED", None
            "selected_doc": "R1",
            "selected_dad": "R1",
            "selection_reason": str,
            "confidence": float,
            "comparison": {
                "pearson": float,
                "area_diff_pct": float,
                "by_fraction": dict,    # COLUMN only
            },
            "warnings": list,
            "replicas": {
                "R1": {eval_data},
                "R2": {eval_data},
            }
        }
    }
}
```

**PENDENT:**
- [ ] Verificar existència `review_sequence()`
- [ ] Integrar funcions de hpsec_replica.py
- [ ] Assegurar output compatible amb export

### 8.4 FASE 5: hpsec_reports.py

**Estat actual:** ⚠️ Orientat a PDF, falta JSON/Excel

**Funcions existents:**
| Funció | Estat | Descripció |
|--------|-------|------------|
| `generate_report_*()` | ✅ | Generació PDF |
| `export_sequence()` | ❓ Verificar | Funció principal |
| JSON export | ❌ Falta | processing_summary.json |
| Excel export | ❌ Falta | Resultats finals |

**Input esperat (de review):**
```python
review_result  # Output de review_sequence()
calibration    # Dades calibració
```

**Output esperat:**
- `CHECK/processing_summary.json` - Tot el JSON traçabilitat
- `CHECK/resultats_SEQXX.xlsx` - Excel amb seleccions
- `CHECK/SEQXX_report.pdf` - Informe PDF (opcional)

**PENDENT:**
- [ ] Verificar/crear `export_sequence()`
- [ ] Moure lògica JSON del pipeline actual a reports
- [ ] Moure lògica Excel del pipeline actual a reports

### 8.5 PIPELINE: hpsec_pipeline.py

**Canvis necessaris:**

```python
# ABANS (incorrecte):
def _step_process():
    files = glob("Resultats_Consolidats/*.xlsx")  # Llegeix processats!
    for f in files:
        eval = evaluate_replica(...)  # Avalua, no processa

# DESPRÉS (correcte):
def _step_process():
    return hpsec_process.process_sequence(
        imported_data,      # De Fase 1
        calibration_data    # De Fase 2
    )

def _step_review():
    return hpsec_review.review_sequence(
        processed_data      # De Fase 3
    )

def _step_export():
    return hpsec_reports.export_sequence(
        review_result,      # De Fase 4
        calibration_data
    )
```

### 8.6 Ordre d'Implementació

| Pas | Tasca | Dependència |
|-----|-------|-------------|
| 1 | Verificar `hpsec_process.process_sequence()` | - |
| 2 | Verificar/crear `hpsec_review.review_sequence()` | Pas 1 |
| 3 | Verificar/crear `hpsec_reports.export_sequence()` | Pas 2 |
| 4 | Refactoritzar pipeline per cridar fases | Pas 1-3 |
| 5 | Tests amb 283_SEQ i 284_SEQ_BP | Pas 4 |
| 6 | Eliminar codi duplicat del pipeline | Pas 5 |
