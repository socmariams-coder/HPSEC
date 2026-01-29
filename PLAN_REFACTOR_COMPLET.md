# PLA DE REFACTORITZACIÓ COMPLET - HPSEC Suite

**Data inici:** 2026-01-29
**Última actualització:** 2026-01-29
**Objectiu:** Traçabilitat total + JSON com a font única + Eliminar duplicacions

---

## 0. REGISTRE DE CANVIS

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

## 2. BUITS DE TRAÇABILITAT (PENDENT)

### 2.1 Mètriques Calculades però NO Guardades a JSON

| Mètrica | Calculada a | Propòsit | Estat |
|---------|-------------|----------|-------|
| `peak_indices` (left/right) | hpsec_replica | Límits integració | ❌ NO GUARDAT |
| `baseline_stats` (p10/p90) | hpsec_replica | Distribució soroll | ❌ NOMÉS std |
| `peak_smoothness` | hpsec_core | Irregularitat pic | ❌ MAI EXPORTAT |
| `r2_status` (VALID/CHECK/INVALID) | hpsec_replica | Qualitat fit | ❌ NOMÉS valor R² |
| `anomaly_score` | hpsec_replica | Puntuació combinada | ❌ MAI EXPORTAT |
| `timeout_intervals` | hpsec_core | Intervals exactes | ❌ NOMÉS count |
| `asymmetry_ratio` | hpsec_replica | Sigma ratio | ❌ INTERN |
| `comparison_by_fraction` | hpsec_replica | Pearson per zona | ❌ NOMÉS global |
| `dad_eval` (complet) | hpsec_replica | Totes wavelengths | ❌ NOMÉS 254nm |

### 2.2 Nivell Traçabilitat Actual: **65-70%**

**Reconstructible des de JSON:**
- ✅ Concentracions (area × factor)
- ✅ Seleccions amb motius
- ✅ Històric calibració

**NO Reconstructible:**
- ❌ Per què un pic és "batman"
- ❌ Intervals exactes de timeout
- ❌ Distribució baseline
- ❌ Qualitat fit (VALID/CHECK)

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

### FASE 2: Ampliar JSON per Traçabilitat (PENDENT)

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

### FASE 3: Refactor hpsec_replica per Exportar Tot (PENDENT)

| Tasca | Estat |
|-------|-------|
| 3.1 Modificar evaluate_replica() per retornar TOTES les mètriques | ⏳ |
| 3.2 Modificar compare_replicas() per retornar comparison_by_fraction | ⏳ |
| 3.3 Actualitzar hpsec_pipeline per guardar totes les mètriques noves | ⏳ |

### FASE 4: Validar GUI Llegeix Correctament (PENDENT)

| Tasca | Estat |
|-------|-------|
| 4.1 HPSEC_Suite ha de mostrar les noves mètriques | ⏳ |
| 4.2 Afegir columnes a la taula QC si escau | ⏳ |

### FASE 5: Tests i Documentació (PENDENT)

| Tasca | Estat |
|-------|-------|
| 5.1 Executar pipeline en 3 SEQs (BP, COLUMN, DUAL) | ⏳ |
| 5.2 Verificar JSON conté totes les mètriques | ⏳ |
| 5.3 Actualitzar CLAUDE.md amb nova estructura | ⏳ |

---

## 5. MÈTRIQUES D'ÈXIT

- [x] Zero duplicacions de funcions entre mòduls
- [ ] JSON conté 100% de les mètriques calculades
- [ ] GUI no fa cap càlcul, només llegeix JSON
- [ ] Traçabilitat: Des de concentració final fins a paràmetres de detecció
- [ ] Tests passen per BP, COLUMN, DUAL

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

- hpsec_consolidate.py és una capa de compatibilitat que re-exporta funcions
- hpsec_import.py conté les funcions d'identificació (is_khp, extract_khp_conc, obtenir_seq)
- hpsec_utils.py conté utilitats matemàtiques (baseline_stats, mode_robust, t_at_max)
- hpsec_core.py conté funcions de detecció (detect_main_peak, detect_batman, detect_timeout)
