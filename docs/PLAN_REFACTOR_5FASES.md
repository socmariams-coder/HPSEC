# Pla de Refactorització: 5 Fases

## Progrés

| Mòdul | Estat | Data | Notes |
|-------|-------|------|-------|
| `hpsec_import.py` | ✅ CREAT | 2026-01-29 | 29 funcions, `import_sequence()` testejat |
| `hpsec_calibrate.py` | ✅ AMPLIAT | 2026-01-29 | v1.3.0 - funcions alineació migrades |
| `hpsec_process.py` | ✅ CREAT | 2026-01-29 | 18 funcions, `process_sequence()` |
| `hpsec_review.py` | ✅ CREAT | 2026-01-29 | v1.1.0 - selecció independent DOC/DAD |
| `hpsec_export.py` | ⬜ PENDENT | - | Crear nou |
| `hpsec_consolidate.py` | 🔄 ACTUALITZAT | 2026-01-29 | v1.10.0 - imports des de hpsec_import + hpsec_process |

---

## Objectiu
Separar `hpsec_consolidate.py` (86 funcions, massa responsabilitats) en mòduls clars seguint el flux de 5 fases.

## Arquitectura Proposada

```
hpsec_core.py          ← JA EXISTEIX (funcions matemàtiques compartides)
hpsec_utils.py         ← JA EXISTEIX (utilitats generals)
hpsec_config.py        ← JA EXISTEIX (configuració)
│
├── hpsec_import.py    ← NOU: Fase 1 - Lectura de dades
├── hpsec_calibrate.py ← EXISTENT: Fase 2 - Calibració KHP (ampliar)
├── hpsec_process.py   ← NOU: Fase 3 - Alineació i càlcul d'àrees
├── hpsec_review.py    ← NOU: Fase 4 - Comparació rèpliques
├── hpsec_export.py    ← NOU: Fase 5 - Escriptura i reports
│
└── hpsec_consolidate.py  ← DEPRECAT: Només wrapper per compatibilitat
```

---

## Assignació de Funcions per Mòdul

### 1. `hpsec_import.py` - FASE IMPORTAR (Lectura de Dades)

**Responsabilitat**: Llegir fitxers font i crear estructura de dades intermèdia.

| Funció | Origen | Notes |
|--------|--------|-------|
| `llegir_doc_uib()` | consolidate | Llegir UIB CSV |
| `llegir_dad_export3d()` | consolidate | Llegir DAD Export3D |
| `llegir_dad_1a()` | consolidate | Llegir DAD1A |
| `llegir_dad_amb_fallback()` | consolidate | Prioritat Export3D → DAD1A |
| `llegir_masterfile_nou()` | consolidate | Llegir nou format master |
| `llegir_master_direct()` | consolidate | Llegir antic format master |
| `detect_master_format()` | consolidate | Detectar format |
| `read_master_date()` | consolidate | Extreure data |
| `extract_doc_from_masterfile()` | consolidate | Extreure segment DOC (nou) |
| `extract_doc_from_master()` | consolidate | Extreure segment DOC (antic) |
| `build_sample_ranges_from_toc_calc()` | consolidate | Construir rangs de mostres |
| `calculate_toc_calc_on_the_fly()` | consolidate | Calcular TOC_CALC si falta |
| `_save_toc_calc_to_masterfile()` | consolidate | Helper per guardar |
| `trobar_excel_mestre()` | consolidate | Trobar master file |
| `list_dad_files()` | consolidate | Llistar fitxers DAD |
| `netejar_nom_uib()` | consolidate | Parsejar nom UIB |
| `get_valid_samples_from_hplc_seq()` | consolidate | Extreure mostres vàlides |
| `match_sample_confidence()` | consolidate | Matching amb confiança |
| `is_sample_in_seq()` | consolidate | Verificar mostra a SEQ |
| `match_uib_to_master()` | consolidate | Match UIB → Master |
| `build_master_sample_index()` | consolidate | Índex de mostres |
| `find_master_row_for_sample()` | consolidate | Trobar fila master |
| `detect_dad_rep_style()` | consolidate | Detectar estil rèplica |
| `dad_sample_rep_from_path()` | consolidate | Parsejar path DAD |
| `target_keys_from_desc()` | consolidate | Generar claus matching |
| `choose_best_candidate()` | consolidate | Seleccionar millor DAD |
| `detect_replica_anomalies()` | consolidate | Detectar errors R12 vs R2 |
| `check_sequence_files()` | consolidate | Verificar fitxers SEQ |
| `is_bp_seq()` | consolidate | Detectar mode BP |

**Nova funció principal**:
```python
def import_sequence(seq_path, config=None, progress_callback=None):
    """
    Fase 1: Importar dades RAW sense processar.

    Returns:
        dict amb:
        - samples: {nom: {uib_data, dad_data, master_info, ...}}
        - khp_samples: llista de mostres KHP trobades
        - master_data: dades del master file
        - mode: "UIB"/"DIRECT"/"DUAL"
        - method: "BP"/"COLUMN"
        - file_check: resultat verificació fitxers
    """
```

---

### 2. `hpsec_calibrate.py` - FASE CALIBRAR ✅ AMPLIAT

**Responsabilitat**: Validació KHP, càlcul shifts, gestió històric.

| Funció | Origen | Estat |
|--------|--------|-------|
| `validate_khp_for_alignment()` | calibrate | ✅ JA EXISTEIX |
| `compare_khp_historical()` | calibrate | ✅ JA EXISTEIX |
| `get_historical_khp_stats()` | calibrate | ✅ JA EXISTEIX |
| `calculate_column_alignment_shifts()` | consolidate | ✅ MIGRAT |
| `find_sibling_alignment()` | consolidate | ✅ MIGRAT |
| `find_nearest_alignment()` | consolidate | ✅ MIGRAT |
| `load_alignment_log()` | consolidate | ✅ MIGRAT |
| `save_alignment_log()` | consolidate | ✅ MIGRAT |
| `add_alignment_entry()` | consolidate | ✅ MIGRAT |
| `get_qaqc_folder()` | consolidate | ✅ MIGRAT |
| `ensure_qaqc_folder()` | consolidate | ✅ MIGRAT |
| `get_alignment_log_path()` | consolidate | ✅ MIGRAT |
| `find_khp_in_folder()` | calibrate | ✅ JA EXISTEIX |
| `find_khp_for_alignment()` | consolidate | ✅ MIGRAT |
| `get_a254_for_alignment()` | consolidate | ✅ MIGRAT |
| `load_khp_history()` | calibrate | ✅ JA EXISTEIX |
| `save_khp_history()` | calibrate | ✅ JA EXISTEIX |
| `calibrate_sequence()` | calibrate | ✅ JA EXISTEIX |

**Nova funció principal**:
```python
def calibrate_sequence(imported_data, config=None, progress_callback=None):
    """
    Fase 2: Validar KHP i calcular alineació.

    Args:
        imported_data: resultat de import_sequence()

    Returns:
        dict amb:
        - khp_validation: {status, issues, metrics, replicas}
        - alignment: {shift_uib, shift_direct, source, khp_file}
        - historical: comparació amb històric
        - ready_to_process: bool (si KHP és vàlid)
    """
```

---

### 3. `hpsec_process.py` - FASE PROCESSAR ✅ CREAT

**Responsabilitat**: Aplicar alineació, calcular àrees, detectar anomalies.

| Funció | Origen | Estat |
|--------|--------|-------|
| `apply_shift()` | consolidate | ✅ MIGRAT |
| `align_signals_by_max()` | consolidate | ✅ MIGRAT |
| `get_baseline_correction()` | consolidate | ✅ MIGRAT |
| `apply_smoothing()` | consolidate | ✅ MIGRAT |
| `truncate_chromatogram()` | consolidate | ✅ MIGRAT |
| `mode_robust()` | consolidate | ✅ MIGRAT |
| `process_dad()` | consolidate | ✅ MIGRAT |
| `find_peak_boundaries()` | consolidate | ✅ MIGRAT |
| `detect_main_peak()` | consolidate | ✅ MIGRAT |
| `calculate_snr_info()` | consolidate | ✅ MIGRAT |
| `analyze_sample_areas()` | consolidate | ✅ MIGRAT |
| `calcular_fraccions_temps()` | consolidate | ✅ MIGRAT |
| `calcular_arees_fraccions_complet()` | consolidate | ✅ MIGRAT |
| `detectar_tmax_senyals()` | consolidate | ✅ MIGRAT |
| `process_sample()` | NOU | ✅ CREAT |
| `process_sequence()` | NOU | ✅ CREAT |
| `calculate_timing_stats()` | consolidate | 🔄 PENDENT |
| `detect_timeout()` | core | ✅ IMPORTAT |
| `detect_batman()` | core | ✅ IMPORTAT |

**Nova funció principal**:
```python
def process_sequence(imported_data, calibration_data, config=None, progress_callback=None):
    """
    Fase 3: Processar mostres amb alineació confirmada.

    Args:
        imported_data: resultat de import_sequence()
        calibration_data: resultat de calibrate_sequence()

    Returns:
        dict amb:
        - samples: {nom: {
            t_aligned, y_aligned,
            areas, fractions,
            anomalies, snr_info, peak_info
        }}
        - timing_stats: estadístiques timing
    """
```

---

### 4. `hpsec_review.py` - FASE REVISAR ✅ CREAT

**Responsabilitat**: Comparar rèpliques, estadístiques, selecció INDEPENDENT DOC/DAD.

| Funció | Origen | Estat |
|--------|--------|-------|
| `compare_replicas()` | replica | ✅ IMPORTAT |
| `evaluate_replica()` | replica | ✅ IMPORTAT |
| `evaluate_dad_multi()` | replica | ✅ IMPORTAT |
| `select_best_replica()` | replica | ✅ IMPORTAT |
| `_select_best_dad()` | NOU | ✅ CREAT |
| `collect_sample_stats()` | NOU | ✅ CREAT |
| `review_sample()` | NOU | ✅ CREAT |
| `review_sequence()` | NOU | ✅ CREAT |
| `apply_selections()` | NOU | ✅ CREAT |
| `_average_replicas()` | NOU | ✅ CREAT |
| `_average_dad()` | NOU | ✅ CREAT |
| `get_review_summary_text()` | NOU | ✅ CREAT |

**CARACTERÍSTICA PRINCIPAL: Selecció Independent DOC/DAD**

```python
def review_sample(replicas, method="COLUMN", config=None):
    """
    Revisar una mostra amb selecció INDEPENDENT DOC i DAD.

    Pot recomanar R1 per DOC però R2 per DAD si la qualitat és diferent.

    Returns:
        dict amb:
        - selected_doc: "R1"/"R2"/"AVERAGE"/None (selecció per DOC)
        - selected_dad: "R1"/"R2"/"AVERAGE"/None (selecció per DAD)
        - selected: "R1"/"R2"/"MIXED"/None (resum - MIXED si DOC!=DAD)
        - comparison: {pearson, area_diff, ...}
        - evaluations: avaluacions DOC per rèplica
        - dad_evaluations: avaluacions DAD per rèplica
        - recommendation_doc: raó selecció DOC
        - recommendation_dad: raó selecció DAD
    """

def apply_selections(review_data, selections, processed_data):
    """
    Aplicar seleccions amb suport independent DOC/DAD.

    Args:
        selections:
            - Simple: {nom: "R1"/"R2"/"AVERAGE"/"DISCARD"}
            - Independent: {nom: {"doc": "R1", "dad": "R2"}}

    Returns:
        dict amb:
        - samples: {nom: {data_doc, data_dad, source_doc, source_dad, is_mixed}}
        - mixed_samples: mostres amb DOC!=DAD
    """
```

---

### 5. `hpsec_export.py` - FASE EXPORTAR (NOU)

**Responsabilitat**: Escriure fitxers finals i generar reports.

| Funció | Origen | Notes |
|--------|--------|-------|
| `write_consolidated_excel()` | consolidate | MOURE AQUÍ |
| `save_file_check_report()` | consolidate | MOURE AQUÍ |
| `apply_renames()` | consolidate | MOURE AQUÍ |
| `generate_consolidation_summary()` | consolidate | MOURE AQUÍ |
| `save_consolidation_summary()` | consolidate | MOURE AQUÍ |

**Noves funcions**:
```python
def export_sequence(review_data, selections, output_path, config=None, options=None):
    """
    Exportar resultats finals.

    Args:
        options: {excel: bool, pdf: bool, copy_selected: bool}

    Returns:
        dict amb:
        - files: llista de fitxers generats
        - summary_path: path al JSON resum
    """

def export_excel_summary(data, path):
    """Generar Excel resum amb seleccions i concentracions."""

def export_pdf_report(data, path):
    """Generar informe PDF professional."""
```

---

### 6. `hpsec_utils.py` - UTILITATS COMPARTIDES (Ampliar)

| Funció | Origen | Notes |
|--------|--------|-------|
| `normalize_key()` | consolidate | MOURE AQUÍ |
| `normalize_rep()` | consolidate | MOURE AQUÍ |
| `split_sample_rep()` | consolidate | MOURE AQUÍ |
| `clean_sample_name()` | consolidate | MOURE AQUÍ |
| `skip_sample_direct()` | consolidate | MOURE AQUÍ |
| `obtenir_seq()` | consolidate | MOURE AQUÍ |
| `seq_tag()` | consolidate | MOURE AQUÍ |
| `mode_robust()` | consolidate | MOURE AQUÍ |
| `is_khp()` | consolidate | MOURE AQUÍ |
| `is_control_injection()` | consolidate | MOURE AQUÍ |

---

## Estratègia de Migració

### Fase 1: Preparació (sense trencar res)
1. Crear nous mòduls buits amb imports
2. Copiar funcions als nous mòduls (sense esborrar de consolidate)
3. Afegir `# DEPRECATED: usar hpsec_xxx.func()` als originals

### Fase 2: Actualitzar Imports
1. Actualitzar `hpsec_consolidate.py` per importar dels nous mòduls
2. Mantenir exports per compatibilitat:
   ```python
   # hpsec_consolidate.py
   from hpsec_import import llegir_doc_uib, llegir_dad_export3d, ...
   from hpsec_process import apply_shift, get_baseline_correction, ...
   # etc.
   ```

### Fase 3: Nova Funció Principal
1. Crear `consolidate_sequence_v2()` que usa el flux de 5 fases
2. Mantenir `consolidate_sequence()` com a wrapper

### Fase 4: Actualitzar Suite
1. Actualitzar `HPSEC_Suite_v3.py` per usar les noves funcions
2. Implementar interfície de 5 passos

### Fase 5: Neteja
1. Esborrar funcions duplicades de consolidate
2. Deixar consolidate només com a wrapper de compatibilitat

---

## Estructura de Dades Intermèdies

### Després de IMPORTAR
```python
imported_data = {
    "seq_path": "/path/to/SEQ",
    "mode": "DUAL",  # UIB/DIRECT/DUAL
    "method": "COLUMN",  # BP/COLUMN
    "master_data": {...},
    "samples": {
        "MOSTRA_001": {
            "replicas": {
                "1": {
                    "uib": {"t": [...], "y": [...], "file": "..."},
                    "direct": {"t": [...], "y": [...], "file": "..."},
                    "dad": {"df": DataFrame, "file": "..."},
                    "master_info": {"row_start": 10, "row_end": 50, ...}
                },
                "2": {...}
            }
        },
        ...
    },
    "khp_samples": ["KHP50_001", "KHP50_002"],
    "file_check": {...}
}
```

### Després de CALIBRAR
```python
calibration_data = {
    "khp_validation": {
        "status": "VALID",
        "issues": [],
        "metrics": {"area": 123.4, "snr": 45.2, ...},
        "replicas": [{"file": "...", "area": 120}, {"file": "...", "area": 126}]
    },
    "alignment": {
        "shift_uib": 0.15,
        "shift_direct": 0.08,
        "source": "LOCAL",  # LOCAL/SIBLING/HISTORIC
        "khp_file": "KHP50_001_R1"
    },
    "historical": {
        "status": "OK",
        "deviation_pct": 5.2
    },
    "ready_to_process": True
}
```

### Després de PROCESSAR
```python
processed_data = {
    "samples": {
        "MOSTRA_001": {
            "replicas": {
                "1": {
                    "t_aligned": [...],
                    "y_doc_net": [...],
                    "y_doc_uib_net": [...],
                    "df_dad_aligned": DataFrame,
                    "areas": {"total": 45.2, "BioP": 12.1, "HS": 8.5, ...},
                    "peak_info": {"t_max": 21.3, "height": 120, ...},
                    "snr_info": {"snr": 35.2, "lod": 2.1, ...},
                    "anomalies": ["TIMEOUT_BB"]
                },
                "2": {...}
            }
        }
    },
    "timing_stats": {...}
}
```

### Després de REVISAR
```python
review_data = {
    "samples": {
        "MOSTRA_001": {
            "comparison": {"pearson": 0.995, "area_diff_pct": 3.2},
            "recommendation": "R1",
            "issues": [],
            "status": "OK"
        },
        "MOSTRA_002": {
            "comparison": {"pearson": 0.971, "area_diff_pct": 15.1},
            "recommendation": "R2",
            "issues": ["BATMAN_R1"],
            "status": "CHECK"
        }
    },
    "summary": {"ok": 45, "check": 3, "fail": 1},
    "alerts": ["MOSTRA_002", "MOSTRA_015"]
}
```

---

## Checklist de Funcionalitats a Preservar

- [ ] Lectura UIB (UTF-16, UTF-8, tab-separated)
- [ ] Lectura DAD Export3D i DAD1A amb fallback
- [ ] Lectura MasterFile nou i antic format
- [ ] Càlcul TOC_CALC on-the-fly si falta
- [ ] Matching mostres UIB ↔ Master amb confiança
- [ ] Detecció mode BP vs COLUMN
- [ ] Detecció mode UIB/DIRECT/DUAL
- [ ] Validació KHP (Batman, Timeout, Històric)
- [ ] Alineació amb sibling/històric si no hi ha KHP local
- [ ] Aplicació shifts temporals
- [ ] Correcció baseline percentil
- [ ] Suavitzat Savitzky-Golay
- [ ] Detecció timeouts (dt > 60s)
- [ ] Detecció Batman peaks
- [ ] Càlcul àrees per fraccions (BioP, HS, BB, SB, LMW)
- [ ] Càlcul SNR, LOD, LOQ
- [ ] Escriptura Excel consolidat (ID, TMAX, AREAS, DOC, DAD)
- [ ] Generació consolidation.json
- [ ] Report File_Check amb orfes i anomalies
- [ ] Històric d'alineacions (Alignment_History.json)
- [ ] Històric KHP (khp_history.json)
- [ ] Timing stats per planificació seqüències

---

## Prioritat d'Implementació

1. **Alta**: `hpsec_import.py` - Base per tot el flux
2. **Alta**: `hpsec_process.py` - Càlculs principals
3. **Mitjana**: `hpsec_export.py` - Escriptura fitxers
4. **Mitjana**: Ampliar `hpsec_calibrate.py` - Funcions alineació
5. **Baixa**: `hpsec_review.py` - Ja existeix parcialment a hpsec_replica.py

---

## Temps Estimat

| Tasca | Complexitat | Risc |
|-------|-------------|------|
| Crear hpsec_import.py | Alta | Mitjà |
| Crear hpsec_process.py | Alta | Mitjà |
| Ampliar hpsec_calibrate.py | Mitjana | Baix |
| Crear hpsec_export.py | Mitjana | Baix |
| Crear hpsec_review.py | Baixa | Baix |
| Actualitzar Suite | Alta | Alt |
| Testing complet | Alta | - |

**Total**: Tasca significativa que requereix atenció al detall per no perdre funcionalitats.
