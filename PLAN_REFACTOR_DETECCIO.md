# Pla de Refactorització: Detecció d'Anomalies HPSEC

**Data**: 2026-01-26
**Versió**: 1.0
**Objectiu**: Unificar la lògica de detecció d'anomalies en un sol mòdul per maximitzar eficiència, robustesa i mantenibilitat.

---

## 1. Estat Actual

### 1.1 Duplicació de Codi

| Funció | hpsec_core.py | hpsec_consolidate.py | hpsec_calibrate.py |
|--------|---------------|----------------------|--------------------|
| `detect_batman()` | ✅ Robust (pic-vall-pic) | ❌ | ⚠️ Simplificat |
| `detect_timeout()` | ❌ | ✅ Robust (dt intervals) | ⚠️ Simplificat (CV) |
| `detect_peak_anomaly()` | ✅ Híbrid | ❌ | ❌ |
| `detect_main_peak()` | ❌ | ❌ | ✅ |
| `detect_all_peaks()` | ❌ | ❌ | ✅ |
| `fit_bigaussian()` | ✅ | ❌ | ❌ |

### 1.2 Problemes Identificats

1. **Duplicació**: Tres implementacions diferents de detecció Batman/timeout
2. **Inconsistència**: Cada mòdul usa criteris lleugerament diferents
3. **Manteniment**: Canvis s'han de fer en múltiples llocs
4. **Testing**: Difícil verificar comportament consistent

---

## 2. Arquitectura Objectiu

### 2.1 Principi: Single Source of Truth

```
hpsec_core.py
├── Detecció d'anomalies (TOTES les funcions)
├── Ajust matemàtic (bi-gaussiana, paràbola)
├── Integració de pics
└── Càlculs estadístics (SNR, Pearson)

hpsec_consolidate.py
├── Lectura/escriptura fitxers
├── Extracció de dades MasterFile
└── import detect_* from hpsec_core

hpsec_calibrate.py
├── Anàlisi KHP (lògica específica)
├── Validació estadística vs històric
├── Gestió CALDATA/KHP_History
└── import detect_* from hpsec_core

DOCtor_BP.py / DOCtor_C.py
├── Workflow de processament
├── Generació de reports
└── import detect_* from hpsec_core
```

### 2.2 Funcions a hpsec_core.py (Versió Final)

```python
# =============================================================================
# DETECCIÓ D'ANOMALIES - FUNCIONS PRINCIPALS
# =============================================================================

def detect_timeout(t_min, threshold_sec=60, major_threshold_sec=70):
    """
    Detecta timeouts basant-se en intervals temporals (dt).

    MÈTODE MÉS ROBUST per detectar pauses del TOC.
    Timeout = interval entre mesures > threshold (defecte 60s)
    Major timeout = recàrrega xeringues (~74s)

    Args:
        t_min: Array de temps en minuts
        threshold_sec: Llindar timeout normal (60s)
        major_threshold_sec: Llindar timeout major (70s)

    Returns:
        dict amb:
            - n_timeouts: int
            - n_major_timeouts: int
            - timeouts: list[dict] amb detalls de cada timeout
            - severity: 'OK'|'INFO'|'WARNING'|'CRITICAL'
            - affected_zones: list[str]
            - total_duration_sec: float (duració total afectada ~2.5-3 min)
    """

def detect_batman(t, y, top_pct=0.20, min_valley_depth=0.01):
    """
    Detecta artefacte Batman (vall al cim del pic).

    CRITERI ESTRICTE: Requereix patró pic-vall-pic.

    Args:
        t, y: Arrays temps i senyal (segment del pic)
        top_pct: Fracció superior a analitzar (0.20 = top 20%)
        min_valley_depth: Profunditat mínima de vall (fracció altura)

    Returns:
        dict amb is_batman, n_valleys, max_depth, valley_info
    """

def detect_peak_anomaly(t, y, top_pct=0.25, min_valley_depth=0.02,
                        smoothness_threshold=70.0):
    """
    Detecció HÍBRIDA: Batman + Smoothness.

    Combina:
    1. Detecció de valls (Batman clar)
    2. Anàlisi de smoothness (anomalies subtils)

    Returns:
        dict amb:
            - is_anomaly: bool
            - anomaly_type: 'BATMAN'|'IRREGULAR'|'OK'
            - smoothness: float (0-100)
            - is_batman: bool
            - is_irregular: bool
    """

def detect_main_peak(t, y, min_prominence_pct=5.0):
    """
    Detecta el pic principal d'un cromatograma.

    Usat per:
    - KHP Column: integrar NOMÉS el pic principal
    - Identificar zona d'interès

    Returns:
        dict amb valid, t_max, peak_idx, left_idx, right_idx, area, height
    """

def detect_all_peaks(t, y, min_prominence_pct=5.0):
    """
    Detecta TOTS els pics significatius.

    Usat per:
    - Validació KHP (no hauria de tenir múltiples pics)
    - Detecció d'interferències

    Returns:
        list[dict] amb info de cada pic
    """

# =============================================================================
# INTEGRACIÓ DE PICS
# =============================================================================

def integrate_chromatogram(t, y, left_idx=None, right_idx=None,
                           baseline=None, mode='full'):
    """
    Integra un cromatograma amb diferents modes.

    Args:
        t, y: Arrays temps i senyal
        left_idx, right_idx: Límits (opcional, per defecte tot)
        baseline: Valor baseline (opcional, calcula si None)
        mode: 'full' | 'main_peak'
            - 'full': Integra tota l'àrea del segment
            - 'main_peak': Integra només el pic principal detectat

    Returns:
        dict amb area, baseline_used, integration_limits, mode
    """
```

---

## 3. Canvis Específics per Mòdul

### 3.1 hpsec_core.py

**AFEGIR** (moure des d'altres mòduls):

```python
# Des de hpsec_consolidate.py (línies 491-646):
TIMEOUT_CONFIG = {...}  # Configuració zones i severitats
def detect_timeout(t_min, threshold_sec=60, major_threshold_sec=70): ...

# Des de hpsec_calibrate.py (línies 192-297):
def detect_main_peak(t, y, min_prominence_pct=5.0): ...
def detect_all_peaks(t, y, min_prominence_pct=5.0): ...

# NOU:
def integrate_chromatogram(t, y, mode='full'): ...
```

**MANTENIR** (ja existents i correctes):
- `detect_batman()` (línies 218-310)
- `detect_peak_anomaly()` (línies 398-453)
- `calc_top_smoothness()` (línies 313-395)
- `fit_bigaussian()` (línies 77-201)
- `repair_with_parabola()` (línies 597-671)

### 3.2 hpsec_consolidate.py

**ELIMINAR**:
- `TIMEOUT_CONFIG` (línies 491-513) → moure a hpsec_core
- `detect_doc_timeouts()` (línies 516-646) → moure a hpsec_core
- `format_timeout_status()` (línies 649-674) → moure a hpsec_core

**AFEGIR** imports:
```python
from hpsec_core import detect_timeout, format_timeout_status
```

**ACTUALITZAR** crides:
- `extract_doc_from_masterfile()`: cridar `detect_timeout()` importat
- `extract_doc_from_master()`: cridar `detect_timeout()` importat

### 3.3 hpsec_calibrate.py

**ELIMINAR**:
- `detect_batman()` local (línies 421-462) → usar de hpsec_core
- `detect_timeout()` local (línies 465-498) → usar de hpsec_core
- `detect_main_peak()` (línies 192-255) → moure a hpsec_core
- `detect_all_peaks()` (línies 258-297) → moure a hpsec_core

**AFEGIR** imports:
```python
from hpsec_core import (
    detect_timeout,
    detect_batman,
    detect_peak_anomaly,
    detect_main_peak,
    detect_all_peaks,
    integrate_chromatogram
)
```

**ACTUALITZAR** `analizar_khp_consolidado()`:
```python
# ABANS (línies 903-904):
has_batman = detect_batman(t_doc, doc_net, config) is not None
has_timeout = detect_timeout(t_doc, doc_net, config, is_bp=is_bp_chromato) is not None

# DESPRÉS:
timeout_info = detect_timeout(t_doc)
has_timeout = timeout_info['n_timeouts'] > 0
timeout_severity = timeout_info['severity']

peak_seg = doc_net[left_idx:right_idx+1]
t_seg = t_doc[left_idx:right_idx+1]
anomaly_info = detect_peak_anomaly(t_seg, peak_seg)
has_batman = anomaly_info['is_batman']
has_irregular = anomaly_info['is_irregular']
smoothness = anomaly_info['smoothness']
```

**AFEGIR** validació KHP específica:
```python
def validate_khp_quality(khp_data, all_peaks, timeout_info, anomaly_info):
    """
    Validació específica per KHP (no aplicable a mostres normals).

    Criteris:
    1. No múltiples pics significatius (>2 pics amb prominència >10%)
    2. No timeout en zona del pic
    3. No batman/irregular
    4. Simetria acceptable (0.5-2.0)
    5. SNR > 50 (KHP ha de ser senyal fort)
    """
    issues = []
    is_valid = True

    # Multi-pic
    if len(all_peaks) > 2:
        issues.append(f"MULTI_PEAK: {len(all_peaks)} pics detectats")
        is_valid = False

    # Timeout
    if timeout_info['severity'] in ['WARNING', 'CRITICAL']:
        issues.append(f"TIMEOUT: {timeout_info['severity']}")
        is_valid = False

    # Anomalia de forma
    if anomaly_info['is_anomaly']:
        issues.append(f"SHAPE: {anomaly_info['anomaly_type']}")
        is_valid = False

    return {'is_valid': is_valid, 'issues': issues}
```

### 3.4 DOCtor_BP.py i DOCtor_C.py

**AFEGIR** imports:
```python
from hpsec_core import detect_timeout, detect_peak_anomaly
```

**ACTUALITZAR** detecció anomalies per usar funcions centralitzades.

---

## 4. Integració KHP: main_peak vs full

### 4.1 Problema

El KHP a Column té un pic únic i definit. Per calibrar correctament:
- **Cal integrar NOMÉS el pic principal** (no baseline ni soroll)
- Les mostres reals tenen múltiples fraccions → integrar TOT

### 4.2 Solució

```python
# A hpsec_calibrate.py, analizar_khp_consolidado():

if is_bp_chromato:
    # BP: pic únic, integrar tot
    integration = integrate_chromatogram(t_doc, doc_net, mode='full')
else:
    # COLUMN: integrar només pic principal
    integration = integrate_chromatogram(t_doc, doc_net, mode='main_peak')

khp_area = integration['area']
```

### 4.3 Implementació `integrate_chromatogram()`

```python
def integrate_chromatogram(t, y, left_idx=None, right_idx=None,
                           baseline=None, mode='full'):
    """
    Integra cromatograma amb mode seleccionable.
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if left_idx is None:
        left_idx = 0
    if right_idx is None:
        right_idx = len(y) - 1

    # Baseline
    if baseline is None:
        baseline = float(np.percentile(y[left_idx:right_idx+1], 10))

    if mode == 'full':
        # Integrar tot el segment
        t_seg = t[left_idx:right_idx+1]
        y_seg = y[left_idx:right_idx+1] - baseline
        y_seg = np.maximum(y_seg, 0)  # No negatius
        area = trapezoid(y_seg, t_seg)

        return {
            'area': float(area),
            'baseline': baseline,
            'left_idx': left_idx,
            'right_idx': right_idx,
            'mode': 'full'
        }

    elif mode == 'main_peak':
        # Detectar i integrar només pic principal
        peak_info = detect_main_peak(t, y)

        if not peak_info['valid']:
            # Fallback a full
            return integrate_chromatogram(t, y, left_idx, right_idx,
                                          baseline, mode='full')

        p_left = peak_info['left_idx']
        p_right = peak_info['right_idx']

        t_peak = t[p_left:p_right+1]
        y_peak = y[p_left:p_right+1] - baseline
        y_peak = np.maximum(y_peak, 0)
        area = trapezoid(y_peak, t_peak)

        return {
            'area': float(area),
            'baseline': baseline,
            'left_idx': p_left,
            'right_idx': p_right,
            'peak_info': peak_info,
            'mode': 'main_peak'
        }
```

---

## 5. Pla d'Execució

### Fase 1: Preparació (Prioritat ALTA)
- [ ] **1.1** Backup complet del codi actual
- [ ] **1.2** Crear branca git `refactor/detection-unification`

### Fase 2: Migració a hpsec_core.py (Prioritat ALTA)
- [ ] **2.1** Moure `TIMEOUT_CONFIG` i `detect_doc_timeouts()` des de hpsec_consolidate.py
- [ ] **2.2** Renombrar a `detect_timeout()` per consistència
- [ ] **2.3** Moure `format_timeout_status()`
- [ ] **2.4** Moure `detect_main_peak()` i `detect_all_peaks()` des de hpsec_calibrate.py
- [ ] **2.5** Crear `integrate_chromatogram()` amb modes 'full' i 'main_peak'
- [ ] **2.6** Afegir exports a `__all__` si existeix

### Fase 3: Actualitzar hpsec_consolidate.py (Prioritat ALTA)
- [ ] **3.1** Eliminar funcions mogudes
- [ ] **3.2** Afegir imports des de hpsec_core
- [ ] **3.3** Actualitzar crides a `detect_timeout()`
- [ ] **3.4** Verificar que consolidació funciona igual

### Fase 4: Actualitzar hpsec_calibrate.py (Prioritat ALTA)
- [ ] **4.1** Eliminar `detect_batman()` i `detect_timeout()` locals
- [ ] **4.2** Eliminar `detect_main_peak()` i `detect_all_peaks()` (moguts)
- [ ] **4.3** Afegir imports des de hpsec_core
- [ ] **4.4** Actualitzar `analizar_khp_consolidado()` per usar noves funcions
- [ ] **4.5** Implementar `validate_khp_quality()` específic
- [ ] **4.6** Actualitzar integració KHP: mode='main_peak' per Column

### Fase 5: Actualitzar DOCtor_BP.py i DOCtor_C.py (Prioritat MITJANA)
- [ ] **5.1** Afegir imports des de hpsec_core
- [ ] **5.2** Substituir deteccions locals per funcions centralitzades
- [ ] **5.3** Verificar comportament consistent

### Fase 6: Testing (Prioritat ALTA)
- [ ] **6.1** Test detect_timeout() amb dades reals (SEQ amb timeouts coneguts)
- [ ] **6.2** Test detect_batman() amb KHP anòmals coneguts
- [ ] **6.3** Test integrate_chromatogram() modes full vs main_peak
- [ ] **6.4** Test complet calibració amb KHP
- [ ] **6.5** Test consolidació amb diverses SEQ

### Fase 7: Documentació (Prioritat MITJANA)
- [ ] **7.1** Actualitzar CLAUDE.md amb nova arquitectura
- [ ] **7.2** Afegir docstrings complets a funcions noves
- [ ] **7.3** Actualitzar TODO_SUITE.txt

---

## 6. Riscos i Mitigacions

| Risc | Probabilitat | Impacte | Mitigació |
|------|--------------|---------|-----------|
| Trencament de funcionalitat existent | Mitjana | Alt | Testing exhaustiu, backup, git branch |
| Incompatibilitat d'arguments | Baixa | Mitjà | Mantenir signatures compatibles |
| Comportament diferent en casos edge | Mitjana | Mitjà | Comparar resultats abans/després |

---

## 7. Mètriques d'Èxit

1. **Zero duplicació**: Cada funció de detecció existeix en UN sol lloc
2. **Consistència**: Mateixos resultats per mateixes dades
3. **Mantenibilitat**: Canvis en detecció es fan en un sol fitxer
4. **Cobertura**: Tots els mòduls usen les funcions centralitzades

---

## 8. Notes Tècniques

### 8.1 Timeout: Duració Total Afectada

Segons l'anàlisi realitzat:
- PRE-timeout: ~0.5 min (transició)
- Timeout: ~1.2 min (74s pausa)
- POST-timeout: ~1.0 min (recuperació)
- **TOTAL: ~2.5-3 min afectats**

La funció `detect_timeout()` ha de retornar aquesta informació per permetre marcar correctament la zona afectada als cromatogrames.

### 8.2 KHP Integració Column

El pic KHP a Column apareix típicament entre 4-8 minuts. La integració `main_peak` ha de:
1. Detectar el pic principal (màxim)
2. Trobar els límits del pic (bases)
3. Integrar només aquesta zona

Això evita incloure:
- Soroll de baseline
- Pics secundaris (si n'hi ha)
- Artefactes

### 8.3 Zones de Severitat Timeout

```python
TIMEOUT_ZONES = {
    "RUN_START": (0, 1),      # Severitat: OK
    "BioP": (0, 18),          # Severitat: WARNING (crític per Column)
    "HS": (18, 23),           # Severitat: CRITICAL (zona més important)
    "BB": (23, 30),           # Severitat: WARNING
    "SB": (30, 40),           # Severitat: WARNING
    "LMW": (40, 70),          # Severitat: INFO
    "POST_RUN": (70, 999),    # Severitat: OK
}
```

---

## 9. Historial de Canvis

| Data | Versió | Descripció |
|------|--------|------------|
| 2026-01-26 | 1.0 | Creació del pla inicial |

