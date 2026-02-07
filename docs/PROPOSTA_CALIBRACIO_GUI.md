# PROPOSTA: Quality Scoring per Calibració KHP

**ESTAT: IMPLEMENTAT** (2026-02-01)

## Resum

Actualització de la GUI de calibració (`calibrate_panel.py`) amb nova lògica de qualitat basada en dades empíriques de 98 rèpliques KHP.

---

## 1. COLUMNES DE LA TAULA (14 columnes)

| # | Nom | Descripció | Font de dades |
|---|-----|-----------|---------------|
| 0 | Rep | Rèplica (R1, R2) | `filename` |
| 1 | Senyal | Direct / UIB | calculat |
| 2 | Àrea | Àrea integrada (mAU·min) | `area` |
| 3 | FWHM | Full Width Half Max (min) | `fwhm_doc` (nou) |
| 4 | RF_V | Response Factor/Vol | `rf_v_doc` (nou) |
| 5 | CR | Concentration Ratio | `cr_doc` (nou) |
| 6 | t_max | Temps retenció (min) | `peak_info.t_max` |
| 7 | Shift | Desplaçament vs referència (s) | `shift_sec` |
| 8 | SNR | Signal-to-Noise | `snr` |
| 9 | Sym | Simetria | `symmetry` |
| 10 | **Pic_J** | Pic amb vall (antic Batman) | `has_batman` |
| 11 | TO | Timeout detectat | `has_timeout`, `timeout_info` |
| 12 | Pics | Pics en zona ±4 min | `all_peaks` (nou) |
| 13 | Q | Quality Score | calculat |
| 14 | Estat | VALID/CHECK/INVALID | calculat |

**Canvi**: Columna "Bat" → "Pic_J"
**Nou**: Columna "Pics" per múltiples pics

---

## 2. QUALITY SCORE - Nova lògica

### Penalitzacions (Q=0 perfecte, Q>100 INVALID)

| Criteri | Condició | Penalització | Justificació |
|---------|----------|--------------|--------------|
| **Pic_J (Batman)** | `has_batman=True` | **+100** | INVALID - pic amb artefacte |
| **Timeout AFECTA pic** | TO dins zona ±2.5min de t_max | **+100** | INVALID - dades corrompudes |
| **Múltiples pics** | >1 pic a ±4min de t_max | **+100** | INVALID - senyal no KHP |
| Timeout fora pic | TO però no afecta t_max | **0** | OK - no afecta integració |
| FWHM elevat | >1.5 min | +20 | WARNING - pic ample |
| Shift elevat (Direct) | >50 s | +10 | INFO |
| Shift elevat (UIB) | >30 s | +10 | INFO |
| SNR baix | <10 | +20 | WARNING - senyal feble |
| Asimetria | <0.5 o >2.5 | +10 | INFO |
| CR baix (COLUMN) | <0.40 | +10 | INFO - altres pics |
| CR baix (BP) | <0.95 | +10 | INFO - no esperat |

### Estat final

| Quality Score | Estat | Color |
|---------------|-------|-------|
| Q ≤ 20 | **OK** | Verd |
| 20 < Q ≤ 50 | **INFO** | Groc |
| 50 < Q ≤ 100 | **CHECK** | Taronja |
| Q > 100 | **INVALID** | Vermell |

---

## 3. DETECCIÓ MÚLTIPLES PICS

Utilitzant dades ja disponibles a `calibration_result.json`:

```python
def count_peaks_in_zone(khp, zone_min=4.0):
    """
    Compta pics dins de ±zone_min del pic principal.

    Returns: nombre de pics en la zona (1 = normal, >1 = múltiples)
    """
    peak_info = khp.get('peak_info', {})
    t_max = peak_info.get('t_max', 0)
    all_peaks = khp.get('all_peaks', [])

    if t_max <= 0 or not all_peaks:
        return 1  # Sense info, assumim OK

    count = 0
    for peak in all_peaks:
        t_peak = peak.get('t', 0)
        if abs(t_peak - t_max) <= zone_min:
            count += 1

    return count
```

**Dades disponibles** (`all_peaks` de 283_SEQ):
```json
"all_peaks": [
  {"idx": 311, "t": 20.73, "height": 583.095, "prominence": 583.2}
]
```

---

## 4. IMPLEMENTACIÓ - Passos

### Pas 1: Actualitzar capçalera taula
```python
self.metrics_table.setColumnCount(15)  # +1 per "Pics"
self.metrics_table.setHorizontalHeaderLabels([
    "Rep", "Senyal", "Àrea", "FWHM", "RF_V", "CR",
    "t_max", "Shift", "SNR", "Sym", "Pic_J", "TO", "Pics", "Q", "Estat"
])
```

### Pas 2: Nova funció `_calculate_quality_score()`
```python
def _calculate_quality_score(self, khp, signal='Direct'):
    """Calcula quality score amb nova lògica."""
    score = 0
    issues = []
    is_bp = khp.get('is_bp', False)

    # === CRITÈRI INVALIDANTS (+100) ===

    # Pic_J (Batman)
    if khp.get('has_batman', False):
        score += 100
        issues.append("Pic_J: pic amb vall")

    # Múltiples pics en zona ±4 min
    n_pics = self._count_peaks_in_zone(khp, zone_min=4.0)
    if n_pics > 1:
        score += 100
        issues.append(f"Múltiples pics: {n_pics} en zona ±4min")

    # Timeout afecta pic
    if khp.get('has_timeout', False):
        if self._timeout_affects_peak(khp):
            score += 100
            issues.append("Timeout afecta pic principal")
        # else: no penalitza (fora pic)

    # === WARNINGS (+20) ===

    fwhm = khp.get('fwhm_doc', 0)
    if fwhm > 1.5:
        score += 20
        issues.append(f"FWHM elevat: {fwhm:.2f} min")

    snr = khp.get('snr', 0)
    if 0 < snr < 10:
        score += 20
        issues.append(f"SNR baix: {snr:.1f}")

    # === INFO (+10) ===

    shift_sec = khp.get('shift_sec', 0)
    shift_limit = 30 if signal == 'UIB' else 50
    if abs(shift_sec) > shift_limit:
        score += 10
        issues.append(f"Shift: {shift_sec:.0f}s (>{shift_limit}s)")

    sym = khp.get('symmetry', 1.0)
    if sym > 0 and (sym < 0.5 or sym > 2.5):
        score += 10
        issues.append(f"Asimetria: {sym:.2f}")

    cr = khp.get('cr_doc', 0)
    if cr > 0:
        if is_bp and cr < 0.95:
            score += 10
            issues.append(f"CR baix BP: {cr:.2f}")
        elif not is_bp and cr < 0.40:
            score += 10
            issues.append(f"CR baix: {cr:.2f}")

    return score, issues
```

### Pas 3: Actualitzar `_update_metrics_table()`
- Afegir columna "Pics" (col 12)
- Moure "Q" a col 13, "Estat" a col 14
- Usar nova funció `_calculate_quality_score()`

---

## 5. BACKEND - Estat de les mètriques

A `hpsec_calibrate.py`, ja existeixen totes les mètriques:

| Mètrica GUI | Clau backend | Estat |
|-------------|--------------|-------|
| FWHM | `fwhm_doc` | ✅ Ja existeix |
| RF_V | `rf_v_doc` | ✅ Ja existeix |
| CR | `concentration_ratio` | ✅ Ja existeix (mapejar a GUI) |
| Pics | `all_peaks`, `all_peaks_count` | ✅ Ja existeix |

**Nota**: La GUI espera `cr_doc` però el backend usa `concentration_ratio`. Cal mapejar a `_update_metrics_table()`:
```python
cr = khp.get('concentration_ratio', khp.get('cr_doc', 0))
```

---

## 6. IMPLEMENTACIÓ COMPLETADA

### Fase 1 (GUI) - ✅ COMPLETAT

**Fitxer**: `gui/widgets/calibrate_panel.py`

**Canvis realitzats**:

1. **Taula actualitzada a 15 columnes**:
   ```
   Rep | Senyal | Àrea | FWHM | RF_V | CR | t_max | Shift | SNR | Sym | Pic_J | TO | Pics | Q | Estat
   ```

2. **Noves funcions afegides**:
   - `_count_peaks_in_zone(khp, zone_min=4.0)` - Compta pics a ±4min del principal
   - `_timeout_affects_peak(khp)` - Determina si timeout afecta el pic
   - `_calculate_quality_score(khp, signal)` - Nova lògica de qualitat

3. **Lògica Quality Score**:
   - Pic_J (Batman): +100 → INVALID
   - Múltiples pics (>1 a ±4min): +100 → INVALID
   - Timeout afecta pic: +100 → INVALID
   - Timeout fora pic: 0 (OK)
   - FWHM >1.5 min: +20
   - SNR <10: +20
   - Shift elevat: +10
   - Asimetria: +10
   - CR baix: +10

4. **Columna TO**:
   - Vermell si afecta pic (+100)
   - Gris si fora pic (OK, no penalitza)

5. **CR mapeig**: `concentration_ratio` → GUI

### Fase 2 (Backend) - NO NECESSÀRIA

Totes les mètriques (`fwhm_doc`, `rf_v_doc`, `concentration_ratio`, `all_peaks`) ja existien al backend.
