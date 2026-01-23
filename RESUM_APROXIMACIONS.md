# RESUM: APROXIMACIONS PER DETECTAR ARTEFACTES UIB

## CONTEXT
Els artefactes UIB són causats per aturades del detector TOC/DOC durant càrrega de xeringues. La UIB continua enviant senyals creant patrons anòmals al cromatograma.

---

## 1. DETECCIÓ STOP/TIMEOUT (IMPLEMENTADA)

### Mètode:
- Divideix senyal en **chunks de 0.1 min** (TIMEOUT_FINE_RES)
- Calcula **pendent** de cada chunk
- Agrupa chunks amb pendent baixa en "illes" (mesetes)
- Valida mesetes amb múltiples criteris

### Criteris de validació:
```python
TIMEOUT_MAX_CV = 0.1%           # CV < 0.1% = meseta totalment plana
TIMEOUT_MIN_DUR = 0.8 min       # Duració mínima
TIMEOUT_MAX_DUR = 3.5 min       # Duració màxima
TIMEOUT_MIN_EDGE_SLOPE = 6.0    # Pendent mínima als bordes
TIMEOUT_MIN_CONTRAST = 3.0      # Ratio bord/meseta
```

### Avantatges:
✓ Detecta **mesetes planes** (parades completes)
✓ CV < 0.1% molt estricte → alta especificitat
✓ Valida bordes (pendents) per evitar falsos positius
✓ Duració controlada (0.8-3.5 min)

### Limitacions:
✗ NO detecta artefactes **NO plans** (rampes, angles, zig-zag)
✗ Només funciona per parades totals del detector
✗ Requereix contrast alt als bordes

---

## 2. DETECCIÓ AMORPHOUS (ANGLES) - IMPLEMENTADA

### Mètode:
- Divideix pic en **segments de 5 punts** solapats
- Calcula **regressió lineal** (R² > 0.98) per cada segment
- Calcula **angles** entre segments lineals consecutius
- Valida angles amb múltiples filtres

### Criteris de validació:
```python
segment_size = 5 punts          # Mida segments
min_r2 = 0.98                   # Linearitat mínima
min_angle_deg = 45°             # Angle mínim significatiu
min_segment_duration = 0.08 min # Duració mínima segment (PROBLEMA!)
CV_THRESHOLD = 20%              # CV màxim per validar angle
```

### Filtres adicionals:
- Single extreme angle (>80°) → rebutjar (artifact numèric)
- Too many angles (>3) → rebutjar (sobre-segmentació)
- Angles concentrated at top (>70%) → rebutjar (artifact al cim)
- Ramp detection: >= 2 segments CV < 5% → amorphous tipus rampa

### Avantatges:
✓ Detecta patrons **NO plans** (zig-zag, escales, rampes)
✓ Captura transicions **abruptes** (angles >45°)
✓ Dual detection: angles bruscos **O** rampes CV baix
✓ Filtres múltiples per reduir falsos positius
✓ Independ de pendent absoluta (detecta angles relatius)

### Limitacions (ACTUAL):
✗ **segment_size=5 massa petit** → duracions fixes (0.107 o 0.267 min)
✗ **min_segment_duration=0.08 massa baix** → angles espuris en gaussianes amples
✗ Sobre-segmentació de corbes suaus crea angles "falsos"
✗ Sensibilitat 100% però especificitat 0% (detecta tot)

---

## 3. DERIVADES (NO IMPLEMENTADA) - ALTERNATIVA POTENCIAL

### Primera derivada (dy/dt):
- Detecta **canvis de pendent**
- Pics = canvis bruscos de velocitat
- Problema: Sensible a soroll

### Segona derivada (d²y/dt²):
- Detecta **canvis de curvatura**
- Pic negatiu = punta convexa (pic gaussià)
- Pic positiu = vall convexa
- Zero = segment recte
- Avantatge: Detecta directament canvis d'angle sense segments
- Problema: Molt sensible a soroll, requereix suavitzat fort

### Comparació amb angles:
| Criteri | Angles (actual) | Derivades |
|---------|----------------|-----------|
| Robustesa soroll | Bona (R² filtra soroll) | Dolenta (requereix suavitzat) |
| Interpretabilitat | Alta (graus) | Baixa (unitats arbitràries) |
| Validació segments | Sí (durada, CV) | No directe |
| Càlcul | Moderat | Ràpid |

---

## PROBLEMA ACTUAL: DURACIONS FIXES

### Per què totes les duracions són 0.107 o 0.267?

**CAUSA**: Sampling rate + subsampling

```python
# detect_amorphous() línia 1171-1177
if len(t) > 200:
    step = 5
    t_work = t[::step]  # Submostrar cada 5 punts

# Línia 1192: segments solapats
for i in range(0, len(t_work) - segment_size, segment_size // 2):
    t_seg = t_work[i:i+segment_size]  # 5 punts
```

Si el **sampling rate original** és constant per fitxer:
- FR fitxers: 0.0214 min/punt → segment 5 punts = 0.107 min
- PTLL fitxers: 0.0534 min/punt → segment 5 punts = 0.267 min

### Conseqüència:
- **Segments massa curts** per distingir rampes reals de gaussianes amples
- Tots els segments del mateix fitxer tenen duració idèntica
- No podem usar durada com a criteri discriminatori

---

## SOLUCIÓ PROPOSADA

### Opció 1: AUGMENTAR segment_size
```python
segment_size = 10-15 punts  # En lloc de 5
```
**Efecte**: Segments 2-3x més llargs → duracions ~0.2-0.4 min (FR) o ~0.5-0.8 min (PTLL)
**Avantatge**: Captura millor rampes llargues vs corbes curtes
**Risc**: Pot perdre angles petits (menos resolució)

### Opció 2: AUGMENTAR min_segment_duration
```python
min_segment_duration = 0.15-0.20 min  # En lloc de 0.08
```
**Efecte**: Només valida angles entre segments llargs
**Avantatge**: Elimina angles espuris de sobre-segmentació
**Risc**: Perd angles reals en fitxers amb sampling baix (FR: 0.107 max)

### Opció 3: COMBINAR segment_size + min_segment_duration
```python
segment_size = 10 punts
min_segment_duration = 0.15 min
```
**Efecte**: Segments més llargs + validació estricta
**Avantatge**: Millor discriminació TP vs FP
**Risc**: Cal validar que detecta tots els amorphous reals

### Opció 4: CRITERI ADAPTATIU
```python
# Detectar sampling rate del fitxer
dt_median = np.median(np.diff(t))
# Ajustar segment_size per tenir ~0.2-0.3 min per segment
target_duration = 0.25  # min
segment_size = max(5, int(target_duration / dt_median))
```
**Efecte**: Duracions consistents independentment del sampling
**Avantatge**: Robust a diferents tipus de fitxers
**Risc**: Més complex d'implementar

---

## AVANTATGES APROXIMACIÓ ANGLES vs ALTERNATIVES

### Per què angles (segments lineals) i NO derivades?

1. **Robustesa**: R² > 0.98 filtra soroll naturalment
2. **Interpretabilitat**: Angle en graus és intuïtiu
3. **Validació rica**: Podem validar durada, CV, posició de cada segment
4. **Dual detection**: Angles bruscos (transicions) + rampes CV baix (continus)
5. **Filtres específics**: TOP concentration, single extreme, too many angles

### Per què angles i NO només CV (com STOP)?

1. **Generalitat**: STOP només detecta CV < 0.1% (mesetes planes)
2. **Angles detecten**: Zig-zag, escales, rampes, transicions
3. **Complementarietat**: STOP (planes) + Amorphous (no planes)

---

## RECOMANACIÓ IMMEDIATA

**TESTEJAR Opció 3**: Augmentar segment_size + min_segment_duration

```python
# Canviar a DOCtor_C.py línia 1132
def detect_amorphous(t, y, segment_size=10, min_r2=0.98,
                     min_angle_deg=45, min_angle_changes=3,
                     min_segment_duration=0.15):
```

**Objectiu**:
- Segments 2x més llargs → ~0.2 min (FR) o ~0.5 min (PTLL)
- Validació min_segment_duration=0.15 → rebutja segments curts
- Mantenir sensibilitat 100% amorphous reals
- Millorar especificitat (reduir FP gaussianes amples)

**Test**: Executar test_amorphous_validation.py amb nous paràmetres
