# Proposta: Integrar validació KHP al pipeline de consolidació

## Problema actual

El pipeline de `hpsec_consolidate.py` calcula els shifts d'alineament usant el KHP
sense verificar la seva qualitat. Si el KHP té problemes (com 272_SEQ amb RATIO_LOW),
els shifts calculats podrien ser incorrectes i afectar TOTES les mostres de la SEQ.

## Flux actual (sense verificació)

```
hpsec_consolidate.py
    ├── Detectar KHP local o sibling
    ├── Calcular shifts: t_max(A254) - t_max(DOC)
    └── Aplicar shifts a TOTES les mostres
```

## Flux proposat (amb verificació)

```
hpsec_consolidate.py
    ├── Detectar KHP local o sibling
    ├── ** VERIFICAR QUALITAT KHP **
    │   ├── Ratio A254/DOC dins rang [0.015, 0.060]
    │   ├── Intensitat dins rang del grup (±2x)
    │   ├── Sense timeout a zona HS (18-23 min)
    │   └── Pic identificable (no INSUFFICIENT_DATA)
    │
    ├── Si KHP VALID:
    │   └── Calcular i aplicar shifts normalment
    │
    └── Si KHP INVALID:
        ├── WARNING: "KHP {nom} té qualitat baixa: {motiu}"
        ├── Buscar KHP alternatiu (sibling, històric)
        └── O: Alinear per màxim de cada mostra (com BP)
```

## Implementació

### Opció A: Funció de validació ràpida integrada

Afegir a `hpsec_consolidate.py`:

```python
def validate_khp_for_alignment(khp_path, intensity_stats=None):
    """
    Validació ràpida del KHP abans d'usar-lo per alineament.

    Criteris crítics (invalidants):
    - RATIO_LOW: ratio A254/DOC < 0.015
    - TIMEOUT_CRITICAL: timeout a zona HS (18-23 min)
    - NO_PEAK: no es pot identificar el pic

    Returns:
        dict amb 'valid': bool, 'issues': list, 'shifts': dict
    """
    # ... implementació
```

### Opció B: Cridar analyze_khp_full.py

Importar les funcions de validació:

```python
from analyze_khp_full import (
    analyze_khp_complete,
    calculate_intensity_stats,
    check_intensity_anomaly
)

# Abans de calcular shifts:
khp_result = analyze_khp_complete(khp_path)
if not khp_result['valid']:
    # WARNING i buscar alternativa
```

## Criteris de validació per alineament

| Criteri | Threshold | Acció si falla |
|---------|-----------|----------------|
| RATIO_LOW | < 0.015 | REBUTJAR |
| TIMEOUT_CRITICAL | a zona HS | REBUTJAR |
| INTENSITY_HIGH | > 5x mean | WARNING |
| NO_PEAK | - | REBUTJAR |
| R2_LOW | < 0.90 | WARNING (no rebutjar) |

**Nota**: Per alineament, només necessitem que el pic sigui identificable.
R2_LOW o CR_LOW no invaliden per alineament (el màxim segueix sent correcte).

## Casos específics detectats

| SEQ | KHP | Problema | Impacte alineament |
|-----|-----|----------|-------------------|
| 271_SEQ_BP | KHP2_271_BP | RATIO_LOW + INTENSITY_HIGH | BP no usa KHP - OK |
| 272_SEQ | KHP2_272 | RATIO_LOW (0.012) | **CRÍTIC** - shifts incorrectes |
| 266_SEQ | KHP5_266 | TIMEOUT_CRITICAL | **CRÍTIC** - pic afectat |
| 257_SEQ | KHP1_257 | INSUFFICIENT_DATA | **CRÍTIC** - no hi ha dades |

## Recomanació

Implementar **Opció A** (funció integrada) per:
1. Mínima dependència externa
2. Validació específica per alineament (menys estricta que validació completa)
3. Feedback immediat durant consolidació

La funció hauria de:
1. Llegir el KHP
2. Verificar que el pic és identificable
3. Verificar ratio A254/DOC
4. Verificar que no hi ha timeout a zona HS
5. Retornar els shifts si és vàlid, o None + warnings si no ho és
