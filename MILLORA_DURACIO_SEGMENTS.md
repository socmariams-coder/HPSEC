# Millora Deteccio Pics Amorfes - Validacio Duracio Segments

## Concepte Original (Usuari)

En lloc de només canviar el threshold d'angles, validar que els angles es produeixin entre segments lineals **prou llargs**. Similar a la deteccio STOP/TIMEOUT que valida duracions (0.8-3.5 min).

Idea: Un pic veritablement amorf ha de tenir SEGMENTS LINEALS LLARGS amb angles bruscos entre ells, no simplement angles en segments petits (que podrien ser soroll).

## Implementacio

### Canvis a `detect_amorphous()` (DOCtor_C.py):

1. **Nou parametre**: `min_segment_duration` (default: 0.08 min)

2. **Calcul de duracio per cada segment**:
   - Cada segment lineal (R² ≥ 0.98) ara te `duration = t_end - t_start`

3. **Validacio d'angles**:
   - ABANS: Comptava TOTS els angles >45° entre segments lineals consecutius
   - ARA: Nomes compta angles si AMBDOS segments adjacents tenen duracio ≥ min_segment_duration

4. **Nous camps retornats**:
   - `n_angle_changes`: Angles VALIDS (entre segments llargs)
   - `n_total_angles`: Angles TOTALS (sense validar duracio)
   - `max_angle`: Maxim angle VALID
   - `max_angle_total`: Maxim angle TOTAL
   - `segments`: Llista de segments lineals amb durada

### Constant de configuracio (DOCtor_C.py, linia 100-102):

```python
AMORPHOUS_MIN_SEGMENT_DURATION = 0.08  # min
```

### Calibracio del threshold:

- **0.5 min**: Massa estricte - filtra TOTS els angles (fins i tot en pics amorfes reals)
- **0.15 min**: Encara massa alt - perd 2/3 pics amorfes reals
- **0.08 min**: OPTIM - Detecta tots els pics amorfes reals ✓
- **0.05 min**: Gairebe equivalent a no filtrar

**Per que 0.08 min?**
- Amb `segment_size=5` punts i submostreig 1:5, els segments tenen ~0.1-0.3 min
- 0.08 min filtra segments MOLT curts (soroll/artifacts) sense eliminar segments reals
- Pics amorfes coneguts tenen segments de 0.10-0.28 min

## Resultats

### Test amb pics amorfes coneguts (test_segment_duration_validation.py):

| Fitxer | Esperat | Detectat | Valid/Total | Max Angle | Resultat |
|--------|---------|----------|-------------|-----------|----------|
| EX1L2503_258_R1 | AMORF | SI | 1/1 | 61.7° | OK ✓ |
| EP_269B_R1 | AMORF | SI | 2/2 | 87.5° | OK ✓ |
| FR2606_283_R1 | AMORF | SI | 3/3 | 89.8° | OK ✓ |

**Conclusio**: Tots els pics amorfes coneguts es detecten correctament amb threshold 0.08 min.

### Test reduccio falsos positius (test_false_positives_reduction.py):

**Falsos positius (NO haurien de detectar-se)**:
- EX10L2503_257_R1: ENCARA detectat (1/1 angles valids, 85°) ❌
- EX10L2506_266_R1: ENCARA detectat (4/4 angles valids, 88°) ❌
- FR2349_272_R1: CORREGIT - ja no detecta ✓

**Veritables positius (han de detectar-se)**:
- EX1L2503_258_R1: OK ✓
- EP_269B_R1: OK ✓
- FR2606_283_R1: OK ✓

**Resum**:
- Falsos positius corregits: 1/3 (33%)
- Veritables positius mantinguts: 3/3 (100%)

## Limitacions

La validacio de duracio de segments **millora** la deteccio pero **no elimina tots els falsos positius**:

1. **Problema**: Els 2 falsos positius restants (EX10L2503_257, EX10L2506_266) tenen tots els angles validats (1/1 i 4/4). Aixo significa que:
   - Els seus segments lineals SÓN prou llargs (≥0.08 min)
   - Els angles SÓN grans (85-88°)
   - Pero el pic NO es realment amorf (es una corba Gaussiana suau)

2. **Per que passa?**: Pics Gaussians suaus poden tenir alguns segments que localment semblen lineals amb angles grans, especialment:
   - A la pujada/baixada rapida del pic
   - Quan hi ha soroll o irregularitats menors
   - Quan el submostreig 1:5 simplifica la corba

## Millores Adicionals Suggerides

Per reduir encara mes els falsos positius (sense perdre veritables positius):

### Opcio 1: Augmentar threshold angle (45° → 60-65°)
- Mes restrictiu: nomes angles MOLT bruscos
- Risc: podria perdre alguns pics amorfes amb angles moderats

### Opcio 2: Validar CONSECUTIVITAT de segments lineals
- Pics amorfes reals tenen SEQUENCIES de segments lineals (escala)
- Pics Gaussians tenen segments lineals ISOLATS
- Implementacio: Comptar "runs" de segments lineals consecutius

### Opcio 3: Combinar amb altres metriques
- Requerir: (angles valids >2) OR (smoothness <15%) OR (asimetria extrema)
- Mes robust: diverses evidencies d'anomalia

### Opcio 4: Augmentar min_segment_duration (0.08 → 0.12-0.15 min)
- Mes restrictiu
- Risc: EP_269B i FR2606 tenen segments de 0.107 min (just al limit)

## Recomanacio

La millora implementada es **CORRECTA I UTIL** pero **PARCIAL**:

✅ **Mantenir** la validacio de duracio de segments (elimina 1/3 FP sense perdre TP)

**Per millora adicional**, suggereixo:
1. Augmentar `AMORPHOUS_MIN_SEGMENT_DURATION` a 0.10 min (filtrara alguns FP mes)
2. Augmentar `min_angle_deg` de 45° a 55-60° (angles mes estrictes)
3. Revisar adaptative threshold: augmentar min_angles requerit per pics mitjans/llargs

Aquests canvis combinats podrien eliminar els 2 FP restants mantenint els 3 TP.

## Fitxers Modificats

1. `DOCtor_C.py`:
   - Linia 100-102: Nova constant `AMORPHOUS_MIN_SEGMENT_DURATION`
   - Linia 1109-1226: Funcio `detect_amorphous()` actualitzada
   - Linia 910: Crida amb parametre `min_segment_duration`

2. Fitxers de test creats:
   - `test_segment_duration_validation.py`: Test amb pics amorfes coneguts
   - `test_false_positives_reduction.py`: Test reduccio falsos positius

## Exemple de Sortida

```
Dur.min    Detectat   Valid    Total    MaxValid   MaxTotal
--------------------------------------------------------------
0.0        SI         1        1        61.7       61.7
0.08       SI         1        1        61.7       61.7      <-- USAT
0.15       NO         0        1        0.0        61.7
```

Mostra com el threshold 0.08 manté la deteccio mentre thresholds mes alts la bloquegen.
