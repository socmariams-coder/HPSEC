# Model Predictiu de Timeouts per Recàrrega de Xeringues TOC

## Paràmetres Verificats

| Paràmetre | Valor | Precisió |
|-----------|-------|----------|
| Cicle recàrrega | **77.2 min** | ±0.03 min |
| Duració mostra HPLC | **78.65 min** | ±0.01 min |
| Deriva per mostra | **1.45 min** | (87 segons) |
| Duració timeout | **74 segons** | - |

## Fórmula de Predicció

Per predir la posició del timeout dins la mostra N:

```
Pos_timeout(N) = T0 - (N-1) × 1.45    (mod 77.2)
```

On:
- **T0** = temps del primer timeout dins la primera mostra (en minuts)
- **N** = número de mostra (1, 2, 3...)
- El resultat és la posició en minuts des de l'inici de la mostra

## Com Determinar T0

T0 depèn de quant temps portava el TOC enregistrant abans de la primera injecció:

```
T0 = (temps_acumulat_TOC mod 77.2)
```

**Exemple pràctic:**
- Si el TOC porta 150 min funcionant quan s'inicia la seqüència
- T0 = 150 mod 77.2 = 72.8 min → timeout a la zona segura (LMW)

## Taula de Referència Ràpida

| T0 (min) | Mostres en zona segura | Primera mostra crítica |
|----------|------------------------|------------------------|
| 77 | 26 | #27 |
| 70 | 21 | #22 |
| 60 | 14 | #15 |
| 50 | 7 | #8 |
| 40 | 0 | #1 (totes crítiques) |
| 30 | 0 | #1 |
| 20 | 0 | #1 |

## Verificació del Model

Testat amb dues seqüències independents:

| Seqüència | T0 | Prediccions | Encerts | Error mitjà |
|-----------|-----|-------------|---------|-------------|
| 272_SEQ | 48.8 min | 29 | 100% | 3 segons |
| 275_SEQ | 20.5 min | 29 | 100% | 3 segons |

## Recomanacions Pràctiques

### 1. Abans d'Iniciar una Seqüència
- Observar quan es produeix un timeout del TOC
- Esperar fins que T0 sigui >60 min (idealment ~77 min)
- Això garanteix que les primeres ~14-25 mostres estaran en zona segura

### 2. Ordenació de Mostres
- Mostres importants: posicions 1-14
- Controls i blancs: posicions 15+
- El timeout deriva 1.45 min per mostra (retrocedeix)

### 3. Seqüències Llargues (>25 mostres)
- Acceptar que algunes mostres tindran timeout en zona crítica
- O dividir en dues seqüències

### 4. Si T0 < 40 min
- TOTES les primeres mostres tindran timeout en zona crítica
- Considerar esperar o acceptar la situació

## Zona Crítica

| Regió | Rang (min) | Impacte |
|-------|------------|---------|
| BioP | 0-18 | CRÍTIC |
| HS | 18-23 | CRÍTIC |
| BB | 23-30 | CRÍTIC |
| SB | 30-40 | CRÍTIC |
| LMW | 40-78 | Segur |

---
*Model verificat: 2025-01-26*
*Precisió: 100% amb error <5 segons*
