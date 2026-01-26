# Informe: Anàlisi de Timeouts per Recàrrega de Xeringues TOC

## Resum Executiu

L'analitzador TOC (Sievers M9e) fa una recàrrega de xeringues cada **76 minuts** (1140 mesures), causant un timeout de **74 segons**. Com que les mostres COLUMN duren **78.6 minuts**, el timeout "deriva" cap enrere ~2.6 min per cada mostra consecutiva.

**Impacte:**
- Mostres 1-14: timeout a zona segura (LMW, >40 min)
- Mostres 15+: timeout progressivament a zona crítica (0-40 min)
- ~53% de mostres en seqüències llargues tindran timeout en zona crítica

## Dades Clau

| Paràmetre | Valor |
|-----------|-------|
| Cicle recàrrega xeringa | 76.0 min (1140 mesures) |
| Duració timeout | 74 segons |
| Duració mostra COLUMN | 78.6 min |
| Zona crítica (integració) | 0-40 min |
| Diferència mostra-cicle | +2.6 min |

## Patró de Deriva

```
Mostra  1: timeout a t=76.0 min -> LMW (segur)
Mostra  2: timeout a t=73.4 min -> LMW (segur)
...
Mostra 14: timeout a t=42.2 min -> LMW (segur)
Mostra 15: timeout a t=39.6 min -> SB (CRÍTIC)
Mostra 16: timeout a t=37.0 min -> SB (CRÍTIC)
...
Mostra 24: timeout a t=16.2 min -> BioP (CRÍTIC)
...
Mostra 30: timeout a t=0.6 min -> BioP (CRÍTIC)
```

## Recomanacions per Tècnics

### 1. Ordre de Mostres (RECOMANAT)
Posar les mostres més importants a les **posicions 1-14** de la seqüència:
- Mostres amb resultats crítics
- Mostres amb pocs replicats
- Mostres difícils de re-analitzar

Posar al final (posicions 15+):
- Blancs i controls
- Mostres amb múltiples replicats
- Mostres menys crítiques

### 2. Seqüències Llargues (>15 mostres)
Opcions:
- **Dividir** en dues seqüències de ~14 mostres
- **Acceptar** que ~50% tindran timeout en zona crítica
- **Intercalar** un blank cada 14 mostres per "resincronitzar"

### 3. Sincronització d'Inici
Si és possible, iniciar la seqüència just després d'una recàrrega natural:
- Observar quan es produeix el timeout
- Esperar 2-3 minuts i iniciar
- Així el primer timeout caurà al final de la mostra 1

### 4. Modificació del Temps d'Anàlisi (Ideal però complex)
Si es pogués reduir el temps d'anàlisi HPLC a <76 min:
- El timeout sempre cauria al final o fora de la mostra
- Eliminaria completament el problema

## Conclusió

El timeout de recàrrega és inevitable (~74s cada 76 min), però el seu impacte es pot minimitzar:

1. **Acció immediata**: Reordenar mostres (importants primer)
2. **Seqüències curtes**: Limitar a 14 mostres per seqüència
3. **Acceptar variabilitat**: En seqüències llargues, ~50% de mostres tindran timeout en zona crítica

## Fitxers Generats

- `comparacio_direct_uib_complet.pdf` - Comparació Direct vs UIB per totes les mostres
- `comparacio_direct_uib_complet_metrics.csv` - Mètriques (Pearson, RMSE, àrees)
- `analisi_timeout_recarrega.png` - Gràfic del patró de deriva
- `timeout_patterns_analysis.png` - Anàlisi de patrons
- `timeout_by_concentration.png` - Anàlisi per concentració

---
*Generat: 2025-01-26*
