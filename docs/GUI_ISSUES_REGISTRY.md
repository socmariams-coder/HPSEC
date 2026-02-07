# HPSEC Suite - Registre d'Incidències GUI

**Creat:** 2026-02-05
**Última actualització:** 2026-02-05

---

## PRINCIPIS GENERALS (Aplicar a TOTES les pantalles)

| ID | Principi | Estat |
|----|----------|-------|
| G01 | Mateixa estructura visual tant si s'importa de zero com des de JSON/manifest | ARREGLAT |
| G02 | Avisos centralitzats en botó únic (no escampats) | ARREGLAT |
| G03 | Botons amb format consistent (CommonToolbar) | ARREGLAT |
| G04 | Botó "Afegir notes" sempre visible a totes les pantalles | ARREGLAT |
| G05 | Botó "OK avisos" ha d'incloure qui els ha revisat (traçabilitat) | ARREGLAT |
| G06 | Columna "Estat/Status" per indicar accions necessàries | ARREGLAT (I08) |

---

## IMPORTAR (Fase 1)

### Estructura i Navegació
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| I01 | Text "prem executar" → canviar per "prem Importar" | BAIXA | ARREGLAT |
| I02 | Botons duplicats "Importar Importar" abans de Següent - unificar | ALTA | ARREGLAT |
| I03 | Botó "Següent" no s'activa quan s'importa de zero | CRÍTICA | ARREGLAT |
| I04 | Estructura diferent quan s'importa des de JSON (botons revisar, confirmar reimportar) | ALTA | ARREGLAT |
| I05 | Pantalla ha de ser IDÈNTICA independentment de la font (zero o JSON) | ALTA | ARREGLAT |

### Suggeriments i Orfes
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| I06 | "1 suggeriment" - no es veu quin és | MITJANA | ARREGLAT |
| I07 | CSV orfes: cal ordenar i veure quins estan buits | MITJANA | ARREGLAT |
| I08 | Columnes Estat UIB i Estat DAD separades per indicar validació/assignació | MITJANA | ARREGLAT |
| I09 | Tots els avisos a "Revisar avisos", no escampats per pantalla | ALTA | ARREGLAT |

### Persistència de Dades
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| I10 | CSV assignats manualment NO es guarden - cal repetir cada vegada | CRÍTICA | ARREGLAT |
| I11 | Avís d'orfes persisteix quan ja s'han assignat | ALTA | ARREGLAT |
| I12 | Botó "Següent" no s'habilita després d'assignacions | ALTA | ARREGLAT |

### DAD/254nm
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| I13 | Quan no hi ha Export3D, no carrega 254 des de MasterFile 3-DAD_KHP | ALTA | ARREGLAT |
| I14 | Fallback a CSV DAD no implementat | ALTA | ARREGLAT |

---

## CALIBRAR (Fase 2)

### Estructura i Navegació
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C01 | Botons duplicats "calibrar calibrar" abans de Següent | ALTA | ARREGLAT |
| C02 | Selector de calibracions perdut quan n'hi ha dues | ALTA | ARREGLAT |
| C03 | Falta resum - va directa al gràfic | MITJANA | REVISAR-UI |

### Gràfics i Càlculs
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C04 | Calibració sense dades 254nm (ni gràfics ni taules) | ALTA | ARREGLAT (via I13) |
| C05 | No hi ha càlcul de bigaussiana (necessari per BP) | ALTA | ARREGLAT |
| C05b | Bigaussian fit no es mostrava al gràfic (només es calculava) | ALTA | ARREGLAT |
| C06 | Històric: falta gràfic de barres (evolució temporal calibracions) | MITJANA | VERIFICAR (ja existeix) |

### Selecció de Rèpliques
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C07 | Columna "Seleccionada" → canviar a "Status" | BAIXA | ARREGLAT |
| C08 | Permetre canviar manualment a "no conforme" amb dropdown | MITJANA | ARREGLAT |
| C09 | Eliminar columna "Outlier" (botó apareix després de taula) | BAIXA | ARREGLAT |

### Navegació
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C15 | Després calibració salta directament a Anàlisi (no deixa revisar) | ALTA | ARREGLAT |

### Històric i Gràfics
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C16 | Botó "?" a històric no feia res - ara mostra llegenda en diàleg | BAIXA | ARREGLAT |
| C17 | Investigar: per què seqüència 284 no es marca com outlier | MITJANA | ARREGLAT |

### Validació i Avisos
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C10 | "Problemes de qualitat" → moure a avisos superiors | ALTA | VERIFICAR (ja inclòs a warnings bar) |
| C11 | Avisos simetria/smoothness irrellevants per BP | MITJANA | ARREGLAT |
| C12 | Molts avisos de desviació a taula històric - origen desconegut | MITJANA | ARREGLAT |

### Errors Tècnics
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| C13 | Error `'>' not supported between NoneType and int` | CRÍTICA | ARREGLAT |
| C14 | Comparacions amb valors None a `_process_khp_group` | CRÍTICA | ARREGLAT |

---

## ANALITZAR (Fase 3)

### Estructura i Navegació
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| A01 | Falta botó "Afegir notes" (viola G04) | MITJANA | ARREGLAT |
| A02 | Barra avisos sense botons "Revisar" / "OK ✓" (viola G05) | ALTA | ARREGLAT |
| A03 | Estructura consistent amb warnings_bar, info_frame, taules, navegació | - | OK |

### Taula de Resultats
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| A07 | Columna "Mostra" ocupa massa espai, amaga altres columnes | ALTA | ARREGLAT |
| A08 | No hi ha dades de UIB a la taula DOC | ALTA | ARREGLAT |
| A09 | Nomenclatura "_b1" estranya per identificar rèpliques | MITJANA | ARREGLAT (simplificat a _R1/_R2) |
| A10 | Botó "Detall" hauria de ser columna o doble-click, no botó inferior | MITJANA | ARREGLAT |

### Funcionalitats
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| A04 | Selector DOC/DAD funcional amb toggle buttons | - | OK |
| A05 | Taules amb selector rèplica (dropdown) i columna Estat | - | OK |
| A06 | Diàleg detall amb gràfics i estadístiques | - | OK |

### Propagació Calibració → Anàlisi
| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| A11 | RF i shift no es mostraven explícitament | ALTA | ARREGLAT |
| A12 | shift_direct no s'actualitzava quan canviava calibració | CRÍTICA | ARREGLAT |
| A13 | Factor 100 a seq 286 no canvia amb calibració - investigar | ALTA | ARREGLAT |
| A14 | Múltiples calibracions (per inj_volume) no es propagaven correctament | ALTA | ARREGLAT |

---

## CONSOLIDAR (Fase 4)

| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| CO01 | (Pendent revisió detallada) | - | PENDENT |

---

## EXPORTAR (Fase 5)

| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| E01 | (Pendent revisió detallada) | - | PENDENT |

---

## DASHBOARD

| ID | Descripció | Prioritat | Estat |
|----|------------|-----------|-------|
| D01 | Dashboard revisat - sense canvis necessaris | - | OK |

---

## HISTÒRIC DE CANVIS

| Data | IDs Afectats | Descripció |
|------|--------------|------------|
| 2026-02-05 | C13, C14 | Arreglat error NoneType comparison a calibració |
| 2026-02-05 | I11, I12 | Arreglat persistència avisos orfes i botó Següent |
| 2026-02-05 | I14 | Afegit fallback CSV DAD |
| 2026-02-05 | I13 | En curs: fallback MasterFile 3-DAD_KHP |
| 2026-02-05 | I03 | Arreglat: emetre import_completed signal |
| 2026-02-05 | I04, I05 | Arreglat: UI consistent entre import fresc i des de manifest |
| 2026-02-05 | I06-I10 | Arreglat: suggeriments visibles, orfes ordenats, avisos centralitzats, auto-save |
| 2026-02-05 | C01 | Arreglat: mateix fix que I02 (task_indicator text) |
| 2026-02-05 | C05 | Arreglat: afegit càlcul bigaussiana a calibració |
| 2026-02-05 | C07, C09 | Arreglat: columna Status, eliminada columna Outlier |
| 2026-02-05 | C11 | Arreglat: filtrat avisos simetria/smoothness per BP |
| 2026-02-05 | A01, A02 | Arreglat: botó Afegir notes i botons Revisar/OK a warnings_bar |
| 2026-02-05 | I-NEW | Arreglat: avisos persistents quan es carrega des de JSON - _show_warnings_bar no respectava _warnings_confirmed |
| 2026-02-05 | I-NEW2 | Arreglat: confirmar suggeriments no guardava warnings_confirmed=True al manifest |
| 2026-02-05 | C08 | Arreglat: columna Status ara és dropdown (Vàlida/Outlier), eliminat botó extern |
| 2026-02-05 | C09 | Arreglat: corregit accés a columna 9 (no existia) → columna 8 |
| 2026-02-05 | C15 | Arreglat: wizard no salta automàticament a Anàlisi després de calibrar |
| 2026-02-05 | C16 | Arreglat: botó "?" ara mostra diàleg amb llegenda completa del gràfic |
| 2026-02-05 | C05b | Arreglat: bigaussian fit ara es mostra al gràfic (línia discontínua verda/taronja) |
| 2026-02-05 | A07 | Arreglat: columnes taula amb ResizeToContents (com Import/Calibrar) |
| 2026-02-05 | A10 | Arreglat: botó Detall amagat, doble-clic a taula funciona + tooltip |
| 2026-02-05 | A11 | Arreglat: RF i shift ara es mostren explícitament a info_frame amb tooltip |
| 2026-02-05 | A12 | Arreglat: _apply_selected_calibration ara actualitza shift_direct correctament |
| 2026-02-05 | I13 | Arreglat: format clau 3-DAD_KHP corregit a {SAMPLE}_{INJ}_R{REP} (e.g., KHP5_1_R1) |
| 2026-02-05 | C02 | Arreglat: selector de condicions ara sempre s'actualitza al mostrar el panel |
| 2026-02-05 | C04 | Arreglat: via I13 - ara les dades 254nm es carreguen correctament |
| 2026-02-05 | A13, A14 | Arreglat: múltiples calibracions per inj_volume - cada mostra usa la calibració que coincideix amb el seu volum d'injecció (50µL→BP_50_2, 100µL→BP_100_2) |
| 2026-02-05 | A09 | Arreglat: Sample_Rep simplificat a NOM_R1/R2 (sense _B1/_B2), el sufix _50 ja distingeix volums |
| 2026-02-05 | A08 | Arreglat: areas_uib ara es calcula sempre (DUAL o només UIB) amb flag is_uib_only |
| 2026-02-05 | C12, C17 | Arreglat: thresholds estrictes (>20% INVALID) i exclude_outliers=True per comparació històrica |
| 2026-02-05 | C12, C17 | Regenerat històric KHP amb update_calibration_validation - quality_issues actualitzats |
| 2026-02-05 | I08 | Arreglat: Columnes Estat UIB i Estat DAD separades (13 columnes en mode DUAL) |
| 2026-02-05 | G01-G06 | Implementat CommonToolbar: estructura unificada, avisos centralitzats, notes, traçabilitat |

---

## NOTES

- **CRÍTICA**: Bloqueja funcionalitat bàsica
- **ALTA**: Afecta experiència d'usuari significativament
- **MITJANA**: Millora desitjable
- **BAIXA**: Cosmètic o menor

---

## PROPOSTA ESTRUCTURA UNIFICADA (Totes les pantalles)

```
+------------------------------------------+
| BARRA AVISOS (si n'hi ha)                |
| [!] Missatge avís    [Revisar] [OK ✓ JM] |
+------------------------------------------+
| CONTINGUT PRINCIPAL                       |
| - Taula/Gràfics/Formularis               |
| - Columna ESTAT per indicar accions      |
+------------------------------------------+
| BARRA INFERIOR                           |
| [Afegir Notes] [Guardar]     [< Ant] [Seg >] |
+------------------------------------------+
```
