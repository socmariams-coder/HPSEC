<!-- Auditoria de robustesa HPSEC — generada 2026-06-30 -->
<!-- 51 troballes verificades · 8 debilitats sistemiques · 15 agents -->

# Informe de robustesa — HPSEC Suite

## 1. Resum executiu

El codi és funcional i els camins feliços generen resultats correctes, però la robustesa està compromesa per un patró transversal dominant: **les fallades es degraden en silenci en lloc de bloquejar o avisar**. Excepcions empassades (`except Exception: pass/continue/debug`), valors per defecte que substitueixen dades absents sense marca (volum d'injecció 100/400 µL, sensibilitat UIB 700 ppb) i resultats parcials reportats com a èxit fan que un operador pugui creure que té dades vàlides quan no és així. El risc de més impacte directe és el **volum d'injecció assumit**, perquè és divisor directe de la ppm i s'escala sense cap senyal a la GUI (reconegut com a PENDING al propi projecte). El segon eix de fragilitat és l'**estat en memòria enverinat**: els mutadors de calibració modifiquen l'objecte cachejat *abans* d'escriure i només invaliden el cache si l'escriptura té èxit, de manera que una escriptura fallida (fitxer bloquejat per OneDrive/Excel) deixa el procés operant amb una calibració fantasma mai persistida. Hi ha també **duplicació de lògica que divergeix** (semàfors, prioritat de net delay, pre_margin, auto-correcció de columnes) i **propagació trencada de camps clau** (`uib_sensitivity`, `inj_volume_source`, `warnings_structured`) que avui són majoritàriament latents però activaran bugs silenciosos en quantificació quan UIB es calibri per sensibilitat. Finalment, l'`if True:` que reescriu incondicionalment `4-TOC_CALC` pot sobreescriure correccions manuals amb un delay de fallback. La majoria de troballes són acotades o recuperables per reimportació, però la combinació silenci + sobreescriptura no atòmica + cache enverinat configura un perfil de risc on els errors no es veuen fins després.

## 2. Debilitats sistèmiques

### S1 — Fallades silencioses (excepcions empassades i defaults que amaguen)
És el patró més estès. Apareix com a `except Exception` que fa `pass`/`continue`/`logger.debug` i com a defaults que substitueixen dades absents sense flag estructurat ni anomalia a la barra d'avisos:
- `quantify_sample` assumeix volum 100/400 µL (`hpsec_analyze.py:1542-1546`) — **impacte directe en ppm**.
- Sensibilitat UIB cau a 700 ppb (`hpsec_analyze.py:2362-2363`).
- Reassignació BP DAD254 falla → `success=True` amb finestres per defecte (`hpsec_import.py:4817-4820`).
- `load_manifest` empassa corrupció i retorna `None` com si fos "primera importació" (`hpsec_import.py:5420-5427`).
- `ensure_data_loaded` retorna en silenci deixant `data_deferred=True`; el worker calibra igualment (`hpsec_import.py:6165-6181`).
- Lectura de config `toc_pre_margin_min` (`hpsec_import.py:2430-2434`), heurístic de volum (`1834-1835`), auto-fix de columnes (`1763-1764`), timestamps de delay (`hpsec_delay.py:136-137,177-178`).

**Per què importa:** el sistema produeix dades quantificades presentades com a vàlides quan en realitat s'han usat suposicions o lectures degradades. L'operador no té com saber-ho. La política documentada del projecte ("no suposar res sobre el volum") es viola a la pràctica.

### S2 — Estat en memòria enverinat (cache mutat in-place abans d'escriure)
`load_calibration_reference` / `load_local_calibrations` / `load_khp_history` retornen el cache **per referència**, i els mutadors (`add_calibration`, `register_calibration`) modifiquen aquest objecte *abans* del `save`, que només invalida el cache *dins* del `try` després d'un `json.dump` reeixit (`hpsec_calibrate.py:890/943/972/977; 395-402; 4008-4052; 2584-2592; 2783-2791`).

**Per què importa:** una escriptura fallida (lock SharePoint/OneDrive) deixa `_cal_ref_cache` apuntant a estructures mutades (activa antiga desactivada, nova cal inserida) mentre el disc no ha canviat i el `mtime` és igual → el següent `load` retorna estat fantasma. Les quantificacions de la sessió usen una calibració mai persistida i, en reiniciar, tot revertesix silenciosament. És el patró de més gravetat perquè afecta el nucli de quantificació.

### S3 — Resultat parcial/buit presentat com a complet
El criteri d'èxit ignora la cobertura:
- Exportació: `success = n_errors==0`, ignora `n_skipped` (`hpsec_export.py:1054-1055`).
- Importació de siblings: carpetes que fallen només es loguen; `import_completed` emet `success:True` amb avisos només del primary (`gui/widgets/import_panel/panel.py:440-482`).
- Camí preloaded emet sense `warnings_structured` → el wizard pinta la fase Importar com 'ok' ignorant blockers reals (`import_panel/panel.py:339-345`).

**Per què importa:** l'usuari creu que té totes les injeccions/exportacions quan part del dataset ha desaparegut sense avís persistent.

### S4 — Propagació trencada de camps riu avall
Valors llegits correctament a la importació es perden en construir un nou `result` o en serialitzar:
- `uib_sensitivity` es perd a `analyze_sample` → `quantify_sample` el rep `None` (`hpsec_analyze.py:1754-1762` vs `1554`); tampoc se serialitza a `analysis_result.json` (`3682-3727`); ni es rellegeix de `0-INFO` al recarregar manifest (`hpsec_import.py:5610,5644-5671`).
- `inj_volume_source` no sobreviu al round-trip del manifest (`hpsec_import.py:5322-5331` vs `5765`).

**Per què importa:** trenca el disseny de routing de calibració UIB v3.0 (sensibilitat 700/1000). Avui és latent (UIB encara no calibrat), però quan coexisteixin cal UIB 700 i 1000 el sistema misrutarà silenciosament cap a la sensibilitat equivocada.

### S5 — Duplicació de lògica que divergeix
La mateixa decisió s'implementa dues vegades amb resultats diferents:
- Semàfors: `_populate_row_estat` calcula `worst` però `_update_row_state` recrea el DOC sense recalcular-lo (`import_panel/panel.py:1189-1250` vs `1824-1928`) → ordenació incorrecta després d'editar una cel·la.
- Prioritat de net delay: `compute_toc_calc` agafa l'última clau sense prioritat; `read_current_delay` prioritza Suite>B12 (`hpsec_import.py:2381-2389` vs `hpsec_delay.py:51-71`).
- `pre_margin`: delay-tool fix 1.5 vs import que llegeix config (`hpsec_delay.py` vs `hpsec_import.py:2429-2434`).
- Auto-correcció de columnes G-L només a `parse_injections`, no a `compute_toc_calc` (`hpsec_import.py:1742-1764` vs `2287-2316`).

**Per què importa:** dos mòduls poden aplicar delays diferents a la mateixa seqüència, o l'usuari valida un mapatge que l'import després descarta. Cap font única de veritat.

### S6 — Lectura posicional / índexs hardcoded
Dades llegides per posició física en lloc de per capçalera/etiqueta:
- Volum d'injecció a `col_list[13]` (`hpsec_migrate_master.py:336-352` + `hpsec_import.py:1811-1835`).
- `uib_sensitivity` a `iloc[4,1]` (B5) sense verificar etiqueta (`hpsec_import.py:578-588`).
- DOC a `col[3]`/`col[5]` (`hpsec_import.py:694-703`).
- Senyal TOC sense guard `sig_col is None` (`hpsec_import.py:685-692`).

**Per què importa:** mateixa classe de fragilitat que les columnes desplaçades — si l'export Agilent/Sievers insereix o reordena una columna/fila, l'índex apunta a una altra magnitud i es desa silenciosament. Avui protegit només pel layout fix del migrate.

### S7 — Escriptura no atòmica (overwrite del fitxer bo abans de validar)
- `save_import_manifest` obre el fitxer definitiu en mode `'w'` sense temp+rename (`hpsec_import.py:5392-5395`).
- `_save_toc_calc_to_masterfile` esborra el full `4-TOC_CALC` abans de validar el `save` (`hpsec_import.py:2233-2265`).
- `import_sequence` reescriu `4-TOC_CALC` incondicionalment (`if True:`) inclús amb delay FALLBACK (`hpsec_import.py:4355-4376` + `2409-2421`).

**Per què importa:** una interrupció a mig camí deixa el fitxer bo invàlid; un delay de fallback sobreescriu correccions manuals amb un warning enganyós ("no trobat" quan en realitat s'ha recalculat).

## 3. Troballes per categoria

### Severitat ALTA

| Severitat | Troballa | file:line | Fix concret |
|---|---|---|---|
| high | `quantify_sample` assumeix volum 100/400 µL quan falta; només `logger.warning`, sense flag ni anomalia | `hpsec_analyze.py:1542-1546` | Marcar `result['volume_source']='assumed'` + `create_anomaly` WARNING/BLOCKER; opcionalment `ppm=None` fins confirmació de l'usuari |
| high | Saves muten l'objecte cachejat abans d'escriure i només invaliden el cache si l'escriptura té èxit → cache enverinat | `hpsec_calibrate.py:890/943/972/977; 395-402; 4008-4052; 2584-2592; 2783-2791` | Invalidar el cache SEMPRE (en `finally` o a l'inici de `save_*`), o fer que `load_*()` retorni `deepcopy` i substituir el cache només després d'un save reeixit |
| high | Auto-correcció columnes G-L només a `parse_injections`, no a `compute_toc_calc` → DOC Direct perdut silenciosament | `hpsec_import.py:1742-1764` vs `2287-2316` | Centralitzar la normalització dins `llegir_masterfile_nou` i escriure el df corregit a `master_data['hplc_seq']`, o reaplicar el guard `is_none` de manera consistent |
| high | `import_sequence` reescriu `4-TOC_CALC` incondicionalment (`if True:`) inclús amb delay FALLBACK; warning enganyós | `hpsec_import.py:4355-4376` + `2409-2421` | Quan `net_delay_source=='FALLBACK'` no desar i emetre blocker. Corregir el text 'no trobat' → 'recalculat' |
| high | El slider de delay BP escriu l'OFFSET com a delay absolut → corromp el Net delay del MasterFile (~7.67→0.3) | `gui/widgets/calibrate_panel/panel.py:2671-2672, 2926-2948` | `new_delay = (self._bp_delay_original or 0) + self._bp_delay_current`; actualitzar el text del diàleg a old→old+offset. (Backup ja és default; no cal afegir-lo) |
| high | Bloc de codi mort 'deprecated' (~44 línies) després d'un `return` en producció | `gui/widgets/analyze_panel/panel.py:2684-2730` | Esborrar 2685-2729; deixar la funció buida amb la docstring de no-op. L'historial viu a git |

### Severitat MITJANA

| Severitat | Troballa | file:line | Fix concret |
|---|---|---|---|
| medium | Escriptura del manifest no atòmica (mode `'w'`, sense temp+rename) | `hpsec_import.py:5392-5395` | Escriure a `output_path+'.tmp'` amb `json.dump(..., default=str)` i `os.replace(tmp, output_path)`; en error esborrar `.tmp` i conservar el manifest anterior |
| medium | `load_manifest` empassa la corrupció i retorna `None` (corromput tractat com 'no existeix') | `hpsec_import.py:5420-5427` | Capturar `json.JSONDecodeError` concret, `logger.warning` amb path, reanomenar a `.corrupt` i senyalitzar a la UI |
| medium | Sensibilitat UIB cau a 700 ppb per defecte sense avís | `hpsec_analyze.py:2362-2363` | Comprovar explícitament `None`; si falta, no estimar o `create_anomaly` WARNING i validar `sens` dins {700,1000} |
| medium | L'exportació marca `success=True` encara que totes les mostres s'hagin saltat | `hpsec_export.py:1054-1055` | `success = n_errors==0 and n_exported>0`; afegir avís amb llista de mostres saltades i motiu |
| medium | Siblings que fallen es descarten i el conjunt es reporta com a èxit parcial | `gui/widgets/import_panel/panel.py:440-482` | Propagar `failed_siblings` (noms+errors) a `import_completed` i mostrar-los persistentment a la barra del wizard |
| medium | Fallada de reassignació BP DAD254 empassada; àrees BP amb finestres per defecte | `hpsec_import.py:4817-4820` | Afegir `create_anomaly` (warning) a `result['warnings_structured']` i flag `bp_reassignment_failed=True` |
| medium | `uib_sensitivity` es perd entre `analyze_sample` i `quantify_sample` (routing UIB v3.0 trencat) | `hpsec_analyze.py:1754-1762; 1554; 2418` | Afegir `result['uib_sensitivity'] = sample_data.get('uib_sensitivity')` al bloc de `analyze_sample`; warning a `quantify_sample` si hi ha senyal UIB però sensitivity `None` |
| medium | `ensure_data_loaded` empassa errors deixant `data_deferred=True`; el worker calibra amb senyals buits | `hpsec_import.py:6165-6166, 6180-6181, 6355`; `gui/widgets/global_calibration_panel.py:127-141` | Marcar `imported_data['load_error']` (o aixecar excepció) als returns primerencs; `CalSeqWorker` avorta amb `self.error.emit()` si `data_deferred` segueix `True` |
| medium | `_force_reimport` queda penjat a `True` quan `_start_cal_worker` surt aviat per worker ocupat | `gui/widgets/global_calibration_panel.py:481-499, 519-546` | Resetejar `_force_reimport` (i avisar) també a l'early-return, o encuar la petició; idealment passar `force_reimport` com a paràmetre per petició |
| medium | `import_from_manifest` no rellegeix mai `uib_sensitivity` de `0-INFO`, ni amb `load_data=True` | `hpsec_import.py:5610, 5644-5671` | Quan `result['uib_sensitivity']` és `None`, rellegir `0-INFO` del MasterFile i completar-lo |
| medium | `uib_sensitivity` llegit per índex hardcoded `iloc[4,1]` (B5) en lloc de per nom | `hpsec_import.py:578-588` + `4287` | Derivar `uib_sensitivity` buscant la clau per nom a `result['info']`; reservar `iloc[4,1]` només com a fallback amb validació de rang |
| medium | `extract_doc_from_masterfile` cau a posicions hardcoded (`col[3]`/`col[5]`) sense avís | `hpsec_import.py:694-703` | Emetre warning estructurat quan s'usa el fallback posicional i validar rang/unitats plausibles de DOC |
| medium | `compute_toc_calc` tria net_delay amb l'ÚLTIMA clau sense prioritat; `read_current_delay` prioritza Suite>B12 | `hpsec_import.py:2381-2389` vs `hpsec_delay.py:51-71` | Extreure la lògica de prioritat de `read_current_delay` a una funció compartida usada per ambdós |
| medium | `pre_margin` divergent: delay-tool fix 1.5, import llegeix config `toc_pre_margin_min` | `hpsec_delay.py:197,237,296` + `panel.py:2944,3110` vs `hpsec_import.py:2429-2434` | Passar `pre_margin_min` des de `get_config()` també a les funcions/callers de `hpsec_delay` |
| medium | Volum d'injecció per índex de columna hardcoded (13) a migrate i import | `hpsec_migrate_master.py:336-352` + `hpsec_import.py:1811-1835` | Detectar la columna de volum per capçalera (`inj`+`vol`/`µL`) abans de l'índex 13; estendre la validació creuada al cas migrate |
| medium | `_read_hplc_timestamps`/`_read_toc_timestamps` degraden en silenci (llista buida / fallback col D) | `hpsec_delay.py:136-137, 177-178` | Propagar warning/anomalia estructurada quan `date_col_idx` és `None` o s'usa el fallback posicional |
| medium | El shift DOC↔254 és un escalar d'UN sol pic KHP aplicat a TOTES les injeccions | `hpsec_calibrate.py:3210; 5313-5316`; aplicat a `hpsec_analyze.py:1851-1854` | Estimar shift per-injecció/per-bloc des del 254 de la pròpia mostra o interpolar entre KHP veïns; anomalia si els shifts dels KHP difereixen > tolerància |
| medium | La importació preloaded emet sense `warnings_structured` → wizard marca Importar 'ok' ignorant blockers | `gui/widgets/import_panel/panel.py:339-345` | Afegir `'warnings_structured': result.get('warnings_structured', [])` a l'emit de `_display_preloaded_data`; o que `_on_import_completed` recorri a `warnings`/manifest quan falti |
| medium | Lògica de semàfors duplicada (`_populate_row_estat` vs `_update_row_state`) que divergeix | `gui/widgets/import_panel/panel.py:1189-1250 i 1824-1928` | Extreure `_compute_row_semaphores(row)` compartida; que `_update_row_state` recalculi i desi `worst` al DOC i unifiqui el tractament de tipus opcionals |
| medium | TODO de funcionalitat pendent (gràfics scatter/Bland-Altman) dins panell de producció | `gui/widgets/compare_panel.py:293` | Treure el TODO i registrar-lo a issue/CLAUDE.md, documentar l'estat amb comentari factual, o implementar els gràfics |

### Severitat BAIXA

| Severitat | Troballa | file:line | Fix concret |
|---|---|---|---|
| low | Auto-fix columnes G-L→A-F: si peta, només DEBUG i continua amb columnes buides | `hpsec_import.py:1763-1764` | Pujar a `logger.warning` i `warnings.append('auto-correcció de columnes ha fallat, revisar MasterFile')` |
| low | Lectura del pre-margin TOC empassa qualsevol error i cau a 1.5 min en silenci | `hpsec_import.py:2430-2434` | Capturar excepcions concretes i `logger.warning('no s'ha pogut llegir toc_pre_margin_min, usant 1.5')` |
| low | Heurístic de columna de volum (índex 13): qualsevol error torna a `None` sense distingir error de 'no detectat' | `hpsec_import.py:1834-1835` | `logger.warning` quan l'heurístic falla per excepció (diferent de 'no trobat') i propagar warning estructurat si no hi ha font de volum fiable |
| low | `uib_sensitivity` no se serialitza a `analysis_result.json` | `hpsec_analyze.py:3682-3727` | Afegir `'uib_sensitivity': sample.get('uib_sensitivity')` al dict de `summarize_sample` (i a nivell de seqüència) |
| low | Recuperació de `seq_date` des del manifest: path i claus incorrectes (CHECK/ sense `data`; `seq_date`/`date` top-level vs `sequence.date`) | `hpsec_calibrate.py:5503-5508` | Usar `load_manifest(seq_path)` i llegir `manifest.get('sequence',{}).get('date')`. Eliminar ruta i claus hardcoded |
| low | Lectura posicional de `uib_sensitivity` a B5 sense verificar l'etiqueta | `hpsec_import.py:584-588` | Llegir per etiqueta dins el bucle (key amb 'uib' i 'range'/'sens'); reservar `iloc[4,1]` com a fallback amb validació de rang |
| low | `inj_volume_source` no sobreviu al round-trip del manifest | `hpsec_import.py:5322-5331` vs `5765`; set a `2139` | Afegir `'inj_volume_source': inj_info.get('inj_volume_source')` al dict 'injection' i propagar-lo a `injection_info` al recarregar |
| low | Cache de calibració basat només en `mtime`: escriptures externes amb la mateixa marca no invaliden | `hpsec_calibrate.py:356-358, 2549-2551, 2751-2753` | Afegir mida/hash ràpid a la clau de cache o exposar una `invalidate_caches()` pública cridada per qualsevol escriptor extern |
| low | Comparació de versió lexicogràfica `version < '3.0'` re-dispararà migració per a ≥ '10.0' | `hpsec_calibrate.py:363-366` | Comparar numèricament: `tuple(int(x) for x in version.split('.')) < (3,0)` o `packaging.version.parse` |
| low | Aplicar nova calibració + requantificar reescriu JSONs però no refresca `processed_data` d'altres panells | `gui/widgets/global_calibration_panel.py:3476-3597` | Emetre senyal `calibration_updated` a tots els panells amb `processed_data` perquè marquin obsolet o recarreguin del JSON, reutilitzant `is_cal_stale` per bloquejar exports caducats |
| low | Primera branca de detecció de senyal TOC (`'toc'+'ppb'`) sense guard `sig_col is None` | `hpsec_import.py:685-692` | Afegir `and sig_col is None` i prioritzar la columna exacta `'TOC (ppb)'` |
| low | migrate escriu B5=`'UIB_range'` amb literal `'None'` i el lector el consumeix com a sensibilitat | `hpsec_migrate_master.py:438-439` | Separar files `'UIB_sensitivity'` (numèric) i `'UIB_range'`; el lector busca la sensibilitat per nom |
| low | `_save_toc_calc_to_masterfile` esborra `4-TOC_CALC` abans de validar l'escriptura | `hpsec_import.py:2233-2265` | Escriure a fitxer temporal i fer replace atòmic; o substituir el full només després de validar |
| low | Resta de baseline incoherent entre branques de càlcul d'àrea KHP (afecta report, no RF) | `hpsec_calibrate.py:3180-3182 / 3303 / 3356-3361` | Recomputar `area_original` amb la mateixa fórmula del cap final (restar `_bl_for_cap`) abans de comparar |
| low | `recompute_area_with_repair`: el guany recuperat s'integra sobre TOT l'eix, no la finestra d'integració | `hpsec_core.py:966-968` | `recovered = trapezoid(max(y_full[left:right+1]-y[left:right+1],0), t[left:right+1])` |
| low | `downsample_to_cadence` per bin-average aplana lleugerament l'àpex de pics estrets (biaix UIB↔Direct ~1-2%) | `hpsec_core.py:128-153` | Integració per bin (trapezoid dins el bin / amplada) i `t_ds` al centre del bin; validar conservació d'àrea sobre gaussiana sintètica |
| low | Lectura de shift amb `or` pot agafar la clau alternativa quan el shift real és 0 | `hpsec_analyze.py:1820-1821` | `shift_uib = data.get('shift_uib', data.get('shift_min_u', 0.0))` i anàleg per direct |
| low | Baseline/soroll KHP suposen que els primers 20-30 punts són baseline neta | `hpsec_calibrate.py:3066 / 3137 / 3318` | Estimar baseline/soroll des d'una finestra robusta lliure de pics (percentil baix) en lloc d'un slice fix |
| low | Fallback de límits per threshold trenca al primer mínim local (soroll) i pot truncar l'àrea | `hpsec_core.py:2244-2247 / 2255-2258` | Suavitzar abans del fallback o exigir que el mínim local estigui per sota d'una fracció de l'amplitud abans de tallar |
| low | `_update_warnings()` construeix la llista d'avisos i la descarta (codi mort no-op) | `gui/widgets/import_panel/panel.py:1982-2015` | Eliminar `_update_warnings()` i les seves 4 crides (els avisos ja viatgen per `import_completed.emit`/`warnings_structured`), o connectar-la realment |

## 4. Full de ruta de remediació prioritzat

**Bloc A — Integritat de la quantificació (fer primer; ataca l'arrel S1+S2, impacte directe en ppm).**
1. **Cache enverinat de calibració** (`hpsec_calibrate.py`, high): fer `load_*()` retornar `deepcopy` i invalidar el cache en `finally`/a l'inici de `save_*`. Elimina l'estat fantasma que contamina totes les quantificacions de la sessió. És l'arrel de S2 i toca el nucli.
2. **Volum d'injecció assumit** (`hpsec_analyze.py:1542-1546`, high): flag `volume_source='assumed'` + anomalia, opcionalment `ppm=None`. Arrel de S1 amb impacte directe. Coordinar amb la detecció per capçalera (`migrate`+`import`, índex 13) i amb el warning de l'heurístic (`1834-1835`) — mateix camí de dades.
3. **Slider de delay BP** (`calibrate_panel/panel.py`, high): `new_delay = original + offset`. Corregeix corrupció activa del Net delay del MasterFile.

**Bloc B — No sobreescriure ni perdre el fitxer bo (S7, escriptura no atòmica).**
4. `import_sequence` `if True:` + FALLBACK (`hpsec_import.py:4355-4376`): no reescriure `4-TOC_CALC` quan el delay és FALLBACK; emetre blocker; corregir el text 'no trobat'→'recalculat'.
5. Auto-correcció de columnes a `compute_toc_calc` (`1742-1764` vs `2287-2316`): centralitzar la normalització dins `llegir_masterfile_nou` i escriure a `master_data['hplc_seq']`. Resol també DOC Direct perdut.
6. Escriptura atòmica del manifest (`5392-5395`) i de `4-TOC_CALC` (`2233-2265`): patró temp+`os.replace` reutilitzable als dos llocs.

**Bloc C — Deixar de mentir sobre l'èxit (S3) i propagar avisos.**
7. Criteri d'èxit de l'export (`hpsec_export.py:1054-1055`), siblings fallits (`import_panel:440-482`) i camí preloaded sense `warnings_structured` (`339-345`): unificar perquè `success` reflecteixi cobertura i els avisos arribin sempre a la barra del wizard. Mateix arrel: el contracte d'`import_completed`/`warnings_structured` és l'únic canal oficial.
8. Eliminar el codi mort que confon (`_update_warnings` no-op, `import_panel:1982-2015`; bloc deprecated `analyze_panel:2684-2730`).

**Bloc D — Excepcions empassades restants (S1) i font única de veritat (S5).**
9. Convertir els `except Exception: pass/continue/debug` que afecten dades en `logger.warning` + anomalia estructurada: `load_manifest` (`5420-5427`), reassignació BP (`4817-4820`), `ensure_data_loaded` (`6165-6181`), timestamps de delay (`hpsec_delay.py:136-137,177-178`).
10. Unificar lògica duplicada: prioritat de net delay (funció compartida `read_current_delay`), `pre_margin` (passar config a `hpsec_delay`), semàfors (`_compute_row_semaphores` compartida). Cada un elimina una divergència entre dos camins.

**Bloc E — Latents UIB i robustesa posicional (S4+S6; fer abans de calibrar UIB per sensibilitat).**
11. Propagar `uib_sensitivity` (analyze→quantify, summarize, recàrrega de manifest) i `inj_volume_source` al round-trip. Avui latents, però són pre-requisit per a un routing 700/1000 correcte.
12. Lectura per etiqueta/capçalera en lloc d'índexs hardcoded (volum col 13, B5, DOC col 3/5, senyal TOC). Inclou separar `UIB_sensitivity` de `UIB_range` al migrate.

**Bloc F — Polishing de baix impacte:** comparació de versió numèrica, cache per hash, fixes d'àrea/baseline/shift, `cross-panel calibration_updated`, TODO de `compare_panel`.

## 5. Comentaris no professionals

| file:line | Comentari actual | Reescriptura / acció |
|---|---|---|
| `gui/widgets/analyze_panel/panel.py:2684-2730` | `return  # v2.2.0: dead path` + `# ----------- Codi original deprecated -----------` + ~44 línies inabastables | Esborrar 2685-2729. Deixar la funció buida amb la docstring de no-op. L'historial viu a git, no al codi |
| `gui/widgets/compare_panel.py:293` | `# TODO: grafics (scatter ppm_col vs ppm_bp, Bland-Altman)` | Treure el TODO del codi i registrar-lo a issue/CLAUDE.md (Feature status), o substituir per comentari factual de l'estat actual |
| `hpsec_analyze.py:178-190, 233` | Làpides amb data: `# NOTA: get_baseline_correction() eliminada (2026-02-02)`, `apply_smoothing migrat`, `align_signals_by_max/apply_shift moguts (2026-02-03)`, `find_peak_boundaries/detect_main_peak` | Esborrar les làpides de **pura eliminació** amb data. Per les funcions **mogudes**, l'import explícit del capçal ja documenta la procedència — eliminar el comentari |
| `hpsec_calibrate.py:1490, 1575-1583, 1861` | `obtenir_seq mogut (2026-01-29)` i anàlegs | Igual: eliminar; conservar només si l'import no deixa clara la procedència |
| `hpsec_reports.py:205` | `is_khp mogut...` | Eliminar (l'import ho documenta) |
| `hpsec_import.py:100` | Anotació de migració/eliminació | Eliminar la làpida |
| `hpsec_import.py:952` | `# Excloure fitxers UIB (contenen "UIB1B") per si de cas` | `# Els CSV UIB1B són cromatogrames UIB, no matrius DAD Export3D; excloure'ls evita contaminar la matriu 3D` |
| `hpsec_core.py:60, 429, 637, 839, 2357`; `hpsec_calibrate.py:2165, 3279`; `hpsec_warnings.py:198-218`; `hpsec_replica.py:74,79,249,946,1164`; `hpsec_analyze.py:831` | Sobrenom informal `batman` als comentaris d'integració de pics | Netejar `batman` dels comentaris **només si** també es renomenen les claus/aliases `batman_*` subjacents (vives com a backward-compat: `detect_batman`, `repair_batman_in_replica`, claus JSON). Altrament el comentari documenta una clau real; substituir per la forma tècnica `irregular_top` de manera coordinada |
| `gui/widgets/process_wizard_panel.py:622, 750, 774, 1455, 1880, 1986, 2421, 2440` | `range(..., 6)  # v2.2.0: 6 fases`, `# v2.2.0: Exportar és l'última (tab 5)` | Treure el prefix de versió; definir `N_PHASES = 6` (i l'índex de l'última tab) com a símbol en lloc de comentar el literal màgic |
| `hpsec_analyze.py:30-44, 3641` | Bloc changelog apilat `# v1.6.0: ... v1.5.0: ... v1.4.0: ... v1.2.0:` | Moure l'historial a CHANGELOG/git; eliminar del codi |
| `hpsec_reports.py:4329` | Anotació de versió inline | Eliminar el prefix de versió; deixar comentari factual si cal |