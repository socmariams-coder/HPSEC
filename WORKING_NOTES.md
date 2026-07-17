# HPSEC Suite — Working Notes (diari de sessions)

Historial detallat de decisions i troballes per sessió. Separat de
CLAUDE.md perquè el context de sessió no carregui ~700 línies de diari;
consultar quan es necessiti el context històric d'un tema concret.

> Last updated: 2026-07-17

### La finestra d'integració NO s'ha de tocar: el blocker ja protegeix la calibració (2026-07-17)

**Conclusió: es TANCA el pendent "ancorar la finestra d'integració al 254 / limitar-la a
n·FWHM". Es va implementar, es va provar contra les dades reals, i es va REVERTIR perquè
empitjora el sistema.** Cap canvi de codi al motor (revertit); el valor de la sessió és la
troballa.

**1. L'escapada de finestra és real i és VIVA** (no és un artefacte de JSON antic: el motor
d'avui reprodueix exactament les àrees desades, comprovat injecció a injecció a la 306).
Mesurat sobre les 32 injeccions de 293/305/306, amplada de finestra en múltiples del FWHM:

| | màx x·FWHM | escapades |
|---|---|---|
| 293 (columna sana) | 18,6 | cap |
| 305 (columna sana) | 12,3 | cap |
| 306 (columna substituïda) | **56,7** | 0,25 R1 (54,6 min = 40,9·FWHM) · 0,25 R2 (56,7) · 0,5 R2 (26,3) |

**2. PERÒ l'escapada NO corromp la calibració acceptada.** A la 306, les dues rèpliques de
0,25 i les dues de 0,5 ja porten `KHP_PEAK_NON_GAUSSIAN` (BLOCKER, `invalidates=True`) —
precisament PERQUÈ la finestra ampla fa que el fit bigaussià no quadri. Aquests nivells no
entren mai a la calibració. L'escapada afecta l'àrea MOSTRADA i el diagnòstic, no el resultat.

**3. El guard n·FWHM és ACTIVAMENT NOCIU.** Amb finestra estreta el pic torna a ajustar-se a
una gaussiana → el blocker DESAPAREIX → dues injeccions genuïnament trencades passen de
correctament rebutjades a admeses com a punts vàlids. Verificat (llindar 22·FWHM, generós:
no dispara mai a 293/305):

| | sense guard | amb guard |
|---|---|---|
| 0,25 ppm | àrea 12.583, `KHP_PEAK_NON_GAUSSIAN` ×2 → nivell INVÀLID | àrea 876, només warning → nivell ADMÈS |
| 0,5 ppm | àrea 3.408, `KHP_PEAK_NON_GAUSSIAN` ×2 → nivell INVÀLID | àrea **550**, només warning → nivell ADMÈS |

El 0,5 ppm queda per SOTA del 0,25 (550 < 876): sobreviu la rèplica trencada. Les injeccions
de 0,25 i 0,5 de la 306 són dolentes de debò — les rèpliques no concorden en ALÇADA neta
(0,25: h_net 617 vs 65 · 0,5: 1.136 vs 444), que la finestra no explica. El blocker les
rebutja per la raó correcta encara que hi arribi per una via indirecta.

**Norma que se'n deriva:** no substituir una detecció que rebutja bé per una correcció que
maquilla l'entrada del detector. Si el pic no és gaussià amb la finestra que el programa obre,
el senyal té un problema real; estrènyer la finestra només amaga el símptoma.

**4. `rf_mass_direct` NO és el pendent de la recta — és el RF d'un sol punt** (el nivell alt).
Comprovat exactament: 293 → 1.621,68/2,0 µg = **810,84** · 306 → 19.520,875/2,0 = **9.760,44**.
Per això surt idèntic amb guard i sense. **No confondre amb els pendents publicats** (293: 795
de la regressió sobre els 6 nivells; Calibration_Reference: 793,9). No són la mateixa magnitud
i no s'han de comparar — cap discrepància, però és un parany fàcil.

**Correcció a documentació existent:** l'avís de `informe_calibracions/LLEGEIX-ME.md` ("l'àrea
del DOC de la 306 no és fiable") és cert per a les àrees de 0,25/0,5 que hi apareixen, però cal
matisar que aquests punts ja estan bloquejats i no entren a cap calibració. La conclusió del v4
sobre COLUMN no depèn d'això.

### Informe d'integració 305/306 + la columna nova és PITJOR que la que substitueix (2026-07-15)

**Entregable:** `Proyectos\Informe_integracio_DOC_305_306_v2.docx` (9 seccions, 4 figures, 6 taules).
Generadors permanents: `compara_integracions.py` (anàlisi + taules) · `fes_figures_integracio.py`
(4 figures, **totes amb les 3 seqs 293/305/306** per demanda de la usuària) · `gen_docx_integracio.py`.

**El defecte de la finestra hi és SEMPRE, però només mossega amb base no plana.** Amplada de la
finestra del programa en múltiples del FWHM: 293 mediana 10,3 (màx 18,6) · 305 mediana 9,7 (màx
12,3) · 306 mediana 13,0 (màx **40,9**). Les 32 injeccions superen 6×FWHM (una integració correcta
= ~4×FWHM). Sobre-integració resultant: 293 ×1,05–1,44 · 305 ×1,02–1,20 · **306 ×1,07 a ×15,0**
(un cas ×173). Depèn de la CONCENTRACIÓ (el senyal aliè capturat és constant, el pic minva): a la
306 inverteix l'ordre dels punts (0,25 ppm → 12.583 > 3 ppm → 11.564). **Les estructures tardanes
(pics ~31 i ~36 min, altiplà 44–72 min) hi són a TOTES les seqs**; a la 305 la finestra s'atura
abans, a la 306 se les empassa.

**Rectes (integració vigent → estreta ancorada ±2·FWHM + base local):**
293: 795/R²0,9998 → 757/R²0,9996 · 305: 7.868/R²0,9982 → 7.718/R²0,9988 · **306: 6.193/R²0,607 →
9.319/R²0,9996**. A 293 i 305 el criteri és indiferent; a la 306 ho decideix tot.

**NO CONVERGEIXEN:** amb la mateixa integració estreta, 306 dona RF 9.319 vs 305 7.718 (+21%).
A 5 ppm l'alçada neta és IDÈNTICA (10.756 = 10.756 ppb) però el FWHM difereix (1,48 vs 1,60) i
l'àrea +15%. Eixamplament a alçada constant amb massa constant (254 ho confirma) → l'excés d'àrea
NO ve de la mostra. Dos factors oberts: (a) la base de 1.400–2.000 ppb amb deriva interna pot no
quedar descrita per una recta ajustada als flancs; (b) el pic de 254 de la 306 és 17% més baix que
el de la 305 (101,6 vs 122,6 mAU) amb la mateixa massa, més del que explicaria el 8% d'eixamplament.

**HI HAVIA MOTIU PER CANVIAR LA COLUMNA? Des de les dades: NO.** Eficiència mesurada al pic de 254
(DAD inline; el DOC no serveix, hi afegeix l'eixamplament del transfer):
| | 293 (20/02) | 305 (07/07) | 306 (14/07) |
|---|---|---|---|
| t_ret | 20,87 | **20,89** (+0,02) | 21,93 (+1,06) |
| FWHM | 0,318 | **0,307** (×0,97) | 0,372 (×1,17) |
| plats N | 23.957 | **25.775** (×1,08) | 19.319 (**×0,81**) |
| asimetria USP | 2,06 | 1,73 | **0,89 (frontal)** |

La columna de la 305, cinc mesos després de la 293, anava IGUAL O MILLOR: mateixa retenció, pic
igual d'estret, +8% de plats, millor simetria. Cap indicador de degradació. **La columna nova (306)
té −19% de plats, pics +17% i asimetria 0,89 = FRONTAL** (no cua): canvi de règim de forma, no
desgast; compatible amb buits/canals al llit o columna no equilibrada. Motius no cromatogràfics
(pressió, fuites, antiguitat, manteniment programat) no són observables aquí → registre del lab.

### La 305 separa els dos efectes + informe v5 (2026-07-15)

**Seq nova 305_SEQ_CAL** (COLUMN, 400 µL, 07/07): mateix detector, mode, volum i mètode que
la 306 (14/07); **només difereix en la columna** (substituïda entre totes dues). Cronologia
real: 293 (20/02) → 304 BP (07/07) → **305 (07/07)** → 306 (14/07).

**La 305 és NETA i aïlla la causa:** pics gaussians (R² fit 0,994–0,999), t_ret constant
21,93 min, rèpliques concordants a tot el rang, 254 nm sense deriva, UIB amb fons 597 (marge)
i **recta RF_mass=7.868, intercept=160, R²=0,9982** (0,25–5 ppm) — usable tal com surt, sense
reintegrar. Contrast amb la 306: fons ×47,5, deriva i caigudes, 254 fins a −7 mAU, UIB saturat
ja a la BASE (999,6), pics no gaussians a 0,25/0,5 ppm, R² 0,607.

**Confirmació del guany del detector amb 3 sèries independents:** alçada neta del DOC a 5 ppm
×7,5 (305) · ×7,6 (306) · ×8,2 (304 BP). A la 305 el factor és estable a TOT el rang (×8,1 /
×7,9 / ×8,6 / ×7,5). Fons del DOC (mediana vs seq anterior del mode): ×22,7 (305), ×47,5 (306),
×12 (304). Pic 254: ×1,10 (305), ×1,09 (304), degradat a la 306 (×0,58 a 0,25 ppm).
→ El guany NO acompanya el canvi de columna (la 305 també en porta una de diferent de la 293):
la inestabilitat és **específica de la columna instal·lada per a la 306**.

**Informe v5** (`Proyectos\Informe_calibracions_COLUMN_BP_v5.docx`): Taula 1 amb 5 columnes
(293→305 · 293→306 · 292→304 · interpretació), §4 reescrit (efecte exclusiu de la 306), figures
de COLUMN amb 3 panells, Taula 3 nova d'injeccions de la 305. Verificat: 6 figures, 6 taules,
capçaleres blanc-sobre-teal correctes a totes.

**CADENA REPARADA (deute meu):** a la v4 només es va desar `gen_docx.py` + les sortides; els
scripts d'extracció i de figures eren d'un sol ús i no es van persistir → l'informe no es podia
regenerar. Ara la carpeta té els tres passos: `extreu_dades.py` (valida contra les files
publicades: 293 i 306 coincideixen exactament) → `fes_figures.py` → `gen_docx.py`.
Notes tècniques: els noms de mostra a 4-TOC_CALC són `KHP{codi}_R{n}`; el `y_doc` de
`replicas_info_uib` va corregit de base i NO serveix per a la figura d'UIB (cal el CSV cru
UTF-16 de `CSV/*UIB1B.CSV`, on es veu el fons real: 50,8 / 659,7 / 999,62); el `y_dad_254` del
JSON sí que és cru.

**bp_data.json CONGELAT:** els calibration_result.json de 292/304 s'han reprocessat i ara tenen
menys rèpliques (292: 9 vs 12 publicades; 304: 8 vs 10). Regenerar-lo canviaria els números de
BP de l'informe → `extreu_dades.py` no el toca sense `--bp`. PENDENT de decidir.

### El salt de magnitud ×8: guany, no sensibilitat + la integració de la 306 (2026-07-15)

**Pregunta (usuària):** salt important de magnitud del senyal sense explicació; és bo per a la
sensibilitat?; a la calibració actual s'agafa àrea d'un segon pic que no toca; cal analitzar a
fons els cromatogrames de COLUMN i les seves integracions per a la darrera seq.

**1. El ×8 NO és sensibilitat, és ESCALA (guany).** BP 292→304: alçada de pic ×8 (227→1.854 ppb
a 5 ppm) però soroll ×6,6–12,3 (0,084→0,554 a 5 ppm; 0,263→3,225 a 0,25 ppm) i baseline ×11,8
(27→320 ppb). S/N no millora; a baixa concentració empitjora.

**2. Es PERD el fons d'escala.** Reproductibilitat entre rèpliques a la 306: 0,25 ppm → h_net
617 vs 15 ppb; 0,5 ppm → 1.137 vs 444 (×2,6). A ≥1 ppm: 2–3% (bona). A la 293 totes les
rèpliques quadraven fins i tot a 0,1 ppm. El règim nou inutilitza 0,25–0,5 ppm tot i el ×8.

**3. Descomposició neta del factor (COLUMN, integració correcta):** àrea ×11,9 = alçada ×7,6
(guany del detector; coincideix amb el ×8 de BP) × eixamplament ×1,35–1,55 (FWHM 1,13→1,53 min,
efecte exclusiu de la columna nova). Quadra amb els dos efectes separats de l'informe v4, ara
quantificats i separats numèricament.

**4. LA 306 NO ESTÀ PERDUDA — ho estava la integració.** Amb finestra estreta ancorada al 254
(±2·FWHM, baseline local): RF_mass=9.319, intercept=−559, **R²=0,9996** (0,25–5 ppm). Amb la
integració vigent: RF_mass=6.193, intercept=5.630, **R²=0,607**. Això CONTRADIU la conclusió
"COLUMN no és aprofitable" de l'informe v4 — cal revisar-la. Referència 293 amb la mateixa
integració estreta: RF_mass=757, intercept=4,5, R²=0,9996 → factor real COLUMN ×12,3.

**5. Causa del bug d'integració (= el "segon pic"):** amb baseline alta i derivant (~1.000–2.000
ppb a la 306) la projecció tangent de `find_peak_boundaries` no troba el final del pic i la
finestra s'escapa: **10,6 a 54,6 min** d'amplada quan el FWHM és 1,5 min (correcte ~5–6 min).
Sobre-integració ×1,1 a ×15 (×173 en una injecció trencada). A la 293 la mateixa funció només
sobre-integra ×1,1–1,4 → el bug NOMÉS es dispara amb baseline alta/derivant. A 0,25 ppm la
finestra va de 5 a 60 min i s'empassa l'altiplà de 44–60 min, que no és KHP: A=13.133 en lloc
de 876. Figura: `informe_calibracions/integracio_306_diagnostic.png`.

**Norma acordada:** no barrejar ni comparar seqs de règims diferents; calibració nova i
independent per al règim nou; velles vs noves només com a diagnòstic.

**PENDENT:** ~~ancorar la finestra d'integració al 254 (o limitar-la a n·FWHM) quan la baseline
derivi~~ → **TANCAT 2026-07-17: NO es fa** (implementat, provat i revertit — treu el blocker que
ja rebutja aquestes injeccions; vegeu l'entrada del 17/07 a dalt). Revisar la conclusió de
l'informe v4 sobre COLUMN: segueix pendent, i NO depèn de la integració.

### DIAGNOSI: reparació de pics a calibració ≠ reparació a SEQ normal (2026-07-14)

**Pregunta (usuària):** el sistema de reparació de pics de calibració és igual al d'una SEQ
normal? No permet passar d'una mostra a una altra. → Confirmat: NO són iguals. Hi ha TRES
camins de reparació amb comportaments diferents. Només diagnosi (cap canvi de codi).

**Camí A — SEQ normal (Analitzar):** `JaggedPeakRepairDialog` via `_open_dialog_with_nav`
(analyze_panel/panel.py:2439). Multi-card (totes rèpliques×senyals), factor global 0,50–1,20
amb preview en viu, ancoratges per card en viu + "Copiar a les altres", Aplicar/Desfer/Descartar,
i navegació ◀▶ entre mostres (`navigate_requested` connectat a panel.py:2502: tanca el diàleg
i obre el mateix per la mostra anterior/següent).

**Camí B — Calibració, botó "Reparar pic"** (calibrate_panel/panel.py:2061 i
global_calibration_panel.py:2522): obre EL MATEIX diàleg amb un adaptador, però:
1. **Botons ◀▶ morts**: el diàleg els pinta sempre, però `navigate_requested` NOMÉS es
   connecta a analyze_panel. A calibració clicar-los no fa res. ← la queixa de la usuària.
2. L'adaptador porta només les rèpliques d'UNA concentració × UN senyal, amb `anomalies=[]`
   → totes les cards neixen "needs_repair", mai "repaired".
3. **BUG pèrdua de dades**: al tancar, el panell "sincronitza" cards→overrides: si la card
   no està 'repaired' i existeix override desat → `remove_manual_repair`
   (calibrate_panel/panel.py:2107-2109; global_calibration_panel.py:2591-2593). Com que els
   overrides existents mai es carreguen a les cards, obrir "Reparar pic" sobre una rèplica JA
   reparada i tancar sense tocar res ESBORRA la reparació desada en silenci i recalcula sense ella.
4. **Δ%àrea del preview ≠ efecte real**: el preview integra amb `calcular_fraccions_temps`
   (tot el cromatograma); el que es persisteix es reaplica amb `recompute_area_with_repair`
   (finestra de pic + baseline d'analizar_khp_data). El cim reparat dibuixat sí que és idèntic
   (repair_with_parabola force=True), però l'àrea anunciada i la que arriba a la recta no
   es calculen igual.

**Camí C — Calibració, doble clic / "Detall"** → `KHPDetailDialog` (khp_detail_dialog.py),
una sola rèplica × senyal:
- Sense navegació (ni entre rèpliques ni concentracions); en Aplicar es tanca sol (accept()).
- Sense control de factor (REPAIR_FACTOR=0,85 fix; `set_manual_repair` sense factor → None,
  mentre que el camí B sí que persisteix el factor).
- Sense preview en viu: cada canvi d'ancoratge → tornar a clicar "Previsualitzar reparació".
- Virtut: preview = persistit (tots dos via `recompute_area_with_repair`, la funció canònica).

**Persistència comuna:** overrides a manual_repairs (CHECK/data), reaplicats deterministes a
`calibrate_from_import` (hpsec_calibrate.py:4801/4876/4903) via `apply_manual_repair_to_khp`.

**Proposta (pendent de decisió):** (1) connectar `navigate_requested` a calibració per navegar
entre files de la taula de mètriques; (2) carregar overrides existents a les cards com a
'repaired' — mata el bug d'esborrat silenciós; (3) unificar el càlcul d'àrea del preview amb
`recompute_area_with_repair`; (4) unificar B i C en un sol diàleg de reparació per calibració
(factor + preview en viu + navegació).

### (2026-07-15) FIX: unificació del sistema de reparació de pics (A = B = C)

Decisió usuària: "tots han de funcionar igual" → el diàleg d'Analitzar (`JaggedPeakRepairDialog`)
és l'únic sistema de reparació a tot arreu. Implementats els 4 punts de la proposta de la diagnosi.

**repair_dialog.py** (analyze_panel):
- `_RepairCard` accepta `saved_anchors` (tupla d'un override desat → inicialitza els spins i els
  ancoratges manuals ABANS del primer preview) i `peak_ctx` (dict peak_idx/left_idx/right_idx/
  baseline/area d'analizar_khp_data). Amb `peak_ctx`, el preview recalcula amb
  `recompute_area_with_repair` → el Δ% mostrat és exactament el que persistirà
  `apply_manual_repair_to_khp` (fix BUG Δ% enganyós). Fallback silenciós al càlcul estàndard
  (fraccions) si no hi ha context. Sense `peak_ctx` ni `_manual_repair` res no canvia (camí A intacte).
- El diàleg llegeix `_manual_repair` i `_peak_ctx` de cada rèplica de l'adaptador; el factor
  global s'inicialitza amb el primer override que en porti.
- `_session_modified` per card + `_modified_keys` a nivell de diàleg. El set del diàleg és la
  font de veritat perquè `_refresh_after_action` remapeja cards→targets (l'ordre dels targets
  canvia quan una anomalia passa d'auto-force a registrada) i el flag ha de seguir la RÈPLICA,
  no el widget. El remapeig també reassigna `_peak_ctx`.
- Helpers de mòdul compartits pels dos panells de calibració:
  `make_calibration_replica_entry` (entrada d'adaptador amb `_peak_ctx`, i si hi ha override:
  estat 'repaired' + `_manual_repair` + backup `y_*_original` perquè "Desfer" funcioni) i
  `sync_repair_cards_to_overrides` (persisteix NOMÉS cards amb `_session_modified`: repaired →
  `set_manual_repair`, no-repaired amb clau existent → `remove_manual_repair`; cards no tocades
  NO es toquen — fix BUG pèrdua de dades silenciosa).

**hpsec_analyze.py** `repair_irregular_top_in_replica`: si `mark_repaired` retorna False i el
fallback de strings tampoc troba res (adaptador de calibració amb `anomalies=[]`), s'AFEGEIX
`{"code": anom_key, "repaired": True, "repair_info": ...}` a la llista — fix del BUG "Aplicar
no persisteix mai al camí B" (l'estat 'repaired' ara sobreviu `_refresh_after_action` i la
sincronització del panell el veu). Per al camí A no canvia res (l'anomalia sempre existeix).

**calibrate_panel/panel.py**:
- `_build_repair_adapter(conc, signal, name=, seq_path=)` injecta per rèplica l'override
  existent (`manual_repairs.json`) i el `_peak_ctx` via el helper compartit.
- Lògica d'obertura extreta de `_on_calib_repair_clicked` a `_open_calib_repair_for(khp)`.
  Navegació ◀▶ connectada: unitats = grups (conc_ppm, senyal) distints en l'ordre de les files
  de `metrics_table` (`_metrics_repair_groups`; les sub-files d'anomalies, sense dict khp a
  col 0 UserRole, se salten). Navegar = tancar, seleccionar la primera fila del grup nou i
  reobrir. La recalibració (`_run_calibrate`, worker async) es fa UN sol cop quan es tanca
  l'últim diàleg de la cadena (guard `_repair_nav_depth` / `_repair_recalc_pending`) — així la
  taula no es reconstrueix sota els peus mentre es navega.
- Detall KHP (`_open_khp_detail`): `repair_requested` → tanca el detall i crida
  `_open_calib_repair_for(khp)`. `_on_detail_repair_applied` eliminat (senyal sense emissor,
  verificat amb grep). `set_manual_repair` fora dels imports del panell (ja només l'usa el helper).

**khp_detail_dialog.py** (camí C fusionat): eliminada la UI pròpia de reparació (fila
d'ancoratges, "Previsualitzar reparació", "Aplicar reparació", `_on_repair_clicked`/
`_apply_repair`/`_on_manual_toggled`/`_repaired_data`/`_anchors_inited`) i el senyal
`repair_applied`. Nou botó "🔧 Reparar pic" que emet `repair_requested = Signal()`. Es manté:
cromatograma (amb overlay del senyal reparat si n'hi ha), mètriques, "Desfer reparació manual"
i el toggle outlier.

**global_calibration_panel.py** `_repair_seq_cal_entry`: adaptador via helper compartit
(overrides carregats → cards neixen 'repaired'); sync només-modificats; navegació ◀▶ per
`self._seq_cal_entries` (índex ±1, reobrint el mateix mètode; si el diàleg venia del preview
del punt, `parent_dialog.accept()` en navegar); recàlcul (`parent_panel.load_seq_cal`) un sol
cop al final de la cadena. El missatge del PROMIG es conserva; l'avís "no s'ha aplicat cap
reparació" desapareix — tancar sense tocar res ara és un no-op legítim (abans esborrava overrides).

**Desviacions del pla (justificades):**
- `_session_modified` per card sol no era segur: `_refresh_after_action` reassigna els cards
  als targets reordenats i el flag hauria quedat al widget equivocat → `_modified_keys`
  (rep_key, signal) al diàleg com a font de veritat, i el flag per card se'n deriva.
- `_peak_ctx.area` usa `area_pre_manual` quan existeix (no l'`area` actual): amb override ja
  aplicat, `area` porta la reparació; el recompute ha de partir de l'àrea pre-override, igual
  que fa `apply_manual_repair_to_khp` en reprocessar.
- El sync es va extreure a un helper compartit a repair_dialog.py (en lloc de duplicar-lo als
  dos panells) perquè el comportament sigui idèntic per construcció.
- Recalibració ajornada al final de la cadena de navegació (no per diàleg): `_run_calibrate` /
  `load_seq_cal` reconstrueixen la taula de forma asíncrona i haurien invalidat les files
  mentre l'usuari encara navega.

**Verificat:** `py_compile` dels 5 fitxers OK; `test_robustesa_audit.py` 28/28 OK; fum GUI
offscreen (33 asserts OK): camí A intacte (2 cards, preview estàndard, apply → 'repaired' +
`_session_modified` + anomalia dict registrada), card amb `_manual_repair`+`_peak_ctx` neix
'repaired' amb anchors 21,4/22,6 i factor 0,90 carregats, `ctx_area_new` del preview ==
`recompute_area_with_repair` (persistit), undo → 'needs_repair', cicle BUG 2 (anomalies=[] →
apply → refresh → segueix 'repaired'), sync amb overrides desats: tancar sense tocar NO escriu
i conserva els overrides / undo d'una rèplica n'esborra només aquella, i `KHPDetailDialog`
emet `repair_requested` amb la UI antiga absent.

### Selecció robusta de rèplica a la calibració KHP (2026-06-26)

**Problema (usuària):** la calibració BP era poc fiable. Anàlisi a fons sobre les 38 rèpliques
de l'historial: les anomalies dominants al KHP són **timeout (recàrrega xeringues TOC) i pics
no-gaussians**, no l'irregular-top (que ja es repara bé, 2/2). El timeout **perfora l'àpex** del
pic (verificat: 292@5ppm rep2 perd −36% d'àrea; la cua post-timeout és artefacte, no es pot
reconstruir de manera fiable — ajust gaussià recupera 241 vs 342 reals).

**Bug arrel:** `_process_khp_group` (hpsec_calibrate.py) feia la **unió** d'anomalies de TOTES
les rèpliques i la passava a `register_calibration`, que invalidava el **nivell sencer** si
qualsevol rèplica tenia un blocker — tot i que la selecció ja triava la rèplica neta. Però el
catàleg (hpsec_warnings.py) marca aquests blockers com **per-rèplica** amb acció literal "triar
l'altra rèplica" (KHP_TIMEOUT_PEAK, KHP_PEAK_NON_GAUSSIAN, TIMEOUT_AT_BOUNDARY, KHP_REPLICA_OUTLIER,
KHP_DOC_SATURATED). Resultat: la 292 perdia 0,1/0,5/5,0 ppm i es quedava amb 3 punts agrupats.

**Fix (hpsec_calibrate.py, `_process_khp_group`):** després de l'auto-detecció d'outlier, si hi ha
rèpliques amb blocker però en queda alguna de neta, es **descarten les bloquejades** (`replicas = _usable`)
i tot el downstream (estadístiques, selecció, unió d'anomalies de validesa) opera només amb les netes
→ el nivell es manté VÀLID amb la rèplica bona. Només si TOTES porten blocker es manté el conjunt
(camí all_invalid existent). Afegit `excluded_replica_anomalies` + `selection.n_replicas_usable/
excluded_anomalous` per traçabilitat. `calibration_anomalies` ara = unió de les NETES (validesa correcta).

**Verificat (292_SEQ_CAL_BP):** 0,1/0,5/5,0 ppm passen a vàlids (rèpliques [1]/[2]/[1]);
recta BP de 3 punts (RF=761, esbiaixada) → **6 punts RF_mass=683, R²=0,998** — quadra amb 114_SEQ_BP (681).

**Paritat reparació (2) FETA:** a `analizar_khp_data` la decisió `detect_peak_anomaly` (L3153, la que
afecta l'àrea) es pren ara sobre senyal SUAVITZAT (`apply_smoothing`), com `analyze_sample`. Reparació i
integració segueixen sobre cru. Efecte: elimina reparacions espúries per soroll del cim (293@1ppm: irregular
True→False). El pre-repair force=True (L3007) es manté: és disseny KHP intencional (pic gaussià pur).

**Calibration_Reference.json REGENERAT (3) (2026-06-26):** creat via `add_calibration()` (esquema v3.0),
signal_scope=direct amb rectes SEPARADES: COLUMN rf_mass=793.9/intercept=28.88 (293_SEQ_CAL, R²=0.9998),
BP rf_mass=682.6/intercept=4.26 (292_SEQ_CAL_BP, R²=0.998). Validat: recuperació 5ppm COLUMN +0.3%, BP −1.1%.
NOTA: COLUMN RF 794 (no el 628 antic) — és correcte amb la integració cap=4 vigent.
Rang fiable ≥0.25–0.5 ppm (0.1 ppm prop LOD: COLUMN +14%, BP −23%).
Scripts de revisió + gràfics a `_results/calibracio_review/` (review_calibracio.py, valida_ppm.py, crear_calibracio.py).
**PENDENT:** UIB (signal_scope=uib) no calibrat encara — només DOC Direct. Validació cross-mode BP↔COLUMN
amb mostres reals: pendent (cal samples dual-mode processats; CHECK/ no és en aquesta còpia de Dades3).

### Reparació amb ancoratges manuals al detall KHP de calibració (2026-06-25)

### Reparació amb ancoratges manuals al detall KHP de calibració (2026-06-25)

**Motiu (usuària):** la calibració tenia una reparació de pic més pobra que l'anàlisi normal
(`repair_dialog.py` ric, amb ancoratges manuals) — només un botó automàtic. La calibració és on
la precisió de l'àrea importa MÉS (fixa el RF), així que ha de tenir el mateix control.

**Canvi (`gui/widgets/calibrate_panel/khp_detail_dialog.py`):**
- Casella "Ancoratges manuals" + camps E (esquerre) / D (dret) en min. Desactivat = automàtic (comportament d'abans).
- `_on_repair_clicked()` passa `anchor_left_t`/`anchor_right_t` a `repair_with_parabola()` quan el mode manual és actiu.
- Botó separat "✓ Aplicar reparació" (abans es reconnectava el mateix botó) → es pot re-previsualitzar amb ancoratges diferents abans d'aplicar.
- `_on_manual_toggled()` inicialitza els camps (rang = [t_min, t_max], default = pic ± 0,8 min COL / 0,4 BP).

**PENDENT: verificació a la GUI** (imports OK; falta provar el flux complet obrint un detall KHP real i comprovar que la nova àrea es propaga a la regressió).

### SEQ_CAL Auto-flow + PDF Dual-Signal (2026-03-04, COMPLETAT)

**Auto-flow (global_calibration_panel.py):**
- `_cal_applied` reemplaçat per `_cal_applied_per_signal` (dict) + `_cal_applied_signals` (set)
- Després d'aplicar Direct, si hi ha UIB → auto-switch al combo + missatge
- Després d'aplicar tots els senyals → missatge "complet" + `show_summary()`
- Combo marca senyals aplicats amb prefix "✓"
- Botó "Aplicar" mostra "✓ Aplicada" per senyal ja aplicat
- `_get_remaining_signals()`: retorna senyals disponibles no aplicats
- Reset complet a `load_seq_cal_data()`

**PDF Dual-Signal (hpsec_reports.py):**
- `generate_dual_calibration_report()`: PDF combinat Direct+UIB
  - Pàg 1: Resum executiu amb taula paràmetres 4 files (Direct COL/BP + UIB COL/BP)
  - Pàg 2-3: Regressió per senyal (scatter + residuals)
  - Pàg 4: Evolució temporal RF (2 subplots: Direct + UIB)
  - Pàg 5: QC Levey-Jennings (2 subplots: Direct + UIB)
  - Pàg 6: Historial calibracions (taula unificada)
  - Pàg 7+: Cromatogrames KHP (ambdós senyals, dedup)
- Si només 1 senyal disponible, genera amb 1 subplot (adaptatiu)

**GUI wiring:**
- `_on_summary_pdf()`: detecta ambdós senyals → genera dual, sinó single
- `_on_generate_cal_report()`: idem (post-apply button)

**Fitxers modificats**: global_calibration_panel.py, hpsec_reports.py

### Export FAIR v2: RAW/PROCESSED + DAD complet + ZIP (2026-03-03, COMPLETAT)

**Implementació completa en 2 sessions:**

**Sessió 1 (fase 1-5 pla original):**
- Noms fitxer sense rèplica: `{sample}_HPSEC_{C|B}.xlsx`
- CSV cromatogrames+resultats, ID sheet 10 seccions
- UI checkboxes CSV+separador, GenerateWorker integrat
- `hpsec_export.py` v2.1.0

**Sessió 2 (ampliació FAIR):**
- `hpsec_export.py` v2.2.0

1. **`write_csv_raw()`**: DOC_Direct_RAW + DOC_UIB_RAW a RESULTATS/RAW/
   - DAD 101λ downsampled a dt=0.04 min via `_downsample_2d()` (bin-average 2D)
   - Rellegeix Export3D complet des de disc via `dad_export3d_path`
   - BP: 1 sola fila a t_max amb totes les λ
   - `_is_numeric()` helper per filtrar columnes λ

2. **`write_csv_processed()`**: DOC_net + 6λ DAD + fraccions + ppm a RESULTATS/PROCESSED/
   - `_get_dad_column()` helper per accedir a DAD per wl

3. **`write_metadata_json()`**: metadata.json amb fingerprints, llista mostres, calibració

4. **`create_export_zip()`**: empaqueta RESULTATS/ en ZIP

5. **`export_sequence()` actualitzat**: nous params `export_raw`, `export_processed`
   (substitueixen `export_csv`). BP RAW/PROCESSED inclòs si COLUMN amb BP vinculat.

6. **`summarize_sample()` (hpsec_analyze.py)**: `dad_export3d_path` propagat al JSON
   d'anàlisi per poder rellegir Export3D complet durant l'export.

7. **UI** (`export_panel.py`): checkboxes RAW/PROCESSED/CSV_SUMMARY/ZIP
   (substitueixen els antics csv_chromatogram_check + csv_summary_check).
   GenerateWorker amb params `export_raw`, `export_processed`, `export_zip`, `export_metadata`.

**DAD downsampling**: dt=0.04 min, ≥6 punts sobre pic més estret (FWHM=0.25 min).
Mida: 5.8 MB → 1.9 MB/mostra (67% reducció).

### Unificació avisos — font única + barra wizard (2026-02-26)

**Problema**: 2 sistemes paral·lels d'avisos (`create_warning()` + `WARNING_DEFINITIONS` i
`create_anomaly()` + `ANOMALY_CATALOG`). El wizard amagava avisos darrere un botó status_indicator;
l'usuari no els veia si no clicava. ~200 línies de codi mort (funcions de migració, filtres, etc).

**Solució implementada:**

1. **`hpsec_warnings.py`**: `ANOMALY_CATALOG` com a font ÚNICA. Eliminats: `WARNING_DEFINITIONS`,
   `create_warning()`, `filter_warnings_by_level()`, `filter_warnings_by_stage()`, `has_blockers()`,
   `dismiss_warning()`, `warnings_summary()`, `migrate_legacy_warning()`, `migrate_warnings_list()`,
   `normalize_warnings()`, `create_warnings_from_timeout_info()`, `create_warnings_from_irregular_top_info()`.
   Afegits codis: `IMP_NO_DATA`, `IMP_MISSING_UIB`, `IMP_MISSING_DAD`, `IMP_ORPHAN_FILES`,
   `IMP_INCOMPLETE`, `CAL_NO_KHP`, `CAL_ALL_REPLICAS_INVALID`, `CAL_GLOBAL_ONLY`,
   `CAL_REPLICA_OUTLIER`, `ANA_NO_CALIBRATION`, `ANA_EMPTY_SAMPLES`.
   `get_max_warning_level()` mantingut com a alias backward-compat (suporta `severity` i `level` keys).

2. **Backends migrats**: `_generate_import_warnings()`, `_generate_calibration_warnings()`,
   `_generate_analysis_warnings()` usen `create_anomaly()` en lloc de `create_warning()`.
   `IMP_EMPTY_CSV` i `IMP_FALLBACK_DAD` eliminats (no accionables).

3. **`WarningBarWidget`** (`process_wizard_panel.py`): barra persistent entre header i tabs.
   Mostra avisos de TOTES les fases completades. Desplegada si ≤3, plegada si >3.
   Color fons adaptat a severitat màxima (vermell/groc/blau). Cada avis mostra icona + missatge + acció.

4. **`WarningReviewDialog` eliminat**: la barra ja mostra tot directament. El status_indicator
   marca avisos com a OK amb un clic (sense diàleg). `WarningSkipDialog` mantingut (per "Següent" amb avisos).

5. **`_update_validation()` simplificada** (`calibrate_panel/panel.py`): ja no recolleix anomalies
   duplicades de `khp_data_direct/uib` — usa `warnings_structured` directament del backend.

6. **`_get_warning_level()` simplificat**: sempre calcula des de `warnings_structured` via
   `get_max_warning_level()`, eliminats fallbacks a `warning_level` key i `warnings` strings.

7. **`_on_*_completed()` actualitzats**: calculen `warning_level` des de `warnings_structured`
   (no del backend `data.get('warning_level')`). Criden `_update_warning_bar()` al final.

**Fitxers modificats**: hpsec_warnings.py, hpsec_import.py, hpsec_calibrate.py, hpsec_analyze.py,
gui/widgets/process_wizard_panel.py, gui/widgets/calibrate_panel/panel.py

### Calibracions independents per senyal/sensibilitat — v3.0 (2026-02-26)

**Problema**: El sistema de calibració tractava Direct i UIB dins d'una sola entrada,
sense distingir sensibilitat UIB (700/1000 ppb). Dades UIB a sensibilitat 700 NO es
poden barrejar amb dades a 1000 per calibrar.

**Solució implementada — Calibration_Reference.json v3.0:**

1. **Estructura nova**: cada entrada a `calibrations[]` cobreix UN sol àmbit (`signal_scope`
   + `uib_sensitivity`). `rf_mass_cal` i `intercept` planers: `{"column": X, "bp": Y}`
   (el senyal ja va definit per `signal_scope`).

2. **`active_calibration_ids`**: dict `{"direct": "CAL_...", "uib": "CAL_...", "uib_700": "..."}`
   — una calibració activa per àmbit. Substitueix l'antic `active_calibration_id` únic.

3. **Migració automàtica v2→v3**: `load_calibration_reference()` detecta v2.0, divideix cada
   entrada antiga en DIRECT + UIB, genera `active_calibration_ids`, guarda i re-llegeix.
   4 entrades antigues → 8 noves. Verificat: RF Direct/COLUMN=752.9, RF UIB/COLUMN=628.

4. **Getters actualitzats**: `get_active_global_calibration(signal, sensitivity)`,
   `get_calibration_for_date(date, signal, sensitivity)`, `get_rf_mass_cal(..., sensitivity)`,
   `get_calibration_intercept(..., sensitivity)`. Helpers `_extract_rf_from_cal()` i
   `_extract_intercept_from_cal()` suporten format planer i nested.

5. **`add_calibration()`**: nous params `signal_scope`, `uib_sensitivity`. Tanca NOMÉS
   calibracions del MATEIX àmbit. ID amb suffix: `CAL_20260226_143000_DIRECT`.

6. **`compute_calibration_fingerprint()`**: si `calibration=None`, hasheja TOTS els
   `active_calibration_ids` (no un sol).

7. **`quantify_sample()` a hpsec_analyze.py**: intercept UIB independent (`intercept_uib`)
   en lloc de reutilitzar l'intercept Direct. `uib_sensitivity` llegit de `sample_result`.

8. **GlobalCalibrationPanel**: vista resum (showEvent) amb taula paràmetres actius,
   scatter regressió amb banda predicció 95%, taula historial calibracions, botó PDF.
   Visible quan no hi ha SEQ_CAL carregada (substitueix "Selecciona una SEQ_CAL").

9. **Cromatogrames KHP PNG**: `save_khp_chromatogram_plot()` i `save_all_khp_chromatograms()`
   a `hpsec_reports.py`. Hook a `calibrate_from_import()` guarda PNGs a
   `SEQ/CHECK/data/khp_plots/`. DOC + baseline + límits integració + àrea ombrejada + 254nm.

10. **PDF calibració amb cromatogrames**: pàgines extra (A4 landscape, GridSpec 3x2)
    amb PNGs dels cromatogrames. `_find_khp_chromatogram_pngs()` busca a
    `regression_data.chromatogram_plots_dir` o `source.seq_references`.

**Fitxers modificats**: hpsec_calibrate.py, hpsec_analyze.py, hpsec_reports.py,
gui/widgets/global_calibration_panel.py, gui/widgets/seq_cal_regression_widget.py,
gui/widgets/history_panel.py

**Call sites fixats** (passaven dict com a primer arg posicional):
- `global_calibration_panel.py` L2237-2240: `get_rf_mass_cal(new_cal, signal=...)` → `get_rf_mass_cal(signal=...)`
- `history_panel.py` L1235-1236: idem

### Unificació sistema d'avisos (2026-02-24)

**Problema**: 3 sistemes paral·lels d'avisos que no s'integren:
1. ANOMALY_CATALOG (18 codis) — usat per analyze, ben estructurat
2. quality_issues (strings lliures) — usat per calibrate, no estructurat
3. WARNING_DEFINITIONS (56 codis) — bridge fràgil amb parsing strings

**Símptomes**: Dashboard "Verificar" sempre buit (hardcoded []), quality_score numèric inintel·ligible,
sense context de mostra ni guia d'acció.

**Solució implementada — ANOMALY_CATALOG com a font única:**

1. **hpsec_warnings.py**: +10 codis KHP (KHP_IRREGULAR_TOP, KHP_MULTI_PEAK, KHP_TIMEOUT_PEAK,
   KHP_SNR_LOW, KHP_RSD_HIGH, KHP_FWHM_HIGH, KHP_ASYMMETRY, KHP_CR_LOW, KHP_BASELINE_DRIFT,
   KHP_NO_DAD). Camp `action` afegit a les 26 entrades. `create_anomaly()` retorna `action`.

2. **hpsec_calibrate.py**: `analizar_khp_data()` genera `calibration_anomalies` amb `create_anomaly()`
   en lloc de strings quality_issues. quality_score derivat automàticament. `calibration_anomalies`
   propagat al return dict, a l'agregació de rèpliques, i a `register_calibration()`.
   `_generate_calibration_warnings()` simplificada: recull anomalies ja estructurades.

3. **sequence_state.py**: Nous camps `calibrate_warnings` i `review_warnings`. `_extract_metadata()`
   extreu avisos blocker/warning de `calibration_anomalies`. `calibrate_state` ara retorna 'warning'
   si KHP local té anomalies (era sempre 'ok').

4. **dashboard_panel.py**: `phases_data` alimenta Verificar i Revisar (era hardcoded []). Tooltip
   prioritza avisos concrets sobre "KHP sibling" genèric. Fallback quality_issues per JSONs antics.

5. **calibrate_panel/panel.py**: Col 15-16 substituïdes: score numèric → badge icona+color
   (✔/ℹ/⚠/✘) amb tooltip que inclou label + acció del catàleg. Fallback per dades sense
   calibration_anomalies. `_update_validation()` prioritza anomalies sobre quality_issues strings.

6. **analyze_panel/panel.py**: `_classify_sample_status()` tooltips enriquits amb `action` del catàleg
   (format: "CRÍTIC: label\n   → acció recomanada").

**WARNING_DEFINITIONS marcat com deprecated** (no eliminat per backward compat JSONs antics).

### Config panel simplificat + contrasenya (2026-02-24)

**Eliminats 22 paràmetres del GUI** (segueixen al JSON, però no editables per l'usuari):
- Detecció d'Anomalies (8 params): interns algorisme
- Càlcul Baseline (6 params): 4 interns + method="mode" + min_noise
- Cromatograma (5 params): max_duration duplicat, smoothing calibrats
- Calibració (3 params): quality_max, min_cals_average, use_historical_fallback

**Contrasenya fixa "LEQUIA"** a `_save_config()` i `_reset_defaults()`.

### Optimització memòria (2026-02-23)

**Problema**: La Suite carregava ~115 MB per seqüència quan en necessitava ~3 MB.
Causa principal: Export3D DAD (101 wavelengths quan en calen 6), DataFrame 2-TOC
retingut en memòria, i fitxers de calibració rellegits del disc repetidament.

**4 blocs implementats (commit f7d03d4):**

1. **DAD filtrat a 6λ durant importació (~95 MB estalvi/SEQ)**:
   - `llegir_dad_export3d(path, wavelengths_to_keep=None)`: filtra columnes just després de llegir CSV
   - 6 call sites actualitzats (find_data_for_injection, import_from_manifest ×3, ensure_data_loaded, import_panel GUI)
   - Wavelengths de `hpsec_config.json → wavelengths.selected` (220, 254, 272, 290, 362)
   - **HCI no afectat**: `compute_hci()` a `hpsec_humic.py` rellegeix Export3D original del disc
     via `dad_export3d_path` (path propagat a `_flatten_samples_for_processing`), no usa el DataFrame en memòria

2. **Alliberar master_data["toc"] (~2 MB estalvi/SEQ)**:
   - `master_data["toc"] = None` al final de `import_sequence()` i `ensure_data_loaded()`
   - Si es necessita de nou (reimportació), `ensure_data_loaded()` rellegeix MasterFile des del disc

3. **Cache calibració amb verificació mtime (estalvi I/O)**:
   - `load_calibration_reference()`: cache `_cal_ref_cache` + `_cal_ref_mtime` (5x speedup)
   - `load_khp_history()`: cache `_khp_cache` + `_khp_mtime` + `_khp_cache_path` (94x speedup)
   - `load_local_calibrations()`: cache `_local_cal_cache` + `_local_cal_mtime` + `_local_cal_path` (31x speedup, 10 call sites)
   - Invalidació automàtica als corresponents `save_*()` functions
   - ~25 crides/sessió passen de lectura disc a lectura memòria

4. **Eliminar "df" redundant de rep_data["direct"] (~12 MB estalvi/SEQ)**:
   - `"df": df_doc` eliminat de 3 llocs (find_data_for_injection, import_from_manifest, ensure_data_loaded)
   - Verificat: cap codi extern accedeix a `rep_data["direct"]["df"]`, només `"t"` i `"y"`

**Safety guards mantinguts** a `hpsec_analyze.py`:
- `_flatten_samples_for_processing` L2254: `len(columns) > 8` → no-op (DAD ja filtrat)
- `analyze_sample` L1617: idem

### UIB Downsample + Saturació (2026-02-23)

**Problema**: UIB CSV té dt=0.005 min (14k punts) vs DOC Direct dt=0.067 min (1.1k punts).
Amb 12.6x més punts, el Savitzky-Golay (finestra 131 pts vs 11 pts) i les derivades es
comporten diferent → límits d'integració i àrees no comparables entre UIB i Direct.
A més, el detector UIB (Sievers M9e) satura a la sensibilitat configurada (700/1000 ppb),
retallant pics d'alta concentració.

**Solució implementada:**

1. **`hpsec_core.py`**: Nova funció `downsample_to_cadence(t, y, target_dt)`.
   - Bin-average: bins uniformes de mida target_dt, mitjana dels punts per bin.
   - Preserva àrea integrada i forma del pic.
   - Auto-detect: si dt_median >= target_dt * 0.8, retorna dades originals.
   - Constant `DOC_TARGET_DT_MIN = 0.0667` (4 segons, cadència TOC).

2. **`hpsec_import.py`**: Downsample aplicat a 3 punts d'entrada UIB:
   - L2708: mostres regulars (`find_data_for_injection`)
   - L4314: KHP via manifest (`import_from_manifest`)
   - L4720: KHP via `ensure_data_loaded`

3. **`gui/widgets/import_panel/panel.py`**: 4t punt d'entrada UIB (reassignació manual).

4. **`hpsec_warnings.py`**: Nova anomalia `UIB_SATURATED` (BLOCKER, invalidates=True).
   - Icon: "SAT", description: "Senyal UIB saturat"

5. **Detecció saturació per forma del pic** (refactored 2026-02-25):
   - **`hpsec_core.py`**: `detect_peak_clipping(t, y)` — detecta retall/clipping per
     **plateau/FWHM ratio**. Gaussiana normal ≈ 0.17, threshold > 0.40 = saturat.
     Inclou estimació automàtica de baseline (median bottom 20%).
     **Independent de qualsevol paràmetre de sensibilitat** — basat en forma intrínseca.
   - **`hpsec_calibrate.py`**: `analizar_khp_data()` crida `detect_peak_clipping` quan
     `doc_source == "uib"`. Guard: Direct MAI entra al codi de saturació.
     Enrichment Direct←UIB: `uib_saturated` propagat de l'entrada UIB (L5020).
   - **`hpsec_analyze.py`**: `analyze_sample()` crida `detect_peak_clipping` per UIB.
   - **`_build_entries()`** (2 implementacions: `hpsec_calibrate.py` + `seq_cal_regression_widget.py`):
     llegeixen `uib_saturated` del backend (ja calculat), guard `signal_name == 'uib'`.
   - **Verificat amb 293_SEQ_CAL**: 5ppm ratio=0.65 → SAT, 2ppm ratio=0.17 → ok
     (y_max=828 > sens=700, però forma normal → correcte: sensibilitat és rang recomanat,
     no límit dur de retall).

6. **`gui/widgets/analyze_panel/panel.py`**: Auto-exclusió i UI:
   - Punts UIB saturats auto-exclosos de la regressió (`_seq_cal_excluded`)
   - Columna anomalies mostra "SAT" en vermell
   - Swap senyal Direct/UIB recalcula exclusions

**Verificació**: Downsample 15000→1125 pts (13.3x), dt 0.005→0.0675 min.

### Informe calibració PDF + regression_data al JSON (2026-02-23)

**Motivació**: L'informe PDF de calibració recalculava la regressió des de KHP_History.
L'usuari va demanar que TOTES les dades vinguessin del JSON, sense recalcular res.

**Canvis implementats:**

1. **`hpsec_calibrate.py`**: `add_calibration()` nou param `regression_data`.
   - `_sanitize_regression_data()`: converteix numpy→Python, genera `stats_per_concentration`
   - El JSON ara guarda: punts (ug_doc, area, residual, y_pred), RMS, stats per conc, model info

2. **`gui/widgets/review_summary_panel.py`**: Passa `regression_data` a `add_calibration()`.
   - Botó "📄 Generar Informe Calibració (PDF)" visible després d'aplicar
   - Scatter miniatura: 7×3 (era 6×2.5), amb recta vigent (taronja) i banda predicció 95%
   - Equació en monospace amb fons blau
   - Comptador retro: "X/Y SEQs seleccionades"
   - Comparació via `format_calibration_comparison_html()` compartit

3. **`gui/widgets/global_calibration_panel.py`**: Botó "📄 Generar Informe PDF" al panell global.

4. **`hpsec_reports.py`**: `generate_calibration_report()` — 5 pàgines PDF:
   - P1: Resum executiu (taula params, equació, stats per concentració)
   - P2: Scatter regressió + residuals (banda predicció 95%)
   - P3: Evolució temporal RF des de KHP_History.json
   - P4: QC Levey-Jennings (desviació % vs recta vigent)
   - P5: Historial calibracions
   - **TOT des de JSON** — NO recalcula regressió

5. **`gui/widgets/analyze_panel/_helpers.py`**: Helpers compartits:
   - `format_calibration_comparison_html()`: taula HTML amb capçalera blava, deltes colorats
   - `compute_prediction_band()`: interval predicció 95% via scipy t-distribution

6. **`gui/widgets/analyze_panel/panel.py`** (Step 3):
   - Comparació usa helper compartit (amb fila equació)
   - Scatter: banda predicció 95%, línies RMS als residuals, etiquetes ppm

**Calibracions antigues** (pre-regression_data) no tindran les pàgines de scatter al PDF.
Es mostra un missatge informatiu i es genera igualment les pàgines 1, 3, 4, 5.

### Revisió calibració KHP (EN CURS)

**Bug crític trobat i fixat — finestra Savitzky-Golay a `find_peak_boundaries()`:**
- `hpsec_core.py` L1112: finestra SG era `n // 20 * 2 + 1` → proporcional al cromatograma (117 pt = 7.9 min)
- Amb finestra massa gran, el suavitzat aplanava el pic i les pendents al punt d'inflexió
  eren massa petites → la projecció tangent donava límits 14x massa amples (27σ vs 4σ esperats)
- **FIX**: finestra SG basada en temps (`sg_target_min = 0.7` min ≈ FWHM típic pic HPLC-SEC)
  → `sg_window = int(0.7 / dt_median)`, amb clamp a [7, n]
- Verificat: amb SG=11pt (0.73 min), projecció tangent dóna 4.1-4.4σ = correcte per gaussiana

**Resultats calibració COLUMN DOC (amb fix SG):**
- 36 entrades OK (de 123 totals), R²=0.977, slope=749, RF mediana=730, CV=16%
- Per concentració: 1ppm RF=724±109, 2ppm RF=726±27, 3ppm RF=828±3, 5ppm RF=749±77
- SEQs 256-274 COLUMN tenien v=100µL al manifest però àrees idèntiques a v=400µL
  → detecció automàtica VOL_SUSPECT: si COLUMN amb v=100 i àrea>400, corregir a v=400

**Calibració BP DOC — PENDENT, alineació temporal en curs:**
- R²=0.019, RF CV=165% — **NO ACCEPTABLE** (amb dades sense alinear)
- **Causa arrel**: el 2-TOC del masterfile és un flux continu de mesures TOC (una cada 4s).
  La Suite assigna files TOC a cada injecció HPLC via un desfase temporal (0-INFO "Net delay").
  Si el desfase és erroni o absent (nan), les files assignades no corresponen al cromatograma
  real → baselines de ~100 ppb (= nivell aigua MQ del TOC), pics desplaçats, àrees incorrectes.

**Delay HPLC↔TOC:**
- El delay (Net delay) és el paràmetre clau per assignar files TOC a cada injecció HPLC.
  Es llegeix del v11 original (full 0-CHECK) i s'aplica al MasterFile (full 0-INFO).
- `hpsec_migrate_master.py` calcula el delay a partir de les hores rellotge HPLC i TOC.
  **IMPORTANT**: aquest càlcul automàtic pot ser erroni (demostrat amb la 156: delay automàtic
  -3.01 vs delay real 7.67 del v11). Sempre verificar amb el v11 original.
- **156_SEQ_BP**: delay corregit de -3.01 a 7.67 min (del v11). Amb delay erroni, les files
  estaven desplaçades -175 rows (~11.7 min) i cada R1 agafava el pic de la injecció anterior.

**Flags de qualitat (khp_reintegrate_doc.py):**
- MULTI_PEAK: CR local (±5 min) < 0.70
- MULTI_PEAK_MILD: CR local 0.70-0.90
- T_RET_ANOMAL: COLUMN fora 18-28 min, BP fora 0-12 min
- VOL_SUSPECT: COLUMN v=100µL amb àrea>400 (probablement v=400µL erroni)
- FALLBACK_MAX: `find_peak_boundaries` ha caigut al fallback threshold

**Script diagnòstic: `khp_reintegrate_doc.py`:**
- Llegeix manifest JSON per baseline i volum (mateixa metodologia que la Suite)
- Extreu DOC del MasterFile 2-TOC amb `extract_doc_from_masterfile()`
- Integra amb `detect_main_peak()` sobre y_net = max(y_raw - baseline, 0)
- Genera CSV (226 entrades) + scatter plots a `REGISTRY/review/`

**Fitxers a REGISTRY/review/:**
- `khp_doc_reintegration.csv`: 226 entrades amb àrea, RF, flags, vol_corrected
- `scatter_doc_clean.png`: Area vs ug_DOC per mode amb recta regressió
- `153_SEQ_diagnostic.png`: Diagnòstic visual pics solapats KHP1/KHP500
- `peak_limits_comparison.png`: Comparació 5 mètodes de límits (tangent, inflexió, ±2σ, ±3σ, 5%)
- `chromatograms/*.png`: 226 cromatogrames individuals

**Regeneració MasterFiles BP:**
- `regenerate_bp_masterfiles.py`: pipeline complet (rawdata → MasterFile → delay → 4-TOC_CALC)
- Dry-run: 45 OK, 5 REVISAR, 3 sense HPLC
- Pendent: usuari verificant delays de les 5 REVISAR (111, 169, 221, 225, 277)
- **111_SEQ_BP és clau** (l'usuari ho marca explícitament)

**Detecció KHP per nom de mostra:**
- SEQs amb concentracions al nom sense "KHP": 111, 113, 114 (KHP pur), 148 (mostra+buff), 225 (HA/FA)
- Solució: convenció `_CAL` al nom de carpeta. Si la SEQ conté "CAL", tota injecció no-exclosa
  (MQ/NaOH/BUFFER/etc.) es tracta com a KHP. Implementat a `_extract_khp_from_masterfile()`.
- SEQs a renombrar per l'usuari: `111→111_CAL`, `113→113_CAL`, `114→114_CAL`
- La 148 i 225 NO són KHP pur (buffer i HA/FA respectivament) → no renombrar.

**Sistemàtica preparació KHP — IMPORTANT:**
- A partir de la **111_SEQ_BP** (inclosa), els KHP es preparen amb **pipetes Pasteur**.
- Les SEQs anteriors a la 111 (072B–109B) tenen sistemàtica desconeguda.
- Pot haver-hi **diferències sistemàtiques** en la preparació entre pre-111 i post-111.
- Considerar separar les rectes de calibració pre/post-111 o verificar si hi ha salt en RF.

**Resultats regressió BP (amb SEQs _CAL):**
- 281 entrades totals (123 COLUMN + 158 BP), 137 OK
- **114_SEQ_BP_CAL: R²=0.9954**, slope=681, intercept=-3.9 → RF_BP=681 (referència!)
  - Test linealitat volum: 5ppm × 6 volums (50-200µL). Resultat perfecte.
  - t_max coherent: 2.4-2.7 min. RF consistent per tots els volums (654-708).
  - RF_BP=681 és ~7% menys que RF_COLUMN=730 — diferència plausible camí hidràulic.
- **113_SEQ_BP_CAL: R²=0.858**, slope=1224, intercept=232
  - L'intercept alt (232) indica àrea de fons significativa a baixa concentració.
  - Àrees inflades per offset: a 0.05ppm, àrea=210 ≈ tot offset, no KHP real.
  - Slope corregit (1224) no és directament comparable, cal model amb intercept.
- **111_SEQ_BP_CAL: PROBLEMÀTICA** — R²=0.025
  - Àrees molt baixes (6-132) per TOTES les conc (incloent 5ppm, que a 114 dóna 335).
  - 16/29 entrades amb MULTI_PEAK flags.
  - Probable causa: delay no prou precís o 4-TOC_CALC no captura el pic complet.
  - Cal revisar: la 111 té 3 blocs de 16 inj (48 total), potser cada bloc necessita
    delay diferent (tèrmica del sistema canvia en 8h de seqüència).
- Totes BP (sense outlier 271): R²=0.31, slope=488, intercept=159
- Post-111 only: R²=0.44, slope=600, intercept=101

**Regressió combinada (millor estimació RF_BP):**
- De la 114 (R²=0.9954): **RF_BP_direct = 681 ± 20** (intercept ≈ 0)
- Coherent amb la tendència de les altres SEQs a concentracions altes (5ppm RF~683)

**Anàlisi integració BP (analyze_integration_bp.py) — COMPLETAT:**
- 6 mètodes comparats: thr1%, thr5%, tangent Agilent, bigauss 3s, bigauss analytic, trapezoid net
- **Tangent (Agilent) confirmat com a millor**: R²=0.978 combinat, 0.999 per 152 sola
- **152_SEQ_BP com a referència primària**: slope=817, intercept=11 (~0), R²=0.9992, n=10
- **156_SEQ_BP exclosa**: intercept=58 (possible contaminació DOC preparació)
- Diferència RF: COLUMN=628 vs BP=817 és efecte d'integració, no del detector
  (el detector TOC és el mateix; calculant àrea efectiva per µg: COLUMN=830, BP=817 = 1.6% diferència)

**Calibration_Reference.json ACTUALITZAT (2026-02-20):**
- `rf_mass_cal.direct.bp`: 915 → **817** (de 152_SEQ_BP tangent)
- `r2.bp`: 0.8213 → **0.9992**
- `n_points.bp`: 7 → **10**
- `intercept.direct.bp`: 0 (mantingut — intercept 152=11 ≈ negligible)

**Fix volums d'injecció (CRÍTIC):**
- `hpsec_migrate_master.py`: volum llegit de col N (Unnamed:13) del v11, NO hardcoded.
  Si no existeix → 0-INFO = "DESCONEGUT" + warning. Detecta volums variables.
- Eliminat BP guard a `hpsec_import.py`: ara l'heurístic index-13 s'aplica a TOTS els modes
- `get_injection_volume()` a `hpsec_calibrate.py`: nou param `manifest_volume` amb prioritat absoluta
- `register_calibration()`: rebutja entrades sense volum (en lloc de suposar 100µL)
- `quantify_sample()` a `hpsec_analyze.py`: warning si volum no ve del manifest
- Propagació `method` a `master_data` per futur ús
- **PENDENT**: els MasterFiles BP existents (generats amb l'antic hardcode BP=100) poden tenir
  volums incorrectes. Cal regenerar els afectats (especialment 107, i qualsevol BP amb v≠100).

### Comparació calibracions BP: 292 vs 152 (2026-02-21)

**292_SEQ_CAL_BP** (referència nova, 6 conc ben distribuïdes):
- slope=647, intercept=2.8, R²=0.9987, n=6 (0.1, 0.25, 0.5, 1, 2, 5 ppm)
- RF_mass molt consistent, intercept ≈ 0

**152_SEQ_BP** (referència antiga):
- slope=812, intercept=14.2, R²=0.9994, n=4 (0.25, 1, 3, 5 ppm)
- RF_mass inconsistent per concentració: 1334 a 0.25ppm vs 834 a 5ppm → suggereix offset de fons

**Comparació directa**: 292 dóna -20.3% menys RF que 152
- A 0.25ppm: 152 àrea=230 vs 292 àrea=174 → 152 inflada per possible offset DOC
- La 292 (6 punts, R²=0.999, bona distribució) és més fiable que la 152 (4 punts, offset)

**Evolució temporal RF:**
- BP: tendència -1.5 RF/SEQ, alta variabilitat (CV ~30%)
- COLUMN: pràcticament estable (tendència ~0/SEQ)

**PENDENT**: Actualitzar `Calibration_Reference.json` BP RF de 817 a ~650 (ref 292)
- L'usuari ha dit "de moment prenem nota, queda pendent d'actualitzar"

### Anàlisi 293_SEQ_CAL COLUMN (2026-02-21)

**Històric COLUMN (pre-293):**
- Global clean (30 entrades, 400µL, 0.25-5ppm): slope=785, intercept=3, R²=0.980
- Recent (>250): slope=751, intercept=40, R²=0.967
- 2ppm only (més estable): RF=797±42

**293_SEQ_CAL COLUMN:**
- 3 entrades (0.1, 0.25, 0.5 ppm a 400µL) — només concentracions baixes
- slope=620, intercept=41.3, R²=0.9982
- **vs referència actual (RF=628, intercept=81): -1.3%** — pràcticament idèntic!
- vs històric global (RF=785): -21.1%
- vs històric recent (RF=751): -17.5%
- **Conclusió**: la referència actual (RF=628+intercept=81) és correcta per COLUMN

### Bugs fixats sessió 2026-02-21

**Concentracions decimals (0.1, 0.25, 0.5 ppm):**
- `get_condition_key()`: `int(conc_ppm)` truncava 0.1→0, 0.25→0, 0.5→0
  → Fix: format decimal amb trailing zero stripping
- 8+ llocs GUI amb `:.0f` o `int(conc)` → tot canviat a `:g`
- Filtres concentració: tolerància absoluta ±1 ppm → relativa 10% (`max(0.01, conc*0.1)`)
- KHP_History.json: netejat entrada antiga amb conc=25.0, fixats 19 condition_keys truncats

**Timeout en KHP:**
- `validate_khp_quality()`: timeout WARNING ara → issues +100 (era: warnings +50)
- Simplificat: INFO=OK, WARNING/CRITICAL=outlier (no cal UI addicional)

**Gràfic històric:**
- Filtrat `qc_history` per concentració i volum abans de passar a `plot_calibration()`

### Refactor GlobalCalibrationPanel — Recta CAL + QC Monitor (2026-02-21)

**Motivació**: El panell barrejava totes les 250+ entrades KHP per fer regressió. Ara separa:
- SEQ_CAL (13 entrades) → Tab "Recta de Calibració" per construir/actualitzar calibració
- Producció (114 entrades) → Tab "Control de Qualitat" amb gràfic Levey-Jennings

**Fitxers modificats:**
- `hpsec_calibrate.py`: + `compute_calibration_fingerprint()` (SHA-256[:16]), + `requantify_analysis_json()` (recalcula ppm sense reprocessar)
- `hpsec_analyze.py`: estampa `calibration_fingerprint` al JSON d'anàlisi
- `hpsec_export.py`: + `patch_excel_calibration()` per patchejar Excels existents
- `gui/widgets/global_calibration_panel.py`: reescriptura completa (2 vistes: CalibrationLineView + QCMonitorView)
- `gui/models/sequence_state.py`: + camp `calibration_fingerprint`, + propietat `is_cal_stale`
- `gui/main_window.py`: connexió senyal `calibration_updated` → dashboard refresh
- `gui/widgets/dashboard_panel.py`: indicador ✔⟳ (taronja) quan calibració obsoleta

**Funcionalitats noves:**
- **CalibrationLineView**: selector SEQ_CAL amb checkboxes, regressió, scatter+residuals, stats per conc, comparació amb vigent, aplicar amb opció retroactiva + requantificació automàtica
- **QCMonitorView**: gràfic Levey-Jennings (desviació % vs recta), línies ±10%/±20%, tendència, indicador EN CONTROL/ATENCIÓ/FORA DE CONTROL
- **requantify_analysis_json()**: modifica NOMÉS ppm/fraccions als JSONs existents des de les àrees (que no canvien). Verificat: RF+10% → ppm -9.1%, àrees intactes
- **calibration_fingerprint**: patró idèntic a config_fingerprint — detecta canvi de calibració al dashboard

**Separació _CAL**: per convenció de nom, `"_CAL" in seq_name.upper()`. 13 entrades de 3 SEQs (111_CAL, 113_CAL, 114_CAL, 292_SEQ_CAL, 293_SEQ_CAL).

### Rename batman → irregular_top + fix integració pics irregulars (2026-02-21)

- Commit `9b12b28`: rename detect_batman→detect_irregular_top a 16 fitxers (168 ocurrències)
- Pre-repair a detect_main_peak: detectar pic irregular → reparar amb paràbola → find_peak_boundaries sobre senyal reparat → integrar sobre senyal original
- Verificat: KHP1 (1ppm) àrea 96.8→304.7, desviació -71%→-8.3%

### Redisseny Wizard — 4 Fases (2026-02-22)

**Implementat en worktree `claude/serene-williamson`, merged a main:**

**Fase 1: Renaming Calibrar → Verificar**
- `process_wizard_panel.py`: TAB_NAMES, tab_names, comments — totes les ocurrències

**Fase 2: Delay diagnostic tool (pas 2 Verificar)**
- `hpsec_delay.py`: backend Net delay (read, estimate impact, update MasterFile)
- `calibrate_panel/panel.py`: secció delay amb indicador shift (colors), slider ±15min,
  spinbox sincronitzat, preview "X files reassignades", botó "Aplicar i Reimportar"
- Cached timestamps per resposta instantània del slider

**Fase 3: Moure regressió SEQ_CAL del pas 2 al pas 3**
- `calibrate_panel/panel.py`: `_detect_seq_cal()` marca `is_seq_cal`, amaga UI normal,
  mostra `_seq_cal_info_group` + delay diagnostic. NO fa regressió (va al pas 3)
- `analyze_panel/panel.py`: secció completa regressió amb taula punts (checkboxes),
  scatter+residuals (matplotlib), RF/intercept/R²/RMS, comparació vigent, recalcular
- NO botó "Aplicar" (va al pas 4)

**Fase 4: Aplicar calibració al pas 4 (Revisar)**
- `review_summary_panel.py`: secció "APLICAR CALIBRACIÓ (SEQ_CAL)" amb:
  - Resum regressió, comparació HTML vigent vs nova, scatter miniatura
  - DateEdit valid_from, checkbox retroactiu, llista SEQs amb checkboxes
  - Botó "Aplicar com a Nova Calibració" → `add_calibration()` + `requantify_analysis_json()`
  - Dashboard refresh automàtic

### Wizard SEQ_CAL — Regressió al wizard (2026-02-21)

**Implementat flux complet per SEQ_CAL al wizard (versió original, ara refactored):**
- Detecció automàtica: `_detect_seq_cal()` (refactored from `_detect_and_run_seq_cal`)
- Regressió moguda de pas 2 a pas 3 (AnalyzePanel)
- Aplicació moguda de pas 2 a pas 4 (ReviewSummaryPanel)
- GlobalCalibrationPanel: consulta-only (sense aplicar/requantificar)

### Redisseny Wizard — Sessió 2026-02-22

**Fase 1: Rename Calibrar → Verificar (COMPLETAT)**
- `process_wizard_panel.py`: TAB_NAMES, tab_names dict
- `dashboard_panel.py`: STAGE_NAMES, columna headers, context menus
- `sequence_state.py`: Phase.CALIBRATE display
- `main_window.py`: comentari
- Commit: `0358c31`

**Fase 2: Delay diagnostic tool (COMPLETAT)**
- `hpsec_delay.py`: NOU — backend per gestió delay HPLC↔TOC
  - `read_current_delay(mf_path)`: llegeix 0-INFO B12
  - `estimate_delay_impact(mf_path, old_delay, new_delay)`: quantes files canvien
  - `update_masterfile_delay(mf_path, net_delay_min)`: actualitza + regenera 4-TOC_CALC + backup
- `gui/widgets/calibrate_panel/panel.py`: secció diagnòstic delay
  - `_build_delay_diagnostic_section()`: UI amb indicador shift, slider, spinbox, impacte, botó
  - `_update_delay_diagnostic(result)`: mostra per BP (sempre) o COLUMN (shift > 2 min)
  - `_on_delay_slider_changed/_on_delay_spin_changed`: sincronitzats bidireccional
  - `_update_delay_impact(new_delay)`: preview en temps real (quantes files canvien)
  - `_delay_apply_and_reimport()`: aplica delay → reimporta → re-verifica
  - Indicador qualitat: verd < 0.5 min, taronja 0.5-2 min, vermell > 2 min
  - Slider: ±15 min, pas 0.1 min
  - Integrat a _on_finished (normal + error path)

**Fase 3: Moure regressió SEQ_CAL al pas 3 — COMPLETAT** (merged)
**Fase 4: Aplicar calibració al pas 4 — COMPLETAT** (merged)

### Sessió 2026-02-22 (continuació) — UIB timeouts + fixes wizard

**Bug: Suite penjada al passar a Verificar (ensure_data_loaded al UI thread):**
- Símptoma: "al reimportar la 288 i passar a verificar cursor ocupat [...] (no respon)"
- Causa: `ensure_data_loaded()` bloquejava el thread principal (carregar MasterFile+CSV+DAD és lent)
- Fix: Mogut `ensure_data_loaded()` dins dels workers (threads):
  - `calibrate_panel/worker.py`: CalibrateWorker.run() crida ensure_data_loaded si data_deferred
  - `analyze_panel/worker.py`: AnalyzeWorker.run() crida ensure_data_loaded si data_deferred
  - `calibrate_panel/panel.py`: eliminat ensure_data_loaded del UI thread (L926-932)
  - `analyze_panel/panel.py`: eliminat ensure_data_loaded del UI thread (L433-440)

**Bug: Preload manifest camps incorrectes (process_wizard_panel.py):**
- `_preload_completed_stages()` creava `imported_data` amb:
  - `manifest.get("method")` → None (hauria de ser `manifest["sequence"]["method"]`)
  - `manifest.get("samples")` → **llista** no dict (crash a `ensure_data_loaded()` que espera `.items()`)
  - `manifest.get("masterfile_path")` → None (hauria de ser `manifest["master_file"]["path"]`)
- Fix: Reescrit per extreure de l'estructura anidada + convertir samples llista→dict
- Afecta a: totes les SEQs al carregar automàticament des del manifest (auto-load)

**UIB timeout estimation (NOU — hpsec_core.py + hpsec_analyze.py):**
- **Context**: El timeout del TOC (recàrrega xeringues Sievers M9e, ~74s cada ~77.2 min)
  afecta tant DOC Direct (gap temporal) com UIB (patró anòmal sense gap temporal).
  UIB mostra un pic espuri (~1.8 min durada, fins +75% sobre baseline) al mateix temps que el timeout Direct.
- **Problema**: No es pot detectar el timeout a UIB per gaps temporals (CSV continu).
  Cal estimar-lo des de DOC Direct o des del model predictiu.
- **Implementació 3 nivells**:
  1. `estimate_timeout_for_uib()` a `hpsec_core.py`: transfereix posicions timeout de DOC Direct a UIB,
     o usa model predictiu (`hpsec_planner.py`) amb T0 i sample_duration
  2. Integrat a `analyze_sample()` a `hpsec_analyze.py`: per cada mostra DUAL, estima UIB timeouts
  3. `_estimate_uib_timeouts_from_sequence()` a `hpsec_analyze.py`: post-processa tota la seqüència
     per extrapolar timeouts a injeccions sense DOC Direct via regressió lineal sobre patró observat
- **Verificació 288_SEQ**: drift consistent -1.4 min/inj (teòric -1.45 per COLUMN 78.65 min)
- **hpsec_planner.py**: mòdul existent (418 línies) amb model complet de predicció timeout, mai usat.
  Constants: TOC_CYCLE_MIN=77.2, TOC_TIMEOUT_SEC=74, SAMPLE_DURATION_CURRENT=78.65
- Zona d'anomalia UIB: t_timeout - 0.2 min a t_timeout + 1.8 min (pre/post marges)

**Investigació 288_SEQ Export3D:**
- Primera importació no reconeixia DAD → reimportació correcta
- Causa: manifest antic sense info DAD; reimportació regenera manifest amb Export3D detectat
- 33 fitxers Export3D correctament detectats al reimportar (dad_source=export3d)

### Canvis sessió 2026-02-21 (continuació)
- **GlobalCalibrationPanel refactor**: 2 vistes (CalibrationLineView + QCMonitorView)
  - `hpsec_calibrate.py`: `compute_calibration_fingerprint()`, `requantify_analysis_json()`
  - `hpsec_export.py`: `patch_excel_calibration()` (openpyxl cell-level patching)
  - `gui/widgets/global_calibration_panel.py`: reescriptura completa (~700 línies)
  - `gui/models/sequence_state.py`: `calibration_fingerprint`, `is_cal_stale`
  - `gui/main_window.py`: `calibration_updated` signal → dashboard refresh
  - `gui/widgets/dashboard_panel.py`: indicador ✔⟳ per SEQs amb calibració obsoleta
  - Commit: `2b43eb0`
- **Fix import re-read MasterFile**: `import_from_manifest(load_data=False)` per auto-load
  - `hpsec_import.py`: param `load_data`, funció `ensure_data_loaded()`
  - `gui/widgets/import_panel/worker.py`: `load_data` param
  - `gui/widgets/import_panel/panel.py`: `_auto_load_from_manifest()` amb `load_data=False`
  - `gui/widgets/analyze_panel/panel.py`: `ensure_data_loaded()` abans d'analitzar
  - `gui/widgets/calibrate_panel/panel.py`: `ensure_data_loaded()` en 3 punts
  - Speedup: 10s → 1ms (factor 10000x) per auto-load SEQs ja importades

### Canvis sessió 2026-02-21 (inici)
- `hpsec_calibrate.py`: fix `get_condition_key()` decimals, fix timeout severity, fix concentration filter tolerance
- `gui/models/sequence_state.py`: `:g` format per concentracions
- `gui/widgets/calibrate_panel/panel.py`: `:g` formats (5 llocs), tolerància relativa (2 llocs), filtre qc_history
- `gui/widgets/analyze_panel/panel.py`: `:g` format KHP display
- `gui/widgets/history_panel.py`: `concs.add(conc)` float, `:g` formats (3 llocs), tolerància relativa
- `gen_cal_analysis.py`: script diagnòstic 4 pàgines PDF (292 vs 152 BP + temporal RF)
- Commits: `faa98f2` (decimal display), `2afdc14` (timeout + history filter)

### Canvis sessió anterior (2026-02-20)
- `Calibration_Reference.json`: BP rf=817, R²=0.999, n=10 (ref: 152_SEQ_BP tangent)
- `hpsec_migrate_master.py`: volum de col N del v11 (no hardcoded), warning si absent
- `hpsec_import.py`: eliminat BP guard volum, warning si cap injecció té volum, propagació method
- `hpsec_calibrate.py`: `get_injection_volume(manifest_volume=)`, `register_calibration()` rebutja vol=None
- `hpsec_analyze.py`: warning si volum no al manifest (heurístic com a fallback)
- `analyze_integration_bp.py`: script comparació 6 mètodes integració BP

### Sessions anteriors (resum)
- Carpetes renombrades: `111→111_CAL`, `113→113_CAL`, `114→114_CAL`
- `regenerate_bp_masterfiles.py`: regeneració MasterFiles + delay + 4-TOC_CALC
- `fix_masterfile_delay.py`: 45/53 SEQs amb delay mesurat aplicat
- `khp_reintegrate_doc.py`: lectura directa MasterFile, mode _CAL
- `hpsec_config.py`, `hpsec_import.py`, `hpsec_migrate_master.py`: pre-margin 1.5 min
- Pipeline 254→DOC: `analizar_khp_data()` reescrit, recalibrate_all_khp.py, calibration_review.py
- validate_khp_quality: 5 nous checks (bigaussian, t_ret, mismatch, no_dad)
- fit_calibration_from_history: mode="ALL", signal="254"
- Dashboard: columna Inj amb detecció importació incompleta
- Derivative integration: find_peak_boundaries amb projecció tangent Agilent
- KHP DAD 254nm: fallback robust des de MasterFile 3-DAD_KHP
- Startup optimization: 24s → <1s (lazy tabs, metadata-only JSON)
- Config panel: 3 tabs, badges impacte, fingerprint SHA-256
