# Informe comparatiu de calibracions COLUMN vs BP

## Per a què serveix

Aquesta carpeta conté la **cadena completa** que genera l'informe comparatiu de les
seqüències de calibratge, en els dos modes cromatogràfics (COLUMN i BP) i els tres
canals (DOC, UIB, 254 nm).

Seqüències: COLUMN **293** (20/02) → **305** (07/07) → **306** (14/07), 400 µL ·
BP **292** (19/02) → **304** (07/07), 100 µL.
Entre la 305 i la 306 es va substituir la columna cromatogràfica.

L'informe separa **dos efectes independents**:

1. **Resposta del detector TOC ×8** — present a les tres seqüències noves
   (305 ×7,5; 306 ×7,6; 304 ×8,2 a 5 ppm) → és del detector.
2. **Inestabilitat de la columna de la 306** — fons ×47,5, deriva i caigudes de base,
   254 nm derivant fins a −7 mAU, UIB saturat ja a la línia de base. La 305, amb el
   mateix detector i una columna diferent, no la presenta (recta lineal R² 0,998 de
   0,25 a 5 ppm), i el mode BP tampoc (bypass de columna).

El pic de 254 nm és invariant a la 305 (×1,10) i a la 304 (×1,09) → mateix analit i
mateixa quantitat injectada.

## Com regenerar l'informe

La cadena té **tres passos**, en aquest ordre:

```
cd C:\Users\maria\Proyectos\HPSEC\informe_calibracions
python -X utf8 extreu_dades.py      # seqs -> cal_compare_data.json
python -X utf8 fes_figures.py       # seqs -> *.png
python -X utf8 gen_docx.py          # dades + figures -> .docx
```

Genera `C:\Users\maria\Proyectos\Informe_calibracions_COLUMN_BP_v5.docx`.
(Si el fitxer està obert al Word, tanca'l abans o canvia la variable `OUT` de `gen_docx.py`.)

**Per afegir una seqüència nova:** processa-la a la Suite (ha d'existir el seu
`CHECK/data/calibration_result.json`), afegeix-la a `SEQS_COLUMN` / `SEQS_BP` als dos
primers scripts, i torna a executar els tres passos.

## Contingut de la carpeta

**Scripts (la cadena):**
- `extreu_dades.py` — llegeix el `calibration_result.json` de cada SEQ i escriu els JSON
  de dades. `--validar` comprova que reprodueix les files ja publicades; `--bp` regenera
  també el fitxer de BP (per defecte NO, vegeu l'avís de sota).
- `fes_figures.py` — genera els 6 PNG de senyals crus. `--bp` inclou també els de BP.
- `gen_docx.py` — munta el .docx a partir dels JSON i els PNG.

**Dades (sortida d'`extreu_dades.py`):**
- `cal_compare_data.json` — injecció a injecció, mode COLUMN (293, 305, 306).
- `bp_data.json` — injecció a injecció, mode BP (292, 304).

**Figures (sortida de `fes_figures.py`):**
- `raw_doc_continu.png` · `raw_254_overlay.png` · `raw_uib_overlay.png` — COLUMN,
  un panell per seqüència.
- `bp_doc_continu.png` · `bp_254_overlay.png` · `bp_uib_overlay.png` — BP.

**Altres:**
- `integracio_306_diagnostic.png` — diagnòstic de la finestra d'integració de la 306
  (no forma part de l'informe; vegeu l'avís de sota).
- `*.v4bak` — còpies de seguretat de les dades i figures de la versió v4.

## Avisos importants

**`bp_data.json` està congelat.** Els `calibration_result.json` de la 292 i la 304 s'han
reprocessat des de la v4 i ara contenen menys rèpliques (292: 9 vs 12 publicades; 304: 8
vs 10). Regenerar-lo amb `--bp` **canviaria els números de BP de l'informe**. Per això
`extreu_dades.py` no el toca per defecte. Cal decidir si es reprocessa l'informe sencer
amb les dades noves de BP.

**L'àrea del DOC de la 306 no és fiable.** La integració de la Suite hi obre finestres de
10,6 a 54,6 min (el pic té un FWHM d'1,5 min; la finestra correcta són ~5–6 min) perquè,
amb la línia de base alta i derivant, la projecció tangent no troba el final del pic i
s'empassa senyal que no és KHP. Reintegrant amb finestra estreta ancorada al 254, la
recta de la 306 passa de R² 0,607 a **R² 0,9996** (RF_mass 9.319). Les àrees de la 306 que
apareixen a l'informe són les de la Suite, tal com les calcula el programa avui. La 293 i
la 305 no estan afectades (sobre-integració ×1,1–1,4 i base plana).

## Estructura de l'informe

§1 Resum · §2 Taula 1 (comparativa per seqüència) · §3 efecte comú del TOC ·
§4 efecte de la columna (exclusiu de la 306) · §5 Figures 1–6 (senyals crus) ·
§6 Taules 2–6 (detall injecció a injecció per seqüència).
Sense apartat de conclusions (per encàrrec). Taules i figures numerades, amb peu i citades al text.

Requisits: `pip install python-docx` · pandas · numpy · scipy · matplotlib.
