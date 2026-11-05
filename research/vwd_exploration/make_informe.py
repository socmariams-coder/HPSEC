# -*- coding: utf-8 -*-
"""Genera l'informe HTML del VWD (307/308) amb figures incrustades (base64)."""
import base64, os

BASE = r"C:\Users\maria\Proyectos\HPSEC\research\vwd_exploration"
OUT  = os.path.join(BASE, "informe_vwd_307_308_v1.html")

def img(rel):
    p = os.path.join(BASE, rel)
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b}"

# figures triades
F_PANO   = img(r"out_308_SEQ_BP\_PANORAMA_308_SEQ_BP.png")
F_MATCH_S= img(r"out_308_SEQ_BP\matched\LQ4174_rep1_matched.png")   # mostra
F_MATCH_B= img(r"out_308_SEQ_BP\matched\MQ1_rep1_matched.png")      # blanc milliQ
F_OVER   = img(r"out_308_SEQ_BP\LQ4174_rep1_overlay.png")           # 3 panells
F_PERIO  = img(r"out_308_SEQ_BP\reactiu\_PERIODOGRAMA_308_SEQ_BP.png")
F_LOCK   = img(r"out_308_SEQ_BP\reactiu\_FASE_LOCK_308_SEQ_BP.png")
F_220254 = img(r"out_308_SEQ_BP\reactiu\_220vs254_308_SEQ_BP.png")

HTML = f"""<!doctype html>
<html lang="ca"><head><meta charset="utf-8">
<title>Informe VWD 307/308</title>
<style>
  @page {{ size: A4; margin: 20mm 18mm 18mm 18mm; }}
  * {{ box-sizing: border-box; }}
  html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  body {{ font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
         color: #1a1a1a; font-size: 10.5pt; line-height: 1.5; margin: 0; }}
  h1 {{ font-size: 19pt; color: #1f3b73; margin: 0 0 2px; line-height: 1.2; }}
  .sub {{ color: #555; font-size: 10pt; margin-bottom: 4px; }}
  .meta {{ color: #777; font-size: 8.5pt; border-bottom: 2px solid #1f3b73;
           padding-bottom: 10px; margin-bottom: 16px; }}
  h2 {{ font-size: 13pt; color: #1f3b73; margin: 22px 0 6px;
        border-bottom: 1px solid #d5dced; padding-bottom: 3px; }}
  h3 {{ font-size: 11pt; color: #27406e; margin: 15px 0 3px; }}
  p {{ margin: 6px 0; }}
  ul {{ margin: 6px 0 6px 0; padding-left: 20px; }}
  li {{ margin: 3px 0; }}
  figure {{ margin: 12px 0 6px; text-align: center; page-break-inside: avoid; }}
  figure img {{ max-width: 100%; border: 1px solid #e3e3e3; border-radius: 3px; }}
  figcaption {{ font-size: 8.8pt; color: #555; margin-top: 4px; text-align: left; }}
  figcaption b {{ color: #1f3b73; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 10px 0; }}
  caption {{ caption-side: top; text-align: left; font-size: 8.8pt; color: #555;
             margin-bottom: 4px; }}
  caption b {{ color: #1f3b73; }}
  th, td {{ padding: 5px 8px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
  thead th {{ border-bottom: 2px solid #1f3b73; color: #1f3b73; }}
  tbody tr:nth-child(even) {{ background: #f6f8fc; }}
  td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .box {{ background: #f6f8fc; border-left: 4px solid #1f3b73;
          padding: 9px 13px; margin: 12px 0; border-radius: 0 3px 3px 0; }}
  .box.key {{ border-left-color: #1f8a4c; background: #f1faf4; }}
  .box.warn {{ border-left-color: #c0392b; background: #fcf2f0; }}
  .box .lbl {{ font-weight: 700; font-size: 8.5pt; letter-spacing: .4px;
               text-transform: uppercase; color: #1f3b73; }}
  .box.key .lbl {{ color: #1f8a4c; }} .box.warn .lbl {{ color: #c0392b; }}
  code {{ background: #eef1f6; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }}
  .foot {{ margin-top: 22px; border-top: 1px solid #d5dced; padding-top: 8px;
           font-size: 8pt; color: #888; }}
  .pb {{ page-break-before: always; }}
</style></head>
<body>

<h1>Detector VWD (UV 220/254 nm) aigües avall del DOC</h1>
<div class="sub">Anàlisi del senyal del nou detector enfront del senyal actual — seqüències 307 i 308</div>
<div class="meta">HPSEC · LEQUIA–UdG · mètode EN DESENVOLUPAMENT · exploració externa (fora de la Suite) · 5 de novembre de 2026</div>

<div class="box">
  <div class="lbl">En una frase</div>
  Tal com està connectat ara (cel·la UV just després del detector de carboni), el VWD
  <b>no dóna encara un senyal d'analit aprofitable</b>: el domina una oscil·lació de base
  instrumental de període ~2,25 min que neix dins o després del DOC, no a la columna.
  No és un problema del mètode d'anàlisi ni de la longitud d'ona: és de muntatge físic.
</div>

<h2>1. Objecte i context</h2>
<p>La sortida del detector de carboni orgànic (DOC; analitzador de COT Sievers M9e, per
oxidació UV amb persulfat) s'ha connectat a un <b>detector UV de longitud d'ona variable
(VWD)</b> que mesura a dues longituds d'ona, <b>220 nm (canal VWD1A)</b> i <b>254 nm
(canal VWD1B)</b>. És un muntatge <b>en desenvolupament</b>. Per decisió de treball es
tracta <b>a banda</b> i <b>no s'introdueix a la Suite</b>; la Suite continua processant
només DOC i DAD.</p>
<p>L'objectiu d'aquest informe és caracteritzar el senyal del VWD i comparar-lo <b>a fons</b>
amb el senyal que ja tenim (el DOC net i el detector de diodes DAD que la Suite processa),
per decidir si el nou detector aporta informació i en quines condicions.</p>
<p>La peça clau del muntatge per interpretar els resultats és l'ordre del flux:
el <b>DAD</b> mesura <b>en línia, aigües amunt</b> (just després de la columna); el
<b>DOC</b> va al mig; i el <b>VWD</b> mesura <b>aigües avall del DOC</b>. El DAD i el VWD
miren els <b>mateixos</b> 220 i 254 nm, però en <b>dos punts diferents del circuit</b>.
Aquesta coincidència de longitud d'ona és la que permet localitzar on apareix qualsevol
artefacte.</p>

<h2>2. Dades i mètode</h2>
<p>S'han analitzat les dues úniques seqüències que porten el VWD amb els dos canals actius,
<b>307 i 308</b> (24 rèpliques cadascuna: patrons KHP, mostres LQ417x i blancs de milliQ).
De cada injecció s'extreu el senyal VWD dels fitxers <code>VWD1A</code>/<code>VWD1B</code>
i es posa al costat del <b>DOC net</b> i del <b>DAD</b> que la Suite <b>ja processa</b>
(comparació equivalent, sobre el pipeline real). L'eina viu a
<code>research/vwd_exploration/</code> i no toca cap codi de producció.</p>
<p>Per cada rèplica es mesura: període i amplitud de l'oscil·lació del VWD, relació
senyal/soroll dins la finestra d'elució enfront de la base (SNR finestra/base), correlació
i desfasament (lag) del VWD contra el DOC i contra el DAD, i la relació d'amplituds entre
220 i 254 nm. Les seqüències 299–306 no es tracten aquí perquè encara no porten els dos
canals VWD.</p>

<h2>3. Resultats</h2>

<h3>3.1 El senyal del VWD el domina una oscil·lació de base</h3>
<p>En totes dues seqüències, el VWD a 220 nm està dominat per una <b>oscil·lació periòdica
de línia de base</b> de període <b>~2,25 min</b> (present en 21 de 24 rèpliques a la 307 i
18 de 24 a la 308; la resta cau en harmònics de 4,5 min). L'amplitud d'aquesta ondulació
dins la finestra d'elució del DOC és pràcticament <b>la mateixa</b> que fora: la relació
senyal/base és de només <b>≈1,0–1,2</b> (Taula 1). És a dir, el "pic" que s'intueix dins
la finestra <b>no és un pic d'analit</b>, sinó un lòbul més de l'oscil·lació (Figura 1).</p>

<table>
  <caption><b>Taula 1.</b> Mètriques agregades (mediana de 24 rèpliques per seqüència).</caption>
  <thead><tr><th>Mètrica</th><th>Seq 307</th><th>Seq 308</th><th>Lectura</th></tr></thead>
  <tbody>
    <tr><td>Període oscil·lació VWD1A (220)</td><td class="n">2,25 min</td><td class="n">2,25 min</td><td>cicle regular, molt per sota d'una injecció (15 min)</td></tr>
    <tr><td>SNR finestra/base · 220 nm</td><td class="n">1,20</td><td class="n">1,00</td><td>≈1 → sense pic per sobre de la base</td></tr>
    <tr><td>SNR finestra/base · 254 nm</td><td class="n">1,20</td><td class="n">1,22</td><td>ídem</td></tr>
    <tr><td>Correlació VWD–DOC (finestra)</td><td class="n">0,95</td><td class="n">0,95</td><td>alta, però el lag no és estable (vegeu 3.4)</td></tr>
    <tr><td>Relació amplitud 220/254</td><td class="n">—</td><td class="n">~100×</td><td>coherent amb l'absortivitat (220 &gt; 254)</td></tr>
    <tr><td>Desfasament 220↔254</td><td class="n">~0 min</td><td class="n">~0,06 min</td><td>els dos canals oscil·len <b>en fase</b></td></tr>
  </tbody>
</table>

<h3>3.2 El canal de 254 nm (VWD1B) no és mort</h3>
<p>El canal de 254 nm <b>sí que dóna senyal</b>: a la seva pròpia escala val ~0,5–1,4 mAU,
unes <b>100 vegades menys</b> que el de 220 nm (Taula 1), cosa coherent amb el fet que
l'absorció UV a 220 és molt més gran que a 254. Porta <b>la mateixa oscil·lació</b> i
<b>en fase</b> amb el de 220. (Correcció d'una lectura inicial que el donava per "pla":
ho semblava només perquè es dibuixava a l'escala del 220, que l'aixafa.)</p>

<figure>
  <img src="{F_OVER}" alt="overlay mostra">
  <figcaption><b>Figura 1.</b> Mostra LQ4174 (seq 308). Esquerra: VWD (220/254) normalitzat
  sobre el DOC — l'ondulació del VWD no segueix la forma del pic de carboni. Centre: magnitud
  real; el 220 oscil·la ±40–100 mAU, el 254 el mateix patró a escala ~100× menor. Dreta: la
  relació espectral 220/254 és estable, pròpia d'una base, no d'un pic que elueix.</figcaption>
</figure>

<h3>3.3 On neix l'oscil·lació: comparació a la mateixa longitud d'ona</h3>
<p>Aquesta és la prova decisiva. Com que el DAD mesura els <b>mateixos</b> 220 i 254 nm
però <b>aigües amunt</b>, podem mirar cada longitud d'ona als dos punts del circuit:</p>
<ul>
  <li>Al <b>mateix</b> 220 nm, el <b>DAD és net</b> (pic definit, base plana) i el
  <b>VWD oscil·la</b> (±100 mAU). La diferència entre els dos no pot ser la mostra ni la
  longitud d'ona: és el <b>tram del circuit</b> que hi ha entremig (Figura 2).</li>
  <li>En un <b>blanc de milliQ</b> (aigua pura, res a absorbir) el DAD és pla a 220 i 254,
  però el <b>VWD segueix oscil·lant igual</b>. Si fos cromatografia o mostra, en un blanc
  no hi hauria res (Figura 3).</li>
</ul>

<figure>
  <img src="{F_MATCH_S}" alt="matched mostra">
  <figcaption><b>Figura 2.</b> Mostra LQ4174 (seq 308), mateixa longitud d'ona als dos punts
  del flux. Dalt 220 nm, baix 254 nm; en gris el <b>DAD (amunt, net)</b>, en color el
  <b>VWD (avall del DOC, ondulat)</b>; ombra blava = finestra del pic de carboni.</figcaption>
</figure>

<figure>
  <img src="{F_MATCH_B}" alt="matched blanc">
  <figcaption><b>Figura 3.</b> Blanc de milliQ (seq 308). El DAD és pla a 220 i 254 (no hi ha
  res a absorbir) però el <b>VWD oscil·la exactament igual</b> que en una mostra → l'ondulació
  <b>no depèn de la mostra</b>.</figcaption>
</figure>

<div class="box key">
  <div class="lbl">Localització</div>
  L'oscil·lació s'introdueix <b>entre el DAD (net, aigües amunt) i el VWD (aigües avall)</b>,
  és a dir <b>dins o just després del detector de carboni (DOC)</b>. No és cromatografia, ni
  analit, ni longitud d'ona: és la <b>cel·la del VWD i el tram que l'alimenta</b>.
</div>

<h3>3.4 No hi ha una retenció estable entre VWD i DOC</h3>
<p>Si el VWD veiés el mateix analit que el DOC, hi hauria un desfasament fix i repetible
entre els dos. No el trobem: el lag de correlació creuada VWD↔DOC <b>dispersa de −3 a +3 min</b>
i en prop d'un terç de les rèpliques arriba al límit de la cerca (±3 min), senyal que no
hi ha un màxim de correlació real (Figura 4, esquerra). La correlació alta de la Taula 1
(Figura 4, dreta) és un efecte de normalitzar dues corbes ondulades, no una coincidència de pics.</p>

<figure>
  <img src="{F_PANO}" alt="histogrames lag i correlacio">
  <figcaption><b>Figura 4.</b> Distribució sobre les 24 injeccions de la seq 308.
  <b>Esquerra</b>: desfasament VWD–DOC; s'escampa de −3 a +3 min i s'amuntega als extrems de
  la cerca (±3 min) → <b>no hi ha un retard fix</b> entre els dos detectors. <b>Dreta</b>: la
  correlació de forma es concentra cap a 0,95, però és correlació entre dues ondulacions, no
  coincidència de pics d'analit.</figcaption>
</figure>

<h3>3.5 Caracterització de l'oscil·lació</h3>
<p>L'ondulació és <b>regular i reproduïble</b> entre injeccions (Figura 5), i els canals
de 220 i 254 nm hi van <b>en fase</b> i amb amplituds en relació ~100× constant (Figura 6).
Tot plegat descriu una <b>modulació comuna de tot el senyal</b> (com una pulsació de base
que afecta per igual les dues longituds d'ona), no un fenomen espectral que distingiria un
analit.</p>

<figure>
  <img src="{F_PERIO}" alt="periodograma">
  <figcaption><b>Figura 5.</b> Periodograma del VWD (seq 308): concentració d'energia al
  voltant de ~2,25 min, consistent entre rèpliques.</figcaption>
</figure>

<figure>
  <img src="{F_220254}" alt="220 vs 254">
  <figcaption><b>Figura 6.</b> 220 nm enfront de 254 nm: mateixa forma, desfasament ~0,
  relació d'amplitud ~100× estable → modulació comuna de base, no firma espectral d'analit.</figcaption>
</figure>

<h2>4. ¿Ho explica la injecció de reactiu del DOC? — hipòtesi refutada</h2>
<p>Com que el detector de carboni injecta <b>àcid</b> i <b>oxidant</b> mitjançant bombes de
xeringa, es va plantejar que l'oscil·lació del VWD fos el <b>cicle de dosificació</b> del
reactiu. L'anàlisi del senyal ho semblava suportar (ondulació periòdica present també en
blancs). <b>Però amb la dosi real llegida del full de DOC exportat, no lliga.</b></p>
<p>Del full <code>2-TOC</code> del MasterFile (Sievers M9e):</p>
<ul>
  <li><b>Cabal constant</b>: àcid 5 µL/min, oxidant 2 µL/min (consignes, no modulades).</li>
  <li>L'únic esdeveniment periòdic del reactiu és la <b>recàrrega de la xeringa</b>: 6 cops
  en tota la seqüència, cada <b>~77 min</b> (~74 s d'aturada), coherent amb buidar una
  xeringa a aquest cabal.</li>
</ul>

<div class="box warn">
  <div class="lbl">Conclusió del punt 4</div>
  Una pertorbació cada <b>~77 min</b> no pot generar una oscil·lació que apareix
  <b>diversos cops dins de cada injecció</b> (~15 min). L'oscil·lació del VWD <b>no és la
  dosificació de reactiu</b>. (El ~2,3 min d'un primer càlcul per FFT era una mesura poc
  robusta, amb deriva lenta i ondulació barrejades; no coincideix amb cap cicle real del reactiu.)
</div>

<h2>5. Diagnòstic</h2>
<p>El que les dades <b>sí</b> permeten afirmar:</p>
<ul>
  <li>El VWD, tal com està connectat, <b>no aporta encara un senyal d'analit</b> en 307/308
  (SNR finestra/base ≈1; sense pic per sobre de la base; sense retenció estable amb el DOC).</li>
  <li>El canal de 254 nm funciona (~100× menor que 220, coherent amb l'absortivitat), però
  arrossega la mateixa oscil·lació.</li>
  <li>L'artefacte està <b>localitzat</b>: neix entre el DAD net i el VWD → <b>dins o després
  del DOC</b>, a la cel·la del VWD i el seu tram d'alimentació. No és el mètode d'anàlisi,
  ni la longitud d'ona, ni la mostra, ni la dosificació de reactiu.</li>
</ul>

<h2>6. Obert i següents passos</h2>
<p>El que <b>no</b> es pot afirmar des de les dades és la causa física concreta de la
pulsació. Són hipòtesis <b>per verificar amb la tècnica</b> (no confirmades per aquestes dades):
ripple de lliurament de les bombes de xeringa dobles, bombolles generades a l'oxidació UV, o
el cicle tèrmic/de mesura del propi M9. El següent pas natural és <b>d'instrumentació</b>:
revisar la instal·lació física del VWD (posició de la cel·la, contrapressió, desgasificació,
estabilitat de flux a la sortida del DOC).</p>
<p>Com a comprovació addicional sobre dades que queda a l'abast: el full de DOC porta la
sincronització HPLC↔TOC (<code>0-INFO</code>: Hora HPLC / Hora TOC / flush), que permet
col·locar cada injecció del VWD sobre el rellotge real del TOC i mirar si les 6 recàrregues
de xeringa (~77 min) deixen cap marca a les injeccions que les travessen. L'eina queda a
punt per refer tota l'anàlisi automàticament amb dades noves quan es reajusti el muntatge.</p>

<div class="foot">
  HPSEC Suite · LEQUIA–UdG · informe generat a partir de l'eina
  <code>research/vwd_exploration/</code> (figures regenerables) · seqüències 307_SEQ_BP i
  308_SEQ_BP · mètode en desenvolupament, no integrat a la Suite · v1 · 2026-11-05.
</div>

</body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("escrit:", OUT, f"({len(HTML)//1024} KB)")
