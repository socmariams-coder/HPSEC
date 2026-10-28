#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPSEC Viewer (HTML) — generador de visor de cromatogrames.

Fa la feina MÍNIMA a partir del MasterFile cru i genera un HTML autocontingut
per NOMÉS visualitzar: segmenta el DOC per injecció, l'alinea amb el DAD
(reaprofitant el motor de la Suite) i deixa un visor on pots passar de mostra
a mostra amb facilitat, amb les fraccions ombrejades (estil ponència).

NO calibra, NO quantifica. Només import + anàlisi qualitativa (do_quantify=False)
per obtenir les traces alineades i les finestres de fracció.

Ús:
    python hpsec_viewer_build.py 290_SEQ 293_SEQ_CAL 306_SEQ_CAL
    python hpsec_viewer_build.py C:/ruta/a/290_SEQ --out visor.html
    python hpsec_viewer_build.py                # llista les seqüències disponibles

Obre el fitxer HTML resultant amb el navegador (doble clic).
"""

import os
import sys
import json
import html
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import hpsec_import as I      # noqa: E402
import hpsec_analyze as A     # noqa: E402
import hpsec_config as C      # noqa: E402


# Paleta de fraccions (to suau, estil ponència)
FRACTION_COLORS = {
    "BioP": "#d9e8f5",
    "HS":   "#e6dcef",
    "BB":   "#fbe6d5",
    "SB":   "#e2efd9",
    "LMW":  "#f0eede",
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


def _default_folder():
    try:
        fs = C.get_data_folders()
        if fs:
            return fs[0]
    except Exception:
        pass
    for cand in (r"C:/Users/maria/Proyectos/Dades3",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dades3")):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


def _resolve_seq(arg, folder):
    """Accepta nom de seqüència o ruta completa; retorna ruta a la carpeta."""
    if os.path.isdir(arg):
        return os.path.abspath(arg)
    cand = os.path.join(folder, arg)
    if os.path.isdir(cand):
        return cand
    return None


def _downsample(t, *ys, target=1200):
    """Redueix la densitat de punts (stride uniforme) per aprimar l'HTML."""
    n = len(t)
    if n <= target:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, target).round().astype(int)
    return (np.asarray(t, float)[idx],
            *[None if y is None else np.asarray(y, float)[idx] for y in ys])


def _col(df, name):
    for c in df.columns:
        if str(c).strip() == name:
            return np.asarray(df[c], float)
    return None


# Fallback si la config no és accessible (p. ex. exe aïllat sense el json):
# valors canònics del pipeline (hpsec_config.json).
FRACTION_FALLBACK = {
    "BioP": ("Biopolímers", 12.0, 18.65),
    "HS":   ("Àcids Húmics", 18.65, 23.0),
    "BB":   ("Building Blocks", 23.0, 29.0),
    "SB":   ("Salt Boundary", 29.0, 32.0),
    "LMW":  ("Low Molecular Weight", 32.0, 78.65),
}


def get_fractions():
    out = []
    try:
        cfg = C.get_config()
        for key in FRACTION_ORDER:
            fr = cfg.get("time_fractions", default={}).get(key)
            if not fr:
                continue
            out.append({
                "key": key,
                "name": str(fr.get("name", key)),
                "start": float(fr["start"]),
                "end": float(fr["end"]),
                "color": FRACTION_COLORS.get(key, "#eeeeee"),
            })
    except Exception:
        out = []
    if not out:  # fallback
        for key in FRACTION_ORDER:
            name, a, b = FRACTION_FALLBACK[key]
            out.append({"key": key, "name": name, "start": a, "end": b,
                        "color": FRACTION_COLORS.get(key, "#eeeeee")})
    return out


def build_items_for_seq(seq_path):
    """Importa + analitza una SEQ i retorna la llista d'items (mostra·rèplica)."""
    seq_name = os.path.basename(seq_path.rstrip("/\\"))
    items = []
    imp = I.import_sequence(seq_path)
    if imp.get("data_deferred"):
        I.ensure_data_loaded(imp)
    # config=None -> el motor usa DEFAULT_PROCESS_CONFIG (dict pla), EXACTAMENT com
    # la Suite (AnalyzeWorker tampoc passa config). NO passar un ConfigManager: el
    # camí principal (detect_main_peak) fa config.get("clau", default_posicional)
    # amb semàntica de dict pla; un ConfigManager interpretaria el default com a 2a
    # clau i retornaria None -> petaria a hpsec_core. El crash de mostres PR
    # ("dict.get() takes no keyword arguments") s'arregla a _should_skip_quantification.
    res = A.analyze_sequence(imp, {}, do_quantify=False)
    sg = res.get("samples_grouped") or {}

    for sname, s in sg.items():
        stype = s.get("sample_type", "")
        reps = s.get("replicas") or {}
        sel = (s.get("selected") or {}).get("doc", "1")
        for rk in sorted(reps.keys()):
            r = reps[rk]
            t_doc = r.get("t_doc")
            y_doc = r.get("y_doc_net")
            if t_doc is None or y_doc is None or len(t_doc) < 10:
                continue
            t_doc = np.asarray(t_doc, float)
            y_doc = np.asarray(y_doc, float)
            y_uib = r.get("y_doc_uib_net")
            y_uib = np.asarray(y_uib, float) if y_uib is not None and len(y_uib) == len(t_doc) else None

            # DAD 254 interpolat sobre l'eix del DOC (mateix rellotge, ja alineat)
            a254 = None
            df = r.get("df_dad")
            if df is not None and hasattr(df, "columns"):
                tcol = _col(df, "time (min)")
                y254 = _col(df, "254")
                if tcol is not None and y254 is not None:
                    a254 = np.interp(t_doc, tcol, y254, left=np.nan, right=np.nan)

            td, yd, yu, ya = _downsample(t_doc, y_doc, y_uib, a254)
            pk = r.get("peak_info") or {}
            items.append({
                "seq": seq_name,
                "sample": sname,
                "rep": str(rk),
                "selected": (str(rk) == str(sel)),
                "type": stype,
                "tmax": (round(float(pk["t_max"]), 3) if pk.get("t_max") is not None else None),
                "t": [round(float(v), 3) for v in td],
                "doc": [None if (yd is None or np.isnan(yd[i])) else round(float(yd[i]), 4) for i in range(len(td))],
                "a254": None if ya is None else [None if np.isnan(ya[i]) else round(float(ya[i]), 5) for i in range(len(td))],
                "uib": None if yu is None else [None if np.isnan(yu[i]) else round(float(yu[i]), 4) for i in range(len(td))],
            })
    return items, seq_name


def build(seq_args, out_path):
    folder = _default_folder()
    all_items = []
    seqs_done = []
    for arg in seq_args:
        sp = _resolve_seq(arg, folder)
        if not sp:
            print(f"  ! no trobo la seqüència: {arg}")
            continue
        print(f"  · processant {os.path.basename(sp)} ...", flush=True)
        try:
            items, name = build_items_for_seq(sp)
            all_items += items
            seqs_done.append(name)
            print(f"    {len(items)} traces", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)

    if not all_items:
        print("Cap traça generada. Res a escriure.")
        return None

    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "fractions": get_fractions(),
        "sequences": seqs_done,
        "items": all_items,
    }
    html_str = render_html(payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"\nHTML escrit: {out_path}  ({len(all_items)} traces, {len(seqs_done)} seqüències)")
    return out_path


def build_one(target, open_after=False, progress=None):
    """Genera l'HTML per UNA seqüència i el desa a la MATEIXA carpeta.

    target pot ser:
      - la carpeta de la seqüència (p. ex. .../290_SEQ)
      - un fitxer MasterFile (.xlsx) -> s'usa la seva carpeta pare

    Retorna la ruta de l'HTML escrit. Pensat per al llançador de tècnics.
    """
    def _say(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    target = os.path.abspath(target)
    if os.path.isfile(target):
        if not target.lower().endswith(".xlsx"):
            raise ValueError("Tria una carpeta de seqüència o un MasterFile (.xlsx).")
        seq_path = os.path.dirname(target)
    elif os.path.isdir(target):
        seq_path = target
    else:
        raise ValueError(f"No existeix: {target}")

    seq_name = os.path.basename(seq_path.rstrip("/\\"))
    _say(f"Llegint {seq_name} i processant el DOC…")
    items, name = build_items_for_seq(seq_path)
    if not items:
        raise RuntimeError(f"No s'han pogut generar cromatogrames de {seq_name} "
                           f"(falta MasterFile o dades?).")

    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "fractions": get_fractions(),
        "sequences": [name],
        "items": items,
    }
    out_path = os.path.join(seq_path, f"{seq_name}_cromatogrames.html")
    _say("Escrivint l'HTML…")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_html(payload))
    _say(f"Fet: {out_path}")
    if open_after:
        try:
            import webbrowser
            webbrowser.open("file:///" + out_path.replace("\\", "/"))
        except Exception:
            pass
    return out_path


def render_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title = "HPSEC · Visor de cromatogrames"
    seqs_txt = html.escape(", ".join(payload["sequences"]))
    return _HTML_TEMPLATE.replace("/*__DATA__*/", data_json) \
                         .replace("__TITLE__", html.escape(title)) \
                         .replace("__SEQS__", seqs_txt) \
                         .replace("__GEN__", html.escape(payload["generated"]))


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ca">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#ffffff; --ink:#1d2733; --muted:#6b7785; --line:#e6ebf0;
    --doc:#0f7f82; --dad:#e07b39; --uib:#8a6fb0; --accent:#0f7f82;
    --card:#ffffff;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --bg:#11161c; --ink:#e8edf2; --muted:#9aa6b2; --line:#263039;
      --doc:#4fd0d3; --dad:#f2a96b; --uib:#b79ce0; --card:#161c23;
    }
  }
  :root[data-theme="dark"]{
    --bg:#11161c; --ink:#e8edf2; --muted:#9aa6b2; --line:#263039;
    --doc:#4fd0d3; --dad:#f2a96b; --uib:#b79ce0; --card:#161c23;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1060px;margin:0 auto;padding:18px 16px 40px;}
  header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px;margin-bottom:6px;}
  h1{font-size:17px;font-weight:650;margin:0;letter-spacing:.2px;}
  .sub{font-size:12px;color:var(--muted);}
  .toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0 8px;}
  select,button{font:inherit;color:var(--ink);background:var(--card);
    border:1px solid var(--line);border-radius:9px;padding:7px 10px;}
  select{min-width:120px;}
  button{cursor:pointer;min-width:40px;}
  button:hover{border-color:var(--accent);}
  button.nav{font-size:16px;font-weight:600;padding:7px 13px;}
  .spacer{flex:1 1 auto;}
  .chk{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--muted);
    border:1px solid var(--line);border-radius:9px;padding:6px 10px;cursor:pointer;user-select:none;}
  .chk input{accent-color:var(--accent);}
  .title-row{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px;margin:6px 2px 2px;}
  .title-row .name{font-size:16px;font-weight:650;}
  .title-row .meta{font-size:12.5px;color:var(--muted);}
  .pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;
    background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent);}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:8px 8px 4px;margin-top:8px;box-shadow:0 1px 2px rgba(0,0,0,.03);}
  canvas{width:100%;display:block;}
  .legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted);
    padding:4px 8px 10px;}
  .legend b{font-weight:600;}
  .sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px;}
  .fracleg{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;font-size:11.5px;color:var(--muted);}
  .fr{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);
    border-radius:8px;padding:3px 8px;}
  .fr i{width:12px;height:12px;border-radius:3px;display:inline-block;}
  .hint{font-size:11.5px;color:var(--muted);margin-top:12px;}
  kbd{font:inherit;font-size:11px;border:1px solid var(--line);border-bottom-width:2px;
    border-radius:5px;padding:1px 5px;background:var(--card);}
  .readout{font-size:12px;color:var(--muted);min-height:16px;margin:2px 4px 0;font-variant-numeric:tabular-nums;}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>HPSEC · Visor de cromatogrames</h1>
    <span class="sub">__SEQS__ · generat __GEN__</span>
  </header>

  <div class="toolbar">
    <label class="sub">Seqüència</label>
    <select id="selSeq"></select>
    <label class="sub">Mostra</label>
    <select id="selSample"></select>
    <button class="nav" id="prev" title="Anterior (←)">◀</button>
    <button class="nav" id="next" title="Següent (→)">▶</button>
    <span class="spacer"></span>
    <label class="chk"><input type="checkbox" id="cDoc" checked> DOC</label>
    <label class="chk"><input type="checkbox" id="cDad" checked> 254 nm</label>
    <label class="chk"><input type="checkbox" id="cUib"> UIB</label>
    <label class="chk"><input type="checkbox" id="cNorm"> Normalitza</label>
    <label class="chk"><input type="checkbox" id="cFrac" checked> Fraccions</label>
    <button id="theme" title="Tema clar/fosc">◐</button>
  </div>

  <div class="title-row">
    <span class="name" id="tName">—</span>
    <span class="meta" id="tMeta"></span>
  </div>

  <div class="card">
    <canvas id="cv" height="440"></canvas>
    <div class="readout" id="readout"></div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="fracleg" id="fracleg"></div>
  <div class="hint">Navega amb <kbd>←</kbd> <kbd>→</kbd> o els desplegables. Passa el ratolí per llegir valors.
     DOC i 254 comparteixen l'eix de temps (ja alineats); «Normalitza» escala 0–1 per comparar formes.</div>
</div>

<script>
const DATA = /*__DATA__*/;
const ITEMS = DATA.items, FR = DATA.fractions;
const $ = id => document.getElementById(id);

// --- estat ---
let idx = 0;
const seqs = DATA.sequences.slice();

function sampleList(seq){
  return ITEMS.map((it,i)=>({it,i})).filter(o=>o.it.seq===seq);
}
function fillSeq(){
  $("selSeq").innerHTML = seqs.map(s=>`<option>${s}</option>`).join("");
}
function fillSamples(seq, keepIdx){
  const list = sampleList(seq);
  $("selSample").innerHTML = list.map(o=>{
    const it=o.it; const star=it.selected?"":" ·2";
    const lbl = `${it.sample}  R${it.rep}${it.type&&it.type!=='SAMPLE'?'  ['+it.type+']':''}`;
    return `<option value="${o.i}">${lbl}</option>`;
  }).join("");
  if(keepIdx!=null){ $("selSample").value = keepIdx; }
  else if(list.length){ $("selSample").value = list[0].i; }
}

// --- color helpers ---
function cssv(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

// --- dibuix ---
const cv = $("cv"), ctx = cv.getContext("2d");
let VIEW = null; // guarda mapeig per hover

function layout(){
  const dpr = window.devicePixelRatio||1;
  const cssW = cv.clientWidth, cssH = 440;
  cv.width = Math.round(cssW*dpr); cv.height = Math.round(cssH*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {w:cssW,h:cssH};
}

function niceTicks(min,max,n){
  if(!(max>min)) return [min];
  const span=max-min, step0=span/n, mag=Math.pow(10,Math.floor(Math.log10(step0)));
  const norm=step0/mag, step = (norm<1.5?1:norm<3?2:norm<7?5:10)*mag;
  const t=[]; let v=Math.ceil(min/step)*step;
  for(;v<=max+1e-9;v+=step) t.push(Math.round(v/step)*step);
  return t;
}

function draw(){
  const {w,h}=layout();
  ctx.clearRect(0,0,w,h);
  const it = ITEMS[idx]; if(!it) return;
  const showDoc=$("cDoc").checked, showDad=$("cDad").checked&&it.a254,
        showUib=$("cUib").checked&&it.uib, norm=$("cNorm").checked, showFrac=$("cFrac").checked;

  const padL=54, padR= (showDad && !norm)?54:16, padT=14, padB=34;
  const x0=padL, x1=w-padR, y0=h-padB, y1=padT;
  const t=it.t; const tmin=t[0], tmax=t[t.length-1];
  const X = v => x0 + (v-tmin)/(tmax-tmin||1)*(x1-x0);

  // rangs Y
  function rng(arr){ let a=Infinity,b=-Infinity; for(const v of arr){ if(v==null||isNaN(v))continue; if(v<a)a=v; if(v>b)b=v;} if(a>b){a=0;b=1;} if(a===b){b=a+1;} return [a,b]; }
  let docR = showDoc? rng(it.doc):[0,1];
  let dadR = showDad? rng(it.a254):[0,1];
  let uibR = showUib? rng(it.uib):[0,1];
  // marge superior
  const pad=(r)=>{const d=(r[1]-r[0])*0.08; return [r[0]-d*0.3, r[1]+d];};
  docR=pad(docR); dadR=pad(dadR); uibR=pad(uibR);

  const Ydoc = v => y0 - (v-docR[0])/(docR[1]-docR[0])*(y0-y1);
  const Ydad = v => y0 - (v-dadR[0])/(dadR[1]-dadR[0])*(y0-y1);
  const Yuib = v => y0 - (v-uibR[0])/(uibR[1]-uibR[0])*(y0-y1);
  const Yn = (v,r)=> y0 - (v-r[0])/(r[1]-r[0])*(y0-y1);
  function Ynorm(arr,v){ const r=rng(arr); return y0 - (v-r[0])/(r[1]-r[0]||1)*(y0-y1); }

  // fraccions
  if(showFrac){
    ctx.save();
    for(const f of FR){
      if(f.end<tmin||f.start>tmax) continue;
      const xa=X(Math.max(f.start,tmin)), xb=X(Math.min(f.end,tmax));
      ctx.fillStyle=f.color; ctx.globalAlpha=0.55;
      ctx.fillRect(xa,y1,xb-xa,y0-y1);
      ctx.globalAlpha=1;
      ctx.fillStyle=cssv("--muted"); ctx.font="11px sans-serif"; ctx.textAlign="center";
      if(xb-xa>22) ctx.fillText(f.key,(xa+xb)/2,y1+12);
    }
    ctx.restore();
  }

  // graella + eixos
  ctx.strokeStyle=cssv("--line"); ctx.fillStyle=cssv("--muted");
  ctx.lineWidth=1; ctx.font="11px sans-serif";
  ctx.textAlign="center"; ctx.textBaseline="top";
  for(const tk of niceTicks(tmin,tmax,8)){
    const x=X(tk); ctx.globalAlpha=.5; ctx.beginPath(); ctx.moveTo(x,y1); ctx.lineTo(x,y0); ctx.stroke(); ctx.globalAlpha=1;
    ctx.fillText(tk.toFixed(tmax-tmin>20?0:1), x, y0+6);
  }
  ctx.fillText("temps (min)", (x0+x1)/2, y0+18);

  // eix esquerra (DOC o normalitzat)
  ctx.textAlign="right"; ctx.textBaseline="middle";
  const leftR = norm? [0,1] : docR;
  for(const tk of niceTicks(leftR[0],leftR[1],6)){
    const y = norm? Yn(tk,[0,1]) : Ydoc(tk);
    ctx.globalAlpha=.35; ctx.strokeStyle=cssv("--line");
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke(); ctx.globalAlpha=1;
    ctx.fillStyle=cssv("--muted"); ctx.fillText(norm?tk.toFixed(1):fmt(tk), x0-6, y);
  }

  // eix dret (254) si no normalitzat
  if(showDad && !norm){
    ctx.textAlign="left"; ctx.fillStyle=cssv("--dad");
    for(const tk of niceTicks(dadR[0],dadR[1],6)){
      const y=Ydad(tk); ctx.fillText(fmt(tk), x1+6, y);
    }
  }

  // línies
  function plot(arr, ymap, color, width, dash){
    ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=width;
    if(dash) ctx.setLineDash(dash);
    ctx.beginPath(); let pen=false;
    for(let i=0;i<t.length;i++){ const v=arr[i];
      if(v==null||isNaN(v)){ pen=false; continue; }
      const x=X(t[i]), y=ymap(v);
      if(!pen){ ctx.moveTo(x,y); pen=true; } else ctx.lineTo(x,y);
    }
    ctx.stroke(); ctx.restore();
  }
  if(showDoc) plot(it.doc, norm? v=>Ynorm(it.doc,v):Ydoc, cssv("--doc"), 1.6, null);
  if(showUib) plot(it.uib, norm? v=>Ynorm(it.uib,v):Yuib, cssv("--uib"), 1.1, [2,2]);
  if(showDad) plot(it.a254, norm? v=>Ynorm(it.a254,v):Ydad, cssv("--dad"), 1.2, [5,3]);

  // marca t_max DOC
  if(it.tmax!=null && it.tmax>=tmin && it.tmax<=tmax){
    const x=X(it.tmax); ctx.save(); ctx.strokeStyle=cssv("--doc"); ctx.globalAlpha=.35;
    ctx.setLineDash([2,3]); ctx.beginPath(); ctx.moveTo(x,y1); ctx.lineTo(x,y0); ctx.stroke(); ctx.restore();
  }

  VIEW={x0,x1,y0,y1,tmin,tmax,X,it,showDoc,showDad,showUib,norm};
}

function fmt(v){ const a=Math.abs(v);
  if(a>=1000) return v.toFixed(0); if(a>=10) return v.toFixed(1);
  if(a>=1) return v.toFixed(2); return v.toFixed(3); }

function nearestIdx(it,tt){ // busca índex de temps més proper
  const t=it.t; let lo=0,hi=t.length-1;
  while(hi-lo>1){ const m=(lo+hi)>>1; if(t[m]<tt) lo=m; else hi=m; }
  return (Math.abs(t[lo]-tt)<=Math.abs(t[hi]-tt))?lo:hi;
}

// hover
cv.addEventListener("mousemove",e=>{
  if(!VIEW) return;
  const r=cv.getBoundingClientRect(); const mx=e.clientX-r.left;
  if(mx<VIEW.x0||mx>VIEW.x1){ $("readout").textContent=""; draw(); return; }
  const tt=VIEW.tmin+(mx-VIEW.x0)/(VIEW.x1-VIEW.x0)*(VIEW.tmax-VIEW.tmin);
  const it=VIEW.it; const i=nearestIdx(it,tt);
  draw();
  const x=VIEW.X(it.t[i]);
  ctx.save(); ctx.strokeStyle=cssv("--muted"); ctx.globalAlpha=.5; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(x,VIEW.y1); ctx.lineTo(x,VIEW.y0); ctx.stroke(); ctx.restore();
  const parts=[`t=${it.t[i].toFixed(2)} min`];
  if(VIEW.showDoc && it.doc[i]!=null) parts.push(`DOC=${fmt(it.doc[i])}`);
  if(VIEW.showDad && it.a254 && it.a254[i]!=null) parts.push(`254=${fmt(it.a254[i])}`);
  if(VIEW.showUib && it.uib && it.uib[i]!=null) parts.push(`UIB=${fmt(it.uib[i])}`);
  $("readout").textContent=parts.join("    ");
});
cv.addEventListener("mouseleave",()=>{ $("readout").textContent=""; draw(); });

function refreshTitle(){
  const it=ITEMS[idx]; if(!it) return;
  $("tName").textContent = `${it.sample}  ·  R${it.rep}`;
  const bits=[];
  bits.push(`<span class="pill">${it.type||'—'}</span>`);
  bits.push(`seq ${it.seq}`);
  if(it.tmax!=null) bits.push(`t<sub>max</sub> DOC = ${it.tmax.toFixed(2)} min`);
  $("tMeta").innerHTML = bits.join("  ·  ");
}

function legend(){
  const it=ITEMS[idx];
  const L=[];
  if($("cDoc").checked) L.push(`<span><i class="sw" style="background:${cssv('--doc')}"></i><b>DOC</b> (TOC/OCD, net)</span>`);
  if($("cDad").checked && it.a254) L.push(`<span><i class="sw" style="background:${cssv('--dad')}"></i><b>254 nm</b> (DAD)</span>`);
  if($("cUib").checked && it.uib) L.push(`<span><i class="sw" style="background:${cssv('--uib')}"></i><b>UIB</b> (UV inline)</span>`);
  $("legend").innerHTML=L.join("");
}

function fracLegend(){
  $("fracleg").innerHTML = FR.map(f=>`<span class="fr"><i style="background:${f.color}"></i>${f.key} · ${f.name} (${f.start}–${f.end})</span>`).join("");
}

function render(){ refreshTitle(); legend(); draw(); }

function go(n){
  const seq=$("selSeq").value; const list=sampleList(seq);
  let pos=list.findIndex(o=>o.i===idx);
  pos=Math.max(0,Math.min(list.length-1,pos+n));
  idx=list[pos].i; $("selSample").value=idx; render();
}

// esdeveniments
$("selSeq").addEventListener("change",()=>{ fillSamples($("selSeq").value); idx=parseInt($("selSample").value); render(); });
$("selSample").addEventListener("change",()=>{ idx=parseInt($("selSample").value); render(); });
$("prev").addEventListener("click",()=>go(-1));
$("next").addEventListener("click",()=>go(1));
for(const id of ["cDoc","cDad","cUib","cNorm","cFrac"]) $(id).addEventListener("change",render);
$("theme").addEventListener("click",()=>{
  const cur=document.documentElement.getAttribute("data-theme");
  document.documentElement.setAttribute("data-theme", cur==="dark"?"light":"dark");
  render();
});
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft"){go(-1);e.preventDefault();}
  else if(e.key==="ArrowRight"){go(1);e.preventDefault();}
});
window.addEventListener("resize",draw);

// init
fillSeq();
fillSamples(seqs[0]);
idx = sampleList(seqs[0])[0].i;
fracLegend(); render();
</script>
</body>
</html>
"""


def main():
    raw = sys.argv[1:]
    out = "hpsec_view.html"
    args = []
    i = 0
    while i < len(raw):
        a = raw[i]
        if a == "--out":
            if i + 1 < len(raw):
                out = raw[i + 1]
                i += 2
                continue
            i += 1
            continue
        if a.startswith("--"):
            i += 1
            continue
        args.append(a)
        i += 1
    out = os.path.abspath(out)

    if not args:
        folder = _default_folder()
        print("Ús: python hpsec_viewer_build.py <SEQ> [<SEQ> ...] [--out fitxer.html]")
        print(f"\nCarpeta de dades: {folder}")
        try:
            seqs = sorted(d for d in os.listdir(folder)
                          if os.path.isdir(os.path.join(folder, d))
                          and (d.endswith("_SEQ") or "_SEQ" in d))
            print(f"Seqüències disponibles ({len(seqs)}):")
            print("  " + "  ".join(seqs[:60]))
        except Exception:
            pass
        return

    print(f"Generant visor HTML -> {out}")
    build(args, out)


if __name__ == "__main__":
    main()
