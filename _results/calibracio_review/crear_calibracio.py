"""Crea (o recrea) el Calibration_Reference.json DOC Direct amb les dues rectes
separades COLUMN i BP, CADASCUNA amb els seus regression_data (punts) perque el
grafic de la Suite els mostri. Usa add_calibration (esquema v3.0).

Patro de produccio: una entrada per mode (com aplicar 2 SEQ_CAL). L'activa porta un
mode; l'altra es troba pel fallback del grafic. Les dues porten rf_mass_cal {column,bp}."""
import os, sys
sys.path.insert(0, r'C:\Users\maria\Proyectos\HPSEC')
import json, numpy as np
import hpsec_calibrate as hc

HIST = r'C:\Users\maria\Proyectos\Dades3\REGISTRY\KHP_History.json'
cals_hist = json.load(open(HIST, encoding='utf-8'))['calibrations']

def replica_has_blocker(r):
    return any(isinstance(a,dict) and a.get('severity')=='blocker'
               for a in r.get('calibration_anomalies',[]))
def wbl(r):
    return sum(10 for a in r.get('calibration_anomalies',[]) if isinstance(a,dict) and a.get('severity')=='blocker')
def select_area(reps):
    usable=[r for r in reps if not replica_has_blocker(r)]
    work=usable if (usable and len(usable)<len(reps)) else reps
    areas=[r['area'] for r in work]
    if len(work)==1: return areas[0]
    rsd=np.std(areas)/np.mean(areas)*100 if np.mean(areas)>0 else 100
    return float(np.mean(areas)) if rsd<10 else min(work,key=wbl)['area']

def collect(mode, seq, vol_default):
    pts=[]
    for c in cals_hist:
        if c.get('mode','').upper()!=mode or c.get('seq_name')!=seq: continue
        conc=c.get('conc_ppm'); vol=c.get('volume_uL') or vol_default
        reps=c.get('replicas_info') or []
        if not conc or conc<=0 or not reps: continue
        pts.append(dict(seq=seq,conc=conc,vol=vol,ug=conc*vol/1000.0,area=select_area(reps)))
    return sorted(pts,key=lambda x:x['conc'])

def regress(pts):
    X=np.array([p['ug'] for p in pts]); Y=np.array([p['area'] for p in pts])
    A=np.vstack([X,np.ones_like(X)]).T; (s,i),_,_,_=np.linalg.lstsq(A,Y,rcond=None)
    pred=s*X+i; r2=1-np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2)
    rms=float(np.sqrt(np.mean((Y-pred)**2)))
    return float(s),float(i),float(r2),rms

def build_regdata(pts, mode):
    s,i,r2,rms=regress(pts)
    points=[]
    for p in pts:
        yp=s*p['ug']+i
        points.append(dict(seq_name=p['seq'],date='2026-06-26',conc_ppm=p['conc'],volume_uL=p['vol'],
                           ug_doc=p['ug'],area=p['area'],rf_mass=p['area']/p['ug'] if p['ug']>0 else 0,
                           residual=p['area']-yp,y_pred=yp,excluded=False))
    return dict(rf_mass_cal=s,intercept=i,r2=r2,n_points=len(pts),residuals_rms=rms,
                model='intercept',signal='direct',mode=mode,points=points), s,i,r2

col=collect('COLUMN','293_SEQ_CAL',400)
bp =collect('BP','292_SEQ_CAL_BP',100)
reg_col,rf_col,int_col,r2_col=build_regdata(col,'column')
reg_bp ,rf_bp ,int_bp ,r2_bp =build_regdata(bp ,'bp')
print(f"COLUMN: rf={rf_col:.1f} int={int_col:.2f} r2={r2_col:.4f} n={len(col)}")
print(f"BP    : rf={rf_bp:.1f} int={int_bp:.2f} r2={r2_bp:.4f} n={len(bp)}")

# --- Esborrar el fitxer existent (el vaig crear jo, sense punts) i recrear net ---
path=hc.get_calibration_reference_path()
if path and os.path.exists(path):
    os.remove(path); print("Esborrat fitxer anterior (sense punts).")
hc._cal_ref_cache=None; hc._cal_ref_mtime=0

rf_vals={"column":round(rf_col,1),"bp":round(rf_bp,1)}
int_vals={"column":round(int_col,2),"bp":round(int_bp,2)}
src={"type":"SEQ_CAL","seq_references":["293_SEQ_CAL","292_SEQ_CAL_BP"],
     "description":"DOC Direct — rectes separades COLUMN/BP (selecció robusta de rèplica, model lliure)"}

# Entrada BP primer (quedarà inactiva), COLUMN després (quedarà ACTIVA) → fallback troba BP
id_bp=hc.add_calibration(rf_vals,src,"2026-06-26",r2=round(r2_bp,4),n_points=len(bp),
                         intercept_values=int_vals,regression_data=reg_bp,signal_scope="direct")
hc._cal_ref_cache=None
id_col=hc.add_calibration(rf_vals,src,"2026-06-26",r2=round(r2_col,4),n_points=len(col),
                          intercept_values=int_vals,regression_data=reg_col,signal_scope="direct")
print("Entrades creades: BP(inactiva)=",id_bp," COLUMN(activa)=",id_col)

# --- Dedup IDs (els 2 add_calibration al mateix segon generen el mateix id) ---
hc._cal_ref_cache=None
ref=hc.load_calibration_reference()
seen=set()
changed=False
for c in ref["calibrations"]:
    cid=c["id"]
    if cid in seen:
        newid=cid+"_"+(c.get("regression_data",{}).get("mode","x").upper())
        c["id"]=newid; changed=True
        # NO és l'actiu (active_calibration_ids apunta al primer); no cal tocar active_ids
    seen.add(c["id"])
if changed:
    hc.save_calibration_reference(ref); hc._cal_ref_cache=None
    print("IDs deduplicats.")

# --- Verificacio ---
hc._cal_ref_cache=None
ref=hc.load_calibration_reference()
print("\nactive_ids:",ref.get("active_calibration_ids"))
for c in ref["calibrations"]:
    rd=c.get("regression_data",{})
    print(f"  entry {c['id']}  active={c.get('is_active')}  reg.mode={rd.get('mode')}  n_punts={len(rd.get('points',[]))}")
for mode in ("column","bp"):
    print(f"  {mode}: rf={hc.get_rf_mass_cal(signal='direct',mode=mode)} int={hc.get_calibration_intercept(signal='direct',mode=mode)}")
