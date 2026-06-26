"""Revisio a fons de la calibracio COLUMN (i BP de referencia) amb la seleccio robusta
de replica aplicada. Genera cens + regressions + grafiques.
Nomes lectura sobre KHP_History; escriu PNGs a la mateixa carpeta."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HIST = r'C:\Users\maria\Proyectos\Dades3\REGISTRY\KHP_History.json'
OUT  = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(HIST, encoding='utf-8'))
cals = d['calibrations']

def replica_has_blocker(r):
    return any(isinstance(a,dict) and a.get('severity')=='blocker'
               for a in r.get('calibration_anomalies',[]))

def select(reps):
    """Retorna (usable_reps, area_sel, excluded_reps) segons el fix."""
    usable=[r for r in reps if not replica_has_blocker(r)]
    if usable and len(usable)<len(reps):
        work=usable; excl=[r for r in reps if replica_has_blocker(r)]
    else:
        work=reps; excl=[]
    areas=[r['area'] for r in work]
    if len(work)==1:
        area=areas[0]
    else:
        rsd=np.std(areas)/np.mean(areas)*100 if np.mean(areas)>0 else 100
        if rsd<10: area=float(np.mean(areas))
        else: area=min(work,key=replica_has_blocker_weight)['area']
    return work, area, excl

def replica_has_blocker_weight(r):
    return sum(10 for a in r.get('calibration_anomalies',[]) if isinstance(a,dict) and a.get('severity')=='blocker')

def regress(X,Y,model='free'):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if len(X)<2: return None
    if model=='origin' or np.all(X==X[0]):
        s=np.sum(X*Y)/np.sum(X*X); i=0.0
    else:
        A=np.vstack([X,np.ones_like(X)]).T
        (s,i),_,_,_=np.linalg.lstsq(A,Y,rcond=None)
    pred=s*X+i
    r2=1-np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2) if np.sum((Y-Y.mean())**2)>0 else 0
    rms=float(np.sqrt(np.mean((Y-pred)**2)))
    return dict(slope=float(s),intercept=float(i),r2=float(r2),rms=rms,n=len(X))

# ---- Recollir punts per mode ----
def collect(mode, seqs=None):
    """Retorna llista de dicts: conc, vol, ug, area_sel, used_reps(list area), excl_reps(list area), seq."""
    out=[]
    for c in cals:
        if c.get('mode','').upper()!=mode: continue
        if seqs and c.get('seq_name') not in seqs: continue
        conc=c.get('conc_ppm'); vol=c.get('volume_uL') or (400 if mode=='COLUMN' else 100)
        if not conc or conc<=0: continue
        reps=c.get('replicas_info') or []
        if not reps: continue
        work,area,excl=select(reps)
        out.append(dict(seq=c.get('seq_name'),conc=conc,vol=vol,ug=conc*vol/1000.0,
                        area=area,used=[r['area'] for r in work],
                        excl=[r['area'] for r in excl]))
    return out

print("="*78); print("CENS COLUMN (amb seleccio robusta aplicada)"); print("="*78)
col=collect('COLUMN')
print(f"{'seq':16}{'conc':>6}{'vol':>6}{'ug':>7}{'area_sel':>10}{'used':>8}{'excl':>6}")
for p in sorted(col,key=lambda x:(x['seq'],x['conc'])):
    print(f"{p['seq'][:16]:16}{p['conc']:>6}{p['vol']:>6}{p['ug']:>7.3f}{p['area']:>10.1f}{len(p['used']):>8}{len(p['excl']):>6}")

# ---- Regressions COLUMN ----
# 293 sol (referencia neta) vs tot COLUMN
col293=[p for p in col if p['seq']=='293_SEQ_CAL']
# excloure 303 (dades trencades, area absurda per 2ppm)
col_clean=[p for p in col if not (p['seq']=='303_SEQ')]
print("\n--- REGRESSIONS COLUMN (model lliure) ---")
for label,pts in [("293_SEQ_CAL sol",col293),("Tot COLUMN (sense 303 trencada)",col_clean),("Tot COLUMN (incloent 303)",col)]:
    r=regress([p['ug'] for p in pts],[p['area'] for p in pts],'free')
    if r: print(f"  {label:34}: RF_mass(slope)={r['slope']:.1f}  intercept={r['intercept']:.2f}  R2={r['r2']:.4f}  RMS={r['rms']:.1f}  n={r['n']}")

bp=collect('BP',seqs={'292_SEQ_CAL_BP'})

# ===================== GRAFIQUES =====================
def plot_calibration(pts, title, fname, model='free', color='#1565C0'):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios':[3,1]},sharex=True)
    # punts usats (per replica) i seleccionats
    for p in pts:
        for a in p['used']:
            ax1.plot(p['ug'],a,'o',color=color,ms=5,alpha=0.45,zorder=3)
        for a in p['excl']:
            ax1.plot(p['ug'],a,'x',color='#D32F2F',ms=9,mew=2,zorder=4)
        ax1.plot(p['ug'],p['area'],'o',color=color,ms=9,mec='k',mew=0.8,zorder=5)
    r=regress([p['ug'] for p in pts],[p['area'] for p in pts],model)
    if r:
        xx=np.linspace(0,max(p['ug'] for p in pts)*1.05,100)
        ax1.plot(xx,r['slope']*xx+r['intercept'],'-',color=color,lw=1.5,zorder=2)
        eq=f"Area = {r['slope']:.1f}·µg + {r['intercept']:.1f}" if model=='free' else f"Area = {r['slope']:.1f}·µg"
        ax1.text(0.03,0.95,f"{eq}\nR² = {r['r2']:.4f}   RMS = {r['rms']:.1f}   n = {r['n']}",
                 transform=ax1.transAxes,va='top',fontsize=10,
                 bbox=dict(boxstyle='round',fc='#E3F2FD',ec=color))
        # residuals %
        for p in pts:
            pred=r['slope']*p['ug']+r['intercept']
            res=(p['area']-pred)/pred*100 if pred>0 else 0
            ax2.plot(p['ug'],res,'o',color=color,ms=7,mec='k',mew=0.6)
        ax2.axhline(0,color='k',lw=0.8); ax2.axhline(5,color='#999',ls=':'); ax2.axhline(-5,color='#999',ls=':')
    ax1.set_ylabel('Àrea (mAU·min)'); ax1.set_title(title,fontsize=11,fontweight='bold')
    ax1.grid(alpha=0.25)
    ax2.set_ylabel('Resid %'); ax2.set_xlabel('µg DOC injectat'); ax2.grid(alpha=0.25)
    # llegenda manual
    ax1.plot([],[],'o',color=color,ms=9,mec='k',label='punt usat (seleccionat)')
    ax1.plot([],[],'o',color=color,ms=5,alpha=0.45,label='rèplica usada')
    ax1.plot([],[],'x',color='#D32F2F',ms=9,mew=2,label='rèplica EXCLOSA (blocker)')
    ax1.legend(fontsize=8,loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,fname),dpi=110); plt.close(fig)
    return r

r1=plot_calibration(col293,"COLUMN — 293_SEQ_CAL (referència neta, model lliure)","col_293.png",'free','#1565C0')
r2=plot_calibration(col_clean,"COLUMN — totes les SEQ (sense 303 trencada)","col_totes.png",'free','#1565C0')
r3=plot_calibration(bp,"BP — 292_SEQ_CAL_BP (selecció robusta, model lliure)","bp_292.png",'free','#2E7D32')

# Grafica combinada COLUMN vs BP (normalitzat per veure pendents)
fig,ax=plt.subplots(figsize=(8,6))
for pts,lab,col_ in [(col293,'COLUMN 293','#1565C0'),(bp,'BP 292','#2E7D32')]:
    xs=[p['ug'] for p in pts]; ys=[p['area'] for p in pts]
    ax.plot(xs,ys,'o',color=col_,ms=8,mec='k',mew=0.6,label=lab)
    r=regress(xs,ys,'free'); xx=np.linspace(0,max(xs)*1.05,100)
    ax.plot(xx,r['slope']*xx+r['intercept'],'-',color=col_,lw=1.4)
ax.set_xlabel('µg DOC injectat'); ax.set_ylabel('Àrea (mAU·min)')
ax.set_title('COLUMN vs BP — rectes de calibració (model lliure)',fontweight='bold')
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'col_vs_bp.png'),dpi=110); plt.close(fig)

print("\nPNGs desats a:", OUT)
for f in ['col_293.png','col_totes.png','bp_292.png','col_vs_bp.png']:
    print("  -", f)
