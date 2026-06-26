"""Validacio de les DUES regressions: arees -> recuperacio de ppm nominal.
Aplica la formula de quantificacio de la Suite amb el RF/intercept ajustat i compara
amb la concentracio coneguda de cada KHP. Genera taula + grafic de recuperacio."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HIST = r'C:\Users\maria\Proyectos\Dades3\REGISTRY\KHP_History.json'
OUT  = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(HIST, encoding='utf-8'))
cals = d['calibrations']

def replica_has_blocker(r):
    return any(isinstance(a,dict) and a.get('severity')=='blocker'
               for a in r.get('calibration_anomalies',[]))
def w(r):
    return sum(10 for a in r.get('calibration_anomalies',[]) if isinstance(a,dict) and a.get('severity')=='blocker')

def select(reps):
    usable=[r for r in reps if not replica_has_blocker(r)]
    work = usable if (usable and len(usable)<len(reps)) else reps
    areas=[r['area'] for r in work]
    if len(work)==1: area=areas[0]
    else:
        rsd=np.std(areas)/np.mean(areas)*100 if np.mean(areas)>0 else 100
        area=float(np.mean(areas)) if rsd<10 else min(work,key=w)['area']
    return area

def regress(X,Y,model='free'):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if model=='origin': s=np.sum(X*Y)/np.sum(X*X); i=0.0
    else:
        A=np.vstack([X,np.ones_like(X)]).T; (s,i),_,_,_=np.linalg.lstsq(A,Y,rcond=None)
    pred=s*X+i; r2=1-np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2)
    return float(s),float(i),float(r2)

def collect(mode, seq):
    out=[]
    for c in cals:
        if c.get('mode','').upper()!=mode or c.get('seq_name')!=seq: continue
        conc=c.get('conc_ppm'); vol=c.get('volume_uL') or (400 if mode=='COLUMN' else 100)
        reps=c.get('replicas_info') or []
        if not conc or conc<=0 or not reps: continue
        out.append(dict(conc=conc,vol=vol,ug=conc*vol/1000.0,area=select(reps)))
    return sorted(out,key=lambda x:x['conc'])

def ppm_calc(area, rf, inter, vol):  # formula Suite
    return max(0.0, area-inter)*1000.0/(rf*vol)

configs=[("COLUMN","293_SEQ_CAL",'#1565C0'),("BP","292_SEQ_CAL_BP",'#2E7D32')]
results={}
for mode,seq,color in configs:
    pts=collect(mode,seq)
    rf,inter,r2=regress([p['ug'] for p in pts],[p['area'] for p in pts],'free')
    print("="*72); print(f"{mode}  ({seq})   RF_mass={rf:.1f}  intercept={inter:.2f}  R2={r2:.4f}")
    print(f"  {'conc_nom':>9}{'area':>10}{'ppm_calc':>10}{'err_ppm':>9}{'err_%':>8}")
    errs=[]
    for p in pts:
        pc=ppm_calc(p['area'],rf,inter,p['vol'])
        e=pc-p['conc']; epct=e/p['conc']*100
        errs.append((p['conc'],epct))
        print(f"  {p['conc']:>9}{p['area']:>10.1f}{pc:>10.3f}{e:>9.3f}{epct:>7.1f}%")
    abs_all=np.mean([abs(e) for _,e in errs])
    abs_hi =np.mean([abs(e) for c,e in errs if c>=0.5])  # sense els molt baixos (prop LOD)
    print(f"  -> error abs mitja: TOTS={abs_all:.1f}%  |  conc>=0.5ppm={abs_hi:.1f}%")
    results[mode]=dict(pts=pts,rf=rf,inter=inter,r2=r2,color=color,errs=errs)

# ---- Grafic recuperacio ppm_calc vs nominal ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.5))
maxppm=5.3
ax1.plot([0,maxppm],[0,maxppm],'--',color='#777',lw=1,label='ideal (y=x)')
for mode in results:
    R=results[mode]
    xs=[p['conc'] for p in R['pts']]
    ys=[ppm_calc(p['area'],R['rf'],R['inter'],p['vol']) for p in R['pts']]
    ax1.plot(xs,ys,'o',color=R['color'],ms=9,mec='k',mew=0.6,label=f"{mode} (RF={R['rf']:.0f})")
ax1.set_xlabel('ppm nominal (KHP)'); ax1.set_ylabel('ppm recuperada (calibració)')
ax1.set_title('Recuperació de ppm — validació de les dues rectes',fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.25); ax1.set_xlim(0,maxppm); ax1.set_ylim(0,maxppm)
# error % vs conc (escala log x per veure els baixos)
ax2.axhline(0,color='k',lw=0.8); ax2.axhline(5,color='#bbb',ls=':'); ax2.axhline(-5,color='#bbb',ls=':')
for mode in results:
    R=results[mode]
    cs=[c for c,_ in R['errs']]; es=[e for _,e in R['errs']]
    ax2.plot(cs,es,'o-',color=R['color'],ms=8,mec='k',mew=0.5,label=mode)
ax2.set_xscale('log'); ax2.set_xlabel('ppm nominal (log)'); ax2.set_ylabel('error recuperació (%)')
ax2.set_title('Error de recuperació per concentració',fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.25,which='both')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'validacio_ppm.png'),dpi=110); plt.close(fig)
print("\nGrafic desat:", os.path.join(OUT,'validacio_ppm.png'))
