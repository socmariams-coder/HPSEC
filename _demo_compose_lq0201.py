"""Demo: composició timeout LQ0201 de 288_SEQ."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hpsec_import import import_sequence, ensure_data_loaded
from hpsec_analyze import analyze_sample
from hpsec_core import (
    detect_timeout, check_timeout_composability, compose_replicas,
    TIMEOUT_CONFIG, map_timeouts_to_injection
)
from hpsec_config import get_config

SEQ = r"C:\Users\Lequia\Desktop\Dades3\288_SEQ"
SAMPLE = "LQ0201"

print(f"affected_zone_pre = {TIMEOUT_CONFIG['affected_zone_pre']} min")

# 1. Importar amb dades completes
print("Importar 288_SEQ (amb dades)...")
data = import_sequence(SEQ)
ensure_data_loaded(data)

# 2. Trobar LQ0201
samples = data.get("samples", {})
sample_data = None
for key, sdata in samples.items():
    if SAMPLE in key:
        sample_data = sdata
        break

if not sample_data:
    print(f"ERROR: {SAMPLE} no trobat. Mostres: {list(samples.keys())[:5]}...")
    sys.exit(1)

reps = sample_data.get("replicas", {})
rep_keys = sorted(reps.keys())
print(f"Rèpliques: {rep_keys}")

r1, r2 = reps[rep_keys[0]], reps[rep_keys[1]]

# Extreure DOC Direct (raw, net si disponible)
def get_doc(rep):
    direct = rep.get("direct", {}) or {}
    t = np.asarray(direct.get("t", []), dtype=float)
    y = np.asarray(direct.get("y", []), dtype=float)
    return t, y

t1, y1 = get_doc(r1)
t2, y2 = get_doc(r2)
print(f"R{rep_keys[0]}: {len(t1)} pts, R{rep_keys[1]}: {len(t2)} pts")

if len(t1) == 0 or len(t2) == 0:
    print("ERROR: Dades DOC buides. Provant amb analyze_sample...")
    # Analyze to get processed data
    config = get_config()
    for rk in rep_keys:
        result = analyze_sample(reps[rk], config=config, mode="COLUMN")
        reps[rk].update(result)
    t1 = np.asarray(reps[rep_keys[0]].get("t_doc", []), dtype=float)
    y1 = np.asarray(reps[rep_keys[0]].get("y_doc_net", []), dtype=float)
    t2 = np.asarray(reps[rep_keys[1]].get("t_doc", []), dtype=float)
    y2 = np.asarray(reps[rep_keys[1]].get("y_doc_net", []), dtype=float)
    print(f"Post-analyze: R{rep_keys[0]}: {len(t1)} pts, R{rep_keys[1]}: {len(t2)} pts")

if len(t1) == 0:
    print("ERROR: No s'han pogut obtenir dades DOC")
    sys.exit(1)

# 3. Detectar timeouts
ti1 = detect_timeout(t1)
ti2 = detect_timeout(t2)

print(f"\nR{rep_keys[0]} timeouts:")
for to in ti1.get("timeouts", []):
    print(f"  gap: {to['t_start_min']:.1f}–{to['t_end_min']:.1f} min, "
          f"afectat: {to['affected_start_min']:.1f}–{to['affected_end_min']:.1f} min")

print(f"R{rep_keys[1]} timeouts:")
for to in ti2.get("timeouts", []):
    print(f"  gap: {to['t_start_min']:.1f}–{to['t_end_min']:.1f} min, "
          f"afectat: {to['affected_start_min']:.1f}–{to['affected_end_min']:.1f} min")

# 4. Composabilitat
comp = check_timeout_composability(ti1, ti2, run_duration_min=70.0)
print(f"\nComposable: {comp['composable']}")
print(f"Cobertura: {comp.get('coverage_pct', 100):.1f}%")
print(f"Solapament: {comp.get('unrepairable_min', 0):.1f} min")
print(f"Segments:")
for s in comp["segments"]:
    chosen = f" (triat R{rep_keys[int(s['chosen'])-1]})" if s.get("chosen") else ""
    print(f"  {s['t_start']:.1f}–{s['t_end']:.1f}: font={s['source']}{chosen}")

if not comp["composable"]:
    print("\nNo composable — sortint")
    sys.exit(0)

# 5. Composar
t_out, y_out, meta = compose_replicas(t1, y1, t2, y2, comp["segments"])

# Verificar chosen
for s in comp["segments"]:
    if s.get("chosen"):
        print(f"  Zona solapament: triat R{rep_keys[int(s['chosen'])-1]} (menys degradat)")

# 6. Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.set_title(f"{SAMPLE} — Composició de rèpliques per timeout", fontsize=13, fontweight="bold")

# R1 i R2
ax.plot(t1, y1, color="#2E86AB", lw=0.8, alpha=0.5, label=f"R{rep_keys[0]}")
ax.plot(t2, y2, color="#E67E22", lw=0.8, alpha=0.5, label=f"R{rep_keys[1]}")

# Compost
ax.plot(t_out, y_out, color="#2C3E50", lw=1.8, label="Compost", zorder=5)

# Zones timeout R1 (blau)
for to in ti1.get("timeouts", []):
    ax.axvspan(to["affected_start_min"], to["affected_end_min"],
               color="#2E86AB", alpha=0.12, zorder=0)
    ax.axvline(to["t_start_min"], color="#2E86AB", ls="--", lw=0.7, alpha=0.5)

# Zones timeout R2 (taronja)
for to in ti2.get("timeouts", []):
    ax.axvspan(to["affected_start_min"], to["affected_end_min"],
               color="#E67E22", alpha=0.12, zorder=0)
    ax.axvline(to["t_start_min"], color="#E67E22", ls="--", lw=0.7, alpha=0.5)

# Zona solapament
for ov in comp.get("overlap_zones", []):
    ax.axvspan(ov[0], ov[1], color="#E74C3C", alpha=0.15, zorder=0,
               label="Solapament")

# Segment annotations
for seg in comp["segments"]:
    if seg["source"] == "unrepairable":
        chosen = seg.get("chosen", "?")
        y_pos = ax.get_ylim()[1] * 0.92
        ax.annotate(f"auto→R{rep_keys[int(chosen)-1]}" if chosen in ("1","2") else "auto",
                    xy=((seg["t_start"]+seg["t_end"])/2, y_pos),
                    ha="center", fontsize=9, color="#E74C3C", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E74C3C", alpha=0.8))

ax.set_xlim(10, 40)
ax.set_ylabel("DOC (ppb)", fontsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
ax.grid(True, alpha=0.15)

# Subplot inferior: segment map
ax2 = axes[1]
colors = {"1": "#2E86AB", "2": "#E67E22", "unrepairable": "#E74C3C"}
for seg in comp["segments"]:
    if seg["t_end"] <= 10 or seg["t_start"] >= 40:
        continue
    c = colors.get(seg["source"], "#999")
    ax2.barh(0, seg["t_end"] - seg["t_start"], left=seg["t_start"],
             height=0.6, color=c, alpha=0.7, edgecolor="white", lw=0.5)
    mid = (seg["t_start"] + seg["t_end"]) / 2
    src = seg["source"]
    if src in ("1", "2"):
        label = f"R{rep_keys[int(src)-1]}"
    else:
        label = f"R{rep_keys[int(seg.get('chosen','1'))-1]}*"
    if seg["t_end"] - seg["t_start"] > 1.5:
        ax2.text(mid, 0, label, ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")

ax2.set_xlim(10, 40)
ax2.set_ylim(-0.5, 0.5)
ax2.set_yticks([])
ax2.set_xlabel("Temps (min)", fontsize=10)
ax2.set_ylabel("Segment", fontsize=9)
ax2.grid(True, alpha=0.15, axis="x")

plt.tight_layout()
out_path = os.path.join(SEQ, "CHECK", "data", f"{SAMPLE}_composition_demo.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot guardat: {out_path}")
