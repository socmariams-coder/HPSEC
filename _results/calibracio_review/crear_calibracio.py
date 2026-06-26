"""Crea el Calibration_Reference.json (DOC Direct) amb les dues rectes separades
COLUMN i BP, usant la funcio de produccio add_calibration (esquema v3.0 correcte).
Valors validats: COLUMN 793.9/28.88 (293_SEQ_CAL), BP 682.6/4.26 (292_SEQ_CAL_BP).
Despres verifica carrega + recuperacio de ppm."""
import sys
sys.path.insert(0, r'C:\Users\maria\Proyectos\HPSEC')
import hpsec_calibrate as hc

# --- comprovar que NO existeix ja (no sobreescriure sense avis) ---
existing = hc.load_calibration_reference()
if existing:
    print("JA existeix Calibration_Reference.json — NO es toca. Surt.")
    sys.exit(0)

cal_id = hc.add_calibration(
    rf_mass_cal_values={"column": 793.9, "bp": 682.6},
    intercept_values={"column": 28.88, "bp": 4.26},
    source={
        "type": "SEQ_CAL",
        "description": "DOC Direct — COLUMN 293_SEQ_CAL + BP 292_SEQ_CAL_BP "
                       "(selecció robusta de rèplica, model lliure, integració cap=4 vigent)",
        "seq_references": ["293_SEQ_CAL", "292_SEQ_CAL_BP"],
    },
    valid_from="2026-06-26",
    r2={"column": 0.9998, "bp": 0.9979},
    n_points={"column": 6, "bp": 6},
    reason="Regeneració calibració DOC Direct (equip nou): rectes separades COLUMN/BP "
           "+ selecció robusta de rèplica + paritat reparació.",
    signal_scope="direct",
)
print("Calibració creada:", cal_id)

# ===================== VERIFICACIÓ =====================
hc._cal_ref_cache = None  # invalidar cache
ref = hc.load_calibration_reference()
print("\nFitxer carregat OK. version:", ref.get("version"),
      "| active_ids:", ref.get("active_calibration_ids"))

print("\n--- get_rf_mass_cal / intercept (signal=direct) ---")
for mode in ("column", "bp"):
    rf = hc.get_rf_mass_cal(signal="direct", mode=mode)
    it = hc.get_calibration_intercept(signal="direct", mode=mode)
    print(f"  {mode:7}: rf_mass={rf}  intercept={it}")

# Recuperació de 5 ppm amb la fórmula de la Suite
print("\n--- recuperació 5 ppm (test) ---")
tests = [("column", 1621.7, 400), ("bp", 341.8, 100)]
for mode, area, vol in tests:
    rf = hc.get_rf_mass_cal(signal="direct", mode=mode)
    it = hc.get_calibration_intercept(signal="direct", mode=mode)
    ppm = max(0.0, area - it) * 1000.0 / (rf * vol)
    print(f"  {mode:7}: area={area} vol={vol} -> ppm={ppm:.3f} (nominal 5.0, err {ppm/5*100-100:+.1f}%)")
