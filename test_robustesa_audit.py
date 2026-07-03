# -*- coding: utf-8 -*-
"""Tests de regressió de robustesa (auditoria 2026-06).

Cada test PROVOCA la condició de problema i comprova el comportament robust nou.
Executar: python test_robustesa_audit.py
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PASS = []
_FAIL = []


def check(name, cond):
    (_PASS if cond else _FAIL).append(name)
    print(("  OK  " if cond else " FALLA ") + name)


# ---------------------------------------------------------------------------
def test_atomic_write_and_cache_poisoning():
    """S2/S7: una escriptura fallida ha d'invalidar el cache i deixar el disc
    intacte (cap calibració fantasma)."""
    print("\n[S2/S7] Cache de calibració + escriptura atòmica")
    import hpsec_calibrate as hc
    tmp = tempfile.mkdtemp()
    ref = os.path.join(tmp, 'Calibration_Reference.json')
    json.dump({"version": "3.0", "calibrations": [{"id": "X"}]}, open(ref, 'w'))
    hc.get_calibration_reference_path = lambda: ref
    hc._cal_ref_cache = None
    hc._cal_ref_mtime = 0

    a = hc.load_calibration_reference()
    a['calibrations'].append({'poison': True})   # mutar la còpia
    b = hc.load_calibration_reference()
    check("load retorna còpia (mutar-la no enverina)", len(b['calibrations']) == 1)

    hc.load_calibration_reference()
    disk_before = open(ref, encoding='utf-8').read()
    orig = hc._atomic_write_json
    hc._atomic_write_json = lambda *a, **k: (_ for _ in ()).throw(IOError('disc ple'))
    res = hc.save_calibration_reference({"version": "3.0", "calibrations": [{"id": "FANTASMA"}]})
    cache_after = hc._cal_ref_cache
    hc._atomic_write_json = orig
    disk_after = open(ref, encoding='utf-8').read()
    nxt = hc.load_calibration_reference()
    check("save fallit -> False", res is False)
    check("cache invalidat despres del fail", cache_after is None)
    check("disc intacte (cap fantasma)", disk_before == disk_after)
    check("propera carrega neta", nxt['calibrations'][0]['id'] == 'X')


# ---------------------------------------------------------------------------
def test_load_manifest_corrupt():
    """S1: un manifest corromput NO s'ha de tractar com 'no existeix' en silenci:
    s'aparta a .corrupt i es retorna None."""
    print("\n[S1] load_manifest amb fitxer corromput")
    import hpsec_import as hi
    seq = tempfile.mkdtemp()
    data_folder = os.path.join(seq, "CHECK", "data")
    os.makedirs(data_folder, exist_ok=True)
    mpath = os.path.join(data_folder, "import_manifest.json")
    open(mpath, 'w', encoding='utf-8').write('{ corromput, no és json }')
    r = hi.load_manifest(seq)
    check("retorna None amb manifest corromput", r is None)
    check("aparta l'original a .corrupt", os.path.exists(mpath + ".corrupt"))
    check("no deixa el manifest corromput al seu lloc", not os.path.exists(mpath))


# ---------------------------------------------------------------------------
def test_volume_assumed_anomaly():
    """S1/#1: si no es troba el volum, quantify_sample no ha d'assumir en silenci:
    marca volume_source='assumed' i emet anomalia ANA_VOLUME_ASSUMED."""
    print("\n[#1] Volum d'injecció assumit -> anomalia")
    import hpsec_analyze as ha
    sample_result = {"processed": True, "sample_name": "TEST", "anomalies": []}
    try:
        ha.quantify_sample(sample_result, calibration_data=None, mode="COLUMN")
    except Exception:
        pass  # pot petar més avall per falta d'àrees; només ens importa la marca
    codes = [a.get("code") for a in sample_result.get("anomalies", []) if isinstance(a, dict)]
    check("s'ha emès ANA_VOLUME_ASSUMED", "ANA_VOLUME_ASSUMED" in codes)


# ---------------------------------------------------------------------------
def test_khp_alignment_checks():
    """#15/A: la comprovació d'alineació del KHP avisa si els pics són massa
    amples (degradació) o si el shift DOC↔254 és inconsistent (pic mal detectat)."""
    print("\n[#15/A] Comprovacions d'alineació derivades del KHP")
    import hpsec_calibrate as hc
    # Cas net: FWHM ~1, shift consistent -> cap avís
    ok = [{"filename": "KHP5_R1", "fwhm_doc": 1.0, "shift_min": -1.9},
          {"filename": "KHP2_R1", "fwhm_doc": 1.0, "shift_min": -1.85}]
    check("cas net: cap avís", len(hc._check_khp_alignment(ok)) == 0)
    # Pics amples
    wide = [{"filename": "KHP5_R1", "fwhm_doc": 3.0, "shift_min": -1.9},
            {"filename": "KHP2_R1", "fwhm_doc": 3.1, "shift_min": -1.9}]
    check("pics amples -> avís", any("amples" in w for w in hc._check_khp_alignment(wide)))
    # Shift inconsistent (un KHP amb punxada equivocada)
    bad = [{"filename": "KHP5_R1", "fwhm_doc": 1.0, "shift_min": -1.9},
           {"filename": "KHP2_R1", "fwhm_doc": 1.0, "shift_min": -1.85},
           {"filename": "KHP1_R1", "fwhm_doc": 1.0, "shift_min": -4.2}]
    warns = hc._check_khp_alignment(bad)
    check("shift inconsistent -> avís que cita el KHP", any("KHP1_R1" in w for w in warns))


def test_khp_measured_delay_recorded():
    """#15: el delay mesurat pel KHP es registra al JSON de processament."""
    import tempfile
    import json as _json
    import hpsec_calibrate as hc
    seq = tempfile.mkdtemp()
    ok = hc._save_khp_measured_delay(seq, {"seq_name": "T", "khp_measured_delay_min": -1.85})
    f = os.path.join(seq, "CHECK", "data", "khp_measured_delay.json")
    check("desa el JSON de delay mesurat", ok and os.path.exists(f))
    d = _json.load(open(f, encoding='utf-8'))
    check("conté el delay mesurat pel KHP", abs(d.get("khp_measured_delay_min") - (-1.85)) < 1e-9)


def test_no_np_trapz_in_production():
    """numpy 2.0 va eliminar np.trapz; els fitxers de producció han d'usar
    np.trapezoid (si no, l'anàlisi peta amb 'no attribute trapz')."""
    print("\n[303] Cap np.trapz als fitxers de producció")
    import glob
    root = os.path.dirname(os.path.abspath(__file__))
    prod = (glob.glob(os.path.join(root, "hpsec_*.py"))
            + [os.path.join(root, "khp_reintegrate_uib.py")])
    bad = [os.path.basename(f) for f in prod
           if os.path.exists(f) and "np.trapz(" in open(f, encoding='utf-8').read()]
    check("cap fitxer de producció usa np.trapz: " + (", ".join(bad) or "cap"), not bad)


def test_303_analyze_ok():
    """303: analitza sense error després d'arreglar np.trapz (abans: totes les
    seqs amb control/NaOH petaven)."""
    seq = r'C:\Users\maria\Proyectos\Dades3\303_SEQ'
    if not os.path.isdir(seq):
        print("\n[303] (saltat: dades no disponibles)")
        return
    print("\n[303] analyze_sequence OK (dades reals)")
    from hpsec_import import import_sequence
    from hpsec_analyze import analyze_sequence
    res = analyze_sequence(import_sequence(seq), None, do_quantify=False)
    check("analyze success", (res or {}).get("success") is True)
    check("sense errors a la llista", not (res or {}).get("errors"))


def test_fair_data_package():
    """FAIR: genera un Frictionless Data Package (results_SEC.csv net + datapackage.json
    amb esquema i unitats + README), tot en anglès, amb ppm i fraccions."""
    print("\n[FAIR] Frictionless Data Package")
    import tempfile
    import json as _json
    import pandas as pd
    from hpsec_fair import generate_data_package
    sg = {"S1": {"sample_type": "SAMPLE", "selected": {"doc": "1", "dad": "1"},
                 "replicas": {"1": {"injection_index": 5, "inj_volume": 400,
                                    "areas": {"DOC": {"BioP": 10.0, "HS": 20.0, "total": 40.0,
                                                      "HS_pct": 50.0},
                                              "A254": {"HS": 5.0, "total": 8.0}}}},
                 "quantification": {"concentration_ppm_direct": 3.4, "hci": 0.7}}}
    out = tempfile.mkdtemp()
    generate_data_package(sg, out, mode="COLUMN", seq_name="TEST", seq_date="2026-01-15")
    check("genera results + datapackage + README", all(
        os.path.exists(os.path.join(out, f))
        for f in ("results_SEC.csv", "datapackage.json", "README.txt")))
    df = pd.read_csv(os.path.join(out, "results_SEC.csv"))
    check("results_SEC.csv net (pandas) amb ppm",
          "conc_ppm_doc" in df.columns and float(df["conc_ppm_doc"].iloc[0]) == 3.4)
    check("porta fraccions DOC", "doc_hs_pct" in df.columns)
    dp = _json.load(open(os.path.join(out, "datapackage.json"), encoding="utf-8"))
    fields = dp["resources"][0]["schema"]["fields"]
    check("datapackage: esquema amb unitats", any(f.get("unit") for f in fields))
    check("datapackage: llicència oberta", dp["licenses"][0]["name"].startswith("CC-BY"))


def test_pre_margin_single_source():
    """#6: hpsec_delay ha de llegir el pre-margin de config (font única)."""
    print("\n[#6] Pre-margin des de config (font única)")
    import hpsec_delay as hd
    val = hd._config_pre_margin()
    check("_config_pre_margin retorna un número", isinstance(val, (int, float)) and val > 0)


# ---------------------------------------------------------------------------
def test_autofix_columns_synthetic():
    """S6/291: un full amb dades a les columnes '.1' (G-L) i originals buides
    s'ha de corregir (i quedar disponible per a TOTS els consumidors)."""
    print("\n[291] Auto-fix de columnes desplaçades (sintètic)")
    import pandas as pd
    import hpsec_import as hi
    df = pd.DataFrame({
        "Line#": [None, None], "Sample Name": [None, None],
        "Line#.1": [1, 2], "Sample Name.1": ["A", "B"],
    })
    fixed, applied = hi._autofix_shifted_columns(df)
    check("detecta i aplica la correcció", applied)
    check("la columna 'Sample Name' té dades després", fixed["Sample Name"].notna().sum() == 2)


def test_291_doc_direct_recovered():
    """291: amb l'auto-fix centralitzat, compute_toc_calc ha de poder calcular
    el 4-TOC_CALC → DOC Direct disponible (cap avís 'DOC Direct no disponible')."""
    seq = r'C:\Users\maria\Proyectos\Dades3\291_SEQ'
    if not os.path.isdir(seq):
        print("\n[291] (saltat: dades no disponibles)")
        return
    print("\n[291] DOC Direct recuperat (dades reals)")
    from hpsec_import import import_sequence
    imp = import_sequence(seq)
    ws = " ".join(str(w) for w in (imp.get("warnings") or []))
    check("import OK", imp.get("success") is True)
    check("cap avís 'DOC Direct no disponible'", "DOC Direct no disponible" not in ws)
    check("cap avís 'sense DOC Direct'", "sense DOC Direct" not in ws)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for t in (test_atomic_write_and_cache_poisoning,
              test_load_manifest_corrupt,
              test_volume_assumed_anomaly,
              test_khp_alignment_checks,
              test_khp_measured_delay_recorded,
              test_no_np_trapz_in_production,
              test_303_analyze_ok,
              test_fair_data_package,
              test_pre_margin_single_source,
              test_autofix_columns_synthetic,
              test_291_doc_direct_recovered):
        try:
            t()
        except Exception as e:
            import traceback
            _FAIL.append(t.__name__ + " (excepció)")
            print(" FALLA (excepció):", e)
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"PASSATS: {len(_PASS)}   FALLATS: {len(_FAIL)}")
    if _FAIL:
        for f in _FAIL:
            print("  x", f)
        sys.exit(1)
    print("TOTS OK")
