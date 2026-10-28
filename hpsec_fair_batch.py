#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador FAIR en bloc — HPSEC Suite.

Recorre seqüències JA ANALITZADES (que tenen CHECK/data/analysis_result.json) i
emet, per cada una, el paquet FAIR Frictionless (results_SEC.csv + traces/ +
datapackage.json + README) SENSE tornar a importar ni analitzar: reutilitza el
`samples_grouped` persistit (que conserva els arrays de senyal) i la calibració/
règim vigents segons la data d'adquisició de la SEQ.

És la peça que faltava: fins ara el FAIR només es generava d'una en una per la
GUI, sobre la SEQ oberta. Això ho fa per totes de cop, reproduïble, des de disc.

Ús:
    python hpsec_fair_batch.py --all
    python hpsec_fair_batch.py 290_SEQ 298_SEQ 303_SEQ
    python hpsec_fair_batch.py C:/ruta/a/290_SEQ --into C:/sortida_fair
    python hpsec_fair_batch.py --all --zip

Per cada SEQ, el paquet es desa a  <SEQ>/RESULTATS/FAIR/  (o a --into/<SEQ>/).

NOTA (traçabilitat): el paquet hereta tot el que hi ha al registre persistit. Si
una SEQ s'ha de tornar a processar (p. ex. reparar batmans a la Suite), cal
regenerar el seu analysis_result.json ABANS de generar el FAIR.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Consola Windows (cp1252): evitar que un accent/guió faci petar el print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import hpsec_analyze as A
import hpsec_fair as F
import hpsec_config as C
from hpsec_calibrate import get_seq_acquisition_date, get_calibration_for_date


def _resolve_seq_paths(args_seqs, use_all):
    """Resol la llista de carpetes SEQ a processar."""
    try:
        folders = C.get_data_folders() or []
    except Exception:
        folders = []
    if not folders:
        folders = [os.getcwd()]

    paths = []
    if use_all:
        for folder in folders:
            try:
                for d in sorted(os.listdir(folder)):
                    full = os.path.join(folder, d)
                    if os.path.isdir(full) and "_SEQ" in d:
                        paths.append(full)
            except OSError:
                continue
    for s in args_seqs:
        if os.path.isdir(s):
            paths.append(os.path.abspath(s))
            continue
        hit = None
        for folder in folders:
            cand = os.path.join(folder, s)
            if os.path.isdir(cand):
                hit = cand
                break
        if hit:
            paths.append(hit)
        else:
            print(f"  ! no trobo la seqüència: {s}")
    # dedup mantenint l'ordre
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _seq_date_str(seq_path, data):
    """Data d'adquisició (YYYY-MM-DD) per resoldre calibració i règim."""
    try:
        d = get_seq_acquisition_date(seq_path)
    except Exception:
        d = None
    if not d:
        d = data.get("analysis_date") or data.get("timestamp")
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10] if d else ""


def generate_for_seq(seq_path, into=None, make_zip=False):
    """Genera el paquet FAIR per UNA seqüència ja analitzada. Retorna dict resum."""
    seq_name = os.path.basename(seq_path.rstrip("/\\"))
    data = A.load_analysis_result(seq_path)
    if not data:
        return {"seq": seq_name, "ok": False, "reason": "sense analysis_result.json (no analitzada)"}

    sg = data.get("samples_grouped") or {}
    if not sg:
        return {"seq": seq_name, "ok": False, "reason": "analysis_result.json sense samples_grouped"}

    mode = data.get("method") or "COLUMN"
    seq_date = _seq_date_str(seq_path, data)

    # Calibració vigent per a la data (per al bloc calibration del datapackage).
    # El calibration_id per fila ja viu al registre (quantification), si es va quantificar.
    cal = None
    try:
        cal = get_calibration_for_date(seq_date, signal="direct") if seq_date else None
    except Exception:
        cal = None

    out_dir = (os.path.join(into, seq_name) if into
               else os.path.join(seq_path, "RESULTATS", "FAIR"))
    os.makedirs(out_dir, exist_ok=True)

    res = F.generate_data_package(
        sg, out_dir, mode=mode, calibration_data=cal,
        seq_name=seq_name, seq_date=seq_date, seq_path=seq_path,
    )
    result = {
        "seq": seq_name, "ok": bool(res.get("success", True)),
        "out": out_dir,
        "n_samples": res.get("n_samples"), "n_traces": res.get("n_traces"),
        "quantified": not data.get("quantification_pending", True),
    }
    if make_zip:
        try:
            import shutil
            zpath = out_dir.rstrip("/\\") + ".zip"
            if os.path.exists(zpath):
                os.remove(zpath)
            shutil.make_archive(out_dir.rstrip("/\\"), "zip", out_dir)
            result["zip"] = zpath
        except Exception as e:
            result["zip_error"] = str(e)
    return result


def main():
    ap = argparse.ArgumentParser(description="Genera el paquet FAIR en bloc des de SEQ ja analitzades.")
    ap.add_argument("seqs", nargs="*", help="Noms o rutes de SEQ (buit amb --all)")
    ap.add_argument("--all", action="store_true", help="Totes les SEQ de les carpetes de dades")
    ap.add_argument("--into", default=None, help="Carpeta de sortida (per defecte <SEQ>/RESULTATS/FAIR)")
    ap.add_argument("--zip", action="store_true", help="Comprimir cada paquet en un .zip")
    a = ap.parse_args()

    if not a.seqs and not a.all:
        ap.print_help()
        return

    paths = _resolve_seq_paths(a.seqs, a.all)
    if not paths:
        print("Cap seqüència a processar.")
        return

    print(f"FAIR en bloc: {len(paths)} sequencia(es)\n")
    ok = skipped = failed = 0
    for p in paths:
        try:
            r = generate_for_seq(p, into=a.into, make_zip=a.zip)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(p):22} {e}")
            failed += 1
            continue
        if not r.get("ok"):
            print(f"  [skip ] {r['seq']:22} {r.get('reason')}")
            skipped += 1
            continue
        q = "quantificat" if r.get("quantified") else "sense quantificar"
        print(f"  [ OK  ] {r['seq']:22} {r.get('n_samples')} mostres, "
              f"{r.get('n_traces')} traces, {q}")
        print(f"          -> {r['out']}")
        ok += 1

    print(f"\nFet: {ok} generades, {skipped} saltades, {failed} amb error.")


if __name__ == "__main__":
    main()
