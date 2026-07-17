# -*- coding: utf-8 -*-
"""Extractor de dades injeccio a injeccio per a l'informe comparatiu de calibracions.

Llegeix el `CHECK/data/calibration_result.json` de cada SEQ_CAL (que ja porta els
cromatogrames per replica: t_doc, y_doc, t_dad, y_dad_254) i genera els fitxers
`cal_compare_data.json` (mode COLUMN) i `bp_data.json` (mode BP) que consumeix
`gen_docx.py`.

L'area de 254 nm es recalcula amb LA MATEIXA cadena de funcions que fa servir la
Suite a `_khp_integrate_254` (detect_main_peak -> find_peak_boundaries -> trapezoid
SENSE restar baseline), de manera que els valors coincideixen amb els del programa.

Us:
    python -X utf8 extreu_dades.py            # regenera els dos JSON
    python -X utf8 extreu_dades.py --validar  # comprova que reprodueix les files existents
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCR))  # per importar hpsec_*

from hpsec_core import detect_main_peak, find_peak_boundaries, get_baseline_stats  # noqa: E402
from scipy.integrate import trapezoid  # noqa: E402

DADES = r"C:\Users\maria\Proyectos\Dades3"

# Seqs de l'informe: (carpeta, mode, sensibilitat UIB documentada)
SEQS_COLUMN = [
    ("293_SEQ_CAL", "COLUMN", 700.0),
    ("305_SEQ_CAL", "COLUMN", 1000.0),
    ("306_SEQ_CAL", "COLUMN", 1000.0),
]
SEQS_BP = [
    ("292_SEQ_CAL_BP", "BP", 700.0),
    ("304_SEQ_CAL_BP", "BP", 1000.0),
]

PROMINENCE_PCT = 5.0  # per defecte de detect_main_peak (config.peak_min_prominence_pct)


def _cal_path(seq: str) -> str:
    return os.path.join(DADES, seq, "CHECK", "data", "calibration_result.json")


def _integra_254(t_dad, y_254, is_bp, mode):
    """Replica _khp_integrate_254 de la Suite. Retorna (t_max_254, area, area_total)."""
    t = np.asarray(t_dad or [], dtype=float)
    y = np.asarray(y_254 or [], dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) <= 10:
        return None, 0.0, 0.0
    info = detect_main_peak(t, y, PROMINENCE_PCT)
    if not (info and info.get("valid")):
        return None, 0.0, 0.0
    pk = info.get("peak_idx", 0)
    bl = get_baseline_stats(t, y, mode=mode).get("mean", 0)
    l_idx, r_idx = find_peak_boundaries(t, y, pk, bl, is_bp=is_bp)
    area = float(trapezoid(y[l_idx:r_idx + 1], t[l_idx:r_idx + 1])) if r_idx > l_idx else 0.0
    area_total = float(trapezoid(np.maximum(y, 0), t))
    return float(info.get("t_max", 0)), area, area_total


def _khp_name(filename: str) -> str:
    fn = filename or ""
    return fn.split("_R")[0] if "_R" in fn else fn


def _uib_lookup(cal: dict) -> dict:
    """{replica_num: entrada uib} a partir de replicas_info_uib."""
    out = {}
    for r in cal.get("replicas_info_uib") or []:
        out[str(r.get("replica_num"))] = r
    return out


def extreu_seq(seq: str, mode: str, uib_sens: float) -> dict:
    path = _cal_path(seq)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existeix {path} — cal processar la SEQ a la Suite.")
    d = json.load(open(path, encoding="utf-8"))
    is_bp = mode == "BP"
    rows = []
    for cal in d.get("calibrations") or []:
        conc = cal.get("conc_ppm")
        vol = cal.get("volume_uL")
        status = "VALID" if (cal.get("status") == "OK") else "INVALID"
        uib_by_rep = _uib_lookup(cal)
        for rep in cal.get("replicas_info") or []:
            rk = str(rep.get("replica_num"))
            bg = rep.get("bigaussian_doc") or {}
            t254, a254, a254_tot = _integra_254(
                rep.get("t_dad"), rep.get("y_dad_254"), is_bp, mode)
            doc_area = rep.get("area") or 0.0
            u = uib_by_rep.get(rk) or {}
            u_y = np.asarray(u.get("y_doc") or [], dtype=float)
            uib_ymax = float(np.nanmax(u_y)) if u_y.size else None
            shift_sec = rep.get("shift_sec")
            row = {
                "khp": _khp_name(rep.get("filename")),
                "conc": conc,
                "rep": rk,
                "vol": vol,
                "doc_area": doc_area,
                "doc_shift": (shift_sec / 60.0) if shift_sec is not None else None,
                "doc_r2": bg.get("r2"),
                "doc_fwhm": rep.get("fwhm_doc"),
                "doc_snr": rep.get("snr"),
                "doc_status": status,
                "doc_anoms": [a.get("code") for a in (rep.get("calibration_anomalies") or [])
                              if isinstance(a, dict)],
                "t254": t254,
                "a254_area": a254,
                "a254_ratio": (doc_area / a254) if a254 else None,
                "uib_area": u.get("area"),
                "uib_ymax": uib_ymax,
                "uib_sat": bool(uib_ymax is not None and uib_ymax >= 999),
                "uib_shift": (u.get("shift_sec") / 60.0) if u.get("shift_sec") is not None else None,
            }
            rows.append(row)
    return {"seq": seq, "method": mode, "uib_sens": uib_sens, "rows": rows}


def _prop(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(a - b) <= tol * max(1.0, abs(b))
    return a == b


def validar():
    """Comprova que l'extractor reprodueix les files ja publicades (293/306/292/304)."""
    ok = True
    for fitxer, seqs in (("cal_compare_data.json", SEQS_COLUMN), ("bp_data.json", SEQS_BP)):
        path = os.path.join(SCR, fitxer)
        if not os.path.exists(path):
            print(f"  [skip] {fitxer} no existeix encara")
            continue
        vell = json.load(open(path, encoding="utf-8"))
        for seq, mode, sens in seqs:
            if seq not in vell:
                print(f"  [nou]  {seq} — no era a {fitxer}, res a validar")
                continue
            nou = extreu_seq(seq, mode, sens)
            ref = {(r["khp"], r["conc"], r["rep"]): r for r in vell[seq]["rows"]}
            for r in nou["rows"]:
                k = (r["khp"], r["conc"], r["rep"])
                if k not in ref:
                    print(f"  [FALTA] {seq} {k} no era al fitxer vell"); ok = False; continue
                for camp in ("doc_area", "doc_r2", "doc_fwhm", "doc_snr", "doc_shift",
                             "t254", "a254_area", "doc_status"):
                    if not _prop(r.get(camp), ref[k].get(camp), tol=1e-4):
                        print(f"  [DIF]  {seq} {k} {camp}: nou={r.get(camp)!r} vell={ref[k].get(camp)!r}")
                        ok = False
            print(f"  {seq}: {len(nou['rows'])} files comprovades")
    print("VALIDACIO:", "OK — l'extractor reprodueix les dades publicades" if ok
          else "HI HA DIFERENCIES (revisa-les abans de regenerar)")
    return ok


def main():
    """Per defecte regenera NOMES cal_compare_data.json (mode COLUMN).

    bp_data.json queda CONGELAT amb els valors publicats a l'informe v4: els
    calibration_result.json de 292/304 s'han reprocessat des d'aleshores i ara
    contenen menys repliques (292: 9 vs 12 publicades; 304: 8 vs 10), de manera
    que regenerar-lo canviaria els numeros de BP de l'informe. Amb --bp es
    regenera igualment, assumint aquest canvi de forma explicita.
    """
    if "--validar" in sys.argv:
        validar()
        return
    col = {seq: extreu_seq(seq, mode, sens) for seq, mode, sens in SEQS_COLUMN}
    with open(os.path.join(SCR, "cal_compare_data.json"), "w", encoding="utf-8") as f:
        json.dump(col, f, ensure_ascii=False, indent=1)
    print("Desat cal_compare_data.json: "
          + ", ".join(f"{k}={len(v['rows'])} files" for k, v in col.items()))

    if "--bp" in sys.argv:
        bp = {seq: extreu_seq(seq, mode, sens) for seq, mode, sens in SEQS_BP}
        with open(os.path.join(SCR, "bp_data.json"), "w", encoding="utf-8") as f:
            json.dump(bp, f, ensure_ascii=False, indent=1)
        print("Desat bp_data.json: "
              + ", ".join(f"{k}={len(v['rows'])} files" for k, v in bp.items()))
    else:
        print("bp_data.json NO tocat (congelat als valors publicats; --bp per regenerar).")


if __name__ == "__main__":
    main()
