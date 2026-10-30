"""
HPSEC Suite — Alineacio de retencio entre sequencies (ancorada al KHP)
======================================================================

Problema
--------
La columna envelleix i el pic del KHP es desplaça de temps de retencio al llarg
de la seva vida (observat: apex A254 del KHP ~20.7-22.4 min segons l'epoca, amb
FWHM estable -> es un DESPLAÇAMENT, no un eixamplament). Per comparar o integrar
cromatogrames de tota la vida de la columna cal portar-los a una MATEIXA posicio
temporal.

Metode (acordat amb LEQUIA)
---------------------------
Desplaçament rigid (1 punt) per sequencia, ancorat a l'apex del KHP a 254 nm:

    delta_t = t_khp_ref - t_khp_obs(seq)
    t_aligned = t_obs + delta_t

aplicat a totes les senyals de la sequencia (DOC Direct/UIB i A254/DAD). NO es fa
warping ni escalat: la deriva es gradual i la forma del pic es conserva, de manera
que un unic desplaçament rigid es suficient.

Nomes COLUMN
------------
En mode BP el KHP elueix a ~2-5 min (un altre eix) i NO s'alinea: es marca
applied=False, reason="BP_exclos".

No destructiu
-------------
Aquest modul NO reescriu els arrays de temps crus. Calcula delta_t i el deixa a
`provenance.retention_align` (per sequencia i per mostra). L'alineacio s'aplica
en DERIVAR (export FAIR, grafics) via `apply_shift`, de manera que el temps cru
es conserva i la transformacio queda documentada i es reversible.

L'ancora ha de ser robusta: un KHP trencat (batman) enverina el desplaçament, per
aixo cal reparar els batmans del KHP ABANS d'alinear. Aqui es filtra per finestra
i FWHM i, si no passa el QC, applied=False (delta_t=0, cap desplaçament fals).
"""

import logging

logger = logging.getLogger(__name__)

# Finestra plausible de l'apex del KHP en COLUMN (min). Fora d'aqui no s'accepta
# com a ancora (evita agafar el volum mort o cues tardanes).
KHP_WINDOW_COLUMN = (18.0, 27.0)

# FWHM maxim acceptable del pic KHP (min). Un KHP molt ample = pic dubtos/trencat.
FWHM_MAX = 1.8

# Referencia temporal canonica del KHP a 254 nm (min) per a COLUMN. Totes les
# sequencies COLUMN es porten a aquesta posicio. Valor dins del rang observat de
# la vida de la columna; es un PARAMETRE (overridable per `t_khp_ref`), no una
# constant fisica. Documentat a la memoria del projecte
# (project_hpsec_deriva_retencio).
DEFAULT_T_KHP_REF = 21.0

# Senyals als quals s'aplica el desplaçament (totes comparteixen eix de temps).
ALIGNED_SIGNALS = ["doc_direct", "doc_uib", "a254"]


def is_column_mode(mode):
    """True si el mode NO es BP (l'alineacio nomes te sentit en COLUMN)."""
    return str(mode or "").upper() != "BP"


def default_reference(mode="COLUMN", regime_id=None):
    """Referencia temporal del KHP (min). De moment unica per a tota la vida de
    la columna; el paramatre regime_id queda per a una futura referencia per
    regim si calgues."""
    return DEFAULT_T_KHP_REF


def qc_khp_apex(apex, fwhm=None, mode="COLUMN"):
    """Valida que l'apex (i opcionalment la FWHM) del KHP sigui una ancora fiable.

    Returns:
        (ok: bool, reason: str)
    """
    if apex is None:
        return False, "sense_apex_khp"
    try:
        apex = float(apex)
    except (TypeError, ValueError):
        return False, "apex_no_numeric"
    lo, hi = KHP_WINDOW_COLUMN
    if not (lo <= apex <= hi):
        return False, f"apex_fora_finestra_{apex:.2f}"
    if fwhm is not None:
        try:
            f = float(fwhm)
            if not (0.0 < f < FWHM_MAX):
                return False, f"fwhm_anomal_{f:.2f}"
        except (TypeError, ValueError):
            pass
    return True, "ok"


def compute_retention_align(t_khp_obs, mode="COLUMN", regime_id=None,
                            fwhm=None, t_khp_ref=None, khp_source=None):
    """Construeix el bloc `retention_align` per a una sequencia.

    Args:
        t_khp_obs: apex observat del KHP a 254 nm (min) en aquesta sequencia.
        mode: "COLUMN" o "BP".
        regime_id: identificador de regim instrumental (nomes metadada).
        fwhm: amplada a mitja alçada del KHP (min), per al QC (opcional).
        t_khp_ref: referencia a la qual portar el KHP (min). Si None -> default.
        khp_source: d'on surt l'apex ("khp_sample", "history", ...), metadada.

    Returns:
        dict amb l'esquema acordat. `applied`/`qc_ok` indiquen si hi ha
        desplaçament; `delta_t` es 0.0 quan no s'aplica.
    """
    base = {
        "model": "shift",
        "mode": str(mode or "").upper(),
        "signals": list(ALIGNED_SIGNALS),
        "regime_id": regime_id,
        "khp_source": khp_source,
    }

    if not is_column_mode(mode):
        base.update({"applied": False, "qc_ok": None, "reason": "BP_exclos",
                     "t_khp_obs": t_khp_obs, "t_khp_ref": None, "delta_t": 0.0})
        return base

    if t_khp_ref is None:
        t_khp_ref = default_reference(mode, regime_id)

    ok, reason = qc_khp_apex(t_khp_obs, fwhm, mode)
    if not ok:
        base.update({"applied": False, "qc_ok": False, "reason": reason,
                     "t_khp_obs": (float(t_khp_obs)
                                   if isinstance(t_khp_obs, (int, float)) else None),
                     "t_khp_ref": float(t_khp_ref), "delta_t": 0.0})
        return base

    delta = float(t_khp_ref) - float(t_khp_obs)
    base.update({"applied": True, "qc_ok": True, "reason": "ok",
                 "t_khp_obs": float(t_khp_obs), "t_khp_ref": float(t_khp_ref),
                 "fwhm": (float(fwhm) if isinstance(fwhm, (int, float)) else None),
                 "delta_t": round(delta, 4)})
    return base


def apply_shift(t, delta_t):
    """Retorna el temps alineat t + delta_t (numpy array). No muta l'entrada.

    Si delta_t es 0/None o t es None, retorna t tal qual."""
    import numpy as np
    if t is None or not delta_t:
        return t
    return np.asarray(t, dtype=float) + float(delta_t)


def sequence_khp_apex(result):
    """Extreu l'apex del KHP a 254 nm (min) d'un analysis_result.

    Ancora = `tmax_signals['A254']` de les repliques del grup KHP (es el marcador
    de retencio del patro; NO `peak_info.t_max`, que es el pic DOC). Mediana de
    les repliques KHP amb apex dins de finestra.

    Returns:
        (apex_min | None, fwhm_min | None, source | None)
    """
    import numpy as np
    sg = (result or {}).get("samples_grouped", {}) or {}
    apexes, fwhms = [], []
    lo, hi = KHP_WINDOW_COLUMN
    for _name, g in sg.items():
        if not (g.get("sample_type") == "KHP" or g.get("analysis_type") == "khp"):
            continue
        for rep in (g.get("replicas") or {}).values():
            if not isinstance(rep, dict):
                continue
            ts = rep.get("tmax_signals") or {}
            a = ts.get("A254")
            if a is None:
                continue
            try:
                a = float(a)
            except (TypeError, ValueError):
                continue
            if not (lo <= a <= hi):
                continue
            apexes.append(a)
            f = rep.get("fwhm_254")
            if isinstance(f, (int, float)):
                fwhms.append(float(f))
    if not apexes:
        return None, None, None
    return (float(np.median(apexes)),
            (float(np.median(fwhms)) if fwhms else None),
            "khp_sample")


def annotate_sequence(result, t_khp_ref=None, regime_id=None):
    """Calcula el bloc `retention_align` d'una sequencia i el desa a:
      - result['retention_align']  (resum de sequencia)
      - cada mostra: provenance['retention_align']  (viatja al FAIR)

    No modifica cap array de temps. Retorna el bloc calculat.
    """
    mode = result.get("method") or result.get("data_mode") or "COLUMN"
    apex, fwhm, source = sequence_khp_apex(result)
    if regime_id is None:
        regime_id = result.get("regime_id")

    align = compute_retention_align(
        apex, mode=mode, regime_id=regime_id, fwhm=fwhm,
        t_khp_ref=t_khp_ref, khp_source=source)

    result["retention_align"] = align

    # Mirall a cada mostra (llista plana i grouped) perque el FAIR i el full ID
    # el trobin al bloc de procedencia de la mostra.
    for s in result.get("samples", []):
        if isinstance(s, dict):
            s.setdefault("provenance", {})["retention_align"] = align
    for g in (result.get("samples_grouped", {}) or {}).values():
        for rep in (g.get("replicas", {}) or {}).values():
            if isinstance(rep, dict):
                rep.setdefault("provenance", {})["retention_align"] = align

    if align.get("applied"):
        logger.info("Retencio: KHP obs=%.3f ref=%.3f delta=%+.3f min",
                    align["t_khp_obs"], align["t_khp_ref"], align["delta_t"])
    else:
        logger.info("Retencio: sense alineacio (%s)", align.get("reason"))
    return align
