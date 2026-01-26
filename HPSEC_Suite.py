"""
HPSEC Suite v1.0
================
Aplicació unificada per al processament de dades HPSEC.

Pipeline:
1. Consolidar - Llegir fitxers .D i crear Excel consolidats
2. Calibrar - Calcular factor de calibració amb KHP
3. Processar - Detectar anomalies i seleccionar millors rèpliques
4. Exportar - Generar fitxers finals i informes PDF

Dissenyat per ser compilat a .exe amb PyInstaller.
"""

import os
import sys
import glob
import re
import json
import threading
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

from scipy.signal import find_peaks, savgol_filter, correlate, correlation_lags
from scipy.integrate import trapezoid
from scipy.stats import pearsonr
import shutil
import platform
import subprocess

# Mòduls HPSEC
from hpsec_replica import (
    evaluate_replica as evaluate_replica_unified,
    select_best_replica as select_best_replica_unified,
    compare_replicas as compare_replicas_unified,
)
from hpsec_consolidate import (
    consolidate_sequence,
    normalize_key, is_khp, mode_robust, obtenir_seq,
    llegir_doc_uib, llegir_dad_export3d, trobar_excel_mestre,
    netejar_nom_uib, extract_doc_from_master,
    get_baseline_correction, apply_smoothing,
    process_dad, detect_main_peak, analyze_sample_areas,
    write_consolidated_excel,
    check_sequence_files, save_file_check_report,
)
from hpsec_reports import (
    generate_consolidation_report,
    generate_chromatograms_report,
    generate_all_reports,
)
from hpsec_calibrate import (
    calibrate_sequence,
    analizar_khp_consolidado, analizar_khp_lote,
    buscar_khp_consolidados, find_khp_in_folder,
    load_local_calibrations, save_local_calibrations,
    get_active_calibration, is_seq_calibrated,
    load_khp_history, save_khp_history,
    get_khp_from_history, get_best_khp_from_history,
    register_calibration, mark_calibration_as_outlier,
    generate_calibration_id, extract_khp_conc,
    get_injection_volume, get_caldata_path, ensure_caldata_folder,
    detect_all_peaks, baseline_stats_time,
    calculate_peak_symmetry, calculate_peak_snr,
    expand_integration_limits_to_baseline,
    detect_batman as detect_batman_cal,
    detect_timeout as detect_timeout_cal,
    DEFAULT_CONFIG as CAL_DEFAULT_CONFIG,
    CALDATA_FOLDER, CALDATA_FILENAME,
    KHP_HISTORY_FILENAME,
)

# PDF generation
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

import warnings
warnings.filterwarnings("ignore")

# Logo path (relatiu a l'script)
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo_STRs.png")
PDF_HEADER = "Anàlisi de matèria orgànica per HPSEC-DAD-DOC"

# Configuració comparació rèpliques
TARGET_WAVELENGTHS_QAQC = [220, 254, 272, 290, 362]
DEFAULT_MIN_CORR = 0.95
DEFAULT_MAX_DIFF = 10.0
MAX_SHIFT_MINUTES = 6.0

# =============================================================================
# MOTOR MATEMÀTIC PER COMPARACIÓ RÈPLIQUES (de PROCESSA_14_2.py)
# =============================================================================

def netejar_baseline(y):
    """Neteja baseline amb percentil 1."""
    if len(y) == 0:
        return y
    y = np.asarray(y, dtype=float)
    baseline = np.percentile(y, 1)
    y_corrected = y - baseline
    y_corrected[y_corrected < 0] = 0
    return y_corrected


def avaluar_qualitat_doc(y):
    """Avalua qualitat DOC - menor score = millor qualitat."""
    if len(y) < 2:
        return 999999
    y_clean = netejar_baseline(y)
    area = np.sum(y_clean)
    if area < 1e-3:
        return 999999
    diffs = np.diff(y_clean)
    roughness = np.sum(np.abs(diffs))
    return roughness / area


def avaluar_qualitat_dad(y):
    """Avalua qualitat DAD - menor score = millor qualitat."""
    if len(y) == 0:
        return 999999
    y = np.asarray(y, dtype=float)
    negatius = y[y < 0]
    score_negatius = np.sum(np.abs(negatius))
    limit_baseline = np.percentile(y, 20)
    baseline_points = y[y <= limit_baseline]
    score_noise = np.std(baseline_points) * len(y) if len(baseline_points) > 0 else 0
    return score_negatius + score_noise


def alinear_senyals(t1, y1, t2, y2, t_min, t_max, threshold=0.0):
    """Alinea dos senyals amb cross-correlation."""
    dt = 0.01
    mask1 = (t1 >= t_min) & (t1 <= t_max)
    mask2 = (t2 >= t_min) & (t2 <= t_max)
    if not np.any(mask1) or not np.any(mask2):
        return None, None, None, 0

    t1_s, y1_s = t1[mask1], y1[mask1]
    t2_s, y2_s = t2[mask2], y2[mask2]

    t_start = max(t1_s.min(), t2_s.min())
    t_end = min(t1_s.max(), t2_s.max())
    t_grid = np.arange(t_start, t_end, dt)

    def reinterp(t_in, y_in, t_target):
        try:
            _, idx = np.unique(t_in, return_index=True)
            return np.interp(t_target, t_in[idx], y_in[idx], left=0, right=0)
        except:
            return np.zeros_like(t_target)

    y1_grid = reinterp(t1_s, y1_s, t_grid)
    y2_grid = reinterp(t2_s, y2_s, t_grid)

    y1_base = netejar_baseline(y1_grid)
    y2_base = netejar_baseline(y2_grid)

    y1_c = y1_base - np.mean(y1_base)
    y2_c = y2_base - np.mean(y2_base)

    if np.std(y1_c) < 1e-6 or np.std(y2_c) < 1e-6:
        shift_time = 0
    else:
        corr = correlate(y1_c, y2_c, mode='full')
        lags = correlation_lags(len(y1_c), len(y2_c), mode='full')
        shift_time = lags[np.argmax(corr)] * dt
        if abs(shift_time) > MAX_SHIFT_MINUTES:
            shift_time = 0

    t2_shifted = t2 + shift_time
    t_final = t_grid
    y1_final = reinterp(t1, y1, t_final)
    y2_final = reinterp(t2_shifted, y2, t_final)

    y1_net = netejar_baseline(y1_final)
    y2_net = netejar_baseline(y2_final)

    y1_net = np.where(y1_net < threshold, 0, y1_net)
    y2_net = np.where(y2_net < threshold, 0, y2_net)

    return t_final, y1_net, y2_net, shift_time


def calcular_estadistiques(t, y1, y2):
    """Calcula correlació Pearson i diferència d'àrea entre dos senyals."""
    if len(y1) < 2:
        return 0, 100
    try:
        corr, _ = pearsonr(y1, y2)
        area1 = trapezoid(y1, t)
        area2 = trapezoid(y2, t)
        if area1 < 1e-9:
            area1 = 1e-9
        diff_pct = abs((area1 - area2) / area1) * 100
        return corr, diff_pct
    except:
        return 0, 100


# =============================================================================
# MÈTRIQUES DAD - Avaluació individual de qualitat
# =============================================================================

# Thresholds per DAD
DAD_DERIVA_WARN = 10.0   # % deriva per WARN
DAD_DERIVA_FAIL = 20.0   # % deriva per FAIL
DAD_SOROLL_WARN = 2.0    # Soroll normalitzat per WARN
DAD_SOROLL_FAIL = 5.0    # Soroll normalitzat per FAIL


def calcular_deriva_baseline_dad(t, y, t_baseline_ini=0, t_baseline_fi=5, t_final_ini=50, t_final_fi=65):
    """
    Calcula la deriva de la línia base del DAD.

    Compara la mitjana de la zona inicial (baseline) amb la zona final.
    Retorna deriva com a percentatge del rang del senyal.

    Args:
        t: array de temps
        y: array de senyal
        t_baseline_ini, t_baseline_fi: rang temporal de la zona baseline inicial
        t_final_ini, t_final_fi: rang temporal de la zona final

    Returns:
        deriva_pct: deriva relativa en percentatge (pot ser negatiu)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 10 or len(y) < 10:
        return 0.0

    # Zona inicial (baseline)
    mask_ini = (t >= t_baseline_ini) & (t <= t_baseline_fi)
    if np.sum(mask_ini) < 5:
        mask_ini = np.arange(min(50, len(t)))  # Primers 50 punts
    baseline_ini = np.mean(y[mask_ini]) if np.any(mask_ini) else 0

    # Zona final
    mask_fi = (t >= t_final_ini) & (t <= t_final_fi)
    if np.sum(mask_fi) < 5:
        mask_fi = np.arange(max(0, len(t)-50), len(t))  # Últims 50 punts
    baseline_fi = np.mean(y[mask_fi]) if np.any(mask_fi) else 0

    # Deriva absoluta
    deriva = baseline_fi - baseline_ini

    # Normalitzar pel rang del senyal
    rang = np.max(y) - np.min(y)
    if rang < 1e-6:
        return 0.0

    deriva_pct = (deriva / rang) * 100
    return float(deriva_pct)


def calcular_soroll_dad(t, y, t_baseline_ini=0, t_baseline_fi=5):
    """
    Calcula el soroll (roughness) del DAD a la zona de baseline.

    Menor valor = menys soroll = millor qualitat.

    Args:
        t: array de temps
        y: array de senyal
        t_baseline_ini, t_baseline_fi: rang temporal de la zona baseline

    Returns:
        soroll: roughness normalitzat
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 10 or len(y) < 10:
        return 0.0

    # Zona baseline
    mask = (t >= t_baseline_ini) & (t <= t_baseline_fi)
    if np.sum(mask) < 10:
        mask = np.arange(min(100, len(t)))

    y_baseline = y[mask]

    if len(y_baseline) < 5:
        return 0.0

    # Roughness = suma de diferències absolutes consecutives
    diffs = np.abs(np.diff(y_baseline))
    roughness = np.mean(diffs)

    # Normalitzar per l'amplitud del senyal complet
    rang = np.max(y) - np.min(y)
    if rang < 1e-6:
        return 0.0

    soroll_norm = (roughness / rang) * 100
    return float(soroll_norm)


def avaluar_qualitat_dad_replica(df_dad, wavelengths, is_bp=False):
    """
    Avalua la qualitat del DAD d'una rèplica individual.

    Args:
        df_dad: DataFrame amb columnes 'time (min)' i les wavelengths
        wavelengths: llista de wavelengths a avaluar
        is_bp: si és mode ByPass (rangs de temps diferents)

    Returns:
        dict amb mètriques per cada wavelength i globals
    """
    result = {
        "valid": False,
        "per_wl": {},
        "deriva_global": 0.0,
        "soroll_global": 0.0,
        "status": "NO_DATA"
    }

    if df_dad is None or df_dad.empty:
        return result

    # Trobar columna de temps
    t_col = [c for c in df_dad.columns if 'time' in str(c).lower()]
    if not t_col:
        return result

    t = pd.to_numeric(df_dad[t_col[0]], errors='coerce').values

    # Rangs segons mode
    if is_bp:
        t_bl_ini, t_bl_fi = 0, 3
        t_fi_ini, t_fi_fi = 12, 18
    else:
        t_bl_ini, t_bl_fi = 0, 5
        t_fi_ini, t_fi_fi = 50, 65

    derives = []
    sorolls = []

    for wl in wavelengths:
        wl_str = str(wl)
        if wl_str not in df_dad.columns:
            continue

        y = pd.to_numeric(df_dad[wl_str], errors='coerce').values

        if len(y) < 10:
            continue

        deriva = calcular_deriva_baseline_dad(t, y, t_bl_ini, t_bl_fi, t_fi_ini, t_fi_fi)
        soroll = calcular_soroll_dad(t, y, t_bl_ini, t_bl_fi)

        result["per_wl"][wl] = {
            "deriva": deriva,
            "soroll": soroll,
            "t": t,
            "y": y
        }

        derives.append(abs(deriva))
        sorolls.append(soroll)

    if not derives:
        return result

    result["valid"] = True
    result["deriva_global"] = float(np.mean(derives))
    result["soroll_global"] = float(np.mean(sorolls))

    # Determinar status
    max_deriva = max(derives)
    max_soroll = max(sorolls)

    if max_deriva >= DAD_DERIVA_FAIL or max_soroll >= DAD_SOROLL_FAIL:
        result["status"] = "FAIL"
    elif max_deriva >= DAD_DERIVA_WARN or max_soroll >= DAD_SOROLL_WARN:
        result["status"] = "WARN"
    else:
        result["status"] = "OK"

    return result


def seleccionar_millor_dad(qual_r1, qual_r2):
    """
    Selecciona la millor rèplica DAD basant-se en deriva i soroll.

    Args:
        qual_r1, qual_r2: resultats de avaluar_qualitat_dad_replica

    Returns:
        (seleccio, motiu): "1" o "2" i el motiu
    """
    if not qual_r1.get("valid") and not qual_r2.get("valid"):
        return "1", "Cap DAD vàlid"

    if not qual_r1.get("valid"):
        return "2", "R1 no vàlid"

    if not qual_r2.get("valid"):
        return "1", "R2 no vàlid"

    der1 = qual_r1["deriva_global"]
    der2 = qual_r2["deriva_global"]
    sroll1 = qual_r1["soroll_global"]
    sroll2 = qual_r2["soroll_global"]

    # Prioritzar menor deriva
    if abs(der1 - der2) > 3:  # Diferència significativa en deriva
        if der1 < der2:
            return "1", f"Menys deriva ({der1:.1f}% vs {der2:.1f}%)"
        else:
            return "2", f"Menys deriva ({der2:.1f}% vs {der1:.1f}%)"

    # Si deriva similar, mirar soroll
    if sroll1 < sroll2:
        return "1", f"Menys soroll ({sroll1:.2f} vs {sroll2:.2f})"
    elif sroll2 < sroll1:
        return "2", f"Menys soroll ({sroll2:.2f} vs {sroll1:.2f})"

    return "1", "Similar qualitat"


# =============================================================================
# VERSIÓ I CONFIGURACIÓ
# =============================================================================
VERSION = "1.0.0"
APP_NAME = "HPSEC Suite"

# Colors corporatius
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "light": "#f8f9fa",
    "dark": "#343a40",
    "white": "#ffffff",
}

# Configuració per defecte
DEFAULT_CONFIG = {
    # Consolidació
    "bp_baseline_win": 1.0,
    "col_baseline_start": 10.0,
    "target_wavelengths": [220, 252, 254, 272, 290, 362],
    "dad_subsample": 5,

    # Calibració
    "khp_pattern": "KHP",
    "peak_min_prominence_pct": 5.0,
    "alignment_threshold_sec": 4.0,  # Umbral per aplicar alineació DOC-DAD

    # Processament
    "timeout_min_height_frac": 0.30,
    "batman_max_sep_min": 0.5,
    "batman_min_height_pct": 15.0,
    "batman_min_sigma": 3.0,
    "ear_min_height_frac": 0.05,

    # Scoring
    "score_timeout": 100,
    "score_batman": 50,
    "score_ear": 5,
    "score_low_area": 20,
    "score_roughness": 1,

    # Fraccions de temps per KPIs
    "time_fractions": {
        "BioP": [0, 18],
        "HS": [18, 23],
        "BB": [23, 30],
        "Carbonats": [30, 40],
        "LMW": [40, 70],
    },
}


# =============================================================================
# GENERACIÓ PDF PROFESSIONAL
# =============================================================================
def get_pdf_styles():
    """Retorna estils personalitzats per als PDFs."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='MainHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=rl_colors.HexColor("#2E86AB"),
        spaceAfter=6,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name='SubHeader',
        parent=styles['Normal'],
        fontSize=10,
        textColor=rl_colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=rl_colors.HexColor("#2E86AB"),
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=0,
        borderColor=rl_colors.HexColor("#2E86AB"),
        borderPadding=4,
    ))

    styles.add(ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
    ))

    styles.add(ParagraphStyle(
        name='SmallText',
        parent=styles['Normal'],
        fontSize=8,
        textColor=rl_colors.HexColor("#888888"),
    ))

    return styles


def create_pdf_header(story, styles, title, seq_name=""):
    """Crea capçalera professional per PDF."""
    # Logo si existeix
    if os.path.exists(LOGO_PATH):
        try:
            img = Image(LOGO_PATH, width=3*cm, height=1.5*cm)
            img.hAlign = 'LEFT'
            story.append(img)
        except:
            pass

    # Títol principal
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(PDF_HEADER, styles['SubHeader']))
    story.append(Paragraph(title, styles['MainHeader']))

    if seq_name:
        story.append(Paragraph(f"Seqüència: {seq_name}", styles['SubHeader']))

    story.append(Paragraph(f"Generat: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['SmallText']))
    story.append(Spacer(1, 0.5*cm))

    # Línia separadora
    line_table = Table([['']], colWidths=[17*cm])
    line_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 1, rl_colors.HexColor("#2E86AB")),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 0.5*cm))


def create_data_table(data, col_widths=None, header_color="#2E86AB"):
    """Crea taula de dades formatada."""
    if not data:
        return None

    tbl = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (0, -1), rl_colors.HexColor("#f5f5f5")),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#cccccc")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#fafafa")]),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


def ensure_check_folder(base_path):
    """Assegura que existeix la carpeta CHECK."""
    check_path = os.path.join(base_path, "CHECK")
    os.makedirs(check_path, exist_ok=True)
    return check_path


# Constant per Samples History (no està a hpsec_calibrate)
SAMPLES_HISTORY_FILENAME = "Samples_History.json"

# =============================================================================
# HISTÒRIC DE MOSTRES (Samples_History.json)
# =============================================================================

def get_samples_history_path(seq_path):
    """Retorna el path del fitxer Samples_History.json (a la carpeta pare de SEQs)."""
    if not seq_path:
        return None
    parent_folder = os.path.dirname(seq_path)
    return os.path.join(parent_folder, SAMPLES_HISTORY_FILENAME)


def load_samples_history(seq_path):
    """
    Carrega l'històric de mostres.
    Retorna una llista de diccionaris amb les mostres.
    """
    history_path = get_samples_history_path(seq_path)
    if not history_path or not os.path.exists(history_path):
        return []

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("samples", [])
    except Exception as e:
        print(f"Error carregant històric mostres: {e}")
        return []


def save_samples_history(seq_path, samples):
    """
    Guarda l'històric de mostres.
    """
    history_path = get_samples_history_path(seq_path)
    if not history_path:
        return False

    try:
        data = {
            "version": "1.0",
            "updated": datetime.now().isoformat(),
            "total_samples": len(samples),
            "samples": samples
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardant històric mostres: {e}")
        return False


def add_sample_to_history(seq_path, sample_data):
    """
    Afegeix una mostra a l'històric.
    Si ja existeix (mateix seq_name + mostra + replica), l'actualitza.

    Args:
        seq_path: Path de la SEQ (per localitzar el JSON)
        sample_data: Dict amb les dades de la mostra
    """
    samples = load_samples_history(seq_path)

    # Buscar si ja existeix
    seq_name = sample_data.get('seq_name', '')
    mostra = sample_data.get('mostra', '')
    replica = sample_data.get('replica', 1)

    updated = False
    for i, s in enumerate(samples):
        if (s.get('seq_name') == seq_name and
            s.get('mostra') == mostra and
            s.get('replica') == replica):
            samples[i] = sample_data
            updated = True
            break

    if not updated:
        samples.append(sample_data)

    return save_samples_history(seq_path, samples)


def add_samples_batch(seq_path, samples_list, replace_seq=True):
    """
    Afegeix múltiples mostres a l'històric.

    Args:
        seq_path: Path de la SEQ
        samples_list: Llista de dicts amb les mostres
        replace_seq: Si True, elimina mostres anteriors de la mateixa SEQ
    """
    samples = load_samples_history(seq_path)
    seq_name = os.path.basename(seq_path)

    if replace_seq:
        # Eliminar mostres anteriors d'aquesta SEQ
        samples = [s for s in samples if s.get('seq_name') != seq_name]

    # Afegir noves mostres
    samples.extend(samples_list)

    return save_samples_history(seq_path, samples)


def get_sample_history(seq_path, mostra_name, mode=None, doc_mode=None):
    """
    Obté l'històric d'una mostra específica.

    Args:
        seq_path: Path de la SEQ (per localitzar el JSON)
        mostra_name: Nom de la mostra a buscar
        mode: Filtrar per mode (COLUMN/BP), None = tots
        doc_mode: Filtrar per font DOC (UIB/DIRECT), None = tots

    Returns:
        Llista de registres de la mostra ordenats per data
    """
    samples = load_samples_history(seq_path)

    # Filtrar per nom de mostra
    filtered = [s for s in samples if s.get('mostra', '').upper() == mostra_name.upper()]

    # Filtrar per mode si especificat
    if mode:
        filtered = [s for s in filtered if s.get('mode') == mode]

    # Filtrar per doc_mode si especificat
    if doc_mode:
        filtered = [s for s in filtered if s.get('doc_mode') == doc_mode]

    # Ordenar per data (més recent primer)
    def parse_date(s):
        date_str = s.get('date', '')
        try:
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.strptime(str(date_str)[:10], fmt[:len(str(date_str)[:10])])
                except:
                    continue
            return datetime.min
        except:
            return datetime.min

    filtered.sort(key=parse_date, reverse=True)

    return filtered


def get_samples_by_seq(seq_path, seq_name=None, mode=None, doc_mode=None, exclude_khp=True):
    """
    Obté totes les mostres d'una SEQ.

    Args:
        seq_path: Path per localitzar el JSON
        seq_name: Nom de la SEQ (si None, usa seq_path)
        mode: Filtrar per mode
        doc_mode: Filtrar per font DOC
        exclude_khp: Excloure mostres KHP

    Returns:
        Llista de mostres de la SEQ
    """
    samples = load_samples_history(seq_path)

    if seq_name is None:
        seq_name = os.path.basename(seq_path)

    filtered = [s for s in samples if s.get('seq_name') == seq_name]

    if mode:
        filtered = [s for s in filtered if s.get('mode') == mode]

    if doc_mode:
        filtered = [s for s in filtered if s.get('doc_mode') == doc_mode]

    if exclude_khp:
        filtered = [s for s in filtered if not is_khp(s.get('mostra', ''))]

    return filtered


def get_samples_stats(seq_path, mostra_name, mode=None):
    """
    Calcula estadístiques d'una mostra (similar a get_equivalent_seqs_stats per KHP).

    Returns:
        Dict amb n_seqs, mean_doc_area, std_doc_area, etc.
    """
    history = get_sample_history(seq_path, mostra_name, mode=mode)

    if not history:
        return None

    doc_areas = [s.get('doc_area', 0) for s in history if s.get('doc_area', 0) > 0]
    a254_areas = [s.get('a254_area', 0) for s in history if s.get('a254_area', 0) > 0]

    if not doc_areas:
        return None

    return {
        "n_registres": len(history),
        "n_seqs": len(set(s.get('seq_name') for s in history)),
        "mean_doc_area": float(np.mean(doc_areas)),
        "std_doc_area": float(np.std(doc_areas)) if len(doc_areas) > 1 else 0,
        "min_doc_area": float(np.min(doc_areas)),
        "max_doc_area": float(np.max(doc_areas)),
        "rsd_doc_area": float(np.std(doc_areas) / np.mean(doc_areas) * 100) if np.mean(doc_areas) > 0 else 0,
        "mean_a254_area": float(np.mean(a254_areas)) if a254_areas else 0,
        "std_a254_area": float(np.std(a254_areas)) if len(a254_areas) > 1 else 0,
        "seqs_list": list(set(s.get('seq_name') for s in history)),
        "modes": list(set(s.get('mode') for s in history)),
    }


def analyze_sample_areas(t_doc, y_doc, df_dad, peak_info, config=None):
    """
    Analitza una mostra i calcula totes les àrees (DOC + DAD wavelengths).

    Args:
        t_doc: Array de temps DOC (minuts)
        y_doc: Array de senyal DOC (net, amb baseline corregida)
        df_dad: DataFrame amb dades DAD (columnes: time (min), 210, 254, 280, etc.)
        peak_info: Dict amb info del pic principal (de detect_main_peak)
        config: Configuració (opcional)

    Returns:
        Dict amb totes les àrees calculades
    """
    from scipy.integrate import trapezoid

    # Longituds d'ona objectiu (de la config)
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362]) if config else [220, 252, 254, 272, 290, 362]

    result = {
        'doc_area': 0.0,
        'doc_t_retention': 0.0,
        'doc_t_start': 0.0,
        'doc_t_end': 0.0,
        'dad_wavelengths': {},  # Per guardar totes les longituds d'ona
        'valid': False,
    }
    # Inicialitzar àrees per cada longitud d'ona objectiu
    for wl in target_wls:
        result[f'a{wl}_area'] = 0.0

    # Verificar que tenim dades
    if t_doc is None or y_doc is None or len(t_doc) < 10:
        return result

    t_doc = np.asarray(t_doc).flatten()
    y_doc = np.asarray(y_doc).flatten()

    # Àrea DOC
    if peak_info and peak_info.get('valid'):
        result['doc_area'] = peak_info.get('area', 0.0)
        result['doc_t_retention'] = peak_info.get('t_max', 0.0)
        result['doc_t_start'] = peak_info.get('t_start', 0.0)
        result['doc_t_end'] = peak_info.get('t_end', 0.0)
        result['valid'] = True

        # Usar límits del pic DOC per integrar DAD
        t_start = result['doc_t_start']
        t_end = result['doc_t_end']

        # Àrees DAD per cada longitud d'ona
        if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
            t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

            # Trobar índexs corresponents als límits del pic DOC
            if t_start > 0 and t_end > t_start:
                dad_left_idx = int(np.searchsorted(t_dad, t_start))
                dad_right_idx = int(np.searchsorted(t_dad, t_end))

                # Assegurar bounds vàlids
                dad_left_idx = max(0, min(dad_left_idx, len(t_dad) - 1))
                dad_right_idx = max(0, min(dad_right_idx, len(t_dad) - 1))

                if dad_right_idx > dad_left_idx:
                    # Calcular àrea per totes les longituds d'ona disponibles
                    for col in df_dad.columns:
                        if col == 'time (min)':
                            continue
                        try:
                            wl_str = str(col)
                            y_wl = pd.to_numeric(df_dad[col], errors='coerce').to_numpy()
                            if len(y_wl) > dad_right_idx:
                                area_wl = float(trapezoid(
                                    y_wl[dad_left_idx:dad_right_idx+1],
                                    t_dad[dad_left_idx:dad_right_idx+1]
                                ))
                                # Guardar amb clau a{wl}_area si és objectiu
                                try:
                                    wl_int = int(wl_str)
                                    if wl_int in target_wls:
                                        result[f'a{wl_int}_area'] = area_wl
                                except ValueError:
                                    pass
                                # Guardar sempre al diccionari general
                                result['dad_wavelengths'][wl_str] = area_wl
                        except:
                            continue

    return result


def create_sample_entry(seq_path, mostra, replica, date, mode, doc_mode,
                        sample_analysis, calibration=None):
    """
    Crea una entrada de mostra per guardar a l'històric.

    Args:
        seq_path: Path de la SEQ
        mostra: Nom de la mostra
        replica: Número de rèplica
        date: Data de la seqüència
        mode: COLUMN o BP
        doc_mode: UIB o DIRECT
        sample_analysis: Resultat de analyze_sample_areas()
        calibration: Dict amb info de calibració (opcional, per calcular concentració)

    Returns:
        Dict preparat per guardar a Samples_History.json
    """
    seq_name = os.path.basename(seq_path)

    entry = {
        "seq_name": seq_name,
        "seq_path": seq_path,
        "date": date,
        "mode": mode,
        "doc_mode": doc_mode,
        "mostra": mostra,
        "replica": replica,
        "is_khp": is_khp(mostra),
        # Àrees DOC
        "doc_area": sample_analysis.get('doc_area', 0.0),
        "doc_t_retention": sample_analysis.get('doc_t_retention', 0.0),
        "doc_t_start": sample_analysis.get('doc_t_start', 0.0),
        "doc_t_end": sample_analysis.get('doc_t_end', 0.0),
        # Àrees DAD (6 longituds d'ona)
        "a220_area": sample_analysis.get('a220_area', 0.0),
        "a252_area": sample_analysis.get('a252_area', 0.0),
        "a254_area": sample_analysis.get('a254_area', 0.0),
        "a272_area": sample_analysis.get('a272_area', 0.0),
        "a290_area": sample_analysis.get('a290_area', 0.0),
        "a362_area": sample_analysis.get('a362_area', 0.0),
        # Totes les longituds d'ona (dict)
        "dad_wavelengths": sample_analysis.get('dad_wavelengths', {}),
        # Concentració (si hi ha calibració)
        "concentration": 0.0,
        "cal_seq": None,
        "cal_factor": None,
    }

    # Calcular concentració si tenim calibració
    if calibration and sample_analysis.get('doc_area', 0) > 0:
        factor = calibration.get('factor', 0)
        if factor > 0:
            entry['concentration'] = sample_analysis['doc_area'] * factor
            entry['cal_seq'] = calibration.get('seq_name', '')
            entry['cal_factor'] = factor

    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONS STANDALONE PER COMPARTIR AMB BATCH
# ═══════════════════════════════════════════════════════════════════════════════

def write_consolidated_excel(out_path, mostra, rep, seq_out, date_master,
                             method, doc_mode, fitxer_doc, fitxer_dad,
                             st_doc, st_dad, t_doc, y_doc_raw, y_doc_net,
                             baseline, df_dad, peak_info, sample_analysis=None,
                             master_file=None, row_start=None, row_end=None):
    """
    Escriu fitxer Excel consolidat amb àrees calculades.

    Funció standalone que pot ser usada tant per la Suite com pel batch.
    """
    sample_analysis = sample_analysis or {}

    id_rows = [
        ("Date", date_master),
        ("Method", method),
        ("Mostra", mostra),
        ("Rèplica", rep),
        ("SEQ", seq_out),
        ("DOC_MODE", doc_mode),
        ("Fitxer_DOC_Original", fitxer_doc),
        ("Fitxer_DAD_Original", fitxer_dad),
        ("Estat_DOC", st_doc),
        ("Estat_DAD", st_dad),
        ("DOC_N_POINTS", len(t_doc)),
        ("DAD_N_POINTS", len(df_dad) if not df_dad.empty else 0),
        # Àrees DOC
        ("PEAK_VALID", peak_info.get("valid", False)),
        ("DOC_AREA", sample_analysis.get("doc_area", peak_info.get("area", 0.0))),
        ("DOC_T_RETENTION", sample_analysis.get("doc_t_retention", peak_info.get("t_max", 0.0))),
        ("DOC_T_START", sample_analysis.get("doc_t_start", peak_info.get("t_start", 0.0))),
        ("DOC_T_END", sample_analysis.get("doc_t_end", peak_info.get("t_end", 0.0))),
        # Àrees DAD (6 longituds d'ona)
        ("A220_AREA", sample_analysis.get("a220_area", 0.0)),
        ("A252_AREA", sample_analysis.get("a252_area", 0.0)),
        ("A254_AREA", sample_analysis.get("a254_area", 0.0)),
        ("A272_AREA", sample_analysis.get("a272_area", 0.0)),
        ("A290_AREA", sample_analysis.get("a290_area", 0.0)),
        ("A362_AREA", sample_analysis.get("a362_area", 0.0)),
    ]

    if doc_mode == "DIRECT" and master_file:
        id_rows.append(("Fitxer_Master", master_file))
        if row_start is not None:
            id_rows.append(("DOC_ROW_START", row_start))
        if row_end is not None:
            id_rows.append(("DOC_ROW_END", row_end))

    df_id = pd.DataFrame(id_rows, columns=["Camp", "Valor"])

    df_doc_out = pd.DataFrame({
        "time (min)": t_doc,
        "DOC (mAU)": y_doc_net,
        "DOC_RAW (mAU)": y_doc_raw,
        "BASELINE (mAU)": baseline,
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_id.to_excel(writer, sheet_name="ID", index=False)
        df_doc_out.to_excel(writer, sheet_name="DOC", index=False)
        if not df_dad.empty:
            df_dad.to_excel(writer, sheet_name="DAD", index=False)


def scan_seq_folders_for_khp(base_path, progress_callback=None):
    """
    Escaneja totes les carpetes SEQ cercant fitxers de calibració KHP.

    Args:
        base_path: Carpeta pare on estan totes les SEQ
        progress_callback: Funció per reportar progrés (opcional)

    Returns:
        Llista de calibracions trobades
    """
    calibrations = []

    # Cercar carpetes que continguin "SEQ" o segueixen patró de seqüències
    seq_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Buscar carpetes SEQ o amb patró típic (dates, etc.)
            if "SEQ" in item.upper() or re.match(r'\d{4}[-_]?\d{2}[-_]?\d{2}', item):
                seq_folders.append(item_path)

    total = len(seq_folders)

    for i, seq_folder in enumerate(seq_folders):
        if progress_callback:
            progress_callback(int(100 * (i + 1) / max(total, 1)), seq_folder)

        # Buscar carpeta CONSOLIDAT o Resultats_Consolidats
        consolidat_path = os.path.join(seq_folder, "CONSOLIDAT")
        resultats_path = os.path.join(seq_folder, "Resultats_Consolidats")

        # Prioritzar Resultats_Consolidats (batch), després CONSOLIDAT (antic)
        if os.path.exists(resultats_path):
            search_path = resultats_path
        elif os.path.exists(consolidat_path):
            search_path = consolidat_path
        else:
            continue

        # Buscar fitxers KHP
        khp_files = glob.glob(os.path.join(search_path, "*KHP*.xlsx"))
        khp_files = [f for f in khp_files if not os.path.basename(f).startswith("~$")]

        if not khp_files:
            continue

        # Determinar mode (COLUMN o BP)
        mode = "BP" if "BP" in os.path.basename(seq_folder).upper() else "COLUMN"

        # Analitzar KHP
        try:
            khp_data = analizar_khp_lote(khp_files, DEFAULT_CONFIG)
            if khp_data:
                # Usar data de la SEQ si disponible, sinó data de modificació del fitxer
                seq_date = khp_data.get('seq_date', '')
                if not seq_date:
                    seq_date = datetime.fromtimestamp(os.path.getmtime(khp_files[0])).isoformat()

                entry = {
                    "seq_name": os.path.basename(seq_folder),
                    "seq_path": seq_folder,
                    "date": seq_date,  # Data de la seqüència
                    "date_processed": datetime.now().isoformat(),
                    "mode": mode,
                    "khp_file": khp_data.get('filename', 'N/A'),
                    "khp_source": "SCAN",
                    "doc_mode": khp_data.get('doc_mode', 'N/A'),  # UIB o DIRECT
                    "conc_ppm": khp_data.get('conc_ppm', 0),
                    "area": khp_data.get('area', 0),
                    "shift_min": khp_data.get('shift_min', 0),
                    "shift_sec": khp_data.get('shift_sec', khp_data.get('shift_min', 0) * 60),
                    "symmetry": khp_data.get('symmetry', 1.0),
                    "snr": khp_data.get('snr', 0),
                    "all_peaks_count": khp_data.get('all_peaks_count', 1),
                    "n_replicas": khp_data.get('n_replicas', 1),
                    "rsd": khp_data.get('rsd', 0),
                    "has_batman": khp_data.get('has_batman', False),
                    "has_timeout": khp_data.get('has_timeout', False),
                    "status": khp_data.get('status', 'OK'),
                    "quality_issues": khp_data.get('quality_issues', []),
                    "is_bp": khp_data.get('is_bp', False),
                }
                calibrations.append(entry)
        except Exception as e:
            print(f"Error escanejant {seq_folder}: {e}")

    # Ordenar per data
    calibrations.sort(key=lambda x: x.get("date", ""), reverse=True)

    return calibrations


def export_history_to_excel(seq_path):
    """
    Exporta l'històric de calibracions a Excel per fàcil consulta.
    """
    calibrations = load_khp_history(seq_path)
    if not calibrations:
        return None

    history_path = get_history_path(seq_path)
    excel_path = os.path.join(os.path.dirname(history_path), KHP_HISTORY_EXCEL_FILENAME)

    # Crear DataFrame amb tota la info rellevant
    rows = []
    for cal in calibrations:
        rows.append({
            "Seqüència": cal.get("seq_name", ""),
            "Data SEQ": cal.get("date", "")[:10] if cal.get("date") else "",
            "Mode": cal.get("mode", ""),
            "SOURCE DOC": cal.get("doc_mode", "N/A"),  # UIB o DIRECT
            "Volum (µL)": cal.get("volume_uL", cal.get("volume", 0)),
            "Fitxer KHP": cal.get("khp_file", ""),
            "Font KHP": cal.get("khp_source", ""),
            "Conc (ppm)": cal.get("conc_ppm", 0),
            "t_max (min)": f"{cal.get('t_retention', 0):.2f}",  # TEMPS RETENCIÓ
            "Àrea": f"{cal.get('area', 0):.2f}",
            "Factor": f"{cal.get('factor', 0):.6f}" if cal.get('factor') else "-",
            "SHIFT (s)": f"{cal.get('shift_sec', 0):.1f}",
            "Simetria": f"{cal.get('symmetry', 1.0):.2f}",
            "SNR": f"{cal.get('snr', 0):.1f}",
            "Pics": cal.get("all_peaks_count", 1),
            "N Rèpliques": cal.get("n_replicas", 1),
            "RSD (%)": f"{cal.get('rsd', 0):.1f}" if cal.get('rsd') else "-",
            "Batman": "Sí" if cal.get("has_batman") else "No",
            "TimeOUT": "Sí" if cal.get("has_timeout") else "No",
            "Límits Exp.": "Sí" if cal.get("limits_expanded", False) else "No",
            "Outlier": "Sí" if cal.get("is_outlier", False) else "No",
            "Estat": "OUTLIER" if cal.get("is_outlier") else cal.get("status", ""),
            "Data Proc.": cal.get("date_processed", "")[:10] if cal.get("date_processed") else "",
        })

    df = pd.DataFrame(rows)

    try:
        df.to_excel(excel_path, index=False, sheet_name="Històric KHP")
        return excel_path
    except Exception as e:
        print(f"Error exportant històric a Excel: {e}")
        return None


def get_khp_from_history(seq_path, target_seq_name, mode="COLUMN"):
    """
    Recupera les dades de KHP d'una seqüència específica de l'històric.
    """
    calibrations = load_khp_history(seq_path)

    for cal in calibrations:
        if cal.get("seq_name") == target_seq_name and cal.get("mode") == mode:
            return cal

    return None


def get_best_khp_from_history(seq_path, mode="COLUMN", exclude_current=True):
    """
    Retorna el millor KHP de l'històric per al mode especificat.
    Prioritza: SNR alt, RSD baix, sense issues de qualitat.
    """
    calibrations = load_khp_history(seq_path)
    current_seq = os.path.basename(seq_path) if seq_path else ""

    candidates = []
    for cal in calibrations:
        if cal.get("mode") != mode:
            continue
        if exclude_current and cal.get("seq_name") == current_seq:
            continue
        if cal.get("status") == "FAIL":
            continue

        # Score: SNR alt és bo, RSD baix és bo, issues de qualitat penalitzen
        score = cal.get("snr", 0) - cal.get("rsd", 0) * 2
        score -= len(cal.get("quality_issues", [])) * 10

        candidates.append((score, cal))

    if not candidates:
        return None

    # Retornar el de millor score
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_equivalent_seqs_stats(seq_path, mode="COLUMN", doc_mode=None, conc_ppm=None, volume_uL=None, exclude_current=True):
    """
    Obté estadístiques de SEQs equivalents de l'històric.

    Args:
        seq_path: Path de la seqüència actual
        mode: COLUMN o BP
        doc_mode: UIB o DIRECT (si None, accepta qualsevol)
        conc_ppm: Concentració KHP (si None, accepta qualsevol)
        volume_uL: Volum d'injecció en µL (si None, accepta qualsevol)
        exclude_current: Excloure la seqüència actual

    Returns:
        Dict amb estadístiques: n_seqs, mean_area, std_area, min_area, max_area, seqs_list
    """
    calibrations = load_khp_history(seq_path)
    current_seq = os.path.basename(seq_path) if seq_path else ""

    equivalent = []
    for cal in calibrations:
        # Excloure actual si cal
        if exclude_current and cal.get("seq_name") == current_seq:
            continue

        # Excloure outliers
        if cal.get("is_outlier", False):
            continue

        # Filtrar per mode (COLUMN/BP)
        if cal.get("mode") != mode:
            continue

        # Filtrar per doc_mode (UIB/DIRECT) si especificat
        if doc_mode is not None and doc_mode != "N/A":
            cal_doc_mode = cal.get("doc_mode", "N/A")
            if cal_doc_mode != "N/A" and cal_doc_mode != doc_mode:
                continue

        # Filtrar per concentració si especificada
        if conc_ppm is not None and cal.get("conc_ppm") != conc_ppm:
            continue

        # Filtrar per volum si especificat
        if volume_uL is not None:
            cal_volume = cal.get("volume_uL", 0)
            if cal_volume > 0 and cal_volume != volume_uL:
                continue

        # Només incloure si té àrea vàlida
        area = cal.get("area", 0)
        if area > 0:
            equivalent.append(cal)

    if not equivalent:
        return None

    # Ordenar per data (més recent primer)
    def parse_date(cal):
        date_str = cal.get("seq_date", cal.get("date", ""))
        try:
            # Intentar diversos formats
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.strptime(str(date_str)[:10], fmt[:len(str(date_str)[:10])])
                except:
                    continue
            return datetime.min
        except:
            return datetime.min

    equivalent_sorted = sorted(equivalent, key=parse_date, reverse=True)

    # Agafar les 5 més recents
    closest_5 = equivalent_sorted[:5]

    areas = [cal.get("area", 0) for cal in equivalent]
    closest_areas = [cal.get("area", 0) for cal in closest_5]

    # Estadístiques de temps de retenció
    t_retentions = [cal.get("t_retention", cal.get("t_doc_max", 0)) for cal in equivalent if cal.get("t_retention", cal.get("t_doc_max", 0)) > 0]

    # Estadístiques A254 ratio
    a254_ratios = [cal.get("a254_doc_ratio", 0) for cal in equivalent if cal.get("a254_doc_ratio", 0) > 0]

    return {
        "n_seqs": len(equivalent),
        "mean_area": float(np.mean(areas)),
        "std_area": float(np.std(areas)),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas)),
        "rsd_area": float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else 0,
        # Estadístiques temps retenció
        "mean_t_retention": float(np.mean(t_retentions)) if t_retentions else 0,
        "std_t_retention": float(np.std(t_retentions)) if len(t_retentions) > 1 else 0,
        # Estadístiques A254 ratio
        "mean_a254_ratio": float(np.mean(a254_ratios)) if a254_ratios else 0,
        "std_a254_ratio": float(np.std(a254_ratios)) if len(a254_ratios) > 1 else 0,
        "n_with_a254": len(a254_ratios),
        # Les 5 més recents ordenades per data (inclou t_retention i a254_ratio)
        "closest_seqs": [(
            cal.get("seq_name", "?"),
            cal.get("area", 0),
            cal.get("seq_date", cal.get("date", ""))[:10],
            cal.get("doc_mode", "N/A"),
            cal.get("t_retention", cal.get("t_doc_max", 0)),
            cal.get("a254_doc_ratio", 0)
        ) for cal in closest_5],
        "all_areas": areas,
        "closest_areas": closest_areas,
        "mode": mode,
        "doc_mode": doc_mode,
    }


# =============================================================================
# FUNCIONS CALIBRACIÓ KHP (de CAL_v1.py)
# =============================================================================
def calculate_integration_limits(t, y, peak_idx, min_width_min=1.0, max_width_min=6.0):
    """
    Calcula els límits d'integració d'un pic KHP usant mètode simplificat.

    LÒGICA SIMPLIFICADA:
    1. Baseline = MODA de tot el senyal (valor més freqüent)
    2. Threshold = baseline + 3*sigma (estadístic)
    3. Límits = des del pic fins que senyal ≤ threshold

    Args:
        t: Array de temps (minuts)
        y: Array de senyal
        peak_idx: Índex del màxim del pic
        min_width_min: Amplada mínima en minuts (default 1.0)
        max_width_min: Amplada màxima en minuts (default 6.0)

    Returns:
        Dict amb left_idx, right_idx, baseline, threshold, width_minutes, etc.
    """
    try:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n < 10:
            return {
                "left_idx": 0, "right_idx": n - 1,
                "baseline": 0, "threshold": 0,
                "width_minutes": float(t[-1] - t[0]) if n > 1 else 0,
                "valid": False, "message": "Senyal massa curt"
            }

        # === 1. BASELINE = MODA de tot el senyal ===
        baseline = mode_robust(y)

        # === 2. STD dels punts propers a baseline ===
        # Considerar "propers" els que estan dins del 20% del rang
        y_range = float(np.max(y) - baseline)
        if y_range <= 0:
            return {
                "left_idx": 0, "right_idx": n - 1,
                "baseline": baseline, "threshold": baseline,
                "width_minutes": float(t[-1] - t[0]),
                "valid": False, "message": "Senyal sense pic"
            }

        mask_baseline = y < (baseline + 0.2 * y_range)
        if np.sum(mask_baseline) > 5:
            std_baseline = float(np.std(y[mask_baseline]))
        else:
            std_baseline = float(np.std(y)) * 0.1  # Fallback

        # Evitar std = 0
        if std_baseline < 1e-6:
            std_baseline = 0.01 * y_range

        # === 3. THRESHOLD = baseline + 3*sigma ===
        threshold = baseline + 3 * std_baseline

        # Calcular límits en índexs basats en temps
        dt = np.mean(np.diff(t)) if n > 1 else 0.01
        max_width_idx = int(max_width_min / dt) if dt > 0 else 300
        min_width_idx = int(min_width_min / dt) if dt > 0 else 50

        # === 4. BUSCAR LÍMIT ESQUERRE ===
        left_idx = peak_idx
        for i in range(peak_idx - 1, max(0, peak_idx - max_width_idx), -1):
            if y[i] <= threshold:
                left_idx = i
                break
            left_idx = i  # Continuar expandint
        else:
            left_idx = max(0, peak_idx - max_width_idx)

        # === 5. BUSCAR LÍMIT DRET ===
        right_idx = peak_idx
        for i in range(peak_idx + 1, min(n, peak_idx + max_width_idx)):
            if y[i] <= threshold:
                right_idx = i
                break
            right_idx = i  # Continuar expandint
        else:
            right_idx = min(n - 1, peak_idx + max_width_idx)

        # === 6. VALIDAR AMPLADA MÍNIMA ===
        current_width_idx = right_idx - left_idx
        if current_width_idx < min_width_idx:
            expand_needed = (min_width_idx - current_width_idx) // 2 + 1
            left_idx = max(0, left_idx - expand_needed)
            right_idx = min(n - 1, right_idx + expand_needed)

        # Assegurar que el pic està dins dels límits
        left_idx = int(min(left_idx, peak_idx - 3))
        right_idx = int(max(right_idx, peak_idx + 3))
        left_idx = max(0, left_idx)
        right_idx = min(n - 1, right_idx)

        # Verificar si els límits arriben a baseline
        left_at_baseline = y[left_idx] <= threshold
        right_at_baseline = y[right_idx] <= threshold
        width_minutes = float(t[right_idx] - t[left_idx])

        return {
            "left_idx": left_idx,
            "right_idx": right_idx,
            "baseline": baseline,
            "std_baseline": std_baseline,
            "threshold": threshold,
            "width_minutes": width_minutes,
            "left_at_baseline": left_at_baseline,
            "right_at_baseline": right_at_baseline,
            "valid": left_at_baseline and right_at_baseline,
            "message": "OK" if (left_at_baseline and right_at_baseline) else "Límits no arriben a baseline"
        }

    except Exception as e:
        print(f"Error calculant límits integració: {e}")
        import traceback
        traceback.print_exc()
        return {
            "left_idx": 0, "right_idx": len(y) - 1 if len(y) > 0 else 0,
            "baseline": 0, "threshold": 0,
            "width_minutes": 0, "valid": False,
            "message": f"Error: {e}"
        }


# Mantenim funció antiga per compatibilitat (crida la nova)
def expand_integration_limits_to_baseline(t, y, left_idx, right_idx, peak_idx,
                                          baseline_threshold_pct=10,
                                          min_width_minutes=1.0,
                                          max_width_minutes=6.0,
                                          is_bp=False):
    """Wrapper per compatibilitat - crida la nova funció simplificada."""
    result = calculate_integration_limits(t, y, peak_idx, min_width_minutes, max_width_minutes)

    # Adaptar format de retorn per compatibilitat
    expanded_left = max(0, left_idx - result["left_idx"])
    expanded_right = max(0, result["right_idx"] - right_idx)

    return {
        "left_idx": result["left_idx"],
        "right_idx": result["right_idx"],
        "expanded_left": expanded_left,
        "expanded_right": expanded_right,
        "baseline": result["baseline"],
        "threshold_value": result["threshold"],
        "original_valid": (expanded_left == 0 and expanded_right == 0),
        "left_at_baseline": result["left_at_baseline"],
        "right_at_baseline": result["right_at_baseline"],
        "width_minutes": result["width_minutes"],
    }


def validate_integration_baseline(t, y, left_idx, right_idx, peak_idx, baseline_threshold_pct=15):
    """
    Valida que els límits d'integració arribin a valors propers a la línia base.
    """
    try:
        y = np.asarray(y)
        peak_height = y[peak_idx]

        # Calcular baseline
        search_range = max(50, (right_idx - left_idx))
        local_region = y[max(0, left_idx - search_range):min(len(y), right_idx + search_range)]
        baseline = np.percentile(local_region, 5)

        effective_height = peak_height - baseline
        if effective_height <= 0:
            return {"valid": True, "message": "Pic no detectat correctament"}

        # Valors als límits
        left_value = y[left_idx] - baseline
        right_value = y[right_idx] - baseline

        left_pct = (left_value / effective_height) * 100 if effective_height > 0 else 0
        right_pct = (right_value / effective_height) * 100 if effective_height > 0 else 0

        left_at_baseline = left_pct <= baseline_threshold_pct
        right_at_baseline = right_pct <= baseline_threshold_pct
        valid = left_at_baseline and right_at_baseline

        if valid:
            message = "OK"
        else:
            issues = []
            if not left_at_baseline:
                issues.append(f"límit esquerre alt ({left_pct:.0f}%)")
            if not right_at_baseline:
                issues.append(f"límit dret alt ({right_pct:.0f}%)")
            message = "Límits integració estrets: " + ", ".join(issues)

        return {
            "valid": valid,
            "left_at_baseline": left_at_baseline,
            "right_at_baseline": right_at_baseline,
            "left_value_pct": left_pct,
            "right_value_pct": right_pct,
            "message": message,
        }

    except Exception as e:
        return {"valid": True, "message": f"Error validació: {e}"}


def buscar_khp_consolidados_gui(input_folder):
    """
    Cerca KHP en tres fases: LOCAL → SIBLINGS → MANUAL
    Retorna (llista_fitxers, origen)
    """
    # FASE 1: LOCAL (Resultats_Consolidats o Consolidat)
    res_cons = os.path.join(input_folder, "Resultats_Consolidats")
    khp_files = find_khp_in_folder(res_cons)

    if khp_files:
        return khp_files, "LOCAL"

    # Fallback a carpeta antiga "Consolidat"
    consolidat_old = os.path.join(input_folder, "Consolidat")
    khp_files = find_khp_in_folder(consolidat_old)

    if khp_files:
        return khp_files, "LOCAL"

    # FASE 2: SIBLINGS (carpetes germanes amb mateix prefix numèric)
    folder_name = os.path.basename(input_folder)
    parent_dir = os.path.dirname(input_folder)

    match = re.match(r'^(\d+)', folder_name)
    if match:
        seq_id = match.group(1)

        try:
            all_folders = [d for d in os.listdir(parent_dir)
                          if os.path.isdir(os.path.join(parent_dir, d))]

            siblings = [d for d in all_folders
                       if d.startswith(seq_id) and d != folder_name]

            for sib in siblings:
                sib_res_cons = os.path.join(parent_dir, sib, "Resultats_Consolidats")
                khp_files = find_khp_in_folder(sib_res_cons)

                if khp_files:
                    return khp_files, f"SIBLING:{sib}"
        except Exception:
            pass

    # FASE 3: MANUAL
    if messagebox.askyesno("KHP NO TROBAT",
                          "No s'ha trobat cap KHP.\n\nVols seleccionar manualment una carpeta amb KHP?"):
        d = filedialog.askdirectory(title="Selecciona Carpeta SEQ amb KHP")
        if d:
            manual_res = os.path.join(d, "Resultats_Consolidats")
            if os.path.exists(manual_res):
                khp_files = find_khp_in_folder(manual_res)
                if khp_files:
                    return khp_files, "MANUAL"
            khp_files = find_khp_in_folder(d)
            if khp_files:
                return khp_files, "MANUAL"

    return [], "NONE"


# analizar_khp_consolidado i analizar_khp_lote importats de hpsec_calibrate
# buscar_khp_consolidados importat de hpsec_calibrate (usar amb gui_callbacks)


# NOTA: Codi de analizar_khp_consolidado i analizar_khp_lote eliminat
# Usar imports de hpsec_calibrate


# =============================================================================
# FUNCIONS CONSOLIDACIÓ - (normalize_key, is_khp, mode_robust, obtenir_seq
# importats de hpsec_consolidate)
# =============================================================================

def seq_tag(seq, bp):
    """Afegeix _BP si és seqüència BP."""
    if bp and "BP" not in seq.upper():
        return f"{seq}_BP"
    return seq


def is_bp_seq(folder, sample_dad_files=None):
    """Detecta si és seqüència BP."""
    name = os.path.basename(os.path.normpath(folder)).upper()
    if re.search(r"(^|[_\-\s])BP($|[_\-\s])", name):
        return True

    if sample_dad_files:
        for p in sample_dad_files[:3]:
            try:
                df, st = llegir_dad_export3d(p)
                if st.startswith("OK") and not df.empty and "time (min)" in df.columns:
                    tmax = pd.to_numeric(df["time (min)"], errors="coerce").max()
                    if pd.notna(tmax) and float(tmax) <= 18.0:
                        return True
            except Exception:
                continue
    return False


def skip_sample_direct(sample_desc):
    """Determina si s'ha d'ignorar una mostra."""
    d = str(sample_desc or "").strip().upper()
    if d in ("", "0"):
        return True
    if "PRE-HPLC" in d or ("POST" in d and "HPLC" in d):
        return True
    return False


def split_sample_rep(sample_desc):
    """Separa nom de mostra i rèplica."""
    s = str(sample_desc or "").strip()
    if not s:
        return "", None
    m = re.match(r"^(.*?)(?:[_\-\s]?R(\d+))\s*$", s, flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2)
    return s, None


def clean_sample_name(sample_desc):
    """Neteja nom de mostra."""
    s = str(sample_desc or "").strip()
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "SAMPLE"


def get_baseline_correction(t, y, mode_type="COL", config=None):
    """Correcció de baseline segons mode BP o COLUMN."""
    config = config or DEFAULT_CONFIG
    if mode_type == "BP":
        mask = t < config["bp_baseline_win"]
        val = mode_robust(y[mask]) if np.sum(mask) > 10 else float(np.nanmin(y))
        return np.full_like(y, val)

    mask = t < config["col_baseline_start"]
    val = mode_robust(y[mask]) if np.sum(mask) > 10 else float(np.nanmin(y))
    return np.full_like(y, val)


def apply_smoothing(y):
    """Aplica suavitzat Savgol."""
    y = np.asarray(y)
    if len(y) < 11:
        return y
    return savgol_filter(y, 11, 3)


def read_master_date(seq_folder):
    """Llegeix data del fitxer mestre."""
    xls = glob.glob(os.path.join(seq_folder, "*.xlsx"))
    for f in xls:
        if "~$" in os.path.basename(f):
            continue
        try:
            xf = pd.ExcelFile(f, engine="openpyxl")
            sheet = None
            for s in xf.sheet_names:
                if str(s).strip().lower() == "0-check":
                    sheet = s
                    break
            if sheet is None:
                sheet = 0
            df = pd.read_excel(xf, sheet_name=sheet, header=None, engine="openpyxl")
            val = df.iloc[1, 1]
            if pd.notna(val):
                return str(val)
        except Exception:
            pass
    return ""


def llegir_doc_uib(path):
    """Llegeix fitxer DOC format UIB (CSV tab-separated)."""
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, header=None, engine="python")
            df = df.iloc[:, [0, 1]]
            df.columns = ["time (min)", "DOC"]
            return df, f"OK{' (UTF-8)' if enc == 'utf-8' else ''}"
        except Exception:
            continue
    return pd.DataFrame(), "Error"


def llegir_dad_export3d(path):
    """Llegeix fitxer DAD Export3D (CSV comma-separated)."""
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep=",", encoding=enc, engine="python")
            if df.shape[1] == 0:
                return pd.DataFrame(), "Buit"
            cols = list(df.columns)
            cols[0] = "time (min)"
            out_cols = [cols[0]]
            for c in cols[1:]:
                sc = str(c).strip()
                try:
                    v = float(sc)
                    out_cols.append(str(int(v)) if v.is_integer() else str(v))
                except Exception:
                    out_cols.append(sc)
            df.columns = out_cols
            return df, f"OK{' (UTF-8)' if enc == 'utf-8' else ''}"
        except Exception:
            continue
    return pd.DataFrame(), "Error"


def netejar_nom_uib(nom_fitxer):
    """Neteja nom de fitxer UIB i extreu mostra/rèplica."""
    stem = os.path.splitext(nom_fitxer)[0]
    stem = re.sub(r"UIB1B\d*", "", stem, flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]

    rep = None
    if tokens:
        last = tokens[-1]
        if last.isdigit():
            rep = last
            tokens = tokens[:-1]
        else:
            m = re.fullmatch(r"R(\d+)", last, flags=re.IGNORECASE)
            if m:
                rep = m.group(1)
                tokens = tokens[:-1]

    if rep is None:
        rep = "1"

    mostra = "_".join(tokens).strip("_")
    if not mostra:
        mostra = os.path.splitext(nom_fitxer)[0]
    return mostra, rep


def trobar_excel_mestre(folder_seq):
    """Troba Excel mestre amb sheets 2-TOC i 4-SEQ_DATA."""
    candidats = []
    for ext in ("*.xlsx", "*.XLSX", "*.xlsm", "*.XLSM"):
        candidats.extend(glob.glob(os.path.join(folder_seq, ext)))
    candidats = [p for p in candidats if not os.path.basename(p).startswith("~$")]

    for p in sorted(candidats):
        try:
            xl = pd.ExcelFile(p, engine="openpyxl")
            s = set(xl.sheet_names)
            if "2-TOC" in s and "4-SEQ_DATA" in s:
                return p
        except Exception:
            continue
    return None


def llegir_master_direct(mestre):
    """Llegeix TOC i SEQ_DATA del mestre."""
    df_toc = pd.read_excel(mestre, sheet_name="2-TOC", header=6, engine="openpyxl")
    df_seq = pd.read_excel(mestre, sheet_name="4-SEQ_DATA", header=0, engine="openpyxl")
    return df_toc, df_seq


def extract_doc_from_master(df_toc, row_ini, row_fi, start_dt):
    """Extreu segment DOC del mestre."""
    time_col = "Date Started" if "Date Started" in df_toc.columns else df_toc.columns[3]
    if "TOC(ppb)" in df_toc.columns:
        sig_col = "TOC(ppb)"
    elif "TC(ppb)" in df_toc.columns:
        sig_col = "TC(ppb)"
    else:
        sig_col = df_toc.columns[5]

    start_idx = max(int(row_ini) - 1, 0)
    end_idx = max(int(row_fi), start_idx)
    seg = df_toc.iloc[start_idx:end_idx].copy()

    seg_time = pd.to_datetime(seg[time_col], errors="coerce")
    t0 = pd.to_datetime(start_dt, errors="coerce")
    t_min = (seg_time - t0).dt.total_seconds() / 60.0

    df_doc = pd.DataFrame({"time (min)": t_min, "DOC": pd.to_numeric(seg[sig_col], errors="coerce")})
    df_doc = df_doc.dropna(subset=["time (min)", "DOC"])
    return df_doc


def list_dad_files(folder_export3d, folder_csv):
    """Llista fitxers DAD disponibles."""
    out = []
    for folder in [folder_export3d, folder_csv]:
        if not folder or not os.path.isdir(folder):
            continue
        for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
            out.extend(glob.glob(os.path.join(folder, ext)))

    out2 = []
    for p in sorted(set(out)):
        bn = os.path.basename(p)
        if bn.startswith("~$"):
            continue
        if re.search(r"UIB1B", bn, flags=re.IGNORECASE):
            continue
        out2.append(p)
    return out2


def normalize_rep(rep):
    """Normalitza número de rèplica."""
    if rep is None:
        return ""
    s = str(rep).strip()
    if not s:
        return ""
    try:
        return str(int(s))
    except Exception:
        return s


def detect_dad_rep_style(dad_files):
    """Detecta estil de nomenclatura de rèpliques."""
    cnt_us = 0
    cnt_r = 0
    for p in dad_files or []:
        stem = os.path.splitext(os.path.basename(p))[0]
        if re.search(r"_[0-9]+$", stem):
            cnt_us += 1
        if re.search(r"[-_\s]R[0-9]+$", stem, flags=re.IGNORECASE):
            cnt_r += 1

    if cnt_us == 0 and cnt_r == 0:
        return ""
    return "UNDERSCORE" if cnt_us >= cnt_r else "R"


def dad_sample_rep_from_path(path):
    """Extreu mostra i rèplica del path DAD."""
    stem = os.path.splitext(os.path.basename(path))[0]

    m = re.match(r"^(.*)_([0-9]+)$", stem)
    if m:
        base = m.group(1).rstrip("-_ ")
        return base, m.group(2), "UNDERSCORE"

    m = re.match(r"^(.*?)[-_\s]R([0-9]+)$", stem, flags=re.IGNORECASE)
    if m:
        base = m.group(1).rstrip("-_ ")
        return base, m.group(2), "R"

    return stem, None, ""


def target_keys_from_desc(sample_desc):
    """Genera claus de matching per una mostra."""
    s = str(sample_desc or "").strip()
    keys = []
    if not s:
        return keys

    keys.append(normalize_key(s))

    tokens = [t for t in re.split(r"\s+", s) if t]
    if tokens:
        last = re.sub(r"[^A-Za-z0-9]+", "", tokens[-1])
        if last and not last.isdigit():
            keys.append(normalize_key(last))

    out = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def choose_best_candidate(target_key, rep, dad_files, used_files=None):
    """Tria el millor fitxer DAD per una mostra."""
    if not dad_files:
        return None, "NO_DATA"

    used = used_files if used_files is not None else set()
    rep_norm = normalize_rep(rep)
    keys = target_keys_from_desc(target_key)
    if not keys:
        return None, "NO_DATA"

    preferred_style = detect_dad_rep_style(dad_files)

    def iter_candidates(style_only):
        for p in dad_files:
            if p in used:
                continue
            s_part, r_part, style = dad_sample_rep_from_path(p)
            if style_only and preferred_style and style and style != preferred_style:
                continue
            yield p, s_part, r_part

    def exact_pass(style_only):
        out = []
        for p, s_part, r_part in iter_candidates(style_only):
            if not r_part:
                continue
            if rep_norm and normalize_rep(r_part) != rep_norm:
                continue
            sk = normalize_key(s_part)
            if sk in keys:
                out.append(p)
        return out

    exact = exact_pass(True) if preferred_style else exact_pass(False)
    if not exact and preferred_style:
        exact = exact_pass(False)

    if len(exact) == 1:
        return exact[0], "OK"
    if len(exact) > 1:
        return None, "NO_DATA"

    def name_only_pass(style_only):
        out = []
        for p, s_part, r_part in iter_candidates(style_only):
            sk = normalize_key(s_part)
            if sk in keys:
                out.append(p)
        return out

    name_only = name_only_pass(True) if preferred_style else name_only_pass(False)
    if len(name_only) == 0 and preferred_style:
        name_only = name_only_pass(False)

    if len(name_only) == 1:
        return name_only[0], "OK"

    return None, "NO_DATA"


def process_dad(df_dad, config=None):
    """Processa DAD: extreu wavelengths i submostreig."""
    config = config or DEFAULT_CONFIG
    if df_dad is None or df_dad.empty:
        return pd.DataFrame()

    target_wls = config["target_wavelengths"]
    cols_to_keep = ["time (min)"]

    for wl in target_wls:
        wl_str = str(wl)
        if wl_str in df_dad.columns:
            cols_to_keep.append(wl_str)

    if len(cols_to_keep) == 1:
        return pd.DataFrame()

    df_filtered = df_dad[cols_to_keep].copy()

    subsample = config["dad_subsample"]
    indices = [0] + list(range(subsample, len(df_filtered), subsample))
    df_sub = df_filtered.iloc[indices].reset_index(drop=True)

    return df_sub


# =============================================================================
# FUNCIONS CORE (de hpsec_utils expandit)
# =============================================================================
def baseline_stats(y, pct_low=10, pct_high=30):
    """Calcula estadístiques de la baseline usant percentils (mètode legacy)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) < 10:
        return {"mean": 0.0, "std": 0.0, "threshold_3sigma": 0.0}

    p_low = np.percentile(y, pct_low)
    p_high = np.percentile(y, pct_high)
    mask = (y >= p_low) & (y <= p_high)
    baseline_points = y[mask]

    if len(baseline_points) < 5:
        baseline_points = y[y <= p_high]

    if len(baseline_points) < 2:
        return {"mean": float(p_low), "std": 0.0, "threshold_3sigma": float(p_low)}

    mean_val = float(np.mean(baseline_points))
    std_val = float(np.std(baseline_points))

    return {
        "mean": mean_val,
        "std": std_val,
        "threshold_3sigma": mean_val + 3.0 * std_val
    }


def baseline_stats_time(t, y, t_start=0, t_end=2.0, fallback_pct=10):
    """
    Calcula estadístiques de la baseline usant una finestra temporal específica.

    Millor per BP on el pic ocupa gran part del cromatograma.

    Args:
        t: array de temps (min)
        y: array de senyal
        t_start: inici finestra baseline (min)
        t_end: final finestra baseline (min)
        fallback_pct: si no hi ha prou punts, usar primers X% del senyal

    Returns:
        dict amb mean, std, threshold_3sigma
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filtrar NaN
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]

    if len(y) < 10:
        return {"mean": 0.0, "std": 0.0, "threshold_3sigma": 0.0}

    # Seleccionar punts dins la finestra temporal
    mask = (t >= t_start) & (t <= t_end)
    baseline_points = y[mask]

    # Fallback: si no hi ha prou punts, usar primers X%
    if len(baseline_points) < 10:
        n_fallback = max(10, int(len(y) * fallback_pct / 100))
        baseline_points = y[:n_fallback]

    if len(baseline_points) < 2:
        return {"mean": float(np.median(y)), "std": 0.0, "threshold_3sigma": float(np.median(y))}

    mean_val = float(np.mean(baseline_points))
    std_val = float(np.std(baseline_points))

    # Evitar std = 0
    if std_val < 1e-6:
        std_val = 0.01 * abs(mean_val) if abs(mean_val) > 0 else 0.01

    return {
        "mean": mean_val,
        "std": std_val,
        "threshold_3sigma": mean_val + 3.0 * std_val
    }


def detect_main_peak(t, y, min_prominence_pct=5.0, is_bp=None):
    """Detecta el pic principal en el senyal amb límits d'integració correctes."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 10 or len(y) < 10:
        return {"valid": False}

    y_max = float(np.nanmax(y))
    if y_max < 1e-6:
        return {"valid": False}

    # Detectar si és BP automàticament si no s'especifica
    if is_bp is None:
        t_max_chromato = float(np.max(t))
        is_bp = t_max_chromato < 20  # Cromatograma curt = BP

    min_prominence = y_max * (min_prominence_pct / 100.0)
    peaks, props = find_peaks(y, prominence=min_prominence, width=3)

    if len(peaks) == 0:
        return {"valid": False}

    idx = int(np.argmax(props["prominences"]))
    main_peak = int(peaks[idx])

    # Calcular baseline segons mode
    n = len(y)
    if is_bp:
        # BP: baseline dels últims punts (després del pic)
        n_edge = max(20, n // 5)
        baseline_level = np.median(y[-n_edge:])
    else:
        n_edge = max(10, n // 10)
        baseline_level = min(np.median(y[:n_edge]), np.median(y[-n_edge:]))

    # Usar find_peak_boundaries per límits correctes
    left_idx, right_idx = find_peak_boundaries(t, y, main_peak, baseline_level, threshold_pct=5.0, is_bp=is_bp)

    # Assegurar rang vàlid
    left_idx = max(0, left_idx)
    right_idx = min(len(y) - 1, right_idx)

    if right_idx > left_idx:
        area = float(trapezoid(y[left_idx:right_idx + 1], t[left_idx:right_idx + 1]))
    else:
        area = 0.0

    return {
        "valid": True,
        "area": area,
        "t_start": float(t[left_idx]),
        "t_max": float(t[main_peak]),
        "t_end": float(t[right_idx]),
        "height": float(y[main_peak]),
        "prominence": float(props["prominences"][idx]),
        "left_idx": left_idx,
        "right_idx": right_idx,
        "peak_idx": main_peak,
        "is_bp": is_bp,
        "baseline_level": float(baseline_level),
    }


def detect_batman(t, y, config):
    """[DEPRECATED] Usar hpsec_replica.evaluate_replica() en lloc d'això."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.size < 10 or y.size < 10:
        return None

    max_sep = config.get("batman_max_sep_min", 0.5)
    min_height_pct = config.get("batman_min_height_pct", 15.0)
    min_sigma = config.get("batman_min_sigma", 3.0)

    bl_stats = baseline_stats(y)
    threshold_sigma = bl_stats["mean"] + min_sigma * bl_stats["std"]

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    signal_range = y_max - y_min
    if signal_range < 1e-6:
        return None

    min_height = y_min + (min_height_pct / 100.0) * signal_range

    peaks, _ = find_peaks(y, prominence=0.05, distance=15)
    if len(peaks) < 2:
        return None

    best = None
    best_drop = -np.inf

    for i in range(len(peaks) - 1):
        left, right = int(peaks[i]), int(peaks[i + 1])
        if right <= left + 2:
            continue

        sep_min = float(t[right] - t[left])
        if sep_min > max_sep:
            continue

        avg_peak = float((y[left] + y[right]) / 2.0)
        if avg_peak < min_height or y[left] < threshold_sigma or y[right] < threshold_sigma:
            continue

        seg = y[left:right + 1]
        idx_v = left + int(np.argmin(seg))
        val_v = float(y[idx_v])

        if avg_peak <= 0:
            continue

        drop_pct = float((avg_peak - val_v) / avg_peak)
        if drop_pct < 0.02 or drop_pct > 0.35:
            continue

        if drop_pct > best_drop:
            best_drop = drop_pct
            best = {
                "drop_pct": drop_pct,
                "t_left": float(t[left]),
                "t_right": float(t[right]),
                "t_valley": float(t[idx_v]),
            }

    return best


def detect_timeout(t, y, config, is_bp=False):
    """[DEPRECATED] Usar hpsec_replica.evaluate_replica() - mètode dt intervals."""
    if len(t) < 20:
        return None

    w_start, w_end = (0.0, 10.0) if is_bp else (15.0, 35.0)
    mask = (t >= w_start) & (t <= w_end)
    if np.sum(mask) < 10:
        return None

    t_w, y_w = t[mask], y[mask]

    baseline = float(np.percentile(y_w, 10))
    peak_val = float(np.max(y_w))
    peak_height = peak_val - baseline
    if peak_height <= 0:
        return None

    min_height_frac = config.get("timeout_min_height_frac", 0.30)
    min_plat_height = baseline + min_height_frac * peak_height

    fine_res = 0.1
    num_steps = int((t_w[-1] - t_w[0]) / fine_res)
    chunks = []

    for i in range(num_steps):
        t_center = t_w[0] + i * fine_res
        mask_c = (t_w >= t_center) & (t_w < t_center + fine_res)
        if np.sum(mask_c) < 2:
            continue
        t_chunk, y_chunk = t_w[mask_c], y_w[mask_c]
        if len(t_chunk) >= 2:
            slope = abs(np.polyfit(t_chunk, y_chunk, 1)[0])
            chunks.append({"slope": slope, "t_start": float(t_chunk[0]), "t_end": float(t_chunk[-1])})

    if not chunks:
        return None

    all_slopes = [c["slope"] for c in chunks]
    thresh = np.percentile(all_slopes, 30)

    islands = []
    current = []
    for c in chunks:
        if c["slope"] <= thresh:
            current.append(c)
        else:
            if current:
                islands.append(current)
                current = []
    if current:
        islands.append(current)

    for isl in islands:
        t_s, t_e = isl[0]["t_start"], isl[-1]["t_end"]
        dur = t_e - t_s
        if not (0.8 <= dur <= 3.0):
            continue

        seg_mask = (t_w >= t_s) & (t_w <= t_e)
        y_mean = float(np.mean(y_w[seg_mask]))

        if y_mean < min_plat_height:
            continue

        return {"t_start": t_s, "t_end": t_e, "dur": dur, "y_mean": y_mean}

    return None


def detect_ears(t, y, config):
    """[DEPRECATED] No funciona bé - eliminat de hpsec_replica."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.size < 10:
        return []

    i_peak = int(np.argmax(y))
    peak_val = float(y[i_peak])
    baseline = float(np.percentile(y, 10))
    peak_height = peak_val - baseline

    if peak_height <= 0:
        return []

    min_frac = config.get("ear_min_height_frac", 0.05)
    min_level = baseline + min_frac * peak_height

    dy = np.diff(y)
    ears = []

    # Pujada
    i = 0
    while i < i_peak:
        if y[i] < min_level:
            i += 1
            continue
        if dy[i] < 0:
            start_idx = i
            while i < i_peak and dy[i] < 0:
                i += 1
            ears.append({"side": "pujada", "t_start": float(t[start_idx]), "t_end": float(t[i])})
        else:
            i += 1

    # Baixada
    i = i_peak
    while i < len(dy):
        if y[i] < min_level:
            i += 1
            continue
        if dy[i] > 0:
            start_idx = i
            while i < len(dy) and dy[i] > 0:
                i += 1
            ears.append({"side": "baixada", "t_start": float(t[start_idx]), "t_end": float(t[i])})
        else:
            i += 1

    return ears


def calcular_kpis(t, y, config):
    """Calcula KPIs per fraccions de temps."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    kpis = {"total": float(trapezoid(y, t))}

    fractions = config.get("time_fractions", DEFAULT_CONFIG["time_fractions"])
    for nom, (t_ini, t_fi) in fractions.items():
        mask = (t >= t_ini) & (t < t_fi)
        if np.sum(mask) > 1:
            kpis[nom] = float(trapezoid(y[mask], t[mask]))
        else:
            kpis[nom] = 0.0

    if kpis["total"] > 0:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 100.0 * kpis[nom] / kpis["total"]

    return kpis


def netejar_baseline(y):
    """Neteja la baseline del senyal."""
    if len(y) == 0:
        return y
    baseline = np.percentile(y, 1)
    y_clean = y - baseline
    y_clean[y_clean < 0] = 0
    return y_clean


def calcular_roughness(y):
    """Calcula roughness normalitzat."""
    if len(y) < 2:
        return 999999
    y_clean = netejar_baseline(y)
    area = np.sum(y_clean)
    if area < 1e-3:
        return 999999
    return np.sum(np.abs(np.diff(y_clean))) / area


# =============================================================================
# DIÀLEG D'HISTÒRIC DE CALIBRACIÓ KHP
# =============================================================================
class KHPHistoryDialog:
    """
    Diàleg per visualitzar i gestionar l'històric de calibracions KHP.
    Permet:
    - Veure totes les calibracions (COLUMN i BP per separat)
    - Seleccionar un KHP anterior per aplicar a la seqüència actual
    - Escananejar carpetes si l'històric no existeix
    - Exportar l'històric a Excel
    """

    def __init__(self, parent, seq_path, apply_callback=None):
        self.parent = parent
        self.seq_path = seq_path
        self.apply_callback = apply_callback
        self.selected_entry = None

        self.window = tk.Toplevel(parent)
        self.window.title("Històric de Calibració KHP")
        self.window.geometry("900x600")
        self.window.transient(parent)
        self.window.grab_set()

        self._build_gui()
        self._load_history()

    def _build_gui(self):
        """Construeix la interfície del diàleg."""
        # Header
        header = tk.Frame(self.window, bg=COLORS["secondary"], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="Històric de Calibració KHP",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["secondary"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=20, pady=10)

        # Frame principal
        main_frame = tk.Frame(self.window, bg=COLORS["white"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Filtres
        filter_frame = tk.Frame(main_frame, bg=COLORS["white"])
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        # Filtre Mode (COLUMN/BP)
        tk.Label(filter_frame, text="Mode:",
                font=("Segoe UI", 10),
                bg=COLORS["white"]).pack(side=tk.LEFT, padx=(0, 5))

        self.mode_var = tk.StringVar(value="TOTS")
        modes = ["TOTS", "COLUMN", "BP"]
        for mode in modes:
            rb = tk.Radiobutton(filter_frame, text=mode, variable=self.mode_var, value=mode,
                               font=("Segoe UI", 10), bg=COLORS["white"],
                               command=self._filter_changed)
            rb.pack(side=tk.LEFT, padx=5)

        # Separador
        tk.Label(filter_frame, text="  |  ",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["secondary"]).pack(side=tk.LEFT)

        # Filtre Source (UIB/DIRECT)
        tk.Label(filter_frame, text="Source:",
                font=("Segoe UI", 10),
                bg=COLORS["white"]).pack(side=tk.LEFT, padx=(0, 5))

        self.source_var = tk.StringVar(value="TOTS")
        sources = ["TOTS", "UIB", "DIRECT"]
        for source in sources:
            rb = tk.Radiobutton(filter_frame, text=source, variable=self.source_var, value=source,
                               font=("Segoe UI", 10), bg=COLORS["white"],
                               command=self._filter_changed)
            rb.pack(side=tk.LEFT, padx=5)

        # Treeview per l'històric
        cols = ["Seqüència", "Data", "Mode", "Source", "Conc", "Vol", "t_max", "Àrea", "Factor", "SNR", "Sim", "SHIFT", "Estat"]
        self.tree = ttk.Treeview(main_frame, columns=cols, show='headings', height=15)

        widths = {"Seqüència": 120, "Data": 75, "Mode": 50, "Source": 50, "Conc": 40, "Vol": 35,
                  "t_max": 45, "Àrea": 55, "Factor": 75, "SNR": 40, "Sim": 40, "SHIFT": 45, "Estat": 55}

        for col in cols:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=widths.get(col, 80), anchor="center")

        # Scrollbars
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(main_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags per colors
        self.tree.tag_configure('OK', background='#c6efce', foreground='#006100')
        self.tree.tag_configure('WARN', background='#fff3cd', foreground='#856404')
        self.tree.tag_configure('FAIL', background='#f8d7da', foreground='#721c24')
        self.tree.tag_configure('HISTORIC', background='#cce5ff', foreground='#004085')
        self.tree.tag_configure('OUTLIER', background='#e0e0e0', foreground='#666666')

        # Frame inferior amb botons
        btn_frame = tk.Frame(self.window, bg=COLORS["light"], height=60)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        btn_frame.pack_propagate(False)

        # Botons esquerra
        left_btns = tk.Frame(btn_frame, bg=COLORS["light"])
        left_btns.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(left_btns, text="Escanejar Carpetes",
                 command=self._scan_folders,
                 bg=COLORS["warning"], fg=COLORS["dark"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(left_btns, text="Exportar a Excel",
                 command=self._export_to_excel,
                 bg=COLORS["success"], fg=COLORS["white"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        self.btn_outlier = tk.Button(left_btns, text="Marcar Outlier",
                 command=self._toggle_outlier,
                 bg="#999999", fg=COLORS["white"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=15, pady=5, state="disabled")
        self.btn_outlier.pack(side=tk.LEFT, padx=5)

        # Botons dreta
        right_btns = tk.Frame(btn_frame, bg=COLORS["light"])
        right_btns.pack(side=tk.RIGHT, padx=10, pady=10)

        self.btn_apply = tk.Button(right_btns, text="Aplicar Seleccionat",
                                   command=self._apply_selected,
                                   bg=COLORS["primary"], fg=COLORS["white"],
                                   font=("Segoe UI", 10, "bold"),
                                   relief="flat", padx=20, pady=5,
                                   state="disabled")
        self.btn_apply.pack(side=tk.LEFT, padx=5)

        tk.Button(right_btns, text="Tancar",
                 command=self.window.destroy,
                 bg=COLORS["dark"], fg=COLORS["white"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=20, pady=5).pack(side=tk.LEFT, padx=5)

        # Bind selecció
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click)

        # Info label
        self.lbl_info = tk.Label(btn_frame, text="Selecciona una entrada per aplicar-la",
                                font=("Segoe UI", 9, "italic"),
                                bg=COLORS["light"], fg=COLORS["dark"])
        self.lbl_info.pack(side=tk.LEFT, padx=20)

    def _load_history(self):
        """Carrega l'històric i l'omple a la taula."""
        self.calibrations = load_khp_history(self.seq_path)

        if not self.calibrations:
            self.lbl_info.configure(text="No hi ha històric. Usa 'Escanejar Carpetes' per generar-lo.")
            return

        self._populate_table()

    def _populate_table(self):
        """Omple la taula amb les calibracions filtrades."""
        # Netejar
        for item in self.tree.get_children():
            self.tree.delete(item)

        mode_filter = self.mode_var.get()
        source_filter = self.source_var.get()

        for cal in self.calibrations:
            # Filtrar per mode (COLUMN/BP)
            if mode_filter != "TOTS" and cal.get("mode") != mode_filter:
                continue

            # Filtrar per source (UIB/DIRECT)
            if source_filter != "TOTS":
                cal_source = cal.get("doc_mode", "N/A")
                if cal_source != source_filter:
                    continue

            # Calcular factor si no existeix
            area = cal.get('area', 0)
            conc = cal.get('conc_ppm', 0)
            factor = cal.get('factor', conc / area if area > 0 else 0)

            # Obtenir nombre de pics
            n_pics = cal.get('all_peaks_count', cal.get('n_peaks', 1))

            # Estat final: considerar outlier
            is_outlier = cal.get("is_outlier", False)
            if is_outlier:
                display_status = "OUTLIER"
            else:
                display_status = cal.get("status", "OK")

            volume = cal.get("volume_uL", cal.get("volume", 0))
            t_retention = cal.get("t_retention", cal.get("t_doc_max", 0))

            values = (
                cal.get("seq_name", ""),
                cal.get("date", "")[:10] if cal.get("date") else "",
                cal.get("mode", ""),
                cal.get("doc_mode", "N/A"),  # SOURCE DOC: UIB o DIRECT
                conc,
                volume,
                f"{t_retention:.1f}",  # TEMPS RETENCIÓ
                f"{area:.1f}",
                f"{factor:.6f}",
                f"{cal.get('snr', 0):.1f}",
                f"{cal.get('symmetry', 1.0):.2f}",
                f"{cal.get('shift_sec', 0):.1f}",
                display_status,
            )

            # Determinar tag
            if is_outlier:
                tag = "OUTLIER"
            elif display_status == "FAIL":
                tag = "FAIL"
            elif display_status == "HISTORIC":
                tag = "HISTORIC"
            elif cal.get("quality_issues"):
                tag = "WARN"
            else:
                tag = "OK"

            self.tree.insert("", tk.END, values=values, tags=(tag,), iid=cal.get("seq_name", ""))

        # Actualitzar info amb filtres actius
        total = len(self.tree.get_children())
        total_all = len(self.calibrations) if self.calibrations else 0
        filter_parts = []
        if mode_filter != "TOTS":
            filter_parts.append(mode_filter)
        if source_filter != "TOTS":
            filter_parts.append(source_filter)
        filter_str = f" [{'/'.join(filter_parts)}]" if filter_parts else ""
        self.lbl_info.configure(text=f"{total} calibracions{filter_str} (de {total_all} total)")

    def _filter_changed(self):
        """Re-filtra quan canvia el mode."""
        self._populate_table()

    def _sort_by(self, col):
        """Ordena la taula per una columna."""
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]

        # Intentar ordenar numèricament si és possible
        try:
            items.sort(key=lambda t: float(t[0].replace(",", ".")), reverse=True)
        except ValueError:
            items.sort(reverse=True)

        for index, (_, k) in enumerate(items):
            self.tree.move(k, "", index)

    def _on_select(self, event):
        """Gestiona la selecció d'una fila."""
        selection = self.tree.selection()
        if selection:
            self.btn_apply.configure(state="normal")
            self.btn_outlier.configure(state="normal")
            seq_name = selection[0]
            # Trobar l'entrada
            for cal in self.calibrations:
                if cal.get("seq_name") == seq_name:
                    self.selected_entry = cal
                    # Actualitzar text del botó segons estat actual
                    if cal.get("is_outlier", False):
                        self.btn_outlier.configure(text="Desmarcar Outlier", bg=COLORS["success"])
                    else:
                        self.btn_outlier.configure(text="Marcar Outlier", bg="#999999")
                    break
        else:
            self.btn_apply.configure(state="disabled")
            self.btn_outlier.configure(state="disabled")
            self.selected_entry = None

    def _on_double_click(self, event):
        """Doble clic per aplicar directament."""
        if self.selected_entry and self.apply_callback:
            self._apply_selected()

    def _toggle_outlier(self):
        """Marca o desmarca la calibració seleccionada com a outlier."""
        if not self.selected_entry:
            return

        seq_name = self.selected_entry.get("seq_name", "")
        mode = self.selected_entry.get("mode", "COLUMN")
        is_outlier = self.selected_entry.get("is_outlier", False)

        # Invertir l'estat
        new_outlier_state = not is_outlier

        action = "marcar com a OUTLIER" if new_outlier_state else "desmarcar com a outlier"
        if not messagebox.askyesno("Confirmar",
                                   f"Vols {action} la calibració:\n\n"
                                   f"SEQ: {seq_name}\n"
                                   f"Mode: {mode}\n"
                                   f"Àrea: {self.selected_entry.get('area', 0):.2f}"):
            return

        # Actualitzar a l'històric
        if mark_calibration_as_outlier(self.seq_path, seq_name, mode, new_outlier_state):
            # Recarregar dades
            self.calibrations = load_khp_history(self.seq_path)
            self._populate_table()

            status = "MARCADA com a outlier" if new_outlier_state else "DESMARCADA com a outlier"
            self.lbl_info.configure(text=f"Calibració {status}")
        else:
            messagebox.showerror("Error", "No s'ha pogut actualitzar l'estat d'outlier.")

    def _apply_selected(self):
        """Aplica la calibració seleccionada."""
        if not self.selected_entry:
            return

        if self.apply_callback:
            self.window.destroy()
            self.apply_callback(self.selected_entry)

    def _scan_folders(self):
        """Escaneja les carpetes SEQ per generar l'històric."""
        if not self.seq_path:
            messagebox.showerror("Error", "Cal seleccionar una carpeta SEQ primer.")
            return

        parent_folder = os.path.dirname(self.seq_path)

        if not messagebox.askyesno("Escanejar Carpetes",
                                  f"S'escanejaran totes les carpetes SEQ a:\n{parent_folder}\n\n"
                                  f"Això pot trigar uns segons. Continuar?"):
            return

        # Crear diàleg de progrés
        progress_win = tk.Toplevel(self.window)
        progress_win.title("Escanejant...")
        progress_win.geometry("400x100")
        progress_win.transient(self.window)

        tk.Label(progress_win, text="Escanejant carpetes SEQ...",
                font=("Segoe UI", 10)).pack(pady=10)

        progress_bar = ttk.Progressbar(progress_win, mode='determinate', length=300)
        progress_bar.pack(pady=10)

        lbl_status = tk.Label(progress_win, text="")
        lbl_status.pack()

        def update_progress(pct, folder):
            progress_bar['value'] = pct
            lbl_status.configure(text=os.path.basename(folder))
            progress_win.update()

        # Escanejar
        try:
            new_calibrations = scan_seq_folders_for_khp(parent_folder, update_progress)

            if new_calibrations:
                # Guardar
                save_khp_history(self.seq_path, new_calibrations)
                self.calibrations = new_calibrations
                self._populate_table()

                messagebox.showinfo("Escaneig Completat",
                                   f"S'han trobat {len(new_calibrations)} calibracions.")
            else:
                messagebox.showwarning("Escaneig Completat",
                                      "No s'ha trobat cap calibració KHP a les carpetes.")

        except Exception as e:
            messagebox.showerror("Error", f"Error durant l'escaneig:\n{str(e)}")

        finally:
            progress_win.destroy()

    def _export_to_excel(self):
        """Exporta l'històric a Excel."""
        if not self.calibrations:
            messagebox.showwarning("Avís", "No hi ha dades per exportar.")
            return

        excel_path = export_history_to_excel(self.seq_path)

        if excel_path:
            messagebox.showinfo("Exportació",
                               f"Històric exportat a:\n{excel_path}")

            # Preguntar si obrir
            if messagebox.askyesno("Obrir Excel?", "Vols obrir el fitxer Excel?"):
                try:
                    os.startfile(excel_path)
                except:
                    pass
        else:
            messagebox.showerror("Error", "No s'ha pogut exportar l'històric.")


# =============================================================================
# FINESTRA D'HISTÒRIC DE MOSTRES
# =============================================================================
class SamplesHistoryWindow:
    """Finestra per visualitzar i filtrar l'històric de mostres."""

    def __init__(self, parent, seq_path=None):
        self.parent = parent
        self.seq_path = seq_path
        self.all_samples = []
        self.filtered_samples = []

        self.window = tk.Toplevel(parent)
        self.window.title("Històric de Mostres")
        self.window.geometry("1100x650")
        self.window.transient(parent)

        self._build_gui()
        self._load_samples()

    def _build_gui(self):
        """Construeix la interfície."""
        # Header
        header = tk.Frame(self.window, bg=COLORS["primary"], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="Històric de Mostres - Àrees DOC i DAD",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["primary"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=20, pady=10)

        # Frame principal
        main_frame = tk.Frame(self.window, bg=COLORS["white"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Filtres (dues línies)
        filter_frame = tk.Frame(main_frame, bg=COLORS["white"])
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        # Línia 1: Mode i DOC Mode
        filter_line1 = tk.Frame(filter_frame, bg=COLORS["white"])
        filter_line1.pack(fill=tk.X, pady=2)

        tk.Label(filter_line1, text="Mode:",
                font=("Segoe UI", 10, "bold"),
                bg=COLORS["white"]).pack(side=tk.LEFT, padx=(0, 5))

        self.mode_var = tk.StringVar(value="TOTS")
        for mode in ["TOTS", "COLUMN", "BP"]:
            rb = tk.Radiobutton(filter_line1, text=mode, variable=self.mode_var, value=mode,
                               font=("Segoe UI", 10), bg=COLORS["white"],
                               command=self._apply_filters)
            rb.pack(side=tk.LEFT, padx=3)

        tk.Label(filter_line1, text="  |  DOC Mode:",
                font=("Segoe UI", 10, "bold"),
                bg=COLORS["white"]).pack(side=tk.LEFT, padx=(10, 5))

        self.doc_mode_var = tk.StringVar(value="TOTS")
        for dm in ["TOTS", "UIB", "DIRECT"]:
            rb = tk.Radiobutton(filter_line1, text=dm, variable=self.doc_mode_var, value=dm,
                               font=("Segoe UI", 10), bg=COLORS["white"],
                               command=self._apply_filters)
            rb.pack(side=tk.LEFT, padx=3)

        # Checkbox KHP
        tk.Label(filter_line1, text="  |  ",
                bg=COLORS["white"]).pack(side=tk.LEFT)
        self.exclude_khp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(filter_line1, text="Excloure KHP", variable=self.exclude_khp_var,
                      font=("Segoe UI", 10), bg=COLORS["white"],
                      command=self._apply_filters).pack(side=tk.LEFT, padx=5)

        # Línia 2: Filtre de text
        filter_line2 = tk.Frame(filter_frame, bg=COLORS["white"])
        filter_line2.pack(fill=tk.X, pady=5)

        tk.Label(filter_line2, text="Filtrar mostra:",
                font=("Segoe UI", 10, "bold"),
                bg=COLORS["white"]).pack(side=tk.LEFT, padx=(0, 5))

        self.filter_text_var = tk.StringVar()
        self.filter_text_var.trace('w', lambda *args: self._apply_filters())
        filter_entry = tk.Entry(filter_line2, textvariable=self.filter_text_var,
                               font=("Segoe UI", 10), width=25)
        filter_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(filter_line2, text="Netejar", command=self._clear_filters,
                 font=("Segoe UI", 9), relief="flat",
                 bg=COLORS["light"]).pack(side=tk.LEFT, padx=5)

        # Info count
        self.lbl_count = tk.Label(filter_line2, text="",
                                 font=("Segoe UI", 10, "italic"),
                                 bg=COLORS["white"], fg=COLORS["secondary"])
        self.lbl_count.pack(side=tk.RIGHT, padx=10)

        # Treeview
        cols = ["SEQ", "Data", "Mostra", "Rep", "Mode", "DOC", "DOC_Area",
                "A220", "A252", "A254", "A272", "A290", "A362"]
        self.tree = ttk.Treeview(main_frame, columns=cols, show='headings', height=18)

        widths = {"SEQ": 100, "Data": 85, "Mostra": 120, "Rep": 35, "Mode": 50, "DOC": 45,
                  "DOC_Area": 70, "A220": 60, "A252": 60, "A254": 60, "A272": 60, "A290": 60, "A362": 60}

        for col in cols:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=widths.get(col, 60), anchor="center")

        # Scrollbars
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(main_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Pack treeview amb scrollbars
        tree_frame = tk.Frame(main_frame, bg=COLORS["white"])
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags per colors
        self.tree.tag_configure('KHP', background='#d4edda', foreground='#155724')
        self.tree.tag_configure('NORMAL', background='white')
        self.tree.tag_configure('ALT', background='#f8f9fa')

        # Frame inferior amb botons
        btn_frame = tk.Frame(self.window, bg=COLORS["light"], height=60)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        btn_frame.pack_propagate(False)

        # Botons
        left_btns = tk.Frame(btn_frame, bg=COLORS["light"])
        left_btns.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(left_btns, text="Recarregar",
                 command=self._load_samples,
                 bg=COLORS["warning"], fg=COLORS["dark"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(left_btns, text="Exportar a Excel",
                 command=self._export_to_excel,
                 bg=COLORS["success"], fg=COLORS["white"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        # Estadístiques
        self.lbl_stats = tk.Label(btn_frame, text="",
                                 font=("Segoe UI", 9),
                                 bg=COLORS["light"], fg=COLORS["dark"])
        self.lbl_stats.pack(side=tk.LEFT, padx=20)

        # Botó tancar
        tk.Button(btn_frame, text="Tancar",
                 command=self.window.destroy,
                 bg=COLORS["dark"], fg=COLORS["white"],
                 font=("Segoe UI", 10),
                 relief="flat", padx=20, pady=5).pack(side=tk.RIGHT, padx=10, pady=10)

        # Bind doble clic per mostrar detalls
        self.tree.bind("<Double-1>", self._on_double_click)

        # Sorting state
        self.sort_column = None
        self.sort_reverse = False

    def _load_samples(self):
        """Carrega les mostres des del JSON."""
        self.all_samples = []

        # Buscar el JSON
        if self.seq_path:
            json_path = get_samples_history_path(self.seq_path)
        else:
            # Buscar al directori actual o parent
            json_path = os.path.join(os.getcwd(), SAMPLES_HISTORY_FILENAME)

        if json_path and os.path.exists(json_path):
            samples = load_samples_history_from_path(json_path)
            if samples:
                self.all_samples = samples

        self._apply_filters()

    def _apply_filters(self):
        """Aplica els filtres a les dades."""
        mode_filter = self.mode_var.get()
        doc_mode_filter = self.doc_mode_var.get()
        text_filter = self.filter_text_var.get().strip().lower()
        exclude_khp = self.exclude_khp_var.get()

        filtered = []
        for s in self.all_samples:
            # Filtre mode
            if mode_filter != "TOTS" and s.get('mode', '') != mode_filter:
                continue
            # Filtre DOC mode
            if doc_mode_filter != "TOTS" and s.get('doc_mode', '') != doc_mode_filter:
                continue
            # Filtre text
            if text_filter and text_filter not in s.get('mostra', '').lower():
                continue
            # Filtre KHP
            if exclude_khp and s.get('is_khp', False):
                continue
            filtered.append(s)

        self.filtered_samples = filtered
        self._populate_tree()
        self._update_stats()

    def _populate_tree(self):
        """Omple el treeview amb les dades filtrades."""
        # Esborrar existents
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Afegir mostres
        for i, s in enumerate(self.filtered_samples):
            tag = 'KHP' if s.get('is_khp', False) else ('ALT' if i % 2 else 'NORMAL')

            values = (
                s.get('seq_name', '-'),
                s.get('date', '-')[:10] if s.get('date') else '-',
                s.get('mostra', '-'),
                s.get('replica', '-'),
                s.get('mode', '-'),
                s.get('doc_mode', '-'),
                f"{s.get('doc_area', 0):.1f}",
                f"{s.get('a220_area', 0):.1f}" if s.get('a220_area', 0) > 0 else "-",
                f"{s.get('a252_area', 0):.1f}" if s.get('a252_area', 0) > 0 else "-",
                f"{s.get('a254_area', 0):.1f}" if s.get('a254_area', 0) > 0 else "-",
                f"{s.get('a272_area', 0):.1f}" if s.get('a272_area', 0) > 0 else "-",
                f"{s.get('a290_area', 0):.1f}" if s.get('a290_area', 0) > 0 else "-",
                f"{s.get('a362_area', 0):.1f}" if s.get('a362_area', 0) > 0 else "-",
            )
            self.tree.insert('', 'end', values=values, tags=(tag,))

        self.lbl_count.configure(text=f"Mostrant {len(self.filtered_samples)} de {len(self.all_samples)} mostres")

    def _update_stats(self):
        """Actualitza estadístiques."""
        if not self.filtered_samples:
            self.lbl_stats.configure(text="")
            return

        doc_areas = [s.get('doc_area', 0) for s in self.filtered_samples if s.get('doc_area', 0) > 0]
        if doc_areas:
            mean_doc = np.mean(doc_areas)
            std_doc = np.std(doc_areas) if len(doc_areas) > 1 else 0
            self.lbl_stats.configure(text=f"DOC: mitjana={mean_doc:.1f}, std={std_doc:.1f}, n={len(doc_areas)}")
        else:
            self.lbl_stats.configure(text="")

    def _sort_by(self, col):
        """Ordena per columna."""
        if self.sort_column == col:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = col
            self.sort_reverse = False

        # Mapeig columna a clau
        col_map = {
            "SEQ": "seq_name", "Data": "date", "Mostra": "mostra", "Rep": "replica",
            "Mode": "mode", "DOC": "doc_mode", "DOC_Area": "doc_area",
            "A220": "a220_area", "A252": "a252_area", "A254": "a254_area",
            "A272": "a272_area", "A290": "a290_area", "A362": "a362_area"
        }

        key = col_map.get(col, col.lower())

        try:
            self.filtered_samples.sort(key=lambda x: x.get(key, 0) or 0, reverse=self.sort_reverse)
        except:
            pass

        self._populate_tree()

    def _clear_filters(self):
        """Neteja els filtres."""
        self.mode_var.set("TOTS")
        self.doc_mode_var.set("TOTS")
        self.filter_text_var.set("")
        self.exclude_khp_var.set(True)
        self._apply_filters()

    def _on_double_click(self, event):
        """Doble clic per mostrar detalls."""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        values = item['values']

        # Construir missatge amb detalls
        msg = f"Mostra: {values[2]}\n"
        msg += f"Rèplica: {values[3]}\n"
        msg += f"SEQ: {values[0]}\n"
        msg += f"Data: {values[1]}\n"
        msg += f"Mode: {values[4]} / {values[5]}\n\n"
        msg += f"DOC Area: {values[6]}\n\n"
        msg += "Àrees DAD:\n"
        msg += f"  A220: {values[7]}\n"
        msg += f"  A252: {values[8]}\n"
        msg += f"  A254: {values[9]}\n"
        msg += f"  A272: {values[10]}\n"
        msg += f"  A290: {values[11]}\n"
        msg += f"  A362: {values[12]}\n"

        messagebox.showinfo("Detalls Mostra", msg)

    def _export_to_excel(self):
        """Exporta les dades filtrades a Excel."""
        if not self.filtered_samples:
            messagebox.showwarning("Avís", "No hi ha dades per exportar.")
            return

        # Demanar path
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="Samples_History_Export.xlsx"
        )

        if not filepath:
            return

        try:
            # Crear DataFrame
            rows = []
            for s in self.filtered_samples:
                rows.append({
                    'SEQ': s.get('seq_name', ''),
                    'Data': s.get('date', ''),
                    'Mostra': s.get('mostra', ''),
                    'Rèplica': s.get('replica', ''),
                    'Mode': s.get('mode', ''),
                    'DOC_Mode': s.get('doc_mode', ''),
                    'Is_KHP': s.get('is_khp', False),
                    'DOC_Area': s.get('doc_area', 0),
                    'DOC_t_retention': s.get('doc_t_retention', 0),
                    'DOC_t_start': s.get('doc_t_start', 0),
                    'DOC_t_end': s.get('doc_t_end', 0),
                    'A220_Area': s.get('a220_area', 0),
                    'A252_Area': s.get('a252_area', 0),
                    'A254_Area': s.get('a254_area', 0),
                    'A272_Area': s.get('a272_area', 0),
                    'A290_Area': s.get('a290_area', 0),
                    'A362_Area': s.get('a362_area', 0),
                })

            df = pd.DataFrame(rows)
            df.to_excel(filepath, index=False)

            messagebox.showinfo("Exportació", f"Exportat a:\n{filepath}")

            if messagebox.askyesno("Obrir?", "Vols obrir el fitxer?"):
                try:
                    os.startfile(filepath)
                except:
                    pass

        except Exception as e:
            messagebox.showerror("Error", f"Error exportant: {e}")


def load_samples_history_from_path(json_path):
    """Carrega històric de mostres des d'un path específic."""
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('samples', [])
    except:
        pass
    return []


# =============================================================================
# CLASSE PRINCIPAL DE L'APLICACIÓ
# =============================================================================
class HPSECSuite:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # Estat de l'aplicació
        self.config = DEFAULT_CONFIG.copy()
        self.seq_path = None
        self.current_step = 0
        self.steps = ["Consolidar", "Calibrar", "Processar", "Exportar"]

        # Dades
        self.consolidated_data = {}
        self.calibration_data = {}
        self.processed_data = {}
        self.selected_replicas = {}

        # Threading
        self.is_processing = False

        self._setup_styles()
        self._build_gui()

    def _setup_styles(self):
        """Configura estils TTK."""
        style = ttk.Style()
        style.theme_use('clam')

        # Botons
        style.configure("Primary.TButton",
                       background=COLORS["primary"],
                       foreground=COLORS["white"],
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))

        style.configure("Success.TButton",
                       background=COLORS["success"],
                       foreground=COLORS["white"],
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))

        style.configure("Step.TLabel",
                       font=("Segoe UI", 11),
                       padding=10)

        style.configure("StepActive.TLabel",
                       font=("Segoe UI", 11, "bold"),
                       background=COLORS["primary"],
                       foreground=COLORS["white"],
                       padding=10)

        style.configure("Title.TLabel",
                       font=("Segoe UI", 14, "bold"),
                       foreground=COLORS["dark"])

        style.configure("Info.TLabel",
                       font=("Segoe UI", 10),
                       foreground=COLORS["dark"])

    def _build_gui(self):
        """Construeix la interfície gràfica."""
        # Header
        self._build_header()

        # Steps indicator
        self._build_steps_indicator()

        # Main content area
        self._build_main_area()

        # Footer with navigation
        self._build_footer()

        # Mostrar primer pas
        self._show_step(0)

    def _build_header(self):
        """Construeix la capçalera."""
        header = tk.Frame(self.root, bg=COLORS["primary"], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text=f"{APP_NAME}",
                font=("Segoe UI", 16, "bold"),
                bg=COLORS["primary"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=20, pady=15)

        tk.Label(header, text=f"v{VERSION}",
                font=("Segoe UI", 10),
                bg=COLORS["primary"], fg=COLORS["light"]).pack(side=tk.LEFT, pady=15)

        # Selector de carpeta
        self.path_var = tk.StringVar(value="Cap carpeta seleccionada")

        btn_folder = tk.Button(header, text="Seleccionar SEQ",
                              command=self._select_folder,
                              bg=COLORS["white"], fg=COLORS["primary"],
                              font=("Segoe UI", 9, "bold"),
                              relief="flat", padx=15, pady=5)
        btn_folder.pack(side=tk.RIGHT, padx=20, pady=15)

        # Botó Històric Mostres
        btn_history = tk.Button(header, text="Històric Mostres",
                               command=self._open_samples_history,
                               bg=COLORS["success"], fg=COLORS["white"],
                               font=("Segoe UI", 9, "bold"),
                               relief="flat", padx=15, pady=5)
        btn_history.pack(side=tk.RIGHT, padx=5, pady=15)

        tk.Label(header, textvariable=self.path_var,
                font=("Segoe UI", 9),
                bg=COLORS["primary"], fg=COLORS["light"]).pack(side=tk.RIGHT, pady=15)

    def _build_steps_indicator(self):
        """Construeix l'indicador de passos."""
        steps_frame = tk.Frame(self.root, bg=COLORS["light"], height=50)
        steps_frame.pack(fill=tk.X)
        steps_frame.pack_propagate(False)

        self.step_labels = []

        for i, step in enumerate(self.steps):
            frame = tk.Frame(steps_frame, bg=COLORS["light"])
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

            # Número i nom del pas
            text = f"{i+1}. {step}"
            lbl = tk.Label(frame, text=text,
                          font=("Segoe UI", 11),
                          bg=COLORS["light"], fg=COLORS["dark"],
                          cursor="hand2")
            lbl.pack(expand=True)
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_step_click(idx))

            self.step_labels.append(lbl)

            # Separador
            if i < len(self.steps) - 1:
                sep = tk.Label(steps_frame, text="→",
                              font=("Segoe UI", 14),
                              bg=COLORS["light"], fg=COLORS["dark"])
                sep.pack(side=tk.LEFT, padx=5)

    def _build_main_area(self):
        """Construeix l'àrea principal de contingut."""
        self.main_frame = tk.Frame(self.root, bg=COLORS["white"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Frames per cada pas (es mostraran/amagaran)
        self.step_frames = []
        for i in range(4):
            frame = tk.Frame(self.main_frame, bg=COLORS["white"])
            self.step_frames.append(frame)

        # Construir contingut de cada pas
        self._build_step_consolidar(self.step_frames[0])
        self._build_step_calibrar(self.step_frames[1])
        self._build_step_processar(self.step_frames[2])
        self._build_step_exportar(self.step_frames[3])

    def _build_step_consolidar(self, parent):
        """Pas 1: Consolidació."""
        # Títol
        tk.Label(parent, text="Pas 1: Consolidació de Dades",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Llegeix els fitxers .D originals i crea fitxers Excel consolidats per cada mostra.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 20))

        # Info frame
        info_frame = tk.LabelFrame(parent, text="Informació",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=COLORS["white"])
        info_frame.pack(fill=tk.X, pady=10)

        self.lbl_con_files = tk.Label(info_frame, text="Fitxers .D detectats: -",
                                      font=("Segoe UI", 10),
                                      bg=COLORS["white"])
        self.lbl_con_files.pack(anchor="w", padx=10, pady=5)

        self.lbl_con_samples = tk.Label(info_frame, text="Mostres identificades: -",
                                        font=("Segoe UI", 10),
                                        bg=COLORS["white"])
        self.lbl_con_samples.pack(anchor="w", padx=10, pady=5)

        self.lbl_con_khp = tk.Label(info_frame, text="Mostres KHP: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_con_khp.pack(anchor="w", padx=10, pady=5)

        # Progress
        self.progress_con = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_con.pack(pady=20)

        self.lbl_con_status = tk.Label(parent, text="",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"], fg=COLORS["dark"])
        self.lbl_con_status.pack()

        # Botó
        self.btn_consolidar = tk.Button(parent, text="Consolidar Dades",
                                        command=self._run_consolidation,
                                        bg=COLORS["primary"], fg=COLORS["white"],
                                        font=("Segoe UI", 11, "bold"),
                                        relief="flat", padx=30, pady=10,
                                        state="disabled")
        self.btn_consolidar.pack(pady=20)

    def _build_step_calibrar(self, parent):
        """Pas 2: Calibració."""
        tk.Label(parent, text="Pas 2: Calibració amb KHP",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Calcula el factor de calibració utilitzant les mostres KHP.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 20))

        # Info frame
        info_frame = tk.LabelFrame(parent, text="Calibració",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=COLORS["white"])
        info_frame.pack(fill=tk.X, pady=10)

        self.lbl_cal_khp = tk.Label(info_frame, text="Mostres KHP trobades: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_cal_khp.pack(anchor="w", padx=10, pady=5)

        self.lbl_cal_factor = tk.Label(info_frame, text="Factor de calibració: -",
                                       font=("Segoe UI", 10, "bold"),
                                       bg=COLORS["white"])
        self.lbl_cal_factor.pack(anchor="w", padx=10, pady=5)

        self.lbl_cal_r2 = tk.Label(info_frame, text="R²: -",
                                   font=("Segoe UI", 10),
                                   bg=COLORS["white"])
        self.lbl_cal_r2.pack(anchor="w", padx=10, pady=5)

        # Gràfic (altura fixa per no empènyer botons)
        self.cal_canvas_frame = tk.Frame(parent, bg=COLORS["light"], height=280)
        self.cal_canvas_frame.pack(fill=tk.X, pady=10)
        self.cal_canvas_frame.pack_propagate(False)  # Mantenir altura fixa

        # Frame per botons
        btn_frame = tk.Frame(parent, bg=COLORS["white"])
        btn_frame.pack(pady=20)

        # Botó calibració
        self.btn_calibrar = tk.Button(btn_frame, text="Executar Calibració",
                                      command=self._run_calibration,
                                      bg=COLORS["primary"], fg=COLORS["white"],
                                      font=("Segoe UI", 11, "bold"),
                                      relief="flat", padx=30, pady=10,
                                      state="disabled")
        self.btn_calibrar.pack(side=tk.LEFT, padx=10)

        # Botó històric KHP
        self.btn_history = tk.Button(btn_frame, text="Històric KHP",
                                     command=self._show_khp_history,
                                     bg=COLORS["secondary"], fg=COLORS["white"],
                                     font=("Segoe UI", 11, "bold"),
                                     relief="flat", padx=30, pady=10,
                                     state="normal")
        self.btn_history.pack(side=tk.LEFT, padx=10)

        # Botó marcar outlier (apareix després de calibració)
        self.btn_outlier = tk.Button(btn_frame, text="❌ Descartar (Outlier)",
                                     command=self._mark_current_as_outlier,
                                     bg=COLORS["warning"], fg=COLORS["dark"],
                                     font=("Segoe UI", 10, "bold"),
                                     relief="flat", padx=20, pady=10)
        # No mostrar inicialment
        self.btn_outlier.pack_forget()

    def _build_step_processar(self, parent):
        """Pas 3: Processament."""
        tk.Label(parent, text="Pas 3: Processament i Selecció",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Detecta anomalies (TimeOUT, Batman, Orelletes) i selecciona les millors rèpliques.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 20))

        # Taula de resultats amb selecció independent DOC/DAD
        table_frame = tk.Frame(parent, bg=COLORS["white"])
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        cols = ["Mostra", "Estat", "DOC R", "DOC Δ%", "Sel DOC", "DAD Qual", "Sel DAD"]
        self.tree_process = ttk.Treeview(table_frame, columns=cols, show='headings', height=12)

        widths = {"Mostra": 140, "Estat": 70, "DOC R": 70, "DOC Δ%": 70,
                  "Sel DOC": 70, "DAD Qual": 90, "Sel DAD": 70}
        for col in cols:
            self.tree_process.heading(col, text=col)
            self.tree_process.column(col, width=widths.get(col, 80), anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_process.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree_process.xview)
        self.tree_process.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_process.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags de colors
        self.tree_process.tag_configure('OK', background='#c6efce', foreground='#006100')
        self.tree_process.tag_configure('WARN', background='#fff3cd', foreground='#856404')
        self.tree_process.tag_configure('FAIL', background='#f8d7da', foreground='#721c24')
        self.tree_process.tag_configure('MODIFIED', background='#cce5ff', foreground='#004085')

        # Double click per obrir gràfic detall
        self.tree_process.bind("<Double-1>", self._open_qaqc_plot)

        # Menú contextual per canviar selecció
        self.context_menu_process = tk.Menu(self.root, tearoff=0)
        self.context_menu_process.add_command(label="DOC → Rèplica 1", command=lambda: self._force_selection("DOC", "1"))
        self.context_menu_process.add_command(label="DOC → Rèplica 2", command=lambda: self._force_selection("DOC", "2"))
        self.context_menu_process.add_separator()
        self.context_menu_process.add_command(label="DAD → Rèplica 1", command=lambda: self._force_selection("DAD", "1"))
        self.context_menu_process.add_command(label="DAD → Rèplica 2", command=lambda: self._force_selection("DAD", "2"))
        self.context_menu_process.add_separator()
        self.context_menu_process.add_command(label="Obrir gràfic QAQC", command=self._open_selected_qaqc_plot)

        self.tree_process.bind("<Button-3>", self._show_process_context_menu)

        # Progress i botó
        self.progress_proc = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_proc.pack(pady=10)

        self.btn_processar = tk.Button(parent, text="Processar Mostres",
                                       command=self._run_processing,
                                       bg=COLORS["primary"], fg=COLORS["white"],
                                       font=("Segoe UI", 11, "bold"),
                                       relief="flat", padx=30, pady=10,
                                       state="disabled")
        self.btn_processar.pack(pady=10)

    def _build_step_exportar(self, parent):
        """Pas 4: Exportació."""
        tk.Label(parent, text="Pas 4: Exportació Final",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Genera els fitxers finals i els informes PDF.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 20))

        # Opcions d'exportació
        options_frame = tk.LabelFrame(parent, text="Opcions d'Exportació",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=COLORS["white"])
        options_frame.pack(fill=tk.X, pady=10)

        self.var_export_excel = tk.BooleanVar(value=True)
        self.var_export_pdf = tk.BooleanVar(value=True)
        self.var_export_kpis = tk.BooleanVar(value=True)

        tk.Checkbutton(options_frame, text="Fitxers Excel finals (FINAL/)",
                      variable=self.var_export_excel,
                      font=("Segoe UI", 10),
                      bg=COLORS["white"]).pack(anchor="w", padx=10, pady=5)

        tk.Checkbutton(options_frame, text="Informe PDF complet (CHECK/)",
                      variable=self.var_export_pdf,
                      font=("Segoe UI", 10),
                      bg=COLORS["white"]).pack(anchor="w", padx=10, pady=5)

        tk.Checkbutton(options_frame, text="Resum KPIs en Excel",
                      variable=self.var_export_kpis,
                      font=("Segoe UI", 10),
                      bg=COLORS["white"]).pack(anchor="w", padx=10, pady=5)

        # Resum
        summary_frame = tk.LabelFrame(parent, text="Resum",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=COLORS["white"])
        summary_frame.pack(fill=tk.X, pady=10)

        self.lbl_exp_total = tk.Label(summary_frame, text="Mostres a exportar: -",
                                      font=("Segoe UI", 10),
                                      bg=COLORS["white"])
        self.lbl_exp_total.pack(anchor="w", padx=10, pady=5)

        self.lbl_exp_ok = tk.Label(summary_frame, text="Sense anomalies: -",
                                   font=("Segoe UI", 10),
                                   bg=COLORS["white"])
        self.lbl_exp_ok.pack(anchor="w", padx=10, pady=5)

        self.lbl_exp_warn = tk.Label(summary_frame, text="Amb anomalies: -",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["white"])
        self.lbl_exp_warn.pack(anchor="w", padx=10, pady=5)

        # Progress i botó
        self.progress_exp = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_exp.pack(pady=10)

        self.btn_exportar = tk.Button(parent, text="Exportar Tot",
                                      command=self._run_export,
                                      bg=COLORS["success"], fg=COLORS["white"],
                                      font=("Segoe UI", 11, "bold"),
                                      relief="flat", padx=30, pady=10,
                                      state="disabled")
        self.btn_exportar.pack(pady=10)

    def _build_footer(self):
        """Construeix el footer amb navegació."""
        footer = tk.Frame(self.root, bg=COLORS["light"], height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        self.btn_prev = tk.Button(footer, text="← Anterior",
                                  command=self._prev_step,
                                  bg=COLORS["dark"], fg=COLORS["white"],
                                  font=("Segoe UI", 10),
                                  relief="flat", padx=20, pady=8,
                                  state="disabled")
        self.btn_prev.pack(side=tk.LEFT, padx=20, pady=12)

        self.btn_next = tk.Button(footer, text="Següent →",
                                  command=self._next_step,
                                  bg=COLORS["primary"], fg=COLORS["white"],
                                  font=("Segoe UI", 10),
                                  relief="flat", padx=20, pady=8)
        self.btn_next.pack(side=tk.RIGHT, padx=20, pady=12)

        # Status
        self.lbl_footer_status = tk.Label(footer, text="Selecciona una carpeta SEQ per començar",
                                          font=("Segoe UI", 9),
                                          bg=COLORS["light"], fg=COLORS["dark"])
        self.lbl_footer_status.pack(expand=True)

    # =========================================================================
    # NAVEGACIÓ
    # =========================================================================
    def _show_step(self, step_idx):
        """Mostra un pas específic."""
        # Amagar tots els frames
        for frame in self.step_frames:
            frame.pack_forget()

        # Mostrar el frame seleccionat
        self.step_frames[step_idx].pack(fill=tk.BOTH, expand=True)
        self.current_step = step_idx

        # Actualitzar indicadors
        for i, lbl in enumerate(self.step_labels):
            if i == step_idx:
                lbl.configure(bg=COLORS["primary"], fg=COLORS["white"],
                             font=("Segoe UI", 11, "bold"))
            elif i < step_idx:
                lbl.configure(bg=COLORS["success"], fg=COLORS["white"],
                             font=("Segoe UI", 11))
            else:
                lbl.configure(bg=COLORS["light"], fg=COLORS["dark"],
                             font=("Segoe UI", 11))

        # Actualitzar botons de navegació
        self.btn_prev.configure(state="normal" if step_idx > 0 else "disabled")
        self.btn_next.configure(state="normal" if step_idx < 3 else "disabled")

        if step_idx == 3:
            self.btn_next.configure(text="Finalitzar")

    def _next_step(self):
        if self.current_step < 3:
            self._show_step(self.current_step + 1)

    def _prev_step(self):
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _on_step_click(self, idx):
        """Permet clicar als passos per navegar."""
        self._show_step(idx)

    # =========================================================================
    # SELECCIÓ DE CARPETA
    # =========================================================================
    def _open_samples_history(self):
        """Obre la finestra d'històric de mostres."""
        SamplesHistoryWindow(self.root, self.seq_path)

    def _select_folder(self):
        """Selecciona la carpeta SEQ."""
        folder = filedialog.askdirectory(title="Selecciona la carpeta SEQ")
        if not folder:
            return

        self.seq_path = folder
        self.path_var.set(os.path.basename(folder))

        # RESET complet quan es canvia de carpeta
        self.is_processing = False  # IMPORTANT: reset flag processament
        self.consolidated_data = {}
        self.calibration_data = {}
        self.selected_replicas = {}

        # Reset GUI labels consolidació
        self.lbl_con_status.configure(text="Pendent de consolidar")
        if hasattr(self, 'progress_con'):
            self.progress_con['value'] = 0

        # Reset GUI labels calibració
        self.lbl_cal_khp.configure(text="KHP: No calibrat", fg=COLORS["dark"])
        self.lbl_cal_factor.configure(text="Àrea: - | Conc: -")
        self.lbl_cal_r2.configure(text="SHIFT: - | Alineació: -")
        self.btn_calibrar.configure(text="Executar Calibració", state="disabled")
        if hasattr(self, 'btn_outlier'):
            self.btn_outlier.pack_forget()  # Amagar botó outlier
        if hasattr(self, 'progress_cal'):
            self.progress_cal['value'] = 0
        if hasattr(self, 'lbl_cal_status'):
            self.lbl_cal_status.configure(text="")

        # Netejar gràfic calibració
        for widget in self.cal_canvas_frame.winfo_children():
            widget.destroy()

        # Netejar taula processament
        for item in self.tree_process.get_children():
            self.tree_process.delete(item)

        # Reset labels exportació
        self.lbl_exp_total.configure(text="Mostres a exportar: -")
        self.lbl_exp_ok.configure(text="Sense anomalies: -")
        self.lbl_exp_warn.configure(text="Amb anomalies: -")

        # Deshabilitar botons que requereixen passos anteriors
        self.btn_calibrar.configure(state="disabled")
        self.btn_processar.configure(state="disabled")
        self.btn_exportar.configure(state="disabled")

        # Escanejar carpeta
        self._scan_folder()

        # Habilitar consolidar
        self.btn_consolidar.configure(state="normal")
        self.lbl_footer_status.configure(text=f"Nova carpeta: {os.path.basename(folder)}")

    def _scan_folder(self):
        """Escaneja la carpeta SEQ."""
        if not self.seq_path:
            return

        # Buscar fitxers .D
        d_files = glob.glob(os.path.join(self.seq_path, "**", "*.D"), recursive=True)

        # Buscar fitxers consolidats existents (Resultats_Consolidats o Consolidat)
        resultats_path = os.path.join(self.seq_path, "Resultats_Consolidats")
        consolidat_path_old = os.path.join(self.seq_path, "Consolidat")

        existing_xlsx = []
        if os.path.exists(resultats_path):
            consolidat_path = resultats_path
            existing_xlsx = glob.glob(os.path.join(resultats_path, "*.xlsx"))
        elif os.path.exists(consolidat_path_old):
            consolidat_path = consolidat_path_old
            existing_xlsx = glob.glob(os.path.join(consolidat_path_old, "*.xlsx"))
        else:
            consolidat_path = resultats_path  # Default per nous

        # Comptar KHP
        khp_count = sum(1 for f in d_files if "KHP" in f.upper())

        # Actualitzar UI
        self.lbl_con_files.configure(text=f"Fitxers .D detectats: {len(d_files)}")
        self.lbl_con_samples.configure(text=f"Excel consolidats existents: {len(existing_xlsx)}")
        self.lbl_con_khp.configure(text=f"Mostres KHP: {khp_count}")

        # Si ja hi ha consolidats, habilitar calibració
        if existing_xlsx:
            self.consolidated_data = {"path": consolidat_path, "files": existing_xlsx}

            # Comprovar si ja hi ha calibració existent (CALDATA)
            is_calibrated, active_cal = is_seq_calibrated(self.seq_path)

            if is_calibrated and active_cal:
                # Mostrar calibració existent
                self._load_existing_calibration(active_cal)
                self.btn_calibrar.configure(text="Recalibrar", state="normal")
            else:
                self.btn_calibrar.configure(text="Executar Calibració", state="normal")

    # =========================================================================
    # PAS 1: CONSOLIDACIÓ
    # =========================================================================
    def _run_consolidation(self):
        """Executa la consolidació en un thread."""
        if self.is_processing:
            return

        self.is_processing = True
        self.btn_consolidar.configure(state="disabled")
        self.lbl_con_status.configure(text="Iniciant consolidació...")

        thread = threading.Thread(target=self._consolidation_thread)
        thread.start()

    def _consolidation_thread(self):
        """Thread de consolidació - Usa mòdul hpsec_consolidate."""
        try:
            def progress_callback(pct, sample):
                self.root.after(0, lambda p=pct, s=sample: self._update_con_progress(p, s))

            # Executar consolidació
            result = consolidate_sequence(
                self.seq_path,
                config=self.config,
                progress_callback=progress_callback
            )

            if not result["success"]:
                error_msg = result["errors"][0] if result["errors"] else "Error desconegut"
                self.root.after(0, lambda: self._consolidation_error(error_msg))
                return

            # Guardar info per _consolidation_done
            self.consolidation_info = {
                "mode": result["mode"],
                "bp": result["bp"],
                "seq": result["seq"],
                "date": result["date"],
                "file_check": result.get("file_check", {}),
                "alignment": result.get("alignment", {}),
            }
            self.consolidation_count = result["processed_count"]
            self.consolidation_files = result["files"]
            self.consolidation_errors = result.get("errors", [])

            self.root.after(0, self._consolidation_done)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._consolidation_error(str(e)))

    def _write_consolidated_excel(self, out_path, mostra, rep, seq_out, date_master,
                                   method, doc_mode, fitxer_doc, fitxer_dad, st_doc, st_dad,
                                   t_doc, y_doc_raw, y_doc_net, baseline, df_dad, peak_info,
                                   sample_analysis=None, master_file=None, row_start=None, row_end=None):
        """Escriu fitxer Excel consolidat - wrapper a funció standalone."""
        write_consolidated_excel(
            out_path, mostra, rep, seq_out, date_master,
            method, doc_mode, fitxer_doc, fitxer_dad, st_doc, st_dad,
            t_doc, y_doc_raw, y_doc_net, baseline, df_dad, peak_info,
            sample_analysis, master_file, row_start, row_end
        )

    def _update_con_progress(self, progress, sample):
        """Actualitza el progrés de consolidació."""
        self.progress_con['value'] = progress
        self.lbl_con_status.configure(text=f"Processant: {sample}")

    def _consolidation_done(self):
        """Consolidació completada."""
        self.is_processing = False
        self.progress_con['value'] = 100
        self.btn_consolidar.configure(state="normal")
        self.btn_calibrar.configure(state="normal")

        # Actualitzar dades
        consolidat_path = os.path.join(self.seq_path, "Resultats_Consolidats")
        xlsx_files = glob.glob(os.path.join(consolidat_path, "*.xlsx"))
        self.consolidated_data = {"path": consolidat_path, "files": xlsx_files}

        # Mostrar info del procés
        info = getattr(self, 'consolidation_info', {})
        count = getattr(self, 'consolidation_count', len(xlsx_files))

        mode_str = info.get('mode', 'N/A')
        bp_str = "SI" if info.get('bp', False) else "NO"
        seq_str = info.get('seq', 'N/A')

        # Comptar parells DOC-DAD i punts
        khp_count = 0
        dad_ok_count = 0
        total_doc_points = 0
        total_dad_points = 0

        for f in xlsx_files:
            fname = os.path.basename(f)
            if is_khp(fname):
                khp_count += 1
            try:
                df_id = pd.read_excel(f, "ID", engine="openpyxl")
                id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))
                if str(id_dict.get("Estat_DAD", "")) == "OK":
                    dad_ok_count += 1
                total_doc_points += int(id_dict.get("DOC_N_POINTS", 0) or 0)
                total_dad_points += int(id_dict.get("DAD_N_POINTS", 0) or 0)
            except:
                pass

        self.lbl_con_status.configure(
            text=f"Completat: {count} fitxers | Mode: {mode_str} | BP: {bp_str} | SEQ: {seq_str}"
        )

        # Actualitzar info panel amb més detalls
        self.lbl_con_files.configure(text=f"Fitxers consolidats: {len(xlsx_files)} | Parells DOC-DAD: {dad_ok_count}/{len(xlsx_files)}")
        self.lbl_con_samples.configure(text=f"Mode: {mode_str} | SEQ: {seq_str} | Punts DOC: {total_doc_points:,}")
        self.lbl_con_khp.configure(text=f"Mostres KHP: {khp_count} | Punts DAD: {total_dad_points:,}")

        # Generar PDFs amb mòdul hpsec_reports
        try:
            generate_all_reports(self.seq_path, xlsx_files, info)
        except Exception as e:
            print(f"Error generant PDFs: {e}")

        # Obrir carpeta CHECK automàticament
        check_folder = os.path.join(self.seq_path, "CHECK")
        if os.path.exists(check_folder):
            try:
                os.startfile(check_folder)
            except:
                pass

        messagebox.showinfo("Consolidació",
            f"Consolidació completada!\n\n"
            f"Fitxers generats: {count}\n"
            f"Mode: {mode_str}\n"
            f"BP: {bp_str}\n"
            f"SEQ: {seq_str}\n\n"
            f"Parells DOC-DAD vàlids: {dad_ok_count}/{len(xlsx_files)}\n"
            f"Mostres KHP: {khp_count}\n\n"
            f"Carpeta CHECK oberta automàticament"
        )

    def _check_dad_status(self, filepath):
        """Comprova si un fitxer té DAD vàlid."""
        try:
            df_id = pd.read_excel(filepath, "ID", engine="openpyxl")
            id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))
            return str(id_dict.get("Estat_DAD", "")) == "OK"
        except:
            return False

    def _consolidation_error(self, error):
        """Error en consolidació."""
        self.is_processing = False
        self.btn_consolidar.configure(state="normal")
        self.lbl_con_status.configure(text=f"Error: {error}")
        messagebox.showerror("Error", f"Error durant la consolidació:\n{error}")

    # =========================================================================
    # PAS 2: CALIBRACIÓ
    # =========================================================================
    def _run_calibration(self):
        """Executa la calibració."""
        if self.is_processing or not self.consolidated_data:
            return

        self.is_processing = True
        self.btn_calibrar.configure(state="disabled")

        thread = threading.Thread(target=self._calibration_thread)
        thread.start()

    def _calibration_thread(self):
        """Thread de calibració - Punt únic amb SHIFT DOC-DAD."""
        try:
            # Cercar KHP: LOCAL → SIBLINGS → MANUAL
            khp_files, khp_source = buscar_khp_consolidados(self.seq_path)

            if not khp_files:
                self.root.after(0, lambda: self._calibration_error("No s'ha trobat cap KHP"))
                return

            self.root.after(0, lambda s=khp_source: self._update_cal_status(f"KHP trobat: {s}"))

            # Analitzar KHP (punt únic o promig de rèpliques)
            khp_data = analizar_khp_lote(khp_files, self.config)

            if not khp_data:
                self.root.after(0, lambda: self._calibration_error("KHP invàlid (no s'ha pogut analitzar)"))
                return

            # Determinar si aplicar alineació
            # En BP NO s'aplica alineació - el cromatograma és massa curt i el pic massa primerenc
            # L'alineació podria portar temps a valors negatius i perdre senyal
            is_bp_mode = khp_data.get('is_bp', False)
            if not is_bp_mode:
                # Detectar BP pel temps del pic o durada del cromatograma
                t_peak = khp_data.get('t_doc_max', 10)
                t_doc_arr = khp_data.get('t_doc')
                if t_doc_arr is not None and len(t_doc_arr) > 0:
                    t_max_chromato = float(np.max(t_doc_arr))
                else:
                    t_max_chromato = 60
                is_bp_mode = t_peak < 10 or t_max_chromato < 20

            threshold_min = self.config["alignment_threshold_sec"] / 60.0
            if is_bp_mode:
                apply_alignment = False  # Mai alinear en BP
            else:
                apply_alignment = abs(khp_data['shift_min']) > threshold_min

            # Actualitzar tots els fitxers amb info KHP
            consolidat_path = self.consolidated_data.get("path", "")
            xlsx_files = glob.glob(os.path.join(consolidat_path, "*.xlsx"))
            xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

            updated_count = 0
            total = len(xlsx_files)

            for i, fpath in enumerate(xlsx_files):
                progress = int(100 * (i + 1) / max(total, 1))
                self.root.after(0, lambda p=progress: self._update_cal_progress(p))

                try:
                    updated = self._update_file_with_khp(fpath, khp_data, apply_alignment)
                    if updated:
                        updated_count += 1
                except Exception as e:
                    print(f"Error actualitzant {os.path.basename(fpath)}: {e}")

            # Guardar dades de calibració
            self.calibration_data = {
                "khp_data": khp_data,
                "source": khp_source,
                "apply_alignment": apply_alignment,
                "updated_files": updated_count,
                "total_files": total,
            }

            # Generar PDF de calibració
            self._generate_calibration_pdf(khp_data, khp_source, apply_alignment, updated_count)

            self.root.after(0, self._calibration_done)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._calibration_error(str(e)))

    def _update_file_with_khp(self, filepath, khp_data, apply_alignment):
        """Actualitza un fitxer Excel amb la informació de calibració KHP."""
        try:
            xls = pd.ExcelFile(filepath, engine="openpyxl")
            df_id = pd.read_excel(xls, "ID", engine="openpyxl")
            df_doc = pd.read_excel(xls, "DOC", engine="openpyxl")
            df_dad = pd.read_excel(xls, "DAD", engine="openpyxl") if "DAD" in xls.sheet_names else pd.DataFrame()

            # Aplicar alineació temporal si cal
            if apply_alignment and "time (min)" in df_doc.columns:
                df_doc["time (min)"] = df_doc["time (min)"] + khp_data['shift_min']

            # Recalcular àrea amb el temps alineat
            t = df_doc["time (min)"].to_numpy()

            # Trobar columna DOC
            doc_col = None
            for c in df_doc.columns:
                if 'doc' in str(c).lower() and 'raw' not in str(c).lower():
                    doc_col = c
                    break
            if doc_col is None:
                doc_col = df_doc.columns[1]

            y = pd.to_numeric(df_doc[doc_col], errors="coerce").to_numpy()
            peak_info = detect_main_peak(t, y, self.config["peak_min_prominence_pct"])

            area_sample = peak_info['area'] if peak_info['valid'] else 0.0

            # Calcular concentració (calibració punt únic)
            if khp_data['area'] > 0 and area_sample > 0:
                conc_sample = (area_sample / khp_data['area']) * khp_data['conc_ppm']
            else:
                conc_sample = 0.0

            # Actualitzar ID
            id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

            id_dict["KHP_FILE"] = khp_data['filename']
            id_dict["KHP_CONC_PPM"] = khp_data['conc_ppm']
            id_dict["KHP_AREA"] = f"{khp_data['area']:.2f}"
            id_dict["SHIFT_KHP_MIN"] = f"{khp_data['shift_min']:.4f}"
            id_dict["ALIGNMENT_APPLIED"] = "YES" if apply_alignment else "NO"
            id_dict["AREA_SAMPLE"] = f"{area_sample:.2f}"
            id_dict["CONC_SAMPLE_PPM"] = f"{conc_sample:.2f}"

            # Info de calibració externa (si aplica)
            if khp_data.get('source_seq'):
                id_dict["CAL_SOURCE_SEQ"] = khp_data['source_seq']
                id_dict["CAL_IS_OUTLIER"] = "YES"
            else:
                id_dict["CAL_SOURCE_SEQ"] = "LOCAL"
                id_dict["CAL_IS_OUTLIER"] = "NO"

            df_id_new = pd.DataFrame(list(id_dict.items()), columns=["Camp", "Valor"])

            # Guardar
            with pd.ExcelWriter(filepath, engine="openpyxl") as w:
                df_id_new.to_excel(w, "ID", index=False)
                df_doc.to_excel(w, "DOC", index=False)
                if not df_dad.empty:
                    df_dad.to_excel(w, "DAD", index=False)

            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def _generate_calibration_pdf(self, khp_data, khp_source, apply_alignment, updated_count):
        """Genera PDF complet d'informe de calibració amb gràfics."""
        try:
            check_path = ensure_check_folder(self.seq_path)
            seq_name = obtenir_seq(self.seq_path)
            pdf_path = os.path.join(check_path, f"REPORT_Calibracio_{seq_name}.pdf")

            # Usar PdfPages per incloure gràfics matplotlib
            with PdfPages(pdf_path) as pdf:
                # === PÀGINA 1: Portada i resum ===
                fig = plt.figure(figsize=(8.27, 11.69))  # A4
                fig.patch.set_facecolor('white')

                # Logo
                if os.path.exists(LOGO_PATH):
                    try:
                        logo_img = plt.imread(LOGO_PATH)
                        ax_logo = fig.add_axes([0.05, 0.88, 0.15, 0.08])
                        ax_logo.imshow(logo_img)
                        ax_logo.axis('off')
                    except:
                        pass

                # Títol
                fig.text(0.5, 0.92, PDF_HEADER, ha='center', fontsize=10, color='#666666')
                fig.text(0.5, 0.87, "INFORME DE CALIBRACIÓ KHP", ha='center', fontsize=18,
                        fontweight='bold', color='#2E86AB')
                fig.text(0.5, 0.83, f"Seqüència: {seq_name}", ha='center', fontsize=12)
                fig.text(0.5, 0.80, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ha='center', fontsize=10, color='#888888')

                # Línia separadora
                ax_line = fig.add_axes([0.1, 0.78, 0.8, 0.001])
                ax_line.axhline(y=0, color='#2E86AB', linewidth=2)
                ax_line.axis('off')

                # Taula resum principal
                ax_table = fig.add_axes([0.1, 0.55, 0.8, 0.20])
                ax_table.axis('off')

                threshold_sec = self.config['alignment_threshold_sec']
                shift_sec = khp_data.get('shift_sec', khp_data.get('shift_min', 0) * 60)

                table_data = [
                    ['PARÀMETRE', 'VALOR', 'ESTAT'],
                    ['Fitxer KHP', khp_data['filename'], ''],
                    ['Origen', khp_source, ''],
                    ['Concentració', f"{khp_data['conc_ppm']} ppm", ''],
                    ['Àrea integrada', f"{khp_data['area']:.2f}", khp_data.get('status', 'OK')],
                    ['SHIFT DOC-DAD', f"{shift_sec:.1f} s", 'ALINEAR' if apply_alignment else 'OK'],
                    ['Simetria pic', f"{khp_data.get('symmetry', 1.0):.2f}", 'OK' if 0.8 <= khp_data.get('symmetry', 1.0) <= 1.5 else 'REVISAR'],
                    ['SNR', f"{khp_data.get('snr', 0):.1f}", 'OK' if khp_data.get('snr', 0) > 10 else 'BAIX'],
                    ['Pics detectats', str(khp_data.get('all_peaks_count', 1)), ''],
                    ['Rèpliques', str(khp_data.get('n_replicas', 1)), ''],
                ]

                if khp_data.get('rsd'):
                    table_data.append(['RSD àrees', f"{khp_data['rsd']:.1f}%", 'OK' if khp_data['rsd'] < 10 else 'ALT'])

                # Colors per estat
                cell_colors = [['#2E86AB']*3]  # Header
                for row in table_data[1:]:
                    estat = row[2]
                    if estat in ['OK', '']:
                        cell_colors.append(['white', 'white', '#d4edda'])
                    elif estat == 'ALINEAR':
                        cell_colors.append(['white', 'white', '#fff3cd'])
                    else:
                        cell_colors.append(['white', 'white', '#f8d7da'])

                tbl = ax_table.table(cellText=table_data, cellColours=cell_colors,
                                    colWidths=[0.4, 0.35, 0.25], loc='center',
                                    cellLoc='left')
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.5)

                # Header style
                for j in range(3):
                    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

                # Informació d'alineació
                if apply_alignment:
                    fig.text(0.1, 0.50, f"ALINEACIÓ APLICADA: Temps DOC desplaçat {shift_sec:.1f}s (umbral: {threshold_sec}s)",
                            fontsize=10, fontweight='bold', color='#856404',
                            bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='#ffc107'))
                else:
                    fig.text(0.1, 0.50, f"Sense alineació necessària (SHIFT {shift_sec:.1f}s < umbral {threshold_sec}s)",
                            fontsize=10, color='#155724')

                # Problemes de qualitat
                quality_issues = khp_data.get('quality_issues', [])
                if quality_issues:
                    fig.text(0.1, 0.45, "Alertes de qualitat:", fontsize=10, fontweight='bold', color='#721c24')
                    for i, issue in enumerate(quality_issues[:4]):
                        fig.text(0.12, 0.43 - i*0.02, f"• {issue}", fontsize=9, color='#721c24')

                # Resum actualització
                fig.text(0.1, 0.32, f"Fitxers actualitzats: {updated_count}", fontsize=11, fontweight='bold')

                # Referència al PDF de cromatogrames per validar patrons
                fig.text(0.1, 0.26, "VALIDACIÓ DE PATRONS:",
                        fontsize=10, fontweight='bold', color='#2E86AB')
                fig.text(0.1, 0.23, f"Per revisar els cromatogrames individuals i validar els patrons KHP,",
                        fontsize=9, color='#333333')
                fig.text(0.1, 0.20, f"consulteu el fitxer: REPORT_Cromatogrames_{seq_name}.pdf",
                        fontsize=9, fontweight='bold', color='#2E86AB')
                fig.text(0.1, 0.17, "(disponible a la carpeta CHECK)",
                        fontsize=8, color='#666666', style='italic')

                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

                # === PÀGINA 2+: Gràfics per cada KHP ===
                all_khp = khp_data.get('all_khp_data', [khp_data])
                replicas = khp_data.get('replicas', all_khp)

                # Detectar mode BP per escala X
                is_bp = khp_data.get('is_bp', False)
                if is_bp:
                    x_min, x_max = -2, 4
                else:
                    x_min, x_max = 0, 70

                for i, khp in enumerate(replicas[:4]):  # Màxim 4 gràfics
                    if khp.get('t_doc') is None:
                        continue

                    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
                    fig.patch.set_facecolor('white')

                    # Títol pàgina
                    is_selected = (khp_data.get('selected_idx', 0) == i) or (khp_data.get('is_average', False) and i == 0)
                    title_color = '#28a745' if is_selected else '#666666'
                    selected_text = " [SELECCIONAT]" if is_selected else ""
                    fig.suptitle(f"KHP Rèplica {i+1}: {khp['filename']}{selected_text}",
                                fontsize=14, fontweight='bold', color=title_color, y=0.98)

                    t_doc = khp.get('t_doc', [])
                    y_doc = khp.get('y_doc', [])
                    t_dad = khp.get('t_dad')
                    y_dad = khp.get('y_dad_254')
                    peak_info = khp.get('peak_info', {})

                    # Convertir a numpy arrays i assegurar 1D
                    t_doc = np.asarray(t_doc).flatten() if t_doc is not None and len(t_doc) > 0 else np.array([])
                    y_doc = np.asarray(y_doc).flatten() if y_doc is not None and len(y_doc) > 0 else np.array([])

                    if len(t_doc) < 10 or len(y_doc) < 10:
                        continue  # Saltar si no hi ha prou dades

                    # === Gràfic 1: DOC amb pic marcat ===
                    ax1 = axes[0, 0]
                    ax1.plot(t_doc, y_doc, 'b-', linewidth=1.5, label='DOC')
                    ax1.set_xlim(x_min, x_max)
                    ax1.set_xlabel('Temps (min)')
                    ax1.set_ylabel('DOC (mAU)', color='blue')
                    ax1.set_title('Senyal DOC amb àrea integrada')
                    ax1.grid(True, alpha=0.3)

                    # Marcar pic principal i àrea
                    if peak_info.get('valid'):
                        t_max = peak_info.get('t_max', 0)
                        ax1.axvline(x=t_max, color='red', linestyle='--', alpha=0.7, label=f'Pic: {t_max:.2f} min')

                        # Ombrar àrea amb bounds checking robust
                        left_idx = khp.get('peak_left_idx', 0)
                        right_idx = khp.get('peak_right_idx', len(t_doc)-1)

                        # Assegurar bounds vàlids
                        left_idx = int(max(0, min(left_idx, len(t_doc) - 1)))
                        right_idx = int(max(0, min(right_idx, len(t_doc) - 1)))

                        if left_idx < right_idx:
                            ax1.fill_between(t_doc[left_idx:right_idx+1], 0,
                                           y_doc[left_idx:right_idx+1],
                                           alpha=0.3, color='blue', label=f'Àrea: {peak_info.get("area", 0):.2f}')

                    # Marcar tots els pics
                    all_peaks = khp.get('all_peaks', [])
                    for pk in all_peaks:
                        ax1.plot(pk['t'], pk['height'], 'ro', markersize=6)

                    ax1.legend(loc='upper right', fontsize=8)

                    # === Gràfic 2: DOC + DAD 254 superposats ===
                    ax2 = axes[0, 1]
                    ax2.plot(t_doc, y_doc, 'b-', linewidth=1.5, label='DOC')
                    ax2.set_xlim(x_min, x_max)
                    ax2.set_xlabel('Temps (min)')
                    ax2.set_ylabel('DOC (mAU)', color='blue')
                    ax2.tick_params(axis='y', labelcolor='blue')

                    if t_dad is not None and y_dad is not None and len(t_dad) > 0:
                        ax2b = ax2.twinx()
                        ax2b.plot(t_dad, y_dad, 'r-', linewidth=1.5, label='DAD 254nm', alpha=0.7)
                        ax2b.set_ylabel('DAD 254nm (mAU)', color='red')
                        ax2b.tick_params(axis='y', labelcolor='red')

                        # Marcar màxims
                        t_doc_max = khp.get('t_doc_max')
                        t_dad_max = khp.get('t_dad_max')
                        if t_doc_max:
                            ax2.axvline(x=t_doc_max, color='blue', linestyle=':', alpha=0.7)
                        if t_dad_max:
                            ax2.axvline(x=t_dad_max, color='red', linestyle=':', alpha=0.7)

                        # Mostrar SHIFT
                        shift = khp.get('shift_min', 0)
                        ax2.set_title(f'DOC vs DAD 254nm (SHIFT: {shift*60:.1f}s)')

                        # Fletxa indicant shift
                        if t_doc_max and t_dad_max:
                            y_arrow = max(y_doc) * 0.9
                            ax2.annotate('', xy=(t_dad_max, y_arrow), xytext=(t_doc_max, y_arrow),
                                        arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                            ax2.text((t_doc_max + t_dad_max)/2, y_arrow*1.05, f'{shift*60:.1f}s',
                                    ha='center', fontsize=9, color='green', fontweight='bold')
                    else:
                        ax2.set_title('DOC (sense DAD disponible)')

                    ax2.grid(True, alpha=0.3)

                    # === Gràfic 3: Zoom al pic ===
                    ax3 = axes[1, 0]
                    if peak_info.get('valid') and len(t_doc) > 10:
                        t_max = peak_info.get('t_max', 25)
                        zoom_start = max(0, t_max - 5)
                        zoom_end = min(float(np.max(t_doc)), t_max + 5)

                        mask_zoom = (t_doc >= zoom_start) & (t_doc <= zoom_end)
                        if np.any(mask_zoom):
                            ax3.plot(t_doc[mask_zoom], y_doc[mask_zoom], 'b-', linewidth=2)

                            # Ombrar àrea amb bounds checking robust
                            left_idx = khp.get('peak_left_idx', 0)
                            right_idx = khp.get('peak_right_idx', len(t_doc)-1)

                            # Assegurar bounds vàlids
                            left_idx = int(max(0, min(left_idx, len(t_doc) - 1)))
                            right_idx = int(max(0, min(right_idx, len(t_doc) - 1)))

                            zoom_left_idx = int(np.searchsorted(t_doc, zoom_start))
                            zoom_right_idx = int(np.searchsorted(t_doc, zoom_end))

                            int_left = max(left_idx, zoom_left_idx)
                            int_right = min(right_idx, zoom_right_idx)

                            # Verificar bounds finals
                            int_left = max(0, min(int_left, len(t_doc) - 1))
                            int_right = max(0, min(int_right, len(t_doc) - 1))

                            if int_right > int_left:
                                ax3.fill_between(t_doc[int_left:int_right+1], 0,
                                               y_doc[int_left:int_right+1],
                                               alpha=0.4, color='green')

                            # Marcar límits d'integració
                            t_start = peak_info.get('t_start', 0)
                            t_end = peak_info.get('t_end', 0)
                            if t_start > 0:
                                ax3.axvline(x=t_start, color='green', linestyle='-', linewidth=2, alpha=0.8)
                            if t_end > 0:
                                ax3.axvline(x=t_end, color='green', linestyle='-', linewidth=2, alpha=0.8)

                            # Marcar pic
                            ax3.axvline(x=t_max, color='red', linestyle='--', linewidth=2)
                            ax3.scatter([t_max], [peak_info.get('height', 0)], color='red', s=100, zorder=5)

                    ax3.set_xlabel('Temps (min)')
                    ax3.set_ylabel('DOC (mAU)')
                    ax3.set_title('Zoom al pic principal')
                    ax3.grid(True, alpha=0.3)

                    # === Gràfic 4: Taula de qualitat ===
                    ax4 = axes[1, 1]
                    ax4.axis('off')

                    quality_data = [
                        ['Mètrica', 'Valor', 'Referència'],
                        ['Àrea DOC', f"{khp.get('area', 0):.2f}", '-'],
                        ['Temps pic', f"{khp.get('t_doc_max', 0):.2f} min", '-'],
                        ['Simetria', f"{khp.get('symmetry', 1.0):.2f}", '0.8-1.5'],
                        ['SNR', f"{khp.get('snr', 0):.1f}", '>10'],
                        ['Pics detectats', str(khp.get('all_peaks_count', 1)), '1'],
                        ['SHIFT', f"{khp.get('shift_min', 0)*60:.1f} s", f'<{threshold_sec}s'],
                        ['Score qualitat', str(khp.get('quality_score', 0)), '0'],
                    ]

                    # Afegir A254 com a check de calibració
                    a254_area = khp.get('a254_area', 0)
                    a254_ratio = khp.get('a254_doc_ratio', 0)
                    if a254_area > 0:
                        quality_data.append(['Àrea A254', f"{a254_area:.2f}", 'CHECK'])
                        quality_data.append(['Ratio DOC/A254', f"{a254_ratio:.3f}", 'Referència'])

                    if khp.get('quality_issues'):
                        quality_data.append(['Alertes', ', '.join(khp['quality_issues'][:2]), '-'])

                    tbl4 = ax4.table(cellText=quality_data, loc='center', cellLoc='center',
                                    colWidths=[0.35, 0.35, 0.3])
                    tbl4.auto_set_font_size(False)
                    tbl4.set_fontsize(9)
                    tbl4.scale(1.2, 1.8)

                    # Header style
                    for j in range(3):
                        tbl4[(0, j)].set_facecolor('#2E86AB')
                        tbl4[(0, j)].set_text_props(color='white', fontweight='bold')

                    fig.tight_layout(rect=[0, 0, 1, 0.96])
                    pdf.savefig(fig, facecolor='white')
                    plt.close(fig)

                # === PÀGINA FINAL: Estadístiques si múltiples rèpliques ===
                stats = khp_data.get('stats')
                if stats and stats.get('n_replicas', 1) > 1:
                    fig = plt.figure(figsize=(8.27, 11.69))
                    fig.patch.set_facecolor('white')

                    fig.text(0.5, 0.95, "Estadístiques de Rèpliques KHP", ha='center',
                            fontsize=16, fontweight='bold', color='#2E86AB')

                    # Taula estadístiques
                    ax_stats = fig.add_axes([0.1, 0.70, 0.8, 0.20])
                    ax_stats.axis('off')

                    stats_data = [
                        ['Estadística', 'Valor'],
                        ['Nombre rèpliques', str(stats['n_replicas'])],
                        ['Àrea mitjana', f"{stats['mean_area']:.2f}"],
                        ['Desv. estàndard àrea', f"{stats['std_area']:.2f}"],
                        ['RSD àrea', f"{stats['rsd_area']:.1f}%"],
                        ['SHIFT mitjà', f"{stats['mean_shift']*60:.1f} s"],
                        ['Desv. estàndard SHIFT', f"{stats['std_shift']*60:.1f} s"],
                        ['Simetria mitjana', f"{stats['mean_symmetry']:.2f}"],
                        ['SNR mitjà', f"{stats['mean_snr']:.1f}"],
                    ]

                    tbl_stats = ax_stats.table(cellText=stats_data, loc='center', cellLoc='left',
                                              colWidths=[0.5, 0.5])
                    tbl_stats.auto_set_font_size(False)
                    tbl_stats.set_fontsize(10)
                    tbl_stats.scale(1.2, 2)

                    for j in range(2):
                        tbl_stats[(0, j)].set_facecolor('#2E86AB')
                        tbl_stats[(0, j)].set_text_props(color='white', fontweight='bold')

                    # Gràfic de barres comparatiu
                    ax_bars = fig.add_axes([0.1, 0.35, 0.8, 0.30])

                    areas = [r.get('area', 0) for r in replicas[:4]]
                    labels = [f"R{i+1}" for i in range(len(areas))]
                    colors_bars = ['#28a745' if i == khp_data.get('selected_idx', 0) else '#2E86AB'
                                  for i in range(len(areas))]

                    bars = ax_bars.bar(labels, areas, color=colors_bars)
                    ax_bars.axhline(y=stats['mean_area'], color='red', linestyle='--',
                                   label=f"Mitjana: {stats['mean_area']:.2f}")
                    ax_bars.set_ylabel('Àrea')
                    ax_bars.set_title('Comparació d\'àrees entre rèpliques')
                    ax_bars.legend()

                    # Valor sobre cada barra
                    for bar, area in zip(bars, areas):
                        ax_bars.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                    f'{area:.1f}', ha='center', va='bottom', fontsize=9)

                    # Decisió
                    if khp_data.get('is_average'):
                        decision_text = f"DECISIÓ: S'utilitza el PROMIG de les rèpliques (RSD {stats['rsd_area']:.1f}% < 10%)"
                        decision_color = '#155724'
                    else:
                        decision_text = f"DECISIÓ: S'utilitza la rèplica amb millor qualitat (RSD {stats['rsd_area']:.1f}% >= 10%)"
                        decision_color = '#856404'

                    fig.text(0.5, 0.28, decision_text, ha='center', fontsize=11,
                            fontweight='bold', color=decision_color,
                            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor=decision_color))

                    pdf.savefig(fig, facecolor='white')
                    plt.close(fig)

                # === PÀGINA COMPARACIÓ HISTÒRICA ===
                # Obtenir dades històriques
                mode_str = khp_data.get('mode', 'COLUMN')
                doc_mode = khp_data.get('doc_mode', 'N/A')
                conc_ppm = khp_data.get('conc_ppm', 0)
                volume_uL = khp_data.get('volume_uL', 0)

                hist_stats = get_equivalent_seqs_stats(
                    self.seq_path,
                    mode=mode_str,
                    doc_mode=doc_mode if doc_mode != "N/A" else None,
                    conc_ppm=conc_ppm if conc_ppm > 0 else None,
                    volume_uL=volume_uL if volume_uL > 0 else None,
                    exclude_current=True
                )

                if hist_stats and hist_stats.get('n_seqs', 0) > 0:
                    fig = plt.figure(figsize=(8.27, 11.69))
                    fig.patch.set_facecolor('white')

                    # Títol
                    fig.text(0.5, 0.95, "Comparació Històrica", ha='center',
                            fontsize=16, fontweight='bold', color='#2E86AB')
                    fig.text(0.5, 0.92, f"Mode: {mode_str} | DOC: {doc_mode} | {conc_ppm} ppm",
                            ha='center', fontsize=10, color='#666666')

                    # Àrea actual
                    current_area = khp_data.get('area', 0)
                    hist_mean = hist_stats['mean_area']
                    hist_std = hist_stats['std_area']

                    # Calcular desviació respecte a la mitjana
                    if hist_std > 0:
                        z_score = (current_area - hist_mean) / hist_std
                        deviation_pct = (current_area - hist_mean) / hist_mean * 100
                    else:
                        z_score = 0
                        deviation_pct = 0

                    # Gràfic de barres comparatiu (5 més recents + actual)
                    ax_bars = fig.add_axes([0.1, 0.55, 0.8, 0.30])

                    closest = hist_stats.get('closest_seqs', [])
                    seq_names = [s[0][:10] for s in closest]  # Noms truncats
                    seq_areas = [s[1] for s in closest]
                    seq_dates = [s[2] for s in closest]

                    # Afegir actual al principi
                    current_seq = obtenir_seq(self.seq_path)
                    all_names = [current_seq[:10] + "\n(ACTUAL)"] + [f"{n}\n{d}" for n, d in zip(seq_names, seq_dates)]
                    all_areas = [current_area] + seq_areas

                    # Colors: actual en verd, històrics en blau
                    colors_bars = ['#28a745'] + ['#2E86AB'] * len(seq_areas)

                    x_pos = range(len(all_names))
                    bars = ax_bars.bar(x_pos, all_areas, color=colors_bars, edgecolor='black', linewidth=0.5)
                    ax_bars.set_xticks(x_pos)
                    ax_bars.set_xticklabels(all_names, fontsize=8, rotation=0)

                    # Línia mitjana i banda ±2σ
                    ax_bars.axhline(y=hist_mean, color='red', linestyle='-', linewidth=2,
                                   label=f"Mitjana: {hist_mean:.2f}")
                    ax_bars.axhline(y=hist_mean + 2*hist_std, color='orange', linestyle='--', linewidth=1,
                                   label=f"+2σ: {hist_mean + 2*hist_std:.2f}")
                    ax_bars.axhline(y=hist_mean - 2*hist_std, color='orange', linestyle='--', linewidth=1,
                                   label=f"-2σ: {hist_mean - 2*hist_std:.2f}")

                    # Omplir banda ±2σ
                    ax_bars.axhspan(hist_mean - 2*hist_std, hist_mean + 2*hist_std,
                                   alpha=0.1, color='orange')

                    ax_bars.set_ylabel('Àrea KHP')
                    ax_bars.set_title('Àrea actual vs 5 SEQs més recents equivalents')
                    ax_bars.legend(loc='upper right', fontsize=8)

                    # Valor sobre cada barra
                    for bar, area in zip(bars, all_areas):
                        ax_bars.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                    f'{area:.1f}', ha='center', va='bottom', fontsize=8)

                    # Taula estadístiques històriques
                    ax_stats = fig.add_axes([0.1, 0.28, 0.8, 0.22])
                    ax_stats.axis('off')

                    # Determinar estat
                    if abs(z_score) < 2:
                        estat = "DINS RANG NORMAL"
                        color_estat = '#155724'
                    elif abs(z_score) < 3:
                        estat = "REVISAR"
                        color_estat = '#856404'
                    else:
                        estat = "FORA DE RANG"
                        color_estat = '#721c24'

                    stats_data = [
                        ['Estadística', 'Valor', 'Descripció'],
                        ['Àrea actual', f"{current_area:.2f}", estat],
                        ['Mitjana històrica', f"{hist_mean:.2f}", f"n={hist_stats['n_seqs']} SEQs"],
                        ['Desv. estàndard', f"{hist_std:.2f}", f"RSD: {hist_stats['rsd_area']:.1f}%"],
                        ['Desviació', f"{deviation_pct:+.1f}%", f"z-score: {z_score:.2f}"],
                        ['Rang històric', f"{hist_stats['min_area']:.2f} - {hist_stats['max_area']:.2f}", "Min - Max"],
                    ]

                    # Afegir temps de retenció si disponible
                    if hist_stats.get('mean_t_retention', 0) > 0:
                        current_t = khp_data.get('t_retention', khp_data.get('t_doc_max', 0))
                        stats_data.append([
                            'Temps retenció',
                            f"{current_t:.2f} min",
                            f"Mitjana: {hist_stats['mean_t_retention']:.2f} min"
                        ])

                    # Afegir A254 ratio si disponible
                    current_a254_ratio = khp_data.get('a254_doc_ratio', 0)
                    hist_a254_ratio = hist_stats.get('mean_a254_ratio', 0)
                    if current_a254_ratio > 0 and hist_a254_ratio > 0:
                        a254_deviation = (current_a254_ratio - hist_a254_ratio) / hist_a254_ratio * 100
                        a254_status = "OK" if abs(a254_deviation) < 15 else "REVISAR"
                        stats_data.append([
                            'Ratio DOC/A254',
                            f"{current_a254_ratio:.3f}",
                            f"Mitjana: {hist_a254_ratio:.3f} ({a254_status})"
                        ])

                    tbl_hist = ax_stats.table(cellText=stats_data, loc='center', cellLoc='center',
                                              colWidths=[0.35, 0.30, 0.35])
                    tbl_hist.auto_set_font_size(False)
                    tbl_hist.set_fontsize(9)
                    tbl_hist.scale(1.2, 1.8)

                    # Header style
                    for j in range(3):
                        tbl_hist[(0, j)].set_facecolor('#2E86AB')
                        tbl_hist[(0, j)].set_text_props(color='white', fontweight='bold')

                    # Color de l'estat
                    tbl_hist[(1, 2)].set_text_props(color=color_estat, fontweight='bold')

                    # Missatge de conclusió
                    if abs(z_score) < 2:
                        msg = f"La calibració actual ({current_area:.2f}) està dins del rang esperat (±2σ de la mitjana històrica)."
                        bg_color = '#d4edda'
                    elif abs(z_score) < 3:
                        msg = f"La calibració actual ({current_area:.2f}) s'allunya {abs(deviation_pct):.1f}% de la mitjana. Revisar si és acceptable."
                        bg_color = '#fff3cd'
                    else:
                        msg = f"ATENCIÓ: La calibració actual ({current_area:.2f}) està fora del rang normal (z={z_score:.1f})."
                        bg_color = '#f8d7da'

                    fig.text(0.5, 0.20, msg, ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=color_estat, linewidth=2))

                    # Peu de pàgina
                    fig.text(0.5, 0.05, f"Dades de {hist_stats['n_seqs']} calibracions equivalents (Mode: {mode_str})",
                            ha='center', fontsize=8, color='#888888')

                    pdf.savefig(fig, facecolor='white')
                    plt.close(fig)

            print(f"PDF calibració generat: {pdf_path}")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error generant PDF calibració: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Error PDF",
                f"Error generant PDF de calibració:\n{str(e)}"))

    def _update_cal_status(self, text):
        """Actualitza l'estat de calibració a la GUI."""
        if hasattr(self, 'lbl_cal_status'):
            self.lbl_cal_status.configure(text=text)

    def _update_cal_progress(self, progress):
        """Actualitza el progrés de calibració."""
        if hasattr(self, 'progress_cal'):
            self.progress_cal['value'] = progress

    def _load_existing_calibration(self, cal_entry):
        """
        Carrega i mostra una calibració existent des de CALDATA.

        Args:
            cal_entry: Diccionari amb les dades de calibració
        """
        try:
            # Intentar re-llegir el fitxer KHP per obtenir dades completes (cromatograma)
            khp_file = cal_entry.get("khp_file", "")
            khp_full_data = None

            # Buscar el fitxer KHP a Resultats_Consolidats
            if khp_file and khp_file != "N/A":
                res_cons = os.path.join(self.seq_path, "Resultats_Consolidats")
                khp_files = find_khp_in_folder(res_cons) if os.path.exists(res_cons) else []

                if khp_files:
                    # Re-analitzar per obtenir t_doc, y_doc, peak_info
                    khp_full_data = analizar_khp_lote(khp_files, self.config)

            # Combinar dades: CALDATA + dades completes si disponibles
            if khp_full_data:
                # Usar dades completes del KHP (amb cromatograma)
                combined_data = khp_full_data.copy()
                # Però mantenir alguns camps de CALDATA (outlier, active, etc.)
                combined_data['is_outlier'] = cal_entry.get('is_outlier', False)
                combined_data['is_active'] = cal_entry.get('is_active', True)
                combined_data['cal_id'] = cal_entry.get('cal_id', '')
            else:
                # Només dades de CALDATA (sense cromatograma)
                combined_data = cal_entry

            # Guardar dades per referència
            self.calibration_data = {
                "khp_data": combined_data,
                "source": cal_entry.get("khp_source", "CALDATA"),
                "apply_alignment": False,
                "from_caldata": True,
            }

            # Extreure dades
            mode = cal_entry.get("mode", "COLUMN")
            doc_mode = cal_entry.get("doc_mode", "N/A")
            conc_ppm = cal_entry.get("conc_ppm", 0)
            area = cal_entry.get("area", 0)
            volume_uL = cal_entry.get("volume_uL", 0)
            t_retention = cal_entry.get("t_retention", 0)
            symmetry = cal_entry.get("symmetry", 1.0)
            snr = cal_entry.get("snr", 0)
            shift_sec = cal_entry.get("shift_sec", cal_entry.get("shift_min", 0) * 60)
            date_processed = cal_entry.get("date_processed", cal_entry.get("date", ""))[:10]
            cal_id = cal_entry.get("cal_id", "N/A")

            # Obtenir estadístiques històriques
            hist_stats = get_equivalent_seqs_stats(
                self.seq_path, mode, doc_mode, conc_ppm, volume_uL, exclude_current=True
            )
            self.calibration_data['hist_stats'] = hist_stats

            # Comparació amb històric
            if hist_stats and hist_stats['n_seqs'] > 0:
                mean_hist = hist_stats['mean_area']
                diff_pct = ((area - mean_hist) / mean_hist * 100) if mean_hist > 0 else 0
                diff_str = f"({diff_pct:+.1f}% vs històric)"
                hist_info = f" | Històric: {mean_hist:.1f}±{hist_stats['std_area']:.1f} (n={hist_stats['n_seqs']})"
            else:
                diff_str = ""
                hist_info = ""

            # === ACTUALITZAR GUI ===

            # Detectar si és outlier
            is_outlier = cal_entry.get('is_outlier', False)
            outlier_source = cal_entry.get('outlier_source_seq', '')

            if is_outlier and outlier_source:
                # === MOSTRAR WARNING OUTLIER ===
                self.calibration_data['is_outlier'] = True
                self.calibration_data['source_seq'] = outlier_source

                self.lbl_cal_khp.configure(
                    text=f"⚠️ OUTLIER - Calibració de: {outlier_source}",
                    fg='#dc3545'  # Vermell
                )
                self.lbl_cal_factor.configure(
                    text=f"Àrea: {area:.2f} | Conc: {conc_ppm} ppm | (calibració externa)",
                    fg='#856404'  # Taronja
                )
                self.lbl_cal_r2.configure(
                    text=f"KHP LOCAL DESCARTAT | Data: {date_processed}",
                    fg='#856404'
                )
            else:
                # === CALIBRACIÓ NORMAL ===
                conditions_str = f"{mode} | {doc_mode} | {volume_uL}µL | {conc_ppm}ppm"
                self.lbl_cal_khp.configure(
                    text=f"✓ CALIBRAT: {cal_entry.get('khp_file', 'N/A')} | {conditions_str}"
                )
                self.lbl_cal_khp.configure(fg=COLORS["success"])

                # Línia 2: Resultats
                self.lbl_cal_factor.configure(
                    text=f"ÀREA: {area:.2f} {diff_str} | t_max: {t_retention:.2f} min{hist_info}"
                )

                # Línia 3: Detalls
                status_text = f"SHIFT: {shift_sec:.1f}s | SNR: {snr:.1f} | Sim: {symmetry:.2f} | Data: {date_processed}"
                self.lbl_cal_r2.configure(text=status_text)
                self.lbl_cal_r2.configure(fg=COLORS["dark"])

            # Mostrar botó outlier
            if hasattr(self, 'btn_outlier'):
                self.btn_outlier.pack(side=tk.LEFT, padx=10)

            # Habilitar històric i processar
            if hasattr(self, 'btn_history'):
                self.btn_history.configure(state="normal")
            self.btn_processar.configure(state="normal")

            # Dibuixar gràfic si tenim dades
            self._draw_calibration_plot()

        except Exception as e:
            print(f"Error carregant calibració existent: {e}")
            import traceback
            traceback.print_exc()
            self.lbl_cal_khp.configure(text=f"Error carregant calibració: {e}")

    def _calibration_done(self):
        """Calibració completada."""
        self.is_processing = False
        self.btn_calibrar.configure(state="normal", text="Recalibrar")
        self.btn_processar.configure(state="normal")

        cal = self.calibration_data
        khp = cal.get("khp_data", {})

        # Info més detallada a la GUI
        source_info = cal.get('source', 'N/A')
        quality_score = khp.get('quality_score', 0)
        symmetry = khp.get('symmetry', 0)
        snr = khp.get('snr', 0)
        current_area = khp.get('area', 0)
        conc_ppm = khp.get('conc_ppm', 0)
        doc_mode = khp.get('doc_mode', 'N/A')  # UIB o DIRECT

        # Obtenir estadístiques de SEQs equivalents (mateix mode, source, volum, concentració)
        is_bp = khp.get('is_bp', False)
        mode = "BP" if is_bp else "COLUMN"
        volume_uL = get_injection_volume(self.seq_path, is_bp)
        hist_stats = get_equivalent_seqs_stats(self.seq_path, mode, doc_mode, conc_ppm, volume_uL, exclude_current=True)

        # Guardar stats per al gràfic
        self.calibration_data['hist_stats'] = hist_stats

        # Mostrar comparació amb històric
        if hist_stats and hist_stats['n_seqs'] > 0:
            mean_hist = hist_stats['mean_area']
            diff_pct = ((current_area - mean_hist) / mean_hist * 100) if mean_hist > 0 else 0
            diff_str = f"({diff_pct:+.1f}% vs històric)"
            source_str = f" [{doc_mode}]" if doc_mode and doc_mode != "N/A" else ""
            hist_info = f" | Històric{source_str}: {mean_hist:.1f}±{hist_stats['std_area']:.1f} (n={hist_stats['n_seqs']})"
        else:
            diff_str = "(sense referència)"
            hist_info = ""

        t_retention = khp.get('t_retention', khp.get('t_doc_max', 0))

        # === LÍNIA 1: Identificació i condicions ===
        # Mode (BP/COLUMN) + Source (UIB/DIRECT) + Volum + Concentració
        conditions_str = f"{mode} | {doc_mode} | {volume_uL}µL | {conc_ppm}ppm"
        self.lbl_cal_khp.configure(text=f"KHP: {khp.get('filename', 'N/A')} | {conditions_str}")

        # === LÍNIA 2: Resultats principals ===
        # Àrea + t_max + comparació històric
        self.lbl_cal_factor.configure(text=f"ÀREA: {current_area:.2f} {diff_str} | t_max: {t_retention:.2f} min{hist_info}")

        shift_sec = khp.get('shift_min', 0) * 60
        align_str = "SÍ" if cal.get("apply_alignment") else "NO"

        # Afegir alertes de qualitat
        quality_issues = khp.get('quality_issues', [])
        baseline_valid = khp.get('baseline_valid', True)

        # === LÍNIA 3: Qualitat i alertes ===
        status_text = f"SHIFT: {shift_sec:.1f}s | Alineació: {align_str} | SNR: {snr:.1f} | Sim: {symmetry:.2f}"

        # Mostrar alertes si n'hi ha
        if quality_issues:
            alert_str = " | ⚠️ " + ", ".join(quality_issues[:2])  # Màxim 2 alertes
            status_text += alert_str

        self.lbl_cal_r2.configure(text=status_text)

        # Canviar color si hi ha problemes o correccions
        limits_expanded = khp.get('limits_expanded', False)
        if khp.get('has_batman') or khp.get('has_timeout'):
            self.lbl_cal_r2.configure(fg=COLORS["warning"])
        elif limits_expanded:
            self.lbl_cal_r2.configure(fg=COLORS["primary"])  # Blau = corregit
        else:
            self.lbl_cal_r2.configure(fg=COLORS["dark"])

        # Dibuixar gràfic
        self._draw_calibration_plot()

        # Registrar calibració a l'històric
        register_calibration(self.seq_path, khp, source_info, mode)

        # Actualitzar botons
        if hasattr(self, 'btn_history'):
            self.btn_history.configure(state="normal")

        # Mostrar botó per marcar com outlier
        if hasattr(self, 'btn_outlier'):
            self.btn_outlier.pack(side=tk.LEFT, padx=10)

        # Obrir carpeta CHECK automàticament
        check_folder = os.path.join(self.seq_path, "CHECK")
        if os.path.exists(check_folder):
            try:
                os.startfile(check_folder)
            except:
                pass

        # Missatge simple - tota la info ja està a la GUI
        messagebox.showinfo("Calibració", f"Calibració completada.\n\nPDFs generats a la carpeta CHECK\n(s'ha obert automàticament)")

    def _calibration_error(self, error):
        """Error en calibració."""
        self.is_processing = False
        self.btn_calibrar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant la calibració:\n{error}")

    def _mark_current_as_outlier(self):
        """Marca la calibració actual com a outlier."""
        if not self.calibration_data or not self.seq_path:
            messagebox.showerror("Error", "No hi ha calibració activa.")
            return

        seq_name = os.path.basename(self.seq_path)
        khp = self.calibration_data.get("khp_data", {})
        is_bp = khp.get('is_bp', False)
        mode = "BP" if is_bp else "COLUMN"

        # Confirmar
        if not messagebox.askyesno("Confirmar Outlier",
                                   f"Marcar la calibració actual com a OUTLIER?\n\n"
                                   f"SEQ: {seq_name}\n"
                                   f"Mode: {mode}\n"
                                   f"Àrea: {khp.get('area', 0):.2f}\n\n"
                                   f"Això exclourà aquesta calibració de les\n"
                                   f"estadístiques i comparacions futures."):
            return

        # Marcar com outlier
        if mark_calibration_as_outlier(self.seq_path, seq_name, mode, is_outlier=True):
            messagebox.showinfo("Outlier",
                               f"Calibració marcada com a OUTLIER.\n\n"
                               f"Serà exclosa de futures comparacions.\n"
                               f"Pots desmarcar-la des de l'Històric KHP.")

            # Actualitzar gràfic (recalcular estadístiques sense l'outlier)
            self._update_calibration_gui()

            # Amagar botó outlier
            self.btn_outlier.pack_forget()
        else:
            messagebox.showerror("Error", "No s'ha pogut marcar com a outlier.")

    def _show_khp_history(self):
        """Mostra el diàleg d'històric de calibracions KHP."""
        KHPHistoryDialog(self.root, self.seq_path, self._apply_historical_khp)

    def _apply_historical_khp(self, khp_entry):
        """
        Aplica un KHP de l'històric a la seqüència actual.

        Args:
            khp_entry: Diccionari amb les dades del KHP seleccionat
        """
        if not khp_entry:
            return

        if not self.consolidated_data or not self.seq_path:
            messagebox.showerror("Error", "Cal tenir dades consolidats primer.")
            return

        # Confirmar
        msg = (f"Aplicar KHP de: {khp_entry.get('seq_name', 'N/A')}\n"
               f"Mode: {khp_entry.get('mode', 'N/A')}\n"
               f"Àrea: {khp_entry.get('area', 0):.2f}\n"
               f"Conc: {khp_entry.get('conc_ppm', 0)} ppm\n\n"
               f"Vols aplicar aquest KHP a la seqüència actual?")

        if not messagebox.askyesno("Aplicar KHP Històric", msg):
            return

        # SEQ font de la calibració
        source_seq_name = khp_entry.get('seq_name', 'UNKNOWN')
        source_seq_path = khp_entry.get('seq_path', '')

        # Crear khp_data compatible
        khp_data = {
            'filename': khp_entry.get('khp_file', 'HISTORIC'),
            'conc_ppm': khp_entry.get('conc_ppm', 0),
            'area': khp_entry.get('area', 0),
            'shift_min': khp_entry.get('shift_min', 0),
            'shift_sec': khp_entry.get('shift_sec', 0),
            'symmetry': khp_entry.get('symmetry', 1.0),
            'snr': khp_entry.get('snr', 0),
            'status': 'HISTORIC',
            'quality_issues': ['Utilitzat de l\'històric'],
            'source_seq': source_seq_name,  # SEQ d'on prové la calibració
            'source_seq_path': source_seq_path,
        }

        # Determinar si aplicar alineació
        threshold_min = self.config["alignment_threshold_sec"] / 60.0
        apply_alignment = abs(khp_data['shift_min']) > threshold_min

        # Actualitzar fitxers
        consolidat_path = self.consolidated_data.get("path", "")
        xlsx_files = glob.glob(os.path.join(consolidat_path, "*.xlsx"))
        xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

        updated_count = 0
        for fpath in xlsx_files:
            try:
                if self._update_file_with_khp(fpath, khp_data, apply_alignment):
                    updated_count += 1
            except Exception as e:
                print(f"Error actualitzant {os.path.basename(fpath)}: {e}")

        # Guardar dades de calibració
        self.calibration_data = {
            "khp_data": khp_data,
            "source": f"HISTORIC: {source_seq_name}",
            "source_seq": source_seq_name,
            "source_seq_path": source_seq_path,
            "is_outlier": True,  # Marcar que aquesta SEQ és outlier
            "apply_alignment": apply_alignment,
            "updated_files": updated_count,
            "total_files": len(xlsx_files),
        }

        # Registrar a CALDATA (marcant com outlier amb font externa)
        try:
            khp_for_caldata = khp_data.copy()
            khp_for_caldata['is_outlier'] = True
            khp_for_caldata['outlier_source_seq'] = source_seq_name

            is_bp = khp_entry.get('mode', 'COLUMN') == 'BP'
            mode_str = 'BP' if is_bp else 'COLUMN'

            register_calibration(
                self.seq_path,
                khp_for_caldata,
                f"HISTORIC:{source_seq_name}",
                mode_str
            )
        except Exception as e:
            print(f"WARN: Error registrant calibració a CALDATA: {e}")

        # Copiar PDF de calibració de la SEQ font
        self._copy_calibration_pdf_from_source(source_seq_path, source_seq_name)

        # Actualitzar GUI
        self.btn_processar.configure(state="normal")
        self._calibration_done_from_history()

    def _copy_calibration_pdf_from_source(self, source_seq_path, source_seq_name):
        """Copia el PDF de calibració de la SEQ font a la SEQ actual."""
        import shutil

        if not source_seq_path or not os.path.exists(source_seq_path):
            print(f"WARN: No es pot accedir a la SEQ font: {source_seq_path}")
            return

        # Buscar PDF de calibració a la SEQ font
        source_check = os.path.join(source_seq_path, "CHECK")
        source_pdf = None

        if os.path.exists(source_check):
            for f in os.listdir(source_check):
                if f.startswith("REPORT_Calibracio") and f.endswith(".pdf"):
                    source_pdf = os.path.join(source_check, f)
                    break

        if not source_pdf or not os.path.exists(source_pdf):
            print(f"WARN: No s'ha trobat PDF de calibració a {source_seq_name}")
            return

        # Copiar a la SEQ actual
        try:
            dest_check = ensure_check_folder(self.seq_path)
            current_seq_name = obtenir_seq(self.seq_path)

            # Nom del PDF destí amb indicació que és d'una altra SEQ
            dest_pdf = os.path.join(dest_check, f"REPORT_Calibracio_{current_seq_name}_FROM_{source_seq_name}.pdf")

            shutil.copy2(source_pdf, dest_pdf)
            print(f"PDF calibració copiat de {source_seq_name}")

            # Crear també un fitxer de text indicant l'outlier
            outlier_info_path = os.path.join(dest_check, f"OUTLIER_INFO_{current_seq_name}.txt")
            with open(outlier_info_path, 'w', encoding='utf-8') as f:
                f.write(f"SEQÜÈNCIA MARCADA COM OUTLIER\n")
                f.write(f"{'='*40}\n\n")
                f.write(f"SEQ Actual: {current_seq_name}\n")
                f.write(f"El KHP d'aquesta seqüència no és vàlid.\n\n")
                f.write(f"Calibració aplicada de: {source_seq_name}\n")
                f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"\nConsulteu el PDF: REPORT_Calibracio_{current_seq_name}_FROM_{source_seq_name}.pdf\n")

        except Exception as e:
            print(f"WARN: Error copiant PDF: {e}")

    def _calibration_done_from_history(self):
        """Actualitza la GUI després d'aplicar KHP de l'històric."""
        cal = self.calibration_data
        khp = cal.get("khp_data", {})

        source_seq = cal.get('source_seq', 'N/A')
        symmetry = khp.get('symmetry', 0)
        snr = khp.get('snr', 0)

        # Mostrar WARNING OUTLIER en vermell/taronja
        self.lbl_cal_khp.configure(
            text=f"⚠️ OUTLIER - Calibració de: {source_seq}",
            fg='#dc3545'  # Vermell
        )
        self.lbl_cal_factor.configure(
            text=f"Àrea: {khp.get('area', 0):.2f} | Conc: {khp.get('conc_ppm', 0)} ppm | (extern)",
            fg='#856404'  # Taronja fosc
        )

        shift_sec = khp.get('shift_min', 0) * 60
        align_str = "SÍ" if cal.get("apply_alignment") else "NO"
        self.lbl_cal_r2.configure(
            text=f"SHIFT: {shift_sec:.1f}s | Alineació: {align_str} | KHP LOCAL DESCARTAT",
            fg='#856404'
        )

        # Dibuixar gràfic amb missatge d'outlier
        self._draw_calibration_plot()

        # Missatge informatiu
        messagebox.showwarning("⚠️ OUTLIER - Calibració Externa",
                              f"Aquesta seqüència s'ha marcat com OUTLIER.\n\n"
                              f"El KHP local no és vàlid.\n\n"
                              f"S'ha aplicat la calibració de:\n"
                              f"  → {source_seq}\n\n"
                              f"Fitxers actualitzats: {cal.get('updated_files', 0)}/{cal.get('total_files', 0)}\n\n"
                              f"Consulteu la carpeta CHECK per més detalls.")

    def _draw_calibration_plot(self):
        """Dibuixa el gràfic de calibració amb ZOOM als pics KHP (tots els replicats)."""
        # Netejar frame
        for widget in self.cal_canvas_frame.winfo_children():
            widget.destroy()

        if not self.calibration_data:
            return

        khp = self.calibration_data.get("khp_data", {})
        if not khp:
            return

        # Si és calibració externa (outlier), mostrar missatge informatiu
        if self.calibration_data.get('is_outlier', False):
            source_seq = self.calibration_data.get('source_seq', 'N/A')

            # Crear un frame amb missatge
            msg_frame = tk.Frame(self.cal_canvas_frame, bg='#fff3cd', relief='ridge', bd=2)
            msg_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            tk.Label(msg_frame,
                    text="⚠️ SEQÜÈNCIA MARCADA COM OUTLIER",
                    font=("Segoe UI", 14, "bold"),
                    bg='#fff3cd', fg='#856404').pack(pady=(20, 10))

            tk.Label(msg_frame,
                    text=f"El KHP d'aquesta seqüència no és vàlid.",
                    font=("Segoe UI", 11),
                    bg='#fff3cd', fg='#856404').pack(pady=5)

            tk.Label(msg_frame,
                    text=f"Calibració aplicada de: {source_seq}",
                    font=("Segoe UI", 12, "bold"),
                    bg='#fff3cd', fg='#155724').pack(pady=10)

            tk.Label(msg_frame,
                    text=f"Consulteu el PDF a la carpeta CHECK per veure els gràfics de calibració.",
                    font=("Segoe UI", 10, "italic"),
                    bg='#fff3cd', fg='#666666').pack(pady=(10, 20))

            return

        # Obtenir tots els replicats si disponibles
        replicas = khp.get('replicas', khp.get('all_khp_data', []))
        if not replicas:
            # Si no hi ha replicas, usar les dades principals com a única rèplica
            replicas = [khp]

        n_replicas = len(replicas)

        # Layout: ZOOM de cada replicat + gràfic comparació històric
        n_cols = min(n_replicas, 2) + 1  # Màxim 2 replicats + 1 comparació
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))
        if n_cols == 1:
            axes = [axes]

        # Colors per replicats
        rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Gràfics ZOOM per cada replicat (màxim 2)
        for i, rep in enumerate(replicas[:2]):
            ax = axes[i]

            t_doc = rep.get('t_doc', [])
            y_doc = rep.get('y_doc', [])
            peak_info = rep.get('peak_info', {})

            # Convertir a numpy
            if t_doc is not None and len(t_doc) > 0:
                t_doc = np.asarray(t_doc).flatten()
                y_doc = np.asarray(y_doc).flatten() if y_doc is not None else np.array([])
            else:
                t_doc = np.array([])
                y_doc = np.array([])

            if len(t_doc) > 10 and len(y_doc) > 10 and peak_info.get('valid'):
                peak_t = peak_info.get('t_max', 0)
                t_start = peak_info.get('t_start', peak_t - 2)
                t_end = peak_info.get('t_end', peak_t + 2)

                # Finestra de zoom: ±3 min al voltant del pic
                zoom_start = max(0, peak_t - 3)
                zoom_end = min(float(np.max(t_doc)), peak_t + 3)

                mask_zoom = (t_doc >= zoom_start) & (t_doc <= zoom_end)
                if np.any(mask_zoom):
                    t_zoom = t_doc[mask_zoom]
                    y_zoom = y_doc[mask_zoom]

                    # Dibuixar senyal
                    ax.plot(t_zoom, y_zoom, color=rep_colors[i], linewidth=2)

                    # Ombrar àrea d'integració
                    left_idx = peak_info.get('left_idx', 0)
                    right_idx = peak_info.get('right_idx', len(t_doc) - 1)
                    left_idx = int(max(0, min(left_idx, len(t_doc) - 1)))
                    right_idx = int(max(0, min(right_idx, len(t_doc) - 1)))

                    zoom_left_idx = int(np.searchsorted(t_doc, zoom_start))
                    zoom_right_idx = int(np.searchsorted(t_doc, zoom_end))
                    int_left = max(left_idx, zoom_left_idx)
                    int_right = min(right_idx, zoom_right_idx)
                    int_left = max(0, min(int_left, len(t_doc) - 1))
                    int_right = max(0, min(int_right, len(t_doc) - 1))

                    if int_right > int_left:
                        ax.fill_between(t_doc[int_left:int_right+1], 0,
                                       y_doc[int_left:int_right+1],
                                       alpha=0.4, color='green')

                    # Línies límits integració (corregits)
                    if t_start > 0:
                        ax.axvline(x=t_start, color='green', linestyle='-', linewidth=2, alpha=0.9)
                    if t_end > 0:
                        ax.axvline(x=t_end, color='green', linestyle='-', linewidth=2, alpha=0.9)

                    # Si s'han expandit els límits, mostrar els originals en vermell puntejat
                    expansion_info = rep.get('expansion_info', peak_info.get('expansion_info', {}))
                    if expansion_info and not expansion_info.get('original_valid', True):
                        orig_left = expansion_info.get('original_left', 0)
                        orig_right = expansion_info.get('original_right', len(t_doc)-1)
                        if 0 <= orig_left < len(t_doc):
                            t_orig_left = t_doc[orig_left]
                            if zoom_start <= t_orig_left <= zoom_end:
                                ax.axvline(x=t_orig_left, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
                        if 0 <= orig_right < len(t_doc):
                            t_orig_right = t_doc[orig_right]
                            if zoom_start <= t_orig_right <= zoom_end:
                                ax.axvline(x=t_orig_right, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

                    # Marcar pic
                    ax.axvline(x=peak_t, color='red', linestyle='--', linewidth=2)
                    peak_h = peak_info.get('height', float(np.max(y_zoom)))
                    ax.scatter([peak_t], [peak_h], color='red', s=80, zorder=5, marker='v')

            # Info del replicat
            rep_area = rep.get('area', peak_info.get('area', 0))
            rep_sym = rep.get('symmetry', 1.0)
            rep_name = rep.get('filename', f'R{i+1}')

            # Afegir indicador si límits expandits
            limits_exp = rep.get('limits_expanded', False)
            exp_marker = " [EXP]" if limits_exp else ""

            ax.set_xlabel("Temps (min)", fontsize=8)
            ax.set_ylabel("DOC (mAU)", fontsize=8)
            ax.set_title(f"{rep_name}{exp_marker}\nÀrea: {rep_area:.1f} | Sim: {rep_sym:.2f}", fontsize=8, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Últim gràfic: Comparació amb històric (barres amb noms SEQ)
        ax_hist = axes[-1]  # Últim eix
        current_area = khp.get('area', 0)
        current_seq_name = self.calibration_data.get('khp_data', {}).get('filename', 'ACTUAL')
        # Simplificar nom per mostrar
        if current_seq_name and len(current_seq_name) > 15:
            current_seq_name = current_seq_name[:12] + "..."
        hist_stats = self.calibration_data.get('hist_stats')

        if hist_stats and hist_stats.get('n_seqs', 0) > 0:
            # Obtenir les 5 SEQs més recents
            closest_seqs = hist_stats.get('closest_seqs', [])
            mean_hist = hist_stats['mean_area']
            std_hist = hist_stats['std_area']

            # Preparar dades per barres
            seq_names = []
            seq_areas = []
            bar_colors = []

            # Afegir SEQs històriques (màxim 5)
            for seq_info in closest_seqs[:5]:
                # Format: (seq_name, area, date, doc_mode, t_retention)
                seq_name = seq_info[0] if len(seq_info) > 0 else "?"
                seq_area = seq_info[1] if len(seq_info) > 1 else 0
                seq_date = seq_info[2] if len(seq_info) > 2 else ""
                # seq_doc_mode = seq_info[3] if len(seq_info) > 3 else "N/A"
                # seq_t_ret = seq_info[4] if len(seq_info) > 4 else 0

                # Extreure codi identificador de la SEQ (ex: "256B" de "SEQ_256B_2024")
                short_name = "?"
                if seq_name:
                    # Buscar patró numèric amb possible lletra (256B, 123, 45A, etc.)
                    import re
                    match = re.search(r'(\d+[A-Za-z]?)', seq_name)
                    if match:
                        short_name = match.group(1)
                    else:
                        # Si no hi ha patró, agafar última part després de _
                        short_name = seq_name.split("_")[-1] if "_" in seq_name else seq_name
                        if len(short_name) > 8:
                            short_name = short_name[:8]
                elif seq_date:
                    short_name = seq_date[:10]
                seq_names.append(short_name)
                seq_areas.append(seq_area)
                bar_colors.append('#a0a0a0')  # Gris per històric

            # Afegir barra actual al final (destacada)
            seq_names.append("ACTUAL")
            seq_areas.append(current_area)
            # Color verd si dins ±2σ, taronja si fora
            if abs(current_area - mean_hist) <= 2 * std_hist:
                bar_colors.append(COLORS["success"])
            else:
                bar_colors.append(COLORS["warning"])

            # Dibuixar barres
            x_pos = range(len(seq_names))
            bars = ax_hist.bar(x_pos, seq_areas, color=bar_colors, edgecolor='black', linewidth=0.5)

            # Línia de mitjana
            ax_hist.axhline(y=mean_hist, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Mitjana: {mean_hist:.1f}')

            # Banda ±2σ
            ax_hist.axhspan(mean_hist - 2*std_hist, mean_hist + 2*std_hist, alpha=0.15, color='blue')

            # Etiquetes de valor a cada barra
            for bar, area in zip(bars, seq_areas):
                ax_hist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{area:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

            # Configurar eixos
            ax_hist.set_xticks(x_pos)
            ax_hist.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=7)
            ax_hist.set_ylabel("Àrea KHP", fontsize=8)

            # Títol amb info de filtre
            diff_pct = ((current_area - mean_hist) / mean_hist * 100) if mean_hist > 0 else 0
            filter_mode = hist_stats.get('mode', 'N/A')
            filter_source = hist_stats.get('doc_mode', '')
            filter_str = filter_mode
            if filter_source and filter_source != "N/A":
                filter_str += f"/{filter_source}"
            ax_hist.set_title(f"Comparació [{filter_str}] (n={hist_stats['n_seqs']}) | Δ={diff_pct:+.1f}%", fontsize=8, fontweight='bold')
            ax_hist.legend(loc='upper right', fontsize=7)
        else:
            # Sense històric - mostrar només mètriques actuals
            shift_sec = khp.get('shift_min', 0) * 60
            symmetry = khp.get('symmetry', 0)
            snr = khp.get('snr', 0)

            metrics = ['Àrea', 'Simetria', 'SNR/10']
            values = [current_area, symmetry, snr/10]
            colors_bar = [COLORS["primary"],
                         COLORS["success"] if 0.8 <= symmetry <= 1.5 else COLORS["warning"],
                         COLORS["success"] if snr > 10 else COLORS["warning"]]

            bars = ax_hist.barh(metrics, values, color=colors_bar, height=0.5)
            for bar, val in zip(bars, values):
                ax_hist.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                            va='center', fontsize=9, fontweight='bold')

            ax_hist.set_title(f"KHP {khp.get('conc_ppm', 0)} ppm (sense històric)", fontsize=9)

        ax_hist.grid(True, alpha=0.3, axis='y')

        # Indicador d'alineació a la part inferior
        align_text = "ALINEACIÓ APLICADA" if self.calibration_data.get("apply_alignment") else "Sense alineació"
        shift_sec = khp.get('shift_min', 0) * 60
        ax_hist.text(0.5, -0.18, f"{align_text} | SHIFT: {shift_sec:.1f}s", transform=ax_hist.transAxes, ha='center', fontsize=7,
                    fontweight='bold', color=COLORS["primary"] if self.calibration_data.get("apply_alignment") else 'gray')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.cal_canvas_frame)
        canvas.draw()
        # NO expand per evitar que empenti els botons fora de vista
        canvas.get_tk_widget().pack(fill=tk.X, expand=False)
        plt.close(fig)

    # =========================================================================
    # PAS 3: PROCESSAMENT
    # =========================================================================
    def _run_processing(self):
        """Executa el processament."""
        if self.is_processing:
            return

        self.is_processing = True
        self.btn_processar.configure(state="disabled")

        # Netejar taula
        for item in self.tree_process.get_children():
            self.tree_process.delete(item)

        thread = threading.Thread(target=self._processing_thread)
        thread.start()

    def _processing_thread(self):
        """Thread de processament amb selecció independent DOC/DAD."""
        try:
            consolidat_path = self.consolidated_data.get("path", "")
            xlsx_files = glob.glob(os.path.join(consolidat_path, "*.xlsx"))

            # Obtenir dades de calibració per convertir àrees a concentracions
            khp_area = 1.0
            khp_conc = 1.0
            if self.calibration_data:
                khp_data = self.calibration_data.get("khp_data", {})
                khp_area = khp_data.get("area", 1.0) or 1.0
                khp_conc = khp_data.get("conc_ppm", 1.0) or 1.0

            # Agrupar per mostra (sense _R1, _R2)
            samples = {}
            for f in xlsx_files:
                name = os.path.basename(f).replace(".xlsx", "")
                match = re.search(r"(.+)_R(\d+)$", name)
                if match:
                    base = match.group(1)
                    rep = match.group(2)
                    if base not in samples:
                        samples[base] = {}
                    samples[base][rep] = f

            total = len(samples)
            count = 0
            check_folder = ensure_check_folder(self.seq_path)

            for sample_name, reps in samples.items():
                count += 1
                progress = int(100 * count / total)

                # Avaluar cada rèplica (DOC)
                evals = {}
                for rep_id, filepath in reps.items():
                    evals[rep_id] = self._evaluate_replica(filepath)

                # Comparar DOC entre rèpliques (Pearson, ΔÀrea)
                doc_comparison = self._compare_doc_replicas(evals)

                # Avaluar qualitat DAD de cada rèplica (deriva, soroll)
                dad_quality = {}
                is_bp = False
                for rep_id, ev in evals.items():
                    if ev.get("valid"):
                        is_bp = ev.get("is_bp", False)
                        # Llegir DataFrame DAD directament del fitxer
                        df_dad = self._load_dad_dataframe(reps[rep_id])
                        dad_quality[rep_id] = avaluar_qualitat_dad_replica(
                            df_dad,
                            self.config["target_wavelengths"],
                            is_bp
                        )

                # Afegir concentració calibrada a cada rèplica
                for rep_id, ev in evals.items():
                    if ev.get("valid"):
                        area = ev.get("area", 0)
                        ev["conc_ppm"] = (area / khp_area) * khp_conc if khp_area > 0 else 0

                        # KPIs en % i en concentració
                        kpis = ev.get("kpis", {})
                        ev["kpis_ppm"] = {}
                        for key, val in kpis.items():
                            if not key.endswith("_pct") and key != "total":
                                ev["kpis_ppm"][key] = (val / khp_area) * khp_conc if khp_area > 0 else 0

                # Seleccionar millor DOC (basant-se en anomalies i qualitat)
                sel_doc, motiu_doc = self._select_best_doc(evals, doc_comparison)

                # Seleccionar millor DAD (basant-se en deriva i soroll)
                qual_r1 = dad_quality.get("1", {"valid": False})
                qual_r2 = dad_quality.get("2", {"valid": False})
                sel_dad, motiu_dad = seleccionar_millor_dad(qual_r1, qual_r2)

                # Determinar estat global
                doc_r = doc_comparison.get("pearson")
                doc_diff = doc_comparison.get("area_diff_pct")
                dad_status = dad_quality.get(sel_dad, {}).get("status", "NO_DATA")

                if doc_r is None or (doc_r < DEFAULT_MIN_CORR and doc_diff and doc_diff > DEFAULT_MAX_DIFF):
                    estat = "FAIL"
                    tag = "FAIL"
                elif doc_r < DEFAULT_MIN_CORR or (doc_diff and doc_diff > DEFAULT_MAX_DIFF) or dad_status == "WARN":
                    estat = "WARN"
                    tag = "WARN"
                else:
                    estat = "OK"
                    tag = "OK"

                # Anomalies DOC de la rèplica seleccionada
                sel_doc_eval = evals.get(sel_doc, {})
                if sel_doc_eval.get("has_timeout") or sel_doc_eval.get("has_batman"):
                    tag = "WARN"
                    estat = "WARN"

                # Valors per la taula: Mostra, Estat, DOC R, DOC Δ%, Sel DOC, DAD Qual, Sel DAD
                doc_r_str = f"{doc_r:.3f}" if doc_r is not None else "-"
                doc_diff_str = f"{doc_diff:.1f}%" if doc_diff is not None else "-"
                dad_qual_str = dad_status
                if dad_status == "WARN":
                    # Afegir info de quina WL falla
                    for rep_id, qual in dad_quality.items():
                        if qual.get("deriva_global", 0) > DAD_DERIVA_WARN:
                            dad_qual_str = f"⚠Der R{rep_id}"
                            break

                values = [sample_name, estat, doc_r_str, doc_diff_str,
                         f"R{sel_doc}", dad_qual_str, f"R{sel_dad}"]

                # Guardar selecció amb DOC i DAD independents
                self.selected_replicas[sample_name] = {
                    "sel_doc": sel_doc,
                    "sel_dad": sel_dad,
                    "reps": reps,
                    "all_evals": evals,
                    "doc_comparison": doc_comparison,
                    "dad_quality": dad_quality,
                    "eval": evals.get(sel_doc, {}),  # Compatibilitat
                }

                # Generar gràfic QAQC PNG (DOC + 6 WL DAD)
                self._generate_qaqc_plot(sample_name, self.selected_replicas[sample_name], check_folder)

                self.root.after(0, lambda v=values, t=tag, p=progress:
                               self._update_proc_progress(v, t, p))

            self.root.after(0, self._processing_done)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._processing_error(str(e)))

    def _load_dad_dataframe(self, filepath):
        """Carrega el DataFrame DAD d'un fitxer Excel."""
        try:
            xls = pd.ExcelFile(filepath)
            if 'DAD' in xls.sheet_names:
                return pd.read_excel(xls, 'DAD')
        except Exception:
            pass
        return pd.DataFrame()

    def _compare_doc_replicas(self, evals):
        """Compara DOC entre rèpliques amb Pearson i diferència d'àrea."""
        comparison = {
            "pearson": None,
            "area_diff_pct": None,
            "shift": 0,
            "n_valid": 0,
        }

        valid_evals = {k: v for k, v in evals.items() if v.get("valid")}
        comparison["n_valid"] = len(valid_evals)

        if len(valid_evals) < 2:
            return comparison

        rep_ids = sorted(valid_evals.keys())

        t1 = np.asarray(valid_evals[rep_ids[0]].get("t", []), dtype=float)
        y1 = np.asarray(valid_evals[rep_ids[0]].get("y", []), dtype=float)
        t2 = np.asarray(valid_evals[rep_ids[1]].get("t", []), dtype=float)
        y2 = np.asarray(valid_evals[rep_ids[1]].get("y", []), dtype=float)

        # Determinar rang d'anàlisi DOC
        is_bp = valid_evals[rep_ids[0]].get("is_bp", False)
        if is_bp:
            doc_min, doc_max = 2.0, 15.0
        else:
            doc_min, doc_max = 15.0, 35.0

        # Alinear i calcular estadístiques
        if len(t1) > 10 and len(t2) > 10:
            try:
                t_aligned, y1_aligned, y2_aligned, shift = alinear_senyals(
                    t1, y1, t2, y2, doc_min, doc_max, threshold=0.0
                )
                if t_aligned is not None:
                    corr, diff = calcular_estadistiques(t_aligned, y1_aligned, y2_aligned)
                    comparison["pearson"] = float(corr)
                    comparison["area_diff_pct"] = float(diff)
                    comparison["shift"] = float(shift)
            except Exception as e:
                print(f"Error comparant DOC: {e}")

        return comparison

    def _select_best_doc(self, evals, doc_comparison):
        """Selecciona la millor rèplica per DOC basant-se en anomalies i qualitat."""
        valid = {k: v for k, v in evals.items() if v.get("valid", False)}

        if not valid:
            return "1", "Cap vàlida"

        if len(valid) == 1:
            rep = list(valid.keys())[0]
            return rep, "Única"

        rep_ids = sorted(valid.keys())

        # Prioritzar per anomalies
        ev1 = valid[rep_ids[0]]
        ev2 = valid[rep_ids[1]]

        anom1 = ev1.get("has_timeout", False) or ev1.get("has_batman", False)
        anom2 = ev2.get("has_timeout", False) or ev2.get("has_batman", False)

        if anom1 and not anom2:
            return rep_ids[1], "R1 anomalia"
        if anom2 and not anom1:
            return rep_ids[0], "R2 anomalia"

        # Si cap o ambdues tenen anomalies, mirar qualitat
        score1 = avaluar_qualitat_doc(np.asarray(ev1.get("y", []), dtype=float))
        score2 = avaluar_qualitat_doc(np.asarray(ev2.get("y", []), dtype=float))

        if score1 < score2:
            return rep_ids[0], "Millor qualitat"
        elif score2 < score1:
            return rep_ids[1], "Millor qualitat"

        return rep_ids[0], "Similar"

    def _generate_qaqc_plot(self, sample_name, data, output_folder):
        """
        Genera gràfic QAQC amb DOC + 6 wavelengths DAD.
        Mostra estadístiques de comparació DOC (Pearson, ΔÀrea) i
        qualitat DAD per cada rèplica (deriva, soroll).
        """
        try:
            os.makedirs(output_folder, exist_ok=True)

            all_evals = data.get("all_evals", {})
            sel_doc = data.get("sel_doc", "1")
            sel_dad = data.get("sel_dad", "1")
            doc_comparison = data.get("doc_comparison", {})
            dad_quality = data.get("dad_quality", {})
            reps = data.get("reps", {})

            # Determinar si és BP
            is_bp = False
            for ev in all_evals.values():
                if ev.get("valid"):
                    is_bp = ev.get("is_bp", False)
                    break
            x_max = 20 if is_bp else 70

            # Wavelengths a mostrar
            wavelengths = self.config.get("target_wavelengths", [220, 252, 254, 272, 290, 362])

            # Crear figura: 1 fila DOC + 6 files DAD = 7 files
            n_wl = len(wavelengths)
            fig_height = 3 + 2 * n_wl  # DOC més gran, DAD més petits
            fig, axes = plt.subplots(1 + n_wl, 1, figsize=(14, fig_height),
                                    gridspec_kw={'height_ratios': [2] + [1]*n_wl})
            plt.subplots_adjust(hspace=0.35, left=0.08, right=0.92, top=0.95, bottom=0.05)

            colors = {"1": "#2E86AB", "2": "#E74C3C"}
            zone_colors = ["#FFA07A", "#98FB98", "#87CEEB", "#DDA0DD", "#F0E68C"]

            # ==================== GRÀFIC DOC ====================
            ax_doc = axes[0]

            # Zones de temps (només per COLUMN)
            if not is_bp:
                for (nom, (t_ini, t_fi)), zc in zip(self.config["time_fractions"].items(), zone_colors):
                    ax_doc.axvspan(t_ini, t_fi, alpha=0.1, color=zc)

            # Dibuixar DOC de cada rèplica
            for rep_id in sorted(all_evals.keys()):
                ev = all_evals[rep_id]
                if not ev.get("valid"):
                    continue

                t = np.asarray(ev.get("t", []), dtype=float)
                y = np.asarray(ev.get("y", []), dtype=float)
                is_selected = rep_id == sel_doc

                style = {
                    'color': colors.get(rep_id, 'gray'),
                    'linewidth': 2.0 if is_selected else 1.0,
                    'alpha': 1.0 if is_selected else 0.5,
                    'linestyle': '-' if is_selected else '--',
                }

                label = f"R{rep_id}"
                if is_selected:
                    label += " (SEL)"
                if ev.get("has_timeout"):
                    label += " [TimeOUT]"
                if ev.get("has_batman"):
                    label += " [Batman]"

                ax_doc.plot(t, y, label=label, **style)

                if is_selected and len(t) > 0:
                    ax_doc.fill_between(t, 0, y, alpha=0.15, color=style['color'])

            # Títol DOC amb estadístiques
            doc_r = doc_comparison.get("pearson")
            doc_diff = doc_comparison.get("area_diff_pct")

            title_parts = [f"DOC - {sample_name}"]
            if doc_r is not None:
                title_parts.append(f"R={doc_r:.3f}")
            if doc_diff is not None:
                title_parts.append(f"ΔÀrea={doc_diff:.1f}%")
            title_parts.append(f"Seleccionat: R{sel_doc}")

            # Color del títol segons qualitat
            if doc_r is not None and doc_r < DEFAULT_MIN_CORR:
                title_color = "#E74C3C"
            elif doc_diff is not None and doc_diff > DEFAULT_MAX_DIFF:
                title_color = "#E67E22"
            else:
                title_color = "#27AE60"

            ax_doc.set_title(" | ".join(title_parts), fontsize=11, fontweight='bold', color=title_color)
            ax_doc.set_xlim(0, x_max)
            ax_doc.set_ylim(bottom=0)
            ax_doc.set_ylabel("DOC", fontsize=10)
            ax_doc.legend(loc='upper right', fontsize=8)
            ax_doc.grid(True, alpha=0.3)

            # ==================== GRÀFICS DAD ====================
            # Carregar DataFrames DAD
            dad_dfs = {}
            for rep_id, filepath in reps.items():
                dad_dfs[rep_id] = self._load_dad_dataframe(filepath)

            for i, wl in enumerate(wavelengths):
                ax_dad = axes[i + 1]
                wl_str = str(wl)

                # Dibuixar cada rèplica
                for rep_id in sorted(dad_dfs.keys()):
                    df_dad = dad_dfs[rep_id]
                    if df_dad.empty or wl_str not in df_dad.columns:
                        continue

                    t_col = [c for c in df_dad.columns if 'time' in str(c).lower()]
                    if not t_col:
                        continue

                    t = pd.to_numeric(df_dad[t_col[0]], errors='coerce').values
                    y = pd.to_numeric(df_dad[wl_str], errors='coerce').values

                    is_selected = rep_id == sel_dad
                    style = {
                        'color': colors.get(rep_id, 'gray'),
                        'linewidth': 1.5 if is_selected else 0.8,
                        'alpha': 1.0 if is_selected else 0.5,
                        'linestyle': '-' if is_selected else '--',
                    }

                    ax_dad.plot(t, y, label=f"R{rep_id}", **style)

                # Estadístiques DAD per aquesta WL
                stats_parts = [f"DAD {wl}nm"]

                for rep_id in sorted(dad_quality.keys()):
                    qual = dad_quality[rep_id]
                    if qual.get("valid") and wl in qual.get("per_wl", {}):
                        wl_stats = qual["per_wl"][wl]
                        der = wl_stats.get("deriva", 0)
                        sroll = wl_stats.get("soroll", 0)

                        # Color per indicar problemes
                        if abs(der) > DAD_DERIVA_WARN:
                            stats_parts.append(f"R{rep_id}: Der={der:.1f}% ⚠")
                        else:
                            stats_parts.append(f"R{rep_id}: Der={der:.1f}%")

                ax_dad.set_title(" | ".join(stats_parts), fontsize=9)
                ax_dad.set_xlim(0, x_max)
                ax_dad.set_ylim(bottom=0)
                ax_dad.set_ylabel(f"{wl}nm", fontsize=8)
                ax_dad.legend(loc='upper right', fontsize=7)
                ax_dad.grid(True, alpha=0.3)

            # Etiqueta eix X només a l'últim
            axes[-1].set_xlabel("Temps (min)", fontsize=10)

            # Text amb selecció final
            fig.text(0.5, 0.01, f"SELECCIÓ FINAL → DOC: Rèplica {sel_doc} | DAD: Rèplica {sel_dad}",
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#E8F6F3', edgecolor='#1ABC9C'))

            # Guardar PNG
            png_path = os.path.join(output_folder, f"QAQC_{sample_name}.png")
            plt.savefig(png_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return png_path

        except Exception as e:
            print(f"Error generant gràfic QAQC per {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compare_replicas(self, evals):
        """Compara rèpliques amb alineació, Pearson i diferències d'àrea per DOC i TOTES les wavelengths DAD."""
        comparison = {
            "n_valid": 0,
            "doc": {
                "pearson": None,
                "area_diff_pct": None,
                "shift": 0,
                "quality_scores": {},
            },
            "dad": {},  # Dict per cada wavelength: {wl: {pearson, area_diff_pct, shift}}
            "best_doc": None,
            "best_dad": None,
        }

        valid_evals = {k: v for k, v in evals.items() if v.get("valid")}
        comparison["n_valid"] = len(valid_evals)

        if len(valid_evals) < 2:
            # Si només hi ha una rèplica, la seleccionem automàticament
            if len(valid_evals) == 1:
                rep = list(valid_evals.keys())[0]
                comparison["best_doc"] = rep
                comparison["best_dad"] = rep
            return comparison

        rep_ids = sorted(valid_evals.keys())

        # ===== COMPARACIÓ DOC =====
        t1 = np.asarray(valid_evals[rep_ids[0]].get("t", []), dtype=float)
        y1 = np.asarray(valid_evals[rep_ids[0]].get("y", []), dtype=float)
        t2 = np.asarray(valid_evals[rep_ids[1]].get("t", []), dtype=float)
        y2 = np.asarray(valid_evals[rep_ids[1]].get("y", []), dtype=float)

        # Determinar rang d'anàlisi DOC (15-35 min per defecte)
        doc_min, doc_max = 15.0, 35.0
        is_bp = valid_evals[rep_ids[0]].get("is_bp", False)
        if is_bp:
            doc_min, doc_max = 2.0, 15.0

        # Alinear i calcular estadístiques DOC
        if len(t1) > 10 and len(t2) > 10:
            try:
                t_aligned, y1_aligned, y2_aligned, shift_doc = alinear_senyals(
                    t1, y1, t2, y2, doc_min, doc_max, threshold=0.0
                )
                if t_aligned is not None:
                    corr_doc, diff_doc = calcular_estadistiques(t_aligned, y1_aligned, y2_aligned)
                    comparison["doc"]["pearson"] = float(corr_doc)
                    comparison["doc"]["area_diff_pct"] = float(diff_doc)
                    comparison["doc"]["shift"] = float(shift_doc)
            except Exception as e:
                print(f"Error comparant DOC: {e}")

        # Calcular quality scores per DOC
        for rep_id in rep_ids:
            y_rep = np.asarray(valid_evals[rep_id].get("y", []), dtype=float)
            comparison["doc"]["quality_scores"][rep_id] = avaluar_qualitat_doc(y_rep)

        # Seleccionar millor DOC (menor score = millor)
        if comparison["doc"]["quality_scores"]:
            comparison["best_doc"] = min(comparison["doc"]["quality_scores"],
                                         key=lambda k: comparison["doc"]["quality_scores"][k])

        # ===== COMPARACIÓ DAD - TOTES LES WAVELENGTHS =====
        dad1 = valid_evals[rep_ids[0]].get("dad_eval", {})
        dad2 = valid_evals[rep_ids[1]].get("dad_eval", {})

        # Rang d'anàlisi DAD (18-25 min per defecte)
        dad_min, dad_max = 18.0, 25.0
        if is_bp:
            dad_min, dad_max = 3.0, 12.0

        if dad1.get("valid") and dad2.get("valid"):
            signals1 = dad1.get("signals", {})
            signals2 = dad2.get("signals", {})

            # Processar totes les wavelengths
            dad_quality_scores = {rep_ids[0]: [], rep_ids[1]: []}

            for wl in TARGET_WAVELENGTHS_QAQC:
                wl_str = str(wl)
                has_w1 = wl in signals1 or wl_str in signals1
                has_w2 = wl in signals2 or wl_str in signals2

                # Obtenir clau correcta
                key1 = wl if wl in signals1 else (wl_str if wl_str in signals1 else None)
                key2 = wl if wl in signals2 else (wl_str if wl_str in signals2 else None)

                if key1 and key2:
                    t_dad1 = np.asarray(signals1[key1].get("t", []), dtype=float)
                    y_dad1 = np.asarray(signals1[key1].get("y_baseline", signals1[key1].get("y", [])), dtype=float)
                    t_dad2 = np.asarray(signals2[key2].get("t", []), dtype=float)
                    y_dad2 = np.asarray(signals2[key2].get("y_baseline", signals2[key2].get("y", [])), dtype=float)

                    comparison["dad"][wl] = {
                        "pearson": None,
                        "area_diff_pct": None,
                        "shift": 0,
                    }

                    if len(t_dad1) > 10 and len(t_dad2) > 10:
                        try:
                            t_aligned, y1_aligned, y2_aligned, shift_dad = alinear_senyals(
                                t_dad1, y_dad1, t_dad2, y_dad2, dad_min, dad_max, threshold=0.0
                            )
                            if t_aligned is not None:
                                corr_dad, diff_dad = calcular_estadistiques(t_aligned, y1_aligned, y2_aligned)
                                comparison["dad"][wl]["pearson"] = float(corr_dad)
                                comparison["dad"][wl]["area_diff_pct"] = float(diff_dad)
                                comparison["dad"][wl]["shift"] = float(shift_dad)
                        except Exception as e:
                            print(f"Error comparant DAD {wl}nm: {e}")

                    # Scores de qualitat
                    dad_quality_scores[rep_ids[0]].append(avaluar_qualitat_dad(y_dad1))
                    dad_quality_scores[rep_ids[1]].append(avaluar_qualitat_dad(y_dad2))

            # Seleccionar millor DAD (mitjana de scores menor = millor)
            if dad_quality_scores[rep_ids[0]] and dad_quality_scores[rep_ids[1]]:
                avg_score_1 = np.mean(dad_quality_scores[rep_ids[0]])
                avg_score_2 = np.mean(dad_quality_scores[rep_ids[1]])
                comparison["best_dad"] = rep_ids[0] if avg_score_1 <= avg_score_2 else rep_ids[1]
                comparison["dad_quality_scores"] = {
                    rep_ids[0]: avg_score_1,
                    rep_ids[1]: avg_score_2
                }

        # Si no s'ha determinat best_dad, usar el millor DOC
        if comparison["best_dad"] is None:
            comparison["best_dad"] = comparison["best_doc"]

        return comparison

    def _evaluate_replica(self, filepath):
        """Avalua una rèplica amb DOC i DAD."""
        try:
            xls = pd.ExcelFile(filepath)
            sheets = xls.sheet_names

            # Llegir ID per obtenir info (Method, etc.)
            id_dict = {}
            if 'ID' in sheets:
                df_id = pd.read_excel(xls, 'ID')
                if 'Camp' in df_id.columns and 'Valor' in df_id.columns:
                    id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

            method = str(id_dict.get("Method", "COLUMN"))
            is_bp = method.upper() == "BP"

            # Llegir DOC
            sheet_doc = 'DOC' if 'DOC' in sheets else ('TOC' if 'TOC' in sheets else None)
            if not sheet_doc:
                return {"valid": False, "reason": "No DOC sheet"}

            df_doc = pd.read_excel(xls, sheet_doc)

            # Trobar columnes DOC
            t_col = [c for c in df_doc.columns if 'time' in c.lower()]
            y_col = [c for c in df_doc.columns if 'doc' in c.lower() and 'raw' not in c.lower()]
            if not y_col:
                y_col = [c for c in df_doc.columns if 'toc' in c.lower()]

            if not t_col or not y_col:
                return {"valid": False, "reason": "Columnes DOC no trobades"}

            t = pd.to_numeric(df_doc[t_col[0]], errors='coerce').values
            y = pd.to_numeric(df_doc[y_col[0]], errors='coerce').values

            mask = np.isfinite(t) & np.isfinite(y)
            t, y = t[mask], y[mask]

            if len(t) < 20:
                return {"valid": False, "reason": "Poques dades"}

            y_clean = netejar_baseline(y)

            # Qualitat DOC
            roughness = calcular_roughness(y)
            bl_stats = baseline_stats(y_clean)

            # Pic principal
            peak_info = detect_main_peak(t, y_clean)
            area = peak_info.get("area", 0) if peak_info["valid"] else 0

            # Anomalies via hpsec_replica (unificat)
            method = "BP" if is_bp else "COLUMN"
            unified_eval = evaluate_replica_unified(t, y_clean, method=method)

            has_timeout = unified_eval.get("timeout", False)
            has_batman = unified_eval.get("batman", False)
            has_irr = unified_eval.get("irr", False)
            timeout_info = unified_eval.get("timeout_info")
            # NOTA: detect_ears eliminat - no funcionava bé
            n_ears = 0

            # Score DOC
            score = (
                roughness * self.config["score_roughness"] +
                (self.config["score_timeout"] if has_timeout else 0) +
                (self.config["score_batman"] if has_batman else 0) +
                (n_ears * self.config["score_ear"]) +
                (self.config["score_low_area"] if area < 50 else 0)
            )

            # KPIs per temps (només per COLUMN, no per BP ni per KHP)
            kpis = {}
            mostra_nom = str(id_dict.get("Mostra", ""))
            is_khp_sample = is_khp(mostra_nom)

            if is_khp_sample:
                # KHP és patró, només àrea total
                kpis = {"total": float(trapezoid(y_clean, t))}
            elif not is_bp:
                kpis = calcular_kpis(t, y_clean, self.config)
            else:
                # Per BP només àrea total
                kpis = {"total": float(trapezoid(y_clean, t))}

            # SNR
            peak_h = peak_info.get("height", np.max(y_clean)) if peak_info["valid"] else float(np.max(y_clean))
            snr = (peak_h - bl_stats["mean"]) / bl_stats["std"] if bl_stats["std"] > 0 else 0.0

            # ===== AVALUACIÓ DAD =====
            dad_eval = {"valid": False}
            dad_scores = {}
            dad_signals = {}  # Guardar senyals per plotting

            if 'DAD' in sheets:
                try:
                    df_dad = pd.read_excel(xls, 'DAD')
                    print(f"  DAD columnes: {list(df_dad.columns)[:8]}...")  # Debug
                    t_dad_col = [c for c in df_dad.columns if 'time' in c.lower()]

                    if t_dad_col:
                        t_dad = pd.to_numeric(df_dad[t_dad_col[0]], errors='coerce').values
                        t_dad_mask = np.isfinite(t_dad)

                        # Avaluar cada wavelength
                        for wl in self.config["target_wavelengths"]:
                            wl_str = str(wl)
                            if wl_str in df_dad.columns:
                                y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').values

                                # Score qualitat DAD: negatius + soroll
                                valid_mask = np.isfinite(y_wl) & t_dad_mask
                                if np.sum(valid_mask) > 10:
                                    t_wl_clean = t_dad[valid_mask]
                                    y_wl_clean = y_wl[valid_mask]

                                    # Guardar senyal per plotting
                                    dad_signals[wl] = {
                                        "t": t_wl_clean.tolist() if hasattr(t_wl_clean, 'tolist') else list(t_wl_clean),
                                        "y": y_wl_clean.tolist() if hasattr(y_wl_clean, 'tolist') else list(y_wl_clean),
                                        "y_baseline": netejar_baseline(y_wl_clean).tolist() if hasattr(netejar_baseline(y_wl_clean), 'tolist') else list(netejar_baseline(y_wl_clean))
                                    }

                                    neg_score = np.sum(np.abs(y_wl_clean[y_wl_clean < 0]))
                                    baseline_pts = y_wl_clean[y_wl_clean <= np.percentile(y_wl_clean, 20)]
                                    noise_score = np.std(baseline_pts) * len(y_wl_clean) if len(baseline_pts) > 0 else 0
                                    dad_scores[wl] = neg_score + noise_score

                                    # Calcular àrea DAD 254 si és el wavelength principal
                                    if wl == 254:
                                        y_bl = np.array(dad_signals[wl]["y_baseline"])
                                        dad_peak = detect_main_peak(t_wl_clean, y_bl)
                                        dad_signals[wl]["area"] = dad_peak.get("area", 0) if dad_peak["valid"] else 0
                                        dad_signals[wl]["peak_info"] = dad_peak

                        if dad_scores:
                            print(f"  DAD vàlid amb {len(dad_signals)} wavelengths: {list(dad_signals.keys())}")  # Debug
                            dad_eval = {
                                "valid": True,
                                "scores": dad_scores,
                                "best_wl": min(dad_scores, key=dad_scores.get) if dad_scores else None,
                                "t_dad": (t_dad[t_dad_mask] if np.any(t_dad_mask) else t_dad).tolist(),
                                "signals": dad_signals,
                            }
                        else:
                            print(f"  DAD sense wavelengths vàlids")  # Debug
                    else:
                        print(f"  DAD sense columna time")  # Debug
                except Exception as e:
                    print(f"  DAD error: {e}")  # Debug
                    dad_eval = {"valid": False, "error": str(e)}
            else:
                print(f"  No hi ha sheet DAD")  # Debug

            return {
                "valid": True,
                "score": score,
                "roughness": roughness,
                "area": area,
                "has_timeout": has_timeout,
                "has_batman": has_batman,
                "n_ears": n_ears,
                "kpis": kpis,
                "snr": snr,
                "t": t,
                "y": y_clean,
                "method": method,
                "is_bp": is_bp,
                "is_khp": is_khp_sample,
                "dad_eval": dad_eval,
                "dad_scores": dad_scores,
                "id_dict": id_dict,
            }

        except Exception as e:
            return {"valid": False, "reason": str(e)}

    def _select_best_replica(self, evals, replica_comparison=None):
        """Selecciona la millor rèplica basant-se en qualitat, anomalies i estadístiques."""
        valid = {k: v for k, v in evals.items() if v.get("valid", False)}

        if not valid:
            return None, "Cap rèplica vàlida"

        if len(valid) == 1:
            rep = list(valid.keys())[0]
            return rep, f"Única vàlida (R{rep})"

        # Ordenar per score (menor = millor)
        sorted_reps = sorted(valid.items(), key=lambda x: x[1]["score"])
        best_rep, best_eval = sorted_reps[0]
        second_rep, second_eval = sorted_reps[1]

        reasons = []

        # Usar la selecció basada en qualitat DOC/DAD si disponible
        if replica_comparison:
            best_doc_from_quality = replica_comparison.get("best_doc")
            best_dad_from_quality = replica_comparison.get("best_dad")

            # Pearson i àrea per DOC
            doc_stats = replica_comparison.get("doc", {})
            pearson_doc = doc_stats.get("pearson")
            area_diff_doc = doc_stats.get("area_diff_pct")

            if pearson_doc is not None:
                if pearson_doc >= DEFAULT_MIN_CORR:
                    reasons.append(f"DOC R={pearson_doc:.3f}")
                else:
                    reasons.append(f"DOC R={pearson_doc:.3f} (baixa)")
            if area_diff_doc is not None:
                if area_diff_doc > DEFAULT_MAX_DIFF:
                    reasons.append(f"DOC ΔÀrea={area_diff_doc:.1f}%")

            # Estadístiques per DAD wavelengths
            dad_stats = replica_comparison.get("dad", {})
            dad_issues = []
            for wl, stats in dad_stats.items():
                p = stats.get("pearson")
                d = stats.get("area_diff_pct")
                if p is not None and p < DEFAULT_MIN_CORR:
                    dad_issues.append(f"{wl}nm:R={p:.2f}")
                if d is not None and d > DEFAULT_MAX_DIFF:
                    dad_issues.append(f"{wl}nm:Δ={d:.0f}%")

            if dad_issues:
                reasons.append(f"DAD: {', '.join(dad_issues[:3])}")

            # Usar selecció per qualitat si disponible
            if best_doc_from_quality and best_doc_from_quality != best_rep:
                # La qualitat DOC suggereix una rèplica diferent
                quality_scores = doc_stats.get("quality_scores", {})
                if quality_scores:
                    reasons.append(f"Qualitat DOC: R{best_doc_from_quality}")
                    best_rep = best_doc_from_quality

        # Raons basades en anomalies (prioritàries)
        if best_eval.get("has_timeout") and not second_eval.get("has_timeout"):
            best_rep = second_rep
            reasons.insert(0, f"R{sorted_reps[0][0]} té TimeOUT")
        elif second_eval.get("has_timeout") and not best_eval.get("has_timeout"):
            reasons.insert(0, f"R{second_rep} té TimeOUT")

        if best_eval.get("has_batman") and not second_eval.get("has_batman"):
            best_rep = second_rep
            reasons.insert(0, f"R{sorted_reps[0][0]} té Batman")
        elif second_eval.get("has_batman") and not best_eval.get("has_batman"):
            reasons.insert(0, f"R{second_rep} té Batman")

        # NOTA: detect_ears eliminat - ja no s'utilitza per selecció

        if not reasons:
            reasons.append(f"Score: {best_eval['score']:.1f} vs {second_eval['score']:.1f}")

        return best_rep, "; ".join(reasons[:4])

    def _update_proc_progress(self, values, tag, progress):
        """Actualitza progrés del processament."""
        self.tree_process.insert("", "end", values=values, tags=(tag,))
        self.progress_proc['value'] = progress

    def _generate_stacked_plot(self, sample_name, data, output_folder):
        """Genera gràfic stacked amb DOC + totes les wavelengths DAD."""
        try:
            os.makedirs(output_folder, exist_ok=True)

            all_evals = data.get("all_evals", {})
            selected_rep = data.get("rep", "1")
            eval_selected = data.get("eval", {})
            is_bp = eval_selected.get("is_bp", False)
            x_max = 20 if is_bp else 70

            # Determinar quantes wavelengths DAD tenim
            dad_wls = []
            for rep_id, ev in all_evals.items():
                if ev.get("valid") and ev.get("dad_eval", {}).get("valid"):
                    signals = ev["dad_eval"].get("signals", {})
                    dad_wls = list(signals.keys())
                    break

            n_plots = 1 + len(dad_wls)  # DOC + DAD wavelengths
            fig_height = max(6, 2.5 * n_plots)

            fig, axes = plt.subplots(n_plots, 1, figsize=(12, fig_height), sharex=True)
            if n_plots == 1:
                axes = [axes]
            plt.subplots_adjust(hspace=0.25)

            colors = {"1": "#2E86AB", "2": "#A23B72"}
            zone_colors = ["#FFA07A", "#98FB98", "#87CEEB", "#DDA0DD", "#F0E68C"]

            # ===== PLOT 1: DOC =====
            ax_doc = axes[0]

            # Zones de temps (només per COLUMN)
            if not is_bp:
                for (nom, (t_ini, t_fi)), zc in zip(self.config["time_fractions"].items(), zone_colors):
                    ax_doc.axvspan(t_ini, t_fi, alpha=0.1, color=zc, label=nom)

            # Dibuixar DOC de cada rèplica
            for rep_id, ev in all_evals.items():
                if not ev.get("valid"):
                    continue
                t = np.array(ev.get("t", []))
                y = np.array(ev.get("y", []))
                is_sel = rep_id == selected_rep

                style = {
                    'color': colors.get(rep_id, 'gray'),
                    'linewidth': 2.0 if is_sel else 1.0,
                    'alpha': 1.0 if is_sel else 0.4,
                    'linestyle': '-' if is_sel else '--',
                    'label': f"R{rep_id} {'(SEL)' if is_sel else ''}"
                }

                ax_doc.plot(t, y, **style)
                if is_sel and len(t) > 0 and len(y) > 0:
                    ax_doc.fill_between(t, 0, y, alpha=0.15, color=style['color'])

            # Marcar anomalies
            if eval_selected.get("has_timeout"):
                ax_doc.text(0.02, 0.95, "⚠ TIMEOUT", transform=ax_doc.transAxes,
                           fontsize=10, color='red', fontweight='bold', va='top')
            if eval_selected.get("has_batman"):
                ax_doc.text(0.02, 0.85, "⚠ BATMAN", transform=ax_doc.transAxes,
                           fontsize=10, color='purple', fontweight='bold', va='top')

            ax_doc.set_xlim(0, x_max)
            ax_doc.set_ylim(bottom=0)
            ax_doc.set_ylabel("DOC", fontsize=10)

            # Títol DOC amb estadístiques de comparació
            replica_comp = data.get("replica_comparison", {})
            doc_stats = replica_comp.get("doc", {})
            pearson_doc = doc_stats.get("pearson")
            area_diff_doc = doc_stats.get("area_diff_pct")

            title_doc = f"{sample_name} - DOC | Àrea: {eval_selected.get('area', 0):.1f} | Score: {eval_selected.get('score', 0):.1f}"
            if pearson_doc is not None:
                ok_pearson = pearson_doc >= DEFAULT_MIN_CORR
                ok_diff = area_diff_doc <= DEFAULT_MAX_DIFF if area_diff_doc is not None else True
                color_title = "green" if (ok_pearson and ok_diff) else "red"
                title_doc += f" | R={pearson_doc:.3f}"
                if area_diff_doc is not None:
                    title_doc += f" | ΔÀrea={area_diff_doc:.1f}%"
            else:
                color_title = "black"

            ax_doc.set_title(title_doc, fontsize=10, fontweight='bold', color=color_title)
            ax_doc.legend(loc='upper right', fontsize=8)
            ax_doc.grid(True, alpha=0.3)

            # ===== PLOTS DAD: una per cada wavelength =====
            for i, wl in enumerate(sorted(dad_wls)):
                ax_dad = axes[i + 1]

                for rep_id, ev in all_evals.items():
                    if not ev.get("valid"):
                        continue
                    dad_eval = ev.get("dad_eval", {})
                    if not dad_eval.get("valid"):
                        continue

                    signals = dad_eval.get("signals", {})
                    if wl not in signals:
                        continue

                    t_wl = np.array(signals[wl].get("t", []))
                    y_wl = np.array(signals[wl].get("y_baseline", signals[wl].get("y", [])))
                    is_sel = rep_id == selected_rep

                    style = {
                        'color': colors.get(rep_id, 'gray'),
                        'linewidth': 1.5 if is_sel else 0.8,
                        'alpha': 1.0 if is_sel else 0.4,
                        'linestyle': '-' if is_sel else '--',
                        'label': f"R{rep_id}"
                    }

                    if len(t_wl) > 0 and len(y_wl) > 0:
                        ax_dad.plot(t_wl, y_wl, **style)

                # Info del DAD amb estadístiques de comparació
                dad_stats = replica_comp.get("dad", {}).get(wl, {})
                pearson_dad = dad_stats.get("pearson")
                area_diff_dad = dad_stats.get("area_diff_pct")

                title_dad = f"DAD {wl} nm"
                color_dad = "black"

                if pearson_dad is not None:
                    ok_p = pearson_dad >= DEFAULT_MIN_CORR
                    ok_d = area_diff_dad <= DEFAULT_MAX_DIFF if area_diff_dad is not None else True
                    color_dad = "black" if (ok_p and ok_d) else "red"
                    title_dad += f" | R={pearson_dad:.3f}"
                    if area_diff_dad is not None:
                        title_dad += f" | ΔÀrea={area_diff_dad:.1f}%"

                ax_dad.set_xlim(0, x_max)
                ax_dad.set_ylim(bottom=0)
                ax_dad.set_ylabel(f"DAD {wl}nm", fontsize=9)
                ax_dad.set_title(title_dad, fontsize=9, color=color_dad)
                ax_dad.legend(loc='upper right', fontsize=7)
                ax_dad.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Temps (min)", fontsize=10)

            fig.tight_layout()
            png_path = os.path.join(output_folder, f"QAQC_{sample_name}.png")
            plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return png_path

        except Exception as e:
            print(f"Error generant gràfic stacked per {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _processing_done(self):
        """Processament completat."""
        self.is_processing = False
        self.progress_proc['value'] = 100
        self.btn_processar.configure(state="normal")
        self.btn_exportar.configure(state="normal")

        # Actualitzar resum exportació
        total = len(self.selected_replicas)
        ok = sum(1 for v in self.selected_replicas.values()
                if v["eval"].get("valid") and not v["eval"].get("has_timeout") and not v["eval"].get("has_batman"))
        warn = total - ok

        self.lbl_exp_total.configure(text=f"Mostres a exportar: {total}")
        self.lbl_exp_ok.configure(text=f"Sense anomalies: {ok}")
        self.lbl_exp_warn.configure(text=f"Amb anomalies: {warn}")

        # Generar PDF de processament
        self._generate_processing_pdf()

        # Comptar gràfics generats a CHECK
        check_folder = ensure_check_folder(self.seq_path)
        n_plots = len(glob.glob(os.path.join(check_folder, "*.png"))) if os.path.exists(check_folder) else 0

        messagebox.showinfo("Processament",
            f"Processament completat!\n\n"
            f"Mostres analitzades: {total}\n"
            f"Sense anomalies: {ok}\n"
            f"Amb anomalies: {warn}\n\n"
            f"Gràfics PNG + PDF: {n_plots} (CHECK/)")

        # Obrir carpeta CHECK automàticament
        try:
            if platform.system() == "Windows":
                os.startfile(check_folder)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", check_folder])
            else:  # Linux
                subprocess.call(["xdg-open", check_folder])
        except Exception as e:
            print(f"No s'ha pogut obrir la carpeta: {e}")

    def _generate_processing_pdf(self):
        """Genera PDF d'informe de processament QA/QC amb gràfics detallats."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            check_path = ensure_check_folder(self.seq_path)
            seq_name = obtenir_seq(self.seq_path)
            pdf_path = os.path.join(check_path, f"REPORT_Processament_{seq_name}.pdf")

            # Obtenir dades de calibració
            khp_area = 1.0
            khp_conc = 1.0
            if self.calibration_data:
                khp_data = self.calibration_data.get("khp_data", {})
                khp_area = khp_data.get("area", 1.0) or 1.0
                khp_conc = khp_data.get("conc_ppm", 1.0) or 1.0

            with PdfPages(pdf_path) as pdf:
                # =================================================================
                # PÀGINA 1: Portada i resum
                # =================================================================
                fig_cover = plt.figure(figsize=(8.27, 11.69))
                fig_cover.patch.set_facecolor('white')

                # Logo
                if os.path.exists(LOGO_PATH):
                    try:
                        logo = plt.imread(LOGO_PATH)
                        ax_logo = fig_cover.add_axes([0.35, 0.85, 0.3, 0.1])
                        ax_logo.imshow(logo)
                        ax_logo.axis('off')
                    except Exception:
                        pass

                # Títol
                fig_cover.text(0.5, 0.78, PDF_HEADER, ha='center', va='top',
                              fontsize=14, fontweight='bold', color=COLORS["primary"])
                fig_cover.text(0.5, 0.73, "INFORME DE PROCESSAMENT QA/QC", ha='center', va='top',
                              fontsize=18, fontweight='bold')
                fig_cover.text(0.5, 0.68, f"Seqüència: {seq_name}", ha='center', va='top',
                              fontsize=12, color='gray')
                fig_cover.text(0.5, 0.64, f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",
                              ha='center', va='top', fontsize=10, color='gray')

                # Estadístiques
                total = len(self.selected_replicas)
                ok = sum(1 for v in self.selected_replicas.values()
                        if v["eval"].get("valid") and not v["eval"].get("has_timeout") and not v["eval"].get("has_batman"))
                timeout_count = sum(1 for v in self.selected_replicas.values() if v["eval"].get("has_timeout"))
                batman_count = sum(1 for v in self.selected_replicas.values() if v["eval"].get("has_batman"))
                column_count = sum(1 for v in self.selected_replicas.values()
                                  if v["eval"].get("valid") and not v["eval"].get("is_bp"))
                bp_count = sum(1 for v in self.selected_replicas.values()
                              if v["eval"].get("valid") and v["eval"].get("is_bp"))

                # Taula resum
                ax_summary = fig_cover.add_axes([0.1, 0.35, 0.8, 0.25])
                ax_summary.axis('off')

                summary_data = [
                    ["RESUM", ""],
                    ["Total mostres processades", str(total)],
                    ["Mostres COLUMN", str(column_count)],
                    ["Mostres BP (By-Pass)", str(bp_count)],
                    ["Mostres OK (sense anomalies)", str(ok)],
                    ["Mostres amb TimeOUT", str(timeout_count)],
                    ["Mostres amb Batman", str(batman_count)],
                    ["Calibració KHP", f"{khp_conc:.1f} ppm (àrea: {khp_area:.1f})"],
                ]

                tbl = ax_summary.table(cellText=summary_data, loc='center', cellLoc='left',
                                       colWidths=[0.5, 0.4])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                tbl.scale(1.0, 2.0)

                for i in range(len(summary_data)):
                    if i == 0:  # Header
                        tbl[(i, 0)].set_facecolor(COLORS["primary"])
                        tbl[(i, 0)].set_text_props(color='white', fontweight='bold')
                        tbl[(i, 1)].set_facecolor(COLORS["primary"])
                    else:
                        tbl[(i, 0)].set_facecolor('#f8f9fa')

                # Llegenda
                fig_cover.text(0.1, 0.28, "Llegenda:", fontsize=10, fontweight='bold')
                fig_cover.text(0.1, 0.24, "• TimeOUT: Platós anormals (saturació del detector)", fontsize=9)
                fig_cover.text(0.1, 0.21, "• Batman: Doble pic (problema d'injecció)", fontsize=9)
                fig_cover.text(0.1, 0.18, "• Ears: Violacions de monotonia en el pic", fontsize=9)
                fig_cover.text(0.1, 0.15, "• Score: Puntuació de qualitat (menor = millor)", fontsize=9)
                fig_cover.text(0.1, 0.12, "• DAD 254: Qualitat del senyal DAD a 254nm", fontsize=9)

                pdf.savefig(fig_cover, dpi=150)
                plt.close(fig_cover)

                # =================================================================
                # PÀGINA 2: Taula detallada de mostres
                # =================================================================
                fig_table = plt.figure(figsize=(8.27, 11.69))
                fig_table.patch.set_facecolor('white')
                fig_table.text(0.5, 0.95, "Detall per Mostra", ha='center', fontsize=14, fontweight='bold')

                ax_tbl = fig_table.add_axes([0.02, 0.1, 0.96, 0.8])
                ax_tbl.axis('off')

                # Crear dades taula amb concentracions
                headers = ["Mostra", "Rep", "Score", "TO", "Bat", "Àrea", "Conc(ppm)", "DAD", "Pearson"]
                rows = []

                for sample_name in sorted(self.selected_replicas.keys()):
                    data = self.selected_replicas[sample_name]
                    eval_d = data.get("eval", {})
                    comparison = data.get("replica_comparison", {})

                    if not eval_d.get("valid"):
                        rows.append([sample_name[:20], "-", "INV", "-", "-", "-", "-", "-", "-"])
                        continue

                    rep = data.get("rep", "?")
                    score = f"{eval_d.get('score', 0):.1f}"
                    timeout = "!" if eval_d.get("has_timeout") else "-"
                    batman = "!" if eval_d.get("has_batman") else "-"
                    area = f"{eval_d.get('area', 0):.0f}"
                    conc = eval_d.get("conc_ppm", 0)
                    conc_str = f"{conc:.2f}" if conc > 0 else "-"
                    dad_ok = "OK" if eval_d.get("dad_eval", {}).get("valid") else "-"
                    pearson = comparison.get("pearson_doc")
                    pearson_str = f"{pearson:.3f}" if pearson is not None else "-"

                    rows.append([sample_name[:20], f"R{rep}", score, timeout, batman, area, conc_str, dad_ok, pearson_str])

                if rows:
                    table_data = [headers] + rows
                    tbl2 = ax_tbl.table(cellText=table_data, loc='upper center', cellLoc='center',
                                        colWidths=[0.18, 0.06, 0.08, 0.06, 0.06, 0.1, 0.12, 0.08, 0.1])
                    tbl2.auto_set_font_size(False)
                    tbl2.set_fontsize(8)
                    tbl2.scale(1.0, 1.5)

                    for j in range(len(headers)):
                        tbl2[(0, j)].set_facecolor(COLORS["primary"])
                        tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

                    # Colorar files amb anomalies
                    for i, row in enumerate(rows, start=1):
                        if row[3] == "!" or row[4] == "!":
                            for j in range(len(headers)):
                                tbl2[(i, j)].set_facecolor('#fff3cd')

                pdf.savefig(fig_table, dpi=150)
                plt.close(fig_table)

                # =================================================================
                # PÀGINES KPI: Fraccions de temps (només COLUMN)
                # =================================================================
                column_samples = [(k, v) for k, v in self.selected_replicas.items()
                                 if v["eval"].get("valid") and not v["eval"].get("is_bp")]

                if column_samples:
                    fig_kpi = plt.figure(figsize=(8.27, 11.69))
                    fig_kpi.patch.set_facecolor('white')
                    fig_kpi.text(0.5, 0.95, "KPIs - Fraccions Cromatogràfiques (mode COLUMN)",
                                ha='center', fontsize=14, fontweight='bold')
                    fig_kpi.text(0.5, 0.92, "Distribució de la matèria orgànica per temps de retenció",
                                ha='center', fontsize=10, color='gray')

                    ax_kpi = fig_kpi.add_axes([0.02, 0.1, 0.96, 0.78])
                    ax_kpi.axis('off')

                    # Capçalera amb % i ppm
                    kpi_headers = ["Mostra", "BioP", "HS", "BB", "Carb", "LMW", "Total", "ppm"]
                    kpi_rows = []

                    for sample_name, data in sorted(column_samples, key=lambda x: x[0]):
                        eval_d = data.get("eval", {})
                        kpis = eval_d.get("kpis", {})
                        kpis_ppm = eval_d.get("kpis_ppm", {})

                        row = [sample_name[:18]]
                        total_pct = 0
                        for nom in ["BioP", "HS", "BB", "Carbonats", "LMW"]:
                            pct = kpis.get(f"{nom}_pct", 0)
                            ppm = kpis_ppm.get(nom, 0)
                            total_pct += pct
                            row.append(f"{pct:.1f}%")
                        row.append(f"{total_pct:.1f}%")
                        row.append(f"{eval_d.get('conc_ppm', 0):.2f}")
                        kpi_rows.append(row)

                    if kpi_rows:
                        kpi_data = [kpi_headers] + kpi_rows
                        tbl_kpi = ax_kpi.table(cellText=kpi_data, loc='upper center', cellLoc='center',
                                               colWidths=[0.18, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12])
                        tbl_kpi.auto_set_font_size(False)
                        tbl_kpi.set_fontsize(8)
                        tbl_kpi.scale(1.0, 1.8)

                        for j in range(len(kpi_headers)):
                            tbl_kpi[(0, j)].set_facecolor('#28a745')
                            tbl_kpi[(0, j)].set_text_props(color='white', fontweight='bold')

                    # Llegenda fraccions
                    fig_kpi.text(0.1, 0.06, "Fraccions: BioP (Biopolímers), HS (Àcids húmics), "
                                "BB (Building Blocks), Carb (Carbonats), LMW (Baix pes molecular)", fontsize=8)

                    pdf.savefig(fig_kpi, dpi=150)
                    plt.close(fig_kpi)

                # =================================================================
                # PÀGINES DE GRÀFICS: DOC + DAD per cada mostra
                # =================================================================
                samples_per_page = 4
                sample_list = list(sorted(self.selected_replicas.items(), key=lambda x: x[0]))

                for page_start in range(0, len(sample_list), samples_per_page):
                    page_samples = sample_list[page_start:page_start + samples_per_page]

                    fig_graphs, axes = plt.subplots(len(page_samples), 2, figsize=(8.27, 11.69))
                    fig_graphs.patch.set_facecolor('white')

                    if len(page_samples) == 1:
                        axes = axes.reshape(1, -1)

                    for idx, (sample_name, data) in enumerate(page_samples):
                        ax_doc = axes[idx, 0]
                        ax_info = axes[idx, 1]

                        eval_d = data.get("eval", {})
                        all_evals = data.get("all_evals", {})
                        comparison = data.get("replica_comparison", {})
                        selected_rep = data.get("rep")

                        if not eval_d.get("valid"):
                            ax_doc.text(0.5, 0.5, f"{sample_name}\n\nINVÀLID", ha='center', va='center',
                                       fontsize=12, color='red')
                            ax_doc.axis('off')
                            ax_info.axis('off')
                            continue

                        # Gràfic DOC + DAD
                        is_bp = eval_d.get("is_bp", False)
                        x_max = 20 if is_bp else 70  # BP fins 20, COLUMN fins 70

                        colors_rep = {"1": COLORS["primary"], "2": COLORS["secondary"]}

                        # Dibuixar DOC de cada rèplica
                        for rep_id, rep_eval in all_evals.items():
                            if not rep_eval.get("valid"):
                                continue
                            t = rep_eval.get("t", [])
                            y = rep_eval.get("y", [])
                            is_selected = rep_id == selected_rep
                            color = colors_rep.get(rep_id, 'gray')

                            ax_doc.plot(t, y, color=color, linewidth=1.5 if is_selected else 0.8,
                                       alpha=1.0 if is_selected else 0.4,
                                       label=f"DOC R{rep_id}")

                        # Dibuixar DAD 254 de la rèplica seleccionada (eix secundari)
                        dad_eval = eval_d.get("dad_eval", {})
                        dad_plotted = False
                        if dad_eval.get("valid"):
                            dad_signals = dad_eval.get("signals", {})
                            if 254 in dad_signals:
                                t_dad = dad_signals[254].get("t", [])
                                y_dad = dad_signals[254].get("y_baseline", [])
                                # Convertir a numpy arrays si són llistes
                                t_dad = np.array(t_dad) if not isinstance(t_dad, np.ndarray) else t_dad
                                y_dad = np.array(y_dad) if not isinstance(y_dad, np.ndarray) else y_dad
                                if len(t_dad) > 0 and len(y_dad) > 0:
                                    ax_dad = ax_doc.twinx()
                                    ax_dad.plot(t_dad, y_dad, color='purple', linewidth=1,
                                               alpha=0.6, linestyle='--', label="DAD 254")
                                    ax_dad.set_ylabel("DAD 254nm", fontsize=7, color='purple')
                                    ax_dad.tick_params(axis='y', labelcolor='purple', labelsize=6)
                                    ax_dad.set_ylim(bottom=0)
                                    dad_plotted = True

                        # Configurar eixos
                        ax_doc.set_xlim(0, x_max)
                        ax_doc.set_ylim(bottom=0)
                        ax_doc.set_xlabel("Temps (min)", fontsize=7)
                        ax_doc.set_ylabel("DOC", fontsize=7)
                        ax_doc.tick_params(axis='both', labelsize=6)
                        dad_status = " + DAD" if dad_plotted else ""
                        ax_doc.set_title(f"{sample_name} ({'BP' if is_bp else 'COLUMN'}){dad_status}", fontsize=9, fontweight='bold')
                        ax_doc.grid(True, alpha=0.3)
                        ax_doc.legend(fontsize=6, loc='upper right')

                        # Panell d'informació
                        ax_info.axis('off')
                        info_text = []
                        info_text.append(f"Rèplica: R{selected_rep}")
                        info_text.append(f"Score: {eval_d.get('score', 0):.2f}")
                        info_text.append(f"Àrea DOC: {eval_d.get('area', 0):.1f}")
                        info_text.append(f"Conc: {eval_d.get('conc_ppm', 0):.2f} ppm")
                        info_text.append(f"DAD: {'OK' if dad_eval.get('valid') else 'NO'}")

                        # Anomalies
                        anomalies = []
                        if eval_d.get("has_timeout"):
                            anomalies.append("TimeOUT")
                        if eval_d.get("has_batman"):
                            anomalies.append("Batman")
                        if eval_d.get("n_ears", 0) > 0:
                            anomalies.append(f"Ears:{eval_d['n_ears']}")
                        info_text.append(f"Anomalies: {', '.join(anomalies) if anomalies else 'Cap'}")

                        # Comparació rèpliques
                        if comparison.get("n_valid", 0) > 1:
                            info_text.append("")
                            info_text.append("Comparació:")
                            if comparison.get("pearson_doc") is not None:
                                r = comparison["pearson_doc"]
                                info_text.append(f"  Pearson: {r:.3f}")
                            if comparison.get("area_diff_pct") is not None:
                                diff = comparison["area_diff_pct"]
                                info_text.append(f"  Dif.àrea: {diff:.1f}%")

                        # Integració per temps (només COLUMN)
                        if not is_bp:
                            kpis = eval_d.get("kpis", {})
                            kpis_ppm = eval_d.get("kpis_ppm", {})
                            info_text.append("")
                            info_text.append("FRACCIONS (min):")
                            fractions_info = [
                                ("BioP", "0-18"),
                                ("HS", "18-23"),
                                ("BB", "23-30"),
                                ("Carb", "30-40"),
                                ("LMW", "40-70"),
                            ]
                            for nom, temps in fractions_info:
                                pct = kpis.get(f"{nom if nom != 'Carb' else 'Carbonats'}_pct", 0)
                                ppm = kpis_ppm.get(nom if nom != 'Carb' else 'Carbonats', 0)
                                info_text.append(f"  {nom}({temps}): {pct:.1f}%")

                        ax_info.text(0.02, 0.98, "\n".join(info_text), transform=ax_info.transAxes,
                                    fontsize=6, verticalalignment='top', fontfamily='monospace',
                                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

                    fig_graphs.tight_layout(pad=0.5)
                    pdf.savefig(fig_graphs, dpi=150)
                    plt.close(fig_graphs)

            print(f"PDF processament generat: {pdf_path}")
            # Mostrar missatge a l'usuari
            self.root.after(0, lambda: messagebox.showinfo("PDF Generat",
                f"Informe PDF generat correctament:\n{pdf_path}"))

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error generant PDF processament: {error_msg}")
            # Mostrar error a l'usuari
            self.root.after(0, lambda: messagebox.showerror("Error PDF",
                f"Error generant PDF de processament:\n{str(e)}"))

    def _processing_error(self, error):
        """Error en processament."""
        self.is_processing = False
        self.btn_processar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant el processament:\n{error}")

    def _show_process_context_menu(self, event):
        """Mostra menú contextual per canviar selecció DOC/DAD."""
        item = self.tree_process.identify_row(event.y)
        if item:
            self.tree_process.selection_set(item)
            self.context_menu_process.post(event.x_root, event.y_root)

    def _force_selection(self, signal_type, replica):
        """Força la selecció de rèplica per DOC o DAD."""
        selection = self.tree_process.selection()
        if not selection:
            return

        for item in selection:
            sample_name = self.tree_process.item(item, 'values')[0]

            if sample_name not in self.selected_replicas:
                continue

            # Actualitzar selecció
            if signal_type == "DOC":
                self.selected_replicas[sample_name]["sel_doc"] = replica
            else:
                self.selected_replicas[sample_name]["sel_dad"] = replica

            # Actualitzar taula
            current_values = list(self.tree_process.item(item, 'values'))
            if signal_type == "DOC":
                current_values[4] = f"R{replica}"  # Columna Sel DOC
            else:
                current_values[6] = f"R{replica}"  # Columna Sel DAD

            self.tree_process.item(item, values=current_values, tags=('MODIFIED',))

            # Regenerar PNG amb la nova selecció
            self._regenerate_qaqc_plot(sample_name)

    def _regenerate_qaqc_plot(self, sample_name):
        """Regenera el gràfic QAQC amb la selecció actual."""
        if sample_name not in self.selected_replicas:
            return

        data = self.selected_replicas[sample_name]
        check_folder = ensure_check_folder(self.seq_path)
        self._generate_qaqc_plot(sample_name, data, check_folder)

    def _open_qaqc_plot(self, event):
        """Obre el gràfic QAQC al fer doble-clic."""
        selection = self.tree_process.selection()
        if not selection:
            return

        item = selection[0]
        sample_name = self.tree_process.item(item, 'values')[0]
        self._open_qaqc_plot_file(sample_name)

    def _open_selected_qaqc_plot(self):
        """Obre el gràfic QAQC de la mostra seleccionada (des del menú)."""
        selection = self.tree_process.selection()
        if not selection:
            return

        item = selection[0]
        sample_name = self.tree_process.item(item, 'values')[0]
        self._open_qaqc_plot_file(sample_name)

    def _open_qaqc_plot_file(self, sample_name):
        """Obre el fitxer PNG del gràfic QAQC."""
        check_folder = ensure_check_folder(self.seq_path)
        png_path = os.path.join(check_folder, f"QAQC_{sample_name}.png")

        if os.path.exists(png_path):
            if platform.system() == "Windows":
                os.startfile(png_path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", png_path])
            else:
                subprocess.call(["xdg-open", png_path])
        else:
            messagebox.showwarning("Gràfic no trobat",
                f"No s'ha trobat el gràfic per {sample_name}.\n"
                f"Processa les mostres primer.")

    def _show_detail(self, event):
        """Mostra detall d'una mostra (legacy, ara obre el PNG)."""
        self._open_qaqc_plot(event)
        return

        selection = self.tree_process.selection()
        if not selection:
            return

        item = selection[0]
        sample_name = self.tree_process.item(item, 'values')[0]

        if sample_name not in self.selected_replicas:
            return

        data = self.selected_replicas[sample_name]
        self._open_detail_window(sample_name, data)

    def _open_detail_window(self, sample_name, data):
        """Obre finestra de detall amb DOC i DAD."""
        detail_win = tk.Toplevel(self.root)
        detail_win.title(f"Detall: {sample_name}")
        detail_win.geometry("1000x800")

        # Crear figura amb 3 subplots
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.2], hspace=0.3, wspace=0.3)

        ax_doc = fig.add_subplot(gs[0, :])  # DOC a dalt (span 2 columnes)
        ax_dad = fig.add_subplot(gs[1, 0])  # DAD a l'esquerra
        ax_comp = fig.add_subplot(gs[1, 1]) # Comparació a la dreta
        ax_tbl = fig.add_subplot(gs[2, :])  # Taula a baix

        colors = {"1": COLORS["primary"], "2": COLORS["secondary"]}
        eval_selected = data.get("eval", {})
        is_bp = eval_selected.get("is_bp", False)
        x_max = 20 if is_bp else 70

        # Gràfic DOC amb totes les rèpliques
        for rep_id, eval_data in data["all_evals"].items():
            if not eval_data.get("valid"):
                continue

            t = eval_data.get("t", [])
            y = eval_data.get("y", [])
            color = colors.get(rep_id, "gray")
            is_selected = data["rep"] == rep_id

            ax_doc.plot(t, y, color=color,
                        label=f"R{rep_id} (Score: {eval_data['score']:.1f})",
                        linewidth=2 if is_selected else 1,
                        alpha=1.0 if is_selected else 0.4)

        ax_doc.set_xlim(0, x_max)
        ax_doc.set_ylim(bottom=0)
        ax_doc.set_title(f"{sample_name} - DOC ({'BP' if is_bp else 'COLUMN'})", fontsize=12, fontweight='bold')
        ax_doc.set_xlabel("Temps (min)")
        ax_doc.set_ylabel("DOC")
        ax_doc.legend(loc='upper right')
        ax_doc.grid(True, alpha=0.3)

        # Gràfic DAD 254nm
        dad_eval = eval_selected.get("dad_eval", {})
        if dad_eval.get("valid"):
            dad_signals = dad_eval.get("signals", {})
            if 254 in dad_signals:
                t_dad = dad_signals[254].get("t", [])
                y_dad = dad_signals[254].get("y_baseline", [])
                if len(t_dad) > 0:
                    ax_dad.plot(t_dad, y_dad, color='purple', linewidth=1.5, label="DAD 254nm")
                    ax_dad.fill_between(t_dad, 0, y_dad, alpha=0.3, color='purple')
                    ax_dad.set_xlim(0, x_max)
                    ax_dad.set_ylim(bottom=0)
                    ax_dad.set_xlabel("Temps (min)")
                    ax_dad.set_ylabel("Absorbància")
                    ax_dad.set_title("DAD 254nm (rèplica seleccionada)")
                    ax_dad.legend()
                    ax_dad.grid(True, alpha=0.3)
        else:
            ax_dad.text(0.5, 0.5, "DAD no disponible", ha='center', va='center',
                       fontsize=12, color='gray')
            ax_dad.axis('off')

        # Gràfic de comparació (barres)
        comparison = data.get("replica_comparison", {})
        if comparison.get("n_valid", 0) > 1:
            metrics = []
            values = []

            if comparison.get("pearson_doc") is not None:
                metrics.append("Pearson\nDOC")
                values.append(comparison["pearson_doc"])

            if comparison.get("pearson_dad") is not None:
                metrics.append("Pearson\nDAD")
                values.append(comparison["pearson_dad"])

            if metrics:
                colors_bar = [COLORS["success"] if v > 0.95 else COLORS["warning"] for v in values]
                bars = ax_comp.barh(metrics, values, color=colors_bar, height=0.5)
                ax_comp.axvline(x=0.95, color='red', linestyle='--', linewidth=1, label="Llindar 0.95")
                ax_comp.set_xlim(0.8, 1.0)
                ax_comp.set_xlabel("Correlació Pearson")
                ax_comp.set_title("Comparació Rèpliques")

                for bar, val in zip(bars, values):
                    ax_comp.text(val - 0.01, bar.get_y() + bar.get_height()/2,
                                f'{val:.4f}', va='center', ha='right', fontsize=9, fontweight='bold')

                # Afegir info diferència àrea
                if comparison.get("area_diff_pct") is not None:
                    ax_comp.text(0.85, 0.1, f"Dif. àrea DOC: {comparison['area_diff_pct']:.1f}%",
                                transform=ax_comp.transAxes, fontsize=9)

                ax_comp.legend(loc='lower right', fontsize=8)
        else:
            ax_comp.text(0.5, 0.5, "Només 1 rèplica vàlida\nNo es pot comparar",
                        ha='center', va='center', fontsize=10, color='gray')
            ax_comp.axis('off')

        # Taula de dades
        ax_tbl.axis('off')
        table_data = [["Rèplica", "Score", "TimeOUT", "Batman", "Ears", "Àrea", "Conc (ppm)", "DAD OK", "Selecció"]]
        for rep_id, eval_data in data["all_evals"].items():
            is_sel = data["rep"] == rep_id
            if eval_data.get("valid"):
                conc = eval_data.get("conc_ppm", 0)
                dad_ok = "Sí" if eval_data.get("dad_eval", {}).get("valid") else "No"
                row = [
                    f"R{rep_id}",
                    f"{eval_data['score']:.1f}",
                    "SÍ" if eval_data.get("has_timeout") else "-",
                    "SÍ" if eval_data.get("has_batman") else "-",
                    str(eval_data.get("n_ears", 0)),
                    f"{eval_data.get('area', 0):.1f}",
                    f"{conc:.2f}" if conc > 0 else "-",
                    dad_ok,
                    "SELECCIONADA" if is_sel else ""
                ]
            else:
                row = [f"R{rep_id}", "INVALID", "-", "-", "-", "-", "-", "-", ""]
            table_data.append(row)

        table = ax_tbl.table(cellText=table_data, loc='center', cellLoc='center',
                            colWidths=[0.08, 0.08, 0.1, 0.1, 0.08, 0.1, 0.12, 0.1, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor(COLORS["primary"])
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Marcar fila seleccionada
        for i, row in enumerate(table_data[1:], start=1):
            if "SELECCIONADA" in row:
                for j in range(len(table_data[0])):
                    table[(i, j)].set_facecolor('#d4edda')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=detail_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        plt.close(fig)

    # =========================================================================
    # PAS 4: EXPORTACIÓ
    # =========================================================================
    def _run_export(self):
        """Executa l'exportació."""
        if self.is_processing:
            return

        self.is_processing = True
        self.btn_exportar.configure(state="disabled")

        thread = threading.Thread(target=self._export_thread)
        thread.start()

    def _export_thread(self):
        """Thread d'exportació."""
        try:
            # Crear carpetes
            final_path = os.path.join(self.seq_path, "FINAL")
            check_path = os.path.join(self.seq_path, "CHECK")
            os.makedirs(final_path, exist_ok=True)
            os.makedirs(check_path, exist_ok=True)

            total = len(self.selected_replicas)
            count = 0

            # Exportar Excel finals
            if self.var_export_excel.get():
                for sample_name, data in self.selected_replicas.items():
                    count += 1
                    progress = int(50 * count / total)

                    self.root.after(0, lambda p=progress: self.progress_exp.configure(value=p))

                    if not data["path"]:
                        continue

                    self._export_sample(sample_name, data, final_path)

            # Generar PDF
            if self.var_export_pdf.get():
                self.root.after(0, lambda: self.progress_exp.configure(value=75))
                self._generate_pdf_report(check_path)

            # Exportar KPIs
            if self.var_export_kpis.get():
                self.root.after(0, lambda: self.progress_exp.configure(value=90))
                self._export_kpis_summary(final_path)

            self.root.after(0, self._export_done)

        except Exception as e:
            self.root.after(0, lambda: self._export_error(str(e)))

    def _export_sample(self, sample_name, data, output_path):
        """
        Exporta una mostra individual combinant DOC i DAD de rèpliques potencialment diferents.
        """
        sel_doc = data.get("sel_doc", data.get("rep", "1"))
        sel_dad = data.get("sel_dad", sel_doc)
        reps = data.get("reps", {})
        all_evals = data.get("all_evals", {})

        # Obtenir eval DOC de la rèplica seleccionada
        eval_doc = all_evals.get(sel_doc, data.get("eval", {}))
        kpis = eval_doc.get("kpis", {})

        output_file = os.path.join(output_path, f"{sample_name}_F.xlsx")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # ID amb info de selecció DOC/DAD
            df_id = pd.DataFrame({
                "Camp": [
                    "Mostra", "Data",
                    "Font DOC", "Font DAD",
                    "Score DOC", "TimeOUT", "Batman", "Orelletes", "Àrea"
                ],
                "Valor": [
                    sample_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    f"Rèplica {sel_doc}",
                    f"Rèplica {sel_dad}",
                    f"{eval_doc.get('score', 0):.1f}",
                    "SI" if eval_doc.get("has_timeout") else "NO",
                    "SI" if eval_doc.get("has_batman") else "NO",
                    str(eval_doc.get("n_ears", 0)),
                    f"{kpis.get('total', 0):.2f}"
                ]
            })
            df_id.to_excel(writer, sheet_name='ID', index=False)

            # DOC de la rèplica seleccionada per DOC
            if "t" in eval_doc and "y" in eval_doc:
                df_doc = pd.DataFrame({
                    "time (min)": eval_doc["t"],
                    "DOC": eval_doc["y"]
                })
                df_doc.to_excel(writer, sheet_name='DOC', index=False)

            # DAD de la rèplica seleccionada per DAD (pot ser diferent!)
            if sel_dad in reps:
                df_dad = self._load_dad_dataframe(reps[sel_dad])
                if not df_dad.empty:
                    # Aplicar neteja de baseline al DAD
                    for col in df_dad.columns:
                        if 'time' not in str(col).lower():
                            try:
                                df_dad[col] = netejar_baseline(pd.to_numeric(df_dad[col], errors='coerce').values)
                            except:
                                pass
                    df_dad.to_excel(writer, sheet_name='DAD', index=False)

            # KPIs (només si no és KHP)
            if not eval_doc.get("is_khp", False):
                kpi_rows = []
                for nom in ["total"] + list(self.config["time_fractions"].keys()):
                    area = kpis.get(nom, 0)
                    pct = kpis.get(f"{nom}_pct", 100 if nom == "total" else 0)
                    kpi_rows.append([nom, f"{area:.2f}", f"{pct:.1f}"])

                df_kpis = pd.DataFrame(kpi_rows, columns=["Fracció", "Àrea", "Percentatge (%)"])
                df_kpis.to_excel(writer, sheet_name='KPIS', index=False)

    def _generate_pdf_report(self, output_path):
        """Genera l'informe PDF."""
        pdf_path = os.path.join(output_path, "REPORT_HPSEC.pdf")

        with PdfPages(pdf_path) as pdf:
            # Portada
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(111)
            ax.axis('off')

            ax.text(0.5, 0.7, "INFORME HPSEC", fontsize=28, fontweight='bold',
                   ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.55, f"Seqüència: {os.path.basename(self.seq_path)}", fontsize=14,
                   ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.45, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=12,
                   ha='center', transform=ax.transAxes)

            total = len(self.selected_replicas)
            ok = sum(1 for v in self.selected_replicas.values()
                    if v["eval"].get("valid") and not v["eval"].get("has_timeout") and not v["eval"].get("has_batman"))

            ax.text(0.5, 0.30, f"Total mostres: {total}", fontsize=14, ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.22, f"OK: {ok}  |  Amb anomalies: {total - ok}", fontsize=14, ha='center', transform=ax.transAxes)

            if self.calibration_data:
                ax.text(0.5, 0.12, f"Factor calibració: {self.calibration_data['slope']:.4f} (R²={self.calibration_data['r2']:.4f})",
                       fontsize=11, ha='center', transform=ax.transAxes)

            pdf.savefig(fig, facecolor='white')
            plt.close(fig)

            # Pàgines per mostra (màxim 20)
            for i, (sample_name, data) in enumerate(list(self.selected_replicas.items())[:20]):
                if not data["eval"].get("valid"):
                    continue

                fig, ax = plt.subplots(figsize=(11, 8))

                eval_data = data["eval"]
                t = eval_data.get("t", [])
                y = eval_data.get("y", [])

                ax.plot(t, y, color=COLORS["primary"], linewidth=1.5)
                ax.set_title(f"{sample_name} (R{data['rep']}, Score: {eval_data['score']:.1f})")
                ax.set_xlabel("Temps (min)")
                ax.set_ylabel("DOC")
                ax.grid(True, alpha=0.3)

                fig.tight_layout()
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

    def _export_kpis_summary(self, output_path):
        """Exporta resum de KPIs."""
        rows = []
        fractions = list(self.config["time_fractions"].keys())

        for sample_name, data in self.selected_replicas.items():
            if not data["eval"].get("valid"):
                continue

            kpis = data["eval"].get("kpis", {})
            row = {"Mostra": sample_name, "Total": kpis.get("total", 0)}
            for nom in fractions:
                row[nom] = kpis.get(nom, 0)
                row[f"{nom} (%)"] = kpis.get(f"{nom}_pct", 0)
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(os.path.join(output_path, "KPIS_Summary.xlsx"), index=False)

    def _export_done(self):
        """Exportació completada."""
        self.is_processing = False
        self.progress_exp['value'] = 100
        self.btn_exportar.configure(state="normal")

        messagebox.showinfo("Exportació",
                           f"Exportació completada!\n\n"
                           f"Fitxers a: {os.path.join(self.seq_path, 'FINAL')}\n"
                           f"Informe a: {os.path.join(self.seq_path, 'CHECK')}")

    def _export_error(self, error):
        """Error en exportació."""
        self.is_processing = False
        self.btn_exportar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant l'exportació:\n{error}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    root = tk.Tk()
    app = HPSECSuite(root)
    root.mainloop()


if __name__ == "__main__":
    main()
