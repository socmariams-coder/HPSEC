"""
Batch Process Script for HPSEC Suite
=====================================
Script per processar múltiples carpetes SEQ de forma massiva.

Funcionalitat:
1. Seleccionar carpeta arrel (GUI)
2. Detectar subcarpetes SEQ (les que tenen *.D)
3. Confirmar amb l'usuari (mostrar llista)
4. Per cada SEQ:
   - Esborrar: CHECK/, CONSOLIDAT/, FINAL/, Resultats_*
   - Consolidar (versió simplificada)
   - Calibrar (si hi ha KHP)
5. Guardar log + mostrar resum
"""

import os
import sys
import glob
import re
import shutil
import time
import gc
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Importacions des de HPSEC_Suite
from HPSEC_Suite import (
    # Config
    DEFAULT_CONFIG,
    COLORS,
    PDF_HEADER,
    LOGO_PATH,
    # Utilitats bàsiques
    obtenir_seq,
    seq_tag,
    is_bp_seq,
    list_dad_files,
    netejar_nom_uib,
    clean_sample_name,
    split_sample_rep,
    normalize_key,
    skip_sample_direct,
    is_khp,
    mode_robust,
    ensure_check_folder,
    # Lectura
    llegir_doc_uib,
    llegir_dad_export3d,
    llegir_master_direct,
    trobar_excel_mestre,
    extract_doc_from_master,
    read_master_date,
    # Processament
    get_baseline_correction,
    apply_smoothing,
    process_dad,
    detect_main_peak,
    choose_best_candidate,
    normalize_rep,
    # Calibració
    find_khp_in_folder,
    analizar_khp_lote,
    # Històric KHP
    register_calibration,
    save_khp_history,
    load_khp_history,
    # Històric Mostres
    analyze_sample_areas,
    create_sample_entry,
    add_samples_batch,
    get_active_calibration,
    # Funcions compartides
    write_consolidated_excel,
    generate_chromatograms_pdf,
)

# =============================================================================
# CONSTANTS
# =============================================================================

FOLDERS_TO_DELETE = [
    "CHECK",
    "CONSOLIDAT",
    "FINAL",
    "Resultats_Consolidats",
    "Resultats_Finals",
]

VERSION = "1.1.0"

# Volums d'injecció per mode i seqüència
# BP: sempre 100 µL
# COLUMN SEQ 256-274: 100 µL
# COLUMN resta: 400 µL
INJECTION_VOLUME_BP = 100  # µL
INJECTION_VOLUME_COL_DEFAULT = 400  # µL
INJECTION_VOLUME_COL_RANGE = (256, 274)  # SEQs amb 100 µL en COLUMN
INJECTION_VOLUME_COL_SPECIAL = 100  # µL per SEQs 256-274


def get_injection_volume(seq_path, is_bp):
    """
    Determina el volum d'injecció segons mode i número de seqüència.

    Args:
        seq_path: Ruta a la carpeta SEQ
        is_bp: Si és mode ByPass

    Returns:
        int: Volum d'injecció en µL
    """
    if is_bp:
        return INJECTION_VOLUME_BP

    # Extreure número de seqüència
    seq_num = obtenir_seq(seq_path)
    try:
        seq_int = int(re.match(r'(\d+)', seq_num).group(1))
    except (AttributeError, ValueError):
        # Si no es pot parsejar, usar default
        return INJECTION_VOLUME_COL_DEFAULT

    # Comprovar si està en el rang especial (256-274)
    if INJECTION_VOLUME_COL_RANGE[0] <= seq_int <= INJECTION_VOLUME_COL_RANGE[1]:
        return INJECTION_VOLUME_COL_SPECIAL

    return INJECTION_VOLUME_COL_DEFAULT

# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================

def is_seq_folder(folder_path):
    """Detecta si una carpeta és una seqüència SEQ (conté carpetes *.D)."""
    if not os.path.isdir(folder_path):
        return False

    # Buscar subcarpetes amb extensió .D
    d_folders = glob.glob(os.path.join(folder_path, "*.D"))
    if d_folders:
        return True

    # O carpetes Export3D / CSV típiques d'una SEQ
    has_export3d = os.path.isdir(os.path.join(folder_path, "Export3D"))
    has_csv = os.path.isdir(os.path.join(folder_path, "CSV"))

    return has_export3d or has_csv


def detect_seq_folders(root_folder):
    """Detecta totes les subcarpetes SEQ dins d'una carpeta arrel."""
    seq_folders = []

    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path) and is_seq_folder(item_path):
            seq_folders.append(item_path)

    return sorted(seq_folders)


def extract_calibration_from_existing(seq_path, config=None, log_func=None):
    """
    Extreu informació de calibració d'una SEQ ja consolidada.

    Returns:
        dict amb dades de calibració o None
    """
    config = config or DEFAULT_CONFIG

    res_cons = os.path.join(seq_path, "Resultats_Consolidats")
    if not os.path.exists(res_cons):
        return None

    # Trobar mode (BP/COL) i doc_mode (UIB/DIRECT) des d'un fitxer consolidat
    xlsx_files = glob.glob(os.path.join(res_cons, "*.xlsx"))
    if not xlsx_files:
        return None

    mode = "COL"
    doc_mode = "?"
    is_bp = False

    # Llegir un fitxer per obtenir mode
    for f in xlsx_files[:1]:
        try:
            df_id = pd.read_excel(f, "ID", engine="openpyxl")
            id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))
            method = str(id_dict.get("Method", "COLUMN"))
            is_bp = method.upper() == "BP"
            mode = "BP" if is_bp else "COL"
            doc_mode = str(id_dict.get("DOC_MODE", "?"))
            break
        except:
            pass

    # Buscar i analitzar KHP
    khp_files = find_khp_in_folder(res_cons)
    if not khp_files:
        return {'has_khp': False, 'mode': mode, 'doc_mode': doc_mode}

    try:
        khp_data = analizar_khp_lote(khp_files, config)
        if khp_data is None:
            return {'has_khp': False, 'mode': mode, 'doc_mode': doc_mode}

        conc = khp_data.get('conc_ppm', 0)
        area = khp_data.get('area', 0)
        shift = khp_data.get('shift_sec', 0)

        volume = get_injection_volume(seq_path, is_bp)

        if conc > 0 and area > 0:
            factor = conc / area
            factor_norm = factor * (volume / 100.0)

            if log_func:
                log_func(f"  KHP: {len(khp_files)}f | {mode}/{doc_mode} {volume}µL | F100={factor_norm:.6f}")

            return {
                'has_khp': True,
                'mode': mode,
                'doc_mode': doc_mode,
                'volume': volume,
                'conc_ppm': conc,
                'area': area,
                'factor': factor,
                'factor_norm': factor_norm,
                'shift': shift,
                'khp_full_data': khp_data,  # Dades completes per register_calibration
                'is_bp': is_bp,
            }
    except Exception as e:
        if log_func:
            log_func(f"  Error llegint KHP: {e}")

    return {'has_khp': False, 'mode': mode, 'doc_mode': doc_mode}


def check_consolidation_status(seq_path, max_age_hours=4):
    """
    Comprova si una SEQ ja està consolidada i quant de temps fa.

    Args:
        seq_path: Ruta a la carpeta SEQ
        max_age_hours: Màxim d'hores per considerar "recent"

    Returns:
        (status_str, is_recent): Estat per mostrar i si és recent
    """
    res_cons = os.path.join(seq_path, "Resultats_Consolidats")

    if not os.path.exists(res_cons):
        return "Pendent", False

    xlsx_files = glob.glob(os.path.join(res_cons, "*.xlsx"))
    if not xlsx_files:
        return "Pendent (buit)", False

    # Trobar el fitxer més recent
    try:
        newest = max(xlsx_files, key=os.path.getmtime)
        mtime = os.path.getmtime(newest)
        age_hours = (time.time() - mtime) / 3600

        n_files = len(xlsx_files)
        khp_files = [f for f in xlsx_files if is_khp(os.path.basename(f))]

        if age_hours < max_age_hours:
            age_str = f"{age_hours:.1f}h" if age_hours >= 1 else f"{age_hours*60:.0f}min"
            return f"Consolidat ({n_files}f, {age_str})", True
        else:
            days = age_hours / 24
            return f"Consolidat ({n_files}f, {days:.0f}d)", False

    except Exception:
        return f"Consolidat ({len(xlsx_files)}f)", False


def clean_seq_folder(seq_path, log_func=None):
    """
    Neteja carpetes antigues d'una SEQ amb gestió robusta d'errors.

    Mètodes (en ordre):
    1. shutil.rmtree directe
    2. Tancar handles + retry
    3. Windows: rd /s /q
    4. Fitxer per fitxer amb chmod
    5. Renombrar com a últim recurs
    """
    deleted = []
    is_windows = sys.platform == 'win32'

    for folder_name in FOLDERS_TO_DELETE:
        folder_path = os.path.join(seq_path, folder_name)
        if not os.path.exists(folder_path):
            continue

        success = False

        # Intent 1: rmtree directe
        try:
            shutil.rmtree(folder_path)
            success = True
        except Exception:
            pass

        # Intent 2: Tancar handles i retry
        if not success and os.path.exists(folder_path):
            gc.collect()
            time.sleep(0.3)
            try:
                shutil.rmtree(folder_path)
                success = True
            except Exception:
                pass

        # Intent 3: Windows - usar rd /s /q (més efectiu)
        if not success and is_windows and os.path.exists(folder_path):
            try:
                import subprocess
                # /s = recursiu, /q = quiet (no preguntar)
                result = subprocess.run(
                    ['cmd', '/c', 'rd', '/s', '/q', folder_path],
                    capture_output=True,
                    timeout=30
                )
                if not os.path.exists(folder_path):
                    success = True
            except Exception:
                pass

        # Intent 4: Fitxer per fitxer amb permisos
        if not success and os.path.exists(folder_path):
            try:
                for root, dirs, files in os.walk(folder_path, topdown=False):
                    for name in files:
                        try:
                            file_path = os.path.join(root, name)
                            os.chmod(file_path, 0o777)
                            os.remove(file_path)
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            dir_path = os.path.join(root, name)
                            os.chmod(dir_path, 0o777)
                            os.rmdir(dir_path)
                        except Exception:
                            pass
                os.chmod(folder_path, 0o777)
                os.rmdir(folder_path)
                if not os.path.exists(folder_path):
                    success = True
            except Exception:
                pass

        # Intent 5: Renombrar (últim recurs)
        if not success and os.path.exists(folder_path):
            try:
                timestamp = datetime.now().strftime('%H%M%S')
                old_name = f"{folder_path}_OLD_{timestamp}"
                os.rename(folder_path, old_name)
                if log_func:
                    log_func(f"    Renombrat: {folder_name}/ -> *_OLD_{timestamp}")
                deleted.append(f"{folder_name}(renamed)")
                continue
            except Exception as e:
                if log_func:
                    log_func(f"    WARN: No s'ha pogut netejar {folder_name}/")
                continue

        if success:
            deleted.append(folder_name)
            if log_func:
                log_func(f"    Esborrat: {folder_name}/")

    return deleted


# =============================================================================
# FUNCIÓ DE CONSOLIDACIÓ (VERSIÓ SIMPLIFICADA)
# =============================================================================

# =============================================================================
# GENERACIÓ PDF CONSOLIDACIÓ
# =============================================================================

def generate_consolidation_pdf(seq_path, info, xlsx_files, config=None, log_func=None):
    """
    Genera PDF de control de consolidació (versió batch sense GUI).

    Args:
        seq_path: Ruta a la carpeta SEQ
        info: Dict amb 'mode', 'bp', 'seq', 'date'
        xlsx_files: Llista de fitxers Excel consolidats
        config: Configuració (opcional)
        log_func: Funció de logging (opcional)
    """
    config = config or DEFAULT_CONFIG

    try:
        check_path = ensure_check_folder(seq_path)
        seq_name = info.get('seq', obtenir_seq(seq_path))
        pdf_path = os.path.join(check_path, f"REPORT_Consolidacio_{seq_name}.pdf")

        with PdfPages(pdf_path) as pdf:
            # =================================================================
            # PÀGINA 1: Portada i resum
            # =================================================================
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor('white')

            # Logo
            if os.path.exists(LOGO_PATH):
                try:
                    logo = plt.imread(LOGO_PATH)
                    ax_logo = fig.add_axes([0.35, 0.85, 0.3, 0.1])
                    ax_logo.imshow(logo)
                    ax_logo.axis('off')
                except Exception:
                    pass

            # Títol
            fig.text(0.5, 0.78, PDF_HEADER, ha='center', va='top',
                    fontsize=14, fontweight='bold', color=COLORS["primary"])
            fig.text(0.5, 0.73, "INFORME DE CONSOLIDACIÓ", ha='center', va='top',
                    fontsize=18, fontweight='bold')
            fig.text(0.5, 0.68, f"Seqüència: {seq_name}", ha='center', va='top',
                    fontsize=12, color='gray')
            fig.text(0.5, 0.64, f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",
                    ha='center', va='top', fontsize=10, color='gray')
            fig.text(0.5, 0.60, "(Processament Batch)", ha='center', va='top',
                    fontsize=9, color='gray', style='italic')

            # Estadístiques
            khp_count = sum(1 for f in xlsx_files if is_khp(os.path.basename(f)))
            dad_ok = sum(1 for f in xlsx_files if _check_dad_status(f))

            # Taula resum
            ax = fig.add_axes([0.1, 0.35, 0.8, 0.25])
            ax.axis('off')

            summary_data = [
                ["CONFIGURACIÓ", "VALOR"],
                ["Carpeta origen", os.path.basename(seq_path)],
                ["Mode DOC", info.get('mode', 'N/A')],
                ["Tipus seqüència", "By-Pass (BP)" if info.get('bp', False) else "COLUMN"],
                ["Data seqüència", str(info.get('date', 'N/A'))],
                ["Total fitxers consolidats", str(len(xlsx_files))],
                ["Mostres KHP (estàndard)", str(khp_count)],
                ["Fitxers amb DAD vàlid", f"{dad_ok} / {len(xlsx_files)}"],
            ]

            tbl = ax.table(cellText=summary_data, loc='center', cellLoc='left',
                          colWidths=[0.5, 0.4])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.0, 2.0)

            for i in range(len(summary_data)):
                if i == 0:
                    tbl[(i, 0)].set_facecolor(COLORS["primary"])
                    tbl[(i, 0)].set_text_props(color='white', fontweight='bold')
                    tbl[(i, 1)].set_facecolor(COLORS["primary"])
                    tbl[(i, 1)].set_text_props(color='white', fontweight='bold')
                else:
                    tbl[(i, 0)].set_facecolor('#f8f9fa')

            # Baseline info
            if info.get('bp', False):
                baseline_info = f"Mode BP: Finestra 0-{config['bp_baseline_win']} min"
            else:
                baseline_info = f"Mode COLUMN: Finestra 0-{config['col_baseline_start']} min"

            fig.text(0.1, 0.28, f"Correcció baseline: {baseline_info}", fontsize=9)
            fig.text(0.1, 0.24, "Suavitzat: Savitzky-Golay (finestra=11, ordre=3)", fontsize=9)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # =================================================================
            # PÀGINA 2: Taula de parells DOC-DAD
            # =================================================================
            fig2 = plt.figure(figsize=(8.27, 11.69))
            fig2.patch.set_facecolor('white')
            fig2.text(0.5, 0.95, "Parells DOC-DAD Consolidats", ha='center',
                     fontsize=14, fontweight='bold')

            ax2 = fig2.add_axes([0.02, 0.05, 0.96, 0.87])
            ax2.axis('off')

            # Crear taula de parells
            headers = ["Mostra", "Rep", "Font DOC", "Fitxer DAD", "Pts DOC", "Pts DAD"]
            rows = []
            row_status = []

            for f in sorted(xlsx_files)[:35]:
                try:
                    df_id = pd.read_excel(f, "ID", engine="openpyxl")
                    id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

                    mostra = str(id_dict.get("Mostra", ""))[:15]
                    if is_khp(mostra):
                        mostra = f"{mostra}*"

                    rep = str(id_dict.get("Rèplica", "-"))

                    # Font DOC
                    if info.get('mode') == 'UIB':
                        doc_file = str(id_dict.get("Fitxer_DOC_Original", "-"))
                        doc_file = doc_file[-20:] if len(doc_file) > 20 else doc_file
                    else:
                        master_file = str(id_dict.get("Fitxer_Master", ""))
                        if master_file:
                            master_name = os.path.basename(master_file)[-15:]
                        else:
                            master_name = "Excel"
                        row_start = id_dict.get("DOC_ROW_START", "?")
                        row_end = id_dict.get("DOC_ROW_END", "?")
                        if row_start != "?" and row_end != "?":
                            doc_file = f"{master_name} [{row_start}-{row_end}]"
                        else:
                            doc_file = master_name

                    dad_file = str(id_dict.get("Fitxer_DAD_Original", "-"))
                    if dad_file != "NO_DATA" and dad_file != "-":
                        dad_file = dad_file[-18:] if len(dad_file) > 18 else dad_file
                    else:
                        dad_file = "MISSING"

                    n_doc = str(id_dict.get("DOC_N_POINTS", "-"))
                    n_dad = str(id_dict.get("DAD_N_POINTS", "-"))

                    # Determinar estat
                    status = "ok"
                    if n_doc == "-" or n_doc == "0" or int(n_doc or 0) < 10:
                        status = "missing_doc"
                    if dad_file == "MISSING" or n_dad == "-" or n_dad == "0":
                        if status == "ok":
                            status = "missing_dad"
                        else:
                            status = "missing_both"

                    rows.append([mostra, rep, doc_file, dad_file, n_doc, n_dad])
                    row_status.append(status)
                except:
                    continue

            if rows:
                table_data = [headers] + rows
                tbl2 = ax2.table(cellText=table_data, loc='upper center', cellLoc='center',
                                colWidths=[0.14, 0.05, 0.28, 0.22, 0.08, 0.08])
                tbl2.auto_set_font_size(False)
                tbl2.set_fontsize(7)
                tbl2.scale(1.0, 1.4)

                for j in range(len(headers)):
                    tbl2[(0, j)].set_facecolor(COLORS["primary"])
                    tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

                # Marcar files segons estat
                for i, (row, status) in enumerate(zip(rows, row_status), start=1):
                    if '*' in row[0]:  # KHP
                        for j in range(len(headers)):
                            tbl2[(i, j)].set_facecolor('#d4edda')
                    elif status.startswith("missing"):
                        for j in range(len(headers)):
                            tbl2[(i, j)].set_facecolor('#f8d7da')
                        if status == "missing_dad" or status == "missing_both":
                            tbl2[(i, 3)].set_text_props(color='red', fontweight='bold')
                        if status == "missing_doc" or status == "missing_both":
                            tbl2[(i, 4)].set_text_props(color='red', fontweight='bold')

            fig2.text(0.1, 0.02, "* KHP (estàndard) | Vermell = MISSING DATA", fontsize=8, style='italic')

            pdf.savefig(fig2, dpi=150)
            plt.close(fig2)

        if log_func:
            log_func(f"  PDF generat: CHECK/REPORT_Consolidacio_{seq_name}.pdf")

        return pdf_path

    except Exception as e:
        if log_func:
            log_func(f"  WARN: Error generant PDF: {e}")
        return None


# generate_chromatograms_pdf: importada de HPSEC_Suite


def _check_dad_status(filepath):
    """Comprova si un fitxer té DAD vàlid."""
    try:
        df_id = pd.read_excel(filepath, "ID", engine="openpyxl")
        id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))
        return str(id_dict.get("Estat_DAD", "")) == "OK"
    except:
        return False


def generate_calibration_pdf(seq_path, cal_data, config=None, log_func=None, khp_full_data=None):
    """
    Genera PDF de calibració per una SEQ (versió batch amb gràfics de TOTS els replicats).

    Args:
        seq_path: Ruta a la carpeta SEQ
        cal_data: Dict amb dades de calibració (factor, area, conc, etc.)
        config: Configuració
        log_func: Funció de logging
        khp_full_data: Dict complet de KHP amb cromatogrames i replicats
    """
    config = config or DEFAULT_CONFIG

    try:
        check_path = ensure_check_folder(seq_path)
        seq_name = obtenir_seq(seq_path)
        pdf_path = os.path.join(check_path, f"REPORT_Calibracio_{seq_name}.pdf")

        with PdfPages(pdf_path) as pdf:
            # === PÀGINA 1: Resum calibració amb tots els replicats ===
            fig = plt.figure(figsize=(8.27, 11.69))
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
            fig.text(0.5, 0.80, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                    ha='center', fontsize=10, color='#888888')

            # Línia separadora
            ax_line = fig.add_axes([0.1, 0.77, 0.8, 0.001])
            ax_line.axhline(y=0, color='#2E86AB', linewidth=2)
            ax_line.axis('off')

            mode = cal_data.get('mode', 'COL')
            doc_mode = cal_data.get('doc_mode', '?')
            volume = cal_data.get('volume', 100)
            conc = cal_data.get('conc_ppm', 0)
            area = cal_data.get('area', 0)
            factor = cal_data.get('factor', 0)
            factor_norm = cal_data.get('factor_norm', 0)
            shift = cal_data.get('shift', 0)

            # Obtenir replicats
            replicas = []
            if khp_full_data:
                replicas = khp_full_data.get('replicas', khp_full_data.get('all_khp_data', []))
                if not replicas and khp_full_data.get('filename'):
                    replicas = [khp_full_data]

            n_replicas = len(replicas)
            is_average = khp_full_data.get('is_average', False) if khp_full_data else False
            selected_idx = khp_full_data.get('selected_idx', 0) if khp_full_data else 0
            rsd = khp_full_data.get('rsd', 0) if khp_full_data else 0

            # Info selecció
            if n_replicas > 1:
                if is_average:
                    selection_text = f"PROMIG de {n_replicas} replicats (RSD: {rsd:.1f}%)"
                    selection_color = '#28a745'
                else:
                    selection_text = f"Seleccionat replicat {selected_idx + 1} de {n_replicas} (RSD alt: {rsd:.1f}%)"
                    selection_color = '#ffc107'
            else:
                selection_text = "Única rèplica"
                selection_color = '#6c757d'

            fig.text(0.5, 0.74, selection_text, ha='center', fontsize=11,
                    fontweight='bold', color=selection_color)

            # Taula resum principal
            ax_table = fig.add_axes([0.1, 0.52, 0.8, 0.18])
            ax_table.axis('off')

            table_data = [
                ['PARÀMETRE', 'VALOR'],
                ['Mode cromatogràfic', mode],
                ['Mode DOC', doc_mode],
                ['Volum injecció', f"{volume} µL"],
                ['Concentració KHP', f"{conc} ppm"],
                ['Àrea (seleccionada)', f"{area:.2f}"],
                ['Factor calibració', f"{factor:.6f}"],
                ['Factor normalitzat (100µL)', f"{factor_norm:.6f}"],
            ]

            cell_colors = [['#2E86AB', '#2E86AB']]
            for _ in table_data[1:]:
                cell_colors.append(['#f8f9fa', 'white'])

            tbl = ax_table.table(cellText=table_data, cellColours=cell_colors,
                                colWidths=[0.5, 0.5], loc='center', cellLoc='left')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.8)

            for j in range(2):
                tbl[(0, j)].set_text_props(color='white', fontweight='bold')

            # === TAULA COMPARATIVA DE TOTS ELS REPLICATS ===
            if n_replicas > 0:
                fig.text(0.5, 0.48, "COMPARACIÓ DE REPLICATS", ha='center', fontsize=12,
                        fontweight='bold', color='#2E86AB')

                ax_reps = fig.add_axes([0.05, 0.20, 0.9, 0.26])
                ax_reps.axis('off')

                # Capçalera
                rep_header = ['#', 'Fitxer', 'DOC', 'A254', 'Ratio', 'SNR', 'SHIFT(s)', 'Estat']
                rep_data = [rep_header]

                for i, rep in enumerate(replicas):
                    fname = rep.get('filename', f'Rep{i+1}')
                    if len(fname) > 20:
                        fname = fname[:17] + '...'

                    rep_area = rep.get('area', 0)
                    rep_a254 = rep.get('a254_area', 0)
                    rep_ratio = rep.get('a254_doc_ratio', 0)
                    rep_snr = rep.get('snr', 0)
                    rep_shift = rep.get('shift_sec', rep.get('shift_min', 0) * 60)

                    # Determinar estat
                    quality_score = rep.get('quality_score', 0)
                    if quality_score >= 50:
                        estat = 'REVISAR'
                    elif quality_score >= 10:
                        estat = 'ALERTA'
                    else:
                        estat = 'OK'

                    # Marcar seleccionat
                    if n_replicas > 1:
                        if is_average:
                            num = f"{i+1}"
                        elif i == selected_idx:
                            num = f"{i+1} *"
                        else:
                            num = f"{i+1}"
                    else:
                        num = "1"

                    rep_data.append([
                        num,
                        fname,
                        f"{rep_area:.1f}",
                        f"{rep_a254:.1f}" if rep_a254 > 0 else "-",
                        f"{rep_ratio:.3f}" if rep_ratio > 0 else "-",
                        f"{rep_snr:.1f}",
                        f"{rep_shift:.1f}",
                        estat
                    ])

                # Colors per files
                rep_colors = [['#2E86AB']*8]  # Header
                for i, rep in enumerate(replicas):
                    quality_score = rep.get('quality_score', 0)
                    if quality_score >= 50:
                        row_color = '#f8d7da'  # Vermell clar
                    elif quality_score >= 10:
                        row_color = '#fff3cd'  # Groc clar
                    else:
                        row_color = '#d4edda'  # Verd clar

                    # Marcar seleccionat amb fons diferent
                    if n_replicas > 1 and not is_average and i == selected_idx:
                        row_color = '#cce5ff'  # Blau clar

                    rep_colors.append([row_color]*8)

                tbl_reps = ax_reps.table(cellText=rep_data, cellColours=rep_colors,
                                        colWidths=[0.05, 0.22, 0.10, 0.10, 0.10, 0.08, 0.10, 0.10],
                                        loc='center', cellLoc='center')
                tbl_reps.auto_set_font_size(False)
                tbl_reps.set_fontsize(8)
                tbl_reps.scale(1, 1.6)

                for j in range(8):
                    tbl_reps[(0, j)].set_text_props(color='white', fontweight='bold')

            # Llegenda columnes
            fig.text(0.1, 0.16, "Llegenda:", fontsize=9, fontweight='bold')
            fig.text(0.1, 0.13, "• DOC: Àrea integrada del pic DOC (carboni orgànic dissolt)", fontsize=8)
            fig.text(0.1, 0.11, "• A254: Àrea integrada del pic DAD a 254nm (UV)", fontsize=8)
            fig.text(0.1, 0.09, "• Ratio: DOC/A254 - validació creuada de la integració", fontsize=8)
            fig.text(0.1, 0.07, "• * = Rèplica seleccionada quan RSD > 10%", fontsize=8, color='#0066cc')

            # Fórmula
            fig.text(0.55, 0.13, "Fórmula:", fontsize=9, fontweight='bold')
            fig.text(0.55, 0.10, f"Conc = Àrea × {factor:.6f}", fontsize=9, family='monospace')
            fig.text(0.55, 0.07, f"(normalitzat: × {factor_norm:.6f} × 100/Vol)", fontsize=8, family='monospace')

            # Referència al PDF de cromatogrames per validar patrons
            fig.text(0.1, 0.04, "VALIDACIÓ:", fontsize=8, fontweight='bold', color='#2E86AB')
            fig.text(0.19, 0.04, f"Consulteu REPORT_Cromatogrames_{seq_name}.pdf per revisar patrons individuals",
                    fontsize=8, color='#333333')

            # Peu de pàgina
            fig.text(0.5, 0.01, f"HPSEC Batch Processor v{VERSION}", ha='center', fontsize=8, color='#888888')

            pdf.savefig(fig, facecolor='white')
            plt.close(fig)

            # === PÀGINES INDIVIDUALS PER CADA REPLICAT ===
            is_bp = mode == 'BP'
            x_max = 20 if is_bp else 70

            for i, rep in enumerate(replicas):
                t_doc = rep.get('t_doc', [])
                y_doc = rep.get('y_doc', [])

                if t_doc is None or y_doc is None:
                    continue
                if not (len(t_doc) > 0 and len(y_doc) > 0):
                    continue

                # Convertir a numpy arrays 1D
                t_doc = np.asarray(t_doc).flatten()
                y_doc = np.asarray(y_doc).flatten()

                if len(t_doc) < 10 or len(y_doc) < 10:
                    continue

                fig_rep, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
                fig_rep.patch.set_facecolor('white')

                # Títol amb indicació de selecció
                fname = rep.get('filename', f'Replicat {i+1}')
                quality_score = rep.get('quality_score', 0)

                if n_replicas > 1:
                    if is_average:
                        title_suffix = "(inclòs al promig)"
                        title_color = '#28a745'
                    elif i == selected_idx:
                        title_suffix = "★ SELECCIONAT"
                        title_color = '#007bff'
                    else:
                        title_suffix = "(descartat)"
                        title_color = '#6c757d'
                else:
                    title_suffix = ""
                    title_color = '#333333'

                fig_rep.suptitle(f"Replicat {i+1}/{n_replicas}: {fname} {title_suffix}",
                               fontsize=12, fontweight='bold', color=title_color, y=0.98)

                # Alertes a dalt
                issues = rep.get('quality_issues', [])
                if issues:
                    alert_text = "⚠️ ALERTES: " + " | ".join(issues)
                    alert_color = '#dc3545' if quality_score >= 50 else '#ffc107'
                    fig_rep.text(0.5, 0.94, alert_text, ha='center', fontsize=9,
                               color=alert_color, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor=alert_color))

                peak_info = rep.get('peak_info', {})
                t_dad = rep.get('t_dad')
                y_dad = rep.get('y_dad_254')

                # === Gràfic 1: DOC complet ===
                ax1 = axes[0, 0]
                ax1.plot(t_doc, y_doc, 'b-', linewidth=1.5, label='DOC')
                ax1.fill_between(t_doc, 0, y_doc, alpha=0.3, color='blue')
                ax1.set_xlim(0, x_max)
                ax1.set_xlabel('Temps (min)')
                ax1.set_ylabel('DOC (mAU)')
                ax1.set_title('Senyal DOC complet')
                ax1.grid(True, alpha=0.3)

                if peak_info.get('valid'):
                    t_max = peak_info.get('t_max', 0)
                    ax1.axvline(x=t_max, color='red', linestyle='--', alpha=0.7)
                    ax1.scatter([t_max], [peak_info.get('height', 0)], color='red', s=50, zorder=5)

                # Marcar tots els pics
                all_peaks = rep.get('all_peaks', [])
                for pk in all_peaks:
                    ax1.plot(pk['t'], pk['height'], 'ro', markersize=4, alpha=0.5)

                ax1.legend(loc='upper right', fontsize=8)

                # === Gràfic 2: DOC + DAD ===
                ax2 = axes[0, 1]
                ax2.plot(t_doc, y_doc, 'b-', linewidth=1.5, label='DOC')
                ax2.set_xlim(0, x_max)
                ax2.set_xlabel('Temps (min)')
                ax2.set_ylabel('DOC (mAU)', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')

                if t_dad is not None and y_dad is not None and len(t_dad) > 0:
                    t_dad = np.array(t_dad) if not isinstance(t_dad, np.ndarray) else t_dad
                    y_dad = np.array(y_dad) if not isinstance(y_dad, np.ndarray) else y_dad

                    ax2b = ax2.twinx()
                    ax2b.plot(t_dad, y_dad, 'r-', linewidth=1.5, label='DAD 254nm', alpha=0.7)
                    ax2b.set_ylabel('DAD 254nm (mAU)', color='red')
                    ax2b.tick_params(axis='y', labelcolor='red')

                    shift_sec = rep.get('shift_sec', rep.get('shift_min', 0) * 60)
                    ax2.set_title(f'DOC vs DAD 254nm (SHIFT: {shift_sec:.1f}s)')
                else:
                    ax2.set_title('DOC (sense DAD)')

                ax2.grid(True, alpha=0.3)

                # === Gràfic 3: Zoom al pic ===
                ax3 = axes[1, 0]
                if peak_info.get('valid'):
                    t_max = peak_info.get('t_max', 25)
                    zoom_start = max(0, t_max - 3)
                    zoom_end = min(max(t_doc), t_max + 3)

                    mask_zoom = (t_doc >= zoom_start) & (t_doc <= zoom_end)
                    if np.any(mask_zoom):
                        ax3.plot(t_doc[mask_zoom], y_doc[mask_zoom], 'b-', linewidth=2)

                        # Ombrar NOMÉS l'àrea d'integració (entre límits del pic)
                        t_start = peak_info.get('t_start', 0)
                        t_end = peak_info.get('t_end', 0)
                        left_idx = peak_info.get('left_idx', 0)
                        right_idx = peak_info.get('right_idx', len(t_doc) - 1)

                        if t_start > 0 and t_end > t_start:
                            # Assegurar índexs són enters vàlids
                            left_idx = int(max(0, min(left_idx, len(t_doc) - 1)))
                            right_idx = int(max(0, min(right_idx, len(t_doc) - 1)))

                            # Usar índexs dels límits d'integració, limitats a la finestra de zoom
                            zoom_left_idx = int(np.searchsorted(t_doc, zoom_start))
                            zoom_right_idx = int(np.searchsorted(t_doc, zoom_end))

                            # Límits d'integració dins la finestra visible
                            int_left = max(left_idx, zoom_left_idx)
                            int_right = min(right_idx, zoom_right_idx)

                            # Assegurar bounds vàlids
                            int_left = int(max(0, min(int_left, len(t_doc) - 1)))
                            int_right = int(max(0, min(int_right, len(t_doc) - 1)))

                            if int_right > int_left:
                                ax3.fill_between(t_doc[int_left:int_right+1], 0,
                                               y_doc[int_left:int_right+1],
                                               alpha=0.4, color='green')

                        # Marcar límits integració amb línies verticals
                        if t_start > 0:
                            ax3.axvline(x=t_start, color='green', linestyle='-', linewidth=2, alpha=0.8)
                        if t_end > 0:
                            ax3.axvline(x=t_end, color='green', linestyle='-', linewidth=2, alpha=0.8)

                        # Pic principal
                        ax3.axvline(x=t_max, color='red', linestyle='--', linewidth=2)

                        rep_area = rep.get('area', 0)
                        ax3.set_title(f'Zoom: {t_max:.2f} min | Àrea: {rep_area:.2f}')

                ax3.set_xlabel('Temps (min)')
                ax3.set_ylabel('DOC (mAU)')
                ax3.grid(True, alpha=0.3)

                # === Gràfic 4: Mètriques qualitat ===
                ax4 = axes[1, 1]
                ax4.axis('off')

                symmetry = rep.get('symmetry', 1.0)
                snr = rep.get('snr', 0)
                shift_sec = rep.get('shift_sec', rep.get('shift_min', 0) * 60)
                rep_area = rep.get('area', 0)
                n_peaks = rep.get('all_peaks_count', 1)
                has_batman = rep.get('has_batman', False)
                has_timeout = rep.get('has_timeout', False)

                threshold = config.get('alignment_threshold_sec', 5)

                # Determinar estats
                sym_ok = 0.8 <= symmetry <= 1.5
                snr_ok = snr > 10
                shift_ok = abs(shift_sec) <= threshold

                quality_data = [
                    ['Mètrica', 'Valor', 'Estat'],
                    ['Simetria', f"{symmetry:.2f}", '✓ OK' if sym_ok else '⚠️ REVISAR'],
                    ['SNR', f"{snr:.1f}", '✓ OK' if snr_ok else '⚠️ BAIX'],
                    ['SHIFT (s)', f"{shift_sec:.1f}", '✓ OK' if shift_ok else '⚠️ ALINEAR'],
                    ['Àrea', f"{rep_area:.2f}", '-'],
                    ['Pics detectats', f"{n_peaks}", '✓ OK' if n_peaks <= 3 else '⚠️ MÚLTIPLES'],
                    ['Doble pic', 'Sí' if has_batman else 'No', '❌ BATMAN' if has_batman else '✓ OK'],
                    ['TimeOUT', 'Sí' if has_timeout else 'No', '❌ REVISAR' if has_timeout else '✓ OK'],
                    ['Puntuació qualitat', f"{quality_score}", '✓' if quality_score < 10 else '⚠️' if quality_score < 50 else '❌'],
                ]

                # Colors
                cell_colors2 = [['#2E86AB']*3]
                for row in quality_data[1:]:
                    estat = row[2]
                    if estat.startswith('✓') or estat == '-':
                        cell_colors2.append(['white', 'white', '#d4edda'])
                    elif estat.startswith('⚠️'):
                        cell_colors2.append(['white', 'white', '#fff3cd'])
                    else:
                        cell_colors2.append(['white', 'white', '#f8d7da'])

                tbl2 = ax4.table(cellText=quality_data, cellColours=cell_colors2,
                                colWidths=[0.4, 0.3, 0.3], loc='center', cellLoc='left')
                tbl2.auto_set_font_size(False)
                tbl2.set_fontsize(9)
                tbl2.scale(1, 1.5)

                for j in range(3):
                    tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

                fig_rep.tight_layout(rect=[0, 0.02, 1, 0.92])
                pdf.savefig(fig_rep, facecolor='white')
                plt.close(fig_rep)

        if log_func:
            n_pages = 1 + len(replicas)
            log_func(f"  PDF calibració: CHECK/REPORT_Calibracio_{seq_name}.pdf ({n_pages} pàgines, {len(replicas)} replicats)")

        return pdf_path

    except Exception as e:
        if log_func:
            log_func(f"  WARN: Error generant PDF calibració: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_pdfs_for_existing(seq_path, config=None, log_func=None):
    """
    Genera PDFs de consolidació i calibració per una SEQ ja consolidada.
    Útil per SEQs saltades o per regenerar PDFs.

    Args:
        seq_path: Ruta a la carpeta SEQ
        config: Configuració
        log_func: Funció de logging

    Returns:
        Dict amb 'consolidation_pdf', 'calibration_pdf', 'info'
    """
    config = config or DEFAULT_CONFIG
    result = {'consolidation_pdf': None, 'calibration_pdf': None, 'info': {}}

    # Trobar carpeta consolidada
    res_cons = os.path.join(seq_path, "Resultats_Consolidats")
    if not os.path.exists(res_cons):
        res_cons = os.path.join(seq_path, "CONSOLIDAT")
    if not os.path.exists(res_cons):
        if log_func:
            log_func(f"  No es troba carpeta consolidada")
        return result

    xlsx_files = glob.glob(os.path.join(res_cons, "*.xlsx"))
    xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

    if not xlsx_files:
        if log_func:
            log_func(f"  No hi ha fitxers Excel consolidats")
        return result

    # Llegir info del primer fitxer
    mode = "COLUMN"
    doc_mode = "?"
    is_bp = False
    date_master = "N/A"
    seq_out = obtenir_seq(seq_path)

    for f in xlsx_files[:1]:
        try:
            df_id = pd.read_excel(f, "ID", engine="openpyxl")
            id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))
            method = str(id_dict.get("Method", "COLUMN"))
            is_bp = method.upper() == "BP"
            mode = method
            doc_mode = str(id_dict.get("DOC_MODE", "?"))
            date_master = str(id_dict.get("Date", "N/A"))
            seq_out = str(id_dict.get("SEQ", seq_out))
            break
        except:
            pass

    info = {
        'mode': doc_mode,
        'bp': is_bp,
        'seq': seq_out,
        'date': date_master
    }
    result['info'] = info

    # Generar PDF consolidació
    try:
        pdf_cons = generate_consolidation_pdf(seq_path, info, xlsx_files, config, log_func)
        result['consolidation_pdf'] = pdf_cons
    except Exception as e:
        if log_func:
            log_func(f"  WARN: Error PDF consolidació: {e}")

    # Generar PDF cromatogrames (DOC + A254)
    try:
        pdf_chrom = generate_chromatograms_pdf(seq_path, xlsx_files, log_func=log_func)
        result['chromatograms_pdf'] = pdf_chrom
    except Exception as e:
        if log_func:
            log_func(f"  WARN: Error PDF cromatogrames: {e}")

    # Buscar KHP i generar PDF calibració
    khp_files = find_khp_in_folder(res_cons)
    if khp_files:
        try:
            khp_data = analizar_khp_lote(khp_files, config)
            if khp_data:
                conc = khp_data.get('conc_ppm', 0)
                area = khp_data.get('area', 0)
                shift = khp_data.get('shift_sec', khp_data.get('shift_min', 0) * 60)
                volume = get_injection_volume(seq_path, is_bp)

                if conc > 0 and area > 0:
                    factor = conc / area
                    factor_norm = factor * (volume / 100.0)

                    cal_data = {
                        'mode': 'BP' if is_bp else 'COL',
                        'doc_mode': doc_mode,
                        'volume': volume,
                        'conc_ppm': conc,
                        'area': area,
                        'factor': factor,
                        'factor_norm': factor_norm,
                        'shift': shift,
                        'khp_full_data': khp_data,  # Dades completes per register_calibration
                        'is_bp': is_bp,
                    }

                    # Passar dades completes de KHP per gràfics
                    pdf_cal = generate_calibration_pdf(seq_path, cal_data, config, log_func, khp_full_data=khp_data)
                    result['calibration_pdf'] = pdf_cal
                    result['cal_data'] = cal_data
        except Exception as e:
            if log_func:
                log_func(f"  WARN: Error PDF calibració: {e}")

    return result


# =============================================================================
# FUNCIÓ DE CONSOLIDACIÓ (VERSIÓ SIMPLIFICADA)
# =============================================================================

# write_consolidated_excel: importada de HPSEC_Suite


def consolidate_seq(seq_path, config=None, log_func=None):
    """
    Consolida una carpeta SEQ (versió sense GUI).

    Args:
        seq_path: Ruta a la carpeta SEQ
        config: Diccionari de configuració (opcional, usa DEFAULT_CONFIG)
        log_func: Funció per logging (opcional)

    Returns:
        dict: Resultat amb 'success', 'count', 'mode', 'bp', 'error'
    """
    config = config or DEFAULT_CONFIG

    result = {
        'success': False,
        'count': 0,
        'mode': None,
        'bp': False,
        'error': None,
    }

    try:
        input_folder = seq_path
        path_csv = os.path.join(input_folder, "CSV")
        path_3d = os.path.join(input_folder, "Export3D")
        path_out = os.path.join(input_folder, "Resultats_Consolidats")

        # Crear carpeta sortida
        os.makedirs(path_out, exist_ok=True)

        # Detectar mode UIB vs DIRECT
        uib_files = []
        if os.path.isdir(path_csv):
            for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
                uib_files.extend(glob.glob(os.path.join(path_csv, f"*UIB1B*{ext}")))

        mode = "UIB" if uib_files else "DIRECT"
        result['mode'] = mode

        # Detectar BP
        dad_pool = list_dad_files(
            path_3d if os.path.isdir(path_3d) else None,
            path_csv if os.path.isdir(path_csv) else None,
        )
        bp_flag = is_bp_seq(input_folder, dad_pool)
        method = "BP" if bp_flag else "COLUMN"
        result['bp'] = bp_flag

        # Obtenir SEQ i data
        seq = obtenir_seq(input_folder)
        seq_out = seq_tag(seq, bp_flag)
        date_master = read_master_date(input_folder)

        if log_func:
            log_func(f"  Mode: {mode} | Tipus: {method} | SEQ: {seq_out}")

        used_dad_files = set()
        processed_count = 0
        sample_entries = []  # Per guardar a Samples_History.json

        if mode == "UIB":
            # MODE UIB: DOC des de CSV UIB1B
            uib_files = sorted(set(uib_files))
            total = len(uib_files)

            for i, f_doc in enumerate(uib_files):
                nom_doc = os.path.basename(f_doc)
                mostra, rep = netejar_nom_uib(nom_doc)

                df_doc, st_doc = llegir_doc_uib(f_doc)
                if df_doc.empty:
                    continue

                t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()
                y_doc = pd.to_numeric(df_doc["DOC"], errors="coerce").to_numpy()

                base = get_baseline_correction(t_doc, y_doc, method, config)
                doc_net = apply_smoothing(y_doc - base)
                doc_net[doc_net < 0] = 0

                # Buscar DAD
                dad_path, st_dad = choose_best_candidate(mostra, rep, dad_pool, used_dad_files)
                df_dad = pd.DataFrame()
                nom_dad = "NO_DATA"

                if dad_path:
                    used_dad_files.add(dad_path)
                    nom_dad = os.path.basename(dad_path)
                    df_dad_raw, st_read = llegir_dad_export3d(dad_path)
                    if st_read.startswith("OK") and not df_dad_raw.empty:
                        df_dad = process_dad(df_dad_raw, config)
                        st_dad = "OK" if not df_dad.empty else "NO_DATA"

                # Detectar pic
                peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])

                # Calcular àrees (DOC + DAD wavelengths)
                sample_analysis = analyze_sample_areas(t_doc, doc_net, df_dad, peak_info, config)

                # Guardar Excel
                out_name = f"{mostra}_{seq_out}_R{rep}.xlsx"
                out_path = os.path.join(path_out, out_name)

                write_consolidated_excel(
                    out_path, mostra, rep, seq_out, date_master, method, "UIB",
                    nom_doc, nom_dad, st_doc, st_dad, t_doc, y_doc, doc_net, base,
                    df_dad, peak_info, sample_analysis
                )

                # Afegir a llista de mostres
                sample_entry = create_sample_entry(
                    seq_path, mostra, rep, date_master, method, "UIB",
                    sample_analysis
                )
                sample_entries.append(sample_entry)

                processed_count += 1

        else:
            # MODE DIRECT: DOC des d'Excel mestre
            mestre = trobar_excel_mestre(input_folder)
            if not mestre:
                result['error'] = "No s'ha trobat Excel mestre"
                return result

            df_toc, df_seq = llegir_master_direct(mestre)

            needed = ["SAMPLE_DESC", "START", "Row initial", "Row Final"]
            for col in needed:
                if col not in df_seq.columns:
                    result['error'] = f"Falta columna: {col}"
                    return result

            df_seq2 = df_seq.copy()
            df_seq2 = df_seq2[df_seq2["SAMPLE_DESC"].apply(lambda x: not skip_sample_direct(x))]
            df_seq2 = df_seq2.dropna(subset=["Row initial", "Row Final"])

            total = len(df_seq2)
            rep_counter = {}

            for i, (_, row) in enumerate(df_seq2.iterrows()):
                sample_desc = str(row["SAMPLE_DESC"]).strip()
                base_name, rep_explicit = split_sample_rep(sample_desc)
                base_key = normalize_key(base_name)

                if rep_explicit is None:
                    rep_counter[base_key] = rep_counter.get(base_key, 0) + 1
                    rep = str(rep_counter[base_key])
                else:
                    rep = str(rep_explicit)

                mostra_clean = clean_sample_name(base_name)

                try:
                    row_ini = int(float(row["Row initial"]))
                    row_fi = int(float(row["Row Final"]))
                except Exception:
                    continue

                df_doc = extract_doc_from_master(df_toc, row_ini, row_fi, row["START"])
                if df_doc.empty:
                    continue

                t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()
                y_doc = pd.to_numeric(df_doc["DOC"], errors="coerce").to_numpy()

                base_arr = get_baseline_correction(t_doc, y_doc, method, config)
                doc_net = apply_smoothing(y_doc - base_arr)
                doc_net[doc_net < 0] = 0

                # Buscar DAD
                dad_path, st_dad = choose_best_candidate(base_name, rep, dad_pool, used_dad_files)
                df_dad = pd.DataFrame()
                nom_dad = "NO_DATA"

                if dad_path:
                    used_dad_files.add(dad_path)
                    nom_dad = os.path.basename(dad_path)
                    df_dad_raw, st_read = llegir_dad_export3d(dad_path)
                    if st_read.startswith("OK") and not df_dad_raw.empty:
                        df_dad = process_dad(df_dad_raw, config)
                        st_dad = "OK" if not df_dad.empty else "NO_DATA"

                # Detectar pic
                peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])

                # Calcular àrees (DOC + DAD wavelengths)
                sample_analysis = analyze_sample_areas(t_doc, doc_net, df_dad, peak_info, config)

                # Guardar Excel
                out_name = f"{mostra_clean}_{seq_out}_R{rep}.xlsx"
                out_path = os.path.join(path_out, out_name)

                write_consolidated_excel(
                    out_path, mostra_clean, rep, seq_out, date_master, method, "DIRECT",
                    os.path.basename(mestre), nom_dad, "OK", st_dad, t_doc, y_doc, doc_net, base_arr,
                    df_dad, peak_info, sample_analysis,
                    master_file=mestre, row_start=row_ini, row_end=row_fi
                )

                # Afegir a llista de mostres
                sample_entry = create_sample_entry(
                    seq_path, mostra_clean, rep, date_master, method, "DIRECT",
                    sample_analysis
                )
                sample_entries.append(sample_entry)

                processed_count += 1

        result['count'] = processed_count
        result['success'] = True
        result['date'] = date_master

        if log_func:
            log_func(f"  Consolidats: {processed_count} fitxers")

        # Guardar mostres a l'històric
        if sample_entries:
            try:
                add_samples_batch(seq_path, sample_entries, replace_seq=True)
                if log_func:
                    log_func(f"  Mostres registrades: {len(sample_entries)} a Samples_History.json")
            except Exception as e:
                if log_func:
                    log_func(f"  WARN: Error guardant mostres a històric: {e}")

        # Generar PDF de consolidació
        xlsx_files = glob.glob(os.path.join(path_out, "*.xlsx"))
        if xlsx_files:
            info = {
                'mode': mode,
                'bp': bp_flag,
                'seq': seq_out,
                'date': date_master
            }
            generate_consolidation_pdf(seq_path, info, xlsx_files, config, log_func)

    except Exception as e:
        result['error'] = str(e)
        if log_func:
            log_func(f"  ERROR: {e}")

    return result


# =============================================================================
# FUNCIÓ DE CALIBRACIÓ (VERSIÓ SIMPLIFICADA)
# =============================================================================

def calibrate_seq(seq_path, config=None, log_func=None, is_bp=False):
    """
    Calibra una carpeta SEQ amb KHP (versió sense GUI).

    Args:
        seq_path: Ruta a la carpeta SEQ
        config: Diccionari de configuració (opcional)
        log_func: Funció per logging (opcional)
        is_bp: Si la seqüència és mode ByPass

    Returns:
        dict: Resultat amb 'success', 'has_khp', 'factor', 'factor_norm', 'volume', 'shift', 'error'
    """
    config = config or DEFAULT_CONFIG

    result = {
        'success': False,
        'has_khp': False,
        'factor': None,
        'factor_norm': None,  # Factor normalitzat a 100 µL
        'volume': None,
        'shift': None,
        'khp_count': 0,
        'error': None,
    }

    try:
        # Buscar KHP consolidats (sense GUI)
        res_cons = os.path.join(seq_path, "Resultats_Consolidats")
        khp_files = find_khp_in_folder(res_cons)

        if not khp_files:
            # Buscar en carpetes germanes amb mateix prefix numèric
            folder_name = os.path.basename(seq_path)
            parent_dir = os.path.dirname(seq_path)

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
                            if log_func:
                                log_func(f"  KHP trobat a: {sib}")
                            break
                except Exception:
                    pass

        if not khp_files:
            if log_func:
                log_func(f"  Sense KHP - saltant calibració")
            result['has_khp'] = False
            result['success'] = True  # No és un error, simplement no hi ha KHP
            return result

        result['has_khp'] = True
        result['khp_count'] = len(khp_files)

        # Analitzar KHP
        khp_data = analizar_khp_lote(khp_files, config)

        if khp_data is None:
            result['error'] = "No s'ha pogut analitzar cap KHP"
            return result

        # Obtenir dades i calcular factor
        # Factor = concentració / àrea (per convertir àrees a ppm)
        conc = khp_data.get('conc_ppm', 0)
        area = khp_data.get('area', 0)
        shift = khp_data.get('shift_sec', 0)  # ja en segons

        # Obtenir volum d'injecció
        volume = get_injection_volume(seq_path, is_bp)
        result['volume'] = volume

        if conc > 0 and area > 0:
            # Factor raw (per aquest volum d'injecció)
            factor = conc / area

            # Factor normalitzat a 100 µL
            # Si volum > 100, l'àrea és proporcionalment més gran
            # Factor_norm = Factor * (Volume / 100)
            factor_norm = factor * (volume / 100.0)

            result['factor'] = factor
            result['factor_norm'] = factor_norm
            result['shift'] = shift
            result['conc_ppm'] = conc
            result['area'] = area
            result['success'] = True
            result['khp_full_data'] = khp_data  # Dades completes per PDF

            # Paràmetres de qualitat
            result['snr'] = khp_data.get('snr', 0)
            result['symmetry'] = khp_data.get('symmetry', 1.0)
            result['all_peaks_count'] = khp_data.get('all_peaks_count', 1)
            result['has_batman'] = khp_data.get('has_batman', False)
            result['has_timeout'] = khp_data.get('has_timeout', False)
            result['quality_issues'] = khp_data.get('quality_issues', [])
            result['n_replicas'] = khp_data.get('n_replicas', 1)
            result['rsd'] = khp_data.get('rsd', 0)
            result['is_bp'] = khp_data.get('is_bp', is_bp)

            if log_func:
                mode_str = "BP" if is_bp else "COL"
                log_func(f"  KHP: {len(khp_files)} fitxers | {mode_str} {volume}µL | Conc: {conc}ppm | Àrea: {area:.1f}")
                log_func(f"       Factor: {factor:.6f} | Factor_100µL: {factor_norm:.6f} | Shift: {shift:.1f}s")
        else:
            result['error'] = f"Dades KHP invàlides (conc={conc}, area={area})"

    except Exception as e:
        result['error'] = str(e)
        if log_func:
            log_func(f"  ERROR calibració: {e}")

    return result


# =============================================================================
# GUI PER SELECCIÓ I CONFIRMACIÓ
# =============================================================================

import threading
import queue

class BatchProcessorGUI:
    """Interfície gràfica per al processament massiu amb threading."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"HPSEC Batch Processor v{VERSION}")
        self.root.geometry("900x650")

        self.root_folder = None
        self.seq_folders = []
        self.log_lines = []
        self.results = {
            'total': 0,
            'ok': 0,
            'errors': 0,
            'no_khp': 0,
        }
        self.calibration_data = []
        self.processing = False
        self.msg_queue = queue.Queue()

        self._setup_gui()
        self._poll_queue()

    def _setup_gui(self):
        """Configura la interfície gràfica."""
        # Frame superior - selecció carpeta
        frame_top = ttk.Frame(self.root, padding=10)
        frame_top.pack(fill="x")

        ttk.Label(frame_top, text="Carpeta arrel:").pack(side="left")
        self.lbl_folder = ttk.Label(frame_top, text="(cap seleccionada)", foreground="gray")
        self.lbl_folder.pack(side="left", padx=10, expand=True, fill="x")

        self.btn_select = ttk.Button(frame_top, text="Seleccionar...", command=self._select_folder)
        self.btn_select.pack(side="right")

        # Frame opcions
        frame_opts = ttk.Frame(self.root, padding=5)
        frame_opts.pack(fill="x", padx=10)

        self.var_clean = tk.BooleanVar(value=False)
        self.chk_clean = ttk.Checkbutton(
            frame_opts,
            text="Esborrar carpetes antigues (CHECK, CONSOLIDAT, Resultats_*) abans de processar",
            variable=self.var_clean
        )
        self.chk_clean.pack(side="left")

        # Frame mig - llista SEQ
        frame_mid = ttk.LabelFrame(self.root, text="Carpetes SEQ detectades", padding=10)
        frame_mid.pack(fill="both", expand=True, padx=10, pady=5)

        # Llista amb scroll
        self.tree = ttk.Treeview(frame_mid, columns=("status",), show="tree headings", height=10)
        self.tree.heading("#0", text="Carpeta SEQ")
        self.tree.heading("status", text="Estat")
        self.tree.column("#0", width=550)
        self.tree.column("status", width=250)

        scrollbar = ttk.Scrollbar(frame_mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Frame log
        frame_log = ttk.LabelFrame(self.root, text="Log", padding=10)
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)

        self.txt_log = tk.Text(frame_log, height=12, state="disabled", font=("Consolas", 9))
        scrollbar_log = ttk.Scrollbar(frame_log, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scrollbar_log.set)

        self.txt_log.pack(side="left", fill="both", expand=True)
        scrollbar_log.pack(side="right", fill="y")

        # Frame inferior - botons
        frame_bottom = ttk.Frame(self.root, padding=10)
        frame_bottom.pack(fill="x")

        self.btn_process = ttk.Button(frame_bottom, text="Processar Tot",
                                       command=self._start_processing, state="disabled")
        self.btn_process.pack(side="left", padx=5)

        self.btn_cancel = ttk.Button(frame_bottom, text="Tancar", command=self._on_close)
        self.btn_cancel.pack(side="right", padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(frame_bottom, mode="determinate", length=300)
        self.progress.pack(side="left", padx=20, expand=True, fill="x")

        self.lbl_progress = ttk.Label(frame_bottom, text="")
        self.lbl_progress.pack(side="left", padx=5)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Tanca l'aplicació."""
        if self.processing:
            if messagebox.askyesno("Confirmar", "Hi ha un processament en curs. Vols sortir igualment?"):
                self.root.destroy()
        else:
            self.root.destroy()

    def _poll_queue(self):
        """Processa missatges de la cua (cridat periòdicament)."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg.get('type')

                if msg_type == 'log':
                    self._do_log(msg['text'])
                elif msg_type == 'progress':
                    self.progress['value'] = msg['value']
                    self.lbl_progress.config(text=msg['text'])
                elif msg_type == 'tree_status':
                    self._do_update_tree(msg['path'], msg['status'])
                elif msg_type == 'done':
                    self._on_processing_done()

        except queue.Empty:
            pass

        # Tornar a comprovar en 100ms
        self.root.after(100, self._poll_queue)

    def _select_folder(self):
        """Selecciona carpeta arrel i detecta SEQ."""
        folder = filedialog.askdirectory(title="Selecciona carpeta arrel amb SEQ")
        if not folder:
            return

        self.root_folder = folder
        self.lbl_folder.config(text=folder, foreground="black")

        # Detectar SEQ
        self.seq_folders = detect_seq_folders(folder)

        # Actualitzar llista amb info de consolidació existent
        self.tree.delete(*self.tree.get_children())

        recent_count = 0
        for seq_path in self.seq_folders:
            name = os.path.basename(seq_path)
            status, is_recent = check_consolidation_status(seq_path)
            if is_recent:
                recent_count += 1
            self.tree.insert("", "end", text=name, values=(status,), iid=seq_path)

        self._do_log(f"Detectades {len(self.seq_folders)} carpetes SEQ ({recent_count} ja consolidades)")

        if self.seq_folders:
            self.btn_process.config(state="normal")
        else:
            self.btn_process.config(state="disabled")
            messagebox.showwarning("Avís", "No s'han detectat carpetes SEQ a la ubicació seleccionada.")

    def _do_log(self, message):
        """Afegeix missatge al log (thread-safe per main thread)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_lines.append(line)

        self.txt_log.config(state="normal")
        self.txt_log.insert("end", line + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    def _log(self, message):
        """Afegeix missatge al log (thread-safe, usa cua)."""
        self.msg_queue.put({'type': 'log', 'text': message})

    def _do_update_tree(self, seq_path, status):
        """Actualitza tree (main thread)."""
        try:
            self.tree.item(seq_path, values=(status,))
        except Exception:
            pass

    def _update_tree_status(self, seq_path, status):
        """Actualitza l'estat d'una SEQ (thread-safe)."""
        self.msg_queue.put({'type': 'tree_status', 'path': seq_path, 'status': status})

    def _update_progress(self, value, text):
        """Actualitza progress bar (thread-safe)."""
        self.msg_queue.put({'type': 'progress', 'value': value, 'text': text})

    def _start_processing(self):
        """Inicia el processament massiu en un thread separat."""
        if not self.seq_folders or self.processing:
            return

        # Confirmar
        clean_msg = "\n\nS'esborraran carpetes antigues primer." if self.var_clean.get() else ""
        skip_msg = "\n\nLes SEQs consolidades recentment (<24h) es saltaran."

        if not messagebox.askyesno("Confirmar",
                                    f"Es processaran {len(self.seq_folders)} carpetes SEQ.{clean_msg}{skip_msg}\n\n"
                                    "Continuar?"):
            return

        self.btn_process.config(state="disabled")
        self.btn_select.config(state="disabled")
        self.chk_clean.config(state="disabled")
        self.processing = True

        # Reset resultats
        self.results = {'total': len(self.seq_folders), 'ok': 0, 'errors': 0, 'no_khp': 0, 'skipped': 0}
        self.calibration_data = []

        # Llançar thread de processament
        thread = threading.Thread(target=self._processing_thread, daemon=True)
        thread.start()

    def _processing_thread(self):
        """Thread de processament (no bloqueja GUI)."""
        try:
            total = len(self.seq_folders)
            config = DEFAULT_CONFIG.copy()
            do_clean = self.var_clean.get()

            for i, seq_path in enumerate(self.seq_folders):
                name = os.path.basename(seq_path)
                progress_pct = int(100 * (i + 1) / total)

                self._update_progress(progress_pct, f"[{i+1}/{total}] {name}")
                self._log(f"[{i+1}/{total}] Processant: {name}")
                self._update_tree_status(seq_path, "Processant...")

                # Comprovar si ja està consolidat recentment
                _, is_recent = check_consolidation_status(seq_path)
                if is_recent and not do_clean:
                    self._log(f"  Saltat (consolidat recentment) - generant PDFs...")

                    # Generar PDFs per SEQ existent
                    pdf_result = generate_pdfs_for_existing(seq_path, config, self._log)

                    # Recollir dades calibració
                    if pdf_result.get('cal_data'):
                        cal_info = pdf_result['cal_data']

                        # Registrar calibració a CALDATA + històric global
                        if cal_info.get('khp_full_data'):
                            try:
                                is_bp = cal_info.get('is_bp', False)
                                mode_str = 'BP' if is_bp else 'COLUMN'
                                register_calibration(
                                    seq_path,
                                    cal_info['khp_full_data'],
                                    "BATCH",
                                    mode_str
                                )
                                self._log(f"  Registrat a CALDATA i històric global")
                            except Exception as e:
                                self._log(f"  AVÍS: Error registrant calibració: {e}")

                        self.calibration_data.append({
                            'seq': name,
                            'mode': cal_info.get('mode', '?'),
                            'doc_mode': cal_info.get('doc_mode', '?'),
                            'volume': cal_info.get('volume', 0),
                            'conc_ppm': cal_info.get('conc_ppm', 0),
                            'area': cal_info.get('area', 0),
                            'factor': cal_info.get('factor', 0),
                            'factor_norm': cal_info.get('factor_norm', 0),
                            'shift': cal_info.get('shift', 0),
                        })
                        self._update_tree_status(seq_path, f"SALTAT+PDF (F100={cal_info['factor_norm']:.6f})")
                    else:
                        self._update_tree_status(seq_path, "SALTAT+PDF (sense KHP)")

                    self.results['skipped'] += 1
                    self.results['ok'] += 1
                    continue

                # Netejar si s'ha demanat
                if do_clean:
                    self._log(f"  Netejant carpetes antigues...")
                    clean_seq_folder(seq_path, self._log)

                # 1. Consolidar
                self._log(f"  Consolidant...")
                cons_result = consolidate_seq(seq_path, config, self._log)

                if not cons_result['success']:
                    self._update_tree_status(seq_path, f"ERROR: {cons_result['error']}")
                    self.results['errors'] += 1
                    continue

                # 1b. Generar PDFs de consolidació i cromatogrames
                is_bp = cons_result.get('bp', False)
                seq_out = cons_result.get('seq', obtenir_seq(seq_path))
                res_cons = os.path.join(seq_path, "Resultats_Consolidats")
                xlsx_files = glob.glob(os.path.join(res_cons, "*.xlsx"))
                xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

                info = {
                    'mode': cons_result.get('mode', '?'),
                    'bp': is_bp,
                    'seq': seq_out,
                }

                try:
                    generate_consolidation_pdf(seq_path, info, xlsx_files, config, self._log)
                except Exception as e:
                    self._log(f"  WARN: Error PDF consolidació: {e}")

                try:
                    generate_chromatograms_pdf(seq_path, xlsx_files, is_bp=is_bp, seq_name=seq_out, log_func=self._log)
                except Exception as e:
                    self._log(f"  WARN: Error PDF cromatogrames: {e}")

                # 2. Calibrar
                self._log(f"  Calibrant...")
                cal_result = calibrate_seq(seq_path, config, self._log, is_bp=is_bp)

                if cal_result['has_khp']:
                    if cal_result['success']:
                        vol = cal_result.get('volume', '?')
                        factor_norm = cal_result.get('factor_norm', 0)
                        status = f"OK ({cons_result['count']}f, {vol}µL, F100={factor_norm:.6f})"
                        self.results['ok'] += 1

                        # Preparar dades per registrar
                        mode_str = 'BP' if is_bp else 'COLUMN'
                        khp_full = cal_result.get('khp_full_data', {})

                        # Registrar calibració (crea CALDATA + actualitza global)
                        try:
                            register_calibration(
                                seq_path,
                                khp_full,  # Dades completes del KHP
                                "BATCH",   # Font
                                mode_str   # Mode
                            )
                            self._log(f"  Registrat a CALDATA i històric global")
                        except Exception as e:
                            self._log(f"  AVÍS: Error registrant calibració: {e}")

                        self.calibration_data.append({
                            'seq': name,
                            'mode': 'BP' if is_bp else 'COL',
                            'doc_mode': cons_result.get('mode', '?'),  # UIB o DIRECT
                            'volume': cal_result.get('volume', 0),
                            'conc_ppm': cal_result.get('conc_ppm', 0),
                            'area': cal_result.get('area', 0),
                            'factor': cal_result.get('factor', 0),
                            'factor_norm': cal_result.get('factor_norm', 0),
                            'shift': cal_result.get('shift', 0),
                            # Paràmetres de qualitat
                            'snr': cal_result.get('snr', 0),
                            'symmetry': cal_result.get('symmetry', 1.0),
                            'all_peaks_count': cal_result.get('all_peaks_count', 1),
                            'has_batman': cal_result.get('has_batman', False),
                            'has_timeout': cal_result.get('has_timeout', False),
                            'quality_issues': cal_result.get('quality_issues', []),
                            'n_replicas': cal_result.get('n_replicas', 1),
                            'rsd': cal_result.get('rsd', 0),
                        })

                        # Generar PDF de calibració amb gràfics
                        try:
                            pdf_cal_data = {
                                'mode': 'BP' if is_bp else 'COL',
                                'doc_mode': cons_result.get('mode', '?'),
                                'volume': cal_result.get('volume', 0),
                                'conc_ppm': cal_result.get('conc_ppm', 0),
                                'area': cal_result.get('area', 0),
                                'factor': cal_result.get('factor', 0),
                                'factor_norm': cal_result.get('factor_norm', 0),
                                'shift': cal_result.get('shift', 0),
                            }
                            # Passar dades completes de KHP per gràfics
                            khp_full = cal_result.get('khp_full_data')
                            generate_calibration_pdf(seq_path, pdf_cal_data, config, self._log, khp_full_data=khp_full)
                        except Exception as e:
                            self._log(f"  AVÍS: No s'ha pogut generar PDF calibració: {e}")
                    else:
                        status = f"WARN: {cal_result['error']}"
                        self.results['errors'] += 1
                else:
                    status = f"OK ({cons_result['count']} fitxers, sense KHP)"
                    self.results['ok'] += 1
                    self.results['no_khp'] += 1

                self._update_tree_status(seq_path, status)

            # Senyalar que hem acabat
            self.msg_queue.put({'type': 'done'})

        except Exception as e:
            self._log(f"ERROR FATAL: {e}")
            self.msg_queue.put({'type': 'done'})

    def _on_processing_done(self):
        """Cridat quan el processament acaba."""
        self.processing = False

        # Guardar calibracions a l'històric KHP_History.json
        self._save_to_khp_history()

        # Generar resum calibració
        self._generate_calibration_summary()

        # Guardar log
        self._save_log()

        # Mostrar resum
        self._show_summary()

        self.btn_process.config(state="normal")
        self.btn_select.config(state="normal")
        self.chk_clean.config(state="normal")

    def _save_to_khp_history(self):
        """Guarda les calibracions a l'històric KHP_History.json."""
        if not self.calibration_data or not self.root_folder:
            return

        try:
            # Carregar històric existent
            # L'històric es guarda a la carpeta pare (root_folder)
            history_path = os.path.join(self.root_folder, "KHP_History.json")

            existing = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        existing = data.get("calibrations", [])
                except:
                    pass

            # Afegir noves calibracions
            for cal_data in self.calibration_data:
                seq_name = cal_data['seq']
                mode = cal_data['mode']
                doc_mode = cal_data.get('doc_mode', '?')

                # Determinar estat segons qualitat
                quality_issues = cal_data.get('quality_issues', [])
                has_batman = cal_data.get('has_batman', False)
                has_timeout = cal_data.get('has_timeout', False)

                if has_batman or has_timeout:
                    status = "WARN"
                elif quality_issues:
                    status = "WARN"
                else:
                    status = "OK"

                entry = {
                    "seq_name": seq_name,
                    "seq_path": os.path.join(self.root_folder, seq_name),
                    "date": datetime.now().isoformat(),
                    "mode": mode,
                    "doc_mode": doc_mode,
                    "khp_file": f"KHP_{cal_data.get('conc_ppm', 0)}",
                    "khp_source": "BATCH",
                    "conc_ppm": cal_data.get('conc_ppm', 0),
                    "area": cal_data.get('area', 0),
                    "shift_min": cal_data.get('shift', 0) / 60.0,
                    "shift_sec": cal_data.get('shift', 0),
                    "volume_uL": cal_data.get('volume', 100),
                    "factor": cal_data.get('factor', 0),
                    "factor_100uL": cal_data.get('factor_norm', 0),
                    "symmetry": cal_data.get('symmetry', 1.0),
                    "snr": cal_data.get('snr', 0),
                    "all_peaks_count": cal_data.get('all_peaks_count', 1),
                    "has_batman": has_batman,
                    "has_timeout": has_timeout,
                    "n_replicas": cal_data.get('n_replicas', 1),
                    "rsd": cal_data.get('rsd', 0),
                    "status": status,
                    "quality_issues": quality_issues,
                }

                # Actualitzar o afegir
                updated = False
                for i, ex in enumerate(existing):
                    if ex.get("seq_name") == seq_name and ex.get("mode") == mode:
                        existing[i] = entry
                        updated = True
                        break

                if not updated:
                    existing.append(entry)

            # Ordenar per data
            existing.sort(key=lambda x: x.get("date", ""), reverse=True)

            # Guardar
            data = {
                "version": "1.0",
                "updated": datetime.now().isoformat(),
                "source": "HPSEC_Batch_Processor",
                "calibrations": existing
            }

            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._do_log(f"Històric KHP guardat: {history_path} ({len(existing)} entrades)")

        except Exception as e:
            self._do_log(f"ERROR guardant històric KHP: {e}")

    def _generate_calibration_summary(self):
        """Genera i mostra resum de calibració de totes les SEQs amb KHP."""
        if not self.calibration_data:
            self._log("\n" + "="*70)
            self._log("RESUM CALIBRACIÓ: Cap SEQ amb KHP processat")
            self._log("="*70)
            return

        self._do_log("\n" + "="*80)
        self._do_log("RESUM CALIBRACIÓ - FACTORS KHP")
        self._do_log("="*80)

        # Capçalera
        header = f"{'SEQ':<18} {'Mode':<4} {'DOC':<6} {'Vol':>4} {'Conc':>5} {'Àrea':>11} {'Factor':>11} {'F_100µL':>11} {'Shift':>6}"
        self._do_log(header)
        self._do_log("-"*80)

        # Dades ordenades per SEQ
        for d in sorted(self.calibration_data, key=lambda x: x['seq']):
            line = (
                f"{d['seq']:<18} "
                f"{d['mode']:<4} "
                f"{d.get('doc_mode', '?'):<6} "
                f"{d['volume']:>4} "
                f"{d['conc_ppm']:>5} "
                f"{d['area']:>11.1f} "
                f"{d['factor']:>11.6f} "
                f"{d['factor_norm']:>11.6f} "
                f"{d['shift']:>5.1f}s"
            )
            self._do_log(line)

        self._do_log("-"*80)

        # Estadístiques Factor_100µL per mode DOC
        factors_norm = [d['factor_norm'] for d in self.calibration_data]

        # Per UIB
        factors_uib = [d['factor_norm'] for d in self.calibration_data if d.get('doc_mode') == 'UIB']
        # Per DIRECT
        factors_direct = [d['factor_norm'] for d in self.calibration_data if d.get('doc_mode') == 'DIRECT']

        if factors_norm:
            mean_f = np.mean(factors_norm)
            std_f = np.std(factors_norm)
            rsd = (std_f / mean_f * 100) if mean_f > 0 else 0

            self._do_log(f"GLOBAL:  Mitjana={mean_f:.6f}  SD={std_f:.6f}  RSD={rsd:.1f}%  (n={len(factors_norm)})")

        if factors_uib:
            mean_uib = np.mean(factors_uib)
            std_uib = np.std(factors_uib) if len(factors_uib) > 1 else 0
            self._do_log(f"UIB:     Mitjana={mean_uib:.6f}  SD={std_uib:.6f}  (n={len(factors_uib)})")

        if factors_direct:
            mean_dir = np.mean(factors_direct)
            std_dir = np.std(factors_direct) if len(factors_direct) > 1 else 0
            self._do_log(f"DIRECT:  Mitjana={mean_dir:.6f}  SD={std_dir:.6f}  (n={len(factors_direct)})")

        self._do_log("="*80 + "\n")

        # Guardar a CSV
        self._save_calibration_csv()

    def _save_calibration_csv(self):
        """Guarda resum de calibració a CSV."""
        if not self.calibration_data or not self.root_folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.root_folder, f"calibration_summary_{timestamp}.csv")

        try:
            df = pd.DataFrame(self.calibration_data)
            # Assegurar que doc_mode existeix
            if 'doc_mode' not in df.columns:
                df['doc_mode'] = '?'
            df = df[['seq', 'mode', 'doc_mode', 'volume', 'conc_ppm', 'area', 'factor', 'factor_norm', 'shift']]
            df.columns = ['SEQ', 'Mode', 'DOC_Mode', 'Volume_uL', 'Conc_ppm', 'Area', 'Factor', 'Factor_100uL', 'Shift_s']
            df = df.sort_values('SEQ')
            df.to_csv(csv_path, index=False, sep=';', decimal=',')
            self._do_log(f"Resum calibració guardat a: {csv_path}")
        except Exception as e:
            self._do_log(f"ERROR guardant CSV calibració: {e}")

    def _save_log(self):
        """Guarda el log a fitxer."""
        if not self.root_folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.root_folder, f"batch_log_{timestamp}.txt")

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"HPSEC Batch Processor v{VERSION}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Carpeta arrel: {self.root_folder}\n")
                f.write(f"Total SEQ: {self.results['total']}\n")
                f.write(f"OK: {self.results['ok']}\n")
                f.write(f"Errors: {self.results['errors']}\n")
                f.write(f"Sense KHP: {self.results['no_khp']}\n")
                f.write(f"{'='*50}\n\n")

                for line in self.log_lines:
                    f.write(line + "\n")

            self._log(f"Log guardat a: {log_path}")
        except Exception as e:
            self._log(f"ERROR guardant log: {e}")

    def _show_summary(self):
        """Mostra resum final."""
        skipped = self.results.get('skipped', 0)
        processed = self.results['ok'] - skipped

        summary = (
            f"Processament completat!\n\n"
            f"Total SEQ: {self.results['total']}\n"
            f"Processades: {processed}\n"
            f"Saltades (recents): {skipped}\n"
            f"Errors: {self.results['errors']}\n"
            f"Sense KHP: {self.results['no_khp']}\n"
            f"Amb KHP: {len(self.calibration_data)}\n\n"
            f"Fitxers generats a carpeta arrel:\n"
            f"- batch_log_*.txt\n"
            f"- calibration_summary_*.csv"
        )
        messagebox.showinfo("Resum", summary)

    def run(self):
        """Inicia l'aplicació."""
        self.root.mainloop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Punt d'entrada principal."""
    print(f"HPSEC Batch Processor v{VERSION}")
    print("="*40)

    app = BatchProcessorGUI()
    app.run()


if __name__ == "__main__":
    main()
