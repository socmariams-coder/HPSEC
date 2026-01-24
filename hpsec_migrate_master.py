"""
hpsec_migrate_master.py - Migració de fitxers mestre antics al nou format MasterFile

Estructura nova:
  0-INFO:       Metadata + sincronització
  1-HPLC-SEQ:   Injeccions ordenades per data + Inj_Index
  2-TOC:        Dades DOC contínues (raw, intocable pels tècnics)
  3-TOC_CALC:   Fórmules: hora_HPLC, Inj_Index, Sample (llegeix de 2-TOC)
  4-DAD_KHP:    Dades DAD només de KHP (opcional, 1 o 2 rèpliques)

Autor: HPSEC Suite
Data: 2025-01
"""

import os
import shutil
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
from datetime import datetime
import glob


# Constants
FLUSH_TIME_MIN = 3.637  # Delay físic DOC-DAD en minuts (constant mesurada amb UIB)
TEMPLATE_PATH = 'C:/Users/Lequia/Desktop/Dades2/xxx_SEQ_MasterFile.xlsx'
TOC_DATA_START_ROW = 8  # Les dades TOC comencen a la fila 8


def adjust_toc_calc(ws_toc_calc, num_toc_rows, toc_data_start=8):
    """
    Ajusta el full TOC_CALC per tenir el nombre correcte de files.

    Args:
        ws_toc_calc: Worksheet de TOC_CALC
        num_toc_rows: Nombre total de files a 2-TOC
        toc_data_start: Fila on comencen les dades a 2-TOC (per defecte 8)
    """
    # Calcular quantes files de dades hi ha (excloent capçaleres)
    num_data_rows = num_toc_rows - toc_data_start + 1
    current_rows = ws_toc_calc.max_row - 1  # Excloure capçalera

    print(f"  TOC_CALC: {current_rows} files actuals, {num_data_rows} necessàries")

    if num_data_rows <= 0:
        return

    # Si necessitem més files, afegir-les
    if num_data_rows > current_rows:
        print(f"  Afegint {num_data_rows - current_rows} files...")
        for i in range(current_rows + 1, num_data_rows + 1):
            row = i + 1  # +1 per la capçalera
            toc_row = toc_data_start + i - 1

            # TOC_Row
            ws_toc_calc.cell(row=row, column=1, value=f"={toc_row}")
            # Date_Started
            ws_toc_calc.cell(row=row, column=2, value=f"='2-TOC'!D{toc_row}")
            # hora_HPLC
            ws_toc_calc.cell(row=row, column=3, value=f"=B{row}+'0-INFO'!$B$12/(24*60)")
            # Inj_Index
            ws_toc_calc.cell(row=row, column=4, value=f"=COUNTIF('1-HPLC-SEQ'!$F:$F,\"<=\"&C{row})")
            # Sample
            ws_toc_calc.cell(row=row, column=5,
                           value=f"=IFERROR(INDEX('1-HPLC-SEQ'!$D:$D,MATCH(D{row},'1-HPLC-SEQ'!$P:$P,0)),\"\")")

            # Format de data
            ws_toc_calc.cell(row=row, column=2).number_format = 'DD/MM/YYYY HH:MM:SS'
            ws_toc_calc.cell(row=row, column=3).number_format = 'DD/MM/YYYY HH:MM:SS'

    # Si sobren files, eliminar-les
    elif num_data_rows < current_rows:
        print(f"  Eliminant {current_rows - num_data_rows} files...")
        ws_toc_calc.delete_rows(num_data_rows + 2, current_rows - num_data_rows)


def create_masterfile_from_scratch(old_filepath):
    """
    Crea MasterFile des de zero (sense plantilla).

    Estructura:
      0-INFO:      Metadata + sincronització
      1-HPLC-SEQ:  Injeccions ordenades + Inj_Index
      2-TOC:       Dades raw
      3-DAD_KHP:   Dades KHP (opcional)
      TOC_CALC:    Fórmules calculades
    """
    print(f"\nCreant MasterFile: {os.path.basename(old_filepath)}")

    # Llegir dades originals
    old_data = read_old_master(old_filepath)
    if old_data is None:
        print("ERROR: No s'han pogut llegir les dades originals")
        return False

    # Extreure info
    info = extract_info_from_check(old_data.get('check'))

    # Detectar method pel nom del folder
    dir_path = os.path.dirname(old_filepath)
    folder_name = os.path.basename(dir_path).upper()
    if '_BP' in folder_name or folder_name.endswith('BP'):
        info['method'] = 'BP'

    # Detectar UIB
    filename = os.path.basename(old_filepath).upper()
    has_uib = 'UIB' in filename
    info['uib_range'] = determine_uib_range(info.get('seq_id'), has_uib)

    print(f"  SEQ-ID: {info.get('seq_id')}")
    print(f"  Method: {info.get('method')}")
    print(f"  UIB_range: {info.get('uib_range')}")

    # Generar path sortida
    seq_id = info.get('seq_id', 'UNKNOWN')
    output_path = os.path.join(dir_path, f"{seq_id}_MasterFile.xlsx")

    # Crear nou workbook
    wb = openpyxl.Workbook()

    # --- 0-INFO ---
    ws_info = wb.active
    ws_info.title = '0-INFO'

    ws_info['A1'] = 'SEQ-ID'
    ws_info['B1'] = info.get('seq_id', '')
    ws_info['A2'] = 'Date'
    ws_info['B2'] = info.get('date', '')
    ws_info['B2'].number_format = 'DD/MM/YYYY'
    ws_info['A3'] = 'Method'
    ws_info['B3'] = info.get('method', 'COLUMN')
    ws_info['A4'] = 'Inj_Volume (uL)'
    ws_info['B4'] = 100 if info.get('method') == 'BP' else 400
    ws_info['A5'] = 'UIB_range'
    ws_info['B5'] = info.get('uib_range') or 'None'

    ws_info['A7'] = 'SINCRONITZACIO HPLC-TOC'
    ws_info['A8'] = 'Hora HPLC'
    ws_info['B8'] = info.get('hora_hplc', '')
    ws_info['A9'] = 'Hora TOC'
    ws_info['B9'] = info.get('hora_toc', '')
    ws_info['A10'] = 'Desfase (min)'
    ws_info['B10'] = '=(B8-B9)*24*60'
    ws_info['A11'] = 'Flush time (min)'
    ws_info['B11'] = FLUSH_TIME_MIN
    ws_info['C11'] = '(constant)'
    ws_info['A12'] = 'Net delay (min)'
    ws_info['B12'] = '=-B10+B11'

    # --- 1-HPLC-SEQ ---
    ws_hplc = wb.create_sheet('1-HPLC-SEQ')
    hplc_date_col = 'F'  # Per defecte
    hplc_sample_col = 'D'
    hplc_index_col = 'P'

    if old_data.get('hplc') is not None:
        df_hplc = old_data['hplc'].copy()

        # Trobar columna de data i sample
        date_col = None
        sample_col_name = None
        for i, col in enumerate(df_hplc.columns):
            if 'acquired date' in col.lower():
                date_col = col
            elif 'sample' in col.lower() and 'name' in col.lower():
                sample_col_name = col

        # IMPORTANT: Filtrar només files amb Sample Name (injeccions reals)
        # Les files sense nom són dades de senyal DAD, no injeccions
        if sample_col_name:
            df_hplc = df_hplc[df_hplc[sample_col_name].notna()].copy()

        if date_col:
            df_hplc[date_col] = pd.to_datetime(df_hplc[date_col])
            df_hplc = df_hplc.sort_values(date_col).reset_index(drop=True)
            df_hplc['Inj_Index'] = range(1, len(df_hplc) + 1)  # Índex seqüencial 1, 2, 3...

            # Actualitzar columnes després del filtre
            hplc_date_col = get_column_letter(list(df_hplc.columns).index(date_col) + 1)
            hplc_sample_col = get_column_letter(list(df_hplc.columns).index(sample_col_name) + 1)

            # Crear Sample_Rep amb lògica de blocs per evitar duplicats
            inj_num_col = None
            for col in df_hplc.columns:
                if col == 'Inj#':
                    inj_num_col = col

            if inj_num_col:
                df_hplc['Inj_Num'] = df_hplc[inj_num_col].fillna(1).astype(int)

                # Detectar combinacions Sample+Inj# duplicades
                df_hplc['_temp_key'] = df_hplc[sample_col_name].astype(str) + '_R' + df_hplc['Inj_Num'].astype(str)
                duplicate_keys = df_hplc['_temp_key'].value_counts()
                samples_need_blocks = set(df_hplc.loc[df_hplc['_temp_key'].isin(duplicate_keys[duplicate_keys > 1].index), sample_col_name])

                # Crear Sample_Rep amb blocs per mostres amb duplicats
                sample_rep_list = []
                sample_block_counter = {}

                for _, row in df_hplc.iterrows():
                    sample = row[sample_col_name]
                    inj_num = row['Inj_Num']

                    if sample in samples_need_blocks:
                        # Mostra amb duplicats: afegir número de bloc
                        if sample not in sample_block_counter:
                            sample_block_counter[sample] = 0

                        # Incrementar bloc quan Inj_Num torna a 1
                        if inj_num == 1:
                            sample_block_counter[sample] += 1

                        block_num = sample_block_counter[sample]
                        sample_rep = f"{sample}_B{block_num}_R{inj_num}"
                    else:
                        # Mostra única: només _R1/_R2
                        sample_rep = f"{sample}_R{inj_num}"

                    sample_rep_list.append(sample_rep)

                df_hplc['Sample_Rep'] = sample_rep_list
                df_hplc.drop(columns=['_temp_key'], inplace=True)
                hplc_sample_col = get_column_letter(list(df_hplc.columns).index('Sample_Rep') + 1)

            hplc_index_col = get_column_letter(list(df_hplc.columns).index('Inj_Index') + 1)

        # Escriure dades
        for r_idx, row in enumerate(dataframe_to_rows(df_hplc, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                # Evitar escriure NaN/None - deixar cel·la buida
                if pd.isna(value):
                    continue
                # Escapar strings que comencin amb = (evitar interpretar com fórmula)
                if isinstance(value, str) and value.startswith('='):
                    value = "'" + value
                ws_hplc.cell(row=r_idx, column=c_idx, value=value)

        print(f"  1-HPLC-SEQ: {len(df_hplc)} files")

    # --- 2-TOC ---
    ws_toc = wb.create_sheet('2-TOC')
    num_toc_rows = 0

    if old_data.get('toc') is not None:
        df_toc = old_data['toc']
        num_toc_rows = len(df_toc)
        for r_idx, row in enumerate(dataframe_to_rows(df_toc, index=False, header=False), 1):
            for c_idx, value in enumerate(row, 1):
                # Evitar escriure NaN/None - deixar cel·la buida
                if pd.isna(value):
                    continue
                # Escapar strings que comencin amb = (evitar interpretar com fórmula)
                if isinstance(value, str) and value.startswith('='):
                    value = "'" + value  # Prefix amb ' per forçar text
                ws_toc.cell(row=r_idx, column=c_idx, value=value)

        print(f"  2-TOC: {num_toc_rows} files")

    # --- 3-DAD_KHP ---
    khp_data = extract_khp_from_dad(old_data.get('dad'))
    if khp_data:
        ws_khp = wb.create_sheet('3-DAD_KHP')
        col_offset = 1
        for khp_name, df_khp in khp_data.items():
            ws_khp.cell(row=1, column=col_offset, value=khp_name)
            ws_khp.cell(row=2, column=col_offset, value='time (min)')
            ws_khp.cell(row=2, column=col_offset + 1, value='value (mAU)')
            for r_idx, row in df_khp.iterrows():
                ws_khp.cell(row=r_idx + 3, column=col_offset, value=row['time'])
                ws_khp.cell(row=r_idx + 3, column=col_offset + 1, value=row['value'])
            col_offset += 3
        print(f"  3-DAD_KHP: {len(khp_data)} KHPs")

    # --- TOC_CALC ---
    # Pre-calcular valors (no fórmules) per màxima velocitat
    ws_toc_calc = wb.create_sheet('TOC_CALC')

    # Capçaleres
    ws_toc_calc['A1'] = 'TOC_Row'
    ws_toc_calc['B1'] = 'Sample'
    ws_toc_calc['C1'] = 'Temps_Relatiu (min)'
    ws_toc_calc['D1'] = 'Inj_Index'

    # Preparar dades per càlcul
    num_data_rows = num_toc_rows - TOC_DATA_START_ROW + 1
    print(f"  TOC_CALC: calculant {num_data_rows} valors...")

    if old_data.get('toc') is not None and old_data.get('hplc') is not None:
        df_toc = old_data['toc']
        df_hplc_orig = old_data['hplc'].copy()

        # Filtrar HPLC a només injeccions amb nom
        sample_col = None
        date_col = None
        for col in df_hplc_orig.columns:
            if 'sample' in col.lower() and 'name' in col.lower():
                sample_col = col
            elif 'acquired date' in col.lower():
                date_col = col

        if sample_col and date_col:
            df_hplc_filt = df_hplc_orig[df_hplc_orig[sample_col].notna()].copy()
            df_hplc_filt[date_col] = pd.to_datetime(df_hplc_filt[date_col])
            df_hplc_filt = df_hplc_filt.sort_values(date_col).reset_index(drop=True)

            # Crear Sample_Rep amb lògica de blocs per evitar duplicats
            inj_num_col = 'Inj#' if 'Inj#' in df_hplc_filt.columns else None
            if inj_num_col:
                df_hplc_filt['Inj_Num'] = df_hplc_filt[inj_num_col].fillna(1).astype(int)

                # Detectar combinacions Sample+Inj# duplicades
                df_hplc_filt['_temp_key'] = df_hplc_filt[sample_col].astype(str) + '_R' + df_hplc_filt['Inj_Num'].astype(str)
                duplicate_keys = df_hplc_filt['_temp_key'].value_counts()
                samples_need_blocks = set(df_hplc_filt.loc[df_hplc_filt['_temp_key'].isin(duplicate_keys[duplicate_keys > 1].index), sample_col])

                # Crear Sample_Rep amb blocs per mostres amb duplicats
                sample_rep_list = []
                sample_block_counter = {}

                for _, row in df_hplc_filt.iterrows():
                    sample = row[sample_col]
                    inj_num = row['Inj_Num']

                    if sample in samples_need_blocks:
                        if sample not in sample_block_counter:
                            sample_block_counter[sample] = 0
                        if inj_num == 1:
                            sample_block_counter[sample] += 1
                        block_num = sample_block_counter[sample]
                        sample_rep = f"{sample}_B{block_num}_R{inj_num}"
                    else:
                        sample_rep = f"{sample}_R{inj_num}"

                    sample_rep_list.append(sample_rep)

                df_hplc_filt['Sample_Rep'] = sample_rep_list
            else:
                df_hplc_filt['Sample_Rep'] = df_hplc_filt[sample_col].astype(str)

            # Arrays per càlcul ràpid
            hplc_times = df_hplc_filt[date_col].values
            hplc_samples = df_hplc_filt['Sample_Rep'].values

            # Obtenir net_delay de 0-INFO (ja calculat)
            net_delay_min = FLUSH_TIME_MIN  # Aproximació inicial

            # Llegir timestamps TOC (columna D, index 3)
            toc_timestamps = pd.to_datetime(df_toc.iloc[TOC_DATA_START_ROW-1:, 3])

            # Calcular per cada fila TOC
            for i, toc_time in enumerate(toc_timestamps):
                row = i + 2  # Fila Excel (1=header)
                toc_row = TOC_DATA_START_ROW + i

                if pd.isna(toc_time):
                    continue

                # hora_HPLC = toc_time + net_delay
                hora_hplc = toc_time + pd.Timedelta(minutes=net_delay_min)

                # Inj_Index = quantes injeccions HPLC <= hora_HPLC
                inj_index = int((hplc_times <= hora_hplc).sum())

                # Sample i Temps_Relatiu
                if inj_index > 0 and inj_index <= len(hplc_samples):
                    sample = hplc_samples[inj_index - 1]
                    inj_start = pd.Timestamp(hplc_times[inj_index - 1])
                    temps_rel = (hora_hplc - inj_start).total_seconds() / 60.0
                else:
                    sample = ''
                    temps_rel = None

                # Escriure valors
                ws_toc_calc.cell(row=row, column=1, value=toc_row)
                ws_toc_calc.cell(row=row, column=2, value=sample)
                if temps_rel is not None:
                    ws_toc_calc.cell(row=row, column=3, value=round(temps_rel, 3))
                ws_toc_calc.cell(row=row, column=4, value=inj_index)

    # Guardar
    wb.save(output_path)
    print(f"  MasterFile creat: {output_path}")

    return True


def extract_seq_number(seq_id):
    """Extreu el número de seqüència del SEQ-ID (ex: '283_SEQ' -> 283)."""
    if not seq_id:
        return None
    import re
    # Buscar primer número al principi del nom
    match = re.match(r'^(\d+)', str(seq_id))
    if match:
        return int(match.group(1))
    return None


def determine_uib_range(seq_id, has_uib_data):
    """
    Determina el rang UIB basat en el número de seqüència.

    - Sense UIB: None
    - SEQ 269-274: 700 ppb
    - SEQ >= 275: 1000 ppb
    - SEQ < 269: None (no tenien UIB)
    """
    if not has_uib_data:
        return None

    seq_num = extract_seq_number(seq_id)
    if seq_num is None:
        return None

    if seq_num < 269:
        return None  # Abans de UIB
    elif seq_num <= 274:
        return '700 ppb'
    else:
        return '1000 ppb'


def backup_original(filepath):
    """Crea còpia de seguretat del fitxer original."""
    if not os.path.exists(filepath):
        return None

    # Generar nom backup amb timestamp
    base, ext = os.path.splitext(filepath)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{base}_backup_{timestamp}{ext}"

    shutil.copy2(filepath, backup_path)
    print(f"Backup creat: {backup_path}")
    return backup_path


def read_old_master(filepath):
    """Llegeix les dades del mestre antic."""
    data = {}

    try:
        # Llegir amb openpyxl per preservar fórmules de SEQ_DATA
        wb_formulas = openpyxl.load_workbook(filepath)
        data['workbook'] = wb_formulas

        # 0-CHECK
        try:
            df_check = pd.read_excel(filepath, sheet_name='0-CHECK', header=None)
            data['check'] = df_check
        except:
            data['check'] = None

        # 1-HPLC-SEQ_RAW
        try:
            df_hplc = pd.read_excel(filepath, sheet_name='1-HPLC-SEQ_RAW')
            data['hplc'] = df_hplc
        except:
            data['hplc'] = None

        # 2-TOC
        try:
            df_toc = pd.read_excel(filepath, sheet_name='2-TOC', header=None)
            data['toc'] = df_toc
        except:
            data['toc'] = None

        # 4-SEQ_DATA - guardar referència al worksheet (no pandas)
        try:
            data['seq_data_ws'] = wb_formulas['4-SEQ_DATA']
        except:
            data['seq_data_ws'] = None

        # 3-DAD_data (per extreure KHP)
        try:
            df_dad = pd.read_excel(filepath, sheet_name='3-DAD_data', header=None)
            data['dad'] = df_dad
        except:
            data['dad'] = None

    except Exception as e:
        print(f"Error llegint {filepath}: {e}")
        return None

    return data


def extract_info_from_check(df_check):
    """Extreu informació del full 0-CHECK antic."""
    info = {
        'seq_id': None,
        'date': None,
        'method': None,
        'hora_hplc': None,
        'hora_toc': None,
        'inj_volume': None,
        'run_time': None,
    }

    if df_check is None:
        return info

    # Buscar valors per etiquetes
    for i, row in df_check.iterrows():
        val0 = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
        val1 = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else None

        if 'SEQ-ID' in val0 or 'SEQ_ID' in val0:
            info['seq_id'] = val1
        elif val0 == 'Date':
            info['date'] = val1
        elif 'HPLC' in val0 and 'TOC' not in val0 and 'delay' not in val0.lower():
            info['hora_hplc'] = val1
        elif 'TOC' in val0 and 'HPLC' not in val0 and 'Flush' not in val0:
            info['hora_toc'] = val1
        # Detectar temps d'execució del mètode principal (Aquagel = 75, Bypass = 10)
        elif 'Aquagel' in val0:
            info['run_time'] = val1  # Hauria de ser 75
        elif 'Bypass' in val0 and info['run_time'] is None:
            info['run_time'] = val1  # Hauria de ser 10

    # Detectar method: PRIMER pel nom de la SEQ, DESPRÉS pel temps d'execució
    if info['seq_id']:
        seq_name = str(info['seq_id']).upper()
        if '_BP' in seq_name or 'SEQ_BP' in seq_name or seq_name.endswith('BP'):
            info['method'] = 'BP'
        else:
            info['method'] = 'COLUMN'
    elif info['run_time']:
        # Si no tenim nom, usar temps: 75 = COLUMN, 10 = BP
        if info['run_time'] == 75:
            info['method'] = 'COLUMN'
        elif info['run_time'] == 10:
            info['method'] = 'BP'
        else:
            info['method'] = 'COLUMN'  # Default
    else:
        info['method'] = 'COLUMN'  # Default

    return info


def extract_khp_from_dad(df_dad):
    """Extreu només les dades de KHP del full DAD."""
    if df_dad is None:
        return None

    khp_data = {}
    khp_count = 0

    # Buscar columnes amb KHP a la fila 0 o 1 (índex i nom)
    for col in range(df_dad.shape[1]):
        idx_val = df_dad.iloc[0, col] if pd.notna(df_dad.iloc[0, col]) else None
        name_val = df_dad.iloc[1, col] if df_dad.shape[0] > 1 and pd.notna(df_dad.iloc[1, col]) else None

        # Detectar si és KHP
        is_khp = False

        if name_val and 'KHP' in str(name_val).upper():
            is_khp = True
            khp_count += 1
            # Usar índex per distingir múltiples KHPs amb el mateix nom
            khp_name = f"{name_val}_R{khp_count}"

        if is_khp and col + 1 < df_dad.shape[1]:
            # Extreure dades (columna següent hauria de tenir els valors)
            # L'estructura és: col=index, col+1=temps?, col+2...col+9=valors per canal
            # Però només agafem temps i valor principal (254nm normalment)

            # Buscar columnes amb dades numèriques per aquest KHP
            time_col = None
            val_col = None

            for c in range(col, min(col + 10, df_dad.shape[1])):
                col_data = df_dad.iloc[2:, c].dropna()
                if len(col_data) > 100:
                    # Té dades
                    if time_col is None:
                        time_col = c
                    elif val_col is None:
                        val_col = c
                        break

            if time_col is not None and val_col is not None:
                times = pd.to_numeric(df_dad.iloc[2:, time_col], errors='coerce')
                vals = pd.to_numeric(df_dad.iloc[2:, val_col], errors='coerce')

                df_khp = pd.DataFrame({
                    'time': times.values,
                    'value': vals.values
                }).dropna()

                if len(df_khp) > 0:
                    khp_data[khp_name] = df_khp

    return khp_data if khp_data else None


def calculate_net_delay(hora_hplc, hora_toc):
    """Calcula el Net delay a partir de les hores."""
    if hora_hplc is None or hora_toc is None:
        return FLUSH_TIME_MIN  # Si no hi ha hores, usar només flush time

    try:
        # Convertir a minuts des de mitjanit
        if hasattr(hora_hplc, 'hour'):
            hplc_min = hora_hplc.hour * 60 + hora_hplc.minute
        else:
            parts = str(hora_hplc).split(':')
            hplc_min = int(parts[0]) * 60 + int(parts[1])

        if hasattr(hora_toc, 'hour'):
            toc_min = hora_toc.hour * 60 + hora_toc.minute
        else:
            parts = str(hora_toc).split(':')
            toc_min = int(parts[0]) * 60 + int(parts[1])

        desfase = hplc_min - toc_min  # en minuts

        # Net delay = -desfase + flush_time
        net_delay = -desfase + FLUSH_TIME_MIN

        return net_delay

    except Exception as e:
        print(f"Error calculant Net delay: {e}")
        return FLUSH_TIME_MIN


def create_new_master(old_data, info, output_path, inj_volume=None):
    """Crea el nou MasterFile."""

    wb = openpyxl.Workbook()

    # --- 0-INFO ---
    ws_info = wb.active
    ws_info.title = '0-INFO'

    # Metadata
    ws_info['A1'] = 'SEQ-ID'
    ws_info['B1'] = info.get('seq_id', '')
    ws_info['A2'] = 'Date'
    ws_info['B2'] = info.get('date', '')
    ws_info['B2'].number_format = 'DD/MM/YYYY'
    ws_info['A3'] = 'Method'
    ws_info['B3'] = info.get('method', 'COLUMN')
    ws_info['A4'] = 'Inj_Volume (uL)'
    ws_info['B4'] = inj_volume if inj_volume else (400 if info.get('method') == 'COLUMN' else 100)
    ws_info['A5'] = 'UIB_range'
    uib_range = info.get('uib_range')
    ws_info['B5'] = uib_range if uib_range else 'None'

    # Sincronització
    ws_info['A7'] = 'SINCRONITZACIO HPLC-TOC'
    ws_info['A8'] = 'Hora HPLC'
    ws_info['B8'] = info.get('hora_hplc', '')
    ws_info['A9'] = 'Hora TOC'
    ws_info['B9'] = info.get('hora_toc', '')
    ws_info['A10'] = 'Desfase (min)'
    ws_info['B10'] = '=(B8-B9)*24*60'  # Fórmula
    ws_info['A11'] = 'Flush time (min)'
    ws_info['B11'] = FLUSH_TIME_MIN
    ws_info['C11'] = '(constant)'
    ws_info['A12'] = 'Net delay (min)'
    ws_info['B12'] = '=-B10+B11'  # Fórmula: -desfase + flush_time

    # --- 1-HPLC-SEQ ---
    ws_hplc = wb.create_sheet('1-HPLC-SEQ')
    if old_data.get('hplc') is not None:
        df_hplc = old_data['hplc'].copy()

        # Trobar columna de data i sample
        date_col = None
        sample_col_name = None
        for col in df_hplc.columns:
            if 'acquired date' in col.lower() or ('injection' in col.lower() and 'date' in col.lower()):
                date_col = col
            elif 'sample' in col.lower() and 'name' in col.lower():
                sample_col_name = col

        # IMPORTANT: Filtrar només files amb Sample Name (injeccions reals)
        if sample_col_name:
            df_hplc = df_hplc[df_hplc[sample_col_name].notna()].copy()

        if date_col:
            # Convertir a datetime i ordenar
            df_hplc[date_col] = pd.to_datetime(df_hplc[date_col])
            df_hplc = df_hplc.sort_values(date_col).reset_index(drop=True)

            # Afegir Inj_Index (índex seqüencial 1, 2, 3...)
            df_hplc['Inj_Index'] = range(1, len(df_hplc) + 1)

        # Escriure dades
        for r_idx, row in enumerate(dataframe_to_rows(df_hplc, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                # Evitar escriure NaN/None - deixar cel·la buida
                if pd.isna(value):
                    continue
                # Escapar strings que comencin amb = (evitar interpretar com fórmula)
                if isinstance(value, str) and value.startswith('='):
                    value = "'" + value
                ws_hplc.cell(row=r_idx, column=c_idx, value=value)

    # --- 2-TOC ---
    ws_toc = wb.create_sheet('2-TOC')
    toc_data_start_row = 8  # Les dades comencen a la fila 8 (després de capçaleres)
    toc_num_rows = 0

    if old_data.get('toc') is not None:
        df_toc = old_data['toc']
        toc_num_rows = len(df_toc)
        for r_idx, row in enumerate(dataframe_to_rows(df_toc, index=False, header=False), 1):
            for c_idx, value in enumerate(row, 1):
                # Evitar escriure NaN/None - deixar cel·la buida
                if pd.isna(value):
                    continue
                # Escapar strings que comencin amb = (evitar interpretar com fórmula)
                if isinstance(value, str) and value.startswith('='):
                    value = "'" + value
                ws_toc.cell(row=r_idx, column=c_idx, value=value)

    # --- 3-TOC_CALC ---
    # Fórmules que calculen hora_HPLC i Inj_Index per cada fila de TOC
    ws_toc_calc = wb.create_sheet('3-TOC_CALC')

    # Capçaleres
    ws_toc_calc['A1'] = 'TOC_Row'
    ws_toc_calc['B1'] = 'Date_Started'
    ws_toc_calc['C1'] = 'hora_HPLC'
    ws_toc_calc['D1'] = 'Inj_Index'
    ws_toc_calc['E1'] = 'Sample'

    # Trobar columna de dates a HPLC-SEQ (per COUNTIF)
    hplc_date_col = 'F'  # Per defecte, actualitzar si cal
    hplc_sample_col = 'D'  # Per defecte
    hplc_index_col = None

    if old_data.get('hplc') is not None:
        df_hplc = old_data['hplc']
        for i, col in enumerate(df_hplc.columns):
            col_letter = get_column_letter(i + 1)
            if 'acquired date' in col.lower():
                hplc_date_col = col_letter
            elif 'sample' in col.lower() and 'name' in col.lower():
                hplc_sample_col = col_letter
        # Inj_Index és l'última columna (afegida per nosaltres)
        hplc_index_col = get_column_letter(len(df_hplc.columns) + 1)

    # Afegir fórmules per cada fila de TOC (començant a fila 8 de TOC)
    if toc_num_rows > 0:
        for i in range(toc_data_start_row, toc_num_rows + 1):
            row = i - toc_data_start_row + 2  # Fila a TOC_CALC (comença a 2)

            # TOC_Row - referència a la fila
            ws_toc_calc.cell(row=row, column=1, value=f'={i}')

            # Date_Started - columna D de TOC
            ws_toc_calc.cell(row=row, column=2, value=f"='2-TOC'!D{i}")

            # hora_HPLC = Date_Started + net_delay (en minuts, convertit a fracció de dia)
            ws_toc_calc.cell(row=row, column=3, value=f"=B{row}+'0-INFO'!$B$12/(24*60)")

            # Inj_Index = COUNTIF de quantes injeccions han començat abans
            ws_toc_calc.cell(row=row, column=4,
                           value=f"=COUNTIF('1-HPLC-SEQ'!${hplc_date_col}:${hplc_date_col},\"<=\"&C{row})")

            # Sample = INDEX per obtenir el nom de la mostra
            ws_toc_calc.cell(row=row, column=5,
                           value=f"=IFERROR(INDEX('1-HPLC-SEQ'!${hplc_sample_col}:${hplc_sample_col},MATCH(D{row},'1-HPLC-SEQ'!${hplc_index_col}:${hplc_index_col},0)),\"\")")

        # Aplicar format de data/hora a columnes B i C
        from openpyxl.styles import numbers
        date_format = 'DD/MM/YYYY HH:MM:SS'
        for row in range(2, toc_num_rows - toc_data_start_row + 3):
            ws_toc_calc.cell(row=row, column=2).number_format = date_format
            ws_toc_calc.cell(row=row, column=3).number_format = date_format

    # --- 4-DAD_KHP (opcional) ---
    khp_data = extract_khp_from_dad(old_data.get('dad'))
    if khp_data:
        ws_khp = wb.create_sheet('4-DAD_KHP')

        col_offset = 1
        for khp_name, df_khp in khp_data.items():
            ws_khp.cell(row=1, column=col_offset, value=khp_name)
            ws_khp.cell(row=2, column=col_offset, value='time (min)')
            ws_khp.cell(row=2, column=col_offset + 1, value='value (mAU)')

            for r_idx, row in df_khp.iterrows():
                ws_khp.cell(row=r_idx + 3, column=col_offset, value=row['time'])
                ws_khp.cell(row=r_idx + 3, column=col_offset + 1, value=row['value'])

            col_offset += 3  # Espai entre KHPs

    # Guardar
    wb.save(output_path)
    print(f"Nou MasterFile creat: {output_path}")
    return True


def migrate_master(old_filepath, inj_volume=None):
    """Migra un fitxer mestre antic al nou format."""

    print(f"\n{'='*60}")
    print(f"Migrant: {old_filepath}")
    print('='*60)

    # 1. Backup
    backup_path = backup_original(old_filepath)
    if not backup_path:
        print("ERROR: No s'ha pogut crear backup")
        return False

    # 2. Llegir dades antigues
    old_data = read_old_master(old_filepath)
    if old_data is None:
        print("ERROR: No s'han pogut llegir les dades")
        return False

    # 3. Extreure info
    info = extract_info_from_check(old_data.get('check'))

    # Override method si el folder conté "_BP"
    dir_path = os.path.dirname(old_filepath)
    folder_name = os.path.basename(dir_path).upper()
    if '_BP' in folder_name or folder_name.endswith('BP'):
        info['method'] = 'BP'

    # Detectar UIB pel nom del fitxer
    filename = os.path.basename(old_filepath).upper()
    has_uib = 'UIB' in filename
    info['uib_range'] = determine_uib_range(info.get('seq_id'), has_uib)

    print(f"  SEQ-ID: {info.get('seq_id')}")
    print(f"  Method: {info.get('method')}")
    print(f"  UIB_range: {info.get('uib_range')}")
    print(f"  Hora HPLC: {info.get('hora_hplc')}")
    print(f"  Hora TOC: {info.get('hora_toc')}")

    # 4. Generar nou path
    seq_id = info.get('seq_id', 'UNKNOWN')
    new_filename = f"{seq_id}_MasterFile.xlsx"
    new_filepath = os.path.join(dir_path, new_filename)

    # 5. Crear nou MasterFile
    success = create_new_master(old_data, info, new_filepath, inj_volume)

    if success:
        print(f"\nMigració completada!")
        print(f"  Original (backup): {backup_path}")
        print(f"  Nou MasterFile: {new_filepath}")

    return success


def migrate_all_in_folder(base_folder):
    """Migra tots els fitxers mestre d'una carpeta."""

    # Buscar fitxers RAWDATA
    # Pattern 1: Si estem a Dades2, buscar en subcarpetes SEQ
    # Pattern 2: Si estem dins una carpeta SEQ, buscar directament
    patterns = [
        f"{base_folder}/*SEQ*/*RAWDATA*.xlsx",  # Subcarpetes SEQ
        f"{base_folder}/*SEQ*/*rawdata*.xlsx",
        f"{base_folder}/*SEQ*/*_v11*.xlsx",     # Format antic (sense RAWDATA al nom)
        f"{base_folder}/*SEQ*/*_v12*.xlsx",
        f"{base_folder}/*RAWDATA*.xlsx",         # Directament al folder
        f"{base_folder}/*rawdata*.xlsx",
        f"{base_folder}/*_v11*.xlsx",
        f"{base_folder}/*_v12*.xlsx",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    # Eliminar duplicats mantenint ordre
    files = list(dict.fromkeys(files))

    # Filtrar els que ja són MasterFile o backups
    files = [f for f in files if 'MasterFile' not in f and 'backup' not in f]

    print(f"\nTrobats {len(files)} fitxers per migrar")

    results = {'success': 0, 'failed': 0, 'skipped': 0}

    for f in sorted(files):
        try:
            # Verificar si ja existeix MasterFile
            dir_path = os.path.dirname(f)
            existing_master = glob.glob(os.path.join(dir_path, '*_MasterFile.xlsx'))

            if existing_master:
                print(f"\nSKIP: {f} (ja existeix MasterFile)")
                results['skipped'] += 1
                continue

            success = migrate_master(f)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1

        except Exception as e:
            print(f"\nERROR migrant {f}: {e}")
            results['failed'] += 1

    print(f"\n{'='*60}")
    print("RESUM MIGRACIÓ")
    print('='*60)
    print(f"  Correctes: {results['success']}")
    print(f"  Fallats: {results['failed']}")
    print(f"  Omesos (ja migrats): {results['skipped']}")

    return results


# --- CLI ---
if __name__ == '__main__':
    import sys

    print("="*60)
    print("HPSEC MasterFile Migration Tool")
    print("="*60)

    if len(sys.argv) < 2:
        print(f"""
Ús:
  python hpsec_migrate_master.py <fitxer.xlsx>              Migra amb plantilla
  python hpsec_migrate_master.py <fitxer.xlsx> --legacy     Migra sense plantilla
  python hpsec_migrate_master.py --all                      Migra tot Dades2

Plantilla: {TEMPLATE_PATH}
        """)
        sys.exit(1)

    arg = sys.argv[1]
    use_template = '--legacy' not in sys.argv

    if arg == '--all':
        base = 'C:/Users/Lequia/Desktop/Dades2'
        migrate_all_in_folder(base)
    elif os.path.isfile(arg):
        if use_template:
            # Nou mètode: crear des de zero
            create_masterfile_from_scratch(arg)
        else:
            # Mètode antic
            migrate_master(arg)
    elif os.path.isdir(arg):
        migrate_all_in_folder(arg)
    else:
        print(f"ERROR: No trobat: {arg}")
        sys.exit(1)
