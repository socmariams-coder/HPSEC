"""
Audit MasterFiles — informe detallat per cada SEQ.
====================================================
Per cada MasterFile reporta:
- Delay: valor, font, si falta
- Volum: 0-INFO vs col13, discrepancies
- Estructura: fulles, columnes
- Acció necessària

Output: CSV amb totes les SEQs + informe text

Usage:
    python _audit_masterfiles.py
"""
import os, csv
import numpy as np
import pandas as pd
from pathlib import Path
from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


def audit_masterfile(seq_path, mf_path):
    """Audit one MasterFile. Returns dict with findings."""
    seq_name = os.path.basename(seq_path)
    method = detect_seq_type(seq_name)
    result = {
        'seq': seq_name,
        'method': method,
        'masterfile': os.path.basename(mf_path),
    }

    try:
        xls = pd.ExcelFile(mf_path)
        result['sheets'] = sorted(xls.sheet_names)
        result['has_dad_khp'] = '3-DAD_KHP' in xls.sheet_names
        result['has_4toc'] = '4-TOC_CALC' in xls.sheet_names
    except Exception as e:
        result['error'] = f"Cannot read: {e}"
        return result

    # === 0-INFO ===
    try:
        df_info = pd.read_excel(mf_path, sheet_name='0-INFO', header=None)
        info = {}
        for _, row in df_info.iterrows():
            if pd.notna(row.iloc[0]):
                key = str(row.iloc[0]).strip()
                val = row.iloc[1] if len(row) > 1 else None
                info[key] = val

        # Delay
        delay_val = None
        for k, v in info.items():
            if 'net delay' in k.lower() and 'suite' not in k.lower():
                try:
                    d = float(v)
                    if not np.isnan(d):
                        delay_val = d
                except (ValueError, TypeError):
                    pass

        result['delay'] = delay_val
        result['delay_status'] = 'OK' if delay_val is not None else 'FALTA'

        # Volume 0-INFO
        vol_info = None
        for k, v in info.items():
            if 'volume' in k.lower():
                try:
                    vol_info = float(v)
                except (ValueError, TypeError):
                    pass
        result['vol_0info'] = vol_info

        # Hores HPLC/TOC
        hora_hplc = info.get('Hora HPLC')
        hora_toc = info.get('Hora TOC')
        result['hora_hplc'] = str(hora_hplc) if hora_hplc else None
        result['hora_toc'] = str(hora_toc) if hora_toc else None

        # Date
        result['date'] = str(info.get('Date', ''))

        # UIB range
        uib = info.get('UIB_range')
        result['uib_range'] = float(uib) if uib and str(uib).strip() else None

    except Exception as e:
        result['error_info'] = str(e)

    # === 1-HPLC-SEQ ===
    try:
        df_hplc = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
        result['n_hplc_cols'] = df_hplc.shape[1]
        result['n_hplc_rows'] = df_hplc.shape[0]
        result['hplc_cols'] = list(df_hplc.columns)

        # Col 13 volume
        col13_vol = None
        col13_name = None
        if df_hplc.shape[1] > 13:
            col13_name = str(df_hplc.columns[13])
            vals = pd.to_numeric(df_hplc.iloc[:, 13], errors='coerce').dropna()
            if len(vals) > 0:
                unique_vals = sorted(vals.unique())
                if all(50 <= v <= 1000 for v in unique_vals):
                    col13_vol = unique_vals
                    result['col13_is_volume'] = True
                else:
                    result['col13_is_volume'] = False
            else:
                result['col13_is_volume'] = False

        result['col13_name'] = col13_name
        result['col13_vol'] = col13_vol

        # Check volume discrepancy
        if vol_info and col13_vol and len(col13_vol) == 1:
            if int(vol_info) != int(col13_vol[0]):
                result['vol_discrepancy'] = f"0-INFO={int(vol_info)} vs col13={int(col13_vol[0])}"
            else:
                result['vol_discrepancy'] = None
        elif col13_vol and len(col13_vol) > 1:
            result['vol_discrepancy'] = f"col13 multiple: {col13_vol}"
        else:
            result['vol_discrepancy'] = None

        # Count sample types
        name_col = None
        for c in df_hplc.columns:
            if 'sample' in str(c).lower() and 'name' in str(c).lower():
                name_col = c
                break
        if name_col:
            names = df_hplc[name_col].dropna().tolist()
            result['n_injections'] = len(names)
            n_khp = sum(1 for n in names if 'khp' in str(n).lower())
            n_mq = sum(1 for n in names if str(n).lower() in ['mq', 'blanc', 'blnc'])
            n_naoh = sum(1 for n in names if 'naoh' in str(n).lower())
            result['n_khp'] = n_khp
            result['n_mq'] = n_mq
            result['n_naoh'] = n_naoh
            result['n_samples'] = len(names) - n_khp - n_mq - n_naoh

    except Exception as e:
        result['error_hplc'] = str(e)

    # === 2-TOC ===
    try:
        df_toc = pd.read_excel(mf_path, sheet_name='2-TOC', header=6)
        result['n_toc_rows'] = len(df_toc)
        result['toc_ok'] = len(df_toc) > 100
    except Exception as e:
        result['toc_ok'] = False
        result['error_toc'] = str(e)

    # === 4-TOC_CALC ===
    if result.get('has_4toc'):
        try:
            df_calc = pd.read_excel(mf_path, sheet_name='4-TOC_CALC')
            n_assigned = len(df_calc[df_calc['Inj_Index'] > 0]['Inj_Index'].unique()) if 'Inj_Index' in df_calc.columns else 0
            result['n_toc_assigned'] = n_assigned
        except Exception:
            result['n_toc_assigned'] = 0

    # === DETERMINE ACTIONS ===
    actions = []

    if result.get('delay') is None:
        actions.append("CALCULAR DELAY")

    if result.get('vol_discrepancy'):
        actions.append(f"VERIFICAR VOLUM: {result['vol_discrepancy']}")

    if result.get('n_hplc_cols', 0) <= 13 and result.get('vol_0info'):
        actions.append("AFEGIR COL VOLUM a 1-HPLC-SEQ")

    if not result.get('toc_ok'):
        actions.append("REVISAR 2-TOC (poques dades)")

    if not result.get('has_4toc'):
        actions.append("FALTA 4-TOC_CALC")
    elif result.get('n_toc_assigned', 0) == 0:
        actions.append("4-TOC_CALC BUIT")

    result['actions'] = actions
    result['needs_action'] = len(actions) > 0

    return result


def main():
    folders = get_data_folders()

    all_seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            mfs = [f for f in os.listdir(full) if 'MasterFile' in f and f.endswith('.xlsx')
                   and '~' not in f and 'pre_vol' not in f and 'backup' not in f]
            if not mfs:
                mfs = [f for f in os.listdir(full) if 'MasterFile' in f and f.endswith('.xlsx')
                       and '~' not in f and 'pre_vol' not in f]
            if mfs:
                all_seqs.append((full, os.path.join(full, mfs[0])))

    print(f"Auditing {len(all_seqs)} MasterFiles...")

    results = []
    for seq_path, mf_path in all_seqs:
        r = audit_masterfile(seq_path, mf_path)
        results.append(r)

    # === REPORT ===
    out = Path('_results/audit_masterfiles')
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_fields = ['seq', 'method', 'date', 'n_injections', 'n_samples', 'n_khp',
                  'delay', 'delay_status', 'vol_0info', 'col13_vol', 'vol_discrepancy',
                  'n_hplc_cols', 'n_toc_rows', 'toc_ok', 'has_4toc', 'n_toc_assigned',
                  'uib_range', 'needs_action', 'actions']
    csv_path = out / 'masterfile_audit.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, delimiter=';', extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = dict(r)
            row['actions'] = ' | '.join(r.get('actions', []))
            row['col13_vol'] = str(r.get('col13_vol', ''))
            w.writerow(row)
    print(f"CSV: {csv_path}")

    # Text report
    report_path = out / 'masterfile_audit.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        # Summary
        n_total = len(results)
        n_need_action = sum(1 for r in results if r.get('needs_action'))
        n_delay_missing = sum(1 for r in results if r.get('delay') is None)
        n_vol_disc = sum(1 for r in results if r.get('vol_discrepancy'))

        f.write(f"AUDIT MASTERFILES\n{'='*60}\n\n")
        f.write(f"Total: {n_total}\n")
        f.write(f"Necessiten accio: {n_need_action}\n")
        f.write(f"Delay falta: {n_delay_missing}\n")
        f.write(f"Volum discrepant: {n_vol_disc}\n\n")

        # Group by action type
        f.write(f"{'='*60}\nDELAY FALTA ({n_delay_missing} SEQs)\n{'='*60}\n")
        for r in results:
            if r.get('delay') is None:
                hplc = r.get('hora_hplc', '?')
                toc = r.get('hora_toc', '?')
                f.write(f"  {r['seq']:25s} ({r['method']:6s}) "
                        f"hplc={hplc}  toc={toc}  "
                        f"vol={r.get('vol_0info','?')}uL  "
                        f"inj={r.get('n_injections','?')}\n")

        f.write(f"\n{'='*60}\nVOLUM DISCREPANT ({n_vol_disc} SEQs)\n{'='*60}\n")
        for r in results:
            if r.get('vol_discrepancy'):
                f.write(f"  {r['seq']:25s} ({r['method']:6s}) "
                        f"{r['vol_discrepancy']}\n")

        # Other actions
        other = [r for r in results if r.get('needs_action')
                 and r.get('delay') is not None and not r.get('vol_discrepancy')]
        if other:
            f.write(f"\n{'='*60}\nALTRES ACCIONS ({len(other)} SEQs)\n{'='*60}\n")
            for r in other:
                f.write(f"  {r['seq']:25s} ({r['method']:6s}) "
                        f"{' | '.join(r['actions'])}\n")

    print(f"Report: {report_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"RESUM")
    print(f"{'='*60}")
    print(f"Total MasterFiles: {n_total}")
    print(f"Necessiten accio: {n_need_action}")
    print(f"  Delay falta: {n_delay_missing}")
    print(f"  Volum discrepant: {n_vol_disc}")
    print(f"  Altres: {n_need_action - n_delay_missing - n_vol_disc}")

    # Volume discrepancies detail
    if n_vol_disc > 0:
        print(f"\n=== VOLUM DISCREPANT (cal decisio teva) ===")
        for r in results:
            if r.get('vol_discrepancy'):
                print(f"  {r['seq']:25s} {r['vol_discrepancy']}")

    print(f"\nCSV complet: {csv_path}")
    print(f"Report text: {report_path}")
    print("Fet!")


if __name__ == "__main__":
    main()
