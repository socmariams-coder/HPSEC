# -*- coding: utf-8 -*-
"""
Inventari de mostres injectades per seqüència (Dades3).
Llegeix tots els MasterFiles i genera un Excel resum.

Ús:
    python inventari_mostres.py
"""

import os, sys, re
from pathlib import Path
import pandas as pd
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

DATA_FOLDER = "C:/Users/Lequia/Desktop/Dades3"
OUTPUT = "C:/Users/Lequia/Desktop/HPSEC/inventari_mostres_dades3.xlsx"


def detect_method(seq_name):
    """COLUMN or BP from folder name."""
    if "_BP" in seq_name.upper():
        return "BP"
    return "COLUMN"


def extract_seq_id(seq_name):
    """Extract numeric ID: '256B_SEQ' -> '256B', '261_SEQ_BP' -> '261'."""
    m = re.match(r'^(\d+\w?)_SEQ', seq_name)
    return m.group(1) if m else seq_name


def read_masterfile(mf_path):
    """Read 1-HPLC-SEQ sheet, return list of sample dicts."""
    try:
        df = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
    except Exception:
        return []

    # Find relevant columns (names vary slightly)
    sample_col = None
    date_col = None
    rep_col = None
    inj_col = None

    for c in df.columns:
        cl = str(c).lower().strip()
        if 'sample' in cl and 'name' in cl:
            sample_col = c
        elif 'acquired' in cl or ('date' in cl and 'inject' in cl):
            date_col = c
        elif 'sample_rep' in cl:
            rep_col = c
        elif 'inj_index' in cl:
            inj_col = c

    if sample_col is None:
        return []

    rows = []
    for _, r in df.iterrows():
        name = r.get(sample_col)
        if pd.isna(name) or str(name).strip() == '':
            continue
        row = {
            "sample_name": str(name).strip(),
        }
        if rep_col and not pd.isna(r.get(rep_col)):
            row["sample_rep"] = str(r[rep_col]).strip()
        if inj_col and not pd.isna(r.get(inj_col)):
            row["inj_index"] = int(r[inj_col])
        if date_col and not pd.isna(r.get(date_col)):
            row["acq_date"] = r[date_col]
        rows.append(row)

    return rows


def read_info_sheet(mf_path):
    """Read 0-INFO for metadata."""
    try:
        df = pd.read_excel(mf_path, sheet_name='0-INFO', header=None)
        info = {}
        for _, r in df.iterrows():
            key = str(r.iloc[0]).strip() if not pd.isna(r.iloc[0]) else ""
            val = r.iloc[1] if len(r) > 1 else None
            if key:
                info[key] = val
        return info
    except Exception:
        return {}


# =============================================================================
# SCAN
# =============================================================================
print("=" * 65)
print("INVENTARI DE MOSTRES — Dades3")
print("=" * 65)

sequences = sorted(Path(DATA_FOLDER).iterdir())
sequences = [s for s in sequences if s.is_dir() and "_SEQ" in s.name.upper()]

all_injections = []     # Full injection list
seq_summary = []        # One row per sequence

for seq_dir in sequences:
    seq_name = seq_dir.name
    seq_id = extract_seq_id(seq_name)
    method = detect_method(seq_name)

    # Find MasterFile
    mf_files = [f for f in seq_dir.glob("*MasterFile*.xlsx") if "backup" not in f.name.lower()]
    if not mf_files:
        seq_summary.append({
            "SEQ_ID": seq_id,
            "SEQ_folder": seq_name,
            "Method": method,
            "n_injections": 0,
            "n_samples": 0,
            "samples": "",
            "date_start": None,
            "date_end": None,
            "status": "NO MASTERFILE",
        })
        continue

    mf_path = str(mf_files[0])
    samples = read_masterfile(mf_path)
    info = read_info_sheet(mf_path)

    if not samples:
        seq_summary.append({
            "SEQ_ID": seq_id,
            "SEQ_folder": seq_name,
            "Method": method,
            "n_injections": 0,
            "n_samples": 0,
            "samples": "",
            "date_start": None,
            "date_end": None,
            "status": "EMPTY",
        })
        continue

    # Unique sample names (without rep suffix)
    unique_names = list(dict.fromkeys(s["sample_name"] for s in samples))

    # Date range
    dates = [s["acq_date"] for s in samples if "acq_date" in s]
    date_start = min(dates) if dates else None
    date_end = max(dates) if dates else None

    seq_summary.append({
        "SEQ_ID": seq_id,
        "SEQ_folder": seq_name,
        "Method": method,
        "n_injections": len(samples),
        "n_samples": len(unique_names),
        "samples": " | ".join(unique_names),
        "date_start": date_start,
        "date_end": date_end,
        "status": "OK",
    })

    # Add to full injection list
    for s in samples:
        all_injections.append({
            "SEQ_ID": seq_id,
            "SEQ_folder": seq_name,
            "Method": method,
            **s,
        })

    print(f"  {seq_name:25s}  {len(samples):3d} inj  {len(unique_names):2d} mostres  {method}")

# =============================================================================
# WRITE EXCEL
# =============================================================================
print(f"\n  Total: {len(sequences)} SEQs, {len(all_injections)} injeccions")

df_summary = pd.DataFrame(seq_summary)
df_injections = pd.DataFrame(all_injections)

with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='SEQS', index=False)
    df_injections.to_excel(writer, sheet_name='INJECTIONS', index=False)

    # Pivot: samples x sequences (presence matrix)
    if len(df_injections) > 0:
        # Unique sample per seq
        pivot_data = df_injections.drop_duplicates(subset=["SEQ_ID", "sample_name"])
        pivot = pivot_data.pivot_table(
            index="sample_name",
            columns="SEQ_ID",
            values="inj_index",
            aggfunc="count",
            fill_value=0
        )
        pivot.to_excel(writer, sheet_name='MATRIX')

print(f"\n  Guardat: {OUTPUT}")
print(f"    - SEQS: resum per seqüència ({len(df_summary)} files)")
print(f"    - INJECTIONS: totes les injeccions ({len(df_injections)} files)")
print(f"    - MATRIX: mostres × seqüències (presència)")
print("=" * 65)
