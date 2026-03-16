import pandas as pd

mf_path = "C:/Users/Lequia/Desktop/Dades3/295_SEQ_BP/295_SEQ_MasterFile .xlsx"

# Read 2-TOC with header=6 (same as the code)
toc_df = pd.read_excel(mf_path, sheet_name="2-TOC", header=6, engine="openpyxl")

print("=== 2-TOC DataFrame ===")
print(f"Shape: {toc_df.shape}")
print(f"Columns: {list(toc_df.columns[:5])}")
print(f"Index range: {toc_df.index[0]} to {toc_df.index[-1]}")
print(f"\nFirst 3 rows (iloc[0:3]):")
print(toc_df.iloc[0:3].to_string())
print(f"\nRows around idx 1007 (which would be Excel row 1015):")
print(toc_df.iloc[1005:1012].to_string())

# Check time column - find dt > 60s
time_col = None
for c in toc_df.columns:
    if 'date' in str(c).lower() or 'time' in str(c).lower() or 'dat' in str(c).lower():
        print(f"\nPotential time column: '{c}' - first value: {toc_df[c].iloc[0]}")

# Look for the timeout around row 1012-1015 (Excel rows 1020-1023)
# Actually row_start for LQ0462_R1 = 1015 (from manifest), so DataFrame idx = 1015-8 = 1007
print(f"\nRows 1003-1010 (Excel rows 1011-1018):")
for i in range(1003, 1010):
    row = toc_df.iloc[i]
    print(f"  df_idx={i} (Excel row {i+8}): {row.values[:4]}")

# Read 4-TOC_CALC to see TOC_Row values for injections 6-8
toc_calc = pd.read_excel(mf_path, sheet_name="4-TOC_CALC", engine="openpyxl")
print(f"\n=== 4-TOC_CALC ===")
print(f"Columns: {list(toc_calc.columns[:10])}")

# Find rows for injections around LQ0462
for col in toc_calc.columns:
    if 'sample' in str(col).lower() or 'name' in str(col).lower() or 'inj' in str(col).lower():
        print(f"\nColumn '{col}' unique values (first 20):")
        print(toc_calc[col].unique()[:20])

# Show all injection rows with TOC_Row info
toc_row_col = None
for c in toc_calc.columns:
    if 'toc_row' in str(c).lower() or 'TOC_Row' in str(c):
        toc_row_col = c
        break

if toc_row_col:
    print(f"\nTOC_Row column found: '{toc_row_col}'")
    # Group by injection to find row ranges
    inj_col = None
    for c in toc_calc.columns:
        if 'seqrow' in str(c).lower() or 'seq' in str(c).lower():
            inj_col = c
            break
    if inj_col:
        print(f"Injection column: '{inj_col}'")
        for inj_num in range(5, 10):
            mask = toc_calc[inj_col] == inj_num
            if mask.any():
                subset = toc_calc[mask]
                toc_rows = subset[toc_row_col]
                name_col = None
                for c in toc_calc.columns:
                    if 'sample' in str(c).lower() or 'name' in str(c).lower():
                        name_col = c
                        break
                name = subset[name_col].iloc[0] if name_col else "?"
                print(f"  Inj {inj_num} ({name}): TOC_Row {int(toc_rows.min())}-{int(toc_rows.max())}, n={len(subset)}")
