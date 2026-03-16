import pandas as pd

mf_path = "C:/Users/Lequia/Desktop/Dades3/295_SEQ_BP/295_SEQ_MasterFile .xlsx"

# Read 4-TOC_CALC
toc_calc = pd.read_excel(mf_path, sheet_name="4-TOC_CALC", engine="openpyxl")

print("=== 4-TOC_CALC full columns ===")
print(f"Columns: {list(toc_calc.columns)}")
print(f"Shape: {toc_calc.shape}")

# Show injection groupings using Inj_Index
print("\nInjection groupings (Inj_Index 4-9):")
for inj_num in range(4, 10):
    mask = toc_calc['Inj_Index'] == inj_num
    if mask.any():
        subset = toc_calc[mask]
        toc_rows = subset['TOC_Row']
        name = subset['Sample'].iloc[0]
        print(f"  Inj {inj_num} ({name}): TOC_Row {int(toc_rows.min())}-{int(toc_rows.max())}, n={len(subset)}")

# Also show the timeout gap - look for large time jumps in 2-TOC
toc_df = pd.read_excel(mf_path, sheet_name="2-TOC", header=6, engine="openpyxl")

# Parse dates and find gaps > 60s
dates = pd.to_datetime(toc_df['Date Started'])
dt_seconds = dates.diff().dt.total_seconds()

print("\n=== Time gaps > 60s in 2-TOC ===")
large_gaps = dt_seconds[dt_seconds > 60]
for idx, gap in large_gaps.items():
    print(f"  df_idx={idx} (Excel row {idx+8}): gap={gap:.1f}s ({gap/60:.2f} min), Rep={toc_df.iloc[idx]['Rep Number']}, time={dates.iloc[idx]}")

# Show 0-INFO sheet
print("\n=== 0-INFO sheet ===")
info_df = pd.read_excel(mf_path, sheet_name="0-INFO", engine="openpyxl")
print(info_df.to_string())
