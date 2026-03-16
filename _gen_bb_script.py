import pathlib
L = []
def a(s): L.append(s)

a('# BB Zone Spectral Characterisation')
a('import sys')
a('sys.stdout.reconfigure(encoding="utf-8")')
a('import os, numpy as np, pandas as pd')
a('sys.path.insert(0, os.path.join("C:" + os.sep, "Users", "Lequia", "Desktop", "HPSEC"))')
a('from hpsec_import import llegir_dad_export3d')

p = pathlib.Path("C:/Users/Lequia/Desktop/HPSEC/_tmp_bb_analysis.py")
p.write_text(chr(10).join(L), encoding="utf-8")
print(f"Written {p.stat().st_size} bytes")