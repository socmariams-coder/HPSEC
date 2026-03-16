# This script generates _tmp_bb_analysis.py
import pathlib
target = pathlib.Path("C:/Users/Lequia/Desktop/HPSEC/_tmp_bb_analysis.py")
target.write_bytes(open(__file__.replace("_gen_bb.py", "_bb_template.txt"), "rb").read())
print(f"Written {target.stat().st_size} bytes")
