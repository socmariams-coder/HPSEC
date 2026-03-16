import json, statistics
from collections import Counter, defaultdict

FILE = "C:/Users/Lequia/Desktop/Dades3/REGISTRY/KHP_History.json"
with open(FILE, "r", encoding="utf-8") as fh:
    data = json.load(fh)
cals = data["calibrations"]
sep = "=" * 90
dash = "-" * 90
print(sep)
print("  KHP_History.json  FULL AUDIT")
print("  File:", FILE)
print("  Version:", data.get("version","?"), " |  Updated:", data.get("updated","?"))
print(sep)
ESSENTIAL = ["area", "conc_ppm", "volume_uL", "seq_name", "mode", "cal_id", "date", "rf_mass", "rf", "snr", "is_bp"]

print()
print(dash)
print("1. ENTRIES WITH conc_ppm = 0 OR MISSING")
print(dash)
count = 0
for e in cals:
    cppm = e.get("conc_ppm", None)
    if cppm is None or cppm == 0:
        count += 1
        print("  cal_id=%-30s  seq=%-25s  conc_ppm=%-6s  date=%-10s  mode=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), cppm, str(e.get("date","?"))[:10], e.get("mode","?")))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("2. ENTRIES WITH area < 0 (negative area)")
print(dash)
count = 0
for e in cals:
    area = e.get("area", None)
    if area is not None and area < 0:
        count += 1
        print("  cal_id=%-30s  seq=%-25s  area=%12.4f  conc=%s  date=%-10s  mode=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), area, e.get("conc_ppm","?"), str(e.get("date","?"))[:10], e.get("mode","?")))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("3. ENTRIES WITH area = 0 (exactly zero)")
print(dash)
count = 0
for e in cals:
    area = e.get("area", None)
    if area is not None and area == 0:
        count += 1
        print("  cal_id=%-30s  seq=%-25s  area=%s  conc=%s  date=%-10s  mode=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), area, e.get("conc_ppm","?"), str(e.get("date","?"))[:10], e.get("mode","?")))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("4. ENTRIES WITH rf_mass <= 0")
print(dash)
count = 0
for e in cals:
    rfm = e.get("rf_mass", None)
    if rfm is None or rfm <= 0:
        count += 1
        print("  cal_id=%-30s  seq=%-25s  rf_mass=%-15s  area=%-12s  conc=%s  date=%-10s  mode=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), str(rfm), str(e.get("area","?")), e.get("conc_ppm","?"), str(e.get("date","?"))[:10], e.get("mode","?")))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("5. ENTRIES MISSING ESSENTIAL FIELDS")
print(dash)
count = 0
for e in cals:
    missing = [f for f in ESSENTIAL if f not in e or e[f] is None]
    if missing:
        count += 1
        print("  cal_id=%-30s  seq=%-25s  MISSING: %s  date=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), missing, str(e.get("date","?"))[:10]))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("6. DUPLICATE seq_name ENTRIES")
print(dash)
seq_counter = Counter(e.get("seq_name","?") for e in cals)
dups = {k: v for k, v in seq_counter.items() if v > 1}
if dups:
    for seq_name, cnt in sorted(dups.items(), key=lambda x: -x[1]):
        print()
        print("  seq_name=%s -- appears %d times:" % (seq_name, cnt))
        entries = [e for e in cals if e.get("seq_name") == seq_name]
        for e in entries:
            a2 = e.get("area", 0)
            area_str = "%.4f" % a2 if isinstance(a2, (int,float)) else str(a2)
            print("    cal_id=%-30s  date=%-19s  mode=%-8s  conc=%s  area=%12s  is_bp=%s  outlier=%s  valid_cal=%s" % (
                e.get("cal_id","?"), str(e.get("date","?"))[:19], str(e.get("mode","?")), e.get("conc_ppm","?"), area_str,
                e.get("is_bp","?"), e.get("is_outlier","?"), e.get("valid_for_calibration","?")))
    print()
    print("  TOTAL duplicate seq_names: %d (covering %d entries)" % (len(dups), sum(dups.values())))
else:
    print("  (none found)")

print()
print(dash)
print("7. VOLUME_uL INCONSISTENCIES")
print(dash)
mode_vol = defaultdict(list)
for e in cals:
    mode = e.get("mode", "?")
    vol = e.get("volume_uL", None)
    conc = e.get("conc_ppm", 0)
    mode_vol[mode].append((vol, conc, e))
for mode, entries in sorted(mode_vol.items()):
    vol_counter = Counter((v, c) for v, c, _ in entries)
    print()
    print("  Mode=%s:" % mode)
    for (vol, conc), cnt in sorted(vol_counter.items(), key=lambda x: str(x[0])):
        print("    volume=%s uL  conc=%s ppm  -- %d entries" % (vol, conc, cnt))
print()
print("  Flagged entries:")
flag_count = 0
for e in cals:
    mode = e.get("mode", "?")
    vol = e.get("volume_uL", None)
    is_bp = e.get("is_bp", False)
    suspect = False
    reason = ""
    if is_bp and vol != 100:
        suspect = True
        reason = "BP should be 100 uL, got %s" % vol
    if mode == "COLUMN" and vol not in (100, 400, None):
        suspect = True
        reason = "COLUMN unusual volume %s" % vol
    if vol is None or vol <= 0:
        suspect = True
        reason = "Invalid volume: %s" % vol
    if suspect:
        flag_count += 1
        print("    cal_id=%-30s  seq=%-25s  volume=%s  mode=%s  reason=%s  date=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), vol, mode, reason, str(e.get("date","?"))[:10]))
if flag_count == 0:
    print("    (no suspicious volumes)")
print("  TOTAL flagged:", flag_count)

print()
print(dash)
print("8. METHOD FIELD CHECK")
print(dash)
has_method = 0
no_method = 0
method_vals = Counter()
for e in cals:
    m = e.get("method", None)
    if m is None or m == "":
        no_method += 1
    else:
        has_method += 1
        method_vals[m] += 1
if method_vals:
    print("  Entries WITH method field:", has_method)
    for m, cnt in method_vals.most_common():
        print("    method=%-30s -- %d entries" % (repr(m), cnt))
else:
    print("  NO entries have a method field at all.")
if no_method > 0:
    print()
    print("  Entries WITHOUT method field:", no_method)
    shown = 0
    for e in cals:
        if e.get("method", None) is None or e.get("method", None) == "":
            if shown < 5:
                print("    cal_id=%-30s  seq=%-25s  date=%s" % (
                    e.get("cal_id","?"), e.get("seq_name","?"), str(e.get("date","?"))[:10]))
                shown += 1
    if no_method > 5:
        print("    ... and %d more" % (no_method - 5))

print()
print(dash)
print("EXTRA: ANOMALOUS RF VALUES (rf < 0)")
print(dash)
rfm_by_mode = defaultdict(list)
for e in cals:
    rfm = e.get("rf_mass", 0) or 0
    conc = e.get("conc_ppm", 0) or 0
    if rfm > 0 and conc > 0:
        rfm_by_mode[e.get("mode","?")].append(rfm)
for mode, vals in sorted(rfm_by_mode.items()):
    if vals:
        mn = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0
        print("  Mode=%s: n=%d, mean_rf_mass=%.2f, std=%.2f, min=%.2f, max=%.2f" % (mode, len(vals), mn, sd, min(vals), max(vals)))
count = 0
for e in cals:
    rf = e.get("rf", 0) or 0
    if rf < 0:
        count += 1
        rfm = e.get("rf_mass", 0) or 0
        ar = e.get("area", 0) or 0
        print("  NEGATIVE RF: cal_id=%-30s  seq=%-25s  rf=%.4f  rf_mass=%.4f  area=%.4f  date=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), rf, rfm, ar, str(e.get("date","?"))[:10]))
if count == 0: print("  (no negative rf values)")
print("  TOTAL negative rf:", count)

print()
print(dash)
print("EXTRA: QUALITY SCORE = 0 BUT valid_for_calibration = True")
print(dash)
count = 0
for e in cals:
    qs = e.get("quality_score", -1)
    vc = e.get("valid_for_calibration", False)
    if qs == 0 and vc:
        count += 1
        ar = e.get("area", 0) or 0
        print("  cal_id=%-30s  seq=%-25s  quality_score=%s  area=%.4f  date=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), qs, ar, str(e.get("date","?"))[:10]))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("EXTRA: STATUS FIELD DISTRIBUTION")
print(dash)
status_counter = Counter(e.get("status","?") for e in cals)
for s, cnt in status_counter.most_common():
    print("  status=%-20s -- %d entries" % (repr(s), cnt))

print()
print(sep)
print("SUMMARY")
print(sep)
total = len(cals)
conc_gt0 = sum(1 for e in cals if (e.get("conc_ppm") or 0) > 0)
conc_eq0 = sum(1 for e in cals if (e.get("conc_ppm") or 0) == 0)
rfm_gt0 = sum(1 for e in cals if (e.get("rf_mass") or 0) > 0)
rfm_le0 = sum(1 for e in cals if e.get("rf_mass") is None or (e.get("rf_mass", 0) <= 0))
outliers_c = sum(1 for e in cals if e.get("is_outlier", False))
valid_cal = sum(1 for e in cals if e.get("valid_for_calibration", False))
active_c = sum(1 for e in cals if e.get("is_active", False))
neg_area = sum(1 for e in cals if (e.get("area") or 0) < 0)
zero_area = sum(1 for e in cals if e.get("area") == 0)
has_batman_c = sum(1 for e in cals if e.get("has_batman", False))
has_timeout_c = sum(1 for e in cals if e.get("has_timeout", False))
mode_counter = Counter(e.get("mode","?") for e in cals)
bp_count = sum(1 for e in cals if e.get("is_bp", False))
conc_counter = Counter(e.get("conc_ppm", 0) for e in cals)
print("  Total entries:                     %d" % total)
print("  Entries with conc_ppm > 0 (KHP):   %d" % conc_gt0)
print("  Entries with conc_ppm = 0 (blanks): %d" % conc_eq0)
print("  Entries with valid rf_mass > 0:     %d" % rfm_gt0)
print("  Entries with rf_mass <= 0/missing:  %d" % rfm_le0)
print("  Entries marked is_outlier=True:     %d" % outliers_c)
print("  Entries valid_for_calibration=True: %d" % valid_cal)
print("  Entries is_active=True:             %d" % active_c)
print("  Entries with negative area:         %d" % neg_area)
print("  Entries with area = 0:              %d" % zero_area)
print("  Entries with has_batman=True:       %d" % has_batman_c)
print("  Entries with has_timeout=True:      %d" % has_timeout_c)
print()
print("  By mode:")
for m, cnt in mode_counter.most_common():
    print("    %-10s -- %d" % (m, cnt))
print("  is_bp=True:", bp_count)
print()
print("  By conc_ppm:")
for c, cnt in sorted(conc_counter.items()):
    print("    %s ppm -- %d entries" % (c, cnt))

print()
print(dash)
print("CROSS-CHECK: conc_ppm > 0 BUT area <= 0")
print(dash)
count = 0
for e in cals:
    conc = e.get("conc_ppm", 0) or 0
    area = e.get("area", 0) or 0
    if conc > 0 and area <= 0:
        count += 1
        rfm = e.get("rf_mass", 0) or 0
        print("  cal_id=%-30s  seq=%-25s  conc=%s  area=%.4f  rf_mass=%.4f  valid_cal=%s  outlier=%s  date=%s  mode=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), conc, area, rfm,
            e.get("valid_for_calibration","?"), e.get("is_outlier","?"),
            str(e.get("date","?"))[:10], e.get("mode","?")))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("CROSS-CHECK: valid_for_calibration=True AND is_outlier=True")
print(dash)
count = 0
for e in cals:
    if e.get("valid_for_calibration") and e.get("is_outlier"):
        count += 1
        print("  cal_id=%-30s  seq=%-25s  date=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), str(e.get("date","?"))[:10]))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(dash)
print("CROSS-CHECK: valid_for_calibration=True AND conc_ppm=0")
print(dash)
count = 0
for e in cals:
    if e.get("valid_for_calibration") and (e.get("conc_ppm",0) or 0) == 0:
        count += 1
        print("  cal_id=%-30s  seq=%-25s  area=%s  date=%s" % (
            e.get("cal_id","?"), e.get("seq_name","?"), e.get("area",0), str(e.get("date","?"))[:10]))
if count == 0: print("  (none found)")
print("  TOTAL:", count)

print()
print(sep)
print("AUDIT COMPLETE")
print(sep)