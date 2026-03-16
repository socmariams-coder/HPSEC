import json, glob, os

for pattern in ['C:/Users/Lequia/Desktop/Dades3/292*/CHECK/data/calibration_result.json',
                'C:/Users/Lequia/Desktop/Dades3/293*/CHECK/data/calibration_result.json']:
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)
        seq = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        print(f"=== {seq} ===")
        # Top-level warnings
        warnings = data.get('warnings', [])
        if warnings:
            print(f"  top warnings: {len(warnings)}")
            for w in warnings[:5]:
                if isinstance(w, dict):
                    print(f"    {w.get('code','')} {w.get('message','')[:80]}")
                else:
                    print(f"    {str(w)[:80]}")
        # khp_data
        for key in ['khp_data_direct', 'khp_data_uib']:
            khp = data.get(key)
            if not khp:
                continue
            if isinstance(khp, dict):
                reps = khp.get('replicas', [khp])
            else:
                reps = [khp]
            for i, rep in enumerate(reps):
                if not isinstance(rep, dict):
                    continue
                anoms = rep.get('calibration_anomalies', [])
                qi = rep.get('quality_issues', [])
                if anoms:
                    codes = [a.get('code','?') for a in anoms if isinstance(a, dict)]
                    print(f"  {key} rep{i} anomalies: {codes}")
                if qi:
                    print(f"  {key} rep{i} quality_issues: {qi[:3]}")
        # calibrations
        for cal_key in ['calibrations_direct', 'calibrations_uib']:
            cals = data.get(cal_key, [])
            for cal in cals:
                anoms = cal.get('calibration_anomalies', [])
                qi = cal.get('quality_issues', [])
                if anoms:
                    codes = [a.get('code','?') for a in anoms if isinstance(a, dict)]
                    print(f"  {cal_key} {cal.get('name_full','?')} anomalies: {codes}")
                if qi:
                    print(f"  {cal_key} {cal.get('name_full','?')} quality_issues: {qi[:3]}")
