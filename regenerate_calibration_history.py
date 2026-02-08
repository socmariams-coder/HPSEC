#!/usr/bin/env python3
"""
Script per regenerar la validació de l'històric de calibracions.
Actualitza quality_issues amb els nous thresholds estrictes (>20% INVALID, >10% WARNING)
i exclou outliers del càlcul de la mitjana històrica.
"""

import os
import sys
import json

# Afegir path del projecte
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpsec_calibrate import (
    load_khp_history,
    save_khp_history,
    validate_khp_quality,
    get_history_path
)


def regenerate_validation(base_path: str):
    """
    Regenera la validació de totes les calibracions a l'històric.

    Args:
        base_path: Path base on cercar SEQs (ex: Dades3)
    """
    print(f"Cercant SEQs a: {base_path}")

    # Trobar una SEQ vàlida
    sample_seq = None
    for item in os.listdir(base_path):
        if "_SEQ" in item:
            sample_seq = os.path.join(base_path, item)
            break

    if not sample_seq:
        print(f"ERROR: No s'ha trobat cap SEQ a {base_path}")
        return

    history_path = get_history_path(sample_seq)
    print(f"Històric: {history_path}")

    if not os.path.exists(history_path):
        print(f"ERROR: No existeix l'històric")
        return

    # Carregar calibracions
    calibrations = load_khp_history(sample_seq)
    print(f"Calibracions carregades: {len(calibrations)}")

    updated = 0
    errors = []

    for cal in calibrations:
        try:
            # Crear khp_data des de la calibració existent
            khp_data = {
                'area': cal.get('area', 0),
                'conc_ppm': cal.get('conc_ppm', 0),
                'volume_uL': cal.get('volume_uL', 0),
                'doc_mode': cal.get('doc_mode', 'N/A'),
                'is_bp': cal.get('is_bp', False),
                'symmetry': cal.get('symmetry', 1.0),
                'snr': cal.get('snr', 0),
                'concentration_ratio': cal.get('concentration_ratio', 1.0),
                'has_batman': cal.get('has_batman', False),
                'limits_expanded': cal.get('limits_expanded', False),
                'uib_sensitivity': cal.get('uib_sensitivity'),
                't_retention': cal.get('t_retention', 0),
            }

            # Timeout i anomaly info
            timeout_info = {'severity': 'CRITICAL' if cal.get('has_timeout') else 'OK'}
            anomaly_info = {'is_batman': cal.get('has_batman', False), 'is_irregular': False}

            # Detectar irregular des d'issues existents
            import re
            existing_issues = cal.get('quality_issues', [])
            for issue in existing_issues:
                if 'irregular' in issue.lower() or 'smoothness' in issue.lower():
                    anomaly_info['is_irregular'] = True
                    match = re.search(r'smoothness[=:\s]*(\d+)', issue, re.I)
                    if match:
                        anomaly_info['smoothness'] = int(match.group(1))

            # Executar validació amb nous thresholds
            validation = validate_khp_quality(
                khp_data=khp_data,
                all_peaks=[],
                timeout_info=timeout_info,
                anomaly_info=anomaly_info,
                seq_path=cal.get('seq_path', base_path)
            )

            # Actualitzar camps
            cal['valid_for_calibration'] = validation.get('is_valid', True)
            cal['calibration_issues'] = validation.get('issues', [])
            cal['calibration_warnings'] = validation.get('warnings', [])
            cal['quality_score'] = validation.get('quality_score', 0)

            # Actualitzar quality_issues (issues + warnings + historical)
            all_issues = validation.get('issues', []) + validation.get('warnings', [])
            hist_comp = validation.get('historical_comparison', {})
            if hist_comp:
                all_issues.extend(hist_comp.get('issues', []))
                all_issues.extend(hist_comp.get('warnings', []))
            cal['quality_issues'] = all_issues

            # Reference comparison
            if validation.get('historical_comparison'):
                cal['reference_comparison'] = validation['historical_comparison']

            # Status
            if cal.get('manual_override') is True:
                cal['status'] = "MANUAL_VALID"
            elif cal.get('manual_override') is False:
                cal['status'] = "MANUAL_INVALID"
            elif not cal['valid_for_calibration']:
                cal['status'] = "INVALID_CAL"
                cal['is_outlier'] = True
                cal['is_active'] = False
            else:
                cal['status'] = "OK"
                if cal.get('manual_override') is None:
                    cal['is_outlier'] = False
                    cal['is_active'] = True

            seq_name = cal.get('seq_name', 'unknown')
            print(f"  {seq_name}: {cal['status']} - issues={len(all_issues)}")
            updated += 1

        except Exception as e:
            errors.append(f"{cal.get('seq_name', 'unknown')}: {e}")
            print(f"  ERROR: {e}")

    # Guardar
    save_khp_history(sample_seq, calibrations)
    print(f"\nResultat:")
    print(f"  Actualitzades: {updated}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    # Usar Dades3 per defecte (on és l'històric actual)
    base = r"C:\Users\Lequia\Desktop\Dades3"

    if len(sys.argv) > 1:
        base = sys.argv[1]

    regenerate_validation(base)
    print("\nFet!")
