"""
DOCtor Validation - Validació dels criteris de detecció
Testa els llindars proposats contra les etiquetes manuals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from datetime import datetime


# ============================================================
# CRITERIS PROPOSATS (ajustar segons resultats)
# ============================================================
THRESH_AS_DIVERGENCE = 0.40    # |As66 - As33| > 0.40 = problema
THRESH_AS66 = 1.20              # As66 > 1.20 = problema
THRESH_PEARSON = 0.997          # Pearson <= 0.997 = problema
# ============================================================


def load_csv():
    """Carrega CSV seleccionat per l'usuari."""
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Selecciona el CSV a validar",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()

    if not csv_path:
        print("ERROR: No s'ha seleccionat cap fitxer")
        return None

    print(f"Carregant: {csv_path}")
    return pd.read_csv(csv_path)


def predict_peak_quality(row, mode="A"):
    """
    Prediu si un pic és problemàtic basant-se en els criteris.

    Modes:
      A: (Delta_As > 0.40) AND (As66 > 1.20)
      B: (As66 > 1.20) AND (Pearson <= 0.997)
      C: (Delta_As > 0.40) AND (Pearson <= 0.997)
      D: (Delta_As > 0.40) OR (As66 > 1.20) - qualsevol dels dos
      E: ((Delta_As > 0.40) OR (As66 > 1.20)) AND (Pearson <= 0.997) - combinada

    Returns: (is_problem, reasons)
    """
    reasons = []

    as_div = abs(row.get("As66", 0) - row.get("As33", 0))
    as66 = row.get("As66", 1.0)
    pearson = row.get("pearson", 1.0)

    crit_div = as_div > THRESH_AS_DIVERGENCE
    crit_as66 = as66 > THRESH_AS66
    crit_pearson = pearson <= THRESH_PEARSON

    if mode == "A":
        # Delta_As AND As66
        is_problem = crit_div and crit_as66
        if crit_div:
            reasons.append(f"As_div={as_div:.2f}>{THRESH_AS_DIVERGENCE}")
        if crit_as66:
            reasons.append(f"As66={as66:.2f}>{THRESH_AS66}")

    elif mode == "B":
        # As66 AND Pearson
        is_problem = crit_as66 and crit_pearson
        if crit_as66:
            reasons.append(f"As66={as66:.2f}>{THRESH_AS66}")
        if crit_pearson:
            reasons.append(f"Pearson={pearson:.4f}<={THRESH_PEARSON}")

    elif mode == "C":
        # Delta_As AND Pearson
        is_problem = crit_div and crit_pearson
        if crit_div:
            reasons.append(f"As_div={as_div:.2f}>{THRESH_AS_DIVERGENCE}")
        if crit_pearson:
            reasons.append(f"Pearson={pearson:.4f}<={THRESH_PEARSON}")

    elif mode == "D":
        # Delta_As OR As66 (qualsevol)
        is_problem = crit_div or crit_as66
        if crit_div:
            reasons.append(f"As_div={as_div:.2f}>{THRESH_AS_DIVERGENCE}")
        if crit_as66:
            reasons.append(f"As66={as66:.2f}>{THRESH_AS66}")

    elif mode == "E":
        # (Delta_As OR As66) AND Pearson - nova regla combinada
        shape_problem = crit_div or crit_as66
        is_problem = shape_problem and crit_pearson
        if is_problem:
            if crit_div:
                reasons.append(f"Δ={as_div:.2f}")
            if crit_as66:
                reasons.append(f"As66={as66:.2f}")
            reasons.append(f"ρ={pearson:.4f}")

    else:
        is_problem = False

    return is_problem, reasons


def compare_replicas(df):
    """
    Compara rèpliques i determina quina és millor.
    Returns: DataFrame amb info de comparació
    """
    results = []

    # Agrupar per mostra
    for (seq, sample), group in df.groupby(["seq", "sample"]):
        r1 = group[group["replica"] == "R1"]
        r2 = group[group["replica"] == "R2"]

        # Per cada pic (per peak_num)
        for peak_num in group["peak_num"].unique():
            p1 = r1[r1["peak_num"] == peak_num]
            p2 = r2[r2["peak_num"] == peak_num]

            if len(p1) == 0 and len(p2) == 0:
                continue

            result = {
                "seq": seq,
                "sample": sample,
                "peak_num": peak_num,
                "has_r1": len(p1) > 0,
                "has_r2": len(p2) > 0,
            }

            if len(p1) > 0:
                p1 = p1.iloc[0]
                result["r1_height"] = p1.get("height", 0)
                result["r1_label"] = p1.get("label", "")
                result["r1_noise"] = p1.get("baseline_noise", 0)
                pred, reasons = predict_peak_quality(p1)
                result["r1_predicted_problem"] = pred
                result["r1_reasons"] = "|".join(reasons)

            if len(p2) > 0:
                p2 = p2.iloc[0]
                result["r2_height"] = p2.get("height", 0)
                result["r2_label"] = p2.get("label", "")
                result["r2_noise"] = p2.get("baseline_noise", 0)
                pred, reasons = predict_peak_quality(p2)
                result["r2_predicted_problem"] = pred
                result["r2_reasons"] = "|".join(reasons)

            # Decisió: quina rèplica és millor?
            if result.get("has_r1") and result.get("has_r2"):
                # Ambdues existeixen
                r1_prob = result.get("r1_predicted_problem", False)
                r2_prob = result.get("r2_predicted_problem", False)

                if r1_prob and not r2_prob:
                    result["best_replica"] = "R2"
                    result["decision_reason"] = "R1 té problemes"
                elif r2_prob and not r1_prob:
                    result["best_replica"] = "R1"
                    result["decision_reason"] = "R2 té problemes"
                elif r1_prob and r2_prob:
                    # Ambdues tenen problemes - triar la de més alçada
                    if result.get("r1_height", 0) >= result.get("r2_height", 0):
                        result["best_replica"] = "R1"
                    else:
                        result["best_replica"] = "R2"
                    result["decision_reason"] = "Ambdues problemàtiques, triar per alçada"
                else:
                    # Cap té problemes - triar per soroll (menys soroll = millor)
                    r1_noise = result.get("r1_noise", 0)
                    r2_noise = result.get("r2_noise", 0)
                    if r1_noise <= r2_noise:
                        result["best_replica"] = "R1"
                    else:
                        result["best_replica"] = "R2"
                    result["decision_reason"] = "Ambdues OK, triar per soroll"

            elif result.get("has_r1"):
                result["best_replica"] = "R1"
                if result.get("r1_predicted_problem"):
                    result["decision_reason"] = "WARNING: Única rèplica amb problemes"
                else:
                    result["decision_reason"] = "Única rèplica disponible"

            elif result.get("has_r2"):
                result["best_replica"] = "R2"
                if result.get("r2_predicted_problem"):
                    result["decision_reason"] = "WARNING: Única rèplica amb problemes"
                else:
                    result["decision_reason"] = "Única rèplica disponible"

            results.append(result)

    return pd.DataFrame(results)


def validate_predictions(df):
    """Valida les prediccions contra les etiquetes manuals amb diferents modes."""
    print("\n" + "="*70)
    print("VALIDACIÓ DE PREDICCIONS - COMPARACIÓ DE MODES")
    print("="*70)
    print(f"\nLlindars utilitzats:")
    print(f"  |As66 - As33| > {THRESH_AS_DIVERGENCE}")
    print(f"  As66 > {THRESH_AS66}")
    print(f"  Pearson <= {THRESH_PEARSON}")

    print(f"\nModes a provar:")
    print(f"  A: (Delta_As > {THRESH_AS_DIVERGENCE}) AND (As66 > {THRESH_AS66})")
    print(f"  B: (As66 > {THRESH_AS66}) AND (Pearson <= {THRESH_PEARSON})")
    print(f"  C: (Delta_As > {THRESH_AS_DIVERGENCE}) AND (Pearson <= {THRESH_PEARSON})")
    print(f"  D: (Delta_As > {THRESH_AS_DIVERGENCE}) OR (As66 > {THRESH_AS66})")
    print(f"  E: ((Delta_As > {THRESH_AS_DIVERGENCE}) OR (As66 > {THRESH_AS66})) AND (Pearson <= {THRESH_PEARSON})")

    # Etiqueta real: OK vs resta
    df["actual_problem"] = df["label"].apply(lambda x: x != "OK" if pd.notna(x) else None)

    # Filtrar pics amb etiqueta
    labeled = df[df["actual_problem"].notna()].copy()

    if len(labeled) == 0:
        print("\nERROR: No hi ha pics etiquetats per validar")
        return df

    # Testar cada mode
    modes = ["A", "B", "C", "D", "E"]
    results = []

    for mode in modes:
        labeled[f"pred_{mode}"], labeled[f"reasons_{mode}"] = zip(
            *labeled.apply(lambda r: predict_peak_quality(r, mode), axis=1)
        )

        tp = ((labeled[f"pred_{mode}"] == True) & (labeled["actual_problem"] == True)).sum()
        tn = ((labeled[f"pred_{mode}"] == False) & (labeled["actual_problem"] == False)).sum()
        fp = ((labeled[f"pred_{mode}"] == True) & (labeled["actual_problem"] == False)).sum()
        fn = ((labeled[f"pred_{mode}"] == False) & (labeled["actual_problem"] == True)).sum()

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total * 100 if total > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "mode": mode, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
        })

    # Taula comparativa
    print(f"\n{'COMPARACIÓ DE MODES':^70}")
    print("-"*70)
    print(f"{'Mode':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-"*70)

    best_mode = None
    best_f1 = 0

    for r in results:
        print(f"{r['mode']:<6} {r['accuracy']:>9.1f}% {r['precision']:>9.1f}% {r['recall']:>9.1f}% {r['f1']:>9.1f}% {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_mode = r["mode"]

    print("-"*70)
    print(f"MILLOR MODE: {best_mode} (F1={best_f1:.1f}%)")

    # Detall del millor mode
    print(f"\n{'DETALL DEL MILLOR MODE (' + best_mode + ')':^70}")
    print("-"*70)

    best_pred = f"pred_{best_mode}"
    best_reasons = f"reasons_{best_mode}"

    # Falsos negatius
    fn_rows = labeled[(labeled[best_pred] == False) & (labeled["actual_problem"] == True)]
    if len(fn_rows) > 0:
        print(f"\nFalsos Negatius ({len(fn_rows)}) - Problemes NO detectats:")
        for _, row in fn_rows.head(10).iterrows():
            as_div = abs(row['As66']-row['As33'])
            print(f"  {row['sample']} P{row['peak_num']} [{row['label']}]: As66={row['As66']:.2f}, Div={as_div:.2f}, Pear={row.get('pearson',0):.4f}")

    # Falsos positius
    fp_rows = labeled[(labeled[best_pred] == True) & (labeled["actual_problem"] == False)]
    if len(fp_rows) > 0:
        print(f"\nFalsos Positius ({len(fp_rows)}) - Falses alarmes:")
        for _, row in fp_rows.head(10).iterrows():
            print(f"  {row['sample']} P{row['peak_num']}: {row[best_reasons]}")

    # Detecció per tipus
    print(f"\nDetecció per tipus de problema:")
    for label in ["NO", "DOBLE", "GHOST", "SMALL"]:
        subset = labeled[labeled["label"] == label]
        if len(subset) == 0:
            continue
        detected = subset[best_pred].sum()
        pct = detected / len(subset) * 100
        print(f"  {label:8s}: {detected:3d}/{len(subset):3d} detectats ({pct:5.1f}%)")

    # Guardar prediccions del millor mode
    df["predicted_problem"] = False
    df["reasons"] = ""
    df.loc[labeled.index, "predicted_problem"] = labeled[best_pred]
    df.loc[labeled.index, "reasons"] = labeled[best_reasons]

    return df


def main():
    print("="*70)
    print("DOCtor VALIDATION")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    df = load_csv()
    if df is None:
        return

    print(f"\nTotal pics carregats: {len(df)}")

    # Validar prediccions
    df = validate_predictions(df)

    # Comparar rèpliques
    print("\n" + "="*70)
    print("SELECCIÓ DE RÈPLIQUES")
    print("="*70)

    replica_df = compare_replicas(df)

    # Resum de decisions
    if len(replica_df) > 0:
        print(f"\nTotal pics analitzats: {len(replica_df)}")
        print("\nDecisions de selecció:")
        for reason, count in replica_df["decision_reason"].value_counts().items():
            print(f"  {count:4d} - {reason}")

        # Validar selecció contra etiquetes
        print("\nValidació de selecció de rèplica:")
        correct = 0
        total_with_both = 0

        for _, row in replica_df.iterrows():
            if row.get("has_r1") and row.get("has_r2"):
                total_with_both += 1
                r1_ok = row.get("r1_label") == "OK"
                r2_ok = row.get("r2_label") == "OK"
                best = row.get("best_replica")

                # Si una és OK i l'altra no, hem de triar la OK
                if r1_ok and not r2_ok:
                    if best == "R1":
                        correct += 1
                elif r2_ok and not r1_ok:
                    if best == "R2":
                        correct += 1
                elif r1_ok and r2_ok:
                    # Ambdues OK, qualsevol elecció és correcta
                    correct += 1

        if total_with_both > 0:
            pct = correct / total_with_both * 100
            print(f"  Encerts en selecció: {correct}/{total_with_both} ({pct:.1f}%)")

    print("\n" + "="*70)
    print("VALIDACIÓ COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n")
    input("Prem ENTER per tancar...")
