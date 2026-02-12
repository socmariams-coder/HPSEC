"""
Genera PDF amb l'analisi del Salt Boundary (SB) al detector DOC.

Inclou:
- Descripcio del fenomen
- Taula de mostres amb SB intens
- Grafics comparatius DOC vs DAD
- Estadistiques per sequencia
- Proposta de limits de fraccions
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime

# -- Paths ----------------------------------------------------------------
TEAMS = Path(
    r"C:\Users\Lequia\OneDrive - Universitat de Girona"
    r"\UdG365_HPLC-DAD - General\Dades"
)
DADES3 = Path(r"C:\Users\Lequia\Desktop\Dades3")
OUTPUT = Path(r"C:\Users\Lequia\Desktop\HPSEC\report_salt_boundary.pdf")

SEQS = {
    "290_SEQ": TEAMS / "290_SEQ",
    "285_SEQ": DADES3 / "285_SEQ",
    "283_SEQ": DADES3 / "283_SEQ",
    "282_SEQ": DADES3 / "282_SEQ",
    "278_SEQ": DADES3 / "278_SEQ",
    "276_SEQ": DADES3 / "276_SEQ",
}

SKIP_NAMES = {"MQ", "NAOH", "BLANC", "BLANK"}


# -- IO -------------------------------------------------------------------

def llegir_uib(path):
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, header=None, engine="python")
            df = df.iloc[:, [0, 1]]
            df.columns = ["t", "DOC"]
            df["t"] = pd.to_numeric(df["t"], errors="coerce")
            df["DOC"] = pd.to_numeric(df["DOC"], errors="coerce")
            df = df.dropna()
            if len(df) > 10:
                return df
        except Exception:
            continue
    return None


def llegir_export3d(path):
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep=",", encoding=enc, engine="python")
            cols = list(df.columns)
            cols[0] = "t"
            out = [cols[0]]
            for c in cols[1:]:
                try:
                    v = float(str(c).strip())
                    out.append(str(int(v)) if v.is_integer() else str(v))
                except Exception:
                    out.append(str(c).strip())
            df.columns = out
            return df
        except Exception:
            continue
    return None


def find_dir(seq_dir, name):
    """Troba subdirectori case-insensitive."""
    for c in seq_dir.iterdir():
        if c.name.lower() == name.lower() and c.is_dir():
            return c
    return None


def find_sb(df, search_min=25, search_max=38):
    """Analitza el pic SB."""
    mask = (df["t"] >= search_min) & (df["t"] <= search_max)
    sub = df[mask].copy()
    if sub.empty:
        return None

    bl_l = df[(df["t"] >= 24) & (df["t"] <= 26)]["DOC"].mean()
    bl_r = df[(df["t"] >= 36) & (df["t"] <= 38)]["DOC"].mean()
    baseline = (bl_l + bl_r) / 2

    idx_max = sub["DOC"].idxmax()
    t_peak = sub.loc[idx_max, "t"]
    doc_peak = sub.loc[idx_max, "DOC"]
    height = doc_peak - baseline

    if height < 5:
        return {"t_peak": t_peak, "height": height, "baseline": baseline,
                "significant": False}

    thr5 = baseline + height * 0.05
    thr50 = baseline + height * 0.50
    rising = sub[sub["t"] < t_peak]
    falling = sub[sub["t"] > t_peak]

    def first_above(s, thr):
        m = s >= thr
        return s[m].iloc[0] if m.any() else search_min

    def first_below(s, thr):
        m = s <= thr
        return s[m].iloc[0] if m.any() else search_max

    t_s5 = first_above(rising.set_index("t")["DOC"], thr5)
    t_e5 = first_below(falling.set_index("t")["DOC"], thr5)
    # For t_s5/t_e5 we need the time index
    r_above = rising[rising["DOC"] >= thr5]
    f_below = falling[falling["DOC"] <= thr5]
    t_s5 = r_above["t"].iloc[0] if len(r_above) > 0 else search_min
    t_e5 = f_below["t"].iloc[0] if len(f_below) > 0 else search_max
    r_above50 = rising[rising["DOC"] >= thr50]
    f_below50 = falling[falling["DOC"] <= thr50]
    t_s50 = r_above50["t"].iloc[0] if len(r_above50) > 0 else search_min
    t_e50 = f_below50["t"].iloc[0] if len(f_below50) > 0 else search_max

    return {
        "t_peak": t_peak, "height": height, "baseline": baseline,
        "t_s5": t_s5, "t_e5": t_e5, "t_s50": t_s50, "t_e50": t_e50,
        "significant": True,
        "pattern": "B" if t_peak > 29.0 else "A",
    }


# -- Carregar dades -------------------------------------------------------

def load_all():
    """Retorna llista de dicts amb totes les dades."""
    records = []
    for seq_name, seq_dir in SEQS.items():
        if not seq_dir.exists():
            print(f"  {seq_name}: no trobat")
            continue
        csv_dir = find_dir(seq_dir, "CSV")
        e3d_dir = find_dir(seq_dir, "Export3d") or find_dir(seq_dir, "Export3D")
        if not csv_dir:
            print(f"  {seq_name}: sense CSV")
            continue

        uib_files = sorted(csv_dir.glob("*UIB1B*"))
        files = [f for f in uib_files
                 if not any(s in f.stem.upper() for s in SKIP_NAMES)]

        for f in files:
            sample = f.stem.replace("_UIB1B", "")
            df_doc = llegir_uib(f)
            if df_doc is None:
                continue

            # DAD (optional)
            df_dad = None
            if e3d_dir:
                dad_path = e3d_dir / f"{sample}.CSV"
                if dad_path.exists():
                    df_dad = llegir_export3d(dad_path)

            sb = find_sb(df_doc)
            records.append({
                "seq": seq_name, "sample": sample,
                "df_doc": df_doc, "df_dad": df_dad, "sb": sb,
            })
        print(f"  {seq_name}: {len([r for r in records if r['seq'] == seq_name])} mostres")
    return records


# -- PDF ------------------------------------------------------------------

def generate_pdf(records):
    with PdfPages(str(OUTPUT)) as pdf:

        # ============================================================
        # PAGE 1: Portada + resum
        # ============================================================
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        title_text = "Salt Boundary (SB) — Artefacte detector DOC"
        ax.text(0.5, 0.82, title_text, ha="center", va="center",
                fontsize=22, fontweight="bold")
        ax.text(0.5, 0.75, f"HPSEC-DAD-DOC  |  {datetime.now().strftime('%d/%m/%Y')}",
                ha="center", va="center", fontsize=12, color="#666666")

        summary = (
            "El Salt Boundary (SB) es un artefacte del detector DOC causat pel\n"
            "front d'elucio de la mostra. La diferencia de forca ionica i pH entre\n"
            "la mostra i la fase movil genera una pertorbacio al senyal DOC que\n"
            "no te corresponencia al DAD (no es materia organica real).\n\n"
            "Caracteristiques clau:\n"
            "  - Apareix al DOC pero NO al DAD (cap absorcio UV)\n"
            "  - Intensitat proporcional a la diferencia de matriu mostra/fase movil\n"
            "  - Repliques identiques (reproduibilitat molt alta)\n"
            "  - No es acumulatiu (no creix entre sequencies)\n\n"
            "S'han identificat dos patrons:\n"
            "  Patro A (normal): pic centrat a ~27 min, altura 20-50 AU\n"
            "  Patro B (intens): pic centrat a ~30.3 min, altura 60-360 AU"
        )
        ax.text(0.08, 0.55, summary, ha="left", va="top",
                fontsize=11, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                          edgecolor="#CCCCCC"))

        # Taula de sequencies analitzades
        seq_stats = {}
        for r in records:
            seq = r["seq"]
            if seq not in seq_stats:
                seq_stats[seq] = {"n": 0, "n_B": 0, "heights": []}
            seq_stats[seq]["n"] += 1
            if r["sb"] and r["sb"]["significant"]:
                seq_stats[seq]["heights"].append(r["sb"]["height"])
                if r["sb"].get("pattern") == "B":
                    seq_stats[seq]["n_B"] += 1

        table_text = f"{'SEQ':12s} {'Total':>6s} {'SB intens':>10s} {'Altura mitj':>12s}\n"
        table_text += "-" * 44 + "\n"
        for seq in SEQS:
            s = seq_stats.get(seq, {"n": 0, "n_B": 0, "heights": []})
            h_mean = np.nanmean(s["heights"]) if s["heights"] else 0
            table_text += f"{seq:12s} {s['n']:6d} {s['n_B']:10d} {h_mean:12.1f}\n"

        ax.text(0.08, 0.13, table_text, ha="left", va="top",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                          edgecolor="#90CAF9"))

        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # PAGE 2: DOC vs DAD — zoom 25-35 min (totes les mostres)
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("DOC vs DAD 254nm — zoom 25-35 min (totes les mostres)",
                     fontsize=14, fontweight="bold", y=0.97)

        ax_doc, ax_dad = axes

        for r in records:
            seq, sample = r["seq"], r["sample"]
            is_290 = "290" in seq
            is_B = r["sb"] and r["sb"].get("pattern") == "B" and "290" not in seq
            if is_290:
                color, lw, alpha = "red", 1.2, 0.7
            elif is_B:
                color, lw, alpha = "orange", 1.0, 0.6
            else:
                color, lw, alpha = "steelblue", 0.5, 0.25

            # DOC
            df = r["df_doc"]
            mask = (df["t"] >= 25) & (df["t"] <= 35)
            ax_doc.plot(df.loc[mask, "t"], df.loc[mask, "DOC"],
                        color=color, lw=lw, alpha=alpha)

            # DAD 254
            df_dad = r["df_dad"]
            if df_dad is not None and "254" in df_dad.columns:
                mask = (df_dad["t"] >= 25) & (df_dad["t"] <= 35)
                ax_dad.plot(df_dad.loc[mask, "t"], df_dad.loc[mask, "254"],
                            color=color, lw=lw, alpha=alpha)

        for ax in [ax_doc, ax_dad]:
            ax.axvline(30, color="gray", ls="--", lw=0.7, alpha=0.5)
            ax.grid(True, alpha=0.2)

        # Legends
        ax_doc.plot([], [], color="red", lw=1.2, label="SEQ 290")
        ax_doc.plot([], [], color="orange", lw=1.0, label="SB intens (altres SEQ)")
        ax_doc.plot([], [], color="steelblue", lw=0.5, alpha=0.4, label="Normal")
        ax_doc.set_xlabel("Temps (min)")
        ax_doc.set_ylabel("DOC (AU)")
        ax_doc.set_title("Senyal DOC (UIB)")
        ax_doc.legend(fontsize=8)

        ax_dad.plot([], [], color="red", lw=1.2, label="SEQ 290")
        ax_dad.plot([], [], color="steelblue", lw=0.5, alpha=0.4, label="Referencia")
        ax_dad.set_xlabel("Temps (min)")
        ax_dad.set_ylabel("Absorbancia 254 nm (AU)")
        ax_dad.set_title("Senyal DAD 254 nm")
        ax_dad.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # PAGE 3: Distribucio SB per sequencia (boxplot + scatter)
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Distribucio del Salt Boundary per sequencia",
                     fontsize=14, fontweight="bold", y=0.97)

        # 3a. Boxplot altura SB
        ax = axes[0]
        seq_names_sorted = list(SEQS.keys())
        data_box = []
        labels_box = []
        for seq in seq_names_sorted:
            heights = [r["sb"]["height"] for r in records
                       if r["seq"] == seq and r["sb"] and r["sb"]["significant"]
                       and not np.isnan(r["sb"]["height"])]
            if heights:
                data_box.append(heights)
                labels_box.append(seq.replace("_SEQ", ""))
        if data_box:
            bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True)
            colors_box = []
            for label in labels_box:
                if "290" in label:
                    colors_box.append("#FFCDD2")
                else:
                    colors_box.append("#BBDEFB")
            for patch, c in zip(bp["boxes"], colors_box):
                patch.set_facecolor(c)
        ax.set_ylabel("Altura SB (AU)")
        ax.set_title("Altura del pic SB")
        ax.grid(True, alpha=0.2, axis="y")

        # 3b. Scatter t_peak vs height (color per patro)
        ax = axes[1]
        for r in records:
            if not r["sb"] or not r["sb"]["significant"]:
                continue
            h = r["sb"]["height"]
            tp = r["sb"]["t_peak"]
            if np.isnan(h):
                continue
            pat = r["sb"].get("pattern", "A")
            if "290" in r["seq"]:
                color, marker = "red", "o"
            elif pat == "B":
                color, marker = "orange", "s"
            else:
                color, marker = "steelblue", "^"
            ax.scatter(tp, h, c=color, marker=marker, s=30, alpha=0.7, edgecolors="none")

        ax.scatter([], [], c="red", marker="o", label="SEQ 290")
        ax.scatter([], [], c="orange", marker="s", label="Patro B (altres)")
        ax.scatter([], [], c="steelblue", marker="^", label="Patro A (normal)")
        ax.set_xlabel("Temps del pic (min)")
        ax.set_ylabel("Altura SB (AU)")
        ax.set_title("Temps vs Intensitat del SB")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # PAGE 4: Espectre UV a t~30 min (290 vs ref)
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Espectre UV a t~30 min — confirma absencia d'absorcio UV",
                     fontsize=14, fontweight="bold", y=0.97)

        # 4a. Espectre
        ax = axes[0]
        for r in records:
            df_dad = r["df_dad"]
            if df_dad is None:
                continue
            t = df_dad["t"].values
            mask = (t >= 29.8) & (t <= 30.2)
            if not mask.any():
                continue
            wl_cols = [c for c in df_dad.columns if c != "t"]
            spectrum = df_dad.loc[mask, wl_cols].mean()
            wls = [float(w) for w in spectrum.index]
            vals = spectrum.values.astype(float)
            is_290 = "290" in r["seq"]
            ax.plot(wls, vals,
                    lw=1.3 if is_290 else 0.5,
                    alpha=0.8 if is_290 else 0.25,
                    color="red" if is_290 else "steelblue")

        ax.plot([], [], color="red", lw=1.3, label="SEQ 290")
        ax.plot([], [], color="steelblue", lw=0.5, alpha=0.4, label="Referencia")
        ax.set_xlabel("Longitud d'ona (nm)")
        ax.set_ylabel("Absorbancia (AU)")
        ax.set_title("Espectre UV a t = 30 min")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # 4b. DOC complet amb zones de fraccions
        ax = axes[1]
        # Pintar zones proposta
        fracs = [
            ("BioP", 10.8, 18, "#E8F5E9"),
            ("HS", 18, 23, "#FFF3E0"),
            ("BB", 23, 26, "#E3F2FD"),
            ("SB", 26, 32, "#FFEBEE"),
            ("LMW", 32, 70, "#F3E5F5"),
        ]
        for name, s, e, c in fracs:
            ax.axvspan(s, e, alpha=0.3, color=c, label=name)
            ax.text((s + e) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0,
                    name, ha="center", va="bottom", fontsize=7, color="#666666")

        for r in records:
            df = r["df_doc"]
            is_290 = "290" in r["seq"]
            ax.plot(df["t"], df["DOC"],
                    lw=0.8 if is_290 else 0.3,
                    alpha=0.7 if is_290 else 0.15,
                    color="red" if is_290 else "steelblue")

        # Re-draw fraction labels at fixed position
        for name, s, e, c in fracs:
            ax.text((s + e) / 2, 5, name, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#444444")

        ax.set_xlabel("Temps (min)")
        ax.set_ylabel("DOC (AU)")
        ax.set_title("Cromatograma DOC complet — fraccions proposades")
        ax.set_xlim(5, 70)
        ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # PAGE 5: Taula detallada de mostres amb SB intens
        # ============================================================
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.5, 0.95, "Mostres amb Salt Boundary intens (Patro B: pic a ~30 min)",
                ha="center", fontsize=14, fontweight="bold")

        # Build table
        intense_records = [r for r in records
                          if r["sb"] and r["sb"]["significant"]
                          and r["sb"].get("pattern") == "B"]

        header = f"{'SEQ':12s} {'Mostra':18s} {'t_pic':>7s} {'Altura':>8s} {'Inici 5%':>9s} {'Fi 5%':>9s} {'FWHM ini':>9s} {'FWHM fi':>9s}"
        lines = [header, "-" * 78]
        for r in sorted(intense_records, key=lambda x: (-x["sb"]["height"])):
            sb = r["sb"]
            lines.append(
                f"{r['seq']:12s} {r['sample']:18s} {sb['t_peak']:7.2f} {sb['height']:8.1f} "
                f"{sb['t_s5']:9.2f} {sb['t_e5']:9.2f} {sb['t_s50']:9.2f} {sb['t_e50']:9.2f}"
            )
        lines.append("")
        lines.append(f"Total: {len(intense_records)} repliques amb SB intens")

        # Group by seq
        by_seq = {}
        for r in intense_records:
            by_seq.setdefault(r["seq"], []).append(r["sample"])
        lines.append("")
        for seq, samples in sorted(by_seq.items()):
            names = sorted(set(s.rsplit("_", 1)[0] for s in samples))
            lines.append(f"  {seq}: {', '.join(names)}")

        ax.text(0.05, 0.88, "\n".join(lines), ha="left", va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA",
                          edgecolor="#DDDDDD"))

        # Limits de fraccions (actualitzats)
        proposal = (
            "LIMITS DE FRACCIONS (ACTUALITZATS)\n"
            "====================================\n\n"
            "  BioP   10.8 - 18.0   Biopolimers\n"
            "  HS     18.0 - 23.0   Acids Humics\n"
            "  BB     23.0 - 26.0   Building Blocks\n"
            "  SB     26.0 - 32.0   Salt Boundary (artefacte DOC)\n"
            "  LMW    32.0 - 70.0   Low Molecular Weight\n\n"
            "La fraccio SB (26-32 min) captura els dos patrons:\n"
            "  Patro A (pic ~27 min): inici 26 - fi 29\n"
            "  Patro B (pic ~30 min): inici 29.4 - fi 31.7\n\n"
            "Aixi s'evita que l'artefacte contamini BB o LMW.\n"
            "Configurat a hpsec_config.json (commit a923b4d)."
        )
        ax.text(0.05, 0.27, proposal, ha="left", va="top",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                          edgecolor="#A5D6A7"))

        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # PAGE 6: Exemples individuals (4 mostres)
        # ============================================================
        # Pick 4 representative samples
        examples = []
        # 1 from 290 with high SB
        for r in records:
            if r["seq"] == "290_SEQ" and "LQ0303" in r["sample"] and "_1" in r["sample"]:
                examples.append(r)
                break
        # 1 from 282 with moderate SB
        for r in records:
            if r["seq"] == "282_SEQ" and "FR2588" in r["sample"] and "_1" in r["sample"]:
                examples.append(r)
                break
        # 1 normal from 278
        for r in records:
            if r["seq"] == "278_SEQ" and "FR2546" in r["sample"] and "_1" in r["sample"]:
                examples.append(r)
                break
        # 1 normal from 285
        for r in records:
            if r["seq"] == "285_SEQ" and "LQ0149" in r["sample"] and "_1" in r["sample"]:
                examples.append(r)
                break

        if examples:
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            fig.suptitle("Exemples individuals — DOC zoom 22-38 min",
                         fontsize=14, fontweight="bold", y=0.97)

            for idx, (ax, r) in enumerate(zip(axes.flat, examples)):
                df = r["df_doc"]
                sb = r["sb"]
                mask = (df["t"] >= 22) & (df["t"] <= 38)
                ax.plot(df.loc[mask, "t"], df.loc[mask, "DOC"],
                        color="red" if "290" in r["seq"] else "#333333", lw=1.2)

                if sb and sb["significant"]:
                    ax.axvline(sb["t_peak"], color="red", ls="--", lw=0.8, alpha=0.6)
                    ax.axvspan(sb["t_s5"], sb["t_e5"], alpha=0.1, color="red")
                    ax.axvspan(sb["t_s50"], sb["t_e50"], alpha=0.15, color="orange")
                    pat = sb.get("pattern", "?")
                    ax.text(0.98, 0.95,
                            f"Patro {pat}\nt_pic={sb['t_peak']:.2f}\naltura={sb['height']:.0f}\n5%: {sb['t_s5']:.1f}-{sb['t_e5']:.1f}",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=8, family="monospace",
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#CCC"))

                # Fraction boundaries (current)
                for t_frac in [23, 26, 32]:
                    ax.axvline(t_frac, color="green", ls="--", lw=0.8, alpha=0.5)
                ax.axvspan(26, 32, alpha=0.06, color="red")

                ax.set_title(f"{r['seq']} / {r['sample']}", fontsize=10)
                ax.set_xlabel("Temps (min)", fontsize=8)
                ax.set_ylabel("DOC (AU)", fontsize=8)
                ax.grid(True, alpha=0.2)

            # Fill remaining axes if less than 4 examples
            for idx in range(len(examples), 4):
                axes.flat[idx].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF generat: {OUTPUT}")


# -- Main -----------------------------------------------------------------

def main():
    print("Carregant dades...")
    records = load_all()
    print(f"\nTotal: {len(records)} mostres de {len(SEQS)} sequencies")
    print("\nGenerant PDF...")
    generate_pdf(records)
    print("Fet!")


if __name__ == "__main__":
    main()
