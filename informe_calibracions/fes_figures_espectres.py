# -*- coding: utf-8 -*-
"""Espectres UV de TOTS els pics d'una injeccio de KHP.

Una injeccio de patro no conte nomes el KHP: hi apareixen altres estructures
(pics de sistema, anions inorganics, artefactes del gradient). L'Export3d del DAD
guarda 101 longituds d'ona (200-400 nm), de manera que cada pic te espectre propi
i es pot classificar sense injectar patrons.

Criteris de lectura:
  - KHP (ftalat) es AROMATIC: banda del benzē amb cua fins a ~280 nm.
  - Els anions inorganics (nitrat, nitrit) absorbeixen fort per sota de 230 nm i
    son transparents per sobre de 240: no tenen banda aromatica.
  - Un pic que nomes "es veu" a 254 perque hi arriba la cua d'una banda de 206 nm
    no es un cromofor de 254.

Genera:
  esp_cromatograma.png  — cromatograma multi-λ amb els pics localitzats
  esp_espectres.png     — espectre de cada pic, cru i normalitzat

Us:  python -X utf8 fes_figures_espectres.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = os.path.dirname(os.path.abspath(__file__))
DADES = r"C:\Users\maria\Proyectos\Dades3"

# Injeccio de referencia: 5 ppm de la 293 (millor relacio senyal/soroll; el KHP
# hi domina) i 0,1 ppm (on el pic de sistema queda resolt i no enterrat).
INJ = ("293_SEQ_CAL", "KHP5_1.CSV")
INJ_BAIXA = ("293_SEQ_CAL", "KHP01_1.CSV")

LAMBDES = [210, 230, 254, 280]
COLORS = {210: "#8E44AD", 230: "#E67E22", 254: "#2563EB", 280: "#27AE60"}


def llegeix(seq, fitxer):
    p = os.path.join(DADES, seq, "Export3d", fitxer)
    df = pd.read_csv(p, encoding="utf-16")
    t = df.iloc[:, 0].to_numpy(dtype=float)
    wl = np.array([float(c) for c in df.columns[1:]])
    A = df.iloc[:, 1:].to_numpy(dtype=float)
    return t, wl, A


def traca(t, wl, A, lam):
    i = int(np.argmin(np.abs(wl - lam)))
    return A[:, i]


def lleva_deriva(t, y, finestra=8.0):
    """Base movil per percentil: el gradient fa derivar el DAD tot el run."""
    dt = float(np.median(np.diff(t)))
    n = max(int(finestra / dt), 21)
    s = pd.Series(y)
    return y - s.rolling(n, center=True, min_periods=1).quantile(0.10).to_numpy()


def troba_pics(t, y, prom):
    idx, props = find_peaks(y, prominence=prom, distance=int(0.5 / np.median(np.diff(t))))
    return [(float(t[i]), float(y[i]), float(p))
            for i, p in zip(idx, props["prominences"])]


def espectre(t, A, t_obj, ample=0.06, sep=1.2):
    """Espectre a t_obj menys una base presa als dos costats."""
    m = np.abs(t - t_obj) <= ample
    b = ((t >= t_obj - sep - 0.5) & (t <= t_obj - sep)) | \
        ((t >= t_obj + sep) & (t <= t_obj + sep + 0.5))
    if m.sum() < 2 or b.sum() < 2:
        return None
    return A[m].mean(axis=0) - A[b].mean(axis=0)


def classifica(wl, s):
    """Retorna (lambda_max, A254/Amax, A280/Amax, veredicte)."""
    i254 = int(np.argmin(np.abs(wl - 254)))
    i280 = int(np.argmin(np.abs(wl - 280)))
    m = wl <= 360
    i_max = int(np.argmax(s[m]))
    lmax = float(wl[m][i_max])
    amax = float(s[m][i_max])
    if amax <= 0:
        return lmax, np.nan, np.nan, "—"
    r254, r280 = s[i254] / amax, s[i280] / amax
    if lmax <= 215 and r254 < 0.02:
        v = "UV llunyà, sense banda aromàtica → inorgànic probable"
    elif r280 > 0.05:
        v = "banda aromàtica marcada"
    elif r254 > 0.02:
        v = "cua fins a 254: aromaticitat feble"
    else:
        v = "UV llunyà"
    return lmax, r254, r280, v


def main():
    seq, fitxer = INJ
    t, wl, A = llegeix(seq, fitxer)
    print(f"{seq} · {fitxer}: {A.shape[0]} punts × {A.shape[1]} λ "
          f"({wl.min():.0f}–{wl.max():.0f} nm)")

    # Deteccio de pics sobre 210 nm: hi surt tot, aromatic o no
    y210 = lleva_deriva(t, traca(t, wl, A, 210))
    pics = troba_pics(t, y210, prom=1.5)
    # dins de la finestra util del run (fora del front d'injeccio i del final)
    pics = [p for p in pics if 2.0 <= p[0] <= 72.0]
    pics.sort(key=lambda x: -x[2])
    pics = sorted(pics[:8], key=lambda x: x[0])

    print(f"\n{'t (min)':>9}{'A210':>9}{'A254':>9}{'λmax':>8}{'A254/Amax':>11}"
          f"{'A280/Amax':>11}  interpretació")
    print("-" * 104)
    y254 = lleva_deriva(t, traca(t, wl, A, 254))
    espectres = []
    for t_p, h210, prom in pics:
        s = espectre(t, A, t_p)
        if s is None:
            continue
        i = int(np.argmin(np.abs(t - t_p)))
        lmax, r254, r280, v = classifica(wl, s)
        espectres.append((t_p, s, lmax, v))
        print(f"{t_p:>9.2f}{h210:>9.2f}{y254[i]:>9.2f}{lmax:>8.0f}"
              f"{r254:>11.3f}{r280:>11.3f}  {v}")

    # ---- Figura 1: cromatograma multi-λ amb els pics marcats ----
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True)
    for lam in LAMBDES:
        y = lleva_deriva(t, traca(t, wl, A, lam))
        axes[0].plot(t, y, color=COLORS[lam], lw=0.8, label=f"{lam} nm")
    for t_p, _, _, _ in espectres:
        axes[0].axvline(t_p, color="#999", ls=":", lw=0.8)
        axes[0].annotate(f"{t_p:.1f}", xy=(t_p, axes[0].get_ylim()[1]),
                         xytext=(0, -12), textcoords="offset points",
                         ha="center", fontsize=7, color="#555")
    axes[0].set_ylabel("absorbància (mAU)", fontsize=10)
    axes[0].set_title(f"{seq} · {fitxer.replace('.CSV','')} — cromatograma DAD a diverses λ "
                      "(base mòbil llevada)", fontsize=11, loc="left")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    for lam in LAMBDES:
        y = lleva_deriva(t, traca(t, wl, A, lam))
        axes[1].plot(t, y, color=COLORS[lam], lw=0.9)
    axes[1].set_ylim(-1, 8)
    axes[1].set_xlabel("t (min)", fontsize=10)
    axes[1].set_ylabel("absorbància (mAU)", fontsize=10)
    axes[1].set_title("mateix cromatograma, escala ampliada", fontsize=11, loc="left")
    axes[1].grid(alpha=0.25)
    for t_p, _, _, _ in espectres:
        axes[1].axvline(t_p, color="#999", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "esp_cromatograma.png"), dpi=110)
    plt.close(fig)
    print("\n   esp_cromatograma.png")

    # ---- Figura 2: espectres ----
    n = len(espectres)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n))
    for (t_p, s, lmax, v), c in zip(espectres, cmap):
        et = f"{t_p:.1f} min (λmax {lmax:.0f})"
        axes[0].plot(wl, s, color=c, lw=1.4, label=et)
        m = wl <= 360
        pic = np.max(s[m])
        if pic > 0:
            axes[1].plot(wl, s / pic, color=c, lw=1.4, label=et)
    for ax, tit in zip(axes, ["Espectres crus", "Normalitzats al màxim"]):
        ax.axvline(254, color="#8E44AD", ls=":", lw=1.0)
        ax.set_xlim(200, 360)
        ax.set_xlabel("longitud d'ona (nm)", fontsize=10)
        ax.set_title(tit, fontsize=11, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("absorbància (mAU)", fontsize=10)
    axes[1].set_ylabel("absorbància normalitzada", fontsize=10)
    fig.suptitle(f"{seq} · {fitxer.replace('.CSV','')} — espectre UV del pic de KHP i de la "
                 "resta de pics de la mateixa injecció", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(SCR, "esp_espectres.png"), dpi=110)
    plt.close(fig)
    print("   esp_espectres.png")


if __name__ == "__main__":
    main()
