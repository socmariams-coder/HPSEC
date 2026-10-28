#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPSEC Viewer (lite) — visor de cromatogrames.

Versió lleugera i independent de la Suite: NOMÉS mostra cromatogrames.
No importa, no analitza, no calibra. Obre una carpeta de dades, tria una
seqüència i dibuixa els senyals DOC (TOC/OCD) i UV (DAD) de cada mostra,
amb superposició i normalització per comparar formes entre regims/concentracions.

Fonts de traça (per cada seqüència, en ordre de preferència):
  1. RESULTATS/traces/<mostra>_SEC.csv   -> DOC + totes les λ DAD (processat, net)
  2. CHECK/data/calibration_result.json  -> KHP per nivell·rèplica (DOC + A254; SEQ_CAL)
  3. CSV/<mostra>_UIB1B.CSV + Export3d/DAD1A  -> senyals UV crus (qualsevol SEQ)

Ús:  python hpsec_viewer.py
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd

# Força el binding Qt a PySide6 ABANS de carregar el backend de matplotlib,
# perquè matplotlib no esculli un altre binding (PyQt) i barregi tipus QWidget.
os.environ.setdefault("QT_API", "pyside6")
from PySide6 import QtWidgets, QtCore  # noqa: E402  (ha d'anar primer)

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ---------------------------------------------------------------------------
# Lectura de fitxers (independent de la Suite)
# ---------------------------------------------------------------------------

def _default_data_folders():
    """Carpetes de dades per defecte (prova la config de la Suite, si hi és)."""
    folders = []
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hpsec_config  # noqa
        folders = list(hpsec_config.get_data_folders() or [])
    except Exception:
        folders = []
    for cand in (r"C:/Users/maria/Proyectos/Dades3",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dades3")):
        if os.path.isdir(cand) and cand not in folders:
            folders.append(cand)
    return folders


def _read_sec_csv(path):
    """RESULTATS/traces/*_SEC.csv -> dict de senyals {nom: (t, y)}."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time_min") or df.columns[0]
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    out = {}
    if "doc" in cols:
        y = pd.to_numeric(df[cols["doc"]], errors="coerce").to_numpy()
        out["DOC"] = (t, y)
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("a") and lc[1:].isdigit():
            y = pd.to_numeric(df[c], errors="coerce").to_numpy()
            out["A" + lc[1:]] = (t, y)
    return out


def _read_uib_raw(path):
    """CSV UIB (tab, UTF-16 o UTF-8) -> (t, y)."""
    for enc in ("utf-16", "utf-8", "latin1"):
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, header=None, engine="python")
            t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            m = t.notna() & y.notna()
            if m.sum() >= 10:
                return t[m].to_numpy(), y[m].to_numpy()
        except Exception:
            continue
    return None


def _read_dad1a_raw(path):
    """CSV DAD1A (tab, UTF-16, sense capçalera) -> (t, y) a la λ exportada."""
    for enc in ("utf-16", "utf-16-le", "utf-8", "latin1"):
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, header=None, engine="python")
            t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            m = t.notna() & y.notna()
            if m.sum() >= 10:
                return t[m].to_numpy(), y[m].to_numpy()
        except Exception:
            continue
    return None


def _read_export3d_254(path):
    """Export3d (comma, UTF-16) -> (t, y) a 254 nm."""
    for enc in ("utf-16", "utf-8", "latin1"):
        try:
            df = pd.read_csv(path, sep=",", encoding=enc, engine="python")
            if df.shape[1] < 2:
                continue
            t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            col254 = None
            for c in df.columns[1:]:
                try:
                    if int(float(str(c).strip())) == 254:
                        col254 = c
                        break
                except Exception:
                    continue
            if col254 is None:
                continue
            y = pd.to_numeric(df[col254], errors="coerce")
            m = t.notna() & y.notna()
            if m.sum() >= 10:
                return t[m].to_numpy(), y[m].to_numpy()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Model de traces per seqüència
# ---------------------------------------------------------------------------

class TraceItem:
    """Una entrada seleccionable a la llista (una mostra/rèplica)."""
    def __init__(self, label, kind, payload):
        self.label = label      # text a la llista
        self.kind = kind        # 'sec' | 'calref' | 'raw'
        self.payload = payload  # info per carregar sota demanda

    def load(self):
        """Retorna dict {nom_senyal: (t, y)}."""
        if self.kind == "sec":
            return _read_sec_csv(self.payload["path"])
        if self.kind == "calref":
            rep = self.payload["rep"]
            out = {}
            t_doc = rep.get("t_doc"); y_doc = rep.get("y_doc")
            if t_doc and y_doc:
                out["DOC"] = (np.asarray(t_doc, float), np.asarray(y_doc, float))
            t_dad = rep.get("t_dad"); y254 = rep.get("y_dad_254")
            if t_dad and y254:
                out["A254"] = (np.asarray(t_dad, float), np.asarray(y254, float))
            return out
        if self.kind == "raw":
            out = {}
            up = self.payload.get("uib")
            if up:
                r = _read_uib_raw(up)
                if r:
                    out["UIB"] = r
            ep = self.payload.get("export3d")
            dp = self.payload.get("dad1a")
            r = _read_export3d_254(ep) if ep else None
            if r is None and dp:
                r = _read_dad1a_raw(dp)
            if r is not None:
                out["A254"] = r
            return out
        return {}


def discover_sequences(folder):
    """Subcarpetes que semblen seqüències HPSEC."""
    seqs = []
    try:
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if not os.path.isdir(p):
                continue
            has = (glob.glob(os.path.join(p, "*MasterFile*.xlsx"))
                   or os.path.isdir(os.path.join(p, "CHECK"))
                   or os.path.isdir(os.path.join(p, "RESULTATS"))
                   or os.path.isdir(os.path.join(p, "CSV")))
            if has:
                seqs.append(name)
    except Exception:
        pass
    return seqs


def build_items(seq_path):
    """Construeix la llista de TraceItem per una seqüència i diu quina font fa servir."""
    items = []

    # 1) RESULTATS/traces/*_SEC.csv
    traces_dir = os.path.join(seq_path, "RESULTATS", "traces")
    if os.path.isdir(traces_dir):
        for f in sorted(glob.glob(os.path.join(traces_dir, "*_SEC.csv"))):
            name = os.path.basename(f)[:-len("_SEC.csv")]
            items.append(TraceItem(name, "sec", {"path": f}))
        if items:
            return items, "RESULTATS/traces (processat)"

    # 2) calibration_result.json (SEQ_CAL)
    calref = os.path.join(seq_path, "CHECK", "data", "calibration_result.json")
    if os.path.isfile(calref):
        try:
            d = json.load(open(calref, encoding="utf-8"))
            cals = d.get("calibrations", [])
            # una sola llista de nivells (evitem duplicar les 5 cals redundants:
            # agrupem per conc i agafem el primer bloc que té traces)
            seen = set()
            for c in cals:
                conc = c.get("conc_ppm")
                mode = c.get("mode", "")
                key = (mode, conc)
                if key in seen:
                    continue
                reps = c.get("replicas_info") or []
                if not any(r.get("t_doc") or r.get("t_dad") for r in reps):
                    continue
                seen.add(key)
                for r in reps:
                    n = r.get("replica_num", "?")
                    lbl = f"KHP {conc} ppm · R{n}"
                    if mode and mode != "COLUMN":
                        lbl = f"[{mode}] " + lbl
                    items.append(TraceItem(lbl, "calref", {"rep": r}))
            if items:
                items.sort(key=lambda it: (it.payload["rep"].get("replica_num", 0)))
                return items, "calibration_result.json (SEQ_CAL)"
        except Exception:
            pass

    # 3) CSV crus (qualsevol SEQ)
    csv_dir = os.path.join(seq_path, "CSV")
    ex_dir = os.path.join(seq_path, "Export3d")
    if os.path.isdir(csv_dir):
        bases = {}
        for f in glob.glob(os.path.join(csv_dir, "*.CSV")) + glob.glob(os.path.join(csv_dir, "*.csv")):
            fn = os.path.basename(f)
            up = fn.upper()
            import re
            m = re.match(r"(.+?)_(UIB1B|DAD1A|VWD1A|VWD1B)\d*\.CSV$", up)
            if not m:
                continue
            base = fn[:m.start(2) - 1]  # retalla "_UIB1B.CSV" etc (respecta majúscules del nom)
            kind = m.group(2)
            bases.setdefault(base, {})
            if kind == "UIB1B":
                bases[base]["uib"] = f
            elif kind in ("DAD1A", "VWD1A"):
                bases[base].setdefault("dad1a", f)
        # Export3d per λ completa (254)
        if os.path.isdir(ex_dir):
            for f in glob.glob(os.path.join(ex_dir, "*.CSV")) + glob.glob(os.path.join(ex_dir, "*.csv")):
                base = os.path.splitext(os.path.basename(f))[0]
                if base in bases:
                    bases[base]["export3d"] = f
        for base in sorted(bases):
            items.append(TraceItem(base, "raw", bases[base]))
        if items:
            return items, "CSV crus (UV: UIB + DAD 254)"

    return items, "(sense cromatogrames trobats)"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

_DOC_SIGNALS = ("DOC", "UIB", "DOC(UIB)")

class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HPSEC Viewer — cromatogrames (lite)")
        self.resize(1180, 760)

        self._folders = _default_data_folders()
        self._items = []          # TraceItem actuals
        self._source_desc = ""

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # --- Panell esquerre (controls) ---
        left = QtWidgets.QVBoxLayout()

        # Carpeta
        fl = QtWidgets.QHBoxLayout()
        self.folder_combo = QtWidgets.QComboBox()
        self.folder_combo.addItems(self._folders)
        self.folder_combo.currentIndexChanged.connect(self._reload_sequences)
        btn_browse = QtWidgets.QPushButton("…")
        btn_browse.setFixedWidth(30)
        btn_browse.setToolTip("Tria una altra carpeta de dades")
        btn_browse.clicked.connect(self._browse_folder)
        fl.addWidget(QtWidgets.QLabel("Carpeta:"))
        fl.addWidget(self.folder_combo, 1)
        fl.addWidget(btn_browse)
        left.addLayout(fl)

        # Seqüència
        self.seq_combo = QtWidgets.QComboBox()
        self.seq_combo.currentIndexChanged.connect(self._on_seq_changed)
        left.addWidget(QtWidgets.QLabel("Seqüència:"))
        left.addWidget(self.seq_combo)

        self.source_lbl = QtWidgets.QLabel("")
        self.source_lbl.setWordWrap(True)
        self.source_lbl.setStyleSheet("color:#555; font-size:11px;")
        left.addWidget(self.source_lbl)

        # Mostres
        left.addWidget(QtWidgets.QLabel("Mostres (selecció múltiple):"))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.sample_list.itemSelectionChanged.connect(self._redraw)
        left.addWidget(self.sample_list, 1)

        # Senyals
        box = QtWidgets.QGroupBox("Senyals i opcions")
        bl = QtWidgets.QVBoxLayout(box)
        self.cb_doc = QtWidgets.QCheckBox("DOC (TOC/OCD)")
        self.cb_doc.setChecked(True)
        self.cb_dad = QtWidgets.QCheckBox("DAD UV")
        self.cb_dad.setChecked(True)
        self.cb_uib = QtWidgets.QCheckBox("UIB (UV inline)")
        self.cb_uib.setChecked(False)
        for cb in (self.cb_doc, self.cb_dad, self.cb_uib):
            cb.stateChanged.connect(self._redraw)
            bl.addWidget(cb)
        wl = QtWidgets.QHBoxLayout()
        wl.addWidget(QtWidgets.QLabel("λ DAD:"))
        self.wl_combo = QtWidgets.QComboBox()
        self.wl_combo.currentIndexChanged.connect(self._redraw)
        wl.addWidget(self.wl_combo, 1)
        bl.addLayout(wl)
        self.cb_norm = QtWidgets.QCheckBox("Normalitza (0–1) per comparar formes")
        self.cb_norm.stateChanged.connect(self._redraw)
        bl.addWidget(self.cb_norm)
        left.addWidget(box)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(330)
        root.addWidget(lw, 0)

        # --- Panell dret (gràfic) ---
        right_w = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_w)
        root.addWidget(right_w, 1)
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, right_w)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 1)

        self.statusBar().showMessage("Llest.")

        self._reload_sequences()

    # --- càrrega ---
    def _browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Tria carpeta de dades")
        if d:
            if self.folder_combo.findText(d) < 0:
                self.folder_combo.addItem(d)
            self.folder_combo.setCurrentText(d)

    def _current_folder(self):
        return self.folder_combo.currentText()

    def _reload_sequences(self):
        folder = self._current_folder()
        self.seq_combo.blockSignals(True)
        self.seq_combo.clear()
        if folder and os.path.isdir(folder):
            self.seq_combo.addItems(discover_sequences(folder))
        self.seq_combo.blockSignals(False)
        self._on_seq_changed()

    def _on_seq_changed(self):
        self.sample_list.clear()
        self._items = []
        seq = self.seq_combo.currentText()
        folder = self._current_folder()
        if not seq:
            self.source_lbl.setText("")
            self._redraw()
            return
        seq_path = os.path.join(folder, seq)
        try:
            self._items, self._source_desc = build_items(seq_path)
        except Exception as e:
            self._items, self._source_desc = [], f"error: {e}"
        self.source_lbl.setText(f"Font: {self._source_desc}  ·  {len(self._items)} mostres")
        for it in self._items:
            self.sample_list.addItem(it.label)
        # selecciona la primera per defecte
        if self._items:
            self.sample_list.setCurrentRow(0)
        self._refresh_wavelengths()
        self._redraw()

    def _refresh_wavelengths(self):
        """Omple el combo de λ segons la font (RESULTATS té totes; la resta, 254)."""
        self.wl_combo.blockSignals(True)
        self.wl_combo.clear()
        wls = set()
        for it in self._items[:1]:  # n'hi ha prou amb la 1a per saber quines λ hi ha
            try:
                sig = it.load()
            except Exception:
                sig = {}
            for k in sig:
                if k.startswith("A") and k[1:].isdigit():
                    wls.add(int(k[1:]))
        if not wls:
            wls = {254}
        for w in sorted(wls):
            self.wl_combo.addItem(f"{w} nm", w)
        idx = self.wl_combo.findData(254)
        self.wl_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.wl_combo.setEnabled(len(wls) > 1)
        self.wl_combo.blockSignals(False)

    # --- dibuix ---
    def _selected_items(self):
        rows = sorted(i.row() for i in self.sample_list.selectedIndexes())
        return [self._items[r] for r in rows if 0 <= r < len(self._items)]

    def _redraw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax2 = None

        sel = self._selected_items()
        wl = self.wl_combo.currentData() or 254
        dad_key = f"A{int(wl)}"
        norm = self.cb_norm.isChecked()
        show_doc = self.cb_doc.isChecked()
        show_dad = self.cb_dad.isChecked()
        show_uib = self.cb_uib.isChecked()

        cmap = matplotlib.colormaps.get_cmap("tab10")
        plotted_doc = plotted_dad = False
        n_err = 0

        for idx, it in enumerate(sel):
            color = cmap(idx % 10)
            try:
                sig = it.load()
            except Exception:
                n_err += 1
                continue

            def _prep(y):
                y = np.asarray(y, float)
                if norm:
                    rng = np.nanmax(y) - np.nanmin(y)
                    if rng > 0:
                        y = (y - np.nanmin(y)) / rng
                return y

            if show_doc and "DOC" in sig:
                t, y = sig["DOC"]
                ax.plot(t, _prep(y), color=color, lw=1.3, label=f"{it.label} · DOC")
                plotted_doc = True
            if show_uib and "UIB" in sig:
                t, y = sig["UIB"]
                ax.plot(t, _prep(y), color=color, lw=1.0, ls=":", label=f"{it.label} · UIB")
                plotted_doc = True
            if show_dad and dad_key in sig:
                t, y = sig[dad_key]
                tgt = ax if norm else (ax2 or ax.twinx())
                if not norm and ax2 is None:
                    ax2 = tgt
                tgt.plot(t, _prep(y), color=color, lw=1.0, ls="--",
                         label=f"{it.label} · {dad_key}")
                plotted_dad = True

        ax.set_xlabel("temps (min)")
        if norm:
            ax.set_ylabel("senyal normalitzat (0–1)")
        else:
            ax.set_ylabel("DOC / UIB (senyal)")
            if ax2 is not None:
                ax2.set_ylabel(f"DAD {int(wl)} nm")
        ax.grid(True, alpha=0.25)

        # llegenda combinada
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2; labels += l2
        if handles:
            ax.legend(handles, labels, fontsize=8, loc="best", framealpha=0.9)

        title = self.seq_combo.currentText()
        if not sel:
            title += "  —  (tria una o més mostres)"
        ax.set_title(title, fontsize=11)

        self.canvas.draw_idle()
        msg = f"{len(sel)} mostra(es)"
        if norm:
            msg += " · normalitzat"
        if n_err:
            msg += f" · {n_err} amb error de lectura"
        self.statusBar().showMessage(msg)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ViewerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
