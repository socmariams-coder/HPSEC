"""
HPSEC Suite - Samples Panel (Mostres)
======================================

Simplified panel:
1. SEQ selector for packaging
2. Global sample inventory (COL<->BP comparison)
3. Packaging section (collapsible)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QAbstractItemView, QSizePolicy, QGroupBox, QCheckBox,
    QFileDialog, QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor, QBrush

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type
from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED,
    PANEL_MARGINS, PANEL_SPACING,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_analysis_json(seq_path):
    """Load analysis_result.json from a SEQ folder."""
    json_path = os.path.join(seq_path, "CHECK", "data", "analysis_result.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", json_path, e)
        return None


def _format_ppm(val):
    if val is None:
        return "\u2014"
    return f"{val:.2f}"


def _format_area(val):
    if val is None or val == 0:
        return "\u2014"
    return f"{val:.1f}"


def _get_selected_replica(sample_data, signal="doc"):
    """Return the selected replica dict for a signal."""
    selected = sample_data.get("selected", {}) or {}
    sel_key = selected.get(signal, selected.get("doc", "1"))
    replicas = sample_data.get("replicas", {}) or {}
    return replicas.get(sel_key, {})


def _extract_sample_metrics(sample_data):
    """Extract display metrics from a sample_data dict."""
    quant = sample_data.get("quantification", {}) or {}
    doc_rep = _get_selected_replica(sample_data, "doc")
    dad_rep = _get_selected_replica(sample_data, "dad")

    areas = (doc_rep.get("areas") or {}).get("DOC") or {}
    dad_info = dad_rep.get("dad_info") or {}

    return {
        "ppm": quant.get("concentration_ppm_direct"),
        "a254": dad_info.get("area_254"),
        "area_total": areas.get("total", 0),
        "sample_type": sample_data.get("sample_type", "SAMPLE"),
    }


# ---------------------------------------------------------------------------
# Background scanner
# ---------------------------------------------------------------------------

class _ScanWorker(QThread):
    """Scan data_folders for all SEQs with analysis_result.json."""
    finished = Signal(list)
    progress = Signal(str)

    def run(self):
        seqs = []
        for folder in get_data_folders():
            if not os.path.isdir(folder):
                continue
            try:
                entries = sorted(os.listdir(folder))
            except OSError:
                continue
            for d in entries:
                full = os.path.join(folder, d)
                if not os.path.isdir(full):
                    continue
                json_path = os.path.join(
                    full, "CHECK", "data", "analysis_result.json")
                if os.path.exists(json_path):
                    method = detect_seq_type(d)
                    seq_num = extract_seq_number(d)
                    mtime = os.path.getmtime(json_path)
                    seqs.append({
                        "name": d,
                        "path": full,
                        "method": method,
                        "seq_num": seq_num,
                        "json_path": json_path,
                        "mtime": mtime,
                    })
        # Sort by seq number descending
        seqs.sort(key=lambda s: s.get("seq_num") or 0, reverse=True)
        self.finished.emit(seqs)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class SamplesDBPanel(QWidget):
    """Panel de mostres -- inventari global COL/BP."""

    sample_selected = Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._initialized = False

        # State
        self._all_seqs = []          # list of seq info dicts
        self._seq_cache = {}         # {seq_path: analysis_data}
        self._inventory = {}         # {sample_name: {col_*, bp_*}}

        # Workers
        self._scan_worker = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PANEL_MARGINS)
        layout.setSpacing(PANEL_SPACING)

        # --- Header row ---
        header_row = QHBoxLayout()

        self._info_label = QLabel("")
        self._info_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        header_row.addWidget(self._info_label)
        header_row.addStretch()

        # Refresh
        self._refresh_btn = QPushButton("Actualitzar")
        self._refresh_btn.clicked.connect(self._scan_seqs)
        header_row.addWidget(self._refresh_btn)

        layout.addLayout(header_row)

        # --- Filter bar ---
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Cercar:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Nom mostra...")
        self._search_edit.setMaximumWidth(220)
        self._search_edit.textChanged.connect(self._filter_inventory_table)
        filt_row.addWidget(self._search_edit)

        filt_row.addSpacing(12)
        filt_row.addWidget(QLabel("SEQ:"))
        self._inv_seq_filter = QComboBox()
        self._inv_seq_filter.addItem("Totes", None)
        self._inv_seq_filter.setMinimumWidth(140)
        self._inv_seq_filter.currentIndexChanged.connect(
            self._filter_inventory_table)
        filt_row.addWidget(self._inv_seq_filter)
        filt_row.addStretch()

        self._inv_count_label = QLabel("")
        self._inv_count_label.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        filt_row.addWidget(self._inv_count_label)

        layout.addLayout(filt_row)

        # --- Comparison table ---
        self._inv_table = QTableWidget()
        self._inv_table.setColumnCount(9)
        self._inv_table.setHorizontalHeaderLabels([
            "Mostra",
            "SEQ COL", "SEQ BP",
            "ppm COL", "ppm BP", "Ratio",
            "A254 COL", "A254 BP", "Ratio A254",
        ])
        self._configure_table(self._inv_table)
        layout.addWidget(self._inv_table, 1)

        # --- Packaging (collapsible) ---
        self._pkg_group = self._build_packaging_section()
        layout.addWidget(self._pkg_group)

    # ------------------------------------------------------------------
    # Packaging section (collapsible)
    # ------------------------------------------------------------------

    def _build_packaging_section(self):
        grp = QGroupBox("Empaquetar")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.setStyleSheet("QGroupBox { font-weight: bold; }")

        lay = QVBoxLayout(grp)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # SEQ selector row
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Seqüència:"))
        self._seq_combo = QComboBox()
        self._seq_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        seq_row.addWidget(self._seq_combo)
        lay.addLayout(seq_row)

        # Content + destination row
        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Contingut:"))
        self._pkg_excel = QCheckBox("Excels")
        self._pkg_excel.setChecked(True)
        opts_row.addWidget(self._pkg_excel)
        self._pkg_summary = QCheckBox("SUMMARY")
        self._pkg_summary.setChecked(True)
        opts_row.addWidget(self._pkg_summary)
        self._pkg_raw = QCheckBox("CSV RAW")
        opts_row.addWidget(self._pkg_raw)
        self._pkg_proc = QCheckBox("CSV PROC")
        opts_row.addWidget(self._pkg_proc)
        self._pkg_pdf = QCheckBox("PDF")
        opts_row.addWidget(self._pkg_pdf)

        opts_row.addSpacing(12)

        # Destination
        opts_row.addWidget(QLabel("Dest\u00ed:"))
        self._pkg_path_edit = QLineEdit()
        self._pkg_path_edit.setReadOnly(True)
        self._pkg_path_edit.setPlaceholderText("Carpeta de sortida...")
        self._pkg_path_edit.setMinimumWidth(200)
        opts_row.addWidget(self._pkg_path_edit)
        browse_btn = QPushButton("\U0001f4c1")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._browse_pkg_dest)
        opts_row.addWidget(browse_btn)
        self._pkg_zip = QCheckBox("ZIP")
        opts_row.addWidget(self._pkg_zip)

        opts_row.addSpacing(12)

        self._pkg_btn = QPushButton("Empaquetar")
        self._pkg_btn.setStyleSheet(
            f"background-color: {COLOR_PRIMARY}; color: white; "
            "font-weight: bold; padding: 4px 16px;")
        self._pkg_btn.clicked.connect(self._on_package)
        opts_row.addWidget(self._pkg_btn)

        opts_row.addStretch()

        lay.addLayout(opts_row)

        return grp

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _configure_table(self, table):
        """Apply consistent table styling."""
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f9f9f9;
                font-size: 11px;
            }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 4px;
                border: none;
                border-bottom: 2px solid #ddd;
                font-weight: bold;
                font-size: 10px;
            }
        """)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            self._scan_seqs()

    def set_current_seq(self, seq_path):
        """Called from wizard to pre-select last processed SEQ."""
        if not seq_path:
            return
        # If already initialized, update combo selection
        if self._initialized and self._all_seqs:
            self._select_seq_in_combo(seq_path)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_seqs(self):
        """Scan all data_folders for analyzed SEQs."""
        self._info_label.setText("Escanejant...")
        self._refresh_btn.setEnabled(False)

        self._scan_worker = _ScanWorker(self)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_finished(self, seqs):
        self._all_seqs = seqs
        self._refresh_btn.setEnabled(True)

        # Populate SEQ combo (for packaging)
        self._seq_combo.blockSignals(True)
        self._seq_combo.clear()
        mw_seq = self.main_window.seq_path

        for i, s in enumerate(seqs):
            label = f"{s['name']}  ({s['method']})"
            if (mw_seq and
                    os.path.normpath(s["path"]) == os.path.normpath(mw_seq)):
                label = f"\u2605 {label}"
            self._seq_combo.addItem(label, s["path"])
        self._seq_combo.blockSignals(False)

        # Pre-select current SEQ in combo
        if mw_seq:
            self._select_seq_in_combo(mw_seq)

        # Populate inventory filter
        self._inv_seq_filter.blockSignals(True)
        self._inv_seq_filter.clear()
        self._inv_seq_filter.addItem("Totes", None)
        for s in seqs:
            self._inv_seq_filter.addItem(s["name"], s["path"])
        self._inv_seq_filter.blockSignals(False)

        n = len(seqs)
        self._info_label.setText(
            f"{n} seq\u00fc\u00e8ncies amb an\u00e0lisi")

        # Build inventory
        self._build_and_show_inventory()

    def _select_seq_in_combo(self, seq_path):
        norm = os.path.normpath(seq_path)
        for i in range(self._seq_combo.count()):
            if os.path.normpath(self._seq_combo.itemData(i) or "") == norm:
                self._seq_combo.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------
    # Inventory (per mostra)
    # ------------------------------------------------------------------

    def _build_and_show_inventory(self):
        """Build global sample inventory from all cached data."""
        if not self._all_seqs:
            return

        # Load all JSONs that aren't cached yet
        for s in self._all_seqs:
            if s["path"] not in self._seq_cache:
                data = _load_analysis_json(s["path"])
                if data:
                    self._seq_cache[s["path"]] = data

        self._inventory = self._build_sample_inventory()
        self._populate_inventory_table()

    def _build_sample_inventory(self):
        """Build {sample_name: {col_seq, bp_seq, col_data, bp_data, ...}}."""
        inventory = {}
        for seq_info in self._all_seqs:
            data = self._seq_cache.get(seq_info["path"])
            if not data:
                continue
            sg = data.get("samples_grouped", {})
            method = seq_info.get("method", "COLUMN")
            is_bp = "BP" in method.upper() if method else False

            for name, sdata in sg.items():
                if sdata.get("sample_type", "SAMPLE") != "SAMPLE":
                    continue
                if name not in inventory:
                    inventory[name] = {}
                if is_bp:
                    inventory[name]["bp_seq"] = seq_info["name"]
                    inventory[name]["bp_path"] = seq_info["path"]
                    inventory[name]["bp_data"] = sdata
                else:
                    inventory[name]["col_seq"] = seq_info["name"]
                    inventory[name]["col_path"] = seq_info["path"]
                    inventory[name]["col_data"] = sdata
        return inventory

    def _populate_inventory_table(self):
        """Fill the inventory table with all samples."""
        table = self._inv_table
        table.setSortingEnabled(False)
        table.setRowCount(0)

        search = self._search_edit.text().strip().lower()
        seq_filter = self._inv_seq_filter.currentData()

        visible_count = 0
        for name in sorted(self._inventory.keys()):
            entry = self._inventory[name]

            # Search filter
            if search and search not in name.lower():
                continue

            # SEQ filter
            if seq_filter:
                col_path = entry.get("col_path")
                bp_path = entry.get("bp_path")
                if col_path != seq_filter and bp_path != seq_filter:
                    continue

            row = table.rowCount()
            table.insertRow(row)
            visible_count += 1

            # Col 0: Mostra
            item = QTableWidgetItem(name)
            item.setData(Qt.UserRole, name)
            table.setItem(row, 0, item)

            # Col 1-2: SEQ COL / BP
            table.setItem(row, 1,
                          QTableWidgetItem(entry.get("col_seq", "\u2014")))
            table.setItem(row, 2,
                          QTableWidgetItem(entry.get("bp_seq", "\u2014")))

            # Extract metrics
            col_m = (_extract_sample_metrics(entry["col_data"])
                     if entry.get("col_data") else {})
            bp_m = (_extract_sample_metrics(entry["bp_data"])
                    if entry.get("bp_data") else {})

            ppm_col = col_m.get("ppm")
            ppm_bp = bp_m.get("ppm")

            # Col 3-4: ppm COL / BP
            item = QTableWidgetItem(_format_ppm(ppm_col))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 3, item)
            item = QTableWidgetItem(_format_ppm(ppm_bp))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 4, item)

            # Col 5: Ratio ppm
            if ppm_col and ppm_bp and ppm_col > 0:
                ratio = ppm_bp / ppm_col
                ratio_item = QTableWidgetItem(f"{ratio:.3f}")
                if abs(ratio - 1.0) > 0.15:
                    ratio_item.setForeground(
                        QBrush(QColor(COLOR_WARNING)))
                ratio_item.setTextAlignment(
                    Qt.AlignRight | Qt.AlignVCenter)
            else:
                ratio_item = QTableWidgetItem("\u2014")
                ratio_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 5, ratio_item)

            # Col 6-7: A254 COL / BP
            a254_col = col_m.get("a254")
            a254_bp = bp_m.get("a254")
            item = QTableWidgetItem(_format_area(a254_col))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 6, item)
            item = QTableWidgetItem(_format_area(a254_bp))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 7, item)

            # Col 8: Ratio A254
            if a254_col and a254_bp and a254_col > 0:
                r = a254_bp / a254_col
                ri = QTableWidgetItem(f"{r:.3f}")
                ri.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            else:
                ri = QTableWidgetItem("\u2014")
                ri.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 8, ri)

        table.setSortingEnabled(True)
        self._inv_count_label.setText(f"{visible_count} mostres")

    def _filter_inventory_table(self):
        """Re-filter the inventory table based on search/seq filter."""
        if self._inventory:
            self._populate_inventory_table()

    # ------------------------------------------------------------------
    # Packaging
    # ------------------------------------------------------------------

    def _browse_pkg_dest(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Selecciona carpeta de sortida")
        if folder:
            self._pkg_path_edit.setText(folder)

    def _on_package(self):
        """Package the currently selected SEQ using export_sequence."""
        idx = self._seq_combo.currentIndex()
        if idx < 0 or idx >= len(self._all_seqs):
            QMessageBox.warning(
                self, "Av\u00eds",
                "Selecciona una seq\u00fc\u00e8ncia primer.")
            return

        seq_info = self._all_seqs[idx]
        data = self._seq_cache.get(seq_info["path"])
        if not data:
            # Try loading it
            data = _load_analysis_json(seq_info["path"])
            if data:
                self._seq_cache[seq_info["path"]] = data

        if not data:
            QMessageBox.warning(
                self, "Av\u00eds",
                "Les dades de la seq\u00fc\u00e8ncia no estan carregades.")
            return

        sg = data.get("samples_grouped", {})
        if not sg:
            QMessageBox.warning(
                self, "Av\u00eds", "No hi ha mostres per empaquetar.")
            return

        # Determine output dir
        output_dir = self._pkg_path_edit.text().strip()
        if not output_dir:
            output_dir = os.path.join(seq_info["path"], "RESULTATS")

        method = seq_info.get("method", "COLUMN")

        try:
            from hpsec_export import export_sequence

            result = export_sequence(
                samples_grouped=sg,
                output_dir=output_dir,
                mode=method,
                seq_path=seq_info["path"],
                export_raw=self._pkg_raw.isChecked(),
                export_processed=self._pkg_proc.isChecked(),
            )

            n = result.get("n_exported", 0)
            errs = result.get("n_errors", 0)
            msg = f"Exportades {n} mostres a:\n{output_dir}"
            if errs > 0:
                msg += f"\n({errs} errors)"

            QMessageBox.information(self, "Empaquetat", msg)

        except Exception as e:
            logger.error("Export error: %s", e)
            QMessageBox.critical(
                self, "Error",
                f"Error durant l'exportaci\u00f3:\n{e}")
