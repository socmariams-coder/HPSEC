"""
HPSEC Suite - Review Panel
===========================

Panel per la fase 4: Revisió i selecció de rèpliques.

Funcionalitats:
- Taula amb totes les mostres i recomanacions
- Selecció independent de rèplica per DOC i DAD
- Visualització de warnings i anomalies
- Diàleg de detall amb gràfics comparatius R1 vs R2
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QDialog, QGroupBox, QGridLayout, QSplitter, QFrame,
    QAbstractItemView, QStyle, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QBrush, QIcon

import numpy as np

# Intentar importar matplotlib per gràfics
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ReviewPanel(QWidget):
    """Panel de revisió de rèpliques."""

    review_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples_grouped = {}
        self.current_sample = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Títol
        header_layout = QHBoxLayout()
        title = QLabel("Revisió de Rèpliques")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        # Llegenda
        legend = QLabel("● Recomanat  ○ Canviat  ⚠ Warning")
        legend.setStyleSheet("color: #666;")
        header_layout.addWidget(legend)
        layout.addLayout(header_layout)

        # Info
        info = QLabel(
            "Revisa les recomanacions automàtiques. Pots canviar la selecció "
            "de rèplica per DOC i DAD de forma independent."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; margin-bottom: 8px;")
        layout.addWidget(info)

        # Taula principal
        self.review_table = QTableWidget()
        self.review_table.setColumnCount(8)
        self.review_table.setHorizontalHeaderLabels([
            "Mostra", "DOC", "DAD", "[DOC] ppm", "SNR", "Status", "Warnings", ""
        ])

        # Configurar columnes
        header = self.review_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Mostra
        header.setSectionResizeMode(1, QHeaderView.Fixed)    # DOC
        header.setSectionResizeMode(2, QHeaderView.Fixed)    # DAD
        header.setSectionResizeMode(3, QHeaderView.Fixed)    # ppm
        header.setSectionResizeMode(4, QHeaderView.Fixed)    # SNR
        header.setSectionResizeMode(5, QHeaderView.Fixed)    # Status
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # Warnings
        header.setSectionResizeMode(7, QHeaderView.Fixed)    # Accions

        self.review_table.setColumnWidth(1, 70)
        self.review_table.setColumnWidth(2, 70)
        self.review_table.setColumnWidth(3, 80)
        self.review_table.setColumnWidth(4, 80)
        self.review_table.setColumnWidth(5, 60)
        self.review_table.setColumnWidth(7, 80)

        self.review_table.setAlternatingRowColors(True)
        self.review_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.review_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.review_table.verticalHeader().setVisible(False)

        # Doble clic per obrir detall
        self.review_table.cellDoubleClicked.connect(self._on_row_double_click)

        layout.addWidget(self.review_table)

        # Resum
        self.summary_frame = QFrame()
        self.summary_frame.setFrameStyle(QFrame.StyledPanel)
        self.summary_frame.setStyleSheet("background-color: #f5f5f5; padding: 8px;")
        summary_layout = QHBoxLayout(self.summary_frame)

        self.summary_label = QLabel()
        self.summary_label.setFont(QFont("Segoe UI", 10))
        summary_layout.addWidget(self.summary_label)

        summary_layout.addStretch()

        # Botons de selecció massiva
        self.all_recommended_btn = QPushButton("Tot Recomanat")
        self.all_recommended_btn.clicked.connect(self._set_all_recommended)
        summary_layout.addWidget(self.all_recommended_btn)

        layout.addWidget(self.summary_frame)

        # Botons de navegació
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.refresh_btn = QPushButton("Actualitzar")
        self.refresh_btn.clicked.connect(self._refresh_data)
        buttons_layout.addWidget(self.refresh_btn)

        self.next_btn = QPushButton("Següent: Exportar →")
        self.next_btn.setStyleSheet("font-weight: bold;")
        self.next_btn.clicked.connect(self._go_next)
        buttons_layout.addWidget(self.next_btn)

        layout.addLayout(buttons_layout)

    def showEvent(self, event):
        """Es crida quan el panel es fa visible."""
        super().showEvent(event)
        self._refresh_data()

    def _refresh_data(self):
        """Actualitza les dades del panel."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            self.summary_label.setText("No hi ha dades processades")
            return

        # Obtenir samples_grouped (nova estructura)
        self.samples_grouped = processed_data.get("samples_grouped", {})

        if not self.samples_grouped:
            # Fallback: crear estructura des de samples
            self._create_grouped_from_samples(processed_data)

        # Omplir taula
        self._populate_table()

        # Habilitar exportar
        self.main_window.enable_tab(5)

    def _create_grouped_from_samples(self, processed_data):
        """Crea samples_grouped si no existeix (compatibilitat)."""
        samples_by_name = {}
        all_samples = processed_data.get("samples", [])

        for s in all_samples:
            name = s.get("name", "UNKNOWN")
            replica = s.get("replica", "1")
            if name not in samples_by_name:
                samples_by_name[name] = {"replicas": {}}
            samples_by_name[name]["replicas"][replica] = s

        # Afegir camps mínims
        for name, data in samples_by_name.items():
            replicas = data["replicas"]
            first_rep = list(replicas.keys())[0] if replicas else "1"
            data["selected"] = {"doc": first_rep, "dad": first_rep}
            data["recommendation"] = {
                "doc": {"replica": first_rep, "score": 0.5, "reason": "Única disponible"},
                "dad": {"replica": first_rep, "score": 0.5, "reason": "Única disponible"}
            }
            data["comparison"] = None
            data["quantification"] = None

        self.samples_grouped = samples_by_name

    def _populate_table(self):
        """Omple la taula amb les dades."""
        self.review_table.setRowCount(0)

        n_warnings = 0
        n_manual = 0

        for sample_name in sorted(self.samples_grouped.keys()):
            sample_data = self.samples_grouped[sample_name]
            row = self.review_table.rowCount()
            self.review_table.insertRow(row)

            replicas = sample_data.get("replicas", {})
            recommendation = sample_data.get("recommendation", {})
            selected = sample_data.get("selected", {"doc": "1", "dad": "1"})
            comparison = sample_data.get("comparison", {})
            quantification = sample_data.get("quantification", {})

            # Col 0: Mostra
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)  # Guardar nom per referència
            self.review_table.setItem(row, 0, item_name)

            # Col 1: DOC replica selector
            doc_combo = QComboBox()
            doc_rec = recommendation.get("doc", {}).get("replica", "1")
            doc_sel = selected.get("doc", doc_rec)

            for rep_num in sorted(replicas.keys()):
                prefix = "● " if rep_num == doc_rec else "○ "
                doc_combo.addItem(f"{prefix}R{rep_num}", rep_num)
                if rep_num == doc_sel:
                    doc_combo.setCurrentIndex(doc_combo.count() - 1)

            doc_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_selection_changed(name, "doc")
            )
            self.review_table.setCellWidget(row, 1, doc_combo)

            # Col 2: DAD replica selector
            dad_combo = QComboBox()
            dad_rec = recommendation.get("dad", {}).get("replica", "1")
            dad_sel = selected.get("dad", dad_rec)

            for rep_num in sorted(replicas.keys()):
                prefix = "● " if rep_num == dad_rec else "○ "
                dad_combo.addItem(f"{prefix}R{rep_num}", rep_num)
                if rep_num == dad_sel:
                    dad_combo.setCurrentIndex(dad_combo.count() - 1)

            dad_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_selection_changed(name, "dad")
            )
            self.review_table.setCellWidget(row, 2, dad_combo)

            # Col 3: Concentració ppm
            conc = quantification.get("concentration_ppm") if quantification else None
            conc_text = f"{conc:.2f}" if conc is not None else "-"
            self.review_table.setItem(row, 3, QTableWidgetItem(conc_text))

            # Col 4: SNR (DOC/DAD)
            doc_selected_data = replicas.get(doc_sel, {})
            dad_selected_data = replicas.get(dad_sel, {})

            snr_doc = doc_selected_data.get("snr_info", {}).get("snr_direct", 0)
            snr_dad = dad_selected_data.get("snr_info_dad", {}).get("A254", {}).get("snr", 0)

            snr_doc_str = f"{snr_doc:.0f}" if snr_doc else "-"
            snr_dad_str = f"{snr_dad:.0f}" if snr_dad else "-"
            self.review_table.setItem(row, 4, QTableWidgetItem(f"{snr_doc_str}/{snr_dad_str}"))

            # Col 5: Status
            has_warnings = False
            if comparison:
                doc_warnings = comparison.get("doc", {}).get("warnings", [])
                dad_warnings = comparison.get("dad", {}).get("warnings", [])
                has_warnings = bool(doc_warnings or dad_warnings)

            is_manual = (doc_sel != doc_rec) or (dad_sel != dad_rec)

            if has_warnings:
                status_item = QTableWidgetItem("⚠")
                status_item.setForeground(QBrush(QColor("#f0ad4e")))
                n_warnings += 1
            elif is_manual:
                status_item = QTableWidgetItem("○")
                status_item.setForeground(QBrush(QColor("#5bc0de")))
                n_manual += 1
            else:
                status_item = QTableWidgetItem("✓")
                status_item.setForeground(QBrush(QColor("#5cb85c")))

            status_item.setTextAlignment(Qt.AlignCenter)
            self.review_table.setItem(row, 5, status_item)

            # Col 6: Warnings
            all_warnings = []
            if comparison:
                all_warnings.extend(comparison.get("doc", {}).get("warnings", []))
                all_warnings.extend(comparison.get("dad", {}).get("warnings", []))

            warnings_text = ", ".join(all_warnings[:3]) if all_warnings else "-"
            if len(all_warnings) > 3:
                warnings_text += f" (+{len(all_warnings)-3})"
            self.review_table.setItem(row, 6, QTableWidgetItem(warnings_text))

            # Col 7: Botó detall
            detail_btn = QPushButton("Detall")
            detail_btn.setFixedWidth(70)
            detail_btn.clicked.connect(lambda checked, name=sample_name: self._show_detail(name))
            self.review_table.setCellWidget(row, 7, detail_btn)

        # Actualitzar resum
        total = len(self.samples_grouped)
        ok = total - n_warnings - n_manual
        self.summary_label.setText(
            f"Total: {total} mostres | ✓ OK: {ok} | ⚠ Warnings: {n_warnings} | ○ Manual: {n_manual}"
        )

    def _on_selection_changed(self, sample_name, signal_type):
        """Gestiona canvi de selecció de rèplica."""
        if sample_name not in self.samples_grouped:
            return

        # Obtenir el combo corresponent
        for row in range(self.review_table.rowCount()):
            item = self.review_table.item(row, 0)
            if item and item.data(Qt.UserRole) == sample_name:
                col = 1 if signal_type == "doc" else 2
                combo = self.review_table.cellWidget(row, col)
                if combo:
                    new_replica = combo.currentData()
                    self.samples_grouped[sample_name]["selected"][signal_type] = new_replica

                    # Recalcular quantificació si canvia DOC
                    if signal_type == "doc":
                        self._recalculate_quantification(sample_name)

                    # Actualitzar SNR i status
                    self._update_row(row, sample_name)
                break

    def _recalculate_quantification(self, sample_name):
        """Recalcula la quantificació quan canvia la rèplica DOC."""
        from hpsec_analyze import quantify_sample

        sample_data = self.samples_grouped[sample_name]
        selected_doc = sample_data["selected"]["doc"]
        selected_replica = sample_data["replicas"].get(selected_doc)

        if selected_replica:
            calibration_data = self.main_window.calibration_data
            method = self.main_window.processed_data.get("method", "COLUMN")
            mode = "BP" if method == "BP" else "COLUMN"

            quantification = quantify_sample(selected_replica, calibration_data, mode=mode)
            sample_data["quantification"] = quantification

    def _update_row(self, row, sample_name):
        """Actualitza una fila després de canvi de selecció."""
        sample_data = self.samples_grouped[sample_name]
        replicas = sample_data.get("replicas", {})
        selected = sample_data.get("selected", {})
        recommendation = sample_data.get("recommendation", {})
        comparison = sample_data.get("comparison", {})
        quantification = sample_data.get("quantification", {})

        doc_sel = selected.get("doc", "1")
        dad_sel = selected.get("dad", "1")
        doc_rec = recommendation.get("doc", {}).get("replica", "1")
        dad_rec = recommendation.get("dad", {}).get("replica", "1")

        # Actualitzar concentració
        conc = quantification.get("concentration_ppm") if quantification else None
        conc_text = f"{conc:.2f}" if conc is not None else "-"
        self.review_table.item(row, 3).setText(conc_text)

        # Actualitzar SNR
        doc_selected_data = replicas.get(doc_sel, {})
        dad_selected_data = replicas.get(dad_sel, {})

        snr_doc = doc_selected_data.get("snr_info", {}).get("snr_direct", 0)
        snr_dad = dad_selected_data.get("snr_info_dad", {}).get("A254", {}).get("snr", 0)

        snr_doc_str = f"{snr_doc:.0f}" if snr_doc else "-"
        snr_dad_str = f"{snr_dad:.0f}" if snr_dad else "-"
        self.review_table.item(row, 4).setText(f"{snr_doc_str}/{snr_dad_str}")

        # Actualitzar status
        has_warnings = False
        if comparison:
            doc_warnings = comparison.get("doc", {}).get("warnings", [])
            dad_warnings = comparison.get("dad", {}).get("warnings", [])
            has_warnings = bool(doc_warnings or dad_warnings)

        is_manual = (doc_sel != doc_rec) or (dad_sel != dad_rec)

        status_item = self.review_table.item(row, 5)
        if has_warnings:
            status_item.setText("⚠")
            status_item.setForeground(QBrush(QColor("#f0ad4e")))
        elif is_manual:
            status_item.setText("○")
            status_item.setForeground(QBrush(QColor("#5bc0de")))
        else:
            status_item.setText("✓")
            status_item.setForeground(QBrush(QColor("#5cb85c")))

    def _set_all_recommended(self):
        """Posa totes les seleccions a la recomanació."""
        for sample_name, sample_data in self.samples_grouped.items():
            recommendation = sample_data.get("recommendation", {})
            sample_data["selected"] = {
                "doc": recommendation.get("doc", {}).get("replica", "1"),
                "dad": recommendation.get("dad", {}).get("replica", "1")
            }

        self._populate_table()

    def _on_row_double_click(self, row, col):
        """Obre el detall en fer doble clic."""
        item = self.review_table.item(row, 0)
        if item:
            sample_name = item.data(Qt.UserRole)
            if sample_name:
                self._show_detail(sample_name)

    def _show_detail(self, sample_name):
        """Mostra el diàleg de detall per una mostra."""
        if sample_name not in self.samples_grouped:
            return

        dialog = DetailDialog(
            sample_name,
            self.samples_grouped[sample_name],
            self.main_window.processed_data.get("method", "COLUMN"),
            parent=self
        )

        if dialog.exec() == QDialog.Accepted:
            # Actualitzar selecció des del diàleg
            new_selection = dialog.get_selection()
            self.samples_grouped[sample_name]["selected"] = new_selection
            self._recalculate_quantification(sample_name)
            self._populate_table()

    def _go_next(self):
        """Guarda seleccions i navega al següent tab."""
        # Actualitzar processed_data amb samples_grouped actualitzat
        if self.main_window.processed_data:
            self.main_window.processed_data["samples_grouped"] = self.samples_grouped

        # Marcar revisió com a completada
        self.main_window.review_data = {
            name: data["selected"] for name, data in self.samples_grouped.items()
        }
        self.main_window.mark_review_completed()

        self.main_window.go_to_tab(5)  # Exportar

    def get_final_selections(self):
        """Retorna les seleccions finals per exportar."""
        return {
            name: {
                "doc_replica": data["selected"]["doc"],
                "dad_replica": data["selected"]["dad"],
                "quantification": data.get("quantification", {})
            }
            for name, data in self.samples_grouped.items()
        }


class DetailDialog(QDialog):
    """Diàleg de detall amb gràfics comparatius R1 vs R2."""

    def __init__(self, sample_name, sample_data, method, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.method = method
        self.is_bp = method.upper() == "BP"

        self.setWindowTitle(f"Detall: {sample_name}")
        self.setMinimumSize(900, 700)

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície del diàleg."""
        layout = QVBoxLayout(self)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # Panel esquerre: Gràfics
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(8, 10), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            graph_layout.addWidget(self.canvas)
            self._plot_comparison()
        else:
            no_plot_label = QLabel("Matplotlib no disponible.\nInstal·la matplotlib per veure gràfics.")
            no_plot_label.setAlignment(Qt.AlignCenter)
            graph_layout.addWidget(no_plot_label)

        splitter.addWidget(graph_widget)

        # Panel dret: Info i selecció
        info_widget = QWidget()
        info_widget.setMaximumWidth(300)
        info_layout = QVBoxLayout(info_widget)

        # Comparació
        comparison = self.sample_data.get("comparison", {})
        if comparison:
            comp_group = QGroupBox("Comparació R1 vs R2")
            comp_layout = QGridLayout(comp_group)

            # DOC
            comp_layout.addWidget(QLabel("<b>DOC</b>"), 0, 0, 1, 2)

            doc_comp = comparison.get("doc", {})
            pearson_doc = doc_comp.get("pearson", 0)
            area_diff_doc = doc_comp.get("area_diff_pct", 0)

            comp_layout.addWidget(QLabel("Pearson:"), 1, 0)
            pearson_label = QLabel(f"{pearson_doc:.4f}")
            if pearson_doc < 0.995:
                pearson_label.setStyleSheet("color: orange; font-weight: bold;")
            comp_layout.addWidget(pearson_label, 1, 1)

            comp_layout.addWidget(QLabel("Diff Àrea:"), 2, 0)
            diff_label = QLabel(f"{area_diff_doc:.1f}%")
            if area_diff_doc > 10:
                diff_label.setStyleSheet("color: orange; font-weight: bold;")
            comp_layout.addWidget(diff_label, 2, 1)

            # Fraccions (COLUMN)
            if not self.is_bp:
                frac_diff = doc_comp.get("fraction_diff_pct", {})
                row = 3
                for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                    diff = frac_diff.get(frac, 0)
                    comp_layout.addWidget(QLabel(f"  {frac}:"), row, 0)
                    frac_label = QLabel(f"{diff:.1f}%")
                    if diff > 15:
                        frac_label.setStyleSheet("color: orange;")
                    comp_layout.addWidget(frac_label, row, 1)
                    row += 1

            # Warnings
            warnings = doc_comp.get("warnings", [])
            if warnings:
                comp_layout.addWidget(QLabel("<b>Warnings:</b>"), row, 0, 1, 2)
                row += 1
                for w in warnings[:5]:
                    comp_layout.addWidget(QLabel(f"  ⚠ {w}"), row, 0, 1, 2)
                    row += 1

            info_layout.addWidget(comp_group)

        # Recomanació
        rec_group = QGroupBox("Recomanació")
        rec_layout = QGridLayout(rec_group)

        recommendation = self.sample_data.get("recommendation", {})

        doc_rec = recommendation.get("doc", {})
        rec_layout.addWidget(QLabel("DOC:"), 0, 0)
        rec_layout.addWidget(QLabel(f"R{doc_rec.get('replica', '?')} ({doc_rec.get('reason', '-')})"), 0, 1)

        dad_rec = recommendation.get("dad", {})
        rec_layout.addWidget(QLabel("DAD:"), 1, 0)
        rec_layout.addWidget(QLabel(f"R{dad_rec.get('replica', '?')} ({dad_rec.get('reason', '-')})"), 1, 1)

        info_layout.addWidget(rec_group)

        # Selecció
        sel_group = QGroupBox("Selecció")
        sel_layout = QGridLayout(sel_group)

        selected = self.sample_data.get("selected", {"doc": "1", "dad": "1"})
        replicas = self.sample_data.get("replicas", {})

        sel_layout.addWidget(QLabel("DOC:"), 0, 0)
        self.doc_combo = QComboBox()
        for rep in sorted(replicas.keys()):
            self.doc_combo.addItem(f"R{rep}", rep)
            if rep == selected.get("doc"):
                self.doc_combo.setCurrentIndex(self.doc_combo.count() - 1)
        sel_layout.addWidget(self.doc_combo, 0, 1)

        sel_layout.addWidget(QLabel("DAD:"), 1, 0)
        self.dad_combo = QComboBox()
        for rep in sorted(replicas.keys()):
            self.dad_combo.addItem(f"R{rep}", rep)
            if rep == selected.get("dad"):
                self.dad_combo.setCurrentIndex(self.dad_combo.count() - 1)
        sel_layout.addWidget(self.dad_combo, 1, 1)

        info_layout.addWidget(sel_group)

        info_layout.addStretch()

        splitter.addWidget(info_widget)
        splitter.setSizes([600, 300])

        layout.addWidget(splitter)

        # Botons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel·lar")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Aplicar")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def _plot_comparison(self):
        """Genera els gràfics comparatius."""
        if not HAS_MATPLOTLIB:
            return

        replicas = self.sample_data.get("replicas", {})
        if len(replicas) < 1:
            return

        rep_keys = sorted(replicas.keys())
        r1_data = replicas.get(rep_keys[0], {})
        r2_data = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None

        self.figure.clear()

        # Crear subplots
        n_plots = 3  # DOC Direct, DOC UIB, DAD 254
        axes = self.figure.subplots(n_plots, 1, sharex=True)

        colors = {'r1': '#2196F3', 'r2': '#FF5722'}  # Blau, Taronja

        # Plot 1: DOC Direct
        ax1 = axes[0]
        t1 = r1_data.get("t_doc")
        y1 = r1_data.get("y_doc_net")

        if t1 is not None and y1 is not None:
            ax1.plot(t1, y1, color=colors['r1'], label='R1', linewidth=1)

            if r2_data:
                t2 = r2_data.get("t_doc")
                y2 = r2_data.get("y_doc_net")
                if t2 is not None and y2 is not None:
                    ax1.plot(t2, y2, color=colors['r2'], label='R2', linewidth=1, linestyle='--')

        ax1.set_ylabel("DOC Direct (mAU)")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: DOC UIB (si disponible)
        ax2 = axes[1]
        y1_uib = r1_data.get("y_doc_uib_net")

        if y1_uib is not None and t1 is not None:
            ax2.plot(t1, y1_uib, color=colors['r1'], label='R1', linewidth=1)

            if r2_data:
                y2_uib = r2_data.get("y_doc_uib_net")
                t2 = r2_data.get("t_doc")
                if y2_uib is not None and t2 is not None:
                    ax2.plot(t2, y2_uib, color=colors['r2'], label='R2', linewidth=1, linestyle='--')

            ax2.set_ylabel("DOC UIB (mAU)")
        else:
            ax2.text(0.5, 0.5, "UIB no disponible", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_ylabel("DOC UIB (mAU)")

        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Plot 3: DAD 254
        ax3 = axes[2]
        df_dad1 = r1_data.get("df_dad")

        if df_dad1 is not None and not df_dad1.empty and '254' in df_dad1.columns:
            t_dad1 = df_dad1['time (min)'].to_numpy()
            y_254_1 = df_dad1['254'].to_numpy()
            ax3.plot(t_dad1, y_254_1, color=colors['r1'], label='R1', linewidth=1)

            if r2_data:
                df_dad2 = r2_data.get("df_dad")
                if df_dad2 is not None and not df_dad2.empty and '254' in df_dad2.columns:
                    t_dad2 = df_dad2['time (min)'].to_numpy()
                    y_254_2 = df_dad2['254'].to_numpy()
                    ax3.plot(t_dad2, y_254_2, color=colors['r2'], label='R2', linewidth=1, linestyle='--')

        ax3.set_ylabel("DAD 254nm (mAU)")
        ax3.set_xlabel("Temps (min)")
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Afegir zones (COLUMN)
        if not self.is_bp:
            zones = [
                (0, 18, "BioP", "#e3f2fd"),
                (18, 23, "HS", "#fff3e0"),
                (23, 30, "BB", "#f3e5f5"),
                (30, 40, "SB", "#e8f5e9"),
                (40, 70, "LMW", "#fce4ec"),
            ]
            for ax in axes:
                for start, end, name, color in zones:
                    ax.axvspan(start, end, alpha=0.2, color=color)

        self.figure.tight_layout()
        self.canvas.draw()

    def get_selection(self):
        """Retorna la selecció actual."""
        return {
            "doc": self.doc_combo.currentData(),
            "dad": self.dad_combo.currentData()
        }
