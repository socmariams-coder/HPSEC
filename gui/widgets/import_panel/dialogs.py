"""
HPSEC Suite - Import Panel Dialogs
==================================

Diàlegs per al panel d'importació.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QDialogButtonBox
)


class OrphanFilesDialog(QDialog):
    """Diàleg per mostrar fitxers orfes."""

    def __init__(self, parent, orphan_files):
        super().__init__(parent)
        self.setWindowTitle("Fitxers Orfes - Revisar Noms")
        self.setMinimumSize(500, 350)

        layout = QVBoxLayout(self)

        info = QLabel(
            "<b>Fitxers no assignats automàticament:</b><br>"
            "Els noms no coincideixen amb el MasterFile. Opcions:<br>"
            "• <b>Revisar noms i reimportar</b>: Verificar que la seqüència del cromatògraf coincideixi amb el MasterFile, després tornar a importar<br>"
            "• <b>Assignar manualment</b>: Doble-clic a la cel·la '-' de la mostra corresponent i seleccionar el fitxer"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        text = QTextEdit()
        text.setReadOnly(True)

        content = ""
        uib_files = orphan_files.get("uib", [])
        dad_files = orphan_files.get("dad", [])

        if uib_files:
            content += "=== UIB (DOC) ===\n"
            for item in uib_files:
                # Suporta tant format antic (string) com nou (dict amb info)
                if isinstance(item, dict):
                    fname = Path(item.get("file", "")).name
                    n_pts = item.get("n_points", 0)
                    status = "BUIT" if n_pts == 0 else f"{n_pts} pts"
                    content += f"  • {fname}  [{status}]\n"
                else:
                    content += f"  • {Path(item).name}\n"
            content += "\n"

        if dad_files:
            content += "=== DAD (254nm) ===\n"
            for item in dad_files:
                if isinstance(item, dict):
                    fname = Path(item.get("file", "")).name
                    n_pts = item.get("n_points", 0)
                    status = "BUIT" if n_pts == 0 else f"{n_pts} pts"
                    content += f"  • {fname}  [{status}]\n"
                else:
                    content += f"  • {Path(item).name}\n"

        text.setText(content or "Cap fitxer orfe.")
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)


class ChromatogramPreviewDialog(QDialog):
    """Diàleg per mostrar preview del cromatograma."""

    def __init__(self, parent, sample_name, replica, sample_data, imported_data):
        super().__init__(parent)
        self.setWindowTitle(f"Preview: {sample_name} (Rep {replica})")
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout(self)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvasQTAgg(fig)

            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            self._plot_data(ax1, ax2, sample_name, replica, imported_data)

            ax1.set_xlabel("Temps (min)")
            ax1.set_ylabel("DOC (mAU)", color="#2E86AB")
            ax2.set_ylabel("DAD 254nm (mAU)", color="#E67E22")
            ax1.set_title(f"{sample_name} - Rèplica {replica}")
            # Només mostrar llegenda si hi ha dades
            if ax1.get_legend_handles_labels()[0]:
                ax1.legend(loc="upper left")
            if ax2.get_legend_handles_labels()[0]:
                ax2.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            fig.tight_layout()
            layout.addWidget(canvas)

        except ImportError:
            info = QLabel(f"""
            <h3>{sample_name} - Rèplica {replica}</h3>
            <p><i>Instal·la matplotlib per veure el gràfic:</i></p>
            <code>pip install matplotlib</code>
            """)
            info.setWordWrap(True)
            layout.addWidget(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _plot_data(self, ax1, ax2, sample_name, replica, imported_data):
        samples = imported_data.get("samples", {})
        sample_info = samples.get(sample_name)
        if not sample_info:
            ax1.text(0.5, 0.5, "Dades no disponibles", ha='center', va='center')
            return

        rep_data = sample_info.get("replicas", {}).get(str(replica))
        if not rep_data:
            ax1.text(0.5, 0.5, f"Rèplica {replica} no trobada", ha='center', va='center')
            return

        direct = rep_data.get("direct", {})
        if direct and direct.get("t") is not None:
            t = direct["t"]
            y = direct.get("y") if direct.get("y") is not None else direct.get("y_raw")
            if y is not None:
                ax1.plot(t, y, color="#2E86AB", label="DOC Direct", linewidth=1)

        uib = rep_data.get("uib", {})
        if uib and uib.get("t") is not None:
            t = uib["t"]
            y = uib.get("y") if uib.get("y") is not None else uib.get("y_raw")
            if y is not None and len(t) > 0 and len(y) > 0:
                ax1.plot(t, y, color="#27AE60", label="DOC UIB", linewidth=1, alpha=0.8)

        dad = rep_data.get("dad", {})
        if dad:
            # Intentar obtenir dades del DAD (pot ser df o arrays separats)
            t_dad = None
            y254 = None

            # Format 1: DataFrame amb columnes "time (min)" i "254"
            df_dad = dad.get("df")
            try:
                import pandas as pd
                if df_dad is not None and isinstance(df_dad, pd.DataFrame) and len(df_dad) > 0:
                    if "time (min)" in df_dad.columns:
                        t_dad = df_dad["time (min)"].values
                    # Buscar columna 254nm (pot ser "254", "254nm", 254, o similar)
                    for col in df_dad.columns:
                        col_str = str(col)
                        if "254" in col_str or col_str == "254":
                            y254 = df_dad[col].values
                            break
            except Exception as e:
                print(f"Warning: Error llegint DataFrame DAD: {e}")

            # Format 2: Arrays separats (t, wavelengths dict)
            if t_dad is None and dad.get("t") is not None:
                t_dad = dad["t"]
                wavelengths = dad.get("wavelengths", {})
                if 254 in wavelengths or "254" in wavelengths:
                    y254 = wavelengths.get(254) or wavelengths.get("254")

            # Plotar si tenim dades
            if t_dad is not None and y254 is not None:
                ax2.plot(t_dad, y254, color="#E67E22", label="DAD 254nm",
                            linewidth=1, linestyle="--", alpha=0.7)
