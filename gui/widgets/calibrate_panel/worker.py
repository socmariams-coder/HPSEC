"""
HPSEC Suite - Calibrate Worker
==============================

Thread worker per calibració asíncrona.
"""

import os
import logging

from PySide6.QtCore import Signal, QThread

from hpsec_calibrate import calibrate_from_import

logger = logging.getLogger(__name__)


class CalibrateWorker(QThread):
    """Worker thread para calibración asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, imported_data, config=None):
        super().__init__()
        self.imported_data = imported_data
        self.config = config

    def run(self):
        try:
            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            # Assegurar dades carregades (pot venir amb data_deferred=True
            # des del preload de metadades). FER-HO AL THREAD per no
            # bloquejar la UI (llegir MasterFile + CSV + Export3D és lent).
            if self.imported_data and self.imported_data.get("data_deferred"):
                from hpsec_import import ensure_data_loaded
                progress_cb(2, "Carregant senyals des del disc...")
                ensure_data_loaded(
                    self.imported_data,
                    config=self.config,
                    progress_callback=lambda pct, msg: progress_cb(
                        2 + int(pct * 0.15), msg  # 2-17% del total
                    ),
                )

            result = calibrate_from_import(
                self.imported_data,
                config=self.config,
                progress_callback=progress_cb
            )
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class SiblingCalibrateWorker(QThread):
    """Worker per calibrar N siblings seqüencialment.

    Cada sibling es calibra independentment amb calibrate_from_import().
    """
    progress = Signal(int, str)
    sibling_finished = Signal(str, dict)  # path, result
    all_finished = Signal(dict)            # {path: result}
    error = Signal(str)

    def __init__(self, sibling_imported, config=None):
        """
        Args:
            sibling_imported: dict {path: imported_data} per cada sibling
            config: config opcional
        """
        super().__init__()
        self.sibling_imported = sibling_imported
        self.config = config

    def run(self):
        results = {}
        paths = list(self.sibling_imported.keys())
        n = len(paths)

        try:
            for i, path in enumerate(paths):
                name = os.path.basename(path)
                imported = self.sibling_imported[path]
                base_pct = int(i / n * 100)

                def progress_cb(pct, msg, _base=base_pct, _n=n, _name=name):
                    global_pct = _base + int(pct / _n)
                    self.progress.emit(global_pct, f"[{_name}] {msg}")

                # ensure_data_loaded si cal
                if imported and imported.get("data_deferred"):
                    from hpsec_import import ensure_data_loaded
                    progress_cb(2, "Carregant senyals...")
                    ensure_data_loaded(
                        imported,
                        config=self.config,
                        progress_callback=lambda pct, msg, _cb=progress_cb: _cb(
                            2 + int(pct * 0.15), msg
                        ),
                    )

                try:
                    result = calibrate_from_import(
                        imported,
                        config=self.config,
                        progress_callback=progress_cb
                    )
                except Exception as e:
                    # Error en un sibling no para els altres
                    logger.warning("Error calibrant %s: %s", name, e)
                    result = {
                        "success": False,
                        "errors": [str(e)],
                        "warnings_structured": [],
                    }

                results[path] = result
                self.sibling_finished.emit(path, result)

            self.progress.emit(100, f"Verificació completada ({n} carpetes)")
            self.all_finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
