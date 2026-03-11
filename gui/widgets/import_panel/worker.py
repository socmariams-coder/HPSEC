"""
HPSEC Suite - Import Worker
===========================

Thread worker per importació asíncrona.
"""

import os
import logging

from PySide6.QtCore import Signal, QThread

from hpsec_import import import_sequence, import_from_manifest, load_manifest, save_import_manifest

logger = logging.getLogger(__name__)


class ImportWorker(QThread):
    """Worker thread per importació asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, use_manifest=False, manifest=None, load_data=True):
        super().__init__()
        self.seq_path = seq_path
        self.use_manifest = use_manifest
        self.manifest = manifest
        self.load_data = load_data

    def run(self):
        try:
            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            if self.use_manifest and self.manifest:
                result = import_from_manifest(
                    self.seq_path,
                    manifest=self.manifest,
                    progress_callback=progress_cb,
                    load_data=self.load_data
                )
            else:
                result = import_sequence(
                    self.seq_path,
                    progress_callback=progress_cb
                )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class SiblingImportWorker(QThread):
    """Worker thread per importar N siblings seqüencialment.

    Cada sibling s'importa de forma independent (import_sequence o import_from_manifest).
    Emet sibling_finished per cada un i all_finished al final.
    """
    progress = Signal(int, str)           # pct global, missatge
    sibling_finished = Signal(str, dict)  # path, result
    all_finished = Signal(dict)           # {path: result}
    error = Signal(str)

    def __init__(self, sibling_paths, load_data=True):
        """
        Args:
            sibling_paths: Llista de paths [primary, sibB, sibC, ...]
            load_data: Si True, carrega dades completes; si False, només metadades
        """
        super().__init__()
        self.sibling_paths = sibling_paths
        self.load_data = load_data

    def run(self):
        results = {}
        n = len(self.sibling_paths)

        try:
            for i, path in enumerate(self.sibling_paths):
                name = os.path.basename(path)
                base_pct = int(i / n * 100)

                def progress_cb(pct, msg, _base=base_pct, _n=n, _name=name):
                    global_pct = _base + int(pct / _n)
                    self.progress.emit(global_pct, f"[{_name}] {msg}")

                # Comprovar si ja hi ha manifest
                manifest = load_manifest(path)

                if manifest and self.load_data:
                    # Manifest existent: carregar des de manifest (ràpid)
                    progress_cb(0, "Carregant des de manifest...")
                    result = import_from_manifest(
                        path, manifest=manifest,
                        progress_callback=progress_cb,
                        load_data=self.load_data
                    )
                elif manifest and not self.load_data:
                    # Metadata-only load
                    progress_cb(0, "Carregant metadades...")
                    result = import_from_manifest(
                        path, manifest=manifest,
                        progress_callback=progress_cb,
                        load_data=False
                    )
                else:
                    # Importació nova
                    progress_cb(0, "Important...")
                    result = import_sequence(path, progress_callback=progress_cb)

                    # Guardar manifest individual
                    if result and result.get("success"):
                        try:
                            save_import_manifest(result)
                        except Exception as e:
                            logger.warning("No s'ha pogut guardar manifest de %s: %s", name, e)

                results[path] = result
                self.sibling_finished.emit(path, result)

            self.progress.emit(100, f"Importació completada ({n} carpetes)")
            self.all_finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
