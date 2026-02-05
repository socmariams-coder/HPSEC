"""
HPSEC Suite - Import Panel Delegates
====================================

Delegates per a la taula d'importació.
"""

from PySide6.QtWidgets import QStyledItemDelegate, QComboBox
from PySide6.QtCore import Qt


class ComboBoxDelegate(QStyledItemDelegate):
    """Delegate per editar amb ComboBox al fer doble-clic."""

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.items = items

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.items)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        idx = editor.findText(value)
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class FileAssignmentDelegate(QStyledItemDelegate):
    """Delegate per assignar fitxers amb ComboBox."""

    def __init__(self, get_options_func, parent=None):
        super().__init__(parent)
        self.get_options_func = get_options_func

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        options = self.get_options_func(index.row(), index.column())
        combo.addItems(options)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        idx = editor.findText(value)
        if idx >= 0:
            editor.setCurrentIndex(idx)
        else:
            editor.setCurrentIndex(0)

    def setModelData(self, editor, model, index):
        new_value = editor.currentText()
        old_value = model.data(index, Qt.DisplayRole)
        # Guardar el fitxer seleccionat (nom real)
        # UserRole ja pot contenir el nom del fitxer original, preservar-lo si no canvia
        current_file = model.data(index, Qt.UserRole)

        # Si l'usuari selecciona un nou fitxer del dropdown, actualitzar UserRole
        if new_value and new_value not in ["-", "(cap)"]:
            # Si és un fitxer diferent al que hi havia, guardar-lo
            if new_value != current_file:
                model.setData(index, new_value, Qt.UserRole)

        # Actualitzar el text mostrat
        model.setData(index, new_value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)
