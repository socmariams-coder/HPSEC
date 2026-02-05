"""
HPSEC Suite - Shared UI Styles
==============================

Constants i estils compartits per tots els panels.
Garanteix consistència visual a tota l'aplicació.
"""

from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt


# =============================================================================
# LAYOUT CONSTANTS
# =============================================================================

# Marges principals del panel (left, top, right, bottom)
PANEL_MARGINS = (16, 12, 16, 12)

# Spacing entre elements
PANEL_SPACING = 12
SECTION_SPACING = 16
COMPACT_SPACING = 8

# Marges per seccions internes
SECTION_MARGINS = (12, 8, 12, 8)
COMPACT_MARGINS = (8, 6, 8, 6)
NO_MARGINS = (0, 0, 0, 0)


# =============================================================================
# TYPOGRAPHY
# =============================================================================

# Títol principal del panel
TITLE_FONT_SIZE = 16
TITLE_FONT_WEIGHT = QFont.Bold

# Subtítols i seccions
SUBTITLE_FONT_SIZE = 14
SUBTITLE_FONT_WEIGHT = QFont.DemiBold

# Text normal
BODY_FONT_SIZE = 12
SMALL_FONT_SIZE = 11
TINY_FONT_SIZE = 10


# =============================================================================
# COLORS
# =============================================================================

# Colors primaris
COLOR_PRIMARY = "#2E86AB"       # Blau principal
COLOR_PRIMARY_DARK = "#1A5276"  # Blau fosc
COLOR_SECONDARY = "#2A9D8F"     # Verd-blau

# Colors d'estat
COLOR_SUCCESS = "#27AE60"       # Verd èxit
COLOR_SUCCESS_LIGHT = "#D5F5E3" # Verd clar (fons)
COLOR_WARNING = "#F39C12"       # Taronja avís
COLOR_WARNING_LIGHT = "#FCF3CF" # Groc clar (fons)
COLOR_ERROR = "#E74C3C"         # Vermell error
COLOR_ERROR_LIGHT = "#FADBD8"   # Rosa clar (fons)

# Colors neutres
COLOR_TEXT = "#2C3E50"          # Text principal
COLOR_TEXT_SECONDARY = "#666666" # Text secundari
COLOR_TEXT_MUTED = "#888888"    # Text apagat
COLOR_BORDER = "#CCCCCC"        # Vores
COLOR_BACKGROUND = "#FFFFFF"    # Fons blanc


# =============================================================================
# STYLESHEETS
# =============================================================================

# Barra d'avisos (warnings)
STYLE_WARNING_BAR = """
    QFrame {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
    }
"""

STYLE_WARNING_TEXT = "color: #856404;"

# Barra d'error
STYLE_ERROR_BAR = """
    QFrame {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 6px;
    }
"""

STYLE_ERROR_TEXT = "color: #721c24;"

# Barra d'èxit
STYLE_SUCCESS_BAR = """
    QFrame {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 6px;
    }
"""

STYLE_SUCCESS_TEXT = "color: #155724;"

# Placeholder text
STYLE_PLACEHOLDER = f"color: {COLOR_TEXT_MUTED}; font-size: 14px;"

# Labels
STYLE_LABEL_BOLD = f"font-weight: bold; color: {COLOR_TEXT};"
STYLE_LABEL_SECONDARY = f"color: {COLOR_TEXT_SECONDARY};"
STYLE_LABEL_MUTED = f"color: {COLOR_TEXT_MUTED}; font-size: {SMALL_FONT_SIZE}px;"

# GroupBox
STYLE_GROUPBOX = "QGroupBox { font-weight: bold; }"
STYLE_GROUPBOX_PRIMARY = f"QGroupBox {{ font-weight: bold; color: {COLOR_PRIMARY_DARK}; }}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_title_font():
    """Crea font per títols principals."""
    font = QFont()
    font.setPointSize(TITLE_FONT_SIZE)
    font.setWeight(TITLE_FONT_WEIGHT)
    return font


def create_subtitle_font():
    """Crea font per subtítols."""
    font = QFont()
    font.setPointSize(SUBTITLE_FONT_SIZE)
    font.setWeight(SUBTITLE_FONT_WEIGHT)
    return font


def apply_panel_layout(layout):
    """Aplica marges i spacing estàndard a un layout de panel."""
    layout.setContentsMargins(*PANEL_MARGINS)
    layout.setSpacing(PANEL_SPACING)


def apply_section_layout(layout):
    """Aplica marges i spacing per seccions internes."""
    layout.setContentsMargins(*SECTION_MARGINS)
    layout.setSpacing(COMPACT_SPACING)


# =============================================================================
# EMPTY STATE / PLACEHOLDER
# =============================================================================

STYLE_EMPTY_STATE = f"""
    QFrame {{
        background-color: #f8f9fa;
        border: 2px dashed {COLOR_BORDER};
        border-radius: 8px;
        padding: 24px;
    }}
"""

STYLE_EMPTY_STATE_TEXT = f"""
    color: {COLOR_TEXT_MUTED};
    font-size: 14px;
"""

STYLE_EMPTY_STATE_ICON = f"""
    color: {COLOR_TEXT_MUTED};
    font-size: 32px;
"""


def create_empty_state_widget(icon: str, title: str, description: str, parent=None):
    """
    Crea un widget d'estat buit amb icona, títol i descripció.

    Args:
        icon: Emoji o caràcter per l'icona (ex: "📁", "📊", "🔬")
        title: Títol breu (ex: "No hi ha dades")
        description: Descripció o instrucció (ex: "Importa una seqüència per començar")
        parent: Widget pare (opcional)

    Returns:
        QFrame configurat com a estat buit
    """
    from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt

    frame = QFrame(parent)
    frame.setStyleSheet(STYLE_EMPTY_STATE)

    layout = QVBoxLayout(frame)
    layout.setAlignment(Qt.AlignCenter)
    layout.setSpacing(8)

    icon_label = QLabel(icon)
    icon_label.setStyleSheet(STYLE_EMPTY_STATE_ICON)
    icon_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(icon_label)

    title_label = QLabel(title)
    title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #495057;")
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)

    desc_label = QLabel(description)
    desc_label.setStyleSheet(STYLE_EMPTY_STATE_TEXT)
    desc_label.setAlignment(Qt.AlignCenter)
    desc_label.setWordWrap(True)
    layout.addWidget(desc_label)

    return frame
