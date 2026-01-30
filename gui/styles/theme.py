"""
HPSEC Suite - Theme & Styles
=============================

Paleta de colores y estilos QSS para un aspecto moderno y profesional.

Para cambiar colores: modificar el dict COLORS
Para cambiar tipografía: modificar FONT_FAMILY y FONT_SIZES
"""

# === PALETA DE COLORES ===
# Basada en azul corporativo UdG con grises neutros

COLORS = {
    # Primarios
    "primary": "#2563EB",          # Azul principal (botones, acciones)
    "primary_hover": "#1D4ED8",    # Azul hover
    "primary_light": "#DBEAFE",    # Azul claro (backgrounds)

    # Secundarios
    "secondary": "#64748B",        # Gris azulado
    "secondary_hover": "#475569",

    # Estados
    "success": "#22C55E",          # Verde OK
    "success_bg": "#DCFCE7",
    "warning": "#F59E0B",          # Naranja warning
    "warning_bg": "#FEF3C7",
    "error": "#EF4444",            # Rojo error
    "error_bg": "#FEE2E2",
    "info": "#3B82F6",             # Azul info
    "info_bg": "#DBEAFE",

    # Fondos
    "background": "#F8FAFC",       # Fondo principal (gris muy claro)
    "surface": "#FFFFFF",          # Superficie (cards, panels)
    "surface_alt": "#F1F5F9",      # Superficie alternativa

    # Bordes
    "border": "#E2E8F0",           # Borde suave
    "border_strong": "#CBD5E1",    # Borde más visible

    # Texto
    "text_primary": "#1E293B",     # Texto principal (casi negro)
    "text_secondary": "#64748B",   # Texto secundario (gris)
    "text_muted": "#94A3B8",       # Texto apagado
    "text_on_primary": "#FFFFFF",  # Texto sobre fondo primario

    # Header
    "header_bg": "#1E40AF",        # Azul oscuro para header
    "header_text": "#FFFFFF",
}

# === TIPOGRAFÍA ===
FONT_FAMILY = "Segoe UI, Roboto, Inter, sans-serif"
FONT_FAMILY_MONO = "Consolas, 'Courier New', monospace"

FONT_SIZES = {
    "h1": "24px",
    "h2": "20px",
    "h3": "16px",
    "body": "13px",
    "small": "11px",
    "tiny": "10px",
}

# === DIMENSIONES ===
SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
}

RADIUS = {
    "sm": "4px",
    "md": "8px",
    "lg": "12px",
}

# === STYLESHEET QSS ===
STYLESHEET = f"""
/* === GENERAL === */
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZES['body']};
    color: {COLORS['text_primary']};
}}

/* === HEADER === */
QFrame#header {{
    background-color: {COLORS['header_bg']};
    border: none;
}}

QLabel#headerTitle {{
    color: {COLORS['header_text']};
    font-weight: bold;
}}

QLabel#headerSubtitle, QLabel#headerInfo {{
    color: rgba(255, 255, 255, 0.8);
}}

/* === TABS === */
QTabWidget::pane {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-top: none;
    border-radius: 0 0 {RADIUS['md']} {RADIUS['md']};
}}

QTabBar::tab {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text_secondary']};
    padding: 12px 24px;
    margin-right: 2px;
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    border-radius: {RADIUS['sm']} {RADIUS['sm']} 0 0;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['surface']};
    color: {COLORS['primary']};
    border-bottom: 2px solid {COLORS['primary']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS['primary_light']};
}}

QTabBar::tab:disabled {{
    color: {COLORS['text_muted']};
    background-color: {COLORS['surface_alt']};
}}

/* === BOTONES === */
QPushButton {{
    background-color: {COLORS['primary']};
    color: {COLORS['text_on_primary']};
    border: none;
    border-radius: {RADIUS['sm']};
    padding: 10px 20px;
    font-weight: 500;
    min-width: 80px;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_hover']};
}}

QPushButton:pressed {{
    background-color: {COLORS['secondary_hover']};
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_muted']};
}}

/* Botón secundario */
QPushButton[secondary="true"] {{
    background-color: transparent;
    color: {COLORS['primary']};
    border: 1px solid {COLORS['primary']};
}}

QPushButton[secondary="true"]:hover {{
    background-color: {COLORS['primary_light']};
}}

/* === INPUTS === */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['sm']};
    padding: 8px 12px;
    selection-background-color: {COLORS['primary_light']};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS['primary']};
    outline: none;
}}

/* === COMBOBOX === */
QComboBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['sm']};
    padding: 8px 12px;
    min-width: 120px;
}}

QComboBox:hover {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

/* === TABLAS === */
QTableView, QTreeView {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['sm']};
    gridline-color: {COLORS['border']};
    selection-background-color: {COLORS['primary_light']};
    selection-color: {COLORS['text_primary']};
}}

QTableView::item, QTreeView::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}

QTableView::item:selected, QTreeView::item:selected {{
    background-color: {COLORS['primary_light']};
}}

QHeaderView::section {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text_secondary']};
    padding: 10px;
    border: none;
    border-bottom: 2px solid {COLORS['border']};
    font-weight: 600;
}}

/* === GROUP BOX === */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['md']};
    margin-top: 16px;
    padding: 16px;
    padding-top: 24px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    background-color: {COLORS['surface']};
    color: {COLORS['text_primary']};
    font-weight: 600;
}}

/* === PROGRESS BAR === */
QProgressBar {{
    background-color: {COLORS['surface_alt']};
    border: none;
    border-radius: {RADIUS['sm']};
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: {RADIUS['sm']};
}}

/* === SCROLLBAR === */
QScrollBar:vertical {{
    background-color: {COLORS['surface_alt']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_strong']};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['secondary']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* === STATUS BAR === */
QStatusBar {{
    background-color: {COLORS['surface_alt']};
    border-top: 1px solid {COLORS['border']};
    padding: 4px;
}}

/* === TOOLTIPS === */
QToolTip {{
    background-color: {COLORS['text_primary']};
    color: {COLORS['surface']};
    border: none;
    padding: 8px;
    border-radius: {RADIUS['sm']};
}}

/* === CARDS (custom class) === */
QFrame[card="true"] {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['md']};
    padding: 16px;
}}

/* === ESTADO INDICATORS === */
QLabel[status="success"] {{
    color: {COLORS['success']};
    background-color: {COLORS['success_bg']};
    padding: 4px 8px;
    border-radius: {RADIUS['sm']};
}}

QLabel[status="warning"] {{
    color: {COLORS['warning']};
    background-color: {COLORS['warning_bg']};
    padding: 4px 8px;
    border-radius: {RADIUS['sm']};
}}

QLabel[status="error"] {{
    color: {COLORS['error']};
    background-color: {COLORS['error_bg']};
    padding: 4px 8px;
    border-radius: {RADIUS['sm']};
}}
"""
