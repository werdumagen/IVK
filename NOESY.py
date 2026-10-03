import sys
import re
import numpy as np
import scipy.linalg as la

# Підтримка як PyQt6, так і PyQt5
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QSlider, QComboBox, QCheckBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
        QTabWidget, QTextEdit, QPushButton, QFileDialog, QMessageBox
    )
    from PyQt6.QtCore import Qt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QSlider, QComboBox, QCheckBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
        QTabWidget, QTextEdit, QPushButton, QFileDialog, QMessageBox
    )
    from PyQt5.QtCore import Qt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# 1. ПАРСЕРИ ТЕКСТОВИХ ДАНИХ (XYZ ТА ЯМР ЗСУВІВ)
# -----------------------------------------------------------------------------
def parse_nmr_text(text):
    """
    Парсить спектр з довільного тексту:
    - Формат: δ 6.44 (s, 1H, H10), 5.60 (s, 1H, H14)
    - Або стовпчик: 6.44 H10
    - Або просто список чисел
    """
    # 1. Формат з дужками: 6.44 (s, 1H, H10)
    p1 = re.findall(r'(\d+\.\d+)\s*\([^,)]*,\s*[^,)]*,\s*([^)]+)\)', text)
    if p1:
        shifts = [float(p[0]) for p in p1]
        labels = [re.sub(r'[^\w()]+', '', p[1]).strip() for p in p1]
        return shifts, labels

    # 2. Порядковий парсинг: "6.44 H10" або "H10 6.44"
    lines = text.strip().splitlines()
    shifts, labels = [], []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        m = re.search(r'[-+]?\d*\.\d+|\d+', line_clean)
        if m:
            shift = float(m.group(0))
            rem = line_clean.replace(m.group(0), '', 1).strip(" ,:;\t()")
            words = rem.split()
            label = words[0] if words else f"H{len(shifts) + 1}"
            shifts.append(shift)
            labels.append(label)
    if shifts:
        return shifts, labels

    # 3. Резерв: витягуємо всі числа з плаваючою точкою
    floats = re.findall(r'\d+\.\d+', text)
    if floats:
        return [float(f) for f in floats], [f"H{i + 1}" for i in range(len(floats))]

    return [], []


def parse_xyz_text(text):
    """
    Витягує координати гідрогенів з XYZ формату (ORCA/Gaussian) або чистих x y z.
    """
    lines = text.strip().splitlines()
    coords = []
    labels = []

    # Спершу шукаємо лінії, що починаються з 'H'
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            elem = parts[0].strip().capitalize()
            if elem == 'H':
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coords.append([x, y, z])
                    labels.append(f"H{len(coords)}")
                except ValueError:
                    pass

    # Якщо атомів H з міткою не знайдено, перевіряємо, чи це просто 3 числа на рядок (x y z)
    if not coords:
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    coords.append([x, y, z])
                    labels.append(f"H{len(coords)}")
                except ValueError:
                    pass

    return np.array(coords) if coords else np.empty((0, 3)), labels


# -----------------------------------------------------------------------------
# 2. МАТЕМАТИЧНИЙ МОДУЛЬ РОЗРАХУНКУ РЕЛАКСАЦІЇ NOESY
# -----------------------------------------------------------------------------
def calculate_noesy_matrix(shifts, coords, tau_m=0.500, tau_c_ps=80.0, freq_mhz=600.0, rho_leak=0.15):
    N = len(shifts)
    if N < 2 or len(coords) != N:
        return np.eye(N), np.zeros((N, N))

    gamma_H = 2.67522e8
    hbar = 1.05457e-34
    mu0 = 4.0 * np.pi * 1e-7
    K = (mu0 / (4.0 * np.pi)) ** 2 * (hbar ** 2) * (gamma_H ** 4)

    tau_c = tau_c_ps * 1e-12
    omega0 = 2.0 * np.pi * freq_mhz * 1e6

    def J(w):
        return tau_c / (1.0 + (w * tau_c) ** 2)

    J0 = J(0.0)
    Jw = J(omega0)
    J2w = J(2.0 * omega0)

    R = np.zeros((N, N))
    distances = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                r_ang = np.linalg.norm(coords[i] - coords[j])
                # Захист від сингулярності (мінімум 0.8 Å)
                r_ang = max(r_ang, 0.8)
                distances[i, j] = r_ang
                r_m = r_ang * 1e-10
                sigma_ij = (K / (r_m ** 6)) * (6.0 * J2w - J0)
                R[i, j] = -sigma_ij

    for i in range(N):
        R[i, i] = -np.sum(R[i, :]) + rho_leak

    # Експонента матриці релаксації Соломона: I(tau_m) = expm(-R * tau_m)
    intensities = la.expm(-R * tau_m)
    return intensities, distances


# -----------------------------------------------------------------------------
# 3. ГОЛОВНЕ ВІКНО ДОДАТКА
# -----------------------------------------------------------------------------
class NOESYCustomApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D NOESY Simulator & Custom Molecular Workspace")
        self.resize(1420, 880)
        self.apply_dark_theme()

        # Поточні робочі дані
        self.shifts = np.array([5.72, 4.45, 4.17, 4.16, 3.49, 2.72, 2.03, 1.96, 1.30, 1.21, 1.05])
        self.labels = ["H10", "H13", "H14", "H26", "H27", "H19", "H8", "H3", "H7", "H4", "Me(3H)"]
        self.coords = np.array([
            [0.10, 2.45, 1.15],
            [1.25, 1.75, -0.45],
            [1.80, 1.15, 0.85],
            [-1.15, 1.40, -0.60],
            [-1.50, 0.85, 0.75],
            [0.20, -0.45, 1.50],
            [1.35, -1.20, -0.35],
            [-1.20, -1.15, -0.45],
            [0.85, -1.85, 0.65],
            [-0.90, -1.75, 0.70],
            [0.15, -0.25, 2.70],
        ])

        # Параметри спектрометра
        self.tau_m = 0.500
        self.tau_c_ps = 80.0
        self.freq_mhz = 600.0
        self.fwhm = 0.035
        self.suppress_diag = False
        self.cmap_name = "Blues_r"

        self.init_ui()
        self.update_simulation()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #16171d;
                color: #e2e8f0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #2d3748;
                background-color: #1a1c24;
            }
            QTabBar::tab {
                background-color: #232733;
                color: #a0aec0;
                padding: 8px 18px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3182ce;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #2d3748;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: bold;
                color: #63b3ed;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel { color: #cbd5e0; }
            QTextEdit {
                background-color: #121318;
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: #edf2f7;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2b6cb0;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover { background-color: #3182ce; }
            QPushButton:pressed { background-color: #1a365d; }
            QTableWidget {
                background-color: #1a1c24;
                gridline-color: #2d3748;
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: #e2e8f0;
            }
            QHeaderView::section {
                background-color: #232733;
                color: #90cdf4;
                padding: 4px;
                border: 1px solid #2d3748;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #2d3748;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal { background: #3182ce; }
            QSlider::handle:horizontal {
                background: #63b3ed;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QComboBox {
                background-color: #232733;
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
            }
        """)

    def init_ui(self):
        main_layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # -------------------------------------------------------------
        # ВКЛАДКА 1: ВІЗУАЛІЗАТОР NOESY
        # -------------------------------------------------------------
        tab_spectrum = QWidget()
        s_layout = QHBoxLayout(tab_spectrum)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        s_layout.addWidget(splitter)

        # Ліва панель керування спектром
        ctrl_panel = QWidget()
        c_vbox = QVBoxLayout(ctrl_panel)
        c_vbox.setContentsMargins(0, 0, 10, 0)

        param_group = QGroupBox("Параметри експерименту")
        p_grid = QGridLayout(param_group)

        p_grid.addWidget(QLabel("Час змішування (τₘ):"), 0, 0)
        self.lbl_tau_m = QLabel("500 ms")
        self.lbl_tau_m.setStyleSheet("color: #63b3ed; font-weight: bold;")
        p_grid.addWidget(self.lbl_tau_m, 0, 1)
        self.slider_tau_m = QSlider(Qt.Orientation.Horizontal)
        self.slider_tau_m.setRange(50, 1500)
        self.slider_tau_m.setValue(500)
        self.slider_tau_m.valueChanged.connect(self.on_sim_params_changed)
        p_grid.addWidget(self.slider_tau_m, 1, 0, 1, 2)

        p_grid.addWidget(QLabel("Час кореляції (τ_c):"), 2, 0)
        self.lbl_tau_c = QLabel("80 ps")
        self.lbl_tau_c.setStyleSheet("color: #63b3ed; font-weight: bold;")
        p_grid.addWidget(self.lbl_tau_c, 2, 1)
        self.slider_tau_c = QSlider(Qt.Orientation.Horizontal)
        self.slider_tau_c.setRange(20, 300)
        self.slider_tau_c.setValue(80)
        self.slider_tau_c.valueChanged.connect(self.on_sim_params_changed)
        p_grid.addWidget(self.slider_tau_c, 3, 0, 1, 2)

        p_grid.addWidget(QLabel("Частота B₀:"), 4, 0)
        self.combo_freq = QComboBox()
        self.combo_freq.addItems(["400 MHz", "500 MHz", "600 MHz", "800 MHz"])
        self.combo_freq.setCurrentText("600 MHz")
        self.combo_freq.currentIndexChanged.connect(self.on_sim_params_changed)
        p_grid.addWidget(self.combo_freq, 4, 1)

        p_grid.addWidget(QLabel("Ширина ліній (FWHM):"), 5, 0)
        self.lbl_fwhm = QLabel("0.035 ppm")
        self.lbl_fwhm.setStyleSheet("color: #63b3ed; font-weight: bold;")
        p_grid.addWidget(self.lbl_fwhm, 5, 1)
        self.slider_fwhm = QSlider(Qt.Orientation.Horizontal)
        self.slider_fwhm.setRange(15, 80)
        self.slider_fwhm.setValue(35)
        self.slider_fwhm.valueChanged.connect(self.on_sim_params_changed)
        p_grid.addWidget(self.slider_fwhm, 6, 0, 1, 2)

        c_vbox.addWidget(param_group)

        disp_group = QGroupBox("Відображення")
        d_vbox = QVBoxLayout(disp_group)
        self.chk_suppress_diag = QCheckBox("Придушити діагональ (Suppress Diag)")
        self.chk_suppress_diag.stateChanged.connect(self.on_sim_params_changed)
        d_vbox.addWidget(self.chk_suppress_diag)

        h_cmap = QHBoxLayout()
        h_cmap.addWidget(QLabel("Палітра:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["Blues_r", "viridis", "inferno", "coolwarm", "mako_r"])
        self.combo_cmap.currentIndexChanged.connect(self.on_sim_params_changed)
        h_cmap.addWidget(self.combo_cmap)
        d_vbox.addLayout(h_cmap)
        c_vbox.addWidget(disp_group)

        btn_go_editor = QPushButton("✏️ Налаштувати зсуви та геометрію...")
        btn_go_editor.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        c_vbox.addWidget(btn_go_editor)

        c_vbox.addStretch()
        ctrl_panel.setMinimumWidth(300)
        splitter.addWidget(ctrl_panel)

        # Центральний графік спектра
        center_panel = QWidget()
        cp_vbox = QVBoxLayout(center_panel)
        self.figure = Figure(figsize=(7, 7), facecolor='#16171d')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        cp_vbox.addWidget(self.toolbar)
        cp_vbox.addWidget(self.canvas)
        splitter.addWidget(center_panel)

        # Права таблиця крос-піків
        right_panel = QWidget()
        r_vbox = QVBoxLayout(right_panel)
        t_group = QGroupBox("Крос-піки NOE (r < 3.8 Å)")
        tg_vbox = QVBoxLayout(t_group)
        self.table_noesy = QTableWidget()
        self.table_noesy.setColumnCount(4)
        self.table_noesy.setHorizontalHeaderLabels(["Протони", "δ₁ - δ₂ (ppm)", "r (Å)", "NOE (%)"])
        self.table_noesy.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_noesy.verticalHeader().setVisible(False)
        tg_vbox.addWidget(self.table_noesy)
        r_vbox.addWidget(t_group)
        right_panel.setMinimumWidth(340)
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 680, 360])
        self.tabs.addTab(tab_spectrum, "📊 2D NOESY Спектр")

        # -------------------------------------------------------------
        # ВКЛАДКА 2: РЕДАКТОР ВХІДНИХ ДАНИХ (XYZ ТА ЯМР ЗСУВІВ)
        # -------------------------------------------------------------
        tab_editor = QWidget()
        ed_layout = QVBoxLayout(tab_editor)

        # Верхня панель дій
        top_bar = QHBoxLayout()
        btn_load_xyz = QPushButton("📂 Завантажити .xyz файл")
        btn_load_xyz.clicked.connect(self.load_xyz_file)
        top_bar.addWidget(btn_load_xyz)

        btn_preset_camph = QPushButton("🧪 Пресет: Camphora")
        btn_preset_camph.clicked.connect(self.load_preset_camphora)
        top_bar.addWidget(btn_preset_camph)

        btn_preset_isocamph = QPushButton("🧪 Пресет: isoCamphora")
        btn_preset_isocamph.clicked.connect(self.load_preset_isocamphora)
        top_bar.addWidget(btn_preset_isocamph)

        top_bar.addStretch()
        ed_layout.addLayout(top_bar)

        # Два текстових редактори (для швидкої вставки)
        text_editors_layout = QHBoxLayout()

        # Лівий: Текст хімічних зсувів
        nmr_box = QGroupBox("1. Вставити хімічні зсуви (1H NMR)")
        nb_layout = QVBoxLayout(nmr_box)
        self.text_nmr_input = QTextEdit()
        self.text_nmr_input.setPlaceholderText(
            "Вставте сюди спектр у довільному форматі:\n"
            "δ 6.44 (s, 1H, H10), 5.60 (s, 1H, H14), 5.09...\n"
            "або стовпчик пар: 6.44 H10\n"
            "5.60 H14"
        )
        nb_layout.addWidget(self.text_nmr_input)
        text_editors_layout.addWidget(nmr_box)

        # Правий: Текст координат XYZ
        xyz_box = QGroupBox("2. Вставити 3D координати (XYZ з ORCA або чисті X Y Z)")
        xb_layout = QVBoxLayout(xyz_box)
        self.text_xyz_input = QTextEdit()
        self.text_xyz_input.setPlaceholderText(
            "Вставте вихідний блок координат ORCA / XYZ:\n"
            "H   0.10   2.45   1.15\n"
            "H   1.25   1.75  -0.45\n"
            "(Програма автоматично відфільтрує атоми гідрогену)"
        )
        xb_layout.addWidget(self.text_xyz_input)
        text_editors_layout.addWidget(xyz_box)

        ed_layout.addLayout(text_editors_layout)

        # Кнопка парсингу текстів у таблицю
        btn_parse_texts = QPushButton("📥 Обробити вставлені тексти в робочу таблицю")
        btn_parse_texts.setStyleSheet("background-color: #319795; font-size: 13px; padding: 8px;")
        btn_parse_texts.clicked.connect(self.parse_and_fill_table)
        ed_layout.addWidget(btn_parse_texts)

        # Робоча інтерактивна таблиця
        tab_box = QGroupBox("3. Інтерактивна робоча таблиця (можна редагувати комірки на льоту)")
        tb_layout = QVBoxLayout(tab_box)
        self.editor_table = QTableWidget()
        self.editor_table.setColumnCount(5)
        self.editor_table.setHorizontalHeaderLabels(["Мітка", "δ (ppm)", "X (Å)", "Y (Å)", "Z (Å)"])
        self.editor_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tb_layout.addWidget(self.editor_table)

        row_ctrls = QHBoxLayout()
        btn_add_row = QPushButton("➕ Додати рядок")
        btn_add_row.clicked.connect(self.add_table_row)
        row_ctrls.addWidget(btn_add_row)

        btn_del_row = QPushButton("➖ Видалити виділений рядок")
        btn_del_row.clicked.connect(self.delete_table_row)
        row_ctrls.addWidget(btn_del_row)
        row_ctrls.addStretch()
        tb_layout.addLayout(row_ctrls)
        ed_layout.addWidget(tab_box)

        # Велика кнопка застосування
        self.btn_apply_data = QPushButton("🚀 ЗАСТОСУВАТИ ДАНІ ТА ПЕРЕРАХУВАТИ NOESY")
        self.btn_apply_data.setStyleSheet("""
            background-color: #38a169;
            color: white;
            font-size: 14px;
            font-weight: bold;
            padding: 12px;
            border-radius: 6px;
        """)
        self.btn_apply_data.clicked.connect(self.apply_data_from_table)
        ed_layout.addWidget(self.btn_apply_data)

        self.tabs.addTab(tab_editor, "⚙️ Введення 3D-геометрії (XYZ) та Зсувів")

        # Заповнюємо початковий стан таблиці
        self.fill_table_from_state()

    # -------------------------------------------------------------------------
    # УПРАВЛІННЯ ДАНИМИ ТА ТАБЛИЦЕЮ
    # -------------------------------------------------------------------------
    def fill_table_from_state(self):
        N = len(self.shifts)
        self.editor_table.setRowCount(N)
        for i in range(N):
            self.editor_table.setItem(i, 0, QTableWidgetItem(str(self.labels[i])))
            self.editor_table.setItem(i, 1, QTableWidgetItem(f"{self.shifts[i]:.2f}"))
            self.editor_table.setItem(i, 2, QTableWidgetItem(f"{self.coords[i, 0]:.4f}"))
            self.editor_table.setItem(i, 3, QTableWidgetItem(f"{self.coords[i, 1]:.4f}"))
            self.editor_table.setItem(i, 4, QTableWidgetItem(f"{self.coords[i, 2]:.4f}"))

    def add_table_row(self):
        row = self.editor_table.rowCount()
        self.editor_table.insertRow(row)
        self.editor_table.setItem(row, 0, QTableWidgetItem(f"H{row + 1}"))
        self.editor_table.setItem(row, 1, QTableWidgetItem("2.00"))
        self.editor_table.setItem(row, 2, QTableWidgetItem("0.0000"))
        self.editor_table.setItem(row, 3, QTableWidgetItem("0.0000"))
        self.editor_table.setItem(row, 4, QTableWidgetItem("0.0000"))

    def delete_table_row(self):
        curr_row = self.editor_table.currentRow()
        if curr_row >= 0:
            self.editor_table.removeRow(curr_row)

    def parse_and_fill_table(self):
        nmr_txt = self.text_nmr_input.toPlainText().strip()
        xyz_txt = self.text_xyz_input.toPlainText().strip()

        if not nmr_txt and not xyz_txt:
            QMessageBox.warning(self, "Попередження",
                                "Будь ласка, вставте хоча б зсуви або XYZ координати у відповідні поля.")
            return

        shifts, labels = parse_nmr_text(nmr_txt) if nmr_txt else ([], [])
        coords, _ = parse_xyz_text(xyz_txt) if xyz_txt else (np.empty((0, 3)), [])

        # Визначаємо кількість рядків
        N = max(len(shifts), len(coords))
        if N == 0:
            QMessageBox.warning(self, "Помилка парсингу", "Не вдалося розпізнати числа у введених текстах.")
            return

        self.editor_table.setRowCount(N)
        for i in range(N):
            l = labels[i] if i < len(labels) else f"H{i + 1}"
            s = f"{shifts[i]:.2f}" if i < len(shifts) else "2.00"
            x = f"{coords[i, 0]:.4f}" if i < len(coords) else "0.0000"
            y = f"{coords[i, 1]:.4f}" if i < len(coords) else "0.0000"
            z = f"{coords[i, 2]:.4f}" if i < len(coords) else "0.0000"

            self.editor_table.setItem(i, 0, QTableWidgetItem(l))
            self.editor_table.setItem(i, 1, QTableWidgetItem(s))
            self.editor_table.setItem(i, 2, QTableWidgetItem(x))
            self.editor_table.setItem(i, 3, QTableWidgetItem(y))
            self.editor_table.setItem(i, 4, QTableWidgetItem(z))

        QMessageBox.information(
            self, "Успіх",
            f"Дані успішно оброблено:\n• Знайдено зсувів: {len(shifts)}\n• Знайдено координат H: {len(coords)}\n"
            "Перевірте зіставлення у таблиці нижче і натисніть зелену кнопку 'ЗАСТОСУВАТИ ДАНІ'."
        )

    def load_xyz_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Виберіть .xyz файл", "", "XYZ Files (*.xyz);;All Files (*)")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.text_xyz_input.setPlainText(content)
            coords, _ = parse_xyz_text(content)
            QMessageBox.information(self, "Файл завантажено",
                                    f"Завантажено {len(coords)} атомів гідрогену (H) з файлу.")

    def load_preset_camphora(self):
        self.shifts = np.array([5.72, 4.45, 4.17, 4.16, 3.49, 2.72, 2.03, 1.96, 1.30, 1.21, 1.05])
        self.labels = ["H10", "H13", "H14", "H26", "H27", "H19", "H8", "H3", "H7", "H4", "Me(3H)"]
        self.coords = np.array([
            [0.10, 2.45, 1.15], [1.25, 1.75, -0.45], [1.80, 1.15, 0.85],
            [-1.15, 1.40, -0.60], [-1.50, 0.85, 0.75], [0.20, -0.45, 1.50],
            [1.35, -1.20, -0.35], [-1.20, -1.15, -0.45], [0.85, -1.85, 0.65],
            [-0.90, -1.75, 0.70], [0.15, -0.25, 2.70],
        ])
        self.fill_table_from_state()
        self.update_simulation()
        self.tabs.setCurrentIndex(0)

    def load_preset_isocamphora(self):
        self.shifts = np.array([5.58, 4.86, 4.43, 3.75, 3.36, 2.44, 2.02, 1.97, 1.32, 1.31, 1.21])
        self.labels = ["H10", "H14", "H13", "H26", "H27", "H19", "H8", "H3", "H7", "Me(3H)", "H4"]
        self.coords = np.array([
            [0.05, 2.40, 1.10], [1.55, 1.45, 0.25], [1.10, 1.85, -0.80],
            [-1.25, 1.30, -0.50], [-1.45, 0.75, 0.80], [-1.85, -0.45, 0.45],
            [1.30, -1.10, -0.30], [-1.10, -1.25, -0.40], [0.75, -1.80, 0.70],
            [2.25, -0.85, 0.50], [-0.85, -1.70, 0.75],
        ])
        self.fill_table_from_state()
        self.update_simulation()
        self.tabs.setCurrentIndex(0)

    def apply_data_from_table(self):
        rows = self.editor_table.rowCount()
        if rows < 2:
            QMessageBox.warning(self, "Помилка", "Потрібно щонайменше 2 протони для розрахунку NOESY!")
            return

        new_labels = []
        new_shifts = []
        new_coords = []

        try:
            for r in range(rows):
                lbl = self.editor_table.item(r, 0).text().strip()
                s = float(self.editor_table.item(r, 1).text())
                x = float(self.editor_table.item(r, 2).text())
                y = float(self.editor_table.item(r, 3).text())
                z = float(self.editor_table.item(r, 4).text())

                new_labels.append(lbl)
                new_shifts.append(s)
                new_coords.append([x, y, z])
        except Exception as e:
            QMessageBox.critical(self, "Помилка даних", f"Перевірте коректність чисел у таблиці: {e}")
            return

        self.labels = new_labels
        self.shifts = np.array(new_shifts)
        self.coords = np.array(new_coords)

        self.update_simulation()
        self.tabs.setCurrentIndex(0)

    # -------------------------------------------------------------------------
    # СИМУЛЯЦІЯ ТА ВІЗУАЛІЗАЦІЯ
    # -------------------------------------------------------------------------
    def on_sim_params_changed(self):
        self.tau_m = self.slider_tau_m.value() / 1000.0
        self.lbl_tau_m.setText(f"{self.slider_tau_m.value()} ms")

        self.tau_c_ps = float(self.slider_tau_c.value())
        self.lbl_tau_c.setText(f"{int(self.tau_c_ps)} ps")

        freq_str = self.combo_freq.currentText()
        self.freq_mhz = float(freq_str.replace(" MHz", ""))

        self.fwhm = self.slider_fwhm.value() / 1000.0
        self.lbl_fwhm.setText(f"{self.fwhm:.3f} ppm")

        self.suppress_diag = self.chk_suppress_diag.isChecked()
        self.cmap_name = self.combo_cmap.currentText()

        self.update_simulation()

    def update_simulation(self):
        intensities, distances = calculate_noesy_matrix(
            self.shifts, self.coords,
            tau_m=self.tau_m,
            tau_c_ps=self.tau_c_ps,
            freq_mhz=self.freq_mhz
        )

        self.populate_cross_peaks_table(intensities, distances)
        self.plot_spectrum(intensities)

    def populate_cross_peaks_table(self, intensities, distances):
        N = len(self.shifts)
        diag_ref = np.median(np.diag(intensities)) if N > 0 else 1.0
        peaks = []

        for i in range(N):
            for j in range(i + 1, N):
                r = distances[i, j]
                noe_pct = abs(intensities[i, j] / diag_ref) * 100.0
                if r < 3.8:
                    peaks.append((self.labels[i], self.labels[j], self.shifts[i], self.shifts[j], r, noe_pct))

        peaks.sort(key=lambda x: x[5], reverse=True)
        self.table_noesy.setRowCount(len(peaks))

        for row, (l1, l2, s1, s2, r, noe) in enumerate(peaks):
            it_pair = QTableWidgetItem(f"{l1} — {l2}")
            it_shift = QTableWidgetItem(f"{s1:.2f} × {s2:.2f}")
            it_r = QTableWidgetItem(f"{r:.2f}")
            it_noe = QTableWidgetItem(f"{noe:.2f}%")

            it_pair.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_shift.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_r.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_noe.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table_noesy.setItem(row, 0, it_pair)
            self.table_noesy.setItem(row, 1, it_shift)
            self.table_noesy.setItem(row, 2, it_r)
            self.table_noesy.setItem(row, 3, it_noe)

    def plot_spectrum(self, intensities):
        self.ax.clear()
        self.ax.set_facecolor("#111216")

        N = len(self.shifts)
        if N == 0:
            self.canvas.draw()
            return

        ppm_min = max(0.0, np.min(self.shifts) - 0.6)
        ppm_max = np.max(self.shifts) + 0.6

        grid_res = 350
        ppm_grid = np.linspace(ppm_min, ppm_max, grid_res)
        X, Y = np.meshgrid(ppm_grid, ppm_grid)
        Z = np.zeros_like(X)

        for i in range(N):
            for j in range(N):
                amp = intensities[i, j]
                if i == j and self.suppress_diag:
                    amp = 0.0

                Z += amp * np.exp(-4.0 * np.log(2) * (
                        ((X - self.shifts[i]) ** 2 + (Y - self.shifts[j]) ** 2) / (self.fwhm ** 2)
                ))

        Z_abs = np.abs(Z)
        max_val = np.max(Z_abs)
        if max_val > 0:
            threshold = 0.008 if self.suppress_diag else 0.015
            levels = np.geomspace(max_val * threshold, max_val * 0.95, 18)
            try:
                self.ax.contour(X, Y, Z_abs, levels=levels, cmap=self.cmap_name, linewidths=0.9)
            except Exception:
                pass

        # Діагональ
        self.ax.plot([ppm_min, ppm_max], [ppm_min, ppm_max], color="#4a5568", linestyle="--", linewidth=0.8, alpha=0.7)

        # Підписи піків уздовж діагоналі
        for s, l in zip(self.shifts, self.labels):
            self.ax.text(s, s, f" {l}", color="#cbd5e0", fontsize=8, verticalalignment='bottom')

        # Осі за стандартом ЯМР (спадання поля зліва направо)
        self.ax.set_xlim(ppm_max, ppm_min)
        self.ax.set_ylim(ppm_max, ppm_min)
        self.ax.set_xlabel("F2 (¹H) [ppm]", color="#e2e8f0", fontsize=11, labelpad=8)
        self.ax.set_ylabel("F1 (¹H) [ppm]", color="#e2e8f0", fontsize=11, labelpad=8)
        self.ax.set_title(
            f"2D NOESY ({N} протонів, {int(self.freq_mhz)} MHz, τₘ = {int(self.tau_m * 1000)} ms)",
            color="#63b3ed", fontsize=12, pad=12, fontweight='bold'
        )

        self.ax.tick_params(colors="#a0aec0", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#2d3748")

        self.ax.grid(True, linestyle=":", color="#2d3748", alpha=0.6)
        self.figure.tight_layout()
        self.canvas.draw()


# -----------------------------------------------------------------------------
# ТОЧКА ВХОДУ
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NOESYCustomApp()
    window.show()
    sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())