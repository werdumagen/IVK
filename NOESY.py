import sys
import os
import json
import re
import numpy as np
import scipy.linalg as la

# Підтримка PyQt6 та PyQt5
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QSlider, QComboBox, QCheckBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
        QTabWidget, QTextEdit, QPushButton, QFileDialog, QMessageBox,
        QListWidget, QLineEdit
    )
    from PyQt6.QtCore import Qt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QSlider, QComboBox, QCheckBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
        QTabWidget, QTextEdit, QPushButton, QFileDialog, QMessageBox,
        QListWidget, QLineEdit
    )
    from PyQt5.QtCore import Qt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# 1. ПАРСЕРИ ТА КОРЕКТНА ОБРОБКА CH3
# -----------------------------------------------------------------------------
def parse_nmr_text_with_multiplicity(text):
    expanded_shifts = []
    expanded_labels = []

    # 1. Формат: 1.15 (s, 3H, Group (3H)) або 6.48 (s, 1H, H10)
    p_full = re.finditer(r'(\d+\.\d+)\s*\([^,)]*,\s*(\d+)H\s*(?:,\s*([^)]*))?\)', text, re.IGNORECASE)
    matches = list(p_full)

    if matches:
        for m in matches:
            shift = float(m.group(1))
            n_h = int(m.group(2))
            base_lbl = m.group(3).strip() if m.group(3) else "H"
            base_lbl = re.sub(r'[^\w()]+', '', base_lbl).strip()

            if n_h == 1:
                expanded_shifts.append(shift)
                expanded_labels.append(base_lbl)
            else:
                for k in range(n_h):
                    expanded_shifts.append(shift)
                    expanded_labels.append(f"{base_lbl}_{chr(97 + k)}")
        return expanded_shifts, expanded_labels

    # 2. Формат: 1.05 s H3 або 6.10 s H1
    p_short = re.finditer(r'(\d+\.\d+)\s+[a-zA-Z]+\s+([a-zA-Z0-9_()]+)', text)
    matches_short = list(p_short)
    if matches_short:
        for m in matches_short:
            shift = float(m.group(1))
            tag = m.group(2).strip()
            m_h3 = re.search(r'(\d+)H|H(\d+)', tag, re.IGNORECASE)
            n_h = 1
            if m_h3:
                cnt = m_h3.group(1) or m_h3.group(2)
                if cnt == '3':
                    n_h = 3

            if n_h == 1:
                expanded_shifts.append(shift)
                expanded_labels.append(tag)
            else:
                for k in range(n_h):
                    expanded_shifts.append(shift)
                    expanded_labels.append(f"{tag}_{chr(97 + k)}")
        return expanded_shifts, expanded_labels

    # 3. Числа через кому / пробіл
    floats = re.findall(r'\d+\.\d+', text)
    for i, f in enumerate(floats):
        expanded_shifts.append(float(f))
        expanded_labels.append(f"H{i + 1}")

    return expanded_shifts, expanded_labels


def parse_xyz_only_hydrogens(text):
    lines = text.strip().splitlines()
    coords = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            elem = parts[0].strip().capitalize()
            if elem == 'H':
                try:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    pass
        elif len(parts) == 3:
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                pass
    return np.array(coords) if coords else np.empty((0, 3))


# -----------------------------------------------------------------------------
# 2. МАТЕМАТИЧНИЙ МОДУЛЬ РОЗРАХУНКУ РЕЛАКСАЦІЇ NOESY
# -----------------------------------------------------------------------------
def calculate_noesy(shifts, coords, tau_m=0.500, tau_c_ps=80.0, freq_mhz=600.0, rho_leak=0.15):
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
                r_ang = max(np.linalg.norm(coords[i] - coords[j]), 0.8)
                distances[i, j] = r_ang
                r_m = r_ang * 1e-10
                sigma_ij = (K / (r_m ** 6)) * (6.0 * J2w - J0)
                R[i, j] = -sigma_ij

    for i in range(N):
        R[i, i] = -np.sum(R[i, :]) + rho_leak

    intensities = la.expm(-R * tau_m)
    return intensities, distances


# -----------------------------------------------------------------------------
# 3. МОДЕЛЬ ДАНИХ ТА СЕРІАЛІЗАЦІЯ СЕСІЇ
# -----------------------------------------------------------------------------
class MoleculeData:
    def __init__(self, name="Нова молекула"):
        self.name = name
        self.labels = []
        self.shifts = np.array([])
        self.coords = np.empty((0, 3))

    def to_dict(self):
        return {
            "name": self.name,
            "labels": list(self.labels),
            "shifts": self.shifts.tolist() if isinstance(self.shifts, np.ndarray) else list(self.shifts),
            "coords": self.coords.tolist() if isinstance(self.coords, np.ndarray) else list(self.coords)
        }

    @classmethod
    def from_dict(cls, d):
        mol = cls(d.get("name", "Молекула"))
        mol.labels = d.get("labels", [])
        mol.shifts = np.array(d.get("shifts", []))
        mol.coords = np.array(d.get("coords", []))
        return mol


# -----------------------------------------------------------------------------
# 4. ГОЛОВНЕ ВІКНО ДОДАТКА
# -----------------------------------------------------------------------------
class NOESYStudioPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D NOESY Studio Pro | Multi-Molecule & Threshold Filtering")
        self.resize(1520, 920)
        self.apply_dark_theme()

        self.molecules = []
        self.current_mol_idx = -1

        # Параметри за замовчуванням
        self.tau_m = 0.500
        self.tau_c_ps = 80.0
        self.freq_mhz = 600.0
        self.fwhm = 0.035
        self.suppress_diag = False
        self.noe_cutoff_pct = 0.10  # Поріг відсікання NOE у % (за замовчуванням 0.10%)
        self.cmap_name = "Blues_r"

        self.init_ui()
        self.add_new_molecule("Молекула 1")

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
                padding: 9px 20px;
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
            QLineEdit, QTextEdit {
                background-color: #121318;
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: #edf2f7;
                font-family: 'SFMono-Regular', Consolas, monospace;
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
            QListWidget {
                background-color: #1a1c24;
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: #e2e8f0;
            }
            QListWidget::item:selected {
                background-color: #2b6cb0;
                color: white;
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

        # -------------------------------------------------------------
        # ВЕРХНІЙ ТУЛБАР: ЗБЕРЕЖЕННЯ ТА ЗАВАНТАЖЕННЯ СЕСІЙ
        # -------------------------------------------------------------
        session_bar = QHBoxLayout()
        session_title = QLabel("💾 Проект сесії:")
        session_title.setStyleSheet("font-weight: bold; color: #63b3ed;")
        session_bar.addWidget(session_title)

        btn_save_session = QPushButton("💾 Зберегти сесію (.json)")
        btn_save_session.setStyleSheet("background-color: #2c5282;")
        btn_save_session.clicked.connect(self.save_session_dialog)
        session_bar.addWidget(btn_save_session)

        btn_load_session = QPushButton("📂 Відкрити сесію (.json)")
        btn_load_session.setStyleSheet("background-color: #2c5282;")
        btn_load_session.clicked.connect(self.load_session_dialog)
        session_bar.addWidget(btn_load_session)

        session_bar.addStretch()
        main_layout.addLayout(session_bar)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # -------------------------------------------------------------
        # ВКЛАДКА 1: ВВЕДЕННЯ ТА РЕДАКТОР ДАНИХ
        # -------------------------------------------------------------
        tab_data = QWidget()
        td_layout = QHBoxLayout(tab_data)
        split_data = QSplitter(Qt.Orientation.Horizontal)
        td_layout.addWidget(split_data)

        # Список молекул
        left_box = QWidget()
        lb_layout = QVBoxLayout(left_box)
        lb_layout.setContentsMargins(0, 0, 8, 0)

        mol_list_group = QGroupBox("Список молекул")
        ml_layout = QVBoxLayout(mol_list_group)

        self.list_molecules = QListWidget()
        self.list_molecules.currentRowChanged.connect(self.on_molecule_selected)
        ml_layout.addWidget(self.list_molecules)

        btn_row_mols = QHBoxLayout()
        btn_add_mol = QPushButton("➕ Додати")
        btn_add_mol.clicked.connect(lambda: self.add_new_molecule())
        btn_row_mols.addWidget(btn_add_mol)

        btn_dup_mol = QPushButton("📋 Дублювати")
        btn_dup_mol.clicked.connect(self.duplicate_current_molecule)
        btn_row_mols.addWidget(btn_dup_mol)

        btn_del_mol = QPushButton("🗑️ Видалити")
        btn_del_mol.clicked.connect(self.delete_current_molecule)
        btn_row_mols.addWidget(btn_del_mol)
        ml_layout.addLayout(btn_row_mols)

        lb_layout.addWidget(mol_list_group)
        left_box.setMinimumWidth(260)
        split_data.addWidget(left_box)

        # Редагування
        right_box = QWidget()
        rb_layout = QVBoxLayout(right_box)
        rb_layout.setContentsMargins(8, 0, 0, 0)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Назва молекули:"))
        self.edit_mol_name = QLineEdit()
        self.edit_mol_name.textChanged.connect(self.on_molecule_name_changed)
        name_row.addWidget(self.edit_mol_name)
        rb_layout.addLayout(name_row)

        inputs_row = QHBoxLayout()

        box_nmr = QGroupBox("1. Хімічні зсуви (1H NMR)")
        bn_layout = QVBoxLayout(box_nmr)
        self.txt_shifts = QTextEdit()
        self.txt_shifts.setPlaceholderText(
            "Вставте зсуви. CH3-групи з '3H' автоматично розгортаються на 3 атоми!\n"
            "Приклад:\n"
            "δ 6.44 (s, 1H, H10), 5.60 (s, 1H, H14), 1.05 (s, 3H, Group(3H))\n"
            "або стовпчиком:\n"
            "6.44 H10\n"
            "1.05 Me_3H"
        )
        bn_layout.addWidget(self.txt_shifts)
        inputs_row.addWidget(box_nmr)

        box_xyz = QGroupBox("2. 3D Координати (XYZ з ORCA / Avogadro)")
        bx_layout = QVBoxLayout(box_xyz)
        self.txt_xyz = QTextEdit()
        self.txt_xyz.setPlaceholderText(
            "Вставте весь .xyz блок (програма відбере тільки H):\n"
            "H   0.10   2.45   1.15\n"
            "H   1.25   1.75  -0.45\n"
            "C   ... (ігнорується)"
        )
        bx_layout.addWidget(self.txt_xyz)

        btn_file_xyz = QPushButton("📂 Завантажити з .xyz файлу")
        btn_file_xyz.clicked.connect(self.load_xyz_file_dialog)
        bx_layout.addWidget(btn_file_xyz)
        inputs_row.addWidget(box_xyz)

        rb_layout.addLayout(inputs_row)

        btn_parse = QPushButton("📥 Скомпонувати у таблицю атомів (з авто-розгортанням CH3)")
        btn_parse.setStyleSheet("background-color: #319795; font-size: 13px; font-weight: bold; padding: 7px;")
        btn_parse.clicked.connect(self.parse_and_build_table)
        rb_layout.addWidget(btn_parse)

        box_tbl = QGroupBox("3. Зіставлені атоми та координати")
        bt_layout = QVBoxLayout(box_tbl)

        self.table_atoms = QTableWidget()
        self.table_atoms.setColumnCount(5)
        self.table_atoms.setHorizontalHeaderLabels(["Мітка", "δ (ppm)", "X (Å)", "Y (Å)", "Z (Å)"])
        self.table_atoms.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bt_layout.addWidget(self.table_atoms)

        row_btns = QHBoxLayout()
        btn_add_r = QPushButton("➕ Додати рядок")
        btn_add_r.clicked.connect(self.add_table_row)
        row_btns.addWidget(btn_add_r)

        btn_del_r = QPushButton("➖ Видалити рядок")
        btn_del_r.clicked.connect(self.delete_table_row)
        row_btns.addWidget(btn_del_r)

        btn_group_ch3 = QPushButton("🔗 Згрупувати виділені у CH3")
        btn_group_ch3.setStyleSheet("background-color: #805ad5;")
        btn_group_ch3.clicked.connect(self.group_selected_to_ch3)
        row_btns.addWidget(btn_group_ch3)

        row_btns.addStretch()
        bt_layout.addLayout(row_btns)
        rb_layout.addWidget(box_tbl)

        btn_save_mol = QPushButton("💾 ЗБЕРЕГТИ ТА РОЗРАХУВАТИ NOESY СПЕКТР")
        btn_save_mol.setStyleSheet("background-color: #38a169; font-size: 14px; font-weight: bold; padding: 10px;")
        btn_save_mol.clicked.connect(self.save_and_calculate_current)
        rb_layout.addWidget(btn_save_mol)

        split_data.addWidget(right_box)
        split_data.setSizes([260, 1100])
        self.tabs.addTab(tab_data, "📝 1. Введення геометрій та зсувів")

        # -------------------------------------------------------------
        # ВКЛАДКА 2: 2D NOESY З 1D ПРОЕКЦІЯМИ ТА ВІДСІКАННЯМ NOE
        # -------------------------------------------------------------
        tab_single = QWidget()
        ts_layout = QHBoxLayout(tab_single)
        split_single = QSplitter(Qt.Orientation.Horizontal)
        ts_layout.addWidget(split_single)

        p_ctrl = QWidget()
        pc_vbox = QVBoxLayout(p_ctrl)
        pc_vbox.setContentsMargins(0, 0, 8, 0)

        sel_grp = QGroupBox("Молекула")
        sg_l = QVBoxLayout(sel_grp)
        self.combo_single_mol = QComboBox()
        self.combo_single_mol.currentIndexChanged.connect(self.on_single_mol_combo_changed)
        sg_l.addWidget(self.combo_single_mol)
        pc_vbox.addWidget(sel_grp)

        param_grp = QGroupBox("Параметри спектрометра")
        pg_grid = QGridLayout(param_grp)

        pg_grid.addWidget(QLabel("Час змішування (τₘ):"), 0, 0)
        self.lbl_tau_m = QLabel("500 ms")
        self.lbl_tau_m.setStyleSheet("color: #63b3ed; font-weight: bold;")
        pg_grid.addWidget(self.lbl_tau_m, 0, 1)
        self.sl_tau_m = QSlider(Qt.Orientation.Horizontal)
        self.sl_tau_m.setRange(50, 1500)
        self.sl_tau_m.setValue(500)
        self.sl_tau_m.valueChanged.connect(self.on_params_changed)
        pg_grid.addWidget(self.sl_tau_m, 1, 0, 1, 2)

        pg_grid.addWidget(QLabel("Час кореляції (τ_c):"), 2, 0)
        self.lbl_tau_c = QLabel("80 ps")
        self.lbl_tau_c.setStyleSheet("color: #63b3ed; font-weight: bold;")
        pg_grid.addWidget(self.lbl_tau_c, 2, 1)
        self.sl_tau_c = QSlider(Qt.Orientation.Horizontal)
        self.sl_tau_c.setRange(20, 300)
        self.sl_tau_c.setValue(80)
        self.sl_tau_c.valueChanged.connect(self.on_params_changed)
        pg_grid.addWidget(self.sl_tau_c, 3, 0, 1, 2)

        pg_grid.addWidget(QLabel("Частота B₀:"), 4, 0)
        self.cb_freq = QComboBox()
        self.cb_freq.addItems(["400 MHz", "500 MHz", "600 MHz", "800 MHz"])
        self.cb_freq.setCurrentText("600 MHz")
        self.cb_freq.currentIndexChanged.connect(self.on_params_changed)
        pg_grid.addWidget(self.cb_freq, 4, 1)

        pg_grid.addWidget(QLabel("FWHM:"), 5, 0)
        self.lbl_fwhm = QLabel("0.035 ppm")
        self.lbl_fwhm.setStyleSheet("color: #63b3ed; font-weight: bold;")
        pg_grid.addWidget(self.lbl_fwhm, 5, 1)
        self.sl_fwhm = QSlider(Qt.Orientation.Horizontal)
        self.sl_fwhm.setRange(15, 80)
        self.sl_fwhm.setValue(35)
        self.sl_fwhm.valueChanged.connect(self.on_params_changed)
        pg_grid.addWidget(self.sl_fwhm, 6, 0, 1, 2)

        pc_vbox.addWidget(param_grp)

        # Блок візуалізації та порогу відсікання
        disp_grp = QGroupBox("Візуалізація та відсікання")
        dg_l = QVBoxLayout(disp_grp)

        # ПОРІГ ВІДСІКАННЯ NOE
        d_grid = QGridLayout()
        d_grid.addWidget(QLabel("Відсікати NOE менше за:"), 0, 0)
        self.lbl_cutoff = QLabel("0.10 %")
        self.lbl_cutoff.setStyleSheet("color: #ecc94b; font-weight: bold; font-size: 13px;")
        d_grid.addWidget(self.lbl_cutoff, 0, 1)

        self.sl_cutoff = QSlider(Qt.Orientation.Horizontal)
        self.sl_cutoff.setRange(0, 300)  # від 0.00% до 3.00%
        self.sl_cutoff.setValue(10)  # за замовчуванням 0.10%
        self.sl_cutoff.valueChanged.connect(self.on_params_changed)
        d_grid.addWidget(self.sl_cutoff, 1, 0, 1, 2)
        dg_l.addLayout(d_grid)

        self.chk_suppress = QCheckBox("Придушити діагональ")
        self.chk_suppress.stateChanged.connect(self.on_params_changed)
        dg_l.addWidget(self.chk_suppress)

        self.chk_show_1d = QCheckBox("Показувати 1D проекції на осях")
        self.chk_show_1d.setChecked(True)
        self.chk_show_1d.stateChanged.connect(self.on_params_changed)
        dg_l.addWidget(self.chk_show_1d)

        h_cm = QHBoxLayout()
        h_cm.addWidget(QLabel("Палітра:"))
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(["Blues_r", "viridis", "inferno", "coolwarm", "mako_r"])
        self.cb_cmap.currentIndexChanged.connect(self.on_params_changed)
        h_cm.addWidget(self.cb_cmap)
        dg_l.addLayout(h_cm)
        pc_vbox.addWidget(disp_grp)

        pc_vbox.addStretch()
        p_ctrl.setMinimumWidth(280)
        split_single.addWidget(p_ctrl)

        # Графік спектра
        center_spec = QWidget()
        cs_l = QVBoxLayout(center_spec)
        self.fig_single = Figure(figsize=(7, 7), facecolor='#16171d')
        self.canvas_single = FigureCanvas(self.fig_single)
        self.tb_single = NavigationToolbar(self.canvas_single, self)
        cs_l.addWidget(self.tb_single)
        cs_l.addWidget(self.canvas_single)
        split_single.addWidget(center_spec)

        # Таблиця крос-піків
        right_tbl = QWidget()
        rt_l = QVBoxLayout(right_tbl)
        self.rt_grp = QGroupBox(f"Крос-піки (NOE ≥ {self.noe_cutoff_pct:.2f}%)")
        rg_l = QVBoxLayout(self.rt_grp)
        self.table_single_noesy = QTableWidget()
        self.table_single_noesy.setColumnCount(4)
        self.table_single_noesy.setHorizontalHeaderLabels(["Протони", "δ₁ - δ₂ (ppm)", "r (Å)", "NOE (%)"])
        self.table_single_noesy.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_single_noesy.verticalHeader().setVisible(False)
        rg_l.addWidget(self.table_single_noesy)
        rt_l.addWidget(self.rt_grp)
        right_tbl.setMinimumWidth(320)
        split_single.addWidget(right_tbl)

        split_single.setSizes([280, 740, 360])
        self.tabs.addTab(tab_single, "📊 2. 2D NOESY Спектр (+1D Проекції)")

        # -------------------------------------------------------------
        # ВКЛАДКА 3: ПОРІВНЯННЯ МОЛЕКУЛ
        # -------------------------------------------------------------
        tab_compare = QWidget()
        tc_layout = QVBoxLayout(tab_compare)

        top_cmp = QHBoxLayout()
        top_cmp.addWidget(QLabel("Молекула А:"))
        self.cb_cmp_a = QComboBox()
        self.cb_cmp_a.currentIndexChanged.connect(self.update_comparison_view)
        top_cmp.addWidget(self.cb_cmp_a)

        top_cmp.addSpacing(30)
        top_cmp.addWidget(QLabel("Молекула Б:"))
        self.cb_cmp_b = QComboBox()
        self.cb_cmp_b.currentIndexChanged.connect(self.update_comparison_view)
        top_cmp.addWidget(self.cb_cmp_b)

        btn_refresh_cmp = QPushButton("🔄 Оновити порівняння")
        btn_refresh_cmp.clicked.connect(self.update_comparison_view)
        top_cmp.addWidget(btn_refresh_cmp)
        top_cmp.addStretch()
        tc_layout.addLayout(top_cmp)

        spec_split = QSplitter(Qt.Orientation.Horizontal)

        w_left = QWidget()
        wl_l = QVBoxLayout(w_left)
        self.fig_cmp_a = Figure(figsize=(5, 5), facecolor='#16171d')
        self.canvas_cmp_a = FigureCanvas(self.fig_cmp_a)
        self.ax_cmp_a = self.fig_cmp_a.add_subplot(111)
        wl_l.addWidget(self.canvas_cmp_a)
        spec_split.addWidget(w_left)

        w_right = QWidget()
        wr_l = QVBoxLayout(w_right)
        self.fig_cmp_b = Figure(figsize=(5, 5), facecolor='#16171d')
        self.canvas_cmp_b = FigureCanvas(self.fig_cmp_b)
        self.ax_cmp_b = self.fig_cmp_b.add_subplot(111)
        wr_l.addWidget(self.canvas_cmp_b)
        spec_split.addWidget(w_right)

        tc_layout.addWidget(spec_split, stretch=3)

        diff_grp = QGroupBox("Діагностичні різниці просторових контактів (Δr > 0.8 Å)")
        dg_layout = QVBoxLayout(diff_grp)
        self.table_diff = QTableWidget()
        self.table_diff.setColumnCount(5)
        self.table_diff.setHorizontalHeaderLabels([
            "Пара протонів", "Відстань в А (Å)", "Відстань в Б (Å)", "Різниця |Δr| (Å)", "Діагностичний висновок"
        ])
        self.table_diff.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_diff.verticalHeader().setVisible(False)
        dg_layout.addWidget(self.table_diff)
        tc_layout.addWidget(diff_grp, stretch=2)

        self.tabs.addTab(tab_compare, "⚖️ 3. Порівняння молекул")

    # -------------------------------------------------------------------------
    # ЗБЕРЕЖЕННЯ ТА ВІДКРИТТЯ СЕСІЇ (.JSON)
    # -------------------------------------------------------------------------
    def save_session_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти сесію проекту", "", "NOESY Session (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"

        if self.current_mol_idx >= 0 and self.current_mol_idx < len(self.molecules):
            self.save_table_to_active_mol()

        session_data = {
            "version": "1.3",
            "molecules": [m.to_dict() for m in self.molecules],
            "current_index": self.current_mol_idx,
            "params": {
                "tau_m": self.tau_m,
                "tau_c_ps": self.tau_c_ps,
                "freq_mhz": self.freq_mhz,
                "fwhm": self.fwhm,
                "suppress_diag": self.suppress_diag,
                "noe_cutoff_pct": self.noe_cutoff_pct,
                "show_1d": self.chk_show_1d.isChecked(),
                "cmap": self.cmap_name
            }
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Успіх", f"Сесію успішно збережено у:\n{os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти сесію: {e}")

    def load_session_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Відкрити файл сесії", "", "NOESY Session (*.json);;All Files (*)")
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            mol_list = data.get("molecules", [])
            if not mol_list:
                QMessageBox.warning(self, "Помилка", "Файл сесії не містить даних молекул.")
                return

            self.molecules = [MoleculeData.from_dict(m) for m in mol_list]
            self.list_molecules.blockSignals(True)
            self.list_molecules.clear()
            for m in self.molecules:
                self.list_molecules.addItem(m.name)
            self.list_molecules.blockSignals(False)

            cur_idx = min(data.get("current_index", 0), len(self.molecules) - 1)
            self.current_mol_idx = cur_idx
            self.list_molecules.setCurrentRow(cur_idx)

            params = data.get("params", {})
            self.tau_m = params.get("tau_m", 0.500)
            self.sl_tau_m.setValue(int(self.tau_m * 1000))
            self.lbl_tau_m.setText(f"{int(self.tau_m * 1000)} ms")

            self.tau_c_ps = params.get("tau_c_ps", 80.0)
            self.sl_tau_c.setValue(int(self.tau_c_ps))
            self.lbl_tau_c.setText(f"{int(self.tau_c_ps)} ps")

            self.freq_mhz = params.get("freq_mhz", 600.0)
            self.cb_freq.setCurrentText(f"{int(self.freq_mhz)} MHz")

            self.fwhm = params.get("fwhm", 0.035)
            self.sl_fwhm.setValue(int(self.fwhm * 1000))
            self.lbl_fwhm.setText(f"{self.fwhm:.3f} ppm")

            self.suppress_diag = params.get("suppress_diag", False)
            self.chk_suppress.setChecked(self.suppress_diag)

            self.noe_cutoff_pct = params.get("noe_cutoff_pct", 0.10)
            self.sl_cutoff.setValue(int(self.noe_cutoff_pct * 100))
            self.lbl_cutoff.setText(f"{self.noe_cutoff_pct:.2f} %")

            self.chk_show_1d.setChecked(params.get("show_1d", True))

            self.cmap_name = params.get("cmap", "Blues_r")
            self.cb_cmap.setCurrentText(self.cmap_name)

            self.sync_combos()
            self.on_molecule_selected(cur_idx)
            self.update_single_simulation()

            QMessageBox.information(self, "Сесію відновлено", f"Завантажено {len(self.molecules)} молекул.")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося прочитати файл сесії: {e}")

    # -------------------------------------------------------------------------
    # УПРАВЛІННЯ МОЛЕКУЛАМИ
    # -------------------------------------------------------------------------
    def add_new_molecule(self, name=None):
        if name is None:
            name = f"Молекула {len(self.molecules) + 1}"
        mol = MoleculeData(name)
        self.molecules.append(mol)
        self.list_molecules.addItem(name)
        self.list_molecules.setCurrentRow(len(self.molecules) - 1)
        self.sync_combos()

    def duplicate_current_molecule(self):
        if self.current_mol_idx < 0:
            return
        curr = self.molecules[self.current_mol_idx]
        new_mol = MoleculeData(f"{curr.name} (Копія)")
        new_mol.labels = list(curr.labels)
        new_mol.shifts = np.copy(curr.shifts)
        new_mol.coords = np.copy(curr.coords)
        self.molecules.append(new_mol)
        self.list_molecules.addItem(new_mol.name)
        self.list_molecules.setCurrentRow(len(self.molecules) - 1)
        self.sync_combos()

    def delete_current_molecule(self):
        if len(self.molecules) <= 1:
            QMessageBox.warning(self, "Попередження", "Має залишитися хоча б одна молекула в списку.")
            return
        row = self.current_mol_idx
        self.molecules.pop(row)
        self.list_molecules.takeItem(row)
        self.current_mol_idx = max(0, row - 1)
        self.list_molecules.setCurrentRow(self.current_mol_idx)
        self.sync_combos()

    def sync_combos(self):
        names = [m.name for m in self.molecules]
        for cb in [self.combo_single_mol, self.cb_cmp_a, self.cb_cmp_b]:
            curr = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            idx = cb.findText(curr)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.blockSignals(False)

        if len(names) >= 2 and self.cb_cmp_a.currentIndex() == self.cb_cmp_b.currentIndex():
            self.cb_cmp_b.setCurrentIndex(1)

    def on_molecule_selected(self, row):
        if row < 0 or row >= len(self.molecules):
            return
        self.current_mol_idx = row
        mol = self.molecules[row]
        self.edit_mol_name.blockSignals(True)
        self.edit_mol_name.setText(mol.name)
        self.edit_mol_name.blockSignals(False)
        self.populate_editor_table(mol.labels, mol.shifts, mol.coords)

    def on_molecule_name_changed(self, text):
        if self.current_mol_idx >= 0:
            self.molecules[self.current_mol_idx].name = text
            item = self.list_molecules.item(self.current_mol_idx)
            if item:
                item.setText(text)
            self.sync_combos()

    def save_table_to_active_mol(self):
        rows = self.table_atoms.rowCount()
        labels, shifts, coords = [], [], []
        for r in range(rows):
            lbl_item = self.table_atoms.item(r, 0)
            s_item = self.table_atoms.item(r, 1)
            x_item = self.table_atoms.item(r, 2)
            y_item = self.table_atoms.item(r, 3)
            z_item = self.table_atoms.item(r, 4)
            if lbl_item and s_item and x_item and y_item and z_item:
                labels.append(lbl_item.text().strip())
                shifts.append(float(s_item.text()))
                coords.append([float(x_item.text()), float(y_item.text()), float(z_item.text())])

        mol = self.molecules[self.current_mol_idx]
        mol.labels = labels
        mol.shifts = np.array(shifts)
        mol.coords = np.array(coords)

    # -------------------------------------------------------------------------
    # РЕДАКТОР ДАНИХ
    # -------------------------------------------------------------------------
    def load_xyz_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Виберіть .xyz файл", "", "XYZ Files (*.xyz);;All Files (*)")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.txt_xyz.setPlainText(content)
            coords = parse_xyz_only_hydrogens(content)
            QMessageBox.information(self, "Файл прочитано", f"Знайдено {len(coords)} атомів гідрогену (H).")

    def parse_and_build_table(self):
        nmr_txt = self.txt_shifts.toPlainText().strip()
        xyz_txt = self.txt_xyz.toPlainText().strip()

        if not nmr_txt and not xyz_txt:
            QMessageBox.warning(self, "Помилка", "Введіть текст зсувів та/або координати XYZ.")
            return

        shifts, labels = parse_nmr_text_with_multiplicity(nmr_txt) if nmr_txt else ([], [])
        coords = parse_xyz_only_hydrogens(xyz_txt) if xyz_txt else np.empty((0, 3))

        N_s = len(shifts)
        N_c = len(coords)

        if N_s != N_c and N_s > 0 and N_c > 0:
            QMessageBox.information(
                self, "Зіставлення",
                f"Розпізнано:\n• Протонів у спектрі (з урахуванням 3H): {N_s}\n• Атомів H у координатах: {N_c}\n\n"
                "Дані внесено в таблицю. Перевірте зіставлення."
            )

        N = max(N_s, N_c)
        self.table_atoms.setRowCount(N)
        for i in range(N):
            l = labels[i] if i < N_s else f"H{i + 1}"
            s = f"{shifts[i]:.2f}" if i < N_s else "2.00"
            x = f"{coords[i, 0]:.4f}" if i < N_c else "0.0000"
            y = f"{coords[i, 1]:.4f}" if i < N_c else "0.0000"
            z = f"{coords[i, 2]:.4f}" if i < N_c else "0.0000"

            self.table_atoms.setItem(i, 0, QTableWidgetItem(l))
            self.table_atoms.setItem(i, 1, QTableWidgetItem(s))
            self.table_atoms.setItem(i, 2, QTableWidgetItem(x))
            self.table_atoms.setItem(i, 3, QTableWidgetItem(y))
            self.table_atoms.setItem(i, 4, QTableWidgetItem(z))

    def group_selected_to_ch3(self):
        sel_ranges = self.table_atoms.selectedRanges()
        rows = []
        for r in sel_ranges:
            rows.extend(range(r.topRow(), r.bottomRow() + 1))
        rows = sorted(list(set(rows)))

        if len(rows) < 2:
            QMessageBox.information(self, "Підказка",
                                    "Виділіть рядки (наприклад, 3 атоми метилу), які належать до однієї CH3 групи.")
            return

        first_shift = self.table_atoms.item(rows[0], 1).text()
        for idx, row in enumerate(rows):
            self.table_atoms.setItem(row, 0, QTableWidgetItem(f"Me(3H)_{chr(97 + idx)}"))
            self.table_atoms.setItem(row, 1, QTableWidgetItem(first_shift))

        QMessageBox.information(self, "CH3 Згруповано", f"Атомам призначено однаковий зсув δ = {first_shift} ppm.")

    def add_table_row(self):
        row = self.table_atoms.rowCount()
        self.table_atoms.insertRow(row)
        self.table_atoms.setItem(row, 0, QTableWidgetItem(f"H{row + 1}"))
        self.table_atoms.setItem(row, 1, QTableWidgetItem("2.00"))
        self.table_atoms.setItem(row, 2, QTableWidgetItem("0.0000"))
        self.table_atoms.setItem(row, 3, QTableWidgetItem("0.0000"))
        self.table_atoms.setItem(row, 4, QTableWidgetItem("0.0000"))

    def delete_table_row(self):
        curr = self.table_atoms.currentRow()
        if curr >= 0:
            self.table_atoms.removeRow(curr)

    def populate_editor_table(self, labels, shifts, coords):
        N = len(shifts)
        self.table_atoms.setRowCount(N)
        for i in range(N):
            self.table_atoms.setItem(i, 0, QTableWidgetItem(str(labels[i])))
            self.table_atoms.setItem(i, 1, QTableWidgetItem(f"{shifts[i]:.2f}"))
            self.table_atoms.setItem(i, 2, QTableWidgetItem(f"{coords[i, 0]:.4f}"))
            self.table_atoms.setItem(i, 3, QTableWidgetItem(f"{coords[i, 1]:.4f}"))
            self.table_atoms.setItem(i, 4, QTableWidgetItem(f"{coords[i, 2]:.4f}"))

    def save_and_calculate_current(self):
        if self.current_mol_idx < 0:
            return

        rows = self.table_atoms.rowCount()
        if rows < 2:
            QMessageBox.warning(self, "Помилка", "Потрібно щонайменше 2 атоми для розрахунку NOESY!")
            return

        try:
            self.save_table_to_active_mol()
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Перевірте коректність чисел: {e}")
            return

        self.combo_single_mol.setCurrentIndex(self.current_mol_idx)
        self.update_single_simulation()
        self.tabs.setCurrentIndex(1)

    # -------------------------------------------------------------------------
    # СИМУЛЯЦІЯ ТА ВІДМАЛЬОВУВАННЯ
    # -------------------------------------------------------------------------
    def on_params_changed(self):
        self.tau_m = self.sl_tau_m.value() / 1000.0
        self.lbl_tau_m.setText(f"{self.sl_tau_m.value()} ms")

        self.tau_c_ps = float(self.sl_tau_c.value())
        self.lbl_tau_c.setText(f"{int(self.tau_c_ps)} ps")

        self.freq_mhz = float(self.cb_freq.currentText().replace(" MHz", ""))
        self.fwhm = self.sl_fwhm.value() / 1000.0
        self.lbl_fwhm.setText(f"{self.fwhm:.3f} ppm")

        self.suppress_diag = self.chk_suppress.isChecked()

        # Оновлення порогу відсікання NOE
        self.noe_cutoff_pct = self.sl_cutoff.value() / 100.0
        self.lbl_cutoff.setText(f"{self.noe_cutoff_pct:.2f} %")
        self.rt_grp.setTitle(f"Крос-піки (NOE ≥ {self.noe_cutoff_pct:.2f}%)")

        self.cmap_name = self.cb_cmap.currentText()

        self.update_single_simulation()

    def on_single_mol_combo_changed(self, idx):
        if idx >= 0:
            self.update_single_simulation()

    def update_single_simulation(self):
        idx = self.combo_single_mol.currentIndex()
        if idx < 0 or idx >= len(self.molecules):
            return

        mol = self.molecules[idx]
        if len(mol.shifts) < 2:
            self.fig_single.clear()
            ax = self.fig_single.add_subplot(111)
            ax.set_facecolor("#16171d")
            ax.text(0.5, 0.5, "Недостатньо даних для розрахунку.\nВведіть зсуви та геометрію на Вкладці 1.",
                    color="#a0aec0", ha='center', va='center')
            self.canvas_single.draw()
            return

        intensities, distances = calculate_noesy(
            mol.shifts, mol.coords,
            tau_m=self.tau_m, tau_c_ps=self.tau_c_ps, freq_mhz=self.freq_mhz
        )

        self.populate_noesy_table(self.table_single_noesy, intensities, distances, mol.shifts, mol.labels)
        self.plot_spectrum_with_projections(intensities, mol.shifts, mol.labels, mol.name)

    def populate_noesy_table(self, table_widget, intensities, distances, shifts, labels):
        N = len(shifts)
        diag_ref = np.median(np.diag(intensities)) if N > 0 else 1.0
        peaks = []

        for i in range(N):
            for j in range(i + 1, N):
                r = distances[i, j]
                noe_pct = abs(intensities[i, j] / diag_ref) * 100.0
                # ВІДСІКАННЯ: додаємо лише ті контакти, які перевищують поріг
                if r < 3.8 and noe_pct >= self.noe_cutoff_pct:
                    peaks.append((labels[i], labels[j], shifts[i], shifts[j], r, noe_pct))

        peaks.sort(key=lambda x: x[4])  # сортуємо за найкоротшою відстанню r
        table_widget.setRowCount(len(peaks))

        for row, (l1, l2, s1, s2, r, noe) in enumerate(peaks):
            it_pair = QTableWidgetItem(f"{l1} — {l2}")
            it_shift = QTableWidgetItem(f"{s1:.2f} × {s2:.2f}")
            it_r = QTableWidgetItem(f"{r:.2f}")
            it_noe = QTableWidgetItem(f"{noe:.2f}%")

            for it in [it_pair, it_shift, it_r, it_noe]:
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            table_widget.setItem(row, 0, it_pair)
            table_widget.setItem(row, 1, it_shift)
            table_widget.setItem(row, 2, it_r)
            table_widget.setItem(row, 3, it_noe)

    def plot_spectrum_with_projections(self, intensities, shifts, labels, title):
        self.fig_single.clear()

        N = len(shifts)
        ppm_min = max(0.0, np.min(shifts) - 0.5)
        ppm_max = np.max(shifts) + 0.5

        grid_res = 280
        ppm_grid = np.linspace(ppm_min, ppm_max, grid_res)

        spec_1d = np.zeros_like(ppm_grid)
        for s in shifts:
            spec_1d += np.exp(-4.0 * np.log(2) * ((ppm_grid - s) ** 2) / (self.fwhm ** 2))
        max_1d = np.max(spec_1d) if np.max(spec_1d) > 0 else 1.0

        show_projections = self.chk_show_1d.isChecked()

        if show_projections:
            gs = self.fig_single.add_gridspec(
                2, 2,
                width_ratios=[1.1, 5.5],
                height_ratios=[1.1, 5.5],
                wspace=0.03, hspace=0.03,
                left=0.08, right=0.96, bottom=0.08, top=0.93
            )

            ax_empty = self.fig_single.add_subplot(gs[0, 0])
            ax_empty.axis('off')

            ax_top = self.fig_single.add_subplot(gs[0, 1])
            ax_left = self.fig_single.add_subplot(gs[1, 0])
            ax_main = self.fig_single.add_subplot(gs[1, 1], sharex=ax_top, sharey=ax_left)

            # Верхня 1D проекція
            ax_top.set_facecolor("#16171d")
            ax_top.plot(ppm_grid, spec_1d, color="#63b3ed", lw=1.2)
            ax_top.fill_between(ppm_grid, 0, spec_1d, color="#3182ce", alpha=0.25)
            ax_top.set_ylim(0, max_1d * 1.15)
            ax_top.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax_top.set_title(f"{title} (600 MHz, τₘ={int(self.tau_m * 1000)} ms)", color="#63b3ed", fontsize=11,
                             fontweight='bold', pad=8)
            for sp in ax_top.spines.values():
                sp.set_color("#2d3748")

            # Ліва 1D проекція
            ax_left.set_facecolor("#16171d")
            ax_left.plot(spec_1d, ppm_grid, color="#63b3ed", lw=1.2)
            ax_left.fill_betweenx(ppm_grid, 0, spec_1d, color="#3182ce", alpha=0.25)
            ax_left.set_xlim(max_1d * 1.15, 0)
            ax_left.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for sp in ax_left.spines.values():
                sp.set_color("#2d3748")
        else:
            ax_main = self.fig_single.add_subplot(111)
            ax_main.set_title(f"{title} (600 MHz, τₘ={int(self.tau_m * 1000)} ms)", color="#63b3ed", fontsize=11,
                              fontweight='bold')

        # 2D Контурний спектр
        ax_main.set_facecolor("#111216")
        X, Y = np.meshgrid(ppm_grid, ppm_grid)
        Z = np.zeros_like(X)

        diag_ref = np.median(np.diag(intensities)) if N > 0 else 1.0

        for i in range(N):
            for j in range(N):
                if i == j:
                    amp = 0.0 if self.suppress_diag else intensities[i, j]
                else:
                    noe_pct = abs(intensities[i, j] / diag_ref) * 100.0
                    # ВІДСІКАННЯ: якщо сигнал менший за поріг, він повністю зануляється
                    if noe_pct < self.noe_cutoff_pct:
                        amp = 0.0
                    else:
                        amp = intensities[i, j]

                if amp != 0.0:
                    Z += amp * np.exp(-4.0 * np.log(2) * (
                            ((X - shifts[i]) ** 2 + (Y - shifts[j]) ** 2) / (self.fwhm ** 2)
                    ))

        Z_abs = np.abs(Z)
        max_val = np.max(Z_abs)
        if max_val > 0:
            threshold = 0.008 if self.suppress_diag else 0.015
            levels = np.geomspace(max_val * threshold, max_val * 0.95, 16)
            try:
                ax_main.contour(X, Y, Z_abs, levels=levels, cmap=self.cmap_name, linewidths=0.9)
            except Exception:
                pass

        ax_main.plot([ppm_min, ppm_max], [ppm_min, ppm_max], color="#4a5568", linestyle="--", linewidth=0.8, alpha=0.7)

        unique_shifts = {}
        for s, l in zip(shifts, labels):
            s_round = round(s, 2)
            if s_round not in unique_shifts:
                unique_shifts[s_round] = l
            else:
                unique_shifts[s_round] += f",{l}"

        for s_round, l_comb in unique_shifts.items():
            ax_main.text(s_round, s_round, f" {l_comb}", color="#cbd5e0", fontsize=7.5, verticalalignment='bottom')

        ax_main.set_xlim(ppm_max, ppm_min)
        ax_main.set_ylim(ppm_max, ppm_min)
        ax_main.set_xlabel("F2 (¹H) [ppm]", color="#e2e8f0", fontsize=10, labelpad=6)
        ax_main.set_ylabel("F1 (¹H) [ppm]", color="#e2e8f0", fontsize=10, labelpad=6)
        ax_main.tick_params(colors="#a0aec0", labelsize=8)
        for spine in ax_main.spines.values():
            spine.set_color("#2d3748")

        self.canvas_single.draw()

    # -------------------------------------------------------------------------
    # ВКЛАДКА 3: ПОРІВНЯННЯ ДВОХ МОЛЕКУЛ
    # -------------------------------------------------------------------------
    def update_comparison_view(self):
        idx_a = self.cb_cmp_a.currentIndex()
        idx_b = self.cb_cmp_b.currentIndex()

        if idx_a < 0 or idx_b < 0 or idx_a >= len(self.molecules) or idx_b >= len(self.molecules):
            return

        mol_a = self.molecules[idx_a]
        mol_b = self.molecules[idx_b]

        int_a, dist_a = calculate_noesy(mol_a.shifts, mol_a.coords, self.tau_m, self.tau_c_ps, self.freq_mhz)
        int_b, dist_b = calculate_noesy(mol_b.shifts, mol_b.coords, self.tau_m, self.tau_c_ps, self.freq_mhz)

        self.plot_spectrum_simple(self.ax_cmp_a, self.canvas_cmp_a, int_a, mol_a.shifts, mol_a.labels,
                                  f"Мол А: {mol_a.name}")
        self.plot_spectrum_simple(self.ax_cmp_b, self.canvas_cmp_b, int_b, mol_b.shifts, mol_b.labels,
                                  f"Мол Б: {mol_b.name}")

        diff_list = []
        dict_a = {}
        for i in range(len(mol_a.labels)):
            for j in range(i + 1, len(mol_a.labels)):
                pair = tuple(sorted([mol_a.labels[i], mol_a.labels[j]]))
                dict_a[pair] = dist_a[i, j]

        dict_b = {}
        for i in range(len(mol_b.labels)):
            for j in range(i + 1, len(mol_b.labels)):
                pair = tuple(sorted([mol_b.labels[i], mol_b.labels[j]]))
                dict_b[pair] = dist_b[i, j]

        common_pairs = set(dict_a.keys()).intersection(set(dict_b.keys()))
        for p in common_pairs:
            r_a = dict_a[p]
            r_b = dict_b[p]
            delta_r = abs(r_a - r_b)
            if delta_r >= 0.7:
                diff_list.append((f"{p[0]} — {p[1]}", r_a, r_b, delta_r))

        diff_list.sort(key=lambda x: x[3], reverse=True)
        self.table_diff.setRowCount(len(diff_list))

        for row, (pair_name, ra, rb, dra) in enumerate(diff_list):
            it_pair = QTableWidgetItem(pair_name)
            it_ra = QTableWidgetItem(f"{ra:.2f}")
            it_rb = QTableWidgetItem(f"{rb:.2f}")
            it_dra = QTableWidgetItem(f"{dra:.2f}")

            conclusion = ""
            if ra < 2.8 and rb > 3.8:
                conclusion = "Сильний NOE тільки в Мол. А"
                it_pair.setForeground(Qt.GlobalColor.green)
            elif rb < 2.8 and ra > 3.8:
                conclusion = "Сильний NOE тільки в Мол. Б"
                it_pair.setForeground(Qt.GlobalColor.cyan)
            else:
                conclusion = "Різниця інтенсивності"

            it_conc = QTableWidgetItem(conclusion)

            for it in [it_pair, it_ra, it_rb, it_dra, it_conc]:
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table_diff.setItem(row, 0, it_pair)
            self.table_diff.setItem(row, 1, it_ra)
            self.table_diff.setItem(row, 2, it_rb)
            self.table_diff.setItem(row, 3, it_dra)
            self.table_diff.setItem(row, 4, it_conc)

    def plot_spectrum_simple(self, ax, canvas, intensities, shifts, labels, title):
        ax.clear()
        ax.set_facecolor("#111216")

        N = len(shifts)
        if N < 2:
            canvas.draw()
            return

        ppm_min = max(0.0, np.min(shifts) - 0.5)
        ppm_max = np.max(shifts) + 0.5

        grid_res = 220
        ppm_grid = np.linspace(ppm_min, ppm_max, grid_res)
        X, Y = np.meshgrid(ppm_grid, ppm_grid)
        Z = np.zeros_like(X)

        diag_ref = np.median(np.diag(intensities)) if N > 0 else 1.0

        for i in range(N):
            for j in range(N):
                if i == j:
                    amp = 0.0 if self.suppress_diag else intensities[i, j]
                else:
                    noe_pct = abs(intensities[i, j] / diag_ref) * 100.0
                    if noe_pct < self.noe_cutoff_pct:
                        amp = 0.0
                    else:
                        amp = intensities[i, j]

                if amp != 0.0:
                    Z += amp * np.exp(-4.0 * np.log(2) * (
                            ((X - shifts[i]) ** 2 + (Y - shifts[j]) ** 2) / (self.fwhm ** 2)
                    ))

        Z_abs = np.abs(Z)
        max_val = np.max(Z_abs)
        if max_val > 0:
            threshold = 0.008 if self.suppress_diag else 0.015
            levels = np.geomspace(max_val * threshold, max_val * 0.95, 14)
            try:
                ax.contour(X, Y, Z_abs, levels=levels, cmap=self.cmap_name, linewidths=0.8)
            except Exception:
                pass

        ax.plot([ppm_min, ppm_max], [ppm_min, ppm_max], color="#4a5568", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlim(ppm_max, ppm_min)
        ax.set_ylim(ppm_max, ppm_min)
        ax.set_xlabel("F2 [ppm]", color="#e2e8f0", fontsize=9)
        ax.set_ylabel("F1 [ppm]", color="#e2e8f0", fontsize=9)
        ax.set_title(title, color="#63b3ed", fontsize=10, fontweight='bold')
        ax.tick_params(colors="#a0aec0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2d3748")

        canvas.draw()


# -----------------------------------------------------------------------------
# ЗАПУСК
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NOESYStudioPro()
    window.show()
    sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())