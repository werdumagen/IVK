# tab_setup.py
import json
from typing import Optional, List
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QSplitter, QMessageBox, QGroupBox,
                             QTabWidget, QFileDialog, QDoubleSpinBox, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal

from core import InteractiveGraphCanvas, DataParsers


class VisualizerTab(QWidget):
    # Сигнал для передачи данных во вторую вкладку
    requestMatching = pyqtSignal(dict, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mol = None
        self.last_graphs = {}
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель (ввод)
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        l_layout.setContentsMargins(5, 5, 5, 5)

        session_l = QHBoxLayout()
        self.btn_save = QPushButton("💾 Сохранить сессию")
        self.btn_load = QPushButton("📂 Загрузить сессию")
        self.btn_save.clicked.connect(self.save_session)
        self.btn_load.clicked.connect(self.load_session)
        session_l.addWidget(self.btn_save)
        session_l.addWidget(self.btn_load)
        l_layout.addLayout(session_l)

        self.tabs_in = QTabWidget()

        t1, t2, t3 = QWidget(), QWidget(), QWidget()
        self.txt_xyz = QTextEdit();
        QVBoxLayout(t1).addWidget(self.txt_xyz)
        self.txt_dft = QTextEdit();
        QVBoxLayout(t2).addWidget(self.txt_dft)

        t3_l = QVBoxLayout(t3)
        fbox = QGroupBox("Параметры фильтрации и J-Matching")
        fl = QHBoxLayout(fbox)
        self.sp_ctol = QDoubleSpinBox();
        self.sp_ctol.setValue(0.25)
        self.sp_jtol = QDoubleSpinBox();
        self.sp_jtol.setValue(0.95)
        self.sp_area = QDoubleSpinBox();
        self.sp_area.setValue(0.20)
        self.chk_art = QCheckBox("Без артефактов");
        self.chk_art.setChecked(True)
        fl.addWidget(QLabel("ΔC Tol:"));
        fl.addWidget(self.sp_ctol)
        fl.addWidget(QLabel("ΔJ Tol:"));
        fl.addWidget(self.sp_jtol)
        fl.addWidget(QLabel("Min Area:"));
        fl.addWidget(self.sp_area)
        fl.addWidget(self.chk_art)
        t3_l.addWidget(fbox)

        self.txt_1d = QTextEdit();
        self.txt_1d.setMaximumHeight(65)
        self.txt_hsqc = QTextEdit()
        self.txt_cosy = QTextEdit()
        t3_l.addWidget(QLabel("1D 1H NMR:"));
        t3_l.addWidget(self.txt_1d)
        t3_l.addWidget(QLabel("2D HSQC:"));
        t3_l.addWidget(self.txt_hsqc)
        t3_l.addWidget(QLabel("2D COSY:"));
        t3_l.addWidget(self.txt_cosy)

        self.tabs_in.addTab(t1, "1. XYZ")
        self.tabs_in.addTab(t2, "2. DFT")
        self.tabs_in.addTab(t3, "3. Эксперимент")
        l_layout.addWidget(self.tabs_in)

        self.btn_plot = QPushButton("⚡ Построить графы (COSY ∧ 1D J-Axiom)")
        self.btn_plot.setStyleSheet("background-color: #0288D1; color: white; font-weight: bold; padding: 11px;")
        self.btn_plot.clicked.connect(self.plot_all)
        l_layout.addWidget(self.btn_plot)

        self.btn_match = QPushButton("🔗 Передать данные и Найти Якоря")
        self.btn_match.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 9px;")
        self.btn_match.clicked.connect(self.launch_matching)
        l_layout.addWidget(self.btn_match)

        splitter.addWidget(left_w)

        # Правая панель (вывод графов)
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)
        self.tabs_out = QTabWidget()
        self.cv_xyz = InteractiveGraphCanvas()
        self.cv_dft = InteractiveGraphCanvas()
        self.cv_exp = InteractiveGraphCanvas()
        self.tabs_out.addTab(self.cv_xyz, "1. Структура (XYZ)")
        self.tabs_out.addTab(self.cv_dft, "2. DFT")
        self.tabs_out.addTab(self.cv_exp, "3. Эксперимент (J-Match)")

        self.table_nodes = QTableWidget()
        self.table_nodes.setColumnCount(5)
        self.table_nodes.setHorizontalHeaderLabels(["ID Узла", "Тип", "13C δ (ppm)", "1H δ (ppm)", "Константы J"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs_out.addTab(self.table_nodes, "📋 Узлы Exp")

        r_layout.addWidget(self.tabs_out)
        splitter.addWidget(right_w)
        splitter.setSizes([450, 600])
        layout.addWidget(splitter)

    def plot_all(self):
        try:
            self.mol, xyz_n, xyz_e = DataParsers.parse_xyz(self.txt_xyz.toPlainText().strip())
            self.cv_xyz.set_graph(xyz_n, xyz_e, False)

            dft_n, dft_e = DataParsers.parse_dft(self.txt_dft.toPlainText().strip())
            self.cv_dft.set_graph(dft_n, dft_e, True)

            exp_n, exp_e = DataParsers.parse_exp(
                self.txt_1d.toPlainText(), self.txt_hsqc.toPlainText(), self.txt_cosy.toPlainText(),
                self.sp_ctol.value(), self.sp_area.value(), self.sp_jtol.value(), self.chk_art.isChecked()
            )
            self.cv_exp.set_graph(exp_n, exp_e, True)

            self.table_nodes.setRowCount(len(exp_n))
            for i, (k, d) in enumerate(exp_n.items()):
                self.table_nodes.setItem(i, 0, QTableWidgetItem(k))
                self.table_nodes.setItem(i, 1, QTableWidgetItem(d.get('type', 'CH')))
                self.table_nodes.setItem(i, 2, QTableWidgetItem(f"{d.get('c', 0):.2f}"))
                self.table_nodes.setItem(i, 3, QTableWidgetItem(", ".join([f"{x:.2f}" for x in d['h_list']])))
                self.table_nodes.setItem(i, 4, QTableWidgetItem(", ".join([f"{x:.1f}" for x in d['j_vals']])))

            self.last_graphs = {
                "xyz_graph": {"nodes": xyz_n, "edges": [list(e) for e in xyz_e]},
                "dft_graph": {"nodes": dft_n, "edges": [list(e) for e in dft_e]},
                "exp_graph": {"nodes": exp_n, "edges": [list(e) for e in exp_e]}
            }
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Сбой построения:\n{e}")

    def launch_matching(self):
        if not self.last_graphs:
            self.plot_all()
        if self.last_graphs:
            self.requestMatching.emit(self.last_graphs, self.mol)

    def save_session(self):
        data = {
            "xyz": self.txt_xyz.toPlainText(), "dft": self.txt_dft.toPlainText(),
            "exp1d": self.txt_1d.toPlainText(), "hsqc": self.txt_hsqc.toPlainText(),
            "cosy": self.txt_cosy.toPlainText(),
            "settings": {"c_tol": self.sp_ctol.value(), "j_tol": self.sp_jtol.value(), "area": self.sp_area.value()}
        }
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", "", "JSON (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить", "", "JSON (*.json)")
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.txt_xyz.setPlainText(data.get("xyz", ""))
            self.txt_dft.setPlainText(data.get("dft", ""))
            self.txt_1d.setPlainText(data.get("exp1d", ""))
            self.txt_hsqc.setPlainText(data.get("hsqc", ""))
            self.txt_cosy.setPlainText(data.get("cosy", ""))
            st = data.get("settings", {})
            if "c_tol" in st: self.sp_ctol.setValue(st["c_tol"])
            if "j_tol" in st: self.sp_jtol.setValue(st["j_tol"])
            if "area" in st: self.sp_area.setValue(st["area"])