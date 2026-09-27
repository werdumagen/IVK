# tab_setup.py
import json
from typing import Optional, List, Dict
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QSplitter, QMessageBox, QGroupBox,
                             QTabWidget, QFileDialog, QDoubleSpinBox, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from core import InteractiveGraphCanvas, DataParsers, InteractiveMolLabel


class VisualizerTab(QWidget):
    requestMatching = pyqtSignal(dict, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mol = None
        self.last_graphs = {}
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

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

        # 1. XYZ
        t1 = QWidget()
        self.txt_xyz = QTextEdit()
        QVBoxLayout(t1).addWidget(self.txt_xyz)

        # 2. DFT + Экранирование + TMS
        t2 = QWidget()
        t2_l = QVBoxLayout(t2)

        tms_lay = QHBoxLayout()
        tms_lay.addWidget(QLabel("TMS 13C:"))
        self.sp_tms_c = QDoubleSpinBox()
        self.sp_tms_c.setRange(0, 300)
        self.sp_tms_c.setValue(188.1)
        tms_lay.addWidget(self.sp_tms_c)
        tms_lay.addWidget(QLabel("TMS 1H:"))
        self.sp_tms_h = QDoubleSpinBox()
        self.sp_tms_h.setRange(0, 50)
        self.sp_tms_h.setValue(31.8)
        tms_lay.addWidget(self.sp_tms_h)
        t2_l.addLayout(tms_lay)

        dft_split = QSplitter(Qt.Orientation.Vertical)

        gb_j = QGroupBox("Матрица J-констант (DFT)")
        l_j = QVBoxLayout(gb_j)
        self.txt_dft = QTextEdit()
        l_j.addWidget(self.txt_dft)
        dft_split.addWidget(gb_j)

        gb_s = QGroupBox("Спектр экранирования (Shielding)")
        l_s = QVBoxLayout(gb_s)
        self.txt_dft_shielding = QTextEdit()
        l_s.addWidget(self.txt_dft_shielding)
        dft_split.addWidget(gb_s)

        t2_l.addWidget(dft_split)

        # 3. Experiment
        t3 = QWidget()
        t3_l = QVBoxLayout(t3)
        fbox = QGroupBox("Параметры фильтрации и J-Matching")
        fl = QHBoxLayout(fbox)
        self.sp_ctol = QDoubleSpinBox()
        self.sp_ctol.setValue(0.25)
        self.sp_jtol = QDoubleSpinBox()
        self.sp_jtol.setValue(0.95)
        self.sp_area = QDoubleSpinBox()
        self.sp_area.setValue(0.20)
        self.chk_art = QCheckBox("Без артефактов")
        self.chk_art.setChecked(True)
        fl.addWidget(QLabel("ΔC Tol:"))
        fl.addWidget(self.sp_ctol)
        fl.addWidget(QLabel("ΔJ Tol:"))
        fl.addWidget(self.sp_jtol)
        fl.addWidget(QLabel("Min Area:"))
        fl.addWidget(self.sp_area)
        fl.addWidget(self.chk_art)
        t3_l.addWidget(fbox)

        self.txt_1d = QTextEdit()
        self.txt_1d.setMaximumHeight(65)
        self.txt_hsqc = QTextEdit()
        self.txt_cosy = QTextEdit()
        t3_l.addWidget(QLabel("1D 1H NMR:"))
        t3_l.addWidget(self.txt_1d)
        t3_l.addWidget(QLabel("2D HSQC:"))
        t3_l.addWidget(self.txt_hsqc)
        t3_l.addWidget(QLabel("2D COSY:"))
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

        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)

        # 2D Молекула на вкладке графов
        gb_mol_setup = QGroupBox("2D Структура (Оранжевый: атом, Зеленый: J-соседи DFT, Синий: связи)")
        gbl_mol_setup = QVBoxLayout(gb_mol_setup)
        self.mol_view = InteractiveMolLabel("2D структура отобразится после построения графов.")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 4px;")
        self.mol_view.setMinimumHeight(240)
        self.mol_view.atomClicked.connect(self.on_mol_atom_clicked)
        gbl_mol_setup.addWidget(self.mol_view)

        self.lbl_mol_info = QLabel("Кликните по атому или ноду для подсветки связности")
        self.lbl_mol_info.setStyleSheet("font-size: 11px; color: #1565C0; padding: 2px;")
        gbl_mol_setup.addWidget(self.lbl_mol_info)
        r_layout.addWidget(gb_mol_setup)

        self.tabs_out = QTabWidget()
        self.cv_xyz = InteractiveGraphCanvas()
        self.cv_dft = InteractiveGraphCanvas()
        self.cv_exp = InteractiveGraphCanvas()

        self.cv_xyz.nodeClicked.connect(self.on_graph_node_clicked)
        self.cv_dft.nodeClicked.connect(self.on_graph_node_clicked)
        self.cv_exp.nodeClicked.connect(self.on_graph_node_clicked)

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
        splitter.setSizes([450, 750])
        layout.addWidget(splitter)

    def plot_all(self):
        try:
            self.mol, xyz_n, xyz_e = DataParsers.parse_xyz(self.txt_xyz.toPlainText().strip())
            self.cv_xyz.set_graph(xyz_n, xyz_e, False)
            self._render_mol()

            dft_n, dft_e = DataParsers.parse_dft(self.txt_dft.toPlainText().strip())

            dft_shifts = DataParsers.parse_dft_shifts(
                self.txt_dft_shielding.toPlainText().strip(),
                self.sp_tms_c.value(),
                self.sp_tms_h.value()
            )
            for k, d in dft_n.items():
                c_idx = int(k.split('_')[1])
                d['c_shift'] = dft_shifts.get(c_idx, None)
                d['h_shifts'] = [dft_shifts.get(h, None) for h in d.get('H', [])]

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

    def highlight_atom_and_connections(self, c_idx: int):
        if not self.mol or c_idx >= self.mol.GetNumAtoms():
            return

        atom = self.mol.GetAtomWithIdx(c_idx)
        own_protons = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
        bonded_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() != 'H']

        dft_edges = self.last_graphs.get("dft_graph", {}).get("edges", [])
        dft_node_name = f"DFT_{c_idx}"
        j_partners = []
        j_partner_protons = []
        j_info_list = []

        for e in dft_edges:
            u, v = e[0], e[1]
            jv = e[2] if len(e) >= 3 else 0.0
            partner_key = None
            if u == dft_node_name: partner_key = v
            elif v == dft_node_name: partner_key = u

            if partner_key:
                p_c_idx = int(partner_key.replace("DFT_", ""))
                j_partners.append(p_c_idx)
                j_info_list.append(f"C_{p_c_idx} ({jv:.1f} Гц)")
                if p_c_idx < self.mol.GetNumAtoms():
                    p_atom = self.mol.GetAtomWithIdx(p_c_idx)
                    j_partner_protons.extend([n.GetIdx() for n in p_atom.GetNeighbors() if n.GetSymbol() == 'H'])

        colors = {}
        colors[c_idx] = (1.0, 0.25, 0.0)
        for h in own_protons: colors[h] = (1.0, 0.65, 0.0)
        for b in bonded_neighbors: colors[b] = (0.12, 0.55, 1.0)
        for jp in j_partners: colors[jp] = (0.15, 0.80, 0.30)
        for jph in j_partner_protons: colors[jph] = (0.45, 0.90, 0.20)

        highlight_bonds = []
        for b in bonded_neighbors:
            bond = self.mol.GetBondBetweenAtoms(c_idx, b)
            if bond: highlight_bonds.append(bond.GetIdx())
        for h in own_protons:
            bond = self.mol.GetBondBetweenAtoms(c_idx, h)
            if bond: highlight_bonds.append(bond.GetIdx())

        all_highlight_atoms = list(colors.keys())
        self._render_mol(highlights=all_highlight_atoms, highlight_colors=colors, highlight_bonds=highlight_bonds)

        bonded_str = ", ".join([f"{self.mol.GetAtomWithIdx(b).GetSymbol()}_{b}" for b in bonded_neighbors]) or "нет"
        j_str = ", ".join(j_info_list) or "нет"
        self.lbl_mol_info.setText(f"Выбран C_{c_idx} | Связан с: {bonded_str} | J-партнеры DFT: {j_str}")

    def _render_mol(self, highlights: Optional[List[int]] = None,
                    highlight_colors: Optional[Dict[int, tuple]] = None,
                    highlight_bonds: Optional[List[int]] = None):
        if not self.mol:
            return
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(600, 240)
            opts = d2d.drawOptions()
            opts.addAtomIndices = True
            opts.highlightBondWidthMultiplier = 3

            if highlights:
                d2d.DrawMolecule(
                    self.mol,
                    highlightAtoms=highlights,
                    highlightAtomColors=highlight_colors,
                    highlightBonds=highlight_bonds
                )
            else:
                d2d.DrawMolecule(self.mol)

            d2d.FinishDrawing()
            pix = QPixmap()
            pix.loadFromData(d2d.GetDrawingText())
            self.mol_view.setPixmap(pix)
            self.mol_view.atom_coords.clear()
            for atom in self.mol.GetAtoms():
                pt = d2d.GetDrawCoords(atom.GetIdx())
                self.mol_view.atom_coords[atom.GetIdx()] = (pt.x, pt.y)
        except Exception as e:
            self.mol_view.setText(f"Ошибка рендера RDKit: {e}")

    def on_mol_atom_clicked(self, atom_idx: int):
        if not self.mol or atom_idx >= self.mol.GetNumAtoms():
            return
        atom = self.mol.GetAtomWithIdx(atom_idx)
        c_idx = atom_idx if atom.GetSymbol() == 'C' else [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'C'][0]

        self.highlight_atom_and_connections(c_idx)
        # Центрируем соответствующий узел на вкладке XYZ
        self.cv_xyz.zoom_to_node(f"C_{c_idx}")

    def on_graph_node_clicked(self, node_id: str):
        c_idx = None
        if node_id.startswith("C_"):
            c_idx = int(node_id.replace("C_", ""))
        elif node_id.startswith("DFT_"):
            c_idx = int(node_id.replace("DFT_", ""))
        elif node_id.startswith("EXP_"):
            for r in range(self.table_nodes.rowCount()):
                it = self.table_nodes.item(r, 0)
                if it and it.text() == node_id:
                    self.table_nodes.selectRow(r)
                    break
            return

        if c_idx is not None:
            self.highlight_atom_and_connections(c_idx)

    def launch_matching(self):
        if not self.last_graphs:
            self.plot_all()
        if self.last_graphs:
            self.requestMatching.emit(self.last_graphs, self.mol)

    def save_session(self):
        data = {
            "xyz": self.txt_xyz.toPlainText(), "dft": self.txt_dft.toPlainText(),
            "dft_shield": self.txt_dft_shielding.toPlainText(),
            "exp1d": self.txt_1d.toPlainText(), "hsqc": self.txt_hsqc.toPlainText(), "cosy": self.txt_cosy.toPlainText(),
            "settings": {"c_tol": self.sp_ctol.value(), "j_tol": self.sp_jtol.value(), "area": self.sp_area.value(),
                         "tms_c": self.sp_tms_c.value(), "tms_h": self.sp_tms_h.value()}
        }
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", "", "JSON (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить", "", "JSON (*.json)")
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.txt_xyz.setPlainText(data.get("xyz", ""))
            self.txt_dft.setPlainText(data.get("dft", ""))
            self.txt_dft_shielding.setPlainText(data.get("dft_shield", ""))
            self.txt_1d.setPlainText(data.get("exp1d", ""))
            self.txt_hsqc.setPlainText(data.get("hsqc", ""))
            self.txt_cosy.setPlainText(data.get("cosy", ""))
            st = data.get("settings", {})
            if "c_tol" in st: self.sp_ctol.setValue(st["c_tol"])
            if "j_tol" in st: self.sp_jtol.setValue(st["j_tol"])
            if "area" in st: self.sp_area.setValue(st["area"])
            if "tms_c" in st: self.sp_tms_c.setValue(st["tms_c"])
            if "tms_h" in st: self.sp_tms_h.setValue(st["tms_h"])