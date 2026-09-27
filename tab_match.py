# tab_match.py
from typing import Optional, List
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSplitter, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor

from rdkit.Chem.Draw import rdMolDraw2D
from core import InteractiveMolLabel


class MatcherTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mol = None
        self.graphs_data = {}
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая часть: Молекула
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        gb = QGroupBox("2D Структура (Подсветка по клику)")
        gbl = QVBoxLayout(gb)
        self.mol_view = InteractiveMolLabel("Нет данных. Сначала постройте графы на первой вкладке.")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.mol_view.atomClicked.connect(self.on_mol_atom_clicked)
        gbl.addWidget(self.mol_view)
        l_layout.addWidget(gb)
        splitter.addWidget(left_w)

        # Правая часть: Таблица
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)
        self.lbl_info = QLabel("Ожидание данных...")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #1565C0; padding: 4px;")
        r_layout.addWidget(self.lbl_info)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Статус", "Углерод (XYZ / DFT)", "Узел Exp", "Тип", "13C δ / 1H δ", "Обоснование"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        r_layout.addWidget(self.table)

        splitter.addWidget(right_w)
        splitter.setSizes([450, 700])
        layout.addWidget(splitter)

    def load_data_and_match(self, graphs_data: dict, mol):
        """Функция принимает данные из первой вкладки напрямую в памяти."""
        self.graphs_data = graphs_data
        self.mol = mol
        self._render_mol()

        xyz_nodes = self.graphs_data.get("xyz_graph", {}).get("nodes", {})
        dft_edges = self.graphs_data.get("dft_graph", {}).get("edges", [])
        exp_nodes = self.graphs_data.get("exp_graph", {}).get("nodes", {})
        exp_edges = self.graphs_data.get("exp_graph", {}).get("edges", [])

        # Вычисляем списки J для узлов
        dft_node_j = {k: [] for k in xyz_nodes}
        for e in dft_edges:
            u, v, j_val = e[0].replace("DFT_", "C_"), e[1].replace("DFT_", "C_"), e[2] if len(e) >= 3 else 7.0
            if u in dft_node_j: dft_node_j[u].append(j_val)
            if v in dft_node_j: dft_node_j[v].append(j_val)
        for k in dft_node_j: dft_node_j[k] = sorted([round(x, 1) for x in dft_node_j[k]])

        exp_node_j = {k: [] for k in exp_nodes}
        for e in exp_edges:
            u, v, j_val = e[0], e[1], e[2] if len(e) >= 3 else 7.0
            if u in exp_node_j: exp_node_j[u].append(j_val)
            if v in exp_node_j: exp_node_j[v].append(j_val)
        for k in exp_node_j: exp_node_j[k] = sorted([round(x, 1) for x in exp_node_j[k]])

        anchors, used_exp = {}, set()

        # 1. Уникальный CH3
        xyz_ch3 = [k for k, v in xyz_nodes.items() if v.get('type') == 'CH3']
        exp_ch3 = [k for k, v in exp_nodes.items() if v.get('type') == 'CH3']
        if len(xyz_ch3) == 1 and len(exp_ch3) == 1:
            anchors[xyz_ch3[0]] = (exp_ch3[0], "Уникальный CH3 (Метоксигруппа)")
            used_exp.add(exp_ch3[0])

        # 2. Изолированные
        for d_node in [k for k in xyz_nodes if not dft_node_j[k]]:
            if d_node in anchors: continue
            d_type = xyz_nodes[d_node].get('type')
            cands = [e for e in exp_nodes if
                     not exp_node_j[e] and e not in used_exp and exp_nodes[e].get('type') == d_type]
            if len(cands) == 1:
                anchors[d_node] = (cands[0], f"Изолированный {d_type}")
                used_exp.add(cands[0])

        # 3. Совпадение J
        for d_node, d_j_list in dft_node_j.items():
            if d_node in anchors or not d_j_list: continue
            cands = []
            for e_node, e_j_list in exp_node_j.items():
                if e_node in used_exp or exp_nodes[e_node].get('type') != xyz_nodes[d_node].get('type') or len(
                    d_j_list) != len(e_j_list): continue
                if all(abs(a - b) <= 1.0 for a, b in zip(d_j_list, e_j_list)):
                    cands.append(e_node)
            if len(cands) == 1:
                anchors[d_node] = (cands[0], f"Точный J-паттерн: {d_j_list} Гц")
                used_exp.add(cands[0])

        self.table.setRowCount(0)
        c_green, c_gray = QColor(220, 255, 220), QColor(245, 245, 245)

        for d_node in sorted(xyz_nodes.keys(), key=lambda x: int(x.split('_')[1])):
            row = self.table.rowCount()
            self.table.insertRow(row)
            c_idx = int(d_node.split('_')[1])
            g_type = xyz_nodes[d_node].get('type', 'CH')

            if d_node in anchors:
                e_node, reason = anchors[d_node]
                ed = exp_nodes[e_node]
                shift_str = f"δC {ed.get('c', 0):.2f} / δH {', '.join(f'{h:.2f}' for h in ed.get('h_list', []))}"
                items = [QTableWidgetItem("Якорь (Strict)"), QTableWidgetItem(f"{d_node} (DFT)"),
                         QTableWidgetItem(e_node), QTableWidgetItem(g_type), QTableWidgetItem(shift_str),
                         QTableWidgetItem(reason)]
                for it in items: it.setBackground(c_green)
            else:
                items = [QTableWidgetItem("Не соотнесено"), QTableWidgetItem(f"{d_node} (DFT)"), QTableWidgetItem("—"),
                         QTableWidgetItem(g_type), QTableWidgetItem("—"), QTableWidgetItem("Топологический поиск")]
                for it in items: it.setBackground(c_gray)

            items[0].setData(Qt.ItemDataRole.UserRole, c_idx)
            for i, it in enumerate(items): self.table.setItem(row, i, it)

        self.lbl_info.setText(f"Строгих якорей найдено: {len(anchors)} из {len(xyz_nodes)}")

    def _render_mol(self, highlights: Optional[List[int]] = None):
        if not self.mol: return
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(500, 450)
            opts = d2d.drawOptions()
            opts.addAtomIndices = True
            opts.highlightBondWidthMultiplier = 3  # СТРОГО INT
            if highlights:
                d2d.DrawMolecule(self.mol, highlightAtoms=highlights,
                                 highlightAtomColors={i: (1.0, 0.40, 0.0) for i in highlights})
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

    def on_table_row_selected(self):
        items = self.table.selectedItems()
        if not items: return
        c_idx = self.table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if c_idx is not None and self.mol and c_idx < self.mol.GetNumAtoms():
            self._render_mol(
                [c_idx] + [n.GetIdx() for n in self.mol.GetAtomWithIdx(c_idx).GetNeighbors() if n.GetSymbol() == 'H'])

    def on_mol_atom_clicked(self, atom_idx: int):
        if not self.mol or atom_idx >= self.mol.GetNumAtoms(): return
        atom = self.mol.GetAtomWithIdx(atom_idx)
        c_idx = atom_idx if atom.GetSymbol() == 'C' else \
        [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'C'][0]
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item.data(Qt.ItemDataRole.UserRole) == c_idx:
                self.table.selectRow(r)
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                break