import sys
import os
import re
import json
import numpy as np
from typing import Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSplitter, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

COV_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03, 'Ca': 1.76,
    'Br': 1.20, 'I': 1.39
}


class InteractiveMolLabel(QLabel):
    atomClicked = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atom_coords = {}

    def mousePressEvent(self, event):
        if not self.atom_coords:
            return
        click_pos = event.pos()
        cx, cy = click_pos.x(), click_pos.y()

        closest_atom = -1
        min_dist = 600

        for idx, (ax, ay) in self.atom_coords.items():
            dist = (cx - ax) ** 2 + (cy - ay) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_atom = idx

        if closest_atom != -1:
            self.atomClicked.emit(closest_atom)


class StrictMatcherWindow(QMainWindow):
    def __init__(self, json_path: str):
        super().__init__()
        self.json_path = json_path
        self.mol = None
        self.graphs_data = {}
        self.setWindowTitle(f"Сопоставление топологии (Строгие Якоря) — {os.path.basename(json_path)}")
        self.resize(1300, 850)
        self._init_ui()
        self.load_data_and_match()

    def _init_ui(self):
        main_w = QWidget()
        layout = QHBoxLayout(main_w)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая часть: 2D структура
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        mol_group = QGroupBox("2D Структура (Подсветка по клику)")
        mg_layout = QVBoxLayout(mol_group)

        self.mol_view = InteractiveMolLabel("2D Структура")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ECEFF1; border-radius: 4px;")
        self.mol_view.setMinimumHeight(450)
        self.mol_view.atomClicked.connect(self.on_mol_atom_clicked)
        mg_layout.addWidget(self.mol_view)
        l_layout.addWidget(mol_group)
        splitter.addWidget(left_w)

        # Правая часть: Таблица якорей
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)

        self.lbl_info = QLabel("Анализ данных...")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #1565C0; padding: 4px;")
        r_layout.addWidget(self.lbl_info)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Статус", "Углерод (XYZ / DFT)", "Узел Exp", "Тип", "13C δ (ppm) / 1H δ", "Обоснование якоря"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        r_layout.addWidget(self.table)

        splitter.addWidget(right_w)
        splitter.setSizes([500, 800])
        layout.addWidget(splitter)
        self.setCentralWidget(main_w)

    def _build_mol_from_xyz(self, xyz_text: str):
        symbols, coords = [], []
        for line in xyz_text.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                m = re.match(r'^([A-Za-z]{1,2})$', parts[0])
                if m:
                    try:
                        symbols.append(m.group(1).capitalize())
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
        if not symbols:
            return

        mol = Chem.RWMol()
        for sym in symbols:
            a = Chem.Atom(sym)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            mol.AddAtom(a)

        n_atoms = len(symbols)
        coords_arr = np.array(coords)
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_cov = COV_RADII.get(symbols[i], 0.77) + COV_RADII.get(symbols[j], 0.77)
                if 0.4 < dist_matrix[i, j] <= 1.22 * r_cov:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        try:
            rdDepictor.Compute2DCoords(mol)
            self.mol = mol
        except Exception:
            pass

    def _render_mol(self, highlights: Optional[List[int]] = None):
        if not self.mol:
            self.mol_view.setText("Координаты молекулы не найдены.")
            return

        w, h = 500, 450
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
            opts = d2d.drawOptions()
            opts.addAtomIndices = True
            opts.highlightBondWidthMultiplier = 3  # Исправлено: строго целое число int

            if highlights:
                colors = {idx: (1.0, 0.40, 0.0) for idx in highlights}
                d2d.DrawMolecule(self.mol, highlightAtoms=highlights, highlightAtomColors=colors)
            else:
                d2d.DrawMolecule(self.mol)

            d2d.FinishDrawing()
            pix = QPixmap()
            pix.loadFromData(d2d.GetDrawingText())
            self.mol_view.setPixmap(pix)

            self.mol_view.atom_coords.clear()
            for atom in self.mol.GetAtoms():
                idx = atom.GetIdx()
                pt = d2d.GetDrawCoords(idx)
                self.mol_view.atom_coords[idx] = (pt.x, pt.y)

        except Exception as e:
            self.mol_view.setText(f"Ошибка рендера: {e}")

    def load_data_and_match(self):
        if not os.path.exists(self.json_path):
            QMessageBox.critical(self, "Ошибка", f"Файл не найден:\n{self.json_path}")
            return

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.graphs_data = json.load(f)

        raw_xyz = self.graphs_data.get("raw_xyz", "")
        if raw_xyz:
            self._build_mol_from_xyz(raw_xyz)
        self._render_mol()

        xyz_nodes = self.graphs_data.get("xyz_graph", {}).get("nodes", {})
        dft_edges = self.graphs_data.get("dft_graph", {}).get("edges", [])
        exp_nodes = self.graphs_data.get("exp_graph", {}).get("nodes", {})
        exp_edges = self.graphs_data.get("exp_graph", {}).get("edges", [])

        # Формирование констант J для каждого узла
        dft_node_j = {k: [] for k in xyz_nodes}
        for edge in dft_edges:
            u, v = edge[0], edge[1]
            j_val = edge[2] if len(edge) >= 3 else 7.0
            u_clean = u.replace("DFT_", "C_")
            v_clean = v.replace("DFT_", "C_")
            if u_clean in dft_node_j:
                dft_node_j[u_clean].append(j_val)
            if v_clean in dft_node_j:
                dft_node_j[v_clean].append(j_val)

        for k in dft_node_j:
            dft_node_j[k] = sorted([round(x, 1) for x in dft_node_j[k]])

        exp_node_j = {k: [] for k in exp_nodes}
        for edge in exp_edges:
            u, v = edge[0], edge[1]
            j_val = edge[2] if len(edge) >= 3 else 7.0
            if u in exp_node_j:
                exp_node_j[u].append(j_val)
            if v in exp_node_j:
                exp_node_j[v].append(j_val)

        for k in exp_node_j:
            exp_node_j[k] = sorted([round(x, 1) for x in exp_node_j[k]])

        matched_anchors = {}
        used_exp = set()

        # 1. Железный якорь: метокси CH3
        xyz_ch3 = [k for k, v in xyz_nodes.items() if v.get('type') == 'CH3']
        exp_ch3 = [k for k, v in exp_nodes.items() if v.get('type') == 'CH3']
        if len(xyz_ch3) == 1 and len(exp_ch3) == 1:
            matched_anchors[xyz_ch3[0]] = (exp_ch3[0], "Уникальный CH3 (Метоксигруппа)")
            used_exp.add(exp_ch3[0])

        # 2. Изолированные узлы (0 связей)
        xyz_isolated = [k for k in xyz_nodes if len(dft_node_j[k]) == 0]
        exp_isolated = [k for k in exp_nodes if len(exp_node_j[k]) == 0 and k not in used_exp]

        for d_node in xyz_isolated:
            if d_node in matched_anchors:
                continue
            d_type = xyz_nodes[d_node].get('type')
            candidates = [e for e in exp_isolated if exp_nodes[e].get('type') == d_type and e not in used_exp]

            if len(candidates) == 1:
                e_match = candidates[0]
                matched_anchors[d_node] = (e_match, f"Уникальный изолированный {d_type}")
                used_exp.add(e_match)

        # 3. Узлы со строгим совпадением набора J
        for d_node, d_j_list in dft_node_j.items():
            if d_node in matched_anchors or not d_j_list:
                continue
            d_type = xyz_nodes[d_node].get('type')

            valid_candidates = []
            for e_node, e_j_list in exp_node_j.items():
                if e_node in used_exp:
                    continue
                if exp_nodes[e_node].get('type') == d_type and len(d_j_list) == len(e_j_list):
                    diffs = [abs(a - b) for a, b in zip(d_j_list, e_j_list)]
                    if all(df <= 1.0 for df in diffs):
                        valid_candidates.append(e_node)

            if len(valid_candidates) == 1:
                e_match = valid_candidates[0]
                matched_anchors[d_node] = (e_match, f"Точный J-паттерн: {d_j_list} Гц")
                used_exp.add(e_match)

        # Заполнение таблицы
        self.table.setRowCount(0)
        c_green = QColor(220, 255, 220)
        c_gray = QColor(245, 245, 245)

        for d_node in sorted(xyz_nodes.keys(), key=lambda x: int(x.split('_')[1])):
            row = self.table.rowCount()
            self.table.insertRow(row)

            c_idx = int(d_node.split('_')[1])
            g_type = xyz_nodes[d_node].get('type', 'CH')

            if d_node in matched_anchors:
                e_node, reason = matched_anchors[d_node]
                edata = exp_nodes[e_node]
                h_str = ", ".join(f"{h:.2f}" for h in edata.get('h_list', []))
                shift_str = f"δC {edata.get('c', 0.0):.2f} / δH {h_str}"

                item_status = QTableWidgetItem("Якорь (Strict)")
                item_dft = QTableWidgetItem(f"{d_node} (DFT_{c_idx})")
                item_exp = QTableWidgetItem(e_node)
                item_type = QTableWidgetItem(g_type)
                item_shift = QTableWidgetItem(shift_str)
                item_reason = QTableWidgetItem(reason)

                item_status.setData(Qt.ItemDataRole.UserRole, c_idx)
                for it in [item_status, item_dft, item_exp, item_type, item_shift, item_reason]:
                    it.setBackground(c_green)
            else:
                item_status = QTableWidgetItem("Не соотнесено")
                item_dft = QTableWidgetItem(f"{d_node} (DFT_{c_idx})")
                item_exp = QTableWidgetItem("—")
                item_type = QTableWidgetItem(g_type)
                item_shift = QTableWidgetItem("—")
                item_reason = QTableWidgetItem("Требуется топологический поиск")

                item_status.setData(Qt.ItemDataRole.UserRole, c_idx)
                for it in [item_status, item_dft, item_exp, item_type, item_shift, item_reason]:
                    it.setBackground(c_gray)

            self.table.setItem(row, 0, item_status)
            self.table.setItem(row, 1, item_dft)
            self.table.setItem(row, 2, item_exp)
            self.table.setItem(row, 3, item_type)
            self.table.setItem(row, 4, item_shift)
            self.table.setItem(row, 5, item_reason)

        tot_nodes = len(xyz_nodes)
        n_matched = len(matched_anchors)
        self.lbl_info.setText(
            f"Файл: {os.path.basename(self.json_path)} | "
            f"Строгих якорей найдено: {n_matched} из {tot_nodes}"
        )

    def on_table_row_selected(self):
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        item = self.table.item(row, 0)
        c_idx = item.data(Qt.ItemDataRole.UserRole)
        if c_idx is not None and self.mol and c_idx < self.mol.GetNumAtoms():
            atom = self.mol.GetAtomWithIdx(c_idx)
            h_indices = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
            self._render_mol(highlights=[c_idx] + h_indices)

    def on_mol_atom_clicked(self, atom_idx: int):
        if not self.mol or atom_idx >= self.mol.GetNumAtoms():
            return
        atom = self.mol.GetAtomWithIdx(atom_idx)
        c_idx = atom_idx if atom.GetSymbol() == 'C' else [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C'][0]

        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item.data(Qt.ItemDataRole.UserRole) == c_idx:
                self.table.selectRow(r)
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file, _ = QFileDialog.getOpenFileName(None, "Выберите экспортированный граф", "graph", "JSON Files (*.json)")

    if json_file and os.path.exists(json_file):
        win = StrictMatcherWindow(json_file)
        win.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)