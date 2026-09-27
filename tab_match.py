# tab_match.py
import sys
import traceback
from typing import Optional, List

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSplitter, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QComboBox, QListView, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap

from rdkit.Chem.Draw import rdMolDraw2D
from core import InteractiveMolLabel, InteractiveGraphCanvas


def handle_exception(exc_type, exc_value, exc_traceback):
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    err_msg = "".join(tb_lines)
    print(err_msg, file=sys.stderr)
    QMessageBox.critical(None, "Ошибка сопоставления", f"Произошел сбой:\n\n{err_msg}")


sys.excepthook = handle_exception


class MatcherTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mol = None
        self.graphs_data = {}
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        gb_mol = QGroupBox("2D Структура (Подсветка по клику)")
        gbl_mol = QVBoxLayout(gb_mol)
        self.mol_view = InteractiveMolLabel("Нет данных. Сначала постройте графы на первой вкладке.")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 4px;")
        self.mol_view.atomClicked.connect(self.on_mol_atom_clicked)
        gbl_mol.addWidget(self.mol_view)
        left_splitter.addWidget(gb_mol)

        gb_graph = QGroupBox("Экспериментальный граф (COSY / J-match)")
        gbl_graph = QVBoxLayout(gb_graph)
        self.canvas_exp = InteractiveGraphCanvas()
        self.canvas_exp.nodeClicked.connect(self.on_graph_node_clicked)
        gbl_graph.addWidget(self.canvas_exp)
        left_splitter.addWidget(gb_graph)

        left_splitter.setSizes([350, 450])
        l_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_w)

        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)
        self.lbl_info = QLabel("Ожидание данных...")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #1565C0; padding: 4px;")
        r_layout.addWidget(self.lbl_info)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Статус", "DFT Узел", "Расчет DFT (δC/δH)", "Соотнесение (Exp Узел)", "Факт Exp (δC/δH)", "Обоснование"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        r_layout.addWidget(self.table)

        main_splitter.addWidget(right_w)
        main_splitter.setSizes([550, 700])
        layout.addWidget(main_splitter)

    def load_data_and_match(self, graphs_data: dict, mol):
        self.graphs_data = graphs_data
        self.mol = mol
        self._render_mol()

        xyz_nodes = self.graphs_data.get("xyz_graph", {}).get("nodes", {})
        dft_nodes = self.graphs_data.get("dft_graph", {}).get("nodes", {})
        dft_edges = self.graphs_data.get("dft_graph", {}).get("edges", [])
        exp_nodes = self.graphs_data.get("exp_graph", {}).get("nodes", {})
        exp_edges = self.graphs_data.get("exp_graph", {}).get("edges", [])

        self.canvas_exp.set_graph(exp_nodes, exp_edges, has_weights=True)

        dft_node_j = {k: [] for k in xyz_nodes}
        for e in dft_edges:
            u, v = e[0].replace("DFT_", "C_"), e[1].replace("DFT_", "C_")
            j_val = e[2] if len(e) >= 3 else 7.0
            if u in dft_node_j: dft_node_j[u].append(j_val)
            if v in dft_node_j: dft_node_j[v].append(j_val)
        for k in dft_node_j:
            dft_node_j[k] = sorted([round(x, 1) for x in dft_node_j[k]])

        exp_node_j = {k: [] for k in exp_nodes}
        for e in exp_edges:
            u, v = e[0], e[1]
            j_val = e[2] if len(e) >= 3 else 7.0
            if u in exp_node_j: exp_node_j[u].append(j_val)
            if v in exp_node_j: exp_node_j[v].append(j_val)
        for k in exp_node_j:
            exp_node_j[k] = sorted([round(x, 1) for x in exp_node_j[k]])

        def get_dft_multiplicity(j_list: List[float]) -> str:
            n_j = len(j_list)
            if n_j == 0: return 's'
            if n_j == 1: return 'd'
            if n_j == 2: return 't' if abs(j_list[0] - j_list[1]) <= 1.0 else 'dd'
            return 'm'

        # Строгая проверка сдвигов без усреднения протонов
        def check_shifts_valid(d_node_key: str, e_node_key: str, c_lim: float = 15.0, h_lim: float = 0.5) -> bool:
            c_idx = int(d_node_key.split('_')[1])
            d_d = dft_nodes.get(f"DFT_{c_idx}", {})
            e_d = exp_nodes.get(e_node_key, {})

            # 1. Проверка углерода
            c_dft = d_d.get('c_shift')
            c_exp = e_d.get('c')
            if c_dft is None or c_exp is None or abs(float(c_dft) - float(c_exp)) > c_lim:
                return False

            # 2. Проверка количества протонов
            h_dfts = sorted([float(x) for x in d_d.get('h_shifts', []) if x is not None])
            h_exps = sorted([float(x) for x in e_d.get('h_list', []) if x is not None])

            # Запрет сопоставления неполных CH2
            if not e_d.get('is_complete', True):
                return False

            if len(h_dfts) != len(h_exps):
                return False

            # 3. Попарная проверка каждого протона
            for hd, he in zip(h_dfts, h_exps):
                if abs(hd - he) > h_lim:
                    return False

            return True

        anchors, used_exp = {}, set()

        # Шаг 1: Автосопоставление по жесткому правилу (ΔC <= 15, ΔH <= 0.5 для КАЖДОГО протона, уникальный кандидат)
        for d_node in sorted(xyz_nodes.keys(), key=lambda x: int(x.split('_')[1])):
            c_idx = int(d_node.split('_')[1])
            d_d = dft_nodes.get(f"DFT_{c_idx}", {})
            d_type = xyz_nodes[d_node].get('type')
            d_mult = get_dft_multiplicity(dft_node_j.get(d_node, []))

            strict_cands = []
            for e_node, e_d in exp_nodes.items():
                if e_node in used_exp or e_d.get('type') != d_type:
                    continue

                if not check_shifts_valid(d_node, e_node, c_lim=15.0, h_lim=0.5):
                    continue

                e_is_s = e_d.get('is_singlet', False)
                e_is_m = e_d.get('is_multiplet', False)
                e_j_len = len(e_d.get('j_vals', []))

                mult_matched = False
                if d_mult == 's' and e_is_s:
                    mult_matched = True
                elif d_mult == 'd' and (e_j_len == 1 and not e_is_s and not e_is_m):
                    mult_matched = True
                elif d_mult not in ['s', 'd']:
                    if e_is_m or (d_mult in ['t', 'dd'] and e_j_len >= 2):
                        mult_matched = True

                if mult_matched:
                    strict_cands.append(e_node)

            if len(strict_cands) == 1:
                e_match = strict_cands[0]
                anchors[d_node] = (e_match, "Строгое совпадение сдвигов (ΔC≤15, ΔH≤0.5) и мультиплетности")
                used_exp.add(e_match)

        # Шаг 2: Уникальный CH3 (если остался)
        xyz_ch3 = [k for k, v in xyz_nodes.items() if v.get('type') == 'CH3' and k not in anchors]
        exp_ch3 = [k for k, v in exp_nodes.items() if v.get('type') == 'CH3' and k not in used_exp]
        if len(xyz_ch3) == 1 and len(exp_ch3) == 1:
            if check_shifts_valid(xyz_ch3[0], exp_ch3[0], c_lim=20.0, h_lim=1.0):
                anchors[xyz_ch3[0]] = (exp_ch3[0], "Уникальный CH3")
                used_exp.add(exp_ch3[0])

        self.table.setRowCount(0)
        c_green, c_gray = QColor(220, 255, 220), QColor(245, 245, 245)

        for d_node in sorted(xyz_nodes.keys(), key=lambda x: int(x.split('_')[1])):
            row = self.table.rowCount()
            self.table.insertRow(row)
            c_idx = int(d_node.split('_')[1])
            g_type = xyz_nodes[d_node].get('type', 'CH')

            dft_key = f"DFT_{c_idx}"
            d_data = dft_nodes.get(dft_key, {})
            c_shift_dft = d_data.get('c_shift')
            h_shifts_dft = d_data.get('h_shifts', [])

            c_s_txt = f"{c_shift_dft:.1f}" if c_shift_dft is not None else "—"
            h_s_txt = ', '.join(f'{h:.2f}' for h in h_shifts_dft if h is not None) if h_shifts_dft else "—"
            dft_str = f"δC {c_s_txt} / δH {h_s_txt}"

            item_status = QTableWidgetItem("Не соотнесено")
            item_dft = QTableWidgetItem(f"{d_node}")
            item_dft_val = QTableWidgetItem(dft_str)
            item_exp_val = QTableWidgetItem("—")
            item_reason = QTableWidgetItem("Топологический поиск")
            item_status.setData(Qt.ItemDataRole.UserRole, c_idx)

            for it in [item_status, item_dft, item_dft_val, item_exp_val, item_reason]:
                it.setBackground(c_gray)

            self.table.setItem(row, 0, item_status)
            self.table.setItem(row, 1, item_dft)
            self.table.setItem(row, 2, item_dft_val)
            self.table.setItem(row, 4, item_exp_val)
            self.table.setItem(row, 5, item_reason)

            cb = QComboBox()
            cb.blockSignals(True)
            cb_view = QListView()
            cb_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            cb.setView(cb_view)
            cb.setMaxVisibleItems(15)
            cb.setStyleSheet("QComboBox { combobox-popup: 0; }")

            cb.addItem("— Не выбрано —", userData=None)
            for ek, ev in exp_nodes.items():
                if ev.get('type') == g_type:
                    tag = "" if ev.get('is_complete', True) else " [1H]"
                    cb.addItem(f"{ek}{tag} (δC {ev.get('c', 0):.1f})", userData=ek)

            if d_node in anchors:
                e_node, reason = anchors[d_node]
                cb_idx = cb.findData(e_node)
                if cb_idx >= 0:
                    cb.setCurrentIndex(cb_idx)

                ed = exp_nodes[e_node]
                shift_str = f"δC {ed.get('c', 0):.2f} / δH {', '.join(f'{h:.2f}' for h in ed.get('h_list', []))}"

                item_status.setText("Авто-Якорь")
                item_exp_val.setText(shift_str)
                item_reason.setText(reason)
                for it in [item_status, item_dft, item_dft_val, item_exp_val, item_reason]:
                    it.setBackground(c_green)

            self.table.setCellWidget(row, 3, cb)
            cb.blockSignals(False)

            cb.currentIndexChanged.connect(
                lambda idx, r=row, d_k=d_node, combo=cb: self.on_manual_match_changed(r, d_k, combo)
            )

        self.lbl_info.setText(f"Строгих якорей найдено: {len(anchors)} из {len(xyz_nodes)}")

    def on_manual_match_changed(self, row, d_node, combo):
        e_node = combo.currentData()
        item_status = self.table.item(row, 0)
        item_dft = self.table.item(row, 1)
        item_dft_val = self.table.item(row, 2)
        item_exp_val = self.table.item(row, 4)
        item_reason = self.table.item(row, 5)

        if not all([item_status, item_dft, item_dft_val, item_exp_val, item_reason]):
            return

        if e_node:
            item_status.setText("Ручное")
            item_reason.setText("Выбрано пользователем")
            ed = self.graphs_data.get("exp_graph", {}).get("nodes", {}).get(e_node, {})
            shift_str = f"δC {ed.get('c', 0):.2f} / δH {', '.join(f'{h:.2f}' for h in ed.get('h_list', []))}"
            item_exp_val.setText(shift_str)

            c_yellow = QColor(255, 235, 150)
            for it in [item_status, item_dft, item_dft_val, item_exp_val, item_reason]:
                it.setBackground(c_yellow)
        else:
            item_status.setText("Не соотнесено")
            item_exp_val.setText("—")
            item_reason.setText("—")
            c_gray = QColor(245, 245, 245)
            for it in [item_status, item_dft, item_dft_val, item_exp_val, item_reason]:
                it.setBackground(c_gray)

    def _render_mol(self, highlights: Optional[List[int]] = None):
        if not self.mol:
            return
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(500, 350)
            opts = d2d.drawOptions()
            opts.addAtomIndices = True
            opts.highlightBondWidthMultiplier = 3
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
        if not items:
            return
        row = items[0].row()
        item = self.table.item(row, 0)
        if item is None:
            return
        c_idx = item.data(Qt.ItemDataRole.UserRole)
        if c_idx is not None and self.mol and c_idx < self.mol.GetNumAtoms():
            atom = self.mol.GetAtomWithIdx(c_idx)
            h_indices = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
            self._render_mol(highlights=[c_idx] + h_indices)

    def on_mol_atom_clicked(self, atom_idx: int):
        if not self.mol or atom_idx >= self.mol.GetNumAtoms():
            return
        atom = self.mol.GetAtomWithIdx(atom_idx)
        c_idx = atom_idx if atom.GetSymbol() == 'C' else \
        [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'C'][0]
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == c_idx:
                self.table.selectRow(r)
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                break

    def on_graph_node_clicked(self, node_id: str):
        c_idx = None
        if node_id.startswith("C_"):
            c_idx = int(node_id.replace("C_", ""))
        elif node_id.startswith("DFT_"):
            c_idx = int(node_id.replace("DFT_", ""))
        elif node_id.startswith("EXP_"):
            for r in range(self.table.rowCount()):
                combo = self.table.cellWidget(r, 3)
                if combo and combo.currentData() == node_id:
                    self.table.selectRow(r)
                    item_0 = self.table.item(r, 0)
                    if item_0:
                        self.table.scrollToItem(item_0, QAbstractItemView.ScrollHint.PositionAtCenter)
                    break
            return

        if c_idx is not None:
            if self.mol and c_idx < self.mol.GetNumAtoms():
                atom = self.mol.GetAtomWithIdx(c_idx)
                self._render_mol([c_idx] + [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H'])
            for r in range(self.table.rowCount()):
                item = self.table.item(r, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == c_idx:
                    self.table.selectRow(r)
                    self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                    break