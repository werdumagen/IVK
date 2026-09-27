# tab_match.py
import sys
import json
import traceback
import numpy as np
from typing import Optional, List, Dict

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSplitter, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QMessageBox, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from core import InteractiveMolLabel, InteractiveGraphCanvas, ScrollableComboBox


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

        gb_mol = QGroupBox("2D Структура (Оранжевый: атом, Зеленый: J-соседи, Синий: связи)")
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

        left_splitter.setSizes([380, 420])
        l_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_w)

        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)

        top_bar = QHBoxLayout()
        self.lbl_info = QLabel("Ожидание данных...")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #1565C0; padding: 4px;")
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()

        self.btn_export_ai = QPushButton("🤖 Экспорт данных для анализа ИИ (JSON)")
        self.btn_export_ai.setStyleSheet(
            "background-color: #6A1B9A; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;"
        )
        self.btn_export_ai.clicked.connect(self.export_ai_data)
        top_bar.addWidget(self.btn_export_ai)
        r_layout.addLayout(top_bar)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Статус", "DFT Узел", "Расчет DFT (δC/δH)", "Соотнесение (Exp Узел)", "Факт Exp (δC/δH)", "Обоснование"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        self.table.cellClicked.connect(self.on_cell_clicked)
        r_layout.addWidget(self.table)

        main_splitter.addWidget(right_w)
        main_splitter.setSizes([550, 700])
        layout.addWidget(main_splitter)

    def _norm_type(self, t: str) -> str:
        return 'CH' if t in ['CH', 'CH1'] else t

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
            if u in dft_node_j:
                dft_node_j[u].append(j_val)
            if v in dft_node_j:
                dft_node_j[v].append(j_val)
        for k in dft_node_j:
            dft_node_j[k] = sorted([round(x, 1) for x in dft_node_j[k]])

        def get_dft_multiplicity(j_list: List[float]) -> str:
            n_j = len(j_list)
            if n_j == 0:
                return 's'
            if n_j == 1:
                return 'd'
            if n_j == 2:
                return 't' if abs(j_list[0] - j_list[1]) <= 1.0 else 'dd'
            return 'm'

        def check_shifts_valid(d_node_key: str, e_node_key: str, c_lim: float = 15.0, h_lim: float = 0.5) -> bool:
            c_idx = int(d_node_key.split('_')[1])
            d_d = dft_nodes.get(f"DFT_{c_idx}", {})
            e_d = exp_nodes.get(e_node_key, {})

            c_dft = d_d.get('c_shift')
            c_exp = e_d.get('c')
            if c_dft is None or c_exp is None or abs(float(c_dft) - float(c_exp)) > c_lim:
                return False

            h_dfts = sorted([float(x) for x in d_d.get('h_shifts', []) if x is not None])
            h_exps = sorted([float(x) for x in e_d.get('h_list', []) if x is not None])

            d_type = self._norm_type(xyz_nodes[d_node_key].get('type'))

            if d_type == 'CH3':
                return abs(float(np.mean(h_dfts)) - float(np.mean(h_exps))) <= 1.0

            if not e_d.get('is_complete', True):
                return False

            if len(h_dfts) != len(h_exps):
                return False

            for hd, he in zip(h_dfts, h_exps):
                if abs(hd - he) > h_lim:
                    return False

            return True

        anchors, used_exp = {}, set()

        # Шаг 1: Автосопоставление метила CH3 (уникальный)
        xyz_ch3 = [k for k, v in xyz_nodes.items() if self._norm_type(v.get('type')) == 'CH3' and k not in anchors]
        exp_ch3 = [k for k, v in exp_nodes.items() if self._norm_type(v.get('type')) == 'CH3' and k not in used_exp]
        if len(xyz_ch3) == 1 and len(exp_ch3) == 1:
            if check_shifts_valid(xyz_ch3[0], exp_ch3[0], c_lim=20.0, h_lim=1.0):
                anchors[xyz_ch3[0]] = (exp_ch3[0], "Уникальный CH3")
                used_exp.add(exp_ch3[0])

        # Шаг 2: Строгое совпадение сдвигов и мультиплетности (ΔC <= 15, ΔH <= 0.5, строго один кандидат)
        for d_node in sorted(xyz_nodes.keys(), key=lambda x: int(x.split('_')[1])):
            if d_node in anchors:
                continue
            c_idx = int(d_node.split('_')[1])
            d_d = dft_nodes.get(f"DFT_{c_idx}", {})
            d_type = self._norm_type(xyz_nodes[d_node].get('type'))
            d_mult = get_dft_multiplicity(dft_node_j.get(d_node, []))

            strict_cands = []
            for e_node, e_d in exp_nodes.items():
                if e_node in used_exp or self._norm_type(e_d.get('type')) != d_type:
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

        self.table.setRowCount(0)
        c_green, c_gray = QColor(220, 255, 220), QColor(245, 245, 245)

        for d_node in sorted(xyz_nodes.keys(), key=lambda x: int(x.split('_')[1])):
            row = self.table.rowCount()
            self.table.insertRow(row)
            c_idx = int(d_node.split('_')[1])
            g_type = self._norm_type(xyz_nodes[d_node].get('type', 'CH'))

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

            cb = ScrollableComboBox()
            cb.blockSignals(True)
            cb.addItem("— Не выбрано —", userData=None)

            for ek, ev in exp_nodes.items():
                if self._norm_type(ev.get('type')) == g_type:
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
                item_exp_val.setData(Qt.ItemDataRole.UserRole, e_node)
                item_exp_val.setToolTip(f"Кликните для перехода к {e_node} на графе")
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
            item_exp_val.setData(Qt.ItemDataRole.UserRole, e_node)
            item_exp_val.setToolTip(f"Кликните для перехода к {e_node} на графе")

            c_yellow = QColor(255, 235, 150)
            for it in [item_status, item_dft, item_dft_val, item_exp_val, item_reason]:
                it.setBackground(c_yellow)
        else:
            item_status.setText("Не соотнесено")
            item_exp_val.setText("—")
            item_exp_val.setData(Qt.ItemDataRole.UserRole, None)
            item_exp_val.setToolTip("")
            item_reason.setText("—")
            c_gray = QColor(245, 245, 245)
            for it in [item_status, item_dft, item_dft_val, item_exp_val, item_reason]:
                it.setBackground(c_gray)

    def on_cell_clicked(self, row: int, column: int):
        """Автозум на экспериментальном графе при клике на столбец 'Факт Exp'."""
        if column == 4:
            self.zoom_to_exp_node_at_row(row)

    def zoom_to_exp_node_at_row(self, row: int):
        item_exp = self.table.item(row, 4)
        e_node = item_exp.data(Qt.ItemDataRole.UserRole) if item_exp else None
        if not e_node:
            combo = self.table.cellWidget(row, 3)
            if combo:
                e_node = combo.currentData()
        if e_node:
            self.canvas_exp.zoom_to_node(e_node)

    def highlight_atom_and_connections(self, c_idx: int):
        if not self.mol or c_idx >= self.mol.GetNumAtoms():
            return

        atom = self.mol.GetAtomWithIdx(c_idx)
        own_protons = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
        bonded_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() != 'H']

        dft_edges = self.graphs_data.get("dft_graph", {}).get("edges", [])
        dft_node_name = f"DFT_{c_idx}"
        j_partners = []
        j_partner_protons = []
        j_info_list = []

        for e in dft_edges:
            u, v = e[0], e[1]
            jv = e[2] if len(e) >= 3 else 0.0
            partner_key = None
            if u == dft_node_name:
                partner_key = v
            elif v == dft_node_name:
                partner_key = u

            if partner_key:
                p_c_idx = int(partner_key.replace("DFT_", ""))
                j_partners.append(p_c_idx)
                j_info_list.append(f"C_{p_c_idx} ({jv:.1f} Гц)")
                if p_c_idx < self.mol.GetNumAtoms():
                    p_atom = self.mol.GetAtomWithIdx(p_c_idx)
                    j_partner_protons.extend([n.GetIdx() for n in p_atom.GetNeighbors() if n.GetSymbol() == 'H'])

        colors = {}
        colors[c_idx] = (1.0, 0.25, 0.0)
        for h in own_protons:
            colors[h] = (1.0, 0.65, 0.0)
        for b in bonded_neighbors:
            colors[b] = (0.12, 0.55, 1.0)
        for jp in j_partners:
            colors[jp] = (0.15, 0.80, 0.30)
        for jph in j_partner_protons:
            colors[jph] = (0.45, 0.90, 0.20)

        highlight_bonds = []
        for b in bonded_neighbors:
            bond = self.mol.GetBondBetweenAtoms(c_idx, b)
            if bond:
                highlight_bonds.append(bond.GetIdx())
        for h in own_protons:
            bond = self.mol.GetBondBetweenAtoms(c_idx, h)
            if bond:
                highlight_bonds.append(bond.GetIdx())

        all_highlight_atoms = list(colors.keys())
        self._render_mol(highlights=all_highlight_atoms, highlight_colors=colors, highlight_bonds=highlight_bonds)

        bonded_str = ", ".join([f"{self.mol.GetAtomWithIdx(b).GetSymbol()}_{b}" for b in bonded_neighbors]) or "нет"
        j_str = ", ".join(j_info_list) or "нет"
        self.lbl_info.setText(f"Выбран C_{c_idx} | Связан с: {bonded_str} | J-партнеры DFT: {j_str}")

    def _render_mol(self, highlights: Optional[List[int]] = None,
                    highlight_colors: Optional[Dict[int, tuple]] = None,
                    highlight_bonds: Optional[List[int]] = None):
        if not self.mol:
            return
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(500, 350)
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

    def on_table_row_selected(self):
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        item = self.table.item(row, 0)
        if item is None:
            return
        c_idx = item.data(Qt.ItemDataRole.UserRole)
        if c_idx is not None:
            self.highlight_atom_and_connections(c_idx)

        if self.table.currentColumn() == 4:
            self.zoom_to_exp_node_at_row(row)

    def on_mol_atom_clicked(self, atom_idx: int):
        if not self.mol or atom_idx >= self.mol.GetNumAtoms():
            return
        atom = self.mol.GetAtomWithIdx(atom_idx)
        c_idx = atom_idx if atom.GetSymbol() == 'C' else [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'C'][0]

        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == c_idx:
                self.table.selectRow(r)
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                break
        self.highlight_atom_and_connections(c_idx)

    def on_graph_node_clicked(self, node_id: str):
        c_idx = None
        if node_id.startswith("C_"):
            c_idx = int(node_id.replace("C_", ""))
        elif node_id.startswith("DFT_"):
            c_idx = int(node_id.replace("DFT_", ""))
        elif node_id.startswith("EXP_"):
            self.canvas_exp.zoom_to_node(node_id)
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
            self.highlight_atom_and_connections(c_idx)
            for r in range(self.table.rowCount()):
                item = self.table.item(r, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == c_idx:
                    self.table.selectRow(r)
                    self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                    break

    def export_ai_data(self):
        if not self.graphs_data:
            QMessageBox.warning(self, "Внимание", "Нет данных для экспорта.")
            return

        xyz_nodes = self.graphs_data.get("xyz_graph", {}).get("nodes", {})
        dft_nodes = self.graphs_data.get("dft_graph", {}).get("nodes", {})
        dft_edges = self.graphs_data.get("dft_graph", {}).get("edges", [])
        exp_nodes = self.graphs_data.get("exp_graph", {}).get("nodes", {})
        exp_edges = self.graphs_data.get("exp_graph", {}).get("edges", [])

        current_matches = []
        assigned_dft = set()
        assigned_exp = set()

        for r in range(self.table.rowCount()):
            d_node_item = self.table.item(r, 1)
            status_item = self.table.item(r, 0)
            reason_item = self.table.item(r, 5)
            combo = self.table.cellWidget(r, 3)

            if not d_node_item or not combo:
                continue

            d_node = d_node_item.text()
            e_node = combo.currentData()
            status = status_item.text() if status_item else "—"
            reason = reason_item.text() if reason_item else "—"

            c_idx = int(d_node.split('_')[1])
            d_d = dft_nodes.get(f"DFT_{c_idx}", {})

            match_info = {
                "dft_node": d_node,
                "status": status,
                "assigned_exp_node": e_node,
                "reason": reason,
                "dft_calculated": {
                    "c_shift": d_d.get("c_shift"),
                    "h_shifts": d_d.get("h_shifts", [])
                }
            }

            if e_node:
                assigned_dft.add(d_node)
                assigned_exp.add(e_node)
                e_d = exp_nodes.get(e_node, {})
                match_info["exp_actual"] = {
                    "c_shift": e_d.get("c"),
                    "h_shifts": e_d.get("h_list", []),
                    "j_vals": e_d.get("j_vals", []),
                    "type": e_d.get("type")
                }
            else:
                match_info["exp_actual"] = None

            current_matches.append(match_info)

        unassigned_dft = []
        for d_node, d_info in xyz_nodes.items():
            if d_node not in assigned_dft:
                c_idx = int(d_node.split('_')[1])
                d_d = dft_nodes.get(f"DFT_{c_idx}", {})

                spin_neighbors = []
                for e in dft_edges:
                    u = e[0].replace("DFT_", "C_")
                    v = e[1].replace("DFT_", "C_")
                    jv = e[2] if len(e) >= 3 else None
                    if u == d_node:
                        spin_neighbors.append({"neighbor": v, "j_hz": jv})
                    elif v == d_node:
                        spin_neighbors.append({"neighbor": u, "j_hz": jv})

                unassigned_dft.append({
                    "dft_node": d_node,
                    "type": d_info.get("type"),
                    "c_shift_calculated": d_d.get("c_shift"),
                    "h_shifts_calculated": d_d.get("h_shifts", []),
                    "spin_neighbors_dft": spin_neighbors
                })

        unassigned_exp = []
        for e_node, e_info in exp_nodes.items():
            if e_node not in assigned_exp:
                cosy_neighbors = []
                for e in exp_edges:
                    u, v = e[0], e[1]
                    jv = e[2] if len(e) >= 3 else None
                    if u == e_node:
                        cosy_neighbors.append({"neighbor": v, "j_hz": jv})
                    elif v == e_node:
                        cosy_neighbors.append({"neighbor": u, "j_hz": jv})

                unassigned_exp.append({
                    "exp_node": e_node,
                    "type": e_info.get("type"),
                    "c_shift_actual": e_info.get("c"),
                    "h_shifts_actual": e_info.get("h_list"),
                    "j_vals_actual": e_info.get("j_vals"),
                    "is_singlet": e_info.get("is_singlet"),
                    "is_multiplet": e_info.get("is_multiplet"),
                    "cosy_neighbors": cosy_neighbors
                })

        ai_payload = {
            "meta_prompt": (
                "Ты — эксперт по спектроскопии ЯМР и квантово-химическому моделированию. "
                "Ниже представлены экспериментальные данные (1H, 13C, HSQC, COSY) и расчетные данные DFT. "
                "Часть сигналов уже надежно соотнесена (якоря). Твоя задача: на основе топологии спиновых графов, "
                "констант спин-спинового взаимодействия (J) и близости химсдвигов досоотнести оставшиеся несоотнесенные узлы."
            ),
            "statistics": {
                "total_carbons": len(xyz_nodes),
                "matched_count": len(assigned_dft),
                "unassigned_count": len(unassigned_dft)
            },
            "current_table_matches": current_matches,
            "unassigned_dft_nodes": unassigned_dft,
            "unassigned_exp_nodes": unassigned_exp,
            "raw_graphs": self.graphs_data
        }

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить данные для ИИ", "nmr_ai_analysis_data.json", "JSON (*.json)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(ai_payload, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Успех", f"Данные сохранены в:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось записать файл:\n{e}")