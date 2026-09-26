import sys
import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QMessageBox, QGroupBox, QAbstractItemView
)
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt, pyqtSignal

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Таблица ковалентных радиусов (Ангстремы)
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
        min_dist = 400  # Радиус чувствительности (20px)

        for idx, (ax, ay) in self.atom_coords.items():
            dist = (cx - ax) ** 2 + (cy - ay) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_atom = idx

        if closest_atom != -1:
            self.atomClicked.emit(closest_atom)


# ==========================================
#         АЛГОРИТМ 1 (ИСХОДНЫЙ)
# ==========================================

class DFTParser1:
    @staticmethod
    def parse_j_matrix(text: str) -> Tuple[Dict[int, str], Dict[Tuple[int, int], float]]:
        atoms = {}
        j_matrix = {}
        col_headers = []

        for line in text.splitlines():
            tokens = line.split()
            if not tokens:
                continue

            if len(tokens) >= 4 and tokens[0].isdigit() and tokens[1].isalpha() and tokens[2].isdigit() and tokens[3].isalpha():
                col_headers = [int(tokens[i]) for i in range(0, len(tokens), 2)]
                continue

            if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isalpha() and '.' in tokens[2]:
                row_idx = int(tokens[0])
                sym = tokens[1].capitalize()
                atoms[row_idx] = sym

                for i, val_str in enumerate(tokens[2:]):
                    if i < len(col_headers):
                        c_idx = col_headers[i]
                        try:
                            val = float(val_str)
                            j_matrix[(row_idx, c_idx)] = val
                            j_matrix[(c_idx, row_idx)] = val
                        except ValueError:
                            pass

        return atoms, j_matrix

    @staticmethod
    def build_spin_graph(atoms: Dict[int, str], j_matrix: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, dict], Dict[Tuple[int, int], int]]:
        c_nodes = {idx: {"sym": sym, "H": []} for idx, sym in atoms.items() if sym == 'C'}

        # 1J_CH
        for (i, j), val in j_matrix.items():
            if abs(val) > 110:
                if atoms.get(i) == 'C' and atoms.get(j) == 'H':
                    if j not in c_nodes[i]["H"]:
                        c_nodes[i]["H"].append(j)
                elif atoms.get(j) == 'C' and atoms.get(i) == 'H':
                    if i not in c_nodes[j]["H"]:
                        c_nodes[j]["H"].append(i)

        spin_nodes = {k: v for k, v in c_nodes.items() if len(v["H"]) > 0}

        edges = {}
        for (i, j), val in j_matrix.items():
            if atoms.get(i) == 'H' and atoms.get(j) == 'H':
                c_i = next((c for c, data in spin_nodes.items() if i in data["H"]), None)
                c_j = next((c for c, data in spin_nodes.items() if j in data["H"]), None)

                # Внутренние геминальные константы игнорируются
                if c_i is not None and c_j is not None and c_i != c_j:
                    pair = tuple(sorted((c_i, c_j)))
                    # Ortho / Vicinal алкильные
                    if abs(val) >= 4.5:
                        edges[pair] = 1
                    # Meta / W-coupling алкильные
                    elif 1.0 < abs(val) < 4.5:
                        edges[pair] = 2

        return spin_nodes, edges


class RDKitGraphBuilder1:
    @staticmethod
    def build_from_xyz(xyz_text: str) -> Tuple[Chem.Mol, Dict[int, dict], Dict[Tuple[int, int], int]]:
        symbols, coords = [], []
        for line in xyz_text.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                m = re.match(r'([A-Za-z]+)', parts[0])
                if m:
                    symbols.append(m.group(1).capitalize())
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if not symbols:
            raise ValueError("Не удалось найти координаты XYZ.")

        n_atoms = len(symbols)
        coords_arr = np.array(coords)
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        mol = Chem.RWMol()
        for sym in symbols:
            a = Chem.Atom(sym)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            mol.AddAtom(a)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_cov = COV_RADII.get(symbols[i], 0.77) + COV_RADII.get(symbols[j], 0.77)
                if 0.4 < dist_matrix[i, j] <= 1.25 * r_cov:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        rdDepictor.Compute2DCoords(mol)

        rd_nodes = {}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
                if h_nbrs:
                    rd_nodes[atom.GetIdx()] = {"H": h_nbrs}

        rd_edges = {}
        topo_dist = Chem.GetDistanceMatrix(mol)
        c_idx_list = list(rd_nodes.keys())

        for i in range(len(c_idx_list)):
            for j in range(i + 1, len(c_idx_list)):
                c1, c2 = c_idx_list[i], c_idx_list[j]
                d = topo_dist[c1, c2]
                pair = tuple(sorted((c1, c2)))
                if d == 1:
                    rd_edges[pair] = 1  # Соседние (Ortho / Vicinal)
                elif d == 2:
                    rd_edges[pair] = 2  # Через один атом (Meta / W-coupling алкилов)

        return mol, rd_nodes, rd_edges


class GraphMatcher1:
    @staticmethod
    def get_components(nodes: List[int], edges: Dict[Tuple[int, int], int]) -> List[List[int]]:
        adj = {n: set() for n in nodes}
        for (u, v) in edges.keys():
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)

        visited = set()
        components = []
        for n in nodes:
            if n not in visited:
                comp, q = [], [n]
                visited.add(n)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)
                components.append(comp)
        return components

    @staticmethod
    def match(dft_nodes: dict, dft_edges: dict, rd_nodes: dict, rd_edges: dict) -> dict:
        dft_comps = GraphMatcher1.get_components(list(dft_nodes.keys()), dft_edges)
        rd_comps = GraphMatcher1.get_components(list(rd_nodes.keys()), rd_edges)

        mapping = {}
        used_rd = set()

        for d_comp in dft_comps:
            d_types = sorted([len(dft_nodes[n]["H"]) for n in d_comp])

            for r_comp in rd_comps:
                if not r_comp or r_comp[0] in used_rd:
                    continue

                r_types = sorted([len(rd_nodes[n]["H"]) for n in r_comp])
                if d_types != r_types:
                    continue

                comp_map = {}
                temp_used = set()

                def backtrack(idx):
                    if idx == len(d_comp):
                        return True
                    d_n = d_comp[idx]
                    req_h = len(dft_nodes[d_n]["H"])

                    for r_n in r_comp:
                        if r_n in temp_used or len(rd_nodes[r_n]["H"]) != req_h:
                            continue

                        consistent = True
                        for m_d, m_r in comp_map.items():
                            d_pair = tuple(sorted([d_n, m_d]))
                            r_pair = tuple(sorted([r_n, m_r]))

                            d_type = dft_edges.get(d_pair, 0)
                            r_type = rd_edges.get(r_pair, 0)

                            if (r_type == 1 and d_type != 1) or (d_type == 1 and r_type != 1):
                                consistent = False
                                break

                            if d_type == 2 and r_type != 2:
                                consistent = False
                                break

                        if consistent:
                            comp_map[d_n] = r_n
                            temp_used.add(r_n)
                            if backtrack(idx + 1):
                                return True
                            del comp_map[d_n]
                            temp_used.remove(r_n)
                    return False

                if backtrack(0):
                    mapping.update(comp_map)
                    used_rd.update(r_comp)
                    break

        return mapping


# ==========================================
#   АЛГОРИТМ 2 (С ПРИВЯЗКОЙ К ЯКОРЯМ АЛГ. 1)
# ==========================================

class DFTParser2:
    @staticmethod
    def build_spin_graph(atoms: Dict[int, str], j_matrix: Dict[Tuple[int, int], float], spin_nodes: dict) -> List[Tuple[int, int]]:
        edges = set()
        for (i, j), val in j_matrix.items():
            if atoms.get(i) == 'H' and atoms.get(j) == 'H' and abs(val) > 1.5:
                c_i = next((c for c, data in spin_nodes.items() if i in data["H"]), None)
                c_j = next((c for c, data in spin_nodes.items() if j in data["H"]), None)
                if c_i is not None and c_j is not None and c_i != c_j:
                    edges.add(tuple(sorted((c_i, c_j))))
        return list(edges)


class RDKitGraphBuilder2:
    @staticmethod
    def build_edges(mol: Chem.Mol, rd_nodes: dict) -> List[Tuple[int, int]]:
        rd_edges = set()
        topo_dist = Chem.GetDistanceMatrix(mol)
        c_idx_list = list(rd_nodes.keys())

        for i in range(len(c_idx_list)):
            for j in range(i + 1, len(c_idx_list)):
                c1, c2 = c_idx_list[i], c_idx_list[j]
                d = topo_dist[c1, c2]
                if d == 1:
                    rd_edges.add((c1, c2))
                elif d == 2 and mol.GetAtomWithIdx(c1).GetIsAromatic() and mol.GetAtomWithIdx(c2).GetIsAromatic():
                    rd_edges.add((c1, c2))
        return list(rd_edges)


class GraphMatcher2:
    @staticmethod
    def match(
        dft_nodes: dict,
        dft_edges: List[Tuple[int, int]],
        rd_nodes: dict,
        rd_edges: List[Tuple[int, int]],
        fixed_anchors: Dict[int, int]
    ) -> Dict[int, int]:
        dft_edges_set = set(dft_edges) | {(v, u) for u, v in dft_edges}
        rd_edges_set = set(rd_edges) | {(v, u) for u, v in rd_edges}

        # Алгоритм 2 рассматривает ТОЛЬКО еще не соотнесенные сигналы и атомы
        unassigned_dft = [d for d in dft_nodes if d not in fixed_anchors]
        unassigned_rd = [r for r in rd_nodes if r not in fixed_anchors.values()]

        if not unassigned_dft or not unassigned_rd:
            return {}

        # Строим компоненты связности среди оставшихся DFT атомов
        adj = {n: set() for n in unassigned_dft}
        for u, v in dft_edges:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)

        visited = set()
        components = []
        for n in unassigned_dft:
            if n not in visited:
                comp, q = [], [n]
                visited.add(n)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)
                components.append(comp)

        # Сортируем компоненты: те, что связаны со старыми якорями Алгоритма 1, решаются первыми
        def comp_anchor_score(comp):
            return sum(1 for d in comp for a in fixed_anchors if (d, a) in dft_edges_set)

        components.sort(key=comp_anchor_score, reverse=True)

        mapping_2 = {}
        current_anchors = dict(fixed_anchors)
        used_rd = set(fixed_anchors.values())

        for comp in components:
            def node_anchor_score(d):
                return sum(1 for a in current_anchors if (d, a) in dft_edges_set)

            comp_sorted = sorted(comp, key=node_anchor_score, reverse=True)
            avail_rd = [r for r in unassigned_rd if r not in used_rd]

            comp_map = {}
            temp_used = set()

            def backtrack(idx):
                if idx == len(comp_sorted):
                    return True
                d_n = comp_sorted[idx]
                req_h = len(dft_nodes[d_n]["H"])

                for r_n in avail_rd:
                    # Алгоритм 2 ни при каких обстоятельствах не трогает атомы Алгоритма 1
                    if r_n in temp_used or r_n in fixed_anchors.values() or len(rd_nodes[r_n]["H"]) != req_h:
                        continue

                    consistent = True
                    # Согласованность с жесткими якорями Алгоритма 1 и ранее найденными парами
                    for m_d, m_r in current_anchors.items():
                        d_has = (d_n, m_d) in dft_edges_set
                        r_has = (r_n, m_r) in rd_edges_set
                        if d_has != r_has:
                            consistent = False
                            break

                    if not consistent:
                        continue

                    # Согласованность внутри текущего компонента
                    for m_d, m_r in comp_map.items():
                        d_has = (d_n, m_d) in dft_edges_set
                        r_has = (r_n, m_r) in rd_edges_set
                        if d_has != r_has:
                            consistent = False
                            break

                    if not consistent:
                        continue

                    comp_map[d_n] = r_n
                    temp_used.add(r_n)
                    if backtrack(idx + 1):
                        return True
                    del comp_map[d_n]
                    temp_used.remove(r_n)

                return False

            if backtrack(0):
                for d, r in comp_map.items():
                    # Защита: отбрасываем всё, что пытается задеть Алгоритм 1
                    if d not in fixed_anchors and r not in fixed_anchors.values():
                        mapping_2[d] = r
                        current_anchors[d] = r
                        used_rd.add(r)

        return mapping_2


# ==========================================
#         ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ==========================================

class DFTMapperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DFT to Structure Auto-Mapper (Двухстадийный)")
        self.resize(1400, 920)
        self.mol = None
        self.rd_nodes = {}
        self._init_ui()

    def _init_ui(self):
        main_w = QWidget()
        layout = QHBoxLayout(main_w)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)

        xyz_group = QGroupBox("1. Координаты структуры (XYZ)")
        xyz_layout = QVBoxLayout()
        self.xyz_input = QTextEdit()
        self.xyz_input.setPlaceholderText("Вставьте декартовы координаты (XYZ)...")
        xyz_layout.addWidget(self.xyz_input)
        self.btn_draw = QPushButton("Построить 2D граф")
        self.btn_draw.clicked.connect(self.load_structure)
        xyz_layout.addWidget(self.btn_draw)
        xyz_group.setLayout(xyz_layout)
        l_layout.addWidget(xyz_group)

        dft_group = QGroupBox("2. Матрица J-констант (DFT)")
        dft_layout = QVBoxLayout()
        self.dft_input = QTextEdit()
        self.dft_input.setPlaceholderText("Вставьте матрицу J-констант...")
        dft_layout.addWidget(self.dft_input)
        dft_group.setLayout(dft_layout)
        l_layout.addWidget(dft_group)

        self.btn_run = QPushButton("⚡ Сопоставить топологию (Алгоритм 1 -> Алгоритм 2)")
        self.btn_run.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 12px; font-size: 13px;")
        self.btn_run.clicked.connect(self.run_mapping)
        l_layout.addWidget(self.btn_run)
        splitter.addWidget(left_w)

        # Правая панель
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)

        self.mol_view = InteractiveMolLabel("2D Структура (Отрисуется после загрузки XYZ)")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.mol_view.setMinimumHeight(420)
        self.mol_view.atomClicked.connect(self.on_atom_clicked_on_image)
        r_layout.addWidget(self.mol_view)

        self.stats_lbl = QLabel("Ожидание данных для сопоставления...")
        self.stats_lbl.setStyleSheet("font-weight: bold; padding: 4px; color: #222;")
        r_layout.addWidget(self.stats_lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Статус", "ID углерода (DFT)", "Протоны (DFT)", "ID углерода (XYZ)", "Тип группы", "Источник сопоставления"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        r_layout.addWidget(self.table)

        splitter.addWidget(right_w)
        splitter.setSizes([450, 950])
        layout.addWidget(splitter)
        self.setCentralWidget(main_w)

    def load_structure(self):
        xyz_text = self.xyz_input.toPlainText().strip()
        if not xyz_text:
            return
        try:
            self.mol, self.rd_nodes, _ = RDKitGraphBuilder1.build_from_xyz(xyz_text)
            self._render_mol()
            self.stats_lbl.setText(f"В структуре найдено углеродов с H: {len(self.rd_nodes)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось распарсить XYZ:\n{e}")

    def _render_mol(self, highlights=None):
        if not self.mol:
            return

        d2d = rdMolDraw2D.MolDraw2DCairo(850, 420)
        opts = d2d.drawOptions()
        opts.addAtomIndices = True
        opts.highlightBondWidthMultiplier = 3

        if highlights:
            colors = {idx: (1.0, 0.5, 0.0) for idx in highlights}
            d2d.DrawMolecule(self.mol, highlightAtoms=highlights, highlightAtomColors=colors)
        else:
            d2d.DrawMolecule(self.mol)

        d2d.FinishDrawing()

        self.mol_view.atom_coords.clear()
        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            pt = d2d.GetDrawCoords(idx)
            self.mol_view.atom_coords[idx] = (pt.x, pt.y)

        pix = QPixmap()
        pix.loadFromData(d2d.GetDrawingText())
        self.mol_view.setPixmap(pix)

    def run_mapping(self):
        xyz_text = self.xyz_input.toPlainText().strip()
        dft_text = self.dft_input.toPlainText().strip()

        if not xyz_text or not dft_text:
            QMessageBox.warning(self, "Внимание", "Заполните оба текстовых поля.")
            return

        try:
            # 1. Загрузка 2D графа структуры
            self.mol, self.rd_nodes, rd_edges_1 = RDKitGraphBuilder1.build_from_xyz(xyz_text)
            self._render_mol()

            # 2. Парсинг матрицы DFT
            atoms, j_matrix = DFTParser1.parse_j_matrix(dft_text)

            # --- ПРОХОД 1: Исходная логика Алгоритма 1 ---
            dft_nodes, dft_edges_1 = DFTParser1.build_spin_graph(atoms, j_matrix)
            mapping_1 = GraphMatcher1.match(dft_nodes, dft_edges_1, self.rd_nodes, rd_edges_1)

            # --- ПРОХОД 2: Алгоритм 2 (только для свободных сигналов, с учетом якорей Алгоритма 1) ---
            dft_edges_2 = DFTParser2.build_spin_graph(atoms, j_matrix, dft_nodes)
            rd_edges_2 = RDKitGraphBuilder2.build_edges(self.mol, self.rd_nodes)
            mapping_2 = GraphMatcher2.match(
                dft_nodes, dft_edges_2, self.rd_nodes, rd_edges_2, fixed_anchors=mapping_1
            )

            # Формирование итоговой таблицы
            self.populate_table(mapping_1, mapping_2, dft_nodes, self.rd_nodes)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Сбой сопоставления:\n{e}")

    def populate_table(self, map_p1: dict, map_p2: dict, dft_nodes: dict, rd_nodes: dict):
        self.table.setRowCount(0)

        # 1. Результаты Алгоритма 1 (Зеленый)
        for d_idx, r_idx in map_p1.items():
            self._add_row("OK (Алг. 1)", d_idx, r_idx, dft_nodes, rd_nodes, "Алгоритм 1", QColor(220, 255, 220))

        # 2. Результаты Алгоритма 2 (Голубой)
        for d_idx, r_idx in map_p2.items():
            # Гарантия приоритета: если алгоритм 2 вдруг указал уже соотнесенный атом, не включаем
            if d_idx in map_p1 or r_idx in map_p1.values():
                continue
            self._add_row("OK (Алг. 2)", d_idx, r_idx, dft_nodes, rd_nodes, "Алгоритм 2 (по якорям)", QColor(215, 240, 255))

        all_matched_d = set(map_p1.keys()) | set(map_p2.keys())
        all_matched_r = set(map_p1.values()) | set(map_p2.values())

        # 3. Не соотнесенные DFT сигналы (Розовый)
        for d_idx in dft_nodes:
            if d_idx not in all_matched_d:
                self._add_row("Только DFT", d_idx, None, dft_nodes, rd_nodes, "Не соотнесено", QColor(255, 225, 225))

        # 4. Не соотнесенные углероды структуры XYZ (Желтый)
        for r_idx in rd_nodes:
            if r_idx not in all_matched_r:
                self._add_row("Только XYZ", None, r_idx, dft_nodes, rd_nodes, "Не соотнесено", QColor(255, 255, 205))

        tot_c = len(rd_nodes)
        n_p1 = len(map_p1)
        n_p2 = len([k for k, v in map_p2.items() if k not in map_p1 and v not in map_p1.values()])
        n_tot = n_p1 + n_p2
        self.stats_lbl.setText(
            f"Сопоставлено всего: {n_tot} из {tot_c} | "
            f"Алгоритм 1: {n_p1} | "
            f"Алгоритм 2 (добор по якорям): {n_p2} | "
            f"Осталось не соотнесено: {tot_c - n_tot}"
        )

    def _add_row(self, status, d_idx, r_idx, dft_nodes, rd_nodes, method, color):
        row = self.table.rowCount()
        self.table.insertRow(row)

        item_status = QTableWidgetItem(status)
        item_status.setBackground(color)
        item_status.setData(Qt.ItemDataRole.UserRole, r_idx if r_idx is not None else -1)

        if d_idx is not None:
            h_list = dft_nodes[d_idx]["H"]
            str_d_idx = f"C {d_idx}"
            str_h = f"H: [{', '.join(map(str, h_list))}]"
            str_type = f"CH{len(h_list)}"
        else:
            str_d_idx, str_h, str_type = "—", "—", "—"

        if r_idx is not None:
            str_r_idx = f"C {r_idx}"
            if str_type == "—":
                str_type = f"CH{len(rd_nodes[r_idx]['H'])}"
        else:
            str_r_idx = "—"

        items = [
            item_status,
            QTableWidgetItem(str_d_idx),
            QTableWidgetItem(str_h),
            QTableWidgetItem(str_r_idx),
            QTableWidgetItem(str_type),
            QTableWidgetItem(method)
        ]

        for i, item in enumerate(items):
            if i > 0:
                item.setBackground(color)
            self.table.setItem(row, i, item)

    def on_atom_clicked_on_image(self, atom_idx):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item.data(Qt.ItemDataRole.UserRole) == atom_idx:
                self.table.selectRow(row)
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                return
        self._render_mol([atom_idx])

    def on_table_row_selected(self):
        items = self.table.selectedItems()
        if not items:
            return
        r_idx = items[0].data(Qt.ItemDataRole.UserRole)
        if r_idx != -1 and r_idx in self.rd_nodes:
            self._render_mol([r_idx])
        else:
            self._render_mol()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DFTMapperApp()
    window.show()
    sys.exit(app.exec())