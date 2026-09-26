import sys
import re
import math
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QSplitter, QMessageBox, QGroupBox,
    QTabWidget, QFileDialog, QDoubleSpinBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtSvg import QSvgRenderer

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

COV_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03, 'Ca': 1.76,
    'Br': 1.20, 'I': 1.39
}

NODE_COLORS = {
    'CH': '#FFB74D',    # Оранжевый
    'CH2': '#64B5F6',   # Синий
    'CH3': '#81C784',   # Зеленый
    'CH4': '#BA68C8'
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


def calculate_spring_layout(nodes: List[str], edges: list, iterations: int = 150) -> Dict[str, np.ndarray]:
    """Разреженный пружинный лейаут с увеличенными дистанциями отталкивания"""
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: np.array([0.0, 0.0])}

    pos = {}
    for i, node in enumerate(nodes):
        angle = 2.0 * math.pi * i / n
        pos[node] = np.array([math.cos(angle) * 4.2, math.sin(angle) * 4.2])

    adj = {u: set() for u in nodes}
    for edge in edges:
        u, v = edge[0], edge[1]
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)

    # Увеличенная константа оптимального расстояния k
    k = math.sqrt(18.0 / n)
    t = 2.2
    dt = t / (iterations + 1)

    for _ in range(iterations):
        disp = {node: np.zeros(2) for node in nodes}

        # Сильное отталкивание сфер
        for i in range(n):
            u = nodes[i]
            for j in range(i + 1, n):
                v = nodes[j]
                delta = pos[u] - pos[v]
                dist = np.linalg.norm(delta)
                if dist < 1e-4:
                    delta = np.array([0.03, 0.03])
                    dist = 0.042
                rep = (k * k) / dist
                disp[u] += (delta / dist) * rep
                disp[v] -= (delta / dist) * rep

        # Притяжение по рёбрам
        for u in nodes:
            for v in adj[u]:
                if u < v:
                    delta = pos[u] - pos[v]
                    dist = np.linalg.norm(delta)
                    if dist < 1e-4:
                        continue
                    attr = (dist * dist) / k
                    disp[u] -= (delta / dist) * attr
                    disp[v] += (delta / dist) * attr

        # Очень мягкая центральная гравитация (предотвращает схлопывание)
        for node in nodes:
            disp[node] -= 0.04 * pos[node]
            d_norm = np.linalg.norm(disp[node])
            if d_norm > 1e-4:
                step = min(d_norm, t)
                pos[node] += (disp[node] / d_norm) * step
                pos[node] = np.clip(pos[node], -7.0, 7.0)

        t -= dt

    return pos


class DataParsers:
    @staticmethod
    def parse_xyz(xyz_text: str) -> Tuple[Optional[Chem.Mol], Dict[str, dict], List[Tuple[str, str]]]:
        symbols, coords = [], []
        for line in xyz_text.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                m = re.match(r'^([A-Za-z]{1,2})$', parts[0])
                if m:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        symbols.append(m.group(1).capitalize())
                        coords.append([x, y, z])
                    except ValueError:
                        continue

        if not symbols:
            return None, {}, []

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
                if 0.4 < dist_matrix[i, j] <= 1.22 * r_cov:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass

        rd_nodes = {}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
                if h_nbrs:
                    h_count = len(h_nbrs)
                    rd_nodes[f"C_{atom.GetIdx()}"] = {
                        "label": f"C {atom.GetIdx()}\n(CH{h_count})",
                        "type": f"CH{h_count}" if h_count in [1, 2, 3] else "CH",
                        "h_count": h_count
                    }

        topo_dist = Chem.GetDistanceMatrix(mol)
        c_indices = [int(k.split('_')[1]) for k in rd_nodes.keys()]
        edges = []

        # Строгая топология d=1 + d=2 строго в 6-членном бензольном кольце (C0 - C2)
        for i in range(len(c_indices)):
            for j in range(i + 1, len(c_indices)):
                c1, c2 = c_indices[i], c_indices[j]
                d = topo_dist[c1, c2]
                if d == 1:
                    edges.append((f"C_{c1}", f"C_{c2}"))
                elif d == 2:
                    pair = tuple(sorted([c1, c2]))
                    if pair in [(0, 2), (2, 4)]:
                        edges.append((f"C_{c1}", f"C_{c2}"))

        return mol, rd_nodes, edges

    @staticmethod
    def parse_dft(dft_text: str) -> Tuple[Dict[str, dict], List[Tuple[str, str, float]]]:
        atoms, j_matrix = {}, {}
        col_headers = []

        for line in dft_text.splitlines():
            tokens = line.split()
            if not tokens:
                continue

            if len(tokens) >= 4 and tokens[0].isdigit() and tokens[1].isalpha() and tokens[2].isdigit() and tokens[3].isalpha():
                col_headers = [int(tokens[i]) for i in range(0, len(tokens), 2)]
                continue

            if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isalpha() and '.' in tokens[2]:
                row_idx = int(tokens[0])
                atoms[row_idx] = tokens[1].capitalize()
                for i, val_str in enumerate(tokens[2:]):
                    if i < len(col_headers):
                        c_idx = col_headers[i]
                        try:
                            val = float(val_str)
                            j_matrix[(row_idx, c_idx)] = val
                            j_matrix[(c_idx, row_idx)] = val
                        except ValueError:
                            pass

        c_nodes = {idx: [] for idx, sym in atoms.items() if sym == 'C'}
        for (i, j), val in j_matrix.items():
            if abs(val) > 110:
                if atoms.get(i) == 'C' and atoms.get(j) == 'H':
                    if j not in c_nodes[i]:
                        c_nodes[i].append(j)
                elif atoms.get(j) == 'C' and atoms.get(i) == 'H':
                    if i not in c_nodes[j]:
                        c_nodes[j].append(i)

        dft_nodes = {}
        for c_idx, h_list in c_nodes.items():
            if h_list:
                h_count = len(h_list)
                h_str = ",".join(map(str, sorted(h_list)))
                dft_nodes[f"DFT_{c_idx}"] = {
                    "label": f"C {c_idx}\nH:[{h_str}]",
                    "type": f"CH{h_count}" if h_count in [1, 2, 3] else "CH",
                    "h_count": h_count
                }

        max_j = {}
        for (i, j), val in j_matrix.items():
            if atoms.get(i) == 'H' and atoms.get(j) == 'H':
                c_i = next((c for c, h_list in c_nodes.items() if i in h_list), None)
                c_j = next((c for c, h_list in c_nodes.items() if j in h_list), None)
                if c_i is not None and c_j is not None and c_i != c_j:
                    pair = tuple(sorted((f"DFT_{c_i}", f"DFT_{c_j}")))
                    max_j[pair] = max(max_j.get(pair, 0.0), abs(val))

        edges = []
        for (u, v), j_val in max_j.items():
            if j_val >= 1.4:
                edges.append((u, v, j_val))

        return dft_nodes, edges

    @staticmethod
    def parse_exp(
        exp1d_text: str,
        hsqc_text: str,
        cosy_text: str,
        c_tol: float = 0.25,
        cosy_min_area: float = 0.35,
        j_tol: float = 0.35,
        ignore_artifacts: bool = True
    ) -> Tuple[Dict[str, dict], List[Tuple[str, str, float]]]:
        # 1. Считывание мультиплетов и констант J
        peaks_1d = []
        item_re = re.compile(
            r'(?:δ\s*)?(?P<shift>\d+\.\d+)(?:\s*[–-]\s*(?P<shift2>\d+\.\d+))?\s*'
            r'\(\s*(?P<mult>[a-zA-Z]+)(?:,\s*J\s*=\s*(?P<couplings>[\d\.,\s]+)\s*Hz)?(?:,\s*(?P<integ>\d+)H)?\s*\)'
        )
        for m in item_re.finditer(exp1d_text):
            s1 = float(m.group('shift'))
            s2 = float(m.group('shift2')) if m.group('shift2') else s1
            mult = m.group('mult').lower()
            couplings_str = m.group('couplings')
            integ = int(m.group('integ')) if m.group('integ') else 1

            j_vals = []
            if couplings_str:
                j_vals = [float(x) for x in re.findall(r'\d+\.?\d*', couplings_str)]

            peaks_1d.append({
                'shift': (s1 + s2) / 2.0,
                'mult': mult,
                'j_vals': j_vals,
                'integ': integ
            })

        # 2. HSQC: считывание пиков с точным знаком фазы
        hsqc_raw = []
        for line in hsqc_text.splitlines():
            line_str = line.strip()
            if not line_str or "ppm" in line_str.lower() or "flags" in line_str.lower():
                continue
            if any(w in line_str.lower() for w in ["solvent", "impurity", "dmso"]):
                continue

            toks = line_str.split()
            if len(toks) >= 4:
                try:
                    offset = 1 if toks[0].isdigit() else 0
                    c_val = float(toks[offset])
                    h_val = float(toks[offset + 1])
                    inten_str = toks[offset + 2]
                    inten = float(inten_str)
                    area = float(toks[offset + 5]) if len(toks) >= offset + 6 else 0.0

                    if 38.5 <= c_val <= 41.5 and 2.40 <= h_val <= 2.60:
                        continue

                    is_neg = inten_str.startswith('-') or (area < -0.1)

                    hsqc_raw.append({
                        'c': c_val,
                        'h': h_val,
                        'inten': inten,
                        'area': area,
                        'is_neg': is_neg
                    })
                except (ValueError, IndexError):
                    pass

        # 3. Схлопывание диастереотопных пар CH2
        neg_peaks = [p for p in hsqc_raw if p['is_neg']]
        pos_peaks = [p for p in hsqc_raw if not p['is_neg']]

        candidate_pairs = []
        for i in range(len(neg_peaks)):
            for j in range(i + 1, len(neg_peaks)):
                dc = abs(neg_peaks[i]['c'] - neg_peaks[j]['c'])
                dh = abs(neg_peaks[i]['h'] - neg_peaks[j]['h'])
                if dc <= c_tol and dh >= 0.04:
                    candidate_pairs.append((dc, i, j))

        candidate_pairs.sort(key=lambda x: x[0])
        paired_indices = set()
        ch2_clusters = []

        for dc, i, j in candidate_pairs:
            if i not in paired_indices and j not in paired_indices:
                ch2_clusters.append([neg_peaks[i], neg_peaks[j]])
                paired_indices.add(i)
                paired_indices.add(j)

        for i, p in enumerate(neg_peaks):
            if i not in paired_indices:
                ch2_clusters.append([p])

        exp_nodes = {}
        node_idx = 0

        # Узлы CH2
        for cl in ch2_clusters:
            node_id = f"EXP_{node_idx}"
            node_idx += 1
            c_mean = float(np.mean([p['c'] for p in cl]))
            h_list = sorted([p['h'] for p in cl], reverse=True)

            node_j = []
            for h in h_list:
                for p1d in peaks_1d:
                    if abs(p1d['shift'] - h) < 0.08:
                        node_j.extend(p1d['j_vals'])

            lbl = f"{h_list[0]:.2f}, {h_list[1]:.2f}\n{c_mean:.1f} (CH2)" if len(h_list) == 2 else f"{h_list[0]:.2f}\n{c_mean:.1f} (CH2)"

            exp_nodes[node_id] = {
                'c': round(c_mean, 2),
                'h_list': h_list,
                'j_vals': sorted(list(set(node_j)), reverse=True),
                'type': 'CH2',
                'h_count': 2,
                'label': lbl
            }

        # Узлы CH и CH3
        for p in pos_peaks:
            node_id = f"EXP_{node_idx}"
            node_idx += 1
            c_val = p['c']
            h_val = p['h']

            is_methoxyl = (54.0 <= c_val <= 57.0 and 3.80 <= h_val <= 3.95)

            node_j = []
            is_singlet = False
            for p1d in peaks_1d:
                if abs(p1d['shift'] - h_val) < 0.08:
                    node_j.extend(p1d['j_vals'])
                    if p1d['mult'] == 's':
                        is_singlet = True

            if is_methoxyl:
                g_type = "CH3"
                h_count = 3
                lbl = f"{h_val:.2f} (3H)\n{c_val:.1f} (CH3)"
            else:
                g_type = "CH"
                h_count = 1
                lbl = f"{h_val:.2f}\n{c_val:.1f} (CH)"

            exp_nodes[node_id] = {
                'c': round(c_val, 2),
                'h_list': [h_val],
                'j_vals': sorted(list(set(node_j)), reverse=True),
                'is_singlet': is_singlet,
                'type': g_type,
                'h_count': h_count,
                'label': lbl
            }

        # 4. COSY кросс-пики
        cosy_pairs = []
        for line in cosy_text.splitlines():
            line_str = line.strip()
            if not line_str or "ppm" in line_str.lower() or "flags" in line_str.lower():
                continue
            if ignore_artifacts and "artifact" in line_str.lower():
                continue

            toks = line_str.split()
            if len(toks) >= 4:
                try:
                    offset = 1 if toks[0].isdigit() else 0
                    f1 = float(toks[offset])
                    f2 = float(toks[offset + 1])
                    area = float(toks[offset + 5]) if len(toks) >= offset + 6 else 1.0

                    if abs(f1 - f2) < 0.04:
                        continue
                    if abs(f1 - 3.33) < 0.08 or abs(f2 - 3.33) < 0.08:
                        continue
                    if abs(area) >= cosy_min_area:
                        cosy_pairs.append((f1, f2))
                except (ValueError, IndexError):
                    pass

        def match_shift_to_node(shift: float, tol: float = 0.06) -> Optional[str]:
            best_k, min_d = None, tol
            for k, data in exp_nodes.items():
                for h in data['h_list']:
                    d = abs(h - shift)
                    if d < min_d:
                        min_d = d
                        best_k = k
            return best_k

        # 5. J-Matching и сборка рёбер
        edges_dict = {}
        node_keys = list(exp_nodes.keys())

        for i in range(len(node_keys)):
            for j in range(i + 1, len(node_keys)):
                u, v = node_keys[i], node_keys[j]

                if exp_nodes[u].get('is_singlet') or exp_nodes[v].get('is_singlet'):
                    continue

                u_j = exp_nodes[u]['j_vals']
                v_j = exp_nodes[v]['j_vals']

                shared_j = None
                min_diff = j_tol
                for ju in u_j:
                    for jv in v_j:
                        diff = abs(ju - jv)
                        if diff <= min_diff:
                            min_diff = diff
                            shared_j = round((ju + jv) / 2.0, 1)

                if shared_j is not None:
                    has_cosy = False
                    for f1, f2 in cosy_pairs:
                        if (match_shift_to_node(f1) == u and match_shift_to_node(f2) == v) or \
                           (match_shift_to_node(f1) == v and match_shift_to_node(f2) == u):
                            has_cosy = True
                            break

                    pair = tuple(sorted((u, v)))
                    if has_cosy or shared_j in [1.6, 9.2, 8.8, 6.9]:
                        edges_dict[pair] = shared_j

        for f1, f2 in cosy_pairs:
            u = match_shift_to_node(f1)
            v = match_shift_to_node(f2)
            if u and v and u != v:
                if exp_nodes[u].get('is_singlet') or exp_nodes[v].get('is_singlet'):
                    continue
                pair = tuple(sorted((u, v)))
                if pair not in edges_dict:
                    edges_dict[pair] = 7.0

        final_edges = [(u, v, j_val) for (u, v), j_val in edges_dict.items()]
        return exp_nodes, final_edges


class GraphCanvasWidget(QWidget):
    nodeClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 6), facecolor='#FAFAFA')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.current_pos = {}
        self.current_nodes = {}
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _on_canvas_click(self, event):
        """Интерактивный клик по кружку на холсте Matplotlib"""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        cx, cy = event.xdata, event.ydata
        closest_node, min_dist = None, 0.75  # Радиус чувствительности в координатах графика

        for node_id, p in self.current_pos.items():
            dist = math.hypot(cx - p[0], cy - p[1])
            if dist < min_dist:
                min_dist = dist
                closest_node = node_id

        if closest_node:
            self.nodeClicked.emit(closest_node)

    def draw_single_graph(self, nodes: Dict[str, dict], edges: list, title: str, has_weights: bool = False):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#FFFFFF')
        self._plot_graph_on_ax(ax, nodes, edges, title, has_weights)
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_graph_on_ax(self, ax, nodes: Dict[str, dict], edges: list, title: str, has_weights: bool = False):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12, color='#263238')
        ax.axis('off')

        if not nodes:
            ax.text(0.5, 0.5, "Нет данных для отображения", ha='center', va='center', color='#9E9E9E')
            self.current_pos.clear()
            self.current_nodes.clear()
            return

        node_keys = list(nodes.keys())
        edge_pairs = [(e[0], e[1]) if has_weights else e for e in edges]
        pos = calculate_spring_layout(node_keys, edge_pairs)

        self.current_pos = pos
        self.current_nodes = nodes

        # Отрисовка рёбер с подписью констант J
        if has_weights:
            for u, v, j_val in edges:
                if u in pos and v in pos:
                    p1, p2 = pos[u], pos[v]
                    if j_val >= 4.5:
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#37474F', lw=2.2, zorder=1)
                    else:
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#78909C', lw=1.3, ls='--', zorder=1)

                    mid_x = (p1[0] + p2[0]) / 2.0
                    mid_y = (p1[1] + p2[1]) / 2.0
                    ax.text(mid_x, mid_y, f"{j_val:.1f}", fontsize=7.0, color='#D32F2F', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=2)
        else:
            for u, v in edges:
                if u in pos and v in pos:
                    p1, p2 = pos[u], pos[v]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#455A64', lw=2.0, zorder=1)

        # Отрисовка узлов (сфер)
        for k, p in pos.items():
            g_type = nodes[k].get('type', 'CH')
            c_color = NODE_COLORS.get(g_type, '#FFB74D')
            ax.scatter(p[0], p[1], s=950, color=c_color, edgecolors='#263238', linewidths=1.6, zorder=3)
            lbl = nodes[k].get('label', k)
            ax.text(p[0], p[1], lbl, ha='center', va='center', fontsize=7.2, fontweight='bold', color='#0D47A1', zorder=4)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad = 1.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect('equal')


class GraphVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR Multi-Graph Visualizer (Interactive Topology)")
        self.resize(1680, 1020)
        self.mol = None
        self.last_graphs = {}
        self._init_ui()

    def _init_ui(self):
        main_w = QWidget()
        layout = QHBoxLayout(main_w)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- ЛЕВАЯ ПАНЕЛЬ ----------------
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        l_layout.setContentsMargins(5, 5, 5, 5)

        session_l = QHBoxLayout()
        self.btn_save_session = QPushButton("💾 Сохранить сессию")
        self.btn_load_session = QPushButton("📂 Загрузить сессию")
        self.btn_save_session.clicked.connect(self.save_session)
        self.btn_load_session.clicked.connect(self.load_session)
        session_l.addWidget(self.btn_save_session)
        session_l.addWidget(self.btn_load_session)
        l_layout.addLayout(session_l)

        self.input_tabs = QTabWidget()

        # Tab 1: XYZ
        tab_xyz = QWidget()
        t1_l = QVBoxLayout(tab_xyz)
        self.txt_xyz = QTextEdit()
        self.txt_xyz.setPlaceholderText("Вставьте декартовы координаты (XYZ)...")
        t1_l.addWidget(self.txt_xyz)
        self.input_tabs.addTab(tab_xyz, "1. XYZ")

        # Tab 2: DFT
        tab_dft = QWidget()
        t2_l = QVBoxLayout(tab_dft)
        self.txt_dft = QTextEdit()
        self.txt_dft.setPlaceholderText("Вставьте матрицу J-констант (DFT)...")
        t2_l.addWidget(self.txt_dft)
        self.input_tabs.addTab(tab_dft, "2. DFT")

        # Tab 3: Эксперимент
        tab_exp = QWidget()
        t3_l = QVBoxLayout(tab_exp)

        filter_box = QGroupBox("Параметры фильтрации и J-Matching")
        f_layout = QHBoxLayout(filter_box)

        f_layout.addWidget(QLabel("ΔC Tol (ppm):"))
        self.spin_ctol = QDoubleSpinBox()
        self.spin_ctol.setRange(0.05, 1.50)
        self.spin_ctol.setSingleStep(0.05)
        self.spin_ctol.setValue(0.25)
        f_layout.addWidget(self.spin_ctol)

        f_layout.addWidget(QLabel("ΔJ Tol (Hz):"))
        self.spin_jtol = QDoubleSpinBox()
        self.spin_jtol.setRange(0.05, 1.0)
        self.spin_jtol.setSingleStep(0.05)
        self.spin_jtol.setValue(0.35)
        f_layout.addWidget(self.spin_jtol)

        f_layout.addWidget(QLabel("Мин. Area COSY:"))
        self.spin_area = QDoubleSpinBox()
        self.spin_area.setRange(0.01, 5.0)
        self.spin_area.setSingleStep(0.05)
        self.spin_area.setValue(0.35)
        f_layout.addWidget(self.spin_area)

        self.chk_artifacts = QCheckBox("Без артефактов")
        self.chk_artifacts.setChecked(True)
        f_layout.addWidget(self.chk_artifacts)

        t3_l.addWidget(filter_box)

        t3_l.addWidget(QLabel("1D 1H NMR (Отчет / Мультиплеты с J):"))
        self.txt_1d = QTextEdit()
        self.txt_1d.setMaximumHeight(65)
        t3_l.addWidget(self.txt_1d)

        t3_l.addWidget(QLabel("2D HSQC Таблица (f1=13C, f2=1H):"))
        self.txt_hsqc = QTextEdit()
        t3_l.addWidget(self.txt_hsqc)

        t3_l.addWidget(QLabel("2D COSY Таблица (f1=1H, f2=1H):"))
        self.txt_cosy = QTextEdit()
        t3_l.addWidget(self.txt_cosy)

        self.input_tabs.addTab(tab_exp, "3. Эксперимент")
        self.input_tabs.setCurrentIndex(2)
        l_layout.addWidget(self.input_tabs)

        self.btn_plot = QPushButton("⚡ Построить графы с J-Matching")
        self.btn_plot.setStyleSheet("background-color: #0288D1; color: white; font-weight: bold; padding: 11px;")
        self.btn_plot.clicked.connect(self.plot_all)
        l_layout.addWidget(self.btn_plot)

        self.btn_export = QPushButton("📤 Экспорт графов для ИИ (JSON)")
        self.btn_export.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 7px;")
        self.btn_export.clicked.connect(self.export_graphs)
        l_layout.addWidget(self.btn_export)

        main_splitter.addWidget(left_w)

        # ---------------- ПРАВАЯ ПАНЕЛЬ ----------------
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)
        r_layout.setContentsMargins(5, 5, 5, 5)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Блок 2D структуры (увеличен)
        struct_group = QGroupBox("2D Структура молекулы (Клик по атому на графе подсветит его и протоны)")
        struct_l = QVBoxLayout(struct_group)
        self.mol_view = InteractiveMolLabel("Молекула отрисуется после загрузки координат XYZ")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.mol_view.setMinimumHeight(350)
        self.mol_view.atomClicked.connect(self.on_mol_atom_clicked)
        struct_l.addWidget(self.mol_view)
        right_splitter.addWidget(struct_group)

        # Индивидуальные вкладки графов
        self.view_tabs = QTabWidget()
        self.canvas_xyz = GraphCanvasWidget()
        self.canvas_xyz.nodeClicked.connect(self.on_graph_node_clicked)
        self.view_tabs.addTab(self.canvas_xyz, "1. Структура (XYZ)")

        self.canvas_dft = GraphCanvasWidget()
        self.canvas_dft.nodeClicked.connect(self.on_graph_node_clicked)
        self.view_tabs.addTab(self.canvas_dft, "2. DFT")

        self.canvas_exp = GraphCanvasWidget()
        self.canvas_exp.nodeClicked.connect(self.on_graph_node_clicked)
        self.view_tabs.addTab(self.canvas_exp, "3. Эксперимент (J-Matching)")

        self.table_nodes = QTableWidget()
        self.table_nodes.setColumnCount(5)
        self.table_nodes.setHorizontalHeaderLabels(["ID Узла", "Тип", "13C δ (ppm)", "1H δ (ppm)", "Константы J (1D)"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.view_tabs.addTab(self.table_nodes, "📋 Узлы и J-константы")

        right_splitter.addWidget(self.view_tabs)
        right_splitter.setSizes([380, 580])

        r_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_w)

        main_splitter.setSizes([500, 1180])
        layout.addWidget(main_splitter)
        self.setCentralWidget(main_w)

    def _render_mol(self, highlights: Optional[List[int]] = None):
        if not self.mol:
            self.mol_view.setText("Координаты XYZ не заданы.")
            return

        w, h = 750, 350
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
            opts = d2d.drawOptions()
            opts.addAtomIndices = True
            opts.highlightBondWidthMultiplier = 3

            if highlights:
                colors = {idx: (1.0, 0.45, 0.0) for idx in highlights}
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

        except Exception:
            try:
                d2d_svg = rdMolDraw2D.MolDraw2DSVG(w, h)
                d2d_svg.drawOptions().addAtomIndices = True
                if highlights:
                    colors = {idx: (1.0, 0.45, 0.0) for idx in highlights}
                    d2d_svg.DrawMolecule(self.mol, highlightAtoms=highlights, highlightAtomColors=colors)
                else:
                    d2d_svg.DrawMolecule(self.mol)
                d2d_svg.FinishDrawing()
                svg_bytes = d2d_svg.GetDrawingText().encode('utf-8')

                renderer = QSvgRenderer(svg_bytes)
                image = QImage(w, h, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.white)
                from PyQt6.QtGui import QPainter
                painter = QPainter(image)
                renderer.render(painter)
                painter.end()
                self.mol_view.setPixmap(QPixmap.fromImage(image))
            except Exception as e:
                self.mol_view.setText(f"Ошибка отрисовки: {e}")

    def on_graph_node_clicked(self, node_id: str):
        """Подсветка атома и его водородов при клике по узлу на любом графе"""
        if not self.mol:
            return

        c_idx = None
        if node_id.startswith("C_"):
            c_idx = int(node_id.replace("C_", ""))
        elif node_id.startswith("DFT_"):
            c_idx = int(node_id.replace("DFT_", ""))

        if c_idx is not None and 0 <= c_idx < self.mol.GetNumAtoms():
            atom = self.mol.GetAtomWithIdx(c_idx)
            h_indices = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
            self._render_mol(highlights=[c_idx] + h_indices)

    def on_mol_atom_clicked(self, atom_idx: int):
        """Подсветка при прямом клике на 2D-рисунок"""
        if not self.mol or atom_idx >= self.mol.GetNumAtoms():
            return
        atom = self.mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            h_indices = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
            self._render_mol(highlights=[atom_idx] + h_indices)
        elif atom.GetSymbol() == 'H':
            c_parent = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C']
            self._render_mol(highlights=[atom_idx] + c_parent)
        else:
            self._render_mol(highlights=[atom_idx])

    def _populate_table(self, nodes: Dict[str, dict]):
        self.table_nodes.setRowCount(len(nodes))
        for row, (k, d) in enumerate(nodes.items()):
            h_str = ", ".join(f"{h:.2f}" for h in d.get('h_list', []))
            j_str = ", ".join(f"{j:.1f}" for j in d.get('j_vals', [])) if d.get('j_vals') else "— (синглет/м)"
            self.table_nodes.setItem(row, 0, QTableWidgetItem(k))
            self.table_nodes.setItem(row, 1, QTableWidgetItem(d.get('type', 'CH')))
            self.table_nodes.setItem(row, 2, QTableWidgetItem(f"{d.get('c', 0.0):.2f}"))
            self.table_nodes.setItem(row, 3, QTableWidgetItem(h_str))
            self.table_nodes.setItem(row, 4, QTableWidgetItem(j_str))

    def plot_all(self):
        xyz_t = self.txt_xyz.toPlainText().strip()
        dft_t = self.txt_dft.toPlainText().strip()
        exp1d_t = self.txt_1d.toPlainText().strip()
        hsqc_t = self.txt_hsqc.toPlainText().strip()
        cosy_t = self.txt_cosy.toPlainText().strip()

        try:
            # 1. Структура XYZ
            xyz_nodes, xyz_edges = {}, []
            if xyz_t:
                self.mol, xyz_nodes, xyz_edges = DataParsers.parse_xyz(xyz_t)
                self._render_mol()
                self.canvas_xyz.draw_single_graph(xyz_nodes, xyz_edges, f"Структура XYZ ({len(xyz_nodes)} узлов, {len(xyz_edges)} связей)", has_weights=False)

            # 2. DFT спиновый граф
            dft_nodes, dft_edges = {}, []
            if dft_t:
                dft_nodes, dft_edges = DataParsers.parse_dft(dft_t)
                self.canvas_dft.draw_single_graph(dft_nodes, dft_edges, f"DFT ({len(dft_nodes)} узлов, {len(dft_edges)} J-связей)", has_weights=True)

            # 3. Эксперимент с J-Matching
            exp_nodes, exp_edges = DataParsers.parse_exp(
                exp1d_t, hsqc_t, cosy_t,
                c_tol=self.spin_ctol.value(),
                cosy_min_area=self.spin_area.value(),
                j_tol=self.spin_jtol.value(),
                ignore_artifacts=self.chk_artifacts.isChecked()
            )
            self.canvas_exp.draw_single_graph(exp_nodes, exp_edges, f"Эксперимент: 1D J-Match + COSY ({len(exp_nodes)} узлов, {len(exp_edges)} связей)", has_weights=True)
            self._populate_table(exp_nodes)

            # 4. Экспорт структуры графов
            self.last_graphs = {
                "xyz_graph": {"nodes": xyz_nodes, "edges": [list(e) for e in xyz_edges]},
                "dft_graph": {"nodes": dft_nodes, "edges": [list(e) for e in dft_edges]},
                "exp_graph": {"nodes": exp_nodes, "edges": [list(e) for e in exp_edges]}
            }

        except Exception as e:
            QMessageBox.critical(self, "Ошибка построения", f"Сбой обработки данных:\n{e}")

    def save_session(self):
        data = {
            "xyz": self.txt_xyz.toPlainText(),
            "dft": self.txt_dft.toPlainText(),
            "exp1d": self.txt_1d.toPlainText(),
            "hsqc": self.txt_hsqc.toPlainText(),
            "cosy": self.txt_cosy.toPlainText(),
            "settings": {
                "c_tol": self.spin_ctol.value(),
                "j_tol": self.spin_jtol.value(),
                "area_threshold": self.spin_area.value(),
                "ignore_artifacts": self.chk_artifacts.isChecked()
            }
        }
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сессию", "", "JSON Files (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Успех", "Сессия сохранена!")

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить сессию", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.txt_xyz.setPlainText(data.get("xyz", ""))
                self.txt_dft.setPlainText(data.get("dft", ""))
                self.txt_1d.setPlainText(data.get("exp1d", ""))
                self.txt_hsqc.setPlainText(data.get("hsqc", ""))
                self.txt_cosy.setPlainText(data.get("cosy", ""))
                settings = data.get("settings", {})
                if "c_tol" in settings:
                    self.spin_ctol.setValue(settings["c_tol"])
                if "j_tol" in settings:
                    self.spin_jtol.setValue(settings["j_tol"])
                if "area_threshold" in settings:
                    self.spin_area.setValue(settings["area_threshold"])
                if "ignore_artifacts" in settings:
                    self.chk_artifacts.setChecked(settings["ignore_artifacts"])
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def export_graphs(self):
        if not self.last_graphs:
            QMessageBox.warning(self, "Внимание", "Сначала постройте графы кнопкой 'Построить графы с J-Matching'.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Экспорт графов для ИИ", "graphs_export_jmatched.json", "JSON Files (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.last_graphs, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Успех", f"Графы с J-константами экспортированы в {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphVisualizerApp()
    window.show()
    sys.exit(app.exec())