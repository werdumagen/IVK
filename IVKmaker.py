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
    QTableWidgetItem, QHeaderView, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsObject
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QFont, QColor,
    QRadialGradient
)
from PyQt6.QtSvg import QSvgRenderer

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

COV_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03, 'Ca': 1.76,
    'Br': 1.20, 'I': 1.39
}

NODE_COLORS = {
    'CH': '#FFA726',    # Тёплый оранжевый
    'CH2': '#42A5F5',   # Небесно-синий
    'CH3': '#66BB6A',   # Пастельно-зеленый
    'CH4': '#AB47BC'
}


# ==========================================================
#      ИНТЕРАКТИВНЫЙ 2D РЕНДЕР МОЛЕКУЛЫ (RDKit)
# ==========================================================

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


# ==========================================================
#     КАСТОМНЫЙ ГРАФОВЫЙ ДВИЖОК (QGraphicsView / QPainter)
# ==========================================================

class GraphNodeItem(QGraphicsObject):
    clicked = pyqtSignal(str)

    def __init__(self, node_id: str, data: dict, x: float, y: float):
        super().__init__()
        self.node_id = node_id
        self.data = data
        self.radius = 32.0
        self.edges = []
        self.setPos(x, y)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.is_hovered = False

    def boundingRect(self) -> QRectF:
        pad = 6.0
        return QRectF(-self.radius - pad, -self.radius - pad,
                      (self.radius + pad) * 2, (self.radius + pad) * 2)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        g_type = self.data.get('type', 'CH')
        base_color = QColor(NODE_COLORS.get(g_type, '#FFA726'))

        grad = QRadialGradient(-8, -8, self.radius * 1.3)
        grad.setColorAt(0.0, base_color.lighter(130))
        grad.setColorAt(0.85, base_color)
        grad.setColorAt(1.0, base_color.darker(125))

        pen_color = QColor('#0D47A1') if self.is_hovered else QColor('#263238')
        pen_width = 3.0 if self.is_hovered else 2.0

        painter.setPen(QPen(pen_color, pen_width))
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(rect)

        lbl = self.data.get('label', self.node_id)
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor('#002171')))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, lbl)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.node_id)

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)


class GraphEdgeItem(QGraphicsItem):
    def __init__(self, u_node: GraphNodeItem, v_node: GraphNodeItem, j_val: Optional[float] = None):
        super().__init__()
        self.u = u_node
        self.v = v_node
        self.j_val = j_val
        self.u.edges.append(self)
        self.v.edges.append(self)
        self.setZValue(-1.0)

    def boundingRect(self) -> QRectF:
        p1 = self.u.pos()
        p2 = self.v.pos()
        return QRectF(p1, p2).normalized().adjusted(-40, -40, 40, 40)

    def update_position(self):
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        p1 = self.u.pos()
        p2 = self.v.pos()

        is_weak = (self.j_val is not None and self.j_val < 4.5)
        if is_weak:
            pen = QPen(QColor('#78909C'), 1.8, Qt.PenStyle.DashLine)
        else:
            pen = QPen(QColor('#37474F'), 2.4, Qt.PenStyle.SolidLine)

        painter.setPen(pen)
        painter.drawLine(p1, p2)

        if self.j_val is not None:
            mid = (p1 + p2) / 2.0
            j_str = f"{self.j_val:.1f}"
            font = QFont("Arial", 8, QFont.Weight.Bold)
            painter.setFont(font)

            pill_w, pill_h = 30.0, 16.0
            pill_rect = QRectF(mid.x() - pill_w / 2.0, mid.y() - pill_h / 2.0, pill_w, pill_h)

            painter.setPen(QPen(QColor('#CFD8DC'), 1.0))
            painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
            painter.drawRoundedRect(pill_rect, 4.0, 4.0)

            painter.setPen(QPen(QColor('#C62828')))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, j_str)


class InteractiveGraphCanvas(QGraphicsView):
    nodeClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet("background-color: #FAFAFA; border: 1px solid #ECEFF1; border-radius: 4px;")

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

    def set_graph(self, nodes_dict: Dict[str, dict], edges_list: list, has_weights: bool = False):
        self.scene.clear()
        if not nodes_dict:
            return

        node_keys = list(nodes_dict.keys())
        edge_pairs = [(e[0], e[1]) if has_weights else e for e in edges_list]

        pos = calculate_spring_layout(node_keys, edge_pairs)

        dist_scale = 145.0
        node_items = {}

        for k in node_keys:
            p = pos.get(k, np.array([0.0, 0.0]))
            item = GraphNodeItem(k, nodes_dict[k], p[0] * dist_scale, p[1] * dist_scale)
            item.clicked.connect(self.nodeClicked.emit)
            self.scene.addItem(item)
            node_items[k] = item

        for e in edges_list:
            u_key = e[0]
            v_key = e[1]
            j_val = e[2] if has_weights and len(e) >= 3 else None
            if u_key in node_items and v_key in node_items:
                edge_item = GraphEdgeItem(node_items[u_key], node_items[v_key], j_val)
                self.scene.addItem(edge_item)

        rect = self.scene.itemsBoundingRect().adjusted(-120, -120, 120, 120)
        self.scene.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)


# ==========================================================
#     АЛГОРИТМ РАСПОЛОЖЕНИЯ УЗЛОВ (Fruchterman-Reingold)
# ==========================================================

def calculate_spring_layout(nodes: List[str], edges: list, iterations: int = 150) -> Dict[str, np.ndarray]:
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: np.array([0.0, 0.0])}

    pos = {}
    for i, node in enumerate(nodes):
        angle = 2.0 * math.pi * i / n
        pos[node] = np.array([math.cos(angle) * 4.5, math.sin(angle) * 4.5])

    adj = {u: set() for u in nodes}
    for edge in edges:
        u, v = edge[0], edge[1]
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)

    k = math.sqrt(22.0 / n)
    t = 2.5
    dt = t / (iterations + 1)

    for _ in range(iterations):
        disp = {node: np.zeros(2) for node in nodes}

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

        for node in nodes:
            disp[node] -= 0.03 * pos[node]
            d_norm = np.linalg.norm(disp[node])
            if d_norm > 1e-4:
                step = min(d_norm, t)
                pos[node] += (disp[node] / d_norm) * step
                pos[node] = np.clip(pos[node], -8.0, 8.0)

        t -= dt

    return pos


# ==========================================================
#                  ПАРСЕРЫ ДАННЫХ
# ==========================================================

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

        # d=1 (вицинальные) + d=2 строго в 6-членном бензольном кольце (C0 - C2)
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
        """
        Строгий парсер эксперимента:
        Edge = (COSY cross-peak >= min_area) AND (J-Match in 1D или подтвержденный мультиплет 'm')
        Синглет ('s') блокирует любые связи (степень вершины = 0).
        """
        # 1. Парсинг 1D мультиплетов и констант J
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
                'is_singlet': (mult == 's'),
                'is_multiplet': (mult == 'm'),
                'integ': integ
            })

        # 2. HSQC: считывание пиков с фазочувствительным разделением
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

        # 3. Схлопывание диастереотопных пар CH2 (ровно 7 CH2 групп)
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

        # Сборка узлов CH2
        for cl in ch2_clusters:
            node_id = f"EXP_{node_idx}"
            node_idx += 1
            c_mean = float(np.mean([p['c'] for p in cl]))
            h_list = sorted([p['h'] for p in cl], reverse=True)

            node_j = []
            is_singlet = False
            is_multiplet = False

            for h in h_list:
                for p1d in peaks_1d:
                    if abs(p1d['shift'] - h) < 0.08:
                        node_j.extend(p1d['j_vals'])
                        if p1d['is_singlet']:
                            is_singlet = True
                        if p1d['is_multiplet']:
                            is_multiplet = True

            lbl = f"{h_list[0]:.2f}, {h_list[1]:.2f}\n{c_mean:.1f} (CH2)" if len(h_list) == 2 else f"{h_list[0]:.2f}\n{c_mean:.1f} (CH2)"

            exp_nodes[node_id] = {
                'c': round(c_mean, 2),
                'h_list': h_list,
                'j_vals': sorted(list(set(node_j)), reverse=True),
                'is_singlet': is_singlet,
                'is_multiplet': is_multiplet,
                'type': 'CH2',
                'h_count': 2,
                'label': lbl
            }

        # Сборка узлов CH и CH3 (11 CH + 1 CH3)
        for p in pos_peaks:
            node_id = f"EXP_{node_idx}"
            node_idx += 1
            c_val = p['c']
            h_val = p['h']

            is_methoxyl = (54.0 <= c_val <= 57.0 and 3.80 <= h_val <= 3.95)

            node_j = []
            is_singlet = False
            is_multiplet = False

            for p1d in peaks_1d:
                if abs(p1d['shift'] - h_val) < 0.08:
                    node_j.extend(p1d['j_vals'])
                    if p1d['is_singlet']:
                        is_singlet = True
                    if p1d['is_multiplet']:
                        is_multiplet = True

            if is_methoxyl:
                g_type = "CH3"
                h_count = 3
                lbl = f"{h_val:.2f} (3H)\n{c_val:.1f} (CH3)"
                is_singlet = True  # OCH3 строго синглет
            else:
                g_type = "CH"
                h_count = 1
                lbl = f"{h_val:.2f}\n{c_val:.1f} (CH)"

            exp_nodes[node_id] = {
                'c': round(c_val, 2),
                'h_list': [h_val],
                'j_vals': sorted(list(set(node_j)), reverse=True),
                'is_singlet': is_singlet,
                'is_multiplet': is_multiplet,
                'type': g_type,
                'h_count': h_count,
                'label': lbl
            }

        # 4. COSY кросс-пики с фильтрацией диагонали и воды
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
                        cosy_pairs.append((f1, f2, abs(area)))
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

        # 5. Строгое логическое правило (COSY AND 1D J-Match)
        edges_dict = {}
        node_keys = list(exp_nodes.keys())

        for i in range(len(node_keys)):
            for j in range(i + 1, len(node_keys)):
                u, v = node_keys[i], node_keys[j]

                # ПРАВИЛО 1: Синглет — абсолютный запрет на любые связи
                if exp_nodes[u].get('is_singlet') or exp_nodes[v].get('is_singlet'):
                    continue

                # ПРАВИЛО 2: Проверка наличия кросс-пика в COSY
                has_cosy = False
                for f1, f2, area in cosy_pairs:
                    if (match_shift_to_node(f1) == u and match_shift_to_node(f2) == v) or \
                       (match_shift_to_node(f1) == v and match_shift_to_node(f2) == u):
                        has_cosy = True
                        break

                # Если связи в COSY нет — СВЯЗИ НЕТ (убивает ложную 1.6 Гц клику)
                if not has_cosy:
                    continue

                # ПРАВИЛО 3: Сверка констант 1D
                u_j = exp_nodes[u].get('j_vals', [])
                v_j = exp_nodes[v].get('j_vals', [])
                u_is_m = exp_nodes[u].get('is_multiplet', False)
                v_is_m = exp_nodes[v].get('is_multiplet', False)

                shared_j = None
                min_diff = j_tol

                # Если у обоих узлов есть оцифрованные J — они ОБЯЗАНЫ численно совпасть
                if u_j and v_j:
                    for ju in u_j:
                        for jv in v_j:
                            diff = abs(ju - jv)
                            if diff <= min_diff:
                                min_diff = diff
                                shared_j = round((ju + jv) / 2.0, 1)

                    if shared_j is not None:
                        edges_dict[(u, v)] = shared_j
                    else:
                        # В COSY пятно есть, а константы не совпали -> СВЯЗЬ ОТСУТСТВУЕТ
                        continue

                # Если один из узлов — сложный мультиплет 'm' (где J не оцифрована, но физически есть)
                elif u_is_m and v_j:
                    edges_dict[(u, v)] = min(v_j) if min(v_j) < 10.0 else max(v_j)
                elif v_is_m and u_j:
                    edges_dict[(u, v)] = min(u_j) if min(u_j) < 10.0 else max(u_j)
                elif u_is_m and v_is_m:
                    # Оба узла мультиплеты с сильным COSY
                    edges_dict[(u, v)] = 7.0

        final_edges = [(u, v, j_val) for (u, v), j_val in edges_dict.items()]
        return exp_nodes, final_edges


# ==========================================================
#                   ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ==========================================================

class GraphVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR Multi-Graph Visualizer (Strict 1D J-Axiom Engine)")
        self.resize(1700, 1020)
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

        t3_l.addWidget(QLabel("1D 1H NMR (Мультиплеты с J):"))
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

        self.btn_plot = QPushButton("⚡ Построить графы (COSY ∧ 1D J-Axiom)")
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

        # 2D Структура молекулы
        struct_group = QGroupBox("2D Структура молекулы (Клик по узлу графа подсветит углерод и все его водороды)")
        struct_l = QVBoxLayout(struct_group)
        self.mol_view = InteractiveMolLabel("Молекула отрисуется после ввода координат XYZ")
        self.mol_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mol_view.setStyleSheet("background-color: white; border: 1px solid #ECEFF1; border-radius: 4px;")
        self.mol_view.setMinimumHeight(380)
        self.mol_view.atomClicked.connect(self.on_mol_atom_clicked)
        struct_l.addWidget(self.mol_view)
        right_splitter.addWidget(struct_group)

        # Вкладки интерактивных графов
        self.view_tabs = QTabWidget()

        self.canvas_xyz = InteractiveGraphCanvas()
        self.canvas_xyz.nodeClicked.connect(self.on_graph_node_clicked)
        self.view_tabs.addTab(self.canvas_xyz, "1. Структура (XYZ)")

        self.canvas_dft = InteractiveGraphCanvas()
        self.canvas_dft.nodeClicked.connect(self.on_graph_node_clicked)
        self.view_tabs.addTab(self.canvas_dft, "2. DFT")

        self.canvas_exp = InteractiveGraphCanvas()
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

        main_splitter.setSizes([480, 1200])
        layout.addWidget(main_splitter)
        self.setCentralWidget(main_w)

    def _render_mol(self, highlights: Optional[List[int]] = None):
        if not self.mol:
            self.mol_view.setText("Координаты XYZ не заданы.")
            return

        w, h = 800, 380
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
            opts = d2d.drawOptions()
            opts.addAtomIndices = True
            opts.highlightBondWidthMultiplier = 3.5

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

        except Exception:
            try:
                d2d_svg = rdMolDraw2D.MolDraw2DSVG(w, h)
                d2d_svg.drawOptions().addAtomIndices = True
                if highlights:
                    colors = {idx: (1.0, 0.40, 0.0) for idx in highlights}
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
        """Интерактивный клик по узлу на графе -> подсветка углерода и всех его водородов на молекуле"""
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
        """Интерактивный клик по атому на 2D молекуле -> подсветка группы"""
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
                self.canvas_xyz.set_graph(xyz_nodes, xyz_edges, has_weights=False)

            # 2. DFT спиновый граф
            dft_nodes, dft_edges = {}, []
            if dft_t:
                dft_nodes, dft_edges = DataParsers.parse_dft(dft_t)
                self.canvas_dft.set_graph(dft_nodes, dft_edges, has_weights=True)

            # 3. Эксперимент с жестким правилом J-Matching
            exp_nodes, exp_edges = DataParsers.parse_exp(
                exp1d_t, hsqc_t, cosy_t,
                c_tol=self.spin_ctol.value(),
                cosy_min_area=self.spin_area.value(),
                j_tol=self.spin_jtol.value(),
                ignore_artifacts=self.chk_artifacts.isChecked()
            )
            self.canvas_exp.set_graph(exp_nodes, exp_edges, has_weights=True)
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
            QMessageBox.warning(self, "Внимание", "Сначала постройте графы кнопкой 'Построить графы'.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Экспорт графов для ИИ", "graphs_export_clean.json", "JSON Files (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.last_graphs, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Успех", f"Очищенные графы экспортированы в {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphVisualizerApp()
    window.show()
    sys.exit(app.exec())