# core.py
import re
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from PyQt6.QtWidgets import (QLabel, QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGraphicsObject, QComboBox, QListView, QAbstractItemView)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QEvent
from PyQt6.QtGui import QPainter, QPen, QBrush, QFont, QColor, QRadialGradient

from rdkit import Chem
from rdkit.Chem import rdDepictor

COV_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03, 'Ca': 1.76,
    'Br': 1.20, 'I': 1.39
}

NODE_COLORS = {
    'CH': '#FFA726',
    'CH1': '#FFA726',
    'CH2': '#42A5F5',
    'CH3': '#66BB6A',
    'CH4': '#AB47BC'
}


class ScrollableComboBox(QComboBox):
    """Выпадающий список со свободным скроллом и отображением полного набора ядер."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        view = QListView(self)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self.setView(view)
        self.setMaxVisibleItems(25)
        view.viewport().installEventFilter(self)

        self.setStyleSheet("""
            QComboBox {
                combobox-popup: 0;
                padding: 3px 6px;
                border: 1px solid #B0BEC5;
                border-radius: 3px;
                background-color: white;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #78909C;
                background-color: white;
                selection-background-color: #BBDEFB;
                selection-color: black;
                min-width: 320px;
            }
            QScrollBar:vertical {
                border: 1px solid #CFD8DC;
                background: #ECEFF1;
                width: 18px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #78909C;
                min-height: 25px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #455A64;
            }
        """)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            view = self.view()
            if view:
                sb = view.verticalScrollBar()
                if sb:
                    delta = event.angleDelta().y()
                    step = -2 if delta > 0 else 2
                    sb.setValue(sb.value() + step)
                    return True
        return super().eventFilter(obj, event)


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
        return QRectF(-self.radius - pad, -self.radius - pad, (self.radius + pad) * 2, (self.radius + pad) * 2)

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
        pen_width = 3.5 if self.is_hovered else 2.0

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
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            pill_w, pill_h = 30.0, 16.0
            pill_rect = QRectF(mid.x() - pill_w / 2.0, mid.y() - pill_h / 2.0, pill_w, pill_h)

            painter.setPen(QPen(QColor('#CFD8DC'), 1.0))
            painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
            painter.drawRoundedRect(pill_rect, 4.0, 4.0)

            painter.setPen(QPen(QColor('#C62828')))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, j_str)


def calculate_spring_layout(nodes: List[str], edges: list, iterations: int = 150) -> Dict[str, np.ndarray]:
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: np.array([0.0, 0.0])}
    pos = {node: np.array([math.cos(2.0 * math.pi * i / n) * 4.5, math.sin(2.0 * math.pi * i / n) * 4.5]) for i, node in enumerate(nodes)}
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
                    delta, dist = np.array([0.03, 0.03]), 0.042
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
                pos[node] += (disp[node] / d_norm) * min(d_norm, t)
                pos[node] = np.clip(pos[node], -8.0, 8.0)
        t -= dt
    return pos


class InteractiveGraphCanvas(QGraphicsView):
    nodeClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setStyleSheet("background-color: #FAFAFA; border: 1px solid #ECEFF1; border-radius: 4px;")
        self.node_items = {}

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

    def set_graph(self, nodes_dict: Dict[str, dict], edges_list: list, has_weights: bool = False):
        self.scene.clear()
        self.node_items = {}
        if not nodes_dict:
            return
        node_keys = list(nodes_dict.keys())
        edge_pairs = [(e[0], e[1]) if has_weights else e for e in edges_list]
        pos = calculate_spring_layout(node_keys, edge_pairs)
        dist_scale = 145.0
        for k in node_keys:
            p = pos.get(k, np.array([0.0, 0.0]))
            item = GraphNodeItem(k, nodes_dict[k], p[0] * dist_scale, p[1] * dist_scale)
            item.clicked.connect(self.nodeClicked.emit)
            self.scene.addItem(item)
            self.node_items[k] = item
        for e in edges_list:
            u_key, v_key = e[0], e[1]
            j_val = e[2] if has_weights and len(e) >= 3 else None
            if u_key in self.node_items and v_key in self.node_items:
                self.scene.addItem(GraphEdgeItem(self.node_items[u_key], self.node_items[v_key], j_val))
        rect = self.scene.itemsBoundingRect().adjusted(-120, -120, 120, 120)
        self.scene.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_to_node(self, node_id: str, scale_factor: float = 1.35):
        """Центрирует и масштабирует холст на выбранном узле графа (вызывается из таблицы)."""
        if hasattr(self, 'node_items') and node_id in self.node_items:
            item = self.node_items[node_id]
            self.resetTransform()
            self.scale(scale_factor, scale_factor)
            self.centerOn(item)
            for nid, nitem in self.node_items.items():
                nitem.is_hovered = (nid == node_id)
                nitem.update()


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
                        symbols.append(m.group(1).capitalize())
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
        if not symbols:
            return None, {}, []

        mol = Chem.RWMol()
        for sym in symbols:
            a = Chem.Atom(sym)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            mol.AddAtom(a)

        coords_arr = np.array(coords)
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
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
                    t_name = "CH" if len(h_nbrs) == 1 else f"CH{len(h_nbrs)}"
                    rd_nodes[f"C_{atom.GetIdx()}"] = {
                        "label": f"C {atom.GetIdx()}\n({t_name})",
                        "type": t_name,
                        "h_count": len(h_nbrs)
                    }

        topo_dist = Chem.GetDistanceMatrix(mol)
        c_idx = [int(k.split('_')[1]) for k in rd_nodes.keys()]
        edges = []
        for i in range(len(c_idx)):
            for j in range(i + 1, len(c_idx)):
                c1, c2 = c_idx[i], c_idx[j]
                if topo_dist[c1, c2] == 1:
                    edges.append((f"C_{c1}", f"C_{c2}"))
                elif topo_dist[c1, c2] == 2:
                    a1, a2 = mol.GetAtomWithIdx(c1), mol.GetAtomWithIdx(c2)
                    if a1.GetIsAromatic() and a2.GetIsAromatic():
                        edges.append((f"C_{c1}", f"C_{c2}"))
        return mol, rd_nodes, edges

    @staticmethod
    def parse_dft_shifts(dft_text: str, tms_c: float, tms_h: float) -> dict:
        shifts = {}
        for line in dft_text.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isalpha():
                idx = int(parts[0])
                sym = parts[1].upper()
                try:
                    iso = float(parts[2])
                    if sym == 'C':
                        shifts[idx] = tms_c - iso
                    elif sym == 'H':
                        shifts[idx] = tms_h - iso
                except ValueError:
                    pass
        return shifts

    @staticmethod
    def parse_dft(dft_text: str) -> Tuple[Dict[str, dict], List[Tuple[str, str, float]]]:
        atoms, j_matrix, col_headers = {}, {}, []
        for line in dft_text.splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) >= 4 and tokens[0].isdigit() and tokens[1].isalpha() and tokens[2].isdigit():
                col_headers = [int(tokens[i]) for i in range(0, len(tokens), 2)]
                continue
            if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isalpha() and '.' in tokens[2]:
                r_idx = int(tokens[0])
                atoms[r_idx] = tokens[1].capitalize()
                for i, v_str in enumerate(tokens[2:]):
                    if i < len(col_headers):
                        try:
                            j_matrix[(r_idx, col_headers[i])] = j_matrix[(col_headers[i], r_idx)] = float(v_str)
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
        for i, h in c_nodes.items():
            if h:
                t_name = "CH" if len(h) == 1 else f"CH{len(h)}"
                dft_nodes[f"DFT_{i}"] = {
                    "label": f"C {i}\nH:[{','.join(map(str, sorted(h)))}]",
                    "type": t_name,
                    "h_count": len(h),
                    "H": h
                }

        max_j = {}
        for (i, j), val in j_matrix.items():
            if atoms.get(i) == 'H' and atoms.get(j) == 'H':
                ci = next((c for c, h in c_nodes.items() if i in h), None)
                cj = next((c for c, h in c_nodes.items() if j in h), None)
                if ci is not None and cj is not None and ci != cj:
                    pair = tuple(sorted((f"DFT_{ci}", f"DFT_{cj}")))
                    max_j[pair] = max(max_j.get(pair, 0.0), abs(val))

        return dft_nodes, [(u, v, jv) for (u, v), jv in max_j.items() if jv >= 1.4]

    @staticmethod
    def parse_exp(exp1d_text: str, hsqc_text: str, cosy_text: str, c_tol: float, cosy_min_area: float, j_tol: float, ignore_artifacts: bool):
        peaks_1d = []
        for m in re.finditer(r'(?:δ\s*)?(?P<shift>\d+\.\d+)(?:\s*[–-]\s*(?P<shift2>\d+\.\d+))?\s*\(\s*(?P<mult>[a-zA-Z]+)(?:,\s*J\s*=\s*(?P<couplings>[\d\.,\s]+)\s*Hz)?(?:,\s*(?P<integ>\d+)H)?\s*\)', exp1d_text):
            s1 = float(m.group('shift'))
            s2 = float(m.group('shift2')) if m.group('shift2') else s1
            mult = m.group('mult').lower()
            integ = int(m.group('integ')) if m.group('integ') else (2 if '2h' in m.group(0).lower() else 1)
            j_vals = [float(x) for x in re.findall(r'\d+\.?\d*', m.group('couplings'))] if m.group('couplings') else []
            peaks_1d.append({
                'shift': (s1 + s2) / 2.0,
                'mult': mult,
                'integ': integ,
                'j_vals': j_vals,
                'is_s': mult == 's',
                'is_m': mult == 'm'
            })

        hsqc_raw = []
        for line in hsqc_text.splitlines():
            line_str = line.strip().lower()
            if not line_str or "ppm" in line_str or "flags" in line_str or "dmso" in line_str:
                continue
            toks = line_str.split()
            if len(toks) >= 4:
                try:
                    off = 1 if toks[0].isdigit() else 0
                    c_v, h_v = float(toks[off]), float(toks[off + 1])
                    if 38.5 <= c_v <= 41.5 and 2.40 <= h_v <= 2.60:
                        continue
                    area = float(toks[off + 5]) if len(toks) >= off + 6 else 0.0
                    is_neg = toks[off + 2].startswith('-') or (area < -0.05)
                    hsqc_raw.append({'c': c_v, 'h': h_v, 'is_neg': is_neg})
                except (ValueError, IndexError):
                    pass

        neg_peaks = [p for p in hsqc_raw if p['is_neg']]
        pos_peaks = [p for p in hsqc_raw if not p['is_neg']]

        exp_nodes = {}
        idx = 0

        paired_neg = set()
        ch2_groups = []

        for i in range(len(neg_peaks)):
            if i in paired_neg:
                continue
            best_j = None
            min_c_diff = c_tol
            for j in range(i + 1, len(neg_peaks)):
                if j in paired_neg:
                    continue
                c_diff = abs(neg_peaks[i]['c'] - neg_peaks[j]['c'])
                h_diff = abs(neg_peaks[i]['h'] - neg_peaks[j]['h'])
                if c_diff <= min_c_diff and h_diff >= 0.04:
                    min_c_diff = c_diff
                    best_j = j

            if best_j is not None:
                ch2_groups.append([neg_peaks[i], neg_peaks[best_j]])
                paired_neg.update([i, best_j])
            else:
                ch2_groups.append([neg_peaks[i]])
                paired_neg.add(i)

        for cl in ch2_groups:
            c_mean = float(np.mean([p['c'] for p in cl]))
            h_raw = sorted([p['h'] for p in cl], reverse=True)

            n_j, is_s, is_m = [], False, False
            mult_list = []
            total_integ = 0
            for h in h_raw:
                matched_p = min(peaks_1d, key=lambda p: abs(p['shift'] - h), default=None)
                if matched_p and abs(matched_p['shift'] - h) < 0.08:
                    n_j.extend(matched_p['j_vals'])
                    if matched_p['is_s']: is_s = True
                    if matched_p['is_m']: is_m = True
                    if matched_p.get('mult'):
                        mult_list.append(matched_p['mult'])
                    total_integ += matched_p['integ']
                else:
                    total_integ += 1

            if len(h_raw) == 1 and total_integ >= 2:
                h_list = [h_raw[0], h_raw[0]]
                is_complete = True
            elif len(h_raw) == 2:
                h_list = h_raw
                is_complete = True
            else:
                h_list = h_raw
                is_complete = False

            mult_str = ", ".join(mult_list) if mult_list else ('s' if is_s else ('m' if is_m else 'm'))

            exp_nodes[f"EXP_{idx}"] = {
                'c': round(c_mean, 2),
                'h_list': h_list,
                'j_vals': sorted(list(set(n_j)), reverse=True),
                'is_singlet': is_s,
                'is_multiplet': is_m,
                'type': 'CH2',
                'mult': mult_str,
                'is_complete': is_complete,
                'integ': len(h_list)
            }
            idx += 1

        for p in pos_peaks:
            matched_p = min(peaks_1d, key=lambda x: abs(x['shift'] - p['h']), default=None)
            integ = matched_p['integ'] if (matched_p and abs(matched_p['shift'] - p['h']) < 0.08) else 1
            is_s = matched_p['is_s'] if matched_p else False
            is_m = matched_p['is_m'] if matched_p else False
            n_j = matched_p['j_vals'] if matched_p else []

            is_ch3 = (54.0 <= p['c'] <= 57.0 and 3.75 <= p['h'] <= 4.0) or (integ >= 3 and is_s and not is_m)
            g_type = "CH3" if is_ch3 else "CH"
            mult_str = 's' if is_ch3 else (matched_p['mult'] if (matched_p and matched_p.get('mult')) else ('s' if is_s else ('m' if is_m else 'm')))

            exp_nodes[f"EXP_{idx}"] = {
                'c': round(p['c'], 2),
                'h_list': [p['h']],
                'j_vals': sorted(list(set(n_j)), reverse=True),
                'is_singlet': is_s or is_ch3,
                'is_multiplet': is_m,
                'type': g_type,
                'mult': mult_str,
                'is_complete': True,
                'integ': 3 if is_ch3 else 1
            }
            idx += 1

        for k, v in exp_nodes.items():
            h_str = ", ".join([f"{h:.2f}" for h in v['h_list']])
            exp_nodes[k]['label'] = f"{h_str}\n{v['c']:.1f} ({v['type']})"

        cosy = []
        for line in cosy_text.splitlines():
            toks = line.strip().split()
            if not toks or ("artifact" in line.lower() and ignore_artifacts):
                continue
            try:
                off = 1 if toks[0].isdigit() else 0
                f1, f2 = float(toks[off]), float(toks[off + 1])
                area = float(toks[off + 5]) if len(toks) >= off + 6 else 1.0
                if abs(f1 - f2) >= 0.04 and abs(area) >= cosy_min_area:
                    cosy.append((f1, f2, abs(area)))
            except (ValueError, IndexError):
                pass

        def match_h(s):
            return min(exp_nodes.keys(), key=lambda k: min([abs(h - s) for h in exp_nodes[k]['h_list']]), default=None)

        edges_dict = {}
        node_keys = list(exp_nodes.keys())
        for i, u in enumerate(node_keys):
            for v in node_keys[i + 1:]:
                if exp_nodes[u]['is_singlet'] or exp_nodes[v]['is_singlet']:
                    continue
                c_area = max([a for f1, f2, a in cosy if (match_h(f1) == u and match_h(f2) == v) or (match_h(f1) == v and match_h(f2) == u)], default=0.0)
                if c_area == 0.0:
                    continue

                uj, vj = exp_nodes[u]['j_vals'], exp_nodes[v]['j_vals']
                if uj and vj:
                    shared_j = None
                    min_d = j_tol
                    for ju in uj:
                        for jv in vj:
                            if abs(ju - jv) <= min_d:
                                min_d = abs(ju - jv)
                                shared_j = round((ju + jv) / 2.0, 1)
                    if shared_j is not None:
                        edges_dict[(u, v)] = shared_j
                elif (exp_nodes[u]['is_multiplet'] or exp_nodes[v]['is_multiplet']) and c_area >= 1.0:
                    edges_dict[(u, v)] = min(uj + vj + [7.0])

        return exp_nodes, [(u, v, j) for (u, v), j in edges_dict.items()]