import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import csv
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

from parsers import (
    parse_shieldings, parse_couplings, parse_xyz_coordinates,
    compute_hsqc_correlations, compute_cosy_correlations,
    COVALENT_RADII, CPK_COLORS
)
from physics import (
    simulate_grouped_h_molecule, simulate_grouped_c13_spectrum,
    simulate_2d_hsqc_map, simulate_2d_cosy_map, render_2d_molecule_on_ax
)
from analysis import (
    analyze_grouped_h_signals, analyze_grouped_c_signals, build_acs_publication_block
)


class FullNMRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()

        self.title("1H, 13C, COSY & HSQC NMR Spectrometer Suite")
        self.geometry("1340x900")
        self.minsize(1100, 740)

        # Calculation records
        self.raw_shieldings = {}
        self.raw_h_shifts = {}
        self.raw_c_shifts = {}
        self.coupl_dict = {}
        self.mol_atoms = {}
        self.h_groups = []
        self.c_groups = []

        self.h_signals = []
        self.c_signals = []
        self.hsqc_correlations = []
        self.cosy_correlations = []

        self.freq_h = 600.0
        self.freq_c = 150.9
        self.lw_h = 0.8
        self.lw_c = 1.5

        # Homonuclear Decoupling / Spin Suppression
        self.decoupled_h_spin = None

        # 3D interactive state
        self.selected_hydrogens = set()
        self.selected_carbons = set()
        self.scatter_indices = []
        self.mol_center = np.zeros(3)
        self.mol_zoom_radius = 5.0
        self.mol_init_radius = 5.0

        # 1H plot state
        self.h_ppm = None
        self.h_spec = None
        self.h_annotations = []
        self.h_selected_groups = set()
        self.h_dragged_ann = None
        self.h_drag_offset = (0.0, 0.0)
        self.h_zoom_start_x = None
        self.h_zoom_rect = None
        self.h_zoom_history = []
        self.h_init_xlim = None
        self.h_init_ylim = None

        # Interactive 2D Mol Overlays (position & size: [x0, y0, width, height])
        self.h_mol_pos = [0.73, 0.54, 0.25, 0.42]
        self.c_mol_pos = [0.73, 0.54, 0.25, 0.42]
        self.show_h_mol = tk.BooleanVar(value=True)
        self.show_c_mol = tk.BooleanVar(value=True)
        self.ax_h_mol = None
        self.ax_c_mol = None
        self.h_drag_mol = False
        self.c_drag_mol = False
        self.h_drag_mol_last = (0, 0)
        self.c_drag_mol_last = (0, 0)

        # 13C plot state
        self.c_ppm = None
        self.c_spec = None
        self.c_annotations = []
        self.c_selected_groups = set()
        self.c_dragged_ann = None
        self.c_drag_offset = (0.0, 0.0)
        self.c_zoom_start_x = None
        self.c_zoom_rect = None
        self.c_zoom_history = []
        self.c_init_xlim = None
        self.c_init_ylim = None

        # 2D COSY state
        self.cosy_zoom_start = None
        self.cosy_zoom_rect = None
        self.cosy_zoom_history = []
        self.cosy_init_lim = (6.0, -0.2)

        # 2D HSQC state
        self.hsqc_zoom_start = None
        self.hsqc_zoom_rect = None
        self.hsqc_zoom_history = []
        self.hsqc_init_xlim = (6.0, -0.2)
        self.hsqc_init_ylim = (170.0, -5.0)

        # Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_input = ttk.Frame(self.notebook)
        self.tab_mol = ttk.Frame(self.notebook)
        self.tab_h_plot = ttk.Frame(self.notebook)
        self.tab_c_plot = ttk.Frame(self.notebook)
        self.tab_cosy = ttk.Frame(self.notebook)
        self.tab_hsqc = ttk.Frame(self.notebook)
        self.tab_table = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_input, text=" 1. Input Data ")
        self.notebook.add(self.tab_mol, text=" 2. 3D Structure ")
        self.notebook.add(self.tab_h_plot, text=" 3. 1H NMR & Decoupling ")
        self.notebook.add(self.tab_c_plot, text=" 4. 13C NMR ")
        self.notebook.add(self.tab_cosy, text=" 5. 2D COSY ")
        self.notebook.add(self.tab_hsqc, text=" 6. 2D HSQC (C-H) ")
        self.notebook.add(self.tab_table, text=" 7. Report & Export ")

        self._build_tab_input()
        self._build_tab_mol()
        self._build_tab_h_plot()
        self._build_tab_c_plot()
        self._build_tab_cosy()
        self._build_tab_hsqc()
        self._build_tab_table()

        self.bind_all("<Return>", self._on_enter_pressed)
        self.bind_all("<KP_Enter>", self._on_enter_pressed)
        self.bind_all("<Delete>", self._on_delete_pressed)
        self.bind_all("<BackSpace>", self._on_delete_pressed)

    def _setup_clipboard(self, widget):
        def do_copy(e=None):
            if isinstance(widget, ttk.Treeview):
                sel = widget.selection()
                if sel:
                    lines = ["\t".join(str(v) for v in widget.item(i, "values")) for i in sel]
                    self.clipboard_clear();
                    self.clipboard_append("\n".join(lines));
                    self.update()
                return "break"
            else:
                widget.event_generate("<<Copy>>");
                return "break"

        def do_paste(e=None):
            if isinstance(widget, (tk.Text, tk.Entry, ttk.Entry)):
                widget.event_generate("<<Paste>>");
                return "break"

        def do_select_all(e=None):
            if isinstance(widget, tk.Text):
                widget.tag_add("sel", "1.0", "end")
            elif isinstance(widget, (tk.Entry, ttk.Entry)):
                widget.select_range(0, tk.END); widget.icursor(tk.END)
            elif isinstance(widget, ttk.Treeview):
                widget.selection_set(widget.get_children())
            return "break"

        widget.bind("<Control-KeyPress-Cyrillic_es>", do_copy);
        widget.bind("<Control-KeyPress-Cyrillic_ES>", do_copy)
        widget.bind("<Control-KeyPress-Cyrillic_em>", do_paste);
        widget.bind("<Control-KeyPress-Cyrillic_EM>", do_paste)
        widget.bind("<Control-KeyPress-Cyrillic_ef>", do_select_all);
        widget.bind("<Control-KeyPress-Cyrillic_EF>", do_select_all)
        if isinstance(widget, ttk.Treeview):
            widget.bind("<Control-c>", do_copy);
            widget.bind("<Control-C>", do_copy)

        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy (Ctrl+C)", command=do_copy)
        if not isinstance(widget, ttk.Treeview): menu.add_command(label="Paste (Ctrl+V)", command=do_paste)
        menu.add_separator()
        menu.add_command(label="Select All (Ctrl+A)", command=do_select_all)
        widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def _safe_paste_to(self, widget):
        try:
            widget.event_generate("<<Paste>>")
        except Exception:
            try:
                widget.insert(tk.INSERT, self.clipboard_get())
            except tk.TclError:
                pass

    # ----------------------------------------------------
    # TAB 1: INPUT DATA
    # ----------------------------------------------------
    def _build_tab_input(self):
        paned = ttk.PanedWindow(self.tab_input, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        f_shield = ttk.LabelFrame(paned, text=" Isotropic Shieldings (1H & 13C) ")
        bar_s = ttk.Frame(f_shield)
        bar_s.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bar_s, text="📋 Paste", command=lambda: self._safe_paste_to(self.txt_shield)).pack(side=tk.LEFT,
                                                                                                     padx=2)
        ttk.Button(bar_s, text="Clear", command=lambda: self.txt_shield.delete("1.0", tk.END)).pack(side=tk.LEFT,
                                                                                                    padx=2)
        self.txt_shield = scrolledtext.ScrolledText(f_shield, wrap=tk.NONE, font=("Consolas", 9))
        self.txt_shield.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._setup_clipboard(self.txt_shield);
        paned.add(f_shield, weight=1)

        f_j = ttk.LabelFrame(paned, text=" Block Matrix of J-Couplings (Hz) ")
        bar_j = ttk.Frame(f_j)
        bar_j.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bar_j, text="📋 Paste", command=lambda: self._safe_paste_to(self.txt_j)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar_j, text="Clear", command=lambda: self.txt_j.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=2)
        self.txt_j = scrolledtext.ScrolledText(f_j, wrap=tk.NONE, font=("Consolas", 9))
        self.txt_j.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._setup_clipboard(self.txt_j);
        paned.add(f_j, weight=1)

        f_xyz = ttk.LabelFrame(paned, text=" 3D Coordinates (XYZ / ORCA Output) ")
        bar_x = ttk.Frame(f_xyz)
        bar_x.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bar_x, text="📋 Paste", command=lambda: self._safe_paste_to(self.txt_xyz)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar_x, text="Clear", command=lambda: self.txt_xyz.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=2)
        self.txt_xyz = scrolledtext.ScrolledText(f_xyz, wrap=tk.NONE, font=("Consolas", 9))
        self.txt_xyz.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._setup_clipboard(self.txt_xyz);
        paned.add(f_xyz, weight=1)

        p_frame = ttk.LabelFrame(self.tab_input, text=" Spectrometer & Standards Reference Settings ")
        p_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(p_frame, text="1H Ref σ (calc):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.ent_sigma_ref_h = ttk.Entry(p_frame, width=9)
        self.ent_sigma_ref_h.grid(row=0, column=1, padx=5, pady=5);
        self.ent_sigma_ref_h.insert(0, "31.60")
        self._setup_clipboard(self.ent_sigma_ref_h)

        ttk.Label(p_frame, text="1H Ref δ (ppm):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.ent_delta_ref_h = ttk.Entry(p_frame, width=9)
        self.ent_delta_ref_h.grid(row=0, column=3, padx=5, pady=5);
        self.ent_delta_ref_h.insert(0, "0.00")
        self._setup_clipboard(self.ent_delta_ref_h)

        ttk.Label(p_frame, text="1H Freq (MHz):").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.ent_freq_h = ttk.Entry(p_frame, width=9)
        self.ent_freq_h.grid(row=0, column=5, padx=5, pady=5);
        self.ent_freq_h.insert(0, "600.0")
        self._setup_clipboard(self.ent_freq_h)
        self.ent_freq_h.bind("<FocusOut>", self._auto_update_c_freq)

        ttk.Label(p_frame, text="1H Linewidth (Hz):").grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        self.ent_lw_h = ttk.Entry(p_frame, width=9)
        self.ent_lw_h.grid(row=0, column=7, padx=5, pady=5);
        self.ent_lw_h.insert(0, "0.8")
        self._setup_clipboard(self.ent_lw_h)

        ttk.Label(p_frame, text="13C Ref σ (calc):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.ent_sigma_ref_c = ttk.Entry(p_frame, width=9)
        self.ent_sigma_ref_c.grid(row=1, column=1, padx=5, pady=5);
        self.ent_sigma_ref_c.insert(0, "181.10")
        self._setup_clipboard(self.ent_sigma_ref_c)

        ttk.Label(p_frame, text="13C Ref δ (ppm):").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.ent_delta_ref_c = ttk.Entry(p_frame, width=9)
        self.ent_delta_ref_c.grid(row=1, column=3, padx=5, pady=5);
        self.ent_delta_ref_c.insert(0, "0.00")
        self._setup_clipboard(self.ent_delta_ref_c)

        ttk.Label(p_frame, text="13C Freq (MHz):").grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
        self.ent_freq_c = ttk.Entry(p_frame, width=9)
        self.ent_freq_c.grid(row=1, column=5, padx=5, pady=5);
        self.ent_freq_c.insert(0, "150.9")
        self._setup_clipboard(self.ent_freq_c)

        ttk.Label(p_frame, text="13C Linewidth (Hz):").grid(row=1, column=6, padx=5, pady=5, sticky=tk.W)
        self.ent_lw_c = ttk.Entry(p_frame, width=9)
        self.ent_lw_c.grid(row=1, column=7, padx=5, pady=5);
        self.ent_lw_c.insert(0, "1.5")
        self._setup_clipboard(self.ent_lw_c)

        btn_load = ttk.Button(p_frame, text="▶ Load & Compute All", command=self.on_load_all_data)
        btn_load.grid(row=0, column=8, rowspan=2, padx=15, pady=5, sticky=tk.NSEW)

    def _auto_update_c_freq(self, event=None):
        try:
            f_h = float(self.ent_freq_h.get())
            self.ent_freq_c.delete(0, tk.END)
            self.ent_freq_c.insert(0, f"{f_h * 0.25144:.1f}")
        except ValueError:
            pass

    # ----------------------------------------------------
    # TAB 2: 3D MOLECULAR STRUCTURE
    # ----------------------------------------------------
    def _build_tab_mol(self):
        paned_mol = ttk.PanedWindow(self.tab_mol, orient=tk.HORIZONTAL)
        paned_mol.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_canvas = ttk.LabelFrame(paned_mol,
                                      text=" 3D Molecular Model (Rotate: LMB Drag | Zoom: Mouse Scroll Wheel) ")
        paned_mol.add(frame_canvas, weight=5)

        top_3d_bar = ttk.Frame(frame_canvas)
        top_3d_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(top_3d_bar, text="⏮ Reset 3D View", command=self.reset_3d_view).pack(side=tk.LEFT, padx=3)
        ttk.Label(top_3d_bar, text="💡 Scroll up to zoom in, scroll down to zoom out",
                  font=("Segoe UI", 8, "italic"), foreground="#555555").pack(side=tk.LEFT, padx=10)

        self.fig_mol = Figure(figsize=(10, 8), dpi=100, facecolor="#ffffff")
        self.fig_mol.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax_mol = self.fig_mol.add_subplot(111, projection='3d')
        self.ax_mol.set_facecolor("#ffffff")
        self.ax_mol.set_axis_off()

        self.canvas_mol = FigureCanvasTkAgg(self.fig_mol, master=frame_canvas)
        self.canvas_mol.draw()
        self.canvas_mol.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas_mol.mpl_connect("pick_event", self._on_3d_atom_picked)
        self.canvas_mol.mpl_connect("scroll_event", self._on_3d_scroll)

        frame_ctrl = ttk.LabelFrame(paned_mol, text=" Grouping Panels ")
        paned_mol.add(frame_ctrl, weight=1)

        self.sub_notebook = ttk.Notebook(frame_ctrl)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        tab_h_sub = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(tab_h_sub, text=" 1H (Protons) ")
        btn_box_h = ttk.Frame(tab_h_sub);
        btn_box_h.pack(fill=tk.X, padx=3, pady=3)
        ttk.Button(btn_box_h, text="⚡ Fold Selected H (Enter)", command=self.merge_selected_hydrogens).pack(fill=tk.X,
                                                                                                            pady=2)
        ttk.Button(btn_box_h, text="↺ Ungroup (Delete)", command=self.ungroup_selected_hydrogens).pack(fill=tk.X,
                                                                                                       pady=2)
        list_frame_h = ttk.Frame(tab_h_sub);
        list_frame_h.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.h_listbox = tk.Listbox(list_frame_h, selectmode=tk.EXTENDED, font=("Consolas", 9), exportselection=False)
        self.h_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        h_scroll = ttk.Scrollbar(list_frame_h, orient=tk.VERTICAL, command=self.h_listbox.yview)
        h_scroll.pack(side=tk.RIGHT, fill=tk.Y);
        self.h_listbox.config(yscrollcommand=h_scroll.set)
        self.h_listbox.bind("<<ListboxSelect>>", self._on_h_listbox_selected)

        tab_c_sub = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(tab_c_sub, text=" 13C (Carbons) ")
        btn_box_c = ttk.Frame(tab_c_sub);
        btn_box_c.pack(fill=tk.X, padx=3, pady=3)
        ttk.Button(btn_box_c, text="⚡ Fold Selected C (Enter)", command=self.merge_selected_carbons).pack(fill=tk.X,
                                                                                                          pady=2)
        ttk.Button(btn_box_c, text="↺ Ungroup (Delete)", command=self.ungroup_selected_carbons).pack(fill=tk.X, pady=2)
        list_frame_c = ttk.Frame(tab_c_sub);
        list_frame_c.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.c_listbox = tk.Listbox(list_frame_c, selectmode=tk.EXTENDED, font=("Consolas", 9), exportselection=False)
        self.c_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        c_scroll = ttk.Scrollbar(list_frame_c, orient=tk.VERTICAL, command=self.c_listbox.yview)
        c_scroll.pack(side=tk.RIGHT, fill=tk.Y);
        self.c_listbox.config(yscrollcommand=c_scroll.set)
        self.c_listbox.bind("<<ListboxSelect>>", self._on_c_listbox_selected)

    def _on_3d_scroll(self, event):
        if event.inaxes != self.ax_mol or not hasattr(self, 'mol_center'): return
        scale = 1.0 / 1.12 if (event.button == 'up' or getattr(event, 'step', 0) > 0) else 1.12
        self.mol_zoom_radius = max(0.5, min(150.0, self.mol_zoom_radius * scale))
        r = self.mol_zoom_radius;
        cx, cy, cz = self.mol_center
        self.ax_mol.set_xlim(cx - r, cx + r);
        self.ax_mol.set_ylim(cy - r, cy + r);
        self.ax_mol.set_zlim(cz - r, cz + r)
        self.canvas_mol.draw_idle()

    def reset_3d_view(self):
        if hasattr(self, 'mol_center'):
            self.mol_zoom_radius = self.mol_init_radius
            r = self.mol_zoom_radius;
            cx, cy, cz = self.mol_center
            self.ax_mol.set_xlim(cx - r, cx + r);
            self.ax_mol.set_ylim(cy - r, cy + r);
            self.ax_mol.set_zlim(cz - r, cz + r)
            self.ax_mol.view_init(elev=20, azim=-60)
            self.canvas_mol.draw_idle()

    # ----------------------------------------------------
    # TAB 3: 1H NMR WITH HOMONUCLEAR DECOUPLING & MOVABLE/RESIZABLE 2D MOL
    # ----------------------------------------------------
    def _build_tab_h_plot(self):
        ctrl_bar = ttk.Frame(self.tab_h_plot)
        ctrl_bar.pack(fill=tk.X, padx=10, pady=(6, 4))

        ttk.Button(ctrl_bar, text="⚡ Merge (Enter)", command=self.merge_selected_signals_plot).pack(side=tk.LEFT,
                                                                                                    padx=3)
        ttk.Button(ctrl_bar, text="↺ Ungroup (Del)", command=self.ungroup_selected_signals_plot).pack(side=tk.LEFT,
                                                                                                      padx=3)
        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # Homonuclear Decoupling Controls
        ttk.Label(ctrl_bar, text="Irradiate:").pack(side=tk.LEFT, padx=(3, 2))
        self.combo_decouple = ttk.Combobox(ctrl_bar, width=7, state="readonly")
        self.combo_decouple.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_bar, text="🔴 Decouple", command=self.apply_h_decoupling).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_bar, text="Clear", command=self.clear_h_decoupling).pack(side=tk.LEFT, padx=2)
        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # 2D Mol Inset Controls
        ttk.Checkbutton(ctrl_bar, text="2D Mol", variable=self.show_h_mol,
                        command=lambda: self._update_h_spectrum(preserve_zoom=True)).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="🔍+ Mol", command=lambda: self.scale_h_mol(1.15)).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="🔍- Mol", command=lambda: self.scale_h_mol(1.0 / 1.15)).pack(side=tk.LEFT, padx=1)
        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        ttk.Button(ctrl_bar, text="⏮ Reset Zoom", command=self.reset_h_zoom).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="📐 Autoscale Y", command=self.autoscale_h_y).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="💾 Save", command=lambda: self.export_image(self.fig_h_spec, "1H_NMR")).pack(
            side=tk.LEFT, padx=3)

        self.fig_h_spec = Figure(figsize=(10, 5), dpi=100, facecolor="#ffffff")
        self.ax_h_spec = self.fig_h_spec.add_subplot(111)
        self._apply_plot_style(self.ax_h_spec)

        self.canvas_h_spec = FigureCanvasTkAgg(self.fig_h_spec, master=self.tab_h_plot)
        self.canvas_h_spec.draw()
        self.canvas_h_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_h_spec.mpl_connect("button_press_event", self._on_h_mouse_press)
        self.canvas_h_spec.mpl_connect("motion_notify_event", self._on_h_mouse_move)
        self.canvas_h_spec.mpl_connect("button_release_event", self._on_h_mouse_release)
        self.canvas_h_spec.mpl_connect("scroll_event", self._on_h_scroll)

    def apply_h_decoupling(self):
        val = self.combo_decouple.get().strip()
        if not val or val == "None":
            messagebox.showinfo("Decoupling", "Select a proton to irradiate/decouple.")
            return
        try:
            self.decoupled_h_spin = int(val.replace("H", ""))
            self._update_h_spectrum(preserve_zoom=True)
        except ValueError:
            pass

    def clear_h_decoupling(self):
        self.decoupled_h_spin = None
        self.combo_decouple.set("None")
        self._update_h_spectrum(preserve_zoom=True)

    def scale_h_mol(self, factor):
        new_w = max(0.12, min(0.70, self.h_mol_pos[2] * factor))
        new_h = max(0.12, min(0.70, self.h_mol_pos[3] * factor))
        cx = self.h_mol_pos[0] + self.h_mol_pos[2] / 2.0
        cy = self.h_mol_pos[1] + self.h_mol_pos[3] / 2.0
        self.h_mol_pos[0] = max(0.0, min(1.0 - new_w, cx - new_w / 2.0))
        self.h_mol_pos[1] = max(0.0, min(1.0 - new_h, cy - new_h / 2.0))
        self.h_mol_pos[2] = new_w
        self.h_mol_pos[3] = new_h
        self._update_h_spectrum(preserve_zoom=True)

    def _on_h_scroll(self, event):
        """Scales 2D molecule window when hovering over it with mouse wheel."""
        if self.ax_h_mol is not None and self.show_h_mol.get():
            try:
                contains, _ = self.ax_h_mol.contains(event)
                if contains:
                    factor = 1.15 if (event.button == 'up' or getattr(event, 'step', 0) > 0) else (1.0 / 1.15)
                    self.scale_h_mol(factor)
                    return
            except Exception:
                pass

    # ----------------------------------------------------
    # TAB 4: 13C NMR WITH MOVABLE/RESIZABLE 2D MOL
    # ----------------------------------------------------
    def _build_tab_c_plot(self):
        ctrl_bar = ttk.Frame(self.tab_c_plot)
        ctrl_bar.pack(fill=tk.X, padx=10, pady=(6, 4))

        ttk.Button(ctrl_bar, text="⚡ Merge (Enter)", command=self.merge_selected_c_signals_plot).pack(side=tk.LEFT,
                                                                                                      padx=3)
        ttk.Button(ctrl_bar, text="↺ Ungroup (Del)", command=self.ungroup_selected_c_signals_plot).pack(side=tk.LEFT,
                                                                                                        padx=3)
        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # 2D Mol Inset Controls
        ttk.Checkbutton(ctrl_bar, text="2D Mol", variable=self.show_c_mol,
                        command=lambda: self._update_c_spectrum(preserve_zoom=True)).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="🔍+ Mol", command=lambda: self.scale_c_mol(1.15)).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="🔍- Mol", command=lambda: self.scale_c_mol(1.0 / 1.15)).pack(side=tk.LEFT, padx=1)
        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        ttk.Button(ctrl_bar, text="⏮ Reset Zoom", command=self.reset_c_zoom).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="📐 Autoscale Y", command=self.autoscale_c_y).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="💾 Save", command=lambda: self.export_image(self.fig_c_spec, "13C_NMR")).pack(
            side=tk.LEFT, padx=3)

        self.fig_c_spec = Figure(figsize=(10, 5), dpi=100, facecolor="#ffffff")
        self.ax_c_spec = self.fig_c_spec.add_subplot(111)
        self._apply_plot_style(self.ax_c_spec)

        self.canvas_c_spec = FigureCanvasTkAgg(self.fig_c_spec, master=self.tab_c_plot)
        self.canvas_c_spec.draw();
        self.canvas_c_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_c_spec.mpl_connect("button_press_event", self._on_c_mouse_press)
        self.canvas_c_spec.mpl_connect("motion_notify_event", self._on_c_mouse_move)
        self.canvas_c_spec.mpl_connect("button_release_event", self._on_c_mouse_release)
        self.canvas_c_spec.mpl_connect("scroll_event", self._on_c_scroll)

    def scale_c_mol(self, factor):
        new_w = max(0.12, min(0.70, self.c_mol_pos[2] * factor))
        new_h = max(0.12, min(0.70, self.c_mol_pos[3] * factor))
        cx = self.c_mol_pos[0] + self.c_mol_pos[2] / 2.0
        cy = self.c_mol_pos[1] + self.c_mol_pos[3] / 2.0
        self.c_mol_pos[0] = max(0.0, min(1.0 - new_w, cx - new_w / 2.0))
        self.c_mol_pos[1] = max(0.0, min(1.0 - new_h, cy - new_h / 2.0))
        self.c_mol_pos[2] = new_w
        self.c_mol_pos[3] = new_h
        self._update_c_spectrum(preserve_zoom=True)

    def _on_c_scroll(self, event):
        if self.ax_c_mol is not None and self.show_c_mol.get():
            try:
                contains, _ = self.ax_c_mol.contains(event)
                if contains:
                    factor = 1.15 if (event.button == 'up' or getattr(event, 'step', 0) > 0) else (1.0 / 1.15)
                    self.scale_c_mol(factor)
                    return
            except Exception:
                pass

    # ----------------------------------------------------
    # TAB 5: 2D COSY WITH 1D PROJECTIONS & CORNER 2D MOL
    # ----------------------------------------------------
    def _build_tab_cosy(self):
        paned = ttk.PanedWindow(self.tab_cosy, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_plot = ttk.LabelFrame(paned, text=" 2D 1H-1H COSY with 1D Projections ")
        paned.add(frame_plot, weight=4)

        ctrl_bar = ttk.Frame(frame_plot)
        ctrl_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ctrl_bar, text="⏮ Reset Zoom", command=self.reset_cosy_zoom).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="💾 Save COSY", command=lambda: self.export_image(self.fig_cosy, "2D_COSY")).pack(
            side=tk.LEFT, padx=3)
        ttk.Label(ctrl_bar, text="💡 1D Projections on top & left | Drag: Zoom window | RMB: Reset zoom",
                  font=("Segoe UI", 8, "italic"), foreground="#444444").pack(side=tk.RIGHT, padx=5)

        self.fig_cosy = Figure(figsize=(8, 8), dpi=100, facecolor="#ffffff")
        gs = self.fig_cosy.add_gridspec(2, 2, width_ratios=[1.2, 5.5], height_ratios=[1.2, 5.5], wspace=0.03,
                                        hspace=0.03)

        self.ax_cosy_corner = self.fig_cosy.add_subplot(gs[0, 0])
        self.ax_cosy_top = self.fig_cosy.add_subplot(gs[0, 1])
        self.ax_cosy_left = self.fig_cosy.add_subplot(gs[1, 0])
        self.ax_cosy_main = self.fig_cosy.add_subplot(gs[1, 1], sharex=self.ax_cosy_top, sharey=self.ax_cosy_left)

        self.canvas_cosy = FigureCanvasTkAgg(self.fig_cosy, master=frame_plot)
        self.canvas_cosy.draw();
        self.canvas_cosy.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas_cosy.mpl_connect("button_press_event", self._on_cosy_mouse_press)
        self.canvas_cosy.mpl_connect("motion_notify_event", self._on_cosy_mouse_move)
        self.canvas_cosy.mpl_connect("button_release_event", self._on_cosy_mouse_release)

        frame_table = ttk.LabelFrame(paned, text=" COSY J-Couplings ")
        paned.add(frame_table, weight=1)
        cols = ("pair", "j_val", "shifts")
        self.tree_cosy = ttk.Treeview(frame_table, columns=cols, show='headings', selectmode='extended')
        self.tree_cosy.heading("pair", text="Pair")
        self.tree_cosy.heading("j_val", text="J (Hz)")
        self.tree_cosy.heading("shifts", text="δ1, δ2 (ppm)")
        self.tree_cosy.column("pair", width=80, anchor=tk.CENTER)
        self.tree_cosy.column("j_val", width=70, anchor=tk.CENTER)
        self.tree_cosy.column("shifts", width=110, anchor=tk.CENTER)
        scr = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree_cosy.yview)
        self.tree_cosy.configure(yscrollcommand=scr.set)
        self.tree_cosy.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0), pady=2)
        scr.pack(side=tk.RIGHT, fill=tk.Y, pady=2, padx=(0, 2))
        self._setup_clipboard(self.tree_cosy)

    # ----------------------------------------------------
    # TAB 6: 2D HSQC WITH 1D PROJECTIONS & CORNER 2D MOL
    # ----------------------------------------------------
    def _build_tab_hsqc(self):
        paned = ttk.PanedWindow(self.tab_hsqc, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_plot = ttk.LabelFrame(paned, text=" 2D HSQC with 1D Projections (CH/CH3 Red, CH2 Blue) ")
        paned.add(frame_plot, weight=4)

        ctrl_bar = ttk.Frame(frame_plot)
        ctrl_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ctrl_bar, text="⏮ Reset Zoom", command=self.reset_hsqc_zoom).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="💾 Save HSQC", command=lambda: self.export_image(self.fig_hsqc, "2D_HSQC")).pack(
            side=tk.LEFT, padx=3)
        ttk.Label(ctrl_bar, text="● Red: CH/CH3  ● Blue: CH2", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT,
                                                                                                  padx=10)

        self.fig_hsqc = Figure(figsize=(8, 8), dpi=100, facecolor="#ffffff")
        gs = self.fig_hsqc.add_gridspec(2, 2, width_ratios=[1.2, 5.5], height_ratios=[1.2, 5.5], wspace=0.03,
                                        hspace=0.03)

        self.ax_hsqc_corner = self.fig_hsqc.add_subplot(gs[0, 0])
        self.ax_hsqc_top = self.fig_hsqc.add_subplot(gs[0, 1])
        self.ax_hsqc_left = self.fig_hsqc.add_subplot(gs[1, 0])
        self.ax_hsqc_main = self.fig_hsqc.add_subplot(gs[1, 1], sharex=self.ax_hsqc_top, sharey=self.ax_hsqc_left)

        self.canvas_hsqc = FigureCanvasTkAgg(self.fig_hsqc, master=frame_plot)
        self.canvas_hsqc.draw();
        self.canvas_hsqc.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas_hsqc.mpl_connect("button_press_event", self._on_hsqc_mouse_press)
        self.canvas_hsqc.mpl_connect("motion_notify_event", self._on_hsqc_mouse_move)
        self.canvas_hsqc.mpl_connect("button_release_event", self._on_hsqc_mouse_release)

        frame_table = ttk.LabelFrame(paned, text=" 1J(C-H) Pairs ")
        paned.add(frame_table, weight=1)
        cols = ("pair", "type", "c_shift", "h_shift")
        self.tree_hsqc = ttk.Treeview(frame_table, columns=cols, show='headings', selectmode='extended')
        self.tree_hsqc.heading("pair", text="Pair")
        self.tree_hsqc.heading("type", text="Type")
        self.tree_hsqc.heading("c_shift", text="13C (ppm)")
        self.tree_hsqc.heading("h_shift", text="1H (ppm)")
        self.tree_hsqc.column("pair", width=80, anchor=tk.CENTER)
        self.tree_hsqc.column("type", width=55, anchor=tk.CENTER)
        self.tree_hsqc.column("c_shift", width=75, anchor=tk.CENTER)
        self.tree_hsqc.column("h_shift", width=75, anchor=tk.CENTER)
        scr = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree_hsqc.yview)
        self.tree_hsqc.configure(yscrollcommand=scr.set)
        self.tree_hsqc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0), pady=2)
        scr.pack(side=tk.RIGHT, fill=tk.Y, pady=2, padx=(0, 2))
        self._setup_clipboard(self.tree_hsqc)

    # ----------------------------------------------------
    # TAB 7: REPORT & EXPORT
    # ----------------------------------------------------
    def _build_tab_table(self):
        top_bar = ttk.Frame(self.tab_table)
        top_bar.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(top_bar, text="💾 Export to CSV / Excel", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="📄 Export to TXT (Report)", command=self.export_txt).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="📋 Copy ACS Block", command=self.copy_acs_to_clipboard).pack(side=tk.LEFT, padx=5)

        cols = ("nuc", "idx", "atom", "shift", "integral", "mult", "couplings")
        self.tree = ttk.Treeview(self.tab_table, columns=cols, show='headings', selectmode='extended')
        self.tree.heading("nuc", text="Nucleus")
        self.tree.heading("idx", text="Atom #")
        self.tree.heading("atom", text="Assignment")
        self.tree.heading("shift", text="δ (ppm)")
        self.tree.heading("integral", text="Integral")
        self.tree.heading("mult", text="Multiplicity")
        self.tree.heading("couplings", text="J Couplings (Hz)")
        self.tree.column("nuc", width=70, anchor=tk.CENTER)
        self.tree.column("idx", width=120, anchor=tk.CENTER)
        self.tree.column("atom", width=120, anchor=tk.CENTER)
        self.tree.column("shift", width=90, anchor=tk.CENTER)
        self.tree.column("integral", width=80, anchor=tk.CENTER)
        self.tree.column("mult", width=110, anchor=tk.CENTER)
        self.tree.column("couplings", width=420, anchor=tk.W)

        scroll_y = ttk.Scrollbar(self.tab_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 10))
        self._setup_clipboard(self.tree)

        f_bottom = ttk.LabelFrame(self.tab_table, text=" Formatted ACS String (1H, 13C, COSY & HSQC) ")
        f_bottom.pack(fill=tk.X, padx=10, pady=10)
        self.txt_acs = tk.Text(f_bottom, height=8, font=("Consolas", 10), wrap=tk.WORD)
        self.txt_acs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._setup_clipboard(self.txt_acs)

    def _apply_plot_style(self, ax):
        ax.set_facecolor("#ffffff")
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        ax.spines['left'].set_color('#888888');
        ax.spines['bottom'].set_color('#333333')
        ax.tick_params(axis='x', colors='#222222', labelsize=9.5)
        ax.tick_params(axis='y', colors='#666666', labelsize=9)
        ax.grid(False)

    # ----------------------------------------------------
    # LOAD & REFRESH LOGIC
    # ----------------------------------------------------
    def on_load_all_data(self):
        raw_shield = self.txt_shield.get("1.0", tk.END).strip()
        raw_j = self.txt_j.get("1.0", tk.END).strip()
        raw_xyz = self.txt_xyz.get("1.0", tk.END).strip()
        if not raw_shield:
            messagebox.showwarning("Warning", "Please paste isotropic shielding data.");
            return

        try:
            sigma_ref_h = float(self.ent_sigma_ref_h.get());
            delta_ref_h = float(self.ent_delta_ref_h.get())
            self.freq_h = float(self.ent_freq_h.get());
            self.lw_h = float(self.ent_lw_h.get())
            sigma_ref_c = float(self.ent_sigma_ref_c.get());
            delta_ref_c = float(self.ent_delta_ref_c.get())
            self.freq_c = float(self.ent_freq_c.get());
            self.lw_c = float(self.ent_lw_c.get())
        except ValueError:
            messagebox.showerror("Error", "Please check numerical input parameters.");
            return

        self.raw_shieldings = parse_shieldings(raw_shield)
        self.coupl_dict = parse_couplings(raw_j)
        self.mol_atoms = parse_xyz_coordinates(raw_xyz)

        all_h = sorted([idx for idx, data in self.raw_shieldings.items() if data['elem'] == 'H'])
        all_c = sorted([idx for idx, data in self.raw_shieldings.items() if data['elem'] == 'C'])
        if not all_h and not all_c:
            messagebox.showerror("Error", "No H or C atoms found in shielding table.");
            return

        self.raw_h_shifts = {p: delta_ref_h + (sigma_ref_h - self.raw_shieldings[p]['iso']) for p in all_h}
        self.raw_c_shifts = {c: delta_ref_c + (sigma_ref_c - self.raw_shieldings[c]['iso']) for c in all_c}
        self.h_groups = [[p] for p in all_h]
        self.c_groups = [[c] for c in all_c]
        self.selected_hydrogens.clear();
        self.selected_carbons.clear()
        self.h_selected_groups.clear();
        self.c_selected_groups.clear()
        self.decoupled_h_spin = None

        h_options = ["None"] + [f"H{p}" for p in all_h]
        self.combo_decouple['values'] = h_options
        self.combo_decouple.set("None")

        self._draw_3d_molecule();
        self._update_h_listbox();
        self._update_c_listbox()
        self._update_h_spectrum(preserve_zoom=False)
        self._update_c_spectrum(preserve_zoom=False)
        self._update_cosy_spectrum(preserve_zoom=False)
        self._update_hsqc_spectrum_and_table(preserve_zoom=False)
        self._populate_combined_table()
        self.notebook.select(self.tab_mol)

    def _refresh_all_spectra(self):
        self._update_h_spectrum(preserve_zoom=True)
        self._update_c_spectrum(preserve_zoom=True)
        self._update_cosy_spectrum(preserve_zoom=True)
        self._update_hsqc_spectrum_and_table(preserve_zoom=True)

    # ----------------------------------------------------
    # 3D MODEL
    # ----------------------------------------------------
    def _draw_3d_molecule(self):
        self.ax_mol.clear();
        self.ax_mol.set_facecolor("#ffffff");
        self.ax_mol.set_axis_off()
        self.scatter_indices.clear()
        if not self.mol_atoms:
            self.ax_mol.text2D(0.5, 0.5, "XYZ coordinates not provided.", ha='center', va='center', color='#666666',
                               transform=self.ax_mol.transAxes)
            self.canvas_mol.draw();
            return

        atom_keys = list(self.mol_atoms.keys())
        coords = np.array([self.mol_atoms[k]['xyz'] for k in atom_keys])
        elems = [self.mol_atoms[k]['elem'] for k in atom_keys]

        n_atoms = len(atom_keys)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                r1 = COVALENT_RADII.get(elems[i], 0.77);
                r2 = COVALENT_RADII.get(elems[j], 0.77)
                if 0.4 < dist < (r1 + r2 + 0.45):
                    self.ax_mol.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]],
                                     [coords[i][2], coords[j][2]], color='#777777', lw=1.8, zorder=1)

        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        colors, sizes = [], []
        for k, el in zip(atom_keys, elems):
            if el == 'H':
                if k in self.selected_hydrogens:
                    colors.append('#FFA500'); sizes.append(140)
                else:
                    in_m = any(k in g and len(g) > 1 for g in self.h_groups)
                    colors.append('#E88888' if in_m else CPK_COLORS.get(el, '#F0F0F0'));
                    sizes.append(90)
            elif el == 'C':
                if k in self.selected_carbons:
                    colors.append('#00BCD4'); sizes.append(220)
                else:
                    in_m = any(k in g and len(g) > 1 for g in self.c_groups)
                    colors.append('#5C8A66' if in_m else CPK_COLORS.get(el, '#333333'));
                    sizes.append(180)
            else:
                colors.append(CPK_COLORS.get(el, '#888888'));
                sizes.append(180)

        self.scatter_indices = atom_keys
        self.ax_mol.scatter(xs, ys, zs, s=sizes, c=colors, edgecolors='#222222', lw=0.9, alpha=0.95, picker=True,
                            pickradius=8, zorder=3)

        for k, (x, y, z), el in zip(atom_keys, coords, elems):
            if el == 'H':
                lbl_color = '#B25E00' if k in self.selected_hydrogens else '#1A1A1A'
                self.ax_mol.text(x + 0.08, y + 0.08, z + 0.08, f"H{k}", fontsize=8, color=lbl_color)
            elif el == 'C':
                lbl_color = '#006064' if k in self.selected_carbons else '#333333'
                self.ax_mol.text(x + 0.08, y + 0.08, z + 0.08, f"C{k}", fontsize=8.5, color=lbl_color,
                                 fontweight='bold')

        max_range = max(np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() * 0.58, 2.0)
        self.mol_center = np.array(
            [(xs.max() + xs.min()) * 0.5, (ys.max() + ys.min()) * 0.5, (zs.max() + zs.min()) * 0.5])
        self.mol_zoom_radius = max_range;
        self.mol_init_radius = max_range
        self.ax_mol.set_box_aspect((1, 1, 1))
        r = self.mol_zoom_radius;
        cx, cy, cz = self.mol_center
        self.ax_mol.set_xlim(cx - r, cx + r);
        self.ax_mol.set_ylim(cy - r, cy + r);
        self.ax_mol.set_zlim(cz - r, cz + r)
        self.canvas_mol.draw()

    def _update_h_listbox(self):
        self.h_listbox.delete(0, tk.END)
        for p in sorted(list(self.raw_h_shifts.keys())):
            shift = self.raw_h_shifts.get(p, 0.0)
            grp = next((g for g in self.h_groups if p in g), [p])
            grp_str = f"[Group: {len(grp)}H]" if len(grp) > 1 else ""
            self.h_listbox.insert(tk.END, f"H{p:<3}  {shift:>6.2f} ppm  {grp_str}")
            if p in self.selected_hydrogens: self.h_listbox.selection_set(self.h_listbox.size() - 1)

    def _update_c_listbox(self):
        self.c_listbox.delete(0, tk.END)
        for c in sorted(list(self.raw_c_shifts.keys())):
            shift = self.raw_c_shifts.get(c, 0.0)
            grp = next((g for g in self.c_groups if c in g), [c])
            grp_str = f"[Group: {len(grp)}C]" if len(grp) > 1 else ""
            self.c_listbox.insert(tk.END, f"C{c:<3}  {shift:>6.1f} ppm  {grp_str}")
            if c in self.selected_carbons: self.c_listbox.selection_set(self.c_listbox.size() - 1)

    def _on_3d_atom_picked(self, event):
        if not hasattr(event, 'ind') or len(event.ind) == 0: return
        idx = event.ind[0]
        if idx < len(self.scatter_indices):
            num = self.scatter_indices[idx]
            if num in self.raw_h_shifts:
                if num in self.selected_hydrogens:
                    self.selected_hydrogens.remove(num)
                else:
                    self.selected_hydrogens.add(num)
                self.sub_notebook.select(0);
                self._draw_3d_molecule();
                self._update_h_listbox()
            elif num in self.raw_c_shifts:
                if num in self.selected_carbons:
                    self.selected_carbons.remove(num)
                else:
                    self.selected_carbons.add(num)
                self.sub_notebook.select(1);
                self._draw_3d_molecule();
                self._update_c_listbox()

    def _on_h_listbox_selected(self, event):
        sel = self.h_listbox.curselection();
        all_p = sorted(list(self.raw_h_shifts.keys()))
        self.selected_hydrogens = {all_p[i] for i in sel if i < len(all_p)};
        self._draw_3d_molecule()

    def _on_c_listbox_selected(self, event):
        sel = self.c_listbox.curselection();
        all_c = sorted(list(self.raw_c_shifts.keys()))
        self.selected_carbons = {all_c[i] for i in sel if i < len(all_c)};
        self._draw_3d_molecule()

    def clear_hydrogen_selection(self):
        self.selected_hydrogens.clear();
        self._draw_3d_molecule();
        self._update_h_listbox()

    def clear_carbon_selection(self):
        self.selected_carbons.clear();
        self._draw_3d_molecule();
        self._update_c_listbox()

    def merge_selected_hydrogens(self):
        if len(self.selected_hydrogens) < 2: return
        new_g = [g for g in self.h_groups if not any(p in self.selected_hydrogens for p in g)]
        new_g.append(sorted(list(self.selected_hydrogens)))
        self.h_groups = new_g;
        self.selected_hydrogens.clear()
        self._draw_3d_molecule();
        self._update_h_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def ungroup_selected_hydrogens(self):
        if not self.selected_hydrogens: return
        new_g = []
        for g in self.h_groups:
            if any(p in self.selected_hydrogens for p in g) and len(g) > 1:
                for p in g: new_g.append([p])
            else:
                new_g.append(g)
        self.h_groups = new_g;
        self.selected_hydrogens.clear()
        self._draw_3d_molecule();
        self._update_h_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def merge_selected_carbons(self):
        if len(self.selected_carbons) < 2: return
        new_g = [g for g in self.c_groups if not any(c in self.selected_carbons for c in g)]
        new_g.append(sorted(list(self.selected_carbons)))
        self.c_groups = new_g;
        self.selected_carbons.clear()
        self._draw_3d_molecule();
        self._update_c_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def ungroup_selected_carbons(self):
        if not self.selected_carbons: return
        new_g = []
        for g in self.c_groups:
            if any(c in self.selected_carbons for c in g) and len(g) > 1:
                for c in g: new_g.append([c])
            else:
                new_g.append(g)
        self.c_groups = new_g;
        self.selected_carbons.clear()
        self._draw_3d_molecule();
        self._update_c_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    # ----------------------------------------------------
    # 1H SPECTRUM WITH SELECTIVE DECOUPLING & 2D MOL
    # ----------------------------------------------------
    def _update_h_spectrum(self, preserve_zoom=False):
        if not self.raw_h_shifts: return
        xlim = self.ax_h_spec.get_xlim() if preserve_zoom and self.h_ppm is not None else None
        ylim = self.ax_h_spec.get_ylim() if preserve_zoom and self.h_spec is not None else None

        self.h_signals = analyze_grouped_h_signals(self.raw_h_shifts, self.coupl_dict, self.h_groups)
        ppm, spec = simulate_grouped_h_molecule(
            self.raw_h_shifts, self.coupl_dict, self.h_groups, self.freq_h, self.lw_h,
            decoupled_spin=self.decoupled_h_spin
        )
        self.h_ppm, self.h_spec = ppm, spec
        self.h_annotations.clear()

        self.ax_h_spec.clear()
        self._apply_plot_style(self.ax_h_spec)
        self.ax_h_spec.axhline(0, color='#b0b0b0', lw=0.7)
        self.ax_h_spec.plot(ppm, spec, color='#801818', lw=1.35)

        max_i = np.max(spec) if np.max(spec) > 0 else 1.0
        self.h_init_xlim = (max(ppm), min(ppm))
        self.h_init_ylim = (-max_i * 0.04, max_i * 1.38)
        self.ax_h_spec.set_xlim(xlim or self.h_init_xlim)
        self.ax_h_spec.set_ylim(ylim or self.h_init_ylim)

        # Draw Decoupling / Irradiation Curve
        if self.decoupled_h_spin is not None and self.decoupled_h_spin in self.raw_h_shifts:
            irr_shift = self.raw_h_shifts[self.decoupled_h_spin]
            band = 0.05
            sat_ppm = np.linspace(irr_shift - band * 2.5, irr_shift + band * 2.5, 200)
            sat_curve = max_i * 0.95 * (1.0 - np.exp(-0.5 * ((sat_ppm - irr_shift) / (band * 0.7)) ** 2))
            self.ax_h_spec.plot(sat_ppm, sat_curve, color='#28a745', lw=1.5, linestyle='-', zorder=4)
            self.ax_h_spec.axvline(irr_shift - band, color='#0056b3', linestyle='-', lw=0.9, alpha=0.85)
            self.ax_h_spec.axvline(irr_shift + band, color='#0056b3', linestyle='-', lw=0.9, alpha=0.85)
            self.ax_h_spec.annotate(f"Irradiated H{self.decoupled_h_spin}\n({irr_shift:.2f} ppm)",
                                    xy=(irr_shift, max_i * 0.2), xytext=(irr_shift, max_i * 0.6),
                                    ha='center', fontsize=8.5, fontweight='bold', color='#155724',
                                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#28a745"))

        # Multiplet labels
        for sig in self.h_signals:
            p_shift = sig['shift']
            idx = np.argmin(np.abs(ppm - p_shift))
            max_v = np.max(spec[max(0, idx - 120):min(len(spec), idx + 120)])
            ann = self.ax_h_spec.annotate(
                f"{sig['label']}\n({p_shift:.2f})", xy=(p_shift, max_v), xytext=(p_shift, max_v * 1.15 + max_i * 0.02),
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec="#6c757d", lw=0.85),
                arrowprops=dict(arrowstyle="->", lw=0.85, color="#495057")
            )
            ann._group_idx = self.h_groups.index(sig['raw_group'])
            self.h_annotations.append(ann)

        self.ax_h_spec.set_xlabel(r"Chemical Shift $\delta$ (ppm)", fontsize=11, fontweight='semibold')
        self.ax_h_spec.set_title(f"1H NMR Spectrum ({self.freq_h:.0f} MHz)", fontsize=12, fontweight='bold')

        # Draw 2D Molecule Inset on top of spectrum
        if self.show_h_mol.get() and self.mol_atoms:
            self.ax_h_mol = self.ax_h_spec.inset_axes(self.h_mol_pos, zorder=10)
            render_2d_molecule_on_ax(self.ax_h_mol, self.mol_atoms, title="2D Structure (drag/scroll)")
        else:
            self.ax_h_mol = None

        self.fig_h_spec.tight_layout()
        self.canvas_h_spec.draw()

    # ----------------------------------------------------
    # 13C SPECTRUM WITH 2D MOL
    # ----------------------------------------------------
    def _update_c_spectrum(self, preserve_zoom=False):
        if not self.raw_c_shifts: return
        xlim = self.ax_c_spec.get_xlim() if preserve_zoom and self.c_ppm is not None else None
        ylim = self.ax_c_spec.get_ylim() if preserve_zoom and self.c_spec is not None else None

        self.c_signals = analyze_grouped_c_signals(self.raw_c_shifts, self.c_groups)
        ppm, spec = simulate_grouped_c13_spectrum(self.raw_c_shifts, self.c_groups, self.freq_c, self.lw_c)
        self.c_ppm, self.c_spec = ppm, spec;
        self.c_annotations.clear()

        self.ax_c_spec.clear()
        self._apply_plot_style(self.ax_c_spec)
        self.ax_c_spec.axhline(0, color='#b0b0b0', lw=0.7)
        self.ax_c_spec.plot(ppm, spec, color='#1e7e34', lw=1.35)

        max_i = np.max(spec) if np.max(spec) > 0 else 1.0
        self.c_init_xlim = (max(ppm), min(ppm))
        self.c_init_ylim = (-max_i * 0.04, max_i * 1.38)
        self.ax_c_spec.set_xlim(xlim or self.c_init_xlim)
        self.ax_c_spec.set_ylim(ylim or self.c_init_ylim)

        for sig in self.c_signals:
            p_shift = sig['shift'];
            idx = np.argmin(np.abs(ppm - p_shift))
            max_v = np.max(spec[max(0, idx - 100):min(len(spec), idx + 100)])
            ann = self.ax_c_spec.annotate(
                f"{sig['label']}\n{p_shift:.1f}", xy=(p_shift, max_v), xytext=(p_shift, max_v * 1.15),
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#155724',
                bbox=dict(boxstyle="round,pad=0.28", fc="#f4faf5", ec="#28a745", lw=0.9),
                arrowprops=dict(arrowstyle="->", lw=0.85, color="#28a745")
            )
            ann._group_idx = self.c_groups.index(sig['raw_group'])
            self.c_annotations.append(ann)

        self.ax_c_spec.set_xlabel(r"Chemical Shift $\delta$ (ppm)", fontsize=11, fontweight='semibold')
        self.ax_c_spec.set_title(f"13C NMR Spectrum ({self.freq_c:.1f} MHz, 1H-Decoupled)", fontsize=12,
                                 fontweight='bold')

        # Draw 2D Molecule Inset on 13C
        if self.show_c_mol.get() and self.mol_atoms:
            self.ax_c_mol = self.ax_c_spec.inset_axes(self.c_mol_pos, zorder=10)
            render_2d_molecule_on_ax(self.ax_c_mol, self.mol_atoms, title="2D Structure (drag/scroll)")
        else:
            self.ax_c_mol = None

        self.fig_c_spec.tight_layout()
        self.canvas_c_spec.draw()

    # ----------------------------------------------------
    # 2D COSY WITH 1D PROJECTIONS
    # ----------------------------------------------------
    def _update_cosy_spectrum(self, preserve_zoom=False):
        if not self.raw_h_shifts: return
        xlim = self.ax_cosy_main.get_xlim() if preserve_zoom else None

        act_shifts = {p: float(np.mean([self.raw_h_shifts[x] for x in g])) for g in self.h_groups for p in g}
        self.cosy_correlations = compute_cosy_correlations(self.coupl_dict, act_shifts, j_min=0.5)

        self.tree_cosy.delete(*self.tree_cosy.get_children())
        for c in self.cosy_correlations:
            self.tree_cosy.insert("", tk.END, values=(f"{c['h1_label']}–{c['h2_label']}", f"{c['j_val']:.2f}",
                                                      f"{c['h1_shift']:.2f}, {c['h2_shift']:.2f}"))

        for a in [self.ax_cosy_corner, self.ax_cosy_top, self.ax_cosy_left, self.ax_cosy_main]:
            a.clear()

        render_2d_molecule_on_ax(self.ax_cosy_corner, self.mol_atoms)

        h_vals = list(act_shifts.values())
        h_min, h_max = max(0.0, min(h_vals) - 0.5), max(h_vals) + 0.5
        ppm_1d, spec_1d = simulate_grouped_h_molecule(self.raw_h_shifts, self.coupl_dict, self.h_groups, self.freq_h,
                                                      self.lw_h)

        self.ax_cosy_top.plot(ppm_1d, spec_1d, color='black', lw=0.95)
        self.ax_cosy_top.axis('off');
        self.ax_cosy_top.set_ylim(0, np.max(spec_1d) * 1.15)

        self.ax_cosy_left.plot(spec_1d, ppm_1d, color='black', lw=0.95)
        self.ax_cosy_left.axis('off');
        self.ax_cosy_left.invert_xaxis()
        self.ax_cosy_left.set_xlim(np.max(spec_1d) * 1.15, 0)

        self._apply_plot_style(self.ax_cosy_main)
        self.ax_cosy_main.grid(True, linestyle='--', alpha=0.45, color='#d0d0d0')
        axis, mat = simulate_2d_cosy_map(h_vals, self.cosy_correlations, h_range=(h_min, h_max))

        if np.max(mat) > 0.05:
            levels = np.linspace(0.12, np.max(mat), 6)
            self.ax_cosy_main.contour(axis, axis, mat, levels=levels, colors='#dc3545', linewidths=1.25)

        self.ax_cosy_main.plot([h_max, h_min], [h_max, h_min], color='#999999', linestyle=':', lw=1.0)
        self.cosy_init_lim = (h_max, h_min)
        self.ax_cosy_main.set_xlim(xlim or self.cosy_init_lim)
        self.ax_cosy_main.set_ylim(xlim or self.cosy_init_lim)
        self.ax_cosy_main.set_xlabel(r"F2: 1H Chemical Shift $\delta$ (ppm)", fontsize=10, fontweight='semibold')
        self.ax_cosy_main.set_ylabel(r"F1: 1H Chemical Shift $\delta$ (ppm)", fontsize=10, fontweight='semibold')
        self.canvas_cosy.draw()

    # ----------------------------------------------------
    # 2D HSQC WITH 1D PROJECTIONS
    # ----------------------------------------------------
    def _update_hsqc_spectrum_and_table(self, preserve_zoom=False):
        if not self.mol_atoms or not self.raw_h_shifts or not self.raw_c_shifts: return
        xlim = self.ax_hsqc_main.get_xlim() if preserve_zoom and self.hsqc_correlations else None
        ylim = self.ax_hsqc_main.get_ylim() if preserve_zoom and self.hsqc_correlations else None

        act_h = {p: float(np.mean([self.raw_h_shifts[x] for x in g])) for g in self.h_groups for p in g}
        act_c = {c: float(np.mean([self.raw_c_shifts[x] for x in g])) for g in self.c_groups for c in g}

        self.hsqc_correlations = compute_hsqc_correlations(self.mol_atoms, act_h, act_c, max_bond_dist=1.30)
        self.tree_hsqc.delete(*self.tree_hsqc.get_children())
        for p in self.hsqc_correlations:
            self.tree_hsqc.insert("", tk.END,
                                  values=(f"{p['c_label']}–{p['h_label']}", p['group_type'], f"{p['c_shift']:.1f}",
                                          f"{p['h_shift']:.2f}"))

        for a in [self.ax_hsqc_corner, self.ax_hsqc_top, self.ax_hsqc_left, self.ax_hsqc_main]:
            a.clear()

        render_2d_molecule_on_ax(self.ax_hsqc_corner, self.mol_atoms)

        ppm_1d_h, spec_1d_h = simulate_grouped_h_molecule(self.raw_h_shifts, self.coupl_dict, self.h_groups,
                                                          self.freq_h, self.lw_h)
        ppm_1d_c, spec_1d_c = simulate_grouped_c13_spectrum(self.raw_c_shifts, self.c_groups, self.freq_c, self.lw_c)

        self.ax_hsqc_top.plot(ppm_1d_h, spec_1d_h, color='black', lw=0.95)
        self.ax_hsqc_top.axis('off');
        self.ax_hsqc_top.set_ylim(0, np.max(spec_1d_h) * 1.15)

        self.ax_hsqc_left.plot(spec_1d_c, ppm_1d_c, color='black', lw=0.95)
        self.ax_hsqc_left.axis('off');
        self.ax_hsqc_left.invert_xaxis()
        self.ax_hsqc_left.set_xlim(np.max(spec_1d_c) * 1.15, 0)

        self._apply_plot_style(self.ax_hsqc_main)
        self.ax_hsqc_main.grid(True, linestyle='--', alpha=0.45, color='#d0d0d0')

        if self.hsqc_correlations:
            h_vals = [p['h_shift'] for p in self.hsqc_correlations];
            c_vals = [p['c_shift'] for p in self.hsqc_correlations]
            h_min, h_max = max(0.0, min(h_vals) - 0.6), max(h_vals) + 0.6
            c_min, c_max = max(0.0, min(c_vals) - 10.0), max(c_vals) + 15.0

            h_ax, c_ax, m_pos, m_neg = simulate_2d_hsqc_map(self.hsqc_correlations, h_range=(h_min, h_max),
                                                            c_range=(c_min, c_max))
            if np.max(m_pos) > 0.05:
                self.ax_hsqc_main.contour(h_ax, c_ax, m_pos, levels=np.linspace(0.12, np.max(m_pos), 5),
                                          colors='#dc3545', linewidths=1.3)
            if np.max(m_neg) > 0.05:
                self.ax_hsqc_main.contour(h_ax, c_ax, m_neg, levels=np.linspace(0.12, np.max(m_neg), 5),
                                          colors='#0056b3', linewidths=1.3)

            for p in self.hsqc_correlations:
                col = '#0056b3' if p['phase'] < 0 else '#dc3545'
                self.ax_hsqc_main.scatter(p['h_shift'], p['c_shift'], s=35, color=col, edgecolors='#ffffff', zorder=4)
                self.ax_hsqc_main.annotate(f"{p['c_label']}-{p['h_label']}\n({p['group_type']})",
                                           xy=(p['h_shift'], p['c_shift']),
                                           xytext=(p['h_shift'] + 0.08, p['c_shift'] + 3.0), fontsize=8,
                                           fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=col, lw=0.8,
                                                     alpha=0.85))

            self.hsqc_init_xlim = (h_max, h_min);
            self.hsqc_init_ylim = (c_max, c_min)

        self.ax_hsqc_main.set_xlim(xlim or self.hsqc_init_xlim)
        self.ax_hsqc_main.set_ylim(ylim or self.hsqc_init_ylim)
        self.ax_hsqc_main.set_xlabel(r"1H Chemical Shift $\delta$ (ppm)", fontsize=10, fontweight='semibold')
        self.ax_hsqc_main.set_ylabel(r"13C Chemical Shift $\delta$ (ppm)", fontsize=10, fontweight='semibold')
        self.canvas_hsqc.draw()

    # ----------------------------------------------------
    # ZOOM / PAN 2D MAPS
    # ----------------------------------------------------
    def _on_cosy_mouse_press(self, event):
        if event.inaxes != self.ax_cosy_main: return
        if event.button == 3:
            if self.cosy_zoom_history:
                prev = self.cosy_zoom_history.pop()
                self.ax_cosy_main.set_xlim(prev[0]);
                self.ax_cosy_main.set_ylim(prev[1])
            else:
                self.ax_cosy_main.set_xlim(self.cosy_init_lim); self.ax_cosy_main.set_ylim(self.cosy_init_lim)
            self.canvas_cosy.draw_idle();
            return
        if event.button == 1: self.cosy_zoom_start = (event.xdata, event.ydata)

    def _on_cosy_mouse_move(self, event):
        if event.inaxes != self.ax_cosy_main or self.cosy_zoom_start is None: return
        if event.xdata is not None and event.ydata is not None:
            if self.cosy_zoom_rect: self.cosy_zoom_rect.remove()
            x0, y0 = self.cosy_zoom_start
            self.cosy_zoom_rect = self.ax_cosy_main.add_patch(
                matplotlib.patches.Rectangle((min(x0, event.xdata), min(y0, event.ydata)),
                                             abs(event.xdata - x0), abs(event.ydata - y0),
                                             color='#dc3545', alpha=0.15, linestyle='--', ec='#dc3545')
            )
            self.canvas_cosy.draw_idle()

    def _on_cosy_mouse_release(self, event):
        if event.button == 1 and self.cosy_zoom_start is not None:
            if self.cosy_zoom_rect: self.cosy_zoom_rect.remove(); self.cosy_zoom_rect = None
            if event.inaxes == self.ax_cosy_main and event.xdata is not None and event.ydata is not None:
                x0, y0 = self.cosy_zoom_start;
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) >= 0.05 and abs(y1 - y0) >= 0.05:
                    self.cosy_zoom_history.append((self.ax_cosy_main.get_xlim(), self.ax_cosy_main.get_ylim()))
                    self.ax_cosy_main.set_xlim(max(x0, x1), min(x0, x1))
                    self.ax_cosy_main.set_ylim(max(y0, y1), min(y0, y1))
                    self.canvas_cosy.draw_idle()
            self.cosy_zoom_start = None

    def reset_cosy_zoom(self):
        self.cosy_zoom_history.clear()
        self.ax_cosy_main.set_xlim(self.cosy_init_lim);
        self.ax_cosy_main.set_ylim(self.cosy_init_lim)
        self.canvas_cosy.draw_idle()

    def _on_hsqc_mouse_press(self, event):
        if event.inaxes != self.ax_hsqc_main: return
        if event.button == 3:
            if self.hsqc_zoom_history:
                prev = self.hsqc_zoom_history.pop();
                self.ax_hsqc_main.set_xlim(prev[0]);
                self.ax_hsqc_main.set_ylim(prev[1])
            else:
                self.ax_hsqc_main.set_xlim(self.hsqc_init_xlim); self.ax_hsqc_main.set_ylim(self.hsqc_init_ylim)
            self.canvas_hsqc.draw_idle();
            return
        if event.button == 1: self.hsqc_zoom_start = (event.xdata, event.ydata)

    def _on_hsqc_mouse_move(self, event):
        if event.inaxes != self.ax_hsqc_main or self.hsqc_zoom_start is None: return
        if event.xdata is not None and event.ydata is not None:
            if self.hsqc_zoom_rect: self.hsqc_zoom_rect.remove()
            x0, y0 = self.hsqc_zoom_start
            self.hsqc_zoom_rect = self.ax_hsqc_main.add_patch(
                matplotlib.patches.Rectangle((min(x0, event.xdata), min(y0, event.ydata)),
                                             abs(event.xdata - x0), abs(event.ydata - y0),
                                             color='#0f4c81', alpha=0.15, linestyle='--', ec='#0f4c81')
            )
            self.canvas_hsqc.draw_idle()

    def _on_hsqc_mouse_release(self, event):
        if event.button == 1 and self.hsqc_zoom_start is not None:
            if self.hsqc_zoom_rect: self.hsqc_zoom_rect.remove(); self.hsqc_zoom_rect = None
            if event.inaxes == self.ax_hsqc_main and event.xdata is not None and event.ydata is not None:
                x0, y0 = self.hsqc_zoom_start;
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) >= 0.05 and abs(y1 - y0) >= 1.0:
                    self.hsqc_zoom_history.append((self.ax_hsqc_main.get_xlim(), self.ax_hsqc_main.get_ylim()))
                    self.ax_hsqc_main.set_xlim(max(x0, x1), min(x0, x1))
                    self.ax_hsqc_main.set_ylim(max(y0, y1), min(y0, y1))
                    self.canvas_hsqc.draw_idle()
            self.hsqc_zoom_start = None

    def reset_hsqc_zoom(self):
        self.hsqc_zoom_history.clear()
        self.ax_hsqc_main.set_xlim(self.hsqc_init_xlim);
        self.ax_hsqc_main.set_ylim(self.hsqc_init_ylim)
        self.canvas_hsqc.draw_idle()

    # ----------------------------------------------------
    # MOUSE HANDLERS (1H & 13C 1D) INCLUDING 2D MOL DRAG
    # ----------------------------------------------------
    def _on_h_mouse_press(self, event):
        # 1. Check if user clicked inside 2D molecule window (Drag start)
        if self.ax_h_mol is not None and self.show_h_mol.get():
            try:
                contains, _ = self.ax_h_mol.contains(event)
                if contains:
                    self.h_drag_mol = True
                    self.h_drag_mol_last = (event.x, event.y)
                    return
            except Exception:
                pass

        if event.inaxes != self.ax_h_spec: return
        if event.button == 3:
            if self.h_zoom_history:
                prev = self.h_zoom_history.pop();
                self.ax_h_spec.set_xlim(prev[0]);
                self.ax_h_spec.set_ylim(prev[1])
            else:
                self.ax_h_spec.set_xlim(self.h_init_xlim); self.ax_h_spec.set_ylim(self.h_init_ylim)
            self.canvas_h_spec.draw_idle();
            return

        is_shift = bool(
            event.key == 'shift' or (hasattr(event, 'guiEvent') and getattr(event.guiEvent, 'state', 0) & 0x0001))
        if event.button == 1:
            for ann in reversed(self.h_annotations):
                if ann.contains(event)[0]:
                    g = getattr(ann, '_group_idx', None)
                    if is_shift and g is not None:
                        if g in self.h_selected_groups:
                            self.h_selected_groups.remove(g)
                        else:
                            self.h_selected_groups.add(g)
                        self._update_h_annotation_highlight();
                        return
                    else:
                        self.h_dragged_ann = ann
                        self.h_drag_offset = (ann.get_position()[0] - event.xdata, ann.get_position()[1] - event.ydata);
                        return
            if not is_shift: self.h_zoom_start_x = event.xdata

    def _on_h_mouse_move(self, event):
        # Dragging 2D Molecule Inset across the spectrum
        if getattr(self, 'h_drag_mol', False) and self.ax_h_mol is not None:
            bbox = self.ax_h_spec.get_window_extent()
            dx = (event.x - self.h_drag_mol_last[0]) / max(bbox.width, 1)
            dy = (event.y - self.h_drag_mol_last[1]) / max(bbox.height, 1)
            self.h_mol_pos[0] = max(0.0, min(1.0 - self.h_mol_pos[2], self.h_mol_pos[0] + dx))
            self.h_mol_pos[1] = max(0.0, min(1.0 - self.h_mol_pos[3], self.h_mol_pos[1] + dy))
            self.h_drag_mol_last = (event.x, event.y)
            self._update_h_spectrum(preserve_zoom=True)
            return

        if event.inaxes != self.ax_h_spec: return
        if self.h_dragged_ann:
            self.h_dragged_ann.set_position((event.xdata + self.h_drag_offset[0], event.ydata + self.h_drag_offset[1]))
            self.canvas_h_spec.draw_idle();
            return
        if self.h_zoom_start_x is not None and event.xdata is not None:
            if self.h_zoom_rect: self.h_zoom_rect.remove()
            self.h_zoom_rect = self.ax_h_spec.axvspan(min(self.h_zoom_start_x, event.xdata),
                                                      max(self.h_zoom_start_x, event.xdata), color='#185a9d',
                                                      alpha=0.20, linestyle='--')
            self.canvas_h_spec.draw_idle()

    def _on_h_mouse_release(self, event):
        if getattr(self, 'h_drag_mol', False):
            self.h_drag_mol = False
            return
        if event.button == 1:
            if self.h_dragged_ann: self.h_dragged_ann = None; return
            if self.h_zoom_start_x is not None:
                if self.h_zoom_rect: self.h_zoom_rect.remove(); self.h_zoom_rect = None
                if event.inaxes == self.ax_h_spec and event.xdata is not None:
                    if abs(event.xdata - self.h_zoom_start_x) >= 0.02:
                        self.h_zoom_history.append((self.ax_h_spec.get_xlim(), self.ax_h_spec.get_ylim()))
                        self.ax_h_spec.set_xlim(max(self.h_zoom_start_x, event.xdata),
                                                min(self.h_zoom_start_x, event.xdata))
                        self.autoscale_h_y()
                self.h_zoom_start_x = None

    def _update_h_annotation_highlight(self):
        for ann in self.h_annotations:
            g = getattr(ann, '_group_idx', None);
            p = ann.get_bbox_patch()
            if g in self.h_selected_groups:
                p.set_facecolor('#FFF3CD');
                p.set_edgecolor('#FF9800');
                p.set_linewidth(2.2);
                ann.set_color('#B25E00')
            else:
                p.set_facecolor('#FFFFFF');
                p.set_edgecolor('#6C757D');
                p.set_linewidth(0.85);
                ann.set_color('#1A1A1A')
        self.canvas_h_spec.draw_idle()

    def merge_selected_signals_plot(self):
        if len(self.h_selected_groups) < 2: return
        merged = [p for idx in self.h_selected_groups for p in self.h_groups[idx]]
        new_g = [g for i, g in enumerate(self.h_groups) if i not in self.h_selected_groups]
        new_g.append(sorted(list(set(merged))))
        self.h_groups = new_g;
        self.h_selected_groups.clear()
        self._draw_3d_molecule();
        self._update_h_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def ungroup_selected_signals_plot(self):
        if not self.h_selected_groups: return
        new_g = []
        for i, g in enumerate(self.h_groups):
            if i in self.h_selected_groups and len(g) > 1:
                for p in g: new_g.append([p])
            else:
                new_g.append(g)
        self.h_groups = new_g;
        self.h_selected_groups.clear()
        self._draw_3d_molecule();
        self._update_h_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def reset_h_zoom(self):
        self.h_zoom_history.clear()
        self.ax_h_spec.set_xlim(self.h_init_xlim);
        self.ax_h_spec.set_ylim(self.h_init_ylim);
        self.canvas_h_spec.draw_idle()

    def autoscale_h_y(self):
        if self.h_ppm is not None and self.h_spec is not None:
            xlim = self.ax_h_spec.get_xlim()
            mask = (self.h_ppm >= min(xlim)) & (self.h_ppm <= max(xlim))
            if np.any(mask):
                max_i = np.max(self.h_spec[mask])
                if max_i > 0: self.ax_h_spec.set_ylim(-max_i * 0.04, max_i * 1.35)
                self.canvas_h_spec.draw_idle()

    def _on_c_mouse_press(self, event):
        if self.ax_c_mol is not None and self.show_c_mol.get():
            try:
                contains, _ = self.ax_c_mol.contains(event)
                if contains:
                    self.c_drag_mol = True
                    self.c_drag_mol_last = (event.x, event.y)
                    return
            except Exception:
                pass

        if event.inaxes != self.ax_c_spec: return
        if event.button == 3:
            if self.c_zoom_history:
                prev = self.c_zoom_history.pop();
                self.ax_c_spec.set_xlim(prev[0]);
                self.ax_c_spec.set_ylim(prev[1])
            else:
                self.ax_c_spec.set_xlim(self.c_init_xlim); self.ax_c_spec.set_ylim(self.c_init_ylim)
            self.canvas_c_spec.draw_idle();
            return

        is_shift = bool(
            event.key == 'shift' or (hasattr(event, 'guiEvent') and getattr(event.guiEvent, 'state', 0) & 0x0001))
        if event.button == 1:
            for ann in reversed(self.c_annotations):
                if ann.contains(event)[0]:
                    g = getattr(ann, '_group_idx', None)
                    if is_shift and g is not None:
                        if g in self.c_selected_groups:
                            self.c_selected_groups.remove(g)
                        else:
                            self.c_selected_groups.add(g)
                        self._update_c_annotation_highlight();
                        return
                    else:
                        self.c_dragged_ann = ann
                        self.c_drag_offset = (ann.get_position()[0] - event.xdata, ann.get_position()[1] - event.ydata);
                        return
            if not is_shift: self.c_zoom_start_x = event.xdata

    def _on_c_mouse_move(self, event):
        if getattr(self, 'c_drag_mol', False) and self.ax_c_mol is not None:
            bbox = self.ax_c_spec.get_window_extent()
            dx = (event.x - self.c_drag_mol_last[0]) / max(bbox.width, 1)
            dy = (event.y - self.c_drag_mol_last[1]) / max(bbox.height, 1)
            self.c_mol_pos[0] = max(0.0, min(1.0 - self.c_mol_pos[2], self.c_mol_pos[0] + dx))
            self.c_mol_pos[1] = max(0.0, min(1.0 - self.c_mol_pos[3], self.c_mol_pos[1] + dy))
            self.c_drag_mol_last = (event.x, event.y)
            self._update_c_spectrum(preserve_zoom=True)
            return

        if event.inaxes != self.ax_c_spec: return
        if self.c_dragged_ann:
            self.c_dragged_ann.set_position((event.xdata + self.c_drag_offset[0], event.ydata + self.c_drag_offset[1]))
            self.canvas_c_spec.draw_idle();
            return
        if self.c_zoom_start_x is not None and event.xdata is not None:
            if self.c_zoom_rect: self.c_zoom_rect.remove()
            self.c_zoom_rect = self.ax_c_spec.axvspan(min(self.c_zoom_start_x, event.xdata),
                                                      max(self.c_zoom_start_x, event.xdata), color='#28a745',
                                                      alpha=0.18, linestyle='--')
            self.canvas_c_spec.draw_idle()

    def _on_c_mouse_release(self, event):
        if getattr(self, 'c_drag_mol', False):
            self.c_drag_mol = False
            return
        if event.button == 1:
            if self.c_dragged_ann: self.c_dragged_ann = None; return
            if self.c_zoom_start_x is not None:
                if self.c_zoom_rect: self.c_zoom_rect.remove(); self.c_zoom_rect = None
                if event.inaxes == self.ax_c_spec and event.xdata is not None:
                    if abs(event.xdata - self.c_zoom_start_x) >= 0.5:
                        self.c_zoom_history.append((self.ax_c_spec.get_xlim(), self.ax_c_spec.get_ylim()))
                        self.ax_c_spec.set_xlim(max(self.c_zoom_start_x, event.xdata),
                                                min(self.c_zoom_start_x, event.xdata))
                        self.autoscale_c_y()
                self.c_zoom_start_x = None

    def _update_c_annotation_highlight(self):
        for ann in self.c_annotations:
            g = getattr(ann, '_group_idx', None);
            p = ann.get_bbox_patch()
            if g in self.c_selected_groups:
                p.set_facecolor('#FFF3CD');
                p.set_edgecolor('#FF9800');
                p.set_linewidth(2.2);
                ann.set_color('#B25E00')
            else:
                p.set_facecolor('#F4FAF5');
                p.set_edgecolor('#28A745');
                p.set_linewidth(0.9);
                ann.set_color('#155724')
        self.canvas_c_spec.draw_idle()

    def merge_selected_c_signals_plot(self):
        if len(self.c_selected_groups) < 2: return
        merged = [c for idx in self.c_selected_groups for c in self.c_groups[idx]]
        new_g = [g for i, g in enumerate(self.c_groups) if i not in self.c_selected_groups]
        new_g.append(sorted(list(set(merged))))
        self.c_groups = new_g;
        self.c_selected_groups.clear()
        self._draw_3d_molecule();
        self._update_c_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def ungroup_selected_c_signals_plot(self):
        if not self.c_selected_groups: return
        new_g = []
        for i, g in enumerate(self.c_groups):
            if i in self.c_selected_groups and len(g) > 1:
                for c in g: new_g.append([c])
            else:
                new_g.append(g)
        self.c_groups = new_g;
        self.c_selected_groups.clear()
        self._draw_3d_molecule();
        self._update_c_listbox()
        self._refresh_all_spectra();
        self._populate_combined_table()

    def reset_c_zoom(self):
        self.c_zoom_history.clear()
        self.ax_c_spec.set_xlim(self.c_init_xlim);
        self.ax_c_spec.set_ylim(self.c_init_ylim);
        self.canvas_c_spec.draw_idle()

    def autoscale_c_y(self):
        if self.c_ppm is not None and self.c_spec is not None:
            xlim = self.ax_c_spec.get_xlim()
            mask = (self.c_ppm >= min(xlim)) & (self.c_ppm <= max(xlim))
            if np.any(mask):
                max_i = np.max(self.c_spec[mask])
                if max_i > 0: self.ax_c_spec.set_ylim(-max_i * 0.04, max_i * 1.35)
                self.canvas_c_spec.draw_idle()

    # ----------------------------------------------------
    # COMBINED REPORT & EXPORT
    # ----------------------------------------------------
    def _populate_combined_table(self):
        self.tree.delete(*self.tree.get_children())
        for s in self.h_signals:
            self.tree.insert("", tk.END,
                             values=(s['nuc'], s['indices'], s['label'], f"{s['shift']:.2f}", s['integral'], s['mult'],
                                     s['j_str']))
        for s in self.c_signals:
            self.tree.insert("", tk.END,
                             values=(s['nuc'], s['indices'], s['label'], f"{s['shift']:.2f}", s['integral'], s['mult'],
                                     s['j_str']))

        full_acs = build_acs_publication_block(
            self.h_signals, self.c_signals, self.hsqc_correlations,
            self.cosy_correlations, self.freq_h, self.freq_c
        )
        self.txt_acs.delete("1.0", tk.END);
        self.txt_acs.insert("1.0", full_acs)

    def export_image(self, fig, tag):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Vector", "*.pdf"), ("SVG Vector", "*.svg"), ("All Files", "*.*")],
            title=f"Save {tag} Figure"
        )
        if filepath:
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            messagebox.showinfo("Success", f"Figure saved to:\n{filepath}")

    def export_csv(self):
        if not self.h_signals and not self.c_signals: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filepath:
            with open(filepath, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(
                    ["Nucleus", "Atom_Numbers", "Assignment", "Chemical_Shift_ppm", "Integral", "Multiplicity",
                     "J_Couplings_Hz"])
                for s in self.h_signals: writer.writerow(
                    [s['nuc'], s['indices'], s['label'], f"{s['shift']:.3f}", s['integral'], s['mult'], s['j_str']])
                for s in self.c_signals: writer.writerow(
                    [s['nuc'], s['indices'], s['label'], f"{s['shift']:.3f}", s['integral'], s['mult'], s['j_str']])
            messagebox.showinfo("Success", f"Data saved to:\n{filepath}")

    def export_txt(self):
        if not self.h_signals and not self.c_signals: return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filepath:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write("=" * 88 + "\n")
                f.write(f"NMR SPECTRAL SUITE REPORT (1H: {self.freq_h:.0f} MHz | 13C: {self.freq_c:.1f} MHz)\n")
                f.write("=" * 88 + "\n\n")
                f.write(
                    f"{'Nuc':<6} | {'Atom #':<12} | {'Assignment':<16} | {'δ (ppm)':<8} | {'Int.':<6} | {'Mult.':<8} | {'J Couplings (Hz)'}\n")
                f.write("-" * 88 + "\n")
                for s in self.h_signals: f.write(
                    f"{s['nuc']:<6} | {s['indices']:<12} | {s['label']:<16} | {s['shift']:<8.3f} | {s['integral']:<6} | {s['mult']:<8} | {s['j_str']}\n")
                f.write("-" * 88 + "\n")
                for s in self.c_signals: f.write(
                    f"{s['nuc']:<6} | {s['indices']:<12} | {s['label']:<16} | {s['shift']:<8.3f} | {s['integral']:<6} | {s['mult']:<8} | {s['j_str']}\n")
                f.write("\n" + "=" * 88 + "\n")
                f.write("ACS FORMATTED CHARACTERIZATION STRING:\n")
                f.write("-" * 88 + "\n")
                f.write(self.txt_acs.get("1.0", tk.END).strip() + "\n")
                f.write("=" * 88 + "\n")
            messagebox.showinfo("Success", f"Report saved to:\n{filepath}")

    def copy_acs_to_clipboard(self):
        text = self.txt_acs.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear();
            self.clipboard_append(text);
            self.update()
            messagebox.showinfo("Clipboard", "Full ACS report copied to clipboard.")

    def _on_enter_pressed(self, event):
        active = self.notebook.index(self.notebook.select())
        if active == 1:
            if self.sub_notebook.index(self.sub_notebook.select()) == 0:
                self.merge_selected_hydrogens()
            else:
                self.merge_selected_carbons()
            return "break"
        elif active == 2:
            self.merge_selected_signals_plot(); return "break"
        elif active == 3:
            self.merge_selected_c_signals_plot(); return "break"

    def _on_delete_pressed(self, event):
        active = self.notebook.index(self.notebook.select())
        if active == 1:
            if self.sub_notebook.index(self.sub_notebook.select()) == 0:
                self.ungroup_selected_hydrogens()
            else:
                self.ungroup_selected_carbons()
            return "break"
        elif active == 2:
            self.ungroup_selected_signals_plot(); return "break"
        elif active == 3:
            self.ungroup_selected_c_signals_plot(); return "break"