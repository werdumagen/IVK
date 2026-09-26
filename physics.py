import collections
import numpy as np
from parsers import COVALENT_RADII, CPK_COLORS, get_2d_projection_coordinates


def spin_operators(n_spins):
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)

    Ix, Iy, Iz = [], [], []
    for i in range(n_spins):
        x_ops, y_ops, z_ops = [id2] * n_spins, [id2] * n_spins, [id2] * n_spins
        x_ops[i], y_ops[i], z_ops[i] = sx, sy, sz

        ix, iy, iz = x_ops[0], y_ops[0], z_ops[0]
        for j in range(1, n_spins):
            ix = np.kron(ix, x_ops[j])
            iy = np.kron(iy, y_ops[j])
            iz = np.kron(iz, z_ops[j])

        Ix.append(ix)
        Iy.append(iy)
        Iz.append(iz)

    return Ix, Iy, Iz


def find_spin_clusters(protons, couplings, j_cutoff=0.4, max_size=8):
    adj = collections.defaultdict(list)
    for i in protons:
        for j in protons:
            if i < j and abs(couplings.get((i, j), 0.0)) >= j_cutoff:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    for p in protons:
        if p not in visited:
            comp = []
            q = [p]
            visited.add(p)
            for node in q:
                comp.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            clusters.append(comp)

    final_clusters = []
    for comp in clusters:
        if len(comp) <= max_size:
            final_clusters.append(comp)
        else:
            sub = find_spin_clusters(comp, couplings, j_cutoff=j_cutoff * 1.8, max_size=max_size)
            final_clusters.extend(sub)

    return final_clusters


def simulate_grouped_h_molecule(shifts, couplings, groups, freq_mhz=600.0, lw_hz=0.8,
                                num_points=15000, decoupled_spin=None):
    all_shifts = list(shifts.values())
    ppm_min = min(all_shifts) - 0.5
    ppm_max = max(all_shifts) + 0.5
    ppm = np.linspace(ppm_min, ppm_max, num_points)
    spectrum = np.zeros_like(ppm)

    gamma = (lw_hz / freq_mhz) / 2.0

    # 1. Merged groups
    for g in groups:
        if len(g) > 1:
            avg_shift = float(np.mean([shifts[p] for p in g]))
            scale = 0.05 if (decoupled_spin is not None and decoupled_spin in g) else 1.0
            spectrum += scale * len(g) * (gamma / (np.pi * ((ppm - avg_shift) ** 2 + gamma ** 2)))

    # 2. Individual spins
    ind_protons = [g[0] for g in groups if len(g) == 1]
    clusters = find_spin_clusters(ind_protons, couplings)

    for comp in clusters:
        n_comp = len(comp)
        if n_comp == 1:
            p = comp[0]
            scale = 0.05 if (decoupled_spin is not None and p == decoupled_spin) else 1.0
            spectrum += scale * 1.0 * (gamma / (np.pi * ((ppm - shifts[p]) ** 2 + gamma ** 2)))
            continue

        sub_shifts = [shifts[p] for p in comp]
        sub_j = {}
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                p_i, p_j = comp[i], comp[j]
                if decoupled_spin is not None and (p_i == decoupled_spin or p_j == decoupled_spin):
                    val = 0.0
                else:
                    val = couplings.get((p_i, p_j), 0.0)
                if abs(val) > 1e-4:
                    sub_j[(i, j)] = val

        Ix, Iy, Iz = spin_operators(n_comp)
        Fx = sum(Ix)
        freqs_hz = np.array(sub_shifts) * freq_mhz
        dim = 2 ** n_comp
        H = np.zeros((dim, dim), dtype=complex)

        for i in range(n_comp):
            H += freqs_hz[i] * Iz[i]
        for (i, j), J_val in sub_j.items():
            H += J_val * (Ix[i] @ Ix[j] + Iy[i] @ Iy[j] + Iz[i] @ Iz[j])

        energies, evecs = np.linalg.eigh(H)
        Fx_eigen = evecs.conj().T @ Fx @ evecs
        intensities = np.abs(Fx_eigen) ** 2

        norm_factor = 2.0 ** (n_comp - 3)
        for a in range(dim):
            for b in range(a + 1, dim):
                I_ab = intensities[a, b] / norm_factor
                if I_ab > 1e-5:
                    v_ab_hz = np.abs(energies[b] - energies[a])
                    pos_ppm = v_ab_hz / freq_mhz
                    spectrum += I_ab * (gamma / (np.pi * ((ppm - pos_ppm) ** 2 + gamma ** 2)))

    return ppm, spectrum


def simulate_grouped_c13_spectrum(c_shifts, c_groups, freq_mhz=150.9, lw_hz=1.5, num_points=20000):
    all_shifts = list(c_shifts.values())
    ppm_min = min(all_shifts) - 10.0
    ppm_max = max(all_shifts) + 10.0
    ppm = np.linspace(ppm_min, ppm_max, num_points)
    spectrum = np.zeros_like(ppm)

    gamma = (lw_hz / freq_mhz) / 2.0
    for g in c_groups:
        if len(g) > 1:
            avg_shift = float(np.mean([c_shifts[c] for c in g]))
            spectrum += len(g) * (gamma / (np.pi * ((ppm - avg_shift) ** 2 + gamma ** 2)))
        else:
            c = g[0]
            spectrum += 1.0 * (gamma / (np.pi * ((ppm - c_shifts[c]) ** 2 + gamma ** 2)))

    return ppm, spectrum


def simulate_2d_hsqc_map(hsqc_corrs, h_range=(0, 6), c_range=(0, 160),
                         grid_h=280, grid_c=360, lw_h_ppm=0.035, lw_c_ppm=0.9):
    h_axis = np.linspace(h_range[0], h_range[1], grid_h)
    c_axis = np.linspace(c_range[0], c_range[1], grid_c)
    H_grid, C_grid = np.meshgrid(h_axis, c_axis)

    matrix_pos = np.zeros_like(H_grid)
    matrix_neg = np.zeros_like(H_grid)

    for peak in hsqc_corrs:
        h0 = peak['h_shift']
        c0 = peak['c_shift']
        phase = peak['phase']
        peak_shape = np.exp(-0.5 * (((H_grid - h0) / lw_h_ppm) ** 2 + ((C_grid - c0) / lw_c_ppm) ** 2))
        if phase > 0:
            matrix_pos += peak_shape
        else:
            matrix_neg += peak_shape

    return h_axis, c_axis, matrix_pos, matrix_neg


def simulate_2d_cosy_map(shifts_list, correlations, h_range=(0, 6), grid=320, lw_ppm=0.035):
    axis = np.linspace(h_range[0], h_range[1], grid)
    F1, F2 = np.meshgrid(axis, axis)
    matrix = np.zeros_like(F1)

    for s in shifts_list:
        matrix += 1.4 * np.exp(-0.5 * (((F1 - s) / lw_ppm) ** 2 + ((F2 - s) / lw_ppm) ** 2))

    for corr in correlations:
        s1 = corr['h1_shift']
        s2 = corr['h2_shift']
        w = min(1.0, (corr['j_val'] / 9.0) * 0.8)
        cp1 = w * np.exp(-0.5 * (((F1 - s1) / lw_ppm) ** 2 + ((F2 - s2) / lw_ppm) ** 2))
        cp2 = w * np.exp(-0.5 * (((F1 - s2) / lw_ppm) ** 2 + ((F2 - s1) / lw_ppm) ** 2))
        matrix += (cp1 + cp2)

    return axis, matrix


def render_2d_molecule_on_ax(ax, mol_atoms, title=""):
    """
    Renders an aligned 2D chemical structure with visible framing and crisp typography.
    """
    ax.clear()
    ax.set_facecolor('#ffffff')

    # Keep border frame visible
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#495057')
        spine.set_linewidth(1.3)
        spine.set_visible(True)

    if not mol_atoms:
        ax.text(0.5, 0.5, "No XYZ Data", ha='center', va='center', color='#888888', fontsize=8, fontweight='bold')
        return

    proj = get_2d_projection_coordinates(mol_atoms)
    keys = list(proj.keys())
    coords_3d = {k: mol_atoms[k]['xyz'] for k in keys}
    n = len(keys)

    # 1. Chemical bonds
    for i in range(n):
        for j in range(i + 1, n):
            k1, k2 = keys[i], keys[j]
            dist_3d = np.linalg.norm(coords_3d[k1] - coords_3d[k2])
            r1 = COVALENT_RADII.get(proj[k1]['elem'], 0.77)
            r2 = COVALENT_RADII.get(proj[k2]['elem'], 0.77)
            if 0.4 < dist_3d < (r1 + r2 + 0.45):
                p1 = proj[k1]['xy']
                p2 = proj[k2]['xy']
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#333333', lw=1.3, zorder=1)

    # 2. Atoms nodes & labels
    for k in keys:
        el = proj[k]['elem']
        xy = proj[k]['xy']
        bg = CPK_COLORS.get(el, '#E0E0E0')
        edge = '#222222'
        size = 130 if el != 'H' else 80

        ax.scatter(xy[0], xy[1], s=size, c=bg, edgecolors=edge, lw=0.9, zorder=2)
        label_text = f"{el}{k}" if el != 'C' else f"C{k}"
        text_color = '#ffffff' if el == 'C' else ('#111111' if el == 'H' else '#ffffff')
        ax.text(xy[0], xy[1], label_text, fontsize=6.5, ha='center', va='center',
                fontweight='bold', color=text_color, zorder=3)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3, color='#333333', fontweight='bold')