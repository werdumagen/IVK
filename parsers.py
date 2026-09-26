import numpy as np

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'CL': 1.02, 'BR': 1.20, 'I': 1.39, 'SI': 1.11
}

CPK_COLORS = {
    'H': '#FFFFFF',
    'C': '#333333',
    'N': '#2B5C8F',
    'O': '#C82828',
    'F': '#7AC235',
    'CL': '#1FB838',
    'BR': '#8A251E',
    'S': '#D9A526',
    'P': '#E66A23',
    'SI': '#9B7BB8'
}


def parse_shieldings(text_data):
    """Parses isotropic nuclear shieldings (sigma_iso) for 1H and 13C."""
    shieldings = {}
    for line in text_data.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            idx = int(parts[0])
            elem = parts[1].upper()
            try:
                iso = float(parts[2])
                shieldings[idx] = {'elem': elem, 'iso': iso}
            except ValueError:
                continue
    return shieldings


def parse_couplings(text_data):
    """Parses block matrices of scalar spin-spin coupling constants (J in Hz)."""
    couplings = {}
    current_cols = []

    for line in text_data.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()

        if len(tokens) >= 2 and len(tokens) % 2 == 0:
            if all(tokens[i].isdigit() and not tokens[i + 1].replace('.', '').replace('-', '').isdigit()
                   for i in range(0, len(tokens), 2)):
                current_cols = [int(tokens[i]) for i in range(0, len(tokens), 2)]
                continue

        if current_cols and len(tokens) >= 2 + len(current_cols) and tokens[0].isdigit():
            row_idx = int(tokens[0])
            values = tokens[2:2 + len(current_cols)]
            for col_idx, val_str in zip(current_cols, values):
                try:
                    j_val = float(val_str)
                    couplings[(row_idx, col_idx)] = j_val
                    couplings[(col_idx, row_idx)] = j_val
                except ValueError:
                    pass
    return couplings


def parse_xyz_coordinates(text_data):
    """Parses Cartesian coordinates of the molecule (XYZ / ORCA format)."""
    atoms = {}
    lines = text_data.strip().splitlines()
    auto_idx = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 1 and parts[0].isdigit():
            continue

        try:
            if len(parts) >= 5 and parts[0].isdigit():
                idx = int(parts[0])
                elem = parts[1].upper()
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                atoms[idx] = {'elem': elem, 'xyz': np.array([x, y, z])}
            elif len(parts) >= 4 and not parts[0].replace('.', '').replace('-', '').isdigit():
                elem = parts[0].upper()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms[auto_idx] = {'elem': elem, 'xyz': np.array([x, y, z])}
                auto_idx += 1
        except ValueError:
            continue

    return atoms


def compute_hsqc_correlations(mol_atoms, h_shifts, c_shifts, max_bond_dist=1.30):
    """Computes direct 1J(C-H) correlations based on bond lengths (< 1.30 A)."""
    if not mol_atoms:
        return []

    correlations = []
    carbon_indices = [idx for idx, data in mol_atoms.items() if data['elem'] == 'C']
    hydrogen_indices = [idx for idx, data in mol_atoms.items() if data['elem'] == 'H']

    for c_idx in carbon_indices:
        c_pos = mol_atoms[c_idx]['xyz']
        attached_h = []
        for h_idx in hydrogen_indices:
            h_pos = mol_atoms[h_idx]['xyz']
            dist = np.linalg.norm(c_pos - h_pos)
            if dist < max_bond_dist:
                attached_h.append(h_idx)

        if attached_h:
            group_type = {1: 'CH', 2: 'CH2', 3: 'CH3'}.get(len(attached_h), 'CH')
            c_shift = c_shifts.get(c_idx, None)

            for h_idx in attached_h:
                h_shift = h_shifts.get(h_idx, None)
                if c_shift is not None and h_shift is not None:
                    correlations.append({
                        'c_idx': c_idx,
                        'c_label': f"C{c_idx}",
                        'h_idx': h_idx,
                        'h_label': f"H{h_idx}",
                        'group_type': group_type,
                        'phase': -1 if group_type == 'CH2' else 1,
                        'c_shift': float(c_shift),
                        'h_shift': float(h_shift)
                    })

    correlations.sort(key=lambda item: item['c_shift'], reverse=True)
    return correlations


def compute_cosy_correlations(couplings, h_shifts, j_min=0.5):
    """Computes 1H-1H scalar couplings (|J| >= j_min)."""
    corrs = []
    protons = sorted(list(h_shifts.keys()))
    n = len(protons)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = protons[i], protons[j]
            j_val = abs(couplings.get((p1, p2), 0.0))
            if j_val >= j_min:
                corrs.append({
                    'h1_idx': p1,
                    'h1_label': f"H{p1}",
                    'h2_idx': p2,
                    'h2_label': f"H{p2}",
                    'j_val': j_val,
                    'h1_shift': h_shifts[p1],
                    'h2_shift': h_shifts[p2]
                })
    corrs.sort(key=lambda x: x['j_val'], reverse=True)
    return corrs


def get_2d_projection_coordinates(mol_atoms):
    """
    Computes an optimal, aligned 2D projection of a 3D molecule using PCA.
    Flattens the structure along the plane of maximum variance.
    """
    if not mol_atoms:
        return {}

    keys = list(mol_atoms.keys())
    coords_3d = np.array([mol_atoms[k]['xyz'] for k in keys])

    # Center coordinates
    center = coords_3d.mean(axis=0)
    centered = coords_3d - center

    # Singular Value Decomposition (PCA)
    _, _, vt = np.linalg.svd(centered)

    # Project onto the top two principal components (flattest view)
    coords_2d = centered @ vt[:2].T

    # Scale coordinates to standard view window (-1 to 1)
    span = np.ptp(coords_2d, axis=0)
    max_span = max(span.max(), 1e-4)
    norm_coords = (coords_2d - coords_2d.mean(axis=0)) / (max_span * 0.55)

    proj_map = {}
    for idx, key in enumerate(keys):
        proj_map[key] = {
            'elem': mol_atoms[key]['elem'],
            'xy': norm_coords[idx]
        }
    return proj_map