import numpy as np


def determine_multiplicity(j_list):
    if not j_list:
        return "s", "-"

    j_vals = [round(abs(item[1]), 2) for item in j_list]
    k = len(j_vals)
    j_str = ", ".join([f"J({item[0]}) = {val:.2f}" for item, val in zip(j_list, j_vals)])

    if k == 1:
        mult = "d"
    elif k == 2:
        mult = "t" if abs(j_vals[0] - j_vals[1]) <= 0.6 else "dd"
    elif k == 3:
        if (max(j_vals) - min(j_vals)) <= 0.6:
            mult = "q"
        elif abs(j_vals[0] - j_vals[1]) <= 0.6:
            mult = "td"
        elif abs(j_vals[1] - j_vals[2]) <= 0.6:
            mult = "dt"
        else:
            mult = "ddd"
    elif k == 4:
        mult = "dddd"
    else:
        mult = "m"

    return mult, j_str


def analyze_grouped_h_signals(shifts, couplings, groups, j_cutoff=0.45):
    signals = []
    ind_protons = [g[0] for g in groups if len(g) == 1]

    for g in groups:
        if len(g) > 1:
            avg_shift = float(np.mean([shifts[p] for p in g]))
            lbl = f"t-Bu ({len(g)}H)" if len(g) == 9 else f"Group ({len(g)}H)"
            signals.append({
                'nuc': '1H',
                'indices': ", ".join(str(p) for p in sorted(g)),
                'label': lbl,
                'shift': avg_shift,
                'integral': f"{len(g)}H",
                'integral_num': len(g),
                'mult': 's',
                'j_str': '-',
                'raw_group': g
            })
        else:
            p = g[0]
            j_interactions = []
            for other_p in ind_protons:
                if other_p != p:
                    j_val = couplings.get((p, other_p), 0.0)
                    if abs(j_val) >= j_cutoff:
                        j_interactions.append((f"H{other_p}", abs(j_val)))

            j_interactions.sort(key=lambda x: x[1], reverse=True)
            mult, j_str = determine_multiplicity(j_interactions)

            signals.append({
                'nuc': '1H',
                'indices': str(p),
                'label': f"H{p}",
                'shift': shifts[p],
                'integral': "1H",
                'integral_num': 1,
                'mult': mult,
                'j_str': j_str,
                'raw_group': g
            })

    signals.sort(key=lambda s: s['shift'], reverse=True)
    return signals


def analyze_grouped_c_signals(c_shifts, c_groups):
    signals = []
    for g in c_groups:
        if len(g) > 1:
            avg_shift = float(np.mean([c_shifts[c] for c in g]))
            lbl = f"t-Bu ({len(g)}C)" if len(g) == 3 else f"Group ({len(g)}C)"
            signals.append({
                'nuc': '13C',
                'indices': ", ".join(str(c) for c in sorted(g)),
                'label': lbl,
                'shift': avg_shift,
                'integral': f"{len(g)}C",
                'integral_num': len(g),
                'mult': 's',
                'j_str': '-',
                'raw_group': g
            })
        else:
            c = g[0]
            signals.append({
                'nuc': '13C',
                'indices': str(c),
                'label': f"C{c}",
                'shift': c_shifts[c],
                'integral': "1C",
                'integral_num': 1,
                'mult': 's',
                'j_str': '-',
                'raw_group': g
            })
    signals.sort(key=lambda s: s['shift'], reverse=True)
    return signals


def build_acs_publication_block(h_signals, c_signals, hsqc_corrs, cosy_corrs, freq_h, freq_c):
    h_parts = []
    for s in h_signals:
        shift_str = f"{s['shift']:.2f}"
        if s['j_str'] != '-':
            j_vals = [t.split('=')[1].strip() for t in s['j_str'].split(',') if '=' in t]
            j_part = f", J = {', '.join(j_vals)} Hz"
        else:
            j_part = ""
        h_parts.append(f"{shift_str} ({s['mult']}{j_part}, {s['integral']}, {s['label']})")

    c_parts = []
    for s in c_signals:
        shift_str = f"{s['shift']:.2f}"
        int_str = f"{s['integral']}, " if s['integral_num'] > 1 else ""
        c_parts.append(f"{shift_str} ({int_str}{s['label']})")

    hsqc_parts = [f"[{p['c_label']}({p['c_shift']:.1f})–{p['h_label']}({p['h_shift']:.2f}), {p['group_type']}]" for p in
                  hsqc_corrs]
    cosy_parts = [f"[{p['h1_label']}–{p['h2_label']}: {p['j_val']:.1f} Hz]" for p in cosy_corrs[:12]]

    acs_text = (
            f"1H NMR ({freq_h:.0f} MHz): δ " + ", ".join(h_parts) + ".\n\n" +
            f"13C NMR ({freq_c:.1f} MHz): δ " + ", ".join(c_parts) + ".\n\n" +
            f"1H–13C HSQC: " + (", ".join(hsqc_parts) if hsqc_parts else "N/A") + ".\n\n" +
            f"1H–1H COSY: " + (", ".join(cosy_parts) if cosy_parts else "None") + "."
    )
    return acs_text