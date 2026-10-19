"""Shared inputs and statistics for post-export Phase 2 analyses."""
from pathlib import Path
import contextlib
import io
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Result/Phase2/C_analysis'
ROIS = ['RT', 'LT', 'RB', 'LB']
B, SEED = 20000, 20261012


def save(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    print(f'{name}: {len(frame)} rows', flush=True)


def selection(unified_labels=None):
    """Reuse the selection notebook without running its export cells."""
    labels = Path(unified_labels or ROOT / 'CXR/unified_labels.csv').resolve()
    ns = {'display': lambda *args: None}
    notebook = json.loads((ROOT / 'Experiment/Phase2/A_selection/a1_reread_selection.ipynb').read_text(encoding='utf-8'))
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        # Keep master construction, but stop before writing selection artifacts.
        if 'for rel_path, table in outputs.items():' in source:
            source = source.replace('OUTPUT_DIR.mkdir(parents=True, exist_ok=True)', '')
            source = source.split('for rel_path, table in outputs.items():')[0]
        source = '\n'.join('ROOT = Path(' + repr(str(ROOT)) + ')' if line.startswith('ROOT = next(')
                           else 'SOURCE_CSV = Path(' + repr(str(labels)) + ')' if line.startswith('SOURCE_CSV = ')
                           else line for line in source.splitlines())
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f'selection_cell_{i}', 'exec'), ns)
    return ns['master'], ns['img_sel'][['uid', 'pi_uid']]


def presentations():
    frame = pd.read_csv(OUT / 'c1_read_results.csv')
    frame = frame.loc[frame.record_type.eq('presentation')].copy()
    for col in ['year', 'read_order', 'seq', 'seq2'] + [f'{prefix}_{roi}' for prefix in ['old', 'new'] for roi in ROIS]:
        frame[col] = frame[col].astype(int)
    assert len(frame) == 1155 and frame.uid2.is_unique
    return frame.sort_values('read_order')


def populations(frame):
    first = frame.loc[frame.read_type.eq('first_read')].copy()
    original = first.loc[first.pi_uid.notna()].copy()
    assert len(first) == 831 and first.uid.is_unique and len(original) == 800
    assert original.groupby('year').size().to_dict() == {2024: 500, 2026: 300}
    original['w'] = 1 / original.pi_uid
    first['w'] = 1.
    all_rows = frame.copy()
    all_rows['w'] = 1.
    return {'original800': original, 'unique831': first, 'presentation1155': all_rows}


def interval(point, boot):
    boot = np.asarray(boot)
    boot = boot[np.isfinite(boot)]
    q = np.quantile(boot, [.025, .05, .95, .975]) if len(boot) else [np.nan] * 4
    return dict(estimate=float(point), ci_low=q[0], ci_high=q[3],
                lower_one_sided=q[1], upper_one_sided=q[2], valid=len(boot))


def calculate(d, bootstrap=B, seed=SEED):
    """Original RB conditional boundary estimands; same draws and patient universe."""
    pats = d.groupby('pid', sort=True).year.first()
    pids = list(pats.index)
    rng = np.random.default_rng(seed)
    draws = np.zeros((bootstrap, len(pids)))
    for year in [2024, 2026]:
        ix = np.flatnonzero(pats.to_numpy() == year)
        draws[:, ix] = rng.multinomial(len(ix), np.repeat(1 / len(ix), len(ix)), size=bootstrap)
    point, boots, counts = {}, {}, []
    for year in [2024, 2026]:
        for k in range(4):
            for direction in ['up', 'down', 'net']:
                sub = d.loc[d.year.eq(year)].copy()
                if direction != 'net':
                    sub = sub.loc[sub.old_RB.eq(k if direction == 'up' else k + 1)]
                event = (sub.new_RB.ge(k + 1) if direction == 'up' else sub.new_RB.le(k)
                         if direction == 'down' else sub.new_RB.ge(k + 1).astype(int) - sub.old_RB.ge(k + 1).astype(int)).astype(float)
                sub['num'], sub['den'] = event * sub.w, sub.w
                mat = sub.groupby('pid')[['num', 'den']].sum().reindex(pids, fill_value=0).to_numpy()
                name = f'{direction}{k}{k+1}_{year}'
                point[name] = mat[:, 0].sum() / mat[:, 1].sum() if mat[:, 1].sum() else np.nan
                sums = draws @ mat
                with np.errstate(divide='ignore', invalid='ignore'):
                    boots[name] = sums[:, 0] / sums[:, 1]
                counts.append(dict(term=name, numerator=float(event.sum()), denominator=len(sub),
                                   weighted_num=float(mat[:, 0].sum()), weighted_den=float(mat[:, 1].sum()), patients=sub.pid.nunique()))
    specs = {
        'T': {'up34_2024': 1, 'down34_2024': -1},
        'U': {'up34_2026': 1, 'down34_2026': -1},
        'D23_2024': {'up23_2024': 1, 'down23_2024': -1},
        'D23_2026': {'up23_2026': 1, 'down23_2026': -1},
        'delta_year': {'T': 1, 'U': -1},
        'delta_cutpoint_2024': {'T': 1, 'D23_2024': -1},
        'delta_cutpoint_2026': {'U': 1, 'D23_2026': -1},
        'C_down': {'down34_2026': 1, 'down34_2024': -1},
        'net_year': {'net34_2024': 1, 'net34_2026': -1},
        'net_cutpoint_2024': {'net34_2024': 1, 'net23_2024': -1},
        'net_cutpoint_2026': {'net34_2026': 1, 'net23_2026': -1},
    }
    for name, terms in specs.items():
        point[name] = sum(point[t] * value for t, value in terms.items())
        boots[name] = sum(boots[t] * value for t, value in terms.items())
    return point, boots, counts
