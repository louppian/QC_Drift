"""Compare historical AI, original reader grades and current rereads; no sequence-context analysis."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from common import B, SEED, ROOT, ROIS, interval, populations, presentations, save

def boundary_rows(a, patients, draw, roi="RB"):
    """Paired patient bootstrap for AI disagreements at each grade boundary."""
    rows, rates = [], {}

    def rate(mask, event):
        sub = a.loc[mask]
        mat = pd.DataFrame({'pid': sub.pid, 'num': event.loc[mask].astype(int),
                            'den': 1}).groupby('pid')[['num', 'den']].sum().reindex(patients, fill_value=0).to_numpy()
        totals = draw @ mat
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = totals[:, 0] / totals[:, 1]
        point = mat[:, 0].sum() / len(sub) if len(sub) else np.nan
        return dict(numerator=int(event.loc[mask].sum()), denominator=len(sub),
                    patients=sub.pid.nunique(), **interval(point, boot)), boot

    metadata = dict(analysis='original800', roi=roi, year=2026, bootstrap=B,
                    seed=SEED, cluster='patient', weighting='equal image')
    for k in range(4):
        for direction, mask, event in [
            ('old_above_ai', a[f'old_{roi}'].gt(k) & a[f'ai_{roi}'].le(k), a[f'new_{roi}'].le(k)),
            ('old_below_ai', a[f'old_{roi}'].le(k) & a[f'ai_{roi}'].gt(k), a[f'new_{roi}'].gt(k)),
        ]:
            for outcome, values in [('changed_to_ai_side', event), ('stayed_on_old_side', ~event)]:
                stats, boot = rate(mask, values)
                rows.append(dict(record_type='ai_boundary', boundary=f'{k}-{k+1}',
                                 direction=direction, metric=outcome, **metadata, **stats))
                if outcome == 'changed_to_ai_side':
                    rates[k, direction] = (stats, boot)
    for direction in ['old_above_ai', 'old_below_ai']:
        high, high_boot = rates[3, direction]
        for k in range(3):
            low, low_boot = rates[k, direction]
            # Same patient draws preserve dependence, including images spanning boundaries.
            rows.append(dict(record_type='ai_boundary_contrast', boundary='3-4',
                             reference_boundary=f'{k}-{k+1}', direction=direction,
                             metric='changed_to_ai_side_34_minus_reference', **metadata,
                             numerator=high['numerator'], denominator=high['denominator'],
                             reference_numerator=low['numerator'], reference_denominator=low['denominator'],
                             **interval(high['estimate'] - low['estimate'], high_boot - low_boot)))
    stats, _ = rate(a[f'ai_{roi}'].eq(3) & a[f'old_{roi}'].eq(4), a[f'new_{roi}'].eq(3))
    rows.append(dict(record_type='ai_transition_rate', **{f'ai_{roi}': 3, f'old_{roi}': 4, f'new_{roi}': 3},
                     metric='new3_given_ai3_old4', **metadata, **stats))
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--historical-root', type=Path, required=True)
    ap.add_argument('--unified-labels', type=Path, default=ROOT / 'CXR/unified_labels.csv')
    ap.add_argument("--roi", choices=ROIS, default="RB")
    args = ap.parse_args()
    roi = args.roi
    suffix = "" if roi == "RB" else "_" + roi
    DATA = args.historical_root
    f = populations(presentations())['original800']

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    idx = {}
    for folder in sorted((DATA / 'results' / 'Jung Soo Kim').glob('*/*_images')):
        npz = folder.with_name(folder.name[:-7] + '.npz')
        if not npz.exists() or 'old' in folder.name.lower():
            continue
        pngs = sorted(folder.glob('*.png'))
        for png in pngs:
            idx.setdefault(sha(png), []).append((str(npz), int(png.stem.split('_')[0]) - 1))
    u = pd.read_csv(args.unified_labels).set_index('uid')
    records = []
    cache = {}
    for pid, g in f[f.year.eq(2026)].groupby('pid'):
        ug = u.loc[g.uid].sort_values('seq')
        rel = ug.iloc[0].image_path.split('/2026 CXR/')[1]
        folder = DATA / 'Dataset' / Path(rel).parent.parent
        result = pd.read_csv(folder / 'result.txt')
        full = u[u.patient_id.eq(pid)].sort_values('seq')
        links = []
        for uid, r in full.iterrows():
            png = DATA / 'Dataset' / r.image_path.split('/2026 CXR/')[1]
            candidates = idx.get(sha(png), [])
            expected = r[ROIS].to_numpy(dtype=int)
            res = result[result.seq.eq(r.seq)].iloc[0]
            assert np.array_equal(expected, res[[f'user_edit_{q}' for q in ROIS]].to_numpy(dtype=int))
            good = []
            for path, k in candidates:
                if path not in cache:
                    with np.load(path, allow_pickle=True) as z:
                        cache[path] = (z['inference'].astype(int)[:, [0, 2, 1, 3]], z['inference_edit'].astype(int))
                ai, ed = cache[path]
                if k < len(ed) and np.array_equal(ed[k], expected) and np.array_equal(ai[k], res[[f'inference_{q}' for q in ROIS]].to_numpy(dtype=int)):
                    good.append((path, k))
            assert good, (pid, uid, 'no validated source')
            links.append((uid, good))
        common = set((x[0] for x in links[0][1]))
        for _, a in links:
            common &= set((x[0] for x in a))
        assert len(common) == 1, (pid, 'ambiguous sequence source', len(common))
        path = next(iter(common))
        ai, ed = cache[path]
        indices = []
        for uid, cands in links:
            ks = [k for p, k in cands if p == path]
            assert len(ks) == 1
            k = ks[0]
            indices.append(k)
            records.append(dict(uid=uid, **{f'ai_{roi}': int(ai[k, ROIS.index(roi)])}, source_npz=path, source_index=k))
        assert indices == sorted(indices) and len(indices) == len(set(indices))
    a = f[f.year.eq(2026)].merge(pd.DataFrame(records), on='uid', validate='one_to_one')
    assert len(a) == 300 and a[f'ai_{roi}'].between(0, 4).all()
    a['old_ai_distance'] = (a[f'old_{roi}'] - a[f'ai_{roi}']).abs()
    a['new_ai_distance'] = (a[f'new_{roi}'] - a[f'ai_{roi}']).abs()
    a['distance_change'] = a.new_ai_distance - a.old_ai_distance
    assert a.w.eq(1).all()
    patients = sorted(a.pid.unique())
    rng = np.random.default_rng(SEED)
    draw = rng.multinomial(len(patients), np.ones(len(patients)) / len(patients), B)

    def stat(d, value, mask=None):
        s = d.copy() if mask is None else d.loc[mask].copy()
        vals = pd.Series(value, index=d.index).reindex(s.index).astype(float)
        mat = pd.DataFrame({'pid': s.pid, 'num': vals, 'den': 1.0}).groupby('pid')[['num', 'den']].sum().reindex(patients, fill_value=0).to_numpy()
        sums = draw @ mat
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = sums[:, 0] / sums[:, 1]
        return dict(numerator=float(vals.sum()), denominator=len(s), patients=s.pid.nunique(), **interval(vals.mean(), boot))
    airesults = {}
    old4 = a[f'old_{roi}'].eq(4)
    down = old4 & a[f'new_{roi}'].le(3)
    changed = a[f'old_{roi}'].ne(a[f'new_{roi}'])
    for label, mask in [('ai4', a[f'ai_{roi}'].eq(4)), ('ai_le3', a[f'ai_{roi}'].le(3))]:
        airesults['down_share_' + label] = stat(a, mask, down)
        airesults['down_rate_' + label] = stat(a, a[f'new_{roi}'].le(3), old4 & mask)
    for label, val in [('old_AI_ACC', a[f'old_{roi}'].eq(a[f'ai_{roi}'])), ('new_AI_ACC', a[f'new_{roi}'].eq(a[f'ai_{roi}'])), ('ACC_change', a[f'new_{roi}'].eq(a[f'ai_{roi}']).astype(int) - a[f'old_{roi}'].eq(a[f'ai_{roi}']).astype(int)), ('old_AI_MAE', a.old_ai_distance), ('new_AI_MAE', a.new_ai_distance), ('MAE_change', a.distance_change)]:
        airesults[label] = stat(a, val)
    for label, event in [('away', a.distance_change.gt(0)), ('toward', a.distance_change.lt(0)), ('equal_distance', a.distance_change.eq(0))]:
        airesults['changed_' + label] = stat(a, event, changed)
    airesults['change_away_minus_toward'] = stat(a, a.distance_change.gt(0).astype(int) - a.distance_change.lt(0).astype(int), changed)
    airesults['down_ai_by_grade'] = a.loc[down].groupby([f'ai_{roi}', f'old_{roi}', f'new_{roi}']).size().rename('n').reset_index().to_dict('records')
    airesults['unchanged'] = int((~changed).sum())
    rows = [dict(record_type='statistic', metric=k, **v, bootstrap=B, seed=SEED, cluster='patient') for k, v in airesults.items() if isinstance(v, dict)]
    rows += [dict(record_type='ai_transition', **r) for r in a.groupby([f'ai_{roi}', f'old_{roi}', f'new_{roi}']).size().rename('n').reset_index().to_dict('records')]
    rows += [dict(record_type='down_ai_by_grade', **r) for r in airesults['down_ai_by_grade']]
    rows.append(dict(record_type='count', metric='unchanged', n=airesults['unchanged']))
    rows += [dict(record_type='observation', **r) for r in a[['uid', 'pid', f'ai_{roi}', f'old_{roi}', f'new_{roi}', 'source_npz', 'source_index']].to_dict('records')]
    # Labels and AI predictions were validated against these source arrays above.
    for path in sorted(a.source_npz.unique()):
        rows.append(dict(record_type='source', source_npz=path, sha256=sha(Path(path))))
    rows += boundary_rows(a, patients, draw, roi)
    save(rows, 'c5_ai_linkage' + suffix)
if __name__ == '__main__':
    main()
