"""Validate returned readings and retain presentations and file audit in one CSV."""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from common import ROIS, save, selection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--returns', type=Path, required=True)
    ap.add_argument('--unified-labels', type=Path)
    args = ap.parse_args()
    master, probabilities = selection(args.unified_labels)
    assert len(master) == 1155 and master.uid2.is_unique
    expected = master.groupby('ms_id').uid2.apply(set).to_dict()
    candidates, audit = {}, []
    for path in sorted(args.returns.glob('*.npz')):
        match = re.match(r'(P2A_M\d+)', path.name)
        if not match:
            continue
        ms = match.group(1)
        reason, timestamp, labels, ids = '', '', None, []
        try:
            with np.load(path, allow_pickle=True) as data:
                ids = np.asarray(data['image_ids'] if 'image_ids' in data.files else data['gallery']).tolist()
                labels = np.asarray(data['labels'] if 'labels' in data.files else data['inference_edit']).astype(float)
                timestamp = str(data['timestamp'].item())
            valid = (labels.shape == (7, 4) and len(ids) == 7 and len(set(ids)) == 7
                     and set(ids) == expected.get(ms) and np.isfinite(labels).all()
                     and (labels >= 0).all() and (labels <= 4).all() and (labels == np.floor(labels)).all())
            if not valid:
                reason = 'invalid identifiers, shape or grades'
        except (ValueError, KeyError, OSError, TypeError) as exc:
            valid, reason = False, str(exc)
        record = dict(record_type='file_audit', file=path.name, ms_id=ms, timestamp=timestamp,
                      valid=bool(valid), all_zero=bool(valid and (labels == 0).all()), chosen=False, reason=reason)
        audit.append(record)
        if valid:
            candidates.setdefault(ms, []).append((timestamp, path.name, ids, labels.astype(int), record))
    if set(candidates) != set(expected):
        raise ValueError(f'Missing valid sequences: {sorted(set(expected) - set(candidates))}')
    records = []
    for ms, values in sorted(candidates.items()):
        timestamp, filename, ids, labels, record = max(values, key=lambda value: (value[0], value[1]))
        record['chosen'] = True
        for uid2, scores in zip(ids, labels):
            records.append(dict(uid2=uid2, saved_at=timestamp, source_file=filename,
                                **{f'new_{roi}': int(scores[i]) for i, roi in enumerate(ROIS)}))
    joined = master.merge(pd.DataFrame(records), on='uid2', how='outer', indicator=True, validate='one_to_one')
    assert joined._merge.eq('both').all()
    joined = joined.sort_values('read_order').drop(columns='_merge')
    joined = joined.merge(probabilities, on='uid', how='left', validate='many_to_one')
    joined['record_type'] = 'presentation'
    assert joined.loc[joined.read_type.eq('first_read')].uid.nunique() == 831
    assert joined.loc[joined.read_type.eq('first_read'), 'pi_uid'].notna().sum() == 800
    save(pd.concat([joined, pd.DataFrame(audit)], ignore_index=True), 'c1_read_results')


if __name__ == '__main__':
    main()
