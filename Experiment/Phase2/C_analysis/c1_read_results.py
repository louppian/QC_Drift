from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Result/Phase2/C_analysis'
OUT.mkdir(parents=True, exist_ok=True)
ap=argparse.ArgumentParser(description='Validate returned reader NPZ files and join all 1155 presentations.')
ap.add_argument('--returns', type=Path, required=True)
ap.add_argument('--master',type=Path,default=ROOT/'Result/Phase2/A_selection/a1_reread_master.csv')
args=ap.parse_args()
DATA=args.returns
ROIS=['RT','LT','RB','LB']
master=pd.read_csv(args.master)
assert len(master)==1155 and master.uid2.is_unique
expected = master.groupby("ms_id").uid2.apply(set).to_dict()
candidates, audit = {}, []
for path in sorted(DATA.glob("*.npz")):
    ms = re.match(r"(P2A_M\d+)", path.name).group(1)
    with np.load(path, allow_pickle=True) as data:
        ids = np.asarray(data["image_ids"] if "image_ids" in data.files else data["gallery"]).tolist()
        key = "labels" if "labels" in data.files else "inference_edit"
        labels = np.asarray(data[key])
        timestamp = str(data["timestamp"].item())
    valid = labels.shape == (7, 4) and len(ids) == 7 and set(ids) == expected.get(ms)
    if valid:
        labels = labels.astype(float)
        valid = bool(np.isfinite(labels).all() and ((labels >= 0) & (labels <= 4)).all()
                     and (labels == np.floor(labels)).all())
    record = dict(file=path.name, ms_id=ms, timestamp=timestamp, valid=valid,
                  all_zero=bool(valid and (labels == 0).all()), chosen=False)
    audit.append(record)
    if valid:
        candidates.setdefault(ms, []).append((timestamp, path.name, ids, labels.astype(int), record))
records = []
conflicts = []
for ms, values in sorted(candidates.items()):
    values.sort(key=lambda x: (x[0], x[1]))
    chosen = values[-1]
    chosen[4]["chosen"] = True
    if len(values) > 1:
        conflicts.append(dict(ms_id=ms, files=[x[1] for x in values],
                              scores_differ=any(not np.array_equal(x[3], chosen[3]) for x in values)))
    for uid2, scores in zip(chosen[2], chosen[3]):
        records.append(dict(uid2=uid2, saved_at=chosen[0], source_file=chosen[1],
                            **{f"new_{r}": int(scores[j]) for j, r in enumerate(ROIS)}))
new = pd.DataFrame(records)
joined = master.merge(new, on="uid2", how="outer", indicator=True, validate="one_to_one")
assert joined._merge.eq("both").all(), joined._merge.value_counts()
joined = joined.sort_values("read_order").drop(columns="_merge")
first = joined[joined.read_type.eq("first_read")].copy()
assert first.uid.is_unique and len(first) == 831
assert joined.uid2.is_unique and len(joined)==1155
pd.DataFrame(audit).to_csv(OUT/'c1_file_audit.csv',index=False,encoding='utf-8-sig')
joined.to_csv(OUT/'c1_read_results.csv',index=False,encoding='utf-8-sig')
print(f'Validated {len(joined)} presentations from {len(candidates)} sequences; latest valid timestamp/name selected.')
