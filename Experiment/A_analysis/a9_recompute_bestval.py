# -*- coding: utf-8 -*-
"""E0 best-val replay. Native ROI and 0.15 validation split restored from the original source."""
from pathlib import Path
from functools import partial
from typing import Dict
import sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from timm.models.vision_transformer import VisionTransformer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Experiment.A_analysis import common as A
B = A.B
ROI = ["RT", "RB", "LT", "LB"]
DISP = ["RT", "LT", "RB", "LB"]
R,C,K = 4,5,7
VAL_FRAC = 0.15
DEVICE = B.DEVICE
REL_BOXES = torch.tensor([[0,0,.5,.5],[.5,0,1,.5],[0,.5,.5,1],[.5,.5,1,1]])
E0 = str(Path(B._REPO)/"checkpoint/E0")
SEEDS = [1,2,42]
cache = None
rng = np.random.default_rng(0)

def vit_base_patch16_512(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

def _adapt_checkpoint_patch_embed(state_dict: Dict, model: VisionTransformer, mode: str = "mean"):
    key = "patch_embed.proj.weight"
    if key not in state_dict:
        return state_dict
    w_ckpt = state_dict[key]
    w_mod = model.patch_embed.proj.weight
    if w_ckpt.shape == w_mod.shape:
        return state_dict
    Cin_ckpt, Cin_mod = w_ckpt.shape[1], w_mod.shape[1]
    if Cin_ckpt == 3 and Cin_mod == 1:
        state_dict[key] = w_ckpt.mean(1, keepdim=True) if mode == "mean" else w_ckpt[:, :1]
    elif Cin_ckpt == 1 and Cin_mod == 3:
        state_dict[key] = w_ckpt.repeat(1, 3, 1, 1)
    return state_dict

def _resize_pos_embed(state_dict: Dict, model: VisionTransformer, verbose: bool = False):
    if "pos_embed" not in state_dict:
        return state_dict
    pe_ckpt = state_dict["pos_embed"]
    pe_mod = model.pos_embed
    if pe_ckpt.shape == pe_mod.shape:
        return state_dict
    cls_ckpt = pe_ckpt[:, :1]
    grid_ckpt = pe_ckpt[:, 1:]
    Nold = grid_ckpt.shape[1]
    gs_old = int(Nold**0.5)
    grid_ckpt = grid_ckpt.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    gs_new = model.patch_embed.grid_size[0]
    grid_new = F.interpolate(grid_ckpt, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
    grid_new = grid_new.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    state_dict["pos_embed"] = torch.cat([cls_ckpt, grid_new], dim=1)
    return state_dict

def load_mae_ckpt_to_512(
    path: str,
    num_classes: int = 4,
    in_chans: int = 1,
    drop_path_rate: float = 0.1,
    global_pool: str = "avg",
    patch_mode: str = "mean",
    verbose: bool = False,
) -> VisionTransformer:
    model = vit_base_patch16_512(
        num_classes=num_classes,
        in_chans=in_chans,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = _adapt_checkpoint_patch_embed(state, model, mode=patch_mode)
    state = _resize_pos_embed(state, model, verbose=verbose)

    if "head.weight" in state:
        if state["head.weight"].shape != model.head.weight.shape:
            state.pop("head.weight", None)
            state.pop("head.bias", None)
            if verbose:
                print(f"Removed head.weight due to shape mismatch.")

    model.load_state_dict(state, strict=False)
    return model

def split_lungs_to_four(mask_bin, min_area=1000):
    """폐 마스크 → 좌/우 폐 각각 상/하 반 → 4 box (정규화 y0,x0,y1,x1), 순서 [RT,RB,LT,LB]. 실패 시 None."""
    comps = [p for p in regionprops(label((mask_bin > 0).astype(np.uint8))) if p.area >= min_area]
    if len(comps) < 2:
        return None
    left, right = min(comps, key=lambda p: p.centroid[1]), max(comps, key=lambda p: p.centroid[1])
    H, W = mask_bin.shape
    out = []
    for reg in (left, right):
        y0, x0, y1, x1 = reg.bbox
        halves = np.linspace(y0, y1, 3)
        for i in range(2):
            ys, ye = int(round(halves[i])), int(round(halves[i + 1]))
            out.append((ys / H, x0 / W, ye / H, x1 / W))
    return out                                    # [RT, RB, LT, LB]

class InhaUHMaskDataset(Dataset):
    def __init__(self, df, cache2026, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cache = cache2026
        self.mode = mode
        self.pat_col = f"pattern{K}"
        self.photo = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.56], [0.17])])
        self.rot = 10 if mode == "train" else 0

    def __len__(self): return len(self.df)

    def _load(self, row):
        if int(row["year"]) == 2024:
            img = Image.open(B._resolve_img(row["uid"])).convert("L")
            mask = np.array(Image.open(B._resolve_mask(row["uid"])).convert("L"))
        else:
            img_u8, mask_u8 = self.cache[A.legacy_raw_path(row)]
            img = Image.fromarray(img_u8); mask = mask_u8
        return img, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, mask_np = self._load(row)
        mask = Image.fromarray(mask_np)
        if img.size != (512, 512): img = img.resize((512, 512), Image.BILINEAR)
        if mask.size != (512, 512): mask = mask.resize((512, 512), Image.NEAREST)
        if self.mode == "train" and self.rot > 0:
            ang = random.uniform(-self.rot, self.rot)
            img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, ang, interpolation=TF.InterpolationMode.NEAREST)
        mask_np = (np.array(mask) > 0).astype(np.float32)

        coords = split_lungs_to_four(mask_np)
        if coords is None:
            coords = REL_BOXES.tolist()
        rel = torch.tensor(coords, dtype=torch.float32)
        roi_masks = np.zeros((R, 512, 512), dtype=np.float32)
        for i in range(R):
            y0, x0, y1, x1 = coords[i]
            y0, y1 = int(y0 * 512), int(y1 * 512); x0, x1 = int(x0 * 512), int(x1 * 512)
            roi_masks[i, y0:y1, x0:x1] = mask_np[y0:y1, x0:x1]
        roi_masks = torch.from_numpy(roi_masks)

        x = self.photo(img)
        if torch.isnan(x).any(): x = torch.zeros_like(x)
        y = torch.tensor([int(row[c]) for c in ROI], dtype=torch.long)
        pat = torch.tensor(int(row[self.pat_col]), dtype=torch.long)
        return x, y, rel, roi_masks, pat, idx

def gen_patterns(df, seed=42):
    Xtr = df.loc[df.split == "train", ROI].to_numpy("float32")
    km = KMeans(n_clusters=K, random_state=seed, n_init=20).fit(Xtr)
    sev = sorted(range(K), key=lambda k: Xtr[km.labels_ == k].sum(1).mean() if (km.labels_ == k).any() else 1e9)
    remap = {old: new for new, old in enumerate(sev)}
    df[f"pattern{K}"] = [remap[c] for c in km.predict(df[ROI].to_numpy("float32"))]
    return df

def compute_priors(df):
    EPS = 1e-7
    tv = df[df.split == "train"]
    labels = tv[ROI].to_numpy(np.int32); pat = tv[f"pattern{K}"].to_numpy(np.int32)
    pi_g = np.zeros((R, R, C, C), np.float32)
    for i in range(R):
        for j in range(R):
            for d in range(C):
                md = labels[:, j] == d; nd = md.sum()
                cnt = np.bincount(labels[md, i], minlength=C).astype(np.float32) if nd else np.zeros(C)
                pi_g[i, j, :, d] = (cnt + EPS) / (nd + EPS * C)
    pi_p = np.zeros((K, R, R, C, C), np.float32)
    for k in range(K):
        lk = labels[pat == k]
        if len(lk) < 10: pi_p[k] = pi_g; continue
        for i in range(R):
            for j in range(R):
                for d in range(C):
                    md = lk[:, j] == d; nd = md.sum()
                    if nd == 0: pi_p[k, i, j, :, d] = pi_g[i, j, :, d]
                    else:
                        cnt = np.bincount(lk[md, i], minlength=C).astype(np.float32)
                        pi_p[k, i, j, :, d] = (cnt + EPS) / (nd + EPS * C)
    return torch.from_numpy(pi_g), torch.from_numpy(pi_p)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P, Y = [], []
    for imgs, lab, rel, roi_masks, pat, _ in loader:
        imgs, rel, roi_masks = imgs.to(DEVICE), rel.to(DEVICE), roi_masks.to(DEVICE)
        out = model(imgs, rel, masks=roi_masks)
        P.append(out["logits_s3"].argmax(-1).cpu()); Y.append(lab)
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    acc = (P == Y).mean(); mae = np.abs(P - Y).mean(); bias = (P - Y).mean()
    per = {ROI[i]: ((P[:, i] == Y[:, i]).mean(), np.abs(P[:, i] - Y[:, i]).mean(), (P[:, i] - Y[:, i]).mean())
           for i in range(R)}
    return acc, mae, bias, per, P, Y

def make_split(df, mode, seed):
    """mode 에 따라 df['split'] 을 train / val / test 로 채운다.

    val 은 **학습 코호트 안에서 환자 단위로** 뗀다 (모델 선택 전용).
    test 는 절대 모델 선택에 쓰지 않는다 — bias 주장의 전제.
    """
    rng = np.random.default_rng(seed)
    if mode == "2026to2024":
        train_year, test_year = 2026, 2024
    elif mode == "2024to2026":
        train_year, test_year = 2024, 2026
    elif mode in ("indomain2024", "indomain2026"):
        train_year = test_year = int(mode[-4:])
    else:
        raise ValueError(mode)

    df["split"] = None
    if train_year != test_year:
        df.loc[df.year == test_year, "split"] = "test"
        pool = np.asarray(df.loc[df.year == train_year, "patient"].unique(), dtype=object)
        rng.shuffle(pool)
        n_val = max(2, int(round(len(pool) * VAL_FRAC)))
        val_pat = set(pool[:n_val])
        df.loc[(df.year == train_year) & (df.patient.isin(val_pat)), "split"] = "val"
        df.loc[(df.year == train_year) & (~df.patient.isin(val_pat)), "split"] = "train"
        df = df[df.split.notna()].copy()
    else:
        sub = df[df.year == train_year].copy()
        pool = np.asarray(sub.patient.unique(), dtype=object); rng.shuffle(pool)
        n_test = max(2, int(round(len(pool) * 0.20)))
        n_val  = max(2, int(round(len(pool) * VAL_FRAC)))
        test_pat = set(pool[:n_test]); val_pat = set(pool[n_test:n_test + n_val])
        sub["split"] = np.where(sub.patient.isin(test_pat), "test",
                        np.where(sub.patient.isin(val_pat), "val", "train"))
        df = sub
    return df, train_year, test_year

@torch.no_grad()
def eval_arm(mode, seed):
    df = A.legacy_manifest()
    df, TY, EY = make_split(df, mode, seed)
    df = gen_patterns(df, seed)
    pi_g, pi_p = compute_priors(df)
    vit = load_mae_ckpt_to_512(str(B.MRM_W), num_classes=B.C, in_chans=1, drop_path_rate=0.1, verbose=False)
    model = B.BrixiaViT512Dynamic(vit, num_regions=B.R, num_classes=B.C, num_patterns=B.K,
                                  proj_dim=B.PROJ_DIM, pi_global=pi_g, pi_patterns=pi_p,
                                  gnn_num_heads=4, gnn_dropout=0.1).to(B.DEVICE)
    ck = torch.load(f"{E0}/{mode}_s{seed}/best_val_model.pth", map_location=B.DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    from torch.utils.data import DataLoader
    te = df[df.split == "test"]
    loader = DataLoader(InhaUHMaskDataset(te, cache, "eval"), batch_size=B.BATCH_SIZE, num_workers=0)
    acc, mae, bias, per, P, Y = evaluate(model, loader)
    del model, vit
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return P, Y, te.patient.to_numpy().astype(str)

def main():
    global cache, rng
    rng = np.random.default_rng(0)
    cache = A.build_full_2026_cache()
    res = {}
    for mode in ["2024to2026", "2026to2024"]:
        Ps, Ys, PTs = [], [], []
        for s in SEEDS:
            print(f"[eval] {mode} s{s}")
            P, Y, PT = eval_arm(mode, s)
            Ps.append(P); Ys.append(Y); PTs.append(PT)
        res[mode] = (np.vstack(Ps), np.vstack(Ys), np.concatenate(PTs))

    fP, fY, fPT = res["2024to2026"]   # fwd: test 2026
    rP, rY, rPT = res["2026to2024"]   # rev: test 2024
    rng = np.random.default_rng(0)

    def pat_means(P, Y, PT, col):
        b = (P[:, col] - Y[:, col]).astype(float) if col is not None else (P - Y).astype(float).mean(1)
        u = np.unique(PT); return u, np.array([b[PT == p].mean() for p in u])

    def boot(u, pb, n=5000):
        return pb[rng.integers(0, len(u), size=(n, len(u)))].mean(1)

    print("\n=== best_val 재산출 (E0 재추론, 3-seed) ===")
    print(f"{'ROI':>7} {'fwd':>8} {'rev':>8} {'δ':>8} {'γ':>8} {'δ 95%CI':>18} {'δ Bonf99%':>18}")
    rows = []
    for name in ["overall"] + DISP:
        col = None if name == "overall" else ROI.index(name)
        uf, pf = pat_means(fP, fY, fPT, col); ur, pr = pat_means(rP, rY, rPT, col)
        fwd, rev = pf.mean(), pr.mean(); delta, gamma = (rev - fwd) / 2, (rev + fwd) / 2
        bd = (boot(ur, pr) - boot(uf, pf)) / 2
        lo, hi = np.percentile(bd, [2.5, 97.5]); loB, hiB = np.percentile(bd, [0.5, 99.5])
        rows.append(dict(roi=name, fwd=fwd, rev=rev, delta=delta, gamma=gamma, ci_lo=lo, ci_hi=hi, bonf_lo=loB, bonf_hi=hiB))
        print(f"{name:>7} {fwd:>+8.3f} {rev:>+8.3f} {delta:>+8.3f} {gamma:>+8.3f} "
              f"[{lo:+.3f},{hi:+.3f}] [{loB:+.3f},{hiB:+.3f}]")
    out = A.RESULT_OUT / "A_analysis/a9_recompute_bestval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "bestval_summary.csv", index=False)
    print("\n[참고] §4.4(최종모델): fwd −0.231 rev +0.004 γ −0.113 | RB δ+0.198 γ−0.151")

if __name__ == "__main__":
    main()
