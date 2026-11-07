"""Unit-grouped cross-validation over ALL disease units (v6). K folds; fold k: test = units of fold k, val = units of fold (k+1)%K, train = the rest. Each unit is tested exactly once.
Folds are balanced by tag-novelty (min L1 distance of the unit's tag profile to any other unit), crop spread (same-crop units go to different folds where possible), unit count and image count; best of N random restarts.
Healthy photos: fixed per-crop 60/20/20 train/val/test across folds (duplicate groups kept together). Writes data/split/splits_cv.csv (img_id, cv0..cvK-1) and unit_folds.csv. splits.csv untouched."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; K = 10; N = 4000
A = pd.read_csv(ROOT / "data/annotation/annotations_v6.csv", keep_default_na=False); nh = A[A.disease_type != "healthy"]
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; vals = sorted({(a, v) for a in ATT for s in nh[a] for v in s.split(";") if v})
UN = sorted(nh.disease_unit.unique()); crop = {u: nh[nh.disease_unit == u].crop.iloc[0] for u in UN}; nimg = nh.disease_unit.value_counts().to_dict()
P = np.stack([[nh[nh.disease_unit == u][a].map(lambda s, v=v: v in s.split(";")).mean() for a, v in vals] for u in UN]); D = np.abs(P[:, None] - P[None]).sum(2); np.fill_diagonal(D, 1e9); nov = D.min(1)
tert = pd.qcut(pd.Series(nov).rank(method="first"), 3, labels=False).to_numpy()
rng = np.random.default_rng(20261103); best = None
def score(fold):
    f = np.array([fold[u] for u in UN]); viol = 0
    for c in set(crop.values()):
        us = [i for i, u in enumerate(UN) if crop[u] == c]
        if len(us) == 1: continue
        for k in range(K):
            held = {k, (k + 1) % K}; left = [i for i in us if f[i] not in held]
            viol += (len(left) == 0) * 100 + sum(f[i] in held for i in us[:0])
        viol += sum(1 for k in range(K) if sum(f[i] == k for i in us) > 1) * 3  # two same-crop units in one fold
    cnt = np.bincount(f, minlength=K); imgs = np.array([sum(nimg[u] for i, u in enumerate(UN) if f[i] == k) for k in range(K)]); nv = np.array([nov[f == k].mean() for k in range(K)]); tt = np.array([[(tert[f == k] == t).sum() for t in range(3)] for k in range(K)])
    return viol + 5 * cnt.std() + imgs.std() / imgs.mean() * 20 + nv.std() / nov.std() * 10 + tt.std() * 2
for it in range(N):
    order = rng.permutation(len(UN)); order = sorted(order, key=lambda i: tert[i]); fold = {}; load = np.zeros(K)
    for i in order:
        u = UN[i]; cand = [k for k in np.argsort(load + rng.random(K) * 0.5) if sum(fold.get(UN[j]) == k for j in range(len(UN)) if crop[UN[j]] == crop[u] and UN[j] in fold) == 0] or list(np.argsort(load)); fold[u] = int(cand[0]); load[cand[0]] += 1
    s = score(fold)
    if best is None or s < best[0]: best = (s, dict(fold))
fold = best[1]; print("best score", round(best[0], 2))
UF = pd.DataFrame(dict(disease_unit=UN, crop=[crop[u] for u in UN], fold=[fold[u] for u in UN], novelty=nov, tertile=tert, n_images=[nimg[u] for u in UN])); UF.to_csv(ROOT / "data/split/unit_folds.csv", index=False)
out = pd.DataFrame({"img_id": A.img_id}); h = (A.disease_type == "healthy").to_numpy(); hrole = np.array(["train"] * len(A), dtype=object)
for c, g in A[h].groupby("crop"):
    d = g.dup_group_id.unique(); rng.shuffle(d); n = len(d); role = {x: ("train" if i < 0.6 * n else "val" if i < 0.8 * n else "test") for i, x in enumerate(d)}; hrole[g.index.map(lambda i: A.index.get_loc(i))] = g.dup_group_id.map(role).to_numpy()
uf = A.disease_unit.map(fold).to_numpy(float)
for k in range(K):
    r = np.where(h, hrole, np.where(uf == k, "test", np.where(uf == (k + 1) % K, "val", "train"))); out[f"cv{k}"] = r
out.to_csv(ROOT / "data/split/splits_cv.csv", index=False)
print(UF.groupby("fold").agg(units=("disease_unit", "size"), imgs=("n_images", "sum"), novelty=("novelty", "mean"), crops=("crop", "nunique")).round(2).to_string())
lone = [c for c in set(crop.values()) if sum(crop[u] == c for u in UN) == 1]; print("single-unit crops (crop unseen when tested):", lone)
bad = []
for k in range(K):
    tr = set(nh[(out.loc[nh.index, f"cv{k}"] == "train")].crop); te = set(nh[(out.loc[nh.index, f"cv{k}"] == "test")].crop); bad += [(k, c) for c in te if c not in tr]
print("test crops with no train unit:", bad)
# attribute-value coverage: units carrying the value in train / test per fold
cov = []
for k in range(K):
    tr = nh[out.loc[nh.index, f"cv{k}"] == "train"]; te = nh[out.loc[nh.index, f"cv{k}"] == "test"]
    for a, v in vals:
        ntr = tr[a].map(lambda s: v in s.split(";")).groupby(tr.disease_unit).any().sum(); nte = te[a].map(lambda s: v in s.split(";")).groupby(te.disease_unit).any().sum(); cov.append((k, a, v, int(ntr), int(nte)))
C = pd.DataFrame(cov, columns=["fold", "attr", "value", "train_units", "test_units"]); print("value-folds with <3 train units:", int((C.train_units < 3).sum()), "| values never tested in a fold:", int(((C.test_units == 0)).sum()), "of", len(C))
print("values with <2 test units overall:", C.groupby(["attr", "value"]).test_units.sum().loc[lambda s: s < 2].to_dict())
