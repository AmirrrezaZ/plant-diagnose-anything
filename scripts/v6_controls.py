"""v6 shortcut baselines and name controls on frozen SigLIP2-L-256 features (no training, CPU).
Unit-macro top-1 over the 116 test units (crop given, candidates = all disease units of the crop), each unit tested once in its CV fold.
Methods: chance (1/#candidates); zero-shot long / name / long+name; name- and description-text SHUFFLED across units of the same crop (control: should drop to ~chance);
crop_prior (image-independent: candidate whose tag profile is closest to the crop's mean TRAIN profile); nearest_seen (kNN train image -> its unit's tag profile -> closest candidate profile).
  python v6_controls.py --tag controls"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="controls"); ap.add_argument("--seeds", type=int, default=20); g = ap.parse_args()
A = pd.read_csv(ROOT / "data/annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(ROOT / "data/split/splits_cv.csv"), on="img_id")
z = np.load(ROOT / "results/cache/embeddings_siglip2l256_all.npz", allow_pickle=True); p = pd.Series(np.arange(len(z["img_id"])), index=z["img_id"]); X = z["feats"][p[A.img_id].values].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = list(Tz["units"]); UI = {u: i for i, u in enumerate(UN)}; scale, bias = float(Tz["scale"]), float(Tz["bias"])
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); crop = A.crop.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr])
unit_crop = np.array([A[A.disease_unit == u].crop.iloc[0] for u in UN]); CM = crop[:, None] == unit_crop[None]
ncrop = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncrop[UI[u]] >= 2 if u in UI else False for u in unit_arr])
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; Y = []
for a in ATT:
    for v in sorted({v for s in A[a] for v in s.split(";") if v and v != "none"}): Y.append(A[a].map(lambda s: v in s.split(";")).to_numpy())
Y = np.stack(Y, 1).astype(np.float32); PROF = np.stack([Y[unit_arr == u].mean(0) for u in UN]); PN = PROF / np.linalg.norm(PROF, axis=1, keepdims=True)
fold = np.full(len(A), -1)
for k in range(10): fold[(A[f"cv{k}"] == "test").to_numpy()] = k
L, N = Tz["unit_long"], Tz["unit_name"]
def shuffle_within_crop(T, rng):
    T2 = T.copy()
    for c in set(unit_crop):
        ix = np.where(unit_crop == c)[0]
        if len(ix) > 1:
            while True:
                pm = rng.permutation(ix)
                if (pm != ix).all(): break
            T2[ix] = T[pm]
    return T2
def unit_macro(S, idx):
    S = np.where(CM[idx], S, -1e18); hit = S.argmax(1) == tgt[idx]; return pd.Series(hit).groupby(unit_arr[idx]).mean()
res = {m: {} for m in ["chance", "zs_long", "zs_name", "zs_long+name", "zs_long_shuffled", "zs_name_shuffled", "crop_prior", "nearest_seen", "crop_prior_unseen_only", "nearest_seen_unseen_only"]}
rng = np.random.default_rng(0); shL = [shuffle_within_crop(L, rng) for _ in range(g.seeds)]; shN = [shuffle_within_crop(N, rng) for _ in range(g.seeds)]
for k in range(10):
    te = np.where((fold == k) & nh & multi)[0]; role = A[f"cv{k}"].to_numpy(); trn = np.where((role == "train") & nh)[0]
    res["chance"].update((1 / np.maximum(CM[te].sum(1), 1)).tolist() and pd.Series(1 / CM[te].sum(1)).groupby(unit_arr[te]).mean().to_dict())
    s = lambda T: scale * X[te] @ T.T + bias
    res["zs_long"].update(unit_macro(s(L), te).to_dict()); res["zs_name"].update(unit_macro(s(N), te).to_dict()); res["zs_long+name"].update(unit_macro(s(L) + s(N), te).to_dict())
    for nm, sh in (("zs_long_shuffled", shL), ("zs_name_shuffled", shN)):
        accs = pd.concat([unit_macro(s(T), te) for T in sh], axis=1).mean(1); res[nm].update(accs.to_dict())
    # crop prior: mean train profile of the crop
    cp = {c: PROF[[UI[u] for u in set(unit_arr[trn][crop[trn] == c])]].mean(0) for c in set(crop[trn])}
    Sprior = np.stack([PN @ (cp[c] / np.linalg.norm(cp[c])) if c in cp else np.zeros(len(UN)) for c in crop[te]]); res["crop_prior"].update(unit_macro(Sprior, te).to_dict())
    # nearest seen class: kNN over same-crop train images
    Ss = np.zeros((len(te), len(UN)))
    for i, j in enumerate(te):
        cand = trn[crop[trn] == crop[j]]
        if len(cand): nn = cand[np.argmax(X[cand] @ X[j])]; Ss[i] = PN @ PN[tgt[nn]]
    res["nearest_seen"].update(unit_macro(Ss, te).to_dict())
    # same two shortcuts but candidates restricted to units NOT seen in training (what a profile-lookup could do if it knew which diseases are new)
    seen = np.zeros(len(UN), bool); seen[list(set(tgt[trn]))] = True
    res["crop_prior_unseen_only"].update(unit_macro(np.where(seen[None], -1e9, Sprior), te).to_dict()); res["nearest_seen_unseen_only"].update(unit_macro(np.where(seen[None], -1e9, Ss), te).to_dict())
out = {m: dict(top1_unit_macro=float(np.mean(list(d.values()))), n_units=len(d), per_unit=d) for m, d in res.items()}
(ROOT / "results/v6").mkdir(exist_ok=True); json.dump(out, open(ROOT / f"results/v6/{g.tag}.json", "w"), indent=1)
for m, d in out.items(): print(f"{m:20s} {d['top1_unit_macro']:.3f}  (n units {d['n_units']})")
