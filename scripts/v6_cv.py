"""v6 unit-grouped CV on frozen SigLIP2-L-256 features. Methods: zero-shot description / name / crop-only; linear tag head + lookup; supervised profile-matching adapter (zero-init => epoch 0 = zero-shot, early stop on val units);
fusion of zero-shot and head. Crop given at inference, candidates = all disease units of the crop. Headline = per-unit top-1 averaged over test units of crops with >= 2 units (unit-macro), bootstrap over units.
  python v6_cv.py --methods zs,head,adapter,fusion --seeds 0,1,2 --tag base"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--methods", default="zs,head,adapter,fusion"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--tag", default="base"); ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--text_adapter", type=int, default=0); ap.add_argument("--styles", default="long,tags"); ap.add_argument("--target", default="unit"); g = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; SEEDS = [int(s) for s in g.seeds.split(",")]; FOLDS = [int(f) for f in g.folds.split(",")]
A = pd.read_csv(ROOT / "data/annotation/annotations_v6.csv", keep_default_na=False); S = pd.read_csv(ROOT / "data/split/splits_cv.csv"); A = A.merge(S, on="img_id")
z = np.load(ROOT / "results/cache/embeddings_siglip2l256_all.npz", allow_pickle=True); p = pd.Series(np.arange(len(z["img_id"])), index=z["img_id"]); X = z["feats"][p[A.img_id].values].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = list(Tz["units"]); UI = {u: i for i, u in enumerate(UN)}; scale, bias = float(Tz["scale"]), float(Tz["bias"])
nh = (A.disease_type != "healthy").to_numpy(); unit_crop = np.array([A[A.disease_unit == u].crop.iloc[0] for u in UN]); crop = A.crop.to_numpy(); unit_arr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr])
CM = (crop[:, None] == unit_crop[None]); ncrop = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncrop[UI[u]] >= 2 if u in UI else False for u in unit_arr])
# --- tag targets for the head
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; cols = []; Y = []; M = []; rel = (A.attr_reliable.astype(str) == "True").to_numpy()
for a in ATT:
    vs = sorted({v for s in A[a] for v in s.split(";") if v and v != "none"}); blank = (A[a] == "").to_numpy()
    m = np.where(~nh, False, ~(blank & ~rel) if a in ("texture", "pattern") else True)
    for v in vs: cols.append((a, v)); Y.append(A[a].map(lambda s: v in s.split(";")).to_numpy()); M.append(m)
Y = np.stack(Y, 1).astype(np.float32); M = np.stack(M, 1)
PROF = np.zeros((len(UN), Y.shape[1]), np.float32)
for u in UN: ix = np.where(unit_arr == u)[0]; PROF[UI[u]] = np.clip((Y[ix] * M[ix]).sum(0) / np.maximum(M[ix].sum(0), 1), 0.03, 0.97)
LPOS, LNEG = np.log(PROF), np.log(1 - PROF)
usz = pd.Series(unit_arr).map(pd.Series(unit_arr).value_counts()).to_numpy(float); wimg = (1 / np.sqrt(usz)); wimg /= wimg.mean()
Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Y, device=dev); Mt = torch.tensor(M.astype(np.float32), device=dev); Wt = torch.tensor(wimg.astype(np.float32), device=dev)
TXT = {k: torch.tensor(Tz["unit_" + k].astype(np.float32), device=dev) for k in ("long", "short", "tags", "name", "crop")}
CLU = list(Tz["clusters"]); CI = {c: i for i, c in enumerate(CLU)}; CLT = {k: torch.tensor(Tz["cluster_" + k].astype(np.float32), device=dev) for k in ("long", "short", "tags")}; clus_of = np.array([CI.get(c, -1) for c in A.group_id]); clus_unit = np.array([UI[u] for u in Tz["cluster_unit"]])
CMt = torch.tensor(CM, device=dev)
def rank_hits(Sc, rows):  # Sc: (len(rows), U) scores; returns correct flags (crop given)
    S2 = np.where(CM[rows], Sc, -1e18); return (S2.argmax(1) == tgt[rows])
def zs_scores(rows, T): return scale * X[rows] @ T.T + bias
def train_head(tr, va, seed):
    torch.manual_seed(seed); W = torch.zeros(X.shape[1], Y.shape[1], device=dev, requires_grad=True); b = torch.full((Y.shape[1],), -1.5, device=dev, requires_grad=True); opt = torch.optim.AdamW([W, b], lr=0.02, weight_decay=0); tt = torch.tensor(tr, device=dev); best = (1e9, None, None); bad = 0
    def L(ii): zz = Xt[ii] * 10 @ W + b; return (F.binary_cross_entropy_with_logits(zz, Yt[ii], reduction="none") * Mt[ii] * Wt[ii][:, None]).sum() / Mt[ii].sum().clamp(min=1)
    vt = torch.tensor(va, device=dev)
    for ep in range(60):
        perm = tt[torch.randperm(len(tt), device=dev)]
        for s in range(0, len(perm), 512): l = L(perm[s:s + 512]) + 1e-4 * (W ** 2).sum(); opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad(): vl = float(L(vt))
        if vl < best[0] - 1e-5: best = (vl, W.detach().clone(), b.detach().clone()); bad = 0
        else:
            bad += 1
            if bad >= 5: break
    return best[1], best[2]
def head_scores(W, b, rows):
    with torch.no_grad(): P = torch.sigmoid(Xt[rows] * 10 @ W + b).cpu().numpy()
    return np.log(np.clip(P, 1e-4, 1)) @ LPOS.T + np.log(np.clip(1 - P, 1e-4, 1)) @ LNEG.T
class Adapter(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.W = torch.nn.Linear(d, d, bias=False); torch.nn.init.zeros_(s.W.weight); s.V = torch.nn.Linear(d, d, bias=False); torch.nn.init.zeros_(s.V.weight); s.ls = torch.nn.Parameter(torch.tensor(float(np.log(scale)))); s.b = torch.nn.Parameter(torch.tensor(bias))
    def img(s, x): return F.normalize(x + s.W(x), dim=-1)
    def txt(s, t): return F.normalize(t + s.V(t), dim=-1) if g.text_adapter else t
    def logits(s, x, t): return s.ls.exp() * s.img(x) @ s.txt(t).T + s.b
def train_adapter(tr, va, seed, train_units):
    torch.manual_seed(seed); net = Adapter(X.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=g.lr, weight_decay=g.wd); styles = g.styles.split(","); tu = torch.tensor(sorted(train_units), device=dev)
    trn = np.array([r for r in tr if tgt[r] in train_units and nh[r]]); trt = torch.tensor(trn, device=dev); tg = torch.tensor(tgt, device=dev); cl = torch.tensor(clus_of, device=dev)
    def val_acc():
        net.eval()
        with torch.no_grad(): Sc = net.logits(Xt[va], TXT["long"]).cpu().numpy()
        net.train(); return rank_hits(Sc, va).mean()
    best = (val_acc(), {k: v.clone() for k, v in net.state_dict().items()}); bad = 0
    for ep in range(40):
        perm = trt[torch.randperm(len(trt), device=dev)]
        for s in range(0, len(perm), 256):
            ii = perm[s:s + 256]; loss = 0
            for st in styles:
                if g.target == "cluster":
                    lg = net.logits(Xt[ii], CLT[st]); m = CMt[tg[ii]][:, 0:0]  # placeholder (unused)
                    ct = torch.tensor(clus_unit, device=dev); cand = (ct[None, :] != tg[ii][:, None]) | (torch.arange(len(CLU), device=dev)[None] == cl[ii][:, None]); samecrop = torch.tensor(unit_crop, device=dev) if False else None
                    cm_ = torch.tensor(CM, device=dev)[ii][:, ct]; trainu = torch.zeros(len(UN), dtype=torch.bool, device=dev); trainu[tu] = True; cm_ = cm_ & trainu[ct][None]
                    lg = lg.masked_fill(~cm_, -1e4); loss = loss + F.cross_entropy(lg, cl[ii])
                else:
                    lg = net.logits(Xt[ii], TXT[st]); mask = CMt[ii] & torch.isin(torch.arange(len(UN), device=dev), tu)[None]; lg = lg.masked_fill(~mask, -1e4); loss = loss + F.cross_entropy(lg, tg[ii], weight=None)
            opt.zero_grad(); (loss / len(styles)).backward(); opt.step()
        a = val_acc()
        if a > best[0] + 1e-4: best = (a, {k: v.clone() for k, v in net.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= 5: break
    net.load_state_dict(best[1]); net.eval(); return net
RES = []
for f in FOLDS:
    role = A[f"cv{f}"].to_numpy(); tr = np.where((role == "train"))[0]; va = np.where((role == "val") & nh)[0]; te = np.where((role == "test") & nh)[0]; train_units = set(tgt[tr[nh[tr]]])
    ev = te[multi[te]]; out = {}
    out["zs_long"] = rank_hits(zs_scores(ev, TXT["long"].cpu().numpy()), ev); out["zs_name"] = rank_hits(zs_scores(ev, TXT["name"].cpu().numpy()), ev); out["zs_crop"] = rank_hits(zs_scores(ev, TXT["crop"].cpu().numpy()), ev)
    Zv = zs_scores(va, TXT["long"].cpu().numpy()); Ze = zs_scores(ev, TXT["long"].cpu().numpy())
    if "head" in g.methods or "fusion" in g.methods:
        Hs = [head_scores(*train_head(tr, va, s), np.concatenate([va, ev])) for s in SEEDS]; H = np.mean(Hs, 0); Hv, He = H[:len(va)], H[len(va):]; out["head"] = rank_hits(He, ev)
    if "fusion" in g.methods:
        def norm(Sc, rows):  # standardise per image over same-crop candidates
            S2 = np.where(CM[rows], Sc, np.nan); mu = np.nanmean(S2, 1, keepdims=True); sd = np.nanstd(S2, 1, keepdims=True) + 1e-6; return (Sc - mu) / sd
        best = (-1, 0)
        for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            a = rank_hits(norm(Zv, va) + lam * norm(Hv, va), va).mean()
            if a > best[0] + 1e-9: best = (a, lam)
        out["fusion"] = rank_hits(norm(Ze, ev) + best[1] * norm(He, ev), ev); out["fusion_lam"] = np.full(len(ev), best[1])
    if "adapter" in g.methods:
        As = []
        for s in SEEDS:
            net = train_adapter(tr, va, s, train_units)
            with torch.no_grad(): As.append(net.logits(Xt[ev], TXT["long"]).cpu().numpy())
        out["adapter"] = rank_hits(np.mean(As, 0), ev)
    for k, v in out.items():
        for r, hit in zip(ev, v): RES.append(dict(fold=f, unit=unit_arr[r], method=k, hit=float(hit)))
    print(f, {k: round(float(np.mean(v)), 3) for k, v in out.items()}, flush=True)
D = pd.DataFrame(RES); D.to_csv(ROOT / f"results/v3/v6cv_{g.tag}.csv", index=False)
U = D.groupby(["method", "unit"]).hit.mean().reset_index(); piv = U.pivot(index="unit", columns="method", values="hit"); rng = np.random.default_rng(0)
def boot(x, B=2000): idx = rng.integers(0, len(x), (B, len(x))); m = x.to_numpy()[idx].mean(1); return np.percentile(m, [2.5, 97.5])
print("\nunit-macro top-1 (n units = %d), 95%% bootstrap over units" % len(piv))
for m in piv.columns:
    if m == "fusion_lam": continue
    lo, hi = boot(piv[m]); extra = ""
    if m != "zs_long" and "zs_long" in piv: d = piv[m] - piv["zs_long"]; l2, h2 = boot(d); extra = f"  diff vs zs_long {d.mean():+.3f} [{l2:+.3f}, {h2:+.3f}]"
    print(f"{m:10s} {piv[m].mean():.3f} [{lo:.3f}, {hi:.3f}]{extra}")
Wimg = D[D.method != "fusion_lam"].groupby("method").hit.mean().round(3).to_dict(); print("image-weighted:", Wimg)
