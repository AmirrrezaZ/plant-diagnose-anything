"""v6: CLIP-L/14 zero-shot (name prompts) and BCVSA-style blend (DINOv2-L -> MLP aligned to CLIP-L/14 text; blended with CLIP zero-shot) on the v6 10-fold unit-grouped CV.
Re-implementation of the BCVSA core (github.com/RakibHossain011/BCVSA); tau/alpha/epoch chosen on VAL units only; two encoders (reference baseline). Crop given, candidates = all units of the crop; unit-macro top-1 over the 116 test units of multi-unit crops.
  python v6_bcvsa.py   -> results/v6/bcvsa.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
A = pd.read_csv(ROOT / "data/annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(ROOT / "data/split/splits_cv.csv"), on="img_id")
def feats(path):
    E = np.load(ROOT / "results/cache" / path, allow_pickle=True); pos = pd.Series(np.arange(len(E["img_id"])), index=E["img_id"]); X = E["feats"][pos[A.img_id].values].astype(np.float32); return X / np.linalg.norm(X, axis=1, keepdims=True)
XD, XC = feats("emb_dinov2l.npz"), feats("emb_clipl14.npz")
nh = (A.disease_type != "healthy").to_numpy(); UNITS = sorted(A[nh].disease_unit.unique()); UI = {u: i for i, u in enumerate(UNITS)}
meta = {u: A[A.disease_unit == u].iloc[0] for u in UNITS}; crop_of = np.array([meta[u].crop for u in UNITS]); CM = A.crop.to_numpy()[:, None] == crop_of[None]
ncrop = pd.Series(crop_of).map(pd.Series(crop_of).value_counts()).to_numpy(); uarr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in uarr]); multi = np.array([ncrop[UI[u]] >= 2 if u in UI else False for u in uarr])
from transformers import CLIPModel, CLIPTokenizer
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
def tembed(txts):
    with torch.no_grad(): t = tok(txts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to("cuda"); return F.normalize(cm.get_text_features(**t).float(), dim=-1).cpu().numpy()
PT = ["a high-resolution photo of a {c} {d} diseased plant leaf", "a close-up image of {c} {d} disease symptoms", "a plant leaf affected by {c} {d}", "a leaf exhibiting signs of {c} {d} disease", "a clear image of a {c} {d} infected leaf"]
T = np.stack([(lambda e: e / np.linalg.norm(e))(tembed([p.format(c=meta[u].crop, d=meta[u].disease_name) for p in PT]).mean(0)) for u in UNITS]).astype(np.float32); Tt = torch.as_tensor(T).cuda(); XDt = torch.as_tensor(XD).cuda(); clip_raw = XC @ T.T
def sm(z, tau): z = z / tau; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def hits(S, rows): S = np.where(CM[rows], S, -1e9); return pd.Series(S.argmax(1) == tgt[rows]).groupby(uarr[rows]).mean()
class Wrap(nn.Module):
    def __init__(s): super().__init__(); s.m = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Dropout(0.25), nn.Linear(2048, 768))
    def forward(s, x): return F.normalize(s.m(x), dim=-1)
def proj(net, rows):
    with torch.no_grad(): return (net(XDt[rows]) @ Tt.T).cpu().numpy()
res = {m: {} for m in ("clipL_zeroshot", "bcvsa_align_only", "bcvsa_blend")}
for k in range(10):
    role = A[f"cv{k}"].to_numpy(); tr = np.where((role == "train") & nh)[0]; va = np.where((role == "val") & nh & multi)[0]; te = np.where((role == "test") & nh & multi)[0]
    seen = sorted({tgt[i] for i in tr}); smap = {u: i for i, u in enumerate(seen)}; ytr = torch.as_tensor([smap[tgt[i]] for i in tr]).cuda(); trt = torch.as_tensor(tr).cuda(); Ts = Tt[seen]
    torch.manual_seed(42); net = Wrap().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    def vs(): net.eval(); return hits(proj(net, va), va).mean()
    best = (vs(), {a: b.clone() for a, b in net.state_dict().items()}, 0); bad = 0
    for ep in range(1, 101):
        net.train(); perm = torch.randperm(len(tr)).cuda()
        for s_ in range(0, len(tr) - 127, 128):
            ii = perm[s_:s_ + 128]; loss = F.cross_entropy(net(XDt[trt[ii]]) @ Ts.T / 0.03, ytr[ii]); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            e = vs()
            if e > best[0] + 1e-4: best = (e, {a: b.clone() for a, b in net.state_dict().items()}, ep); bad = 0
            else:
                bad += 1
                if bad >= 3: break
    net.load_state_dict(best[1]); net.eval(); aln_va, clip_va = proj(net, va), clip_raw[va]; bs = (-1, 0.05, 1.0)
    for tau in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        for alpha in np.round(np.linspace(0, 1, 11), 2):
            sc = hits(alpha * sm(aln_va, tau) + (1 - alpha) * sm(clip_va, tau), va).mean()
            if sc > bs[0]: bs = (sc, tau, alpha)
    _, tau, alpha = bs; Sa = proj(net, te)
    res["clipL_zeroshot"].update(hits(clip_raw[te], te).to_dict()); res["bcvsa_align_only"].update(hits(Sa, te).to_dict()); res["bcvsa_blend"].update(hits(alpha * sm(Sa, tau) + (1 - alpha) * sm(clip_raw[te], tau), te).to_dict())
    print("fold", k, "epoch", best[2], "val", round(float(best[0]), 3), "tau/alpha", tau, alpha, flush=True)
out = {m: dict(top1_unit_macro=float(np.mean(list(d.values()))), n_units=len(d), per_unit=d) for m, d in res.items()}
(ROOT / "results/v6").mkdir(exist_ok=True); json.dump(out, open(ROOT / "results/v6/bcvsa.json", "w"), indent=1)
for m, d in out.items(): print(f"{m:20s} {d['top1_unit_macro']:.3f} (n {d['n_units']})")
