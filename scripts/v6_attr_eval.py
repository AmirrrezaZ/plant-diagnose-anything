"""v6 image -> symptom transfer test (the paper's claim): predict each attribute's value(s) of images of HELD-OUT diseases.
Frozen SigLIP2-L-256 image features (cached). Hit = top-1 predicted value is among the group's annotated values (group-level labels!). Unit-macro over test units; 10 unit-grouped CV folds.
Methods: prior (most common train value), zero-shot attribute-value prompts, linear head trained on train units initialised at the zero-shot text weights (zsinit) or at 0 (scratch); epoch picked on val units.
  python v6_attr_eval.py -> results/cv/attr_transfer.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; torch.manual_seed(0)
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
Z = np.load(ROOT / "results/cache/embeddings_siglip2l256_all.npz", allow_pickle=True); pos = {i: k for k, i in enumerate(Z["img_id"])}; X = F.normalize(torch.tensor(Z["feats"][[pos[i] for i in A.img_id]]), dim=-1).cuda()
TL = np.load(ROOT / "results/cache/v6_text_llm.npz", allow_pickle=True); AM = [str(m).split("|") for m in TL["attr_meta"]]; AE = torch.tensor(TL["attr_emb"]).cuda()
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]; UI = {u: i for i, u in enumerate(UN)}
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); crop = A.crop.to_numpy(); ucrop = pd.Series([u.split("|")[0] for u in UN]).value_counts(); multi = np.array([ucrop.get(u.split("|")[0], 0) >= 2 for u in unit_arr])
ATTS = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; scale = float(Tz["scale"])
def unit_macro(hit, idx): return float(pd.Series(hit).groupby(unit_arr[idx]).mean().mean())
out = {a: {m: [] for m in ("prior", "cropprior", "zs", "zsinit", "scratch")} for a in ATTS}; nunits = {a: [] for a in ATTS}
for fold in range(10):
    role = A[f"cv{fold}"].to_numpy(); tr = np.where((role == "train") & nh)[0]; va = np.where((role == "val") & nh & multi)[0]; te = np.where((role == "test") & nh & multi)[0]
    for a in ATTS:
        ix = [i for i, (aa, v) in enumerate(AM) if aa == a]; names = [AM[i][1] for i in ix]; V = len(names); ni = {n: k for k, n in enumerate(names)}
        Y = np.zeros((len(A), V), np.float32); ok = np.zeros(len(A), bool)
        for k, s in enumerate(A[a].to_numpy()):
            vs = [v for v in s.split(";") if v in ni]
            if vs: ok[k] = True; Y[k, [ni[v] for v in vs]] = 1 / len(vs)
        trm, vam, tem = tr[ok[tr]], va[ok[va]], te[ok[te]]
        if len(tem) == 0 or len(trm) == 0: continue
        hit_of = lambda S, idx: Y[idx][np.arange(len(idx)), S.argmax(1)] > 0
        prior = np.bincount(np.array(Y[trm].argmax(1)), minlength=V).argmax(); out[a]["prior"].append(unit_macro(Y[tem][:, prior] > 0, tem))
        cp = {}
        for c in set(crop[trm]):
            m_ = trm[crop[trm] == c]; cp[c] = np.bincount(Y[m_].argmax(1), minlength=V).argmax()
        out[a]["cropprior"].append(unit_macro(np.array([Y[k, cp.get(crop[k], prior)] > 0 for k in tem]), tem))
        W0 = AE[ix]; zs = (X[tem] @ W0.T).cpu().numpy(); out[a]["zs"].append(unit_macro(hit_of(zs, tem), tem))
        for name, init in (("zsinit", W0.clone()), ("scratch", torch.zeros(V, 1024, device="cuda"))):
            W = torch.nn.Parameter(init); b = torch.nn.Parameter(torch.zeros(V, device="cuda")); opt = torch.optim.AdamW([W, b], lr=2e-3, weight_decay=0.01); Yt = torch.tensor(Y).cuda(); best = (-1, None)
            cnt = pd.Series(unit_arr[trm]).value_counts(); w = (1 / np.sqrt(pd.Series(unit_arr[trm]).map(cnt).to_numpy(float))); w /= w.sum(); rng = np.random.default_rng(fold)
            for ep in range(40):
                for _ in range(20):
                    bi = torch.tensor(trm[rng.choice(len(trm), 256, p=w)]).cuda(); lg = scale * X[bi] @ W.T + b; loss = -(Yt[bi] * F.log_softmax(lg, 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
                if len(vam):
                    with torch.no_grad(): s = unit_macro(hit_of((scale * X[vam] @ W.T + b).cpu().numpy(), vam), vam)
                    if s > best[0] + 1e-4: best = (s, (W.detach().clone(), b.detach().clone()))
            if best[1] is None: best = (0, (W.detach().clone(), b.detach().clone()))
            with torch.no_grad(): out[a][name].append(unit_macro(hit_of((scale * X[tem] @ best[1][0].T + best[1][1]).cpu().numpy(), tem), tem))
        nunits[a].append(len(set(unit_arr[tem])))
    print("fold", fold, flush=True)
rng = np.random.default_rng(0); res = {}
for a in ATTS:
    res[a] = {}
    for m in out[a]: res[a][m] = float(np.mean(out[a][m]))
    for m in ("cropprior", "zs", "zsinit", "scratch"):
        d = np.array(out[a][m]) - np.array(out[a]["prior"]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; res[a][f"{m}_minus_prior"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(a, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in res[a].items()}, flush=True)
(ROOT / "results/cv").mkdir(parents=True, exist_ok=True); json.dump(dict(per_fold=out, summary=res), open(ROOT / "results/cv/attr_transfer.json", "w"), indent=1)

for a in ATTS:
    for m in ("zsinit", "scratch"):
        d = np.array(out[a][m]) - np.array(out[a]["cropprior"]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; res[a][f"{m}_minus_cropprior"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(a, "cropprior", round(res[a]["cropprior"], 3), "zsinit-cropprior", [round(x, 3) for x in res[a]["zsinit_minus_cropprior"]], "scratch-cropprior", [round(x, 3) for x in res[a]["scratch_minus_cropprior"]], flush=True)
json.dump(dict(per_fold=out, summary=res), open(ROOT / "results/cv/attr_transfer.json", "w"), indent=1)
