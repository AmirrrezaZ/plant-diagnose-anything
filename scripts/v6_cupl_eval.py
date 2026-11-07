"""Training-free: average SigLIP2 text embeddings of 6 blind LLM paraphrases per disease (v6_cupl.py) vs the blind tag-profile text; CV test units (10 folds) and DG-full types. No training.
  python v6_cupl_eval.py -> results/cv/cupl.json"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; dev = "cuda"
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
Z = np.load(ROOT / "results/cache/embeddings_siglip2l256_all.npz", allow_pickle=True); pos = {i: k for k, i in enumerate(Z["img_id"])}; X = F.normalize(torch.tensor(Z["feats"][[pos[i] for i in A.img_id]], device=dev), dim=-1)
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]; UI = {u: i for i, u in enumerate(UN)}; scale = float(Tz["scale"])
TL = np.load(ROOT / "results/cache/v6_text_llm.npz", allow_pickle=True); DGT = [str(x) for x in TL["dg_types"]]; PP = json.load(open(DATA / "profiles/llm_paraphrases.json"))
from transformers import AutoModel, AutoProcessor
sn = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(sn); sm = AutoModel.from_pretrained(sn, dtype=torch.float32).to(dev).eval()
def emb(ts):
    with torch.no_grad():
        tk = proc(text=[t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev); f = sm.get_text_features(**tk); f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
def para(keys, crop_prefix):
    out = []
    for k in keys:
        c = k.split("|")[0]; E = emb([(f"a photo of a {c} plant. " if crop_prefix else "") + s for s in PP[k.lower()]]); out.append(F.normalize(E.mean(0), dim=0))
    return torch.stack(out)
T_L = torch.tensor(TL["pw_long"], device=dev); T_tags = torch.tensor(TL["pw_tags"], device=dev); T_name = torch.tensor(TL["pw_name"], device=dev)
T_P = para(UN, False); T_Pc = para(UN, True)
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr]); crop = A.crop.to_numpy(); unit_crop = np.array([u.split("|")[0] for u in UN]); CM = crop[:, None] == unit_crop[None]
uc = pd.Series(unit_crop).value_counts(); multi = np.array([uc.get(u.split("|")[0], 0) >= 2 for u in unit_arr])
def um(T, idx): S = np.where(CM[idx], (scale * X[idx] @ T.T).cpu().numpy(), -1e18); return float(pd.Series(S.argmax(1) == tgt[idx]).groupby(unit_arr[idx]).mean().mean())
V = {"blind_long": T_L, "blind_tags": T_tags, "name": T_name, "paraphrases": T_P, "paraphrases+crop": T_Pc, "paraphrases+tags": F.normalize(T_P + T_tags, dim=-1), "paraphrases+name": F.normalize(T_P + T_name, dim=-1), "paraphrases+tags+name": F.normalize(T_P + T_tags + T_name, dim=-1)}
res = {k: [] for k in V}
for f in range(10):
    te = np.where((A[f"cv{f}"].to_numpy() == "test") & nh & multi)[0]
    for k, T in V.items(): res[k].append(um(T, te))
rng = np.random.default_rng(0); summ = {}
for k in V:
    d = np.array(res[k]) - np.array(res["blind_long"]); b = [d[rng.integers(0, 10, 10)].mean() for _ in range(10000)]; summ[k] = dict(mean=float(np.mean(res[k])), minus_blind_long=[float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    print("CV", k, round(summ[k]["mean"], 3), [round(x, 3) for x in summ[k]["minus_blind_long"]], flush=True)
# DG-full
ALIAS = {"maize": "corn", "brinjal": "eggplant", "paddy": "rice", "capsicum": "bell pepper", "peas": "pea", "arhar": "pigeon pea"}
dg = pd.read_csv(DATA / "external/digigreen/annotations.csv"); dg = dg[dg.diagnosis.fillna("").str.count(";") == 0].copy(); dg["diag"] = dg.diagnosis.str.strip().str.lower(); dg = dg[dg.diag != "healthy"]
dg["crop_l"] = dg.crop.str.strip().str.lower(); dg["type"] = dg.crop_l + "|" + dg.diag; dg = dg[dg.type.isin(DGT)].copy(); dg["crop_a"] = dg.crop_l.map(lambda c: ALIAS.get(c, c))
cnt_t = dg.type.value_counts(); dg = dg[dg.type.map(cnt_t) >= 3]; nt = dg.groupby("crop_a").type.nunique(); dg = dg[dg.crop_a.map(nt) >= 2].reset_index(drop=True); types = sorted(dg.type.unique())
Dz = np.load(ROOT / "results/cache/emb_digigreen_l256.npz", allow_pickle=True); dp = {str(f): k for k, f in enumerate(Dz["files"])}; Xd = F.normalize(torch.tensor(Dz["feats"][[dp[Path(f).name] if Path(f).name in dp else dp[f] for f in dg.image_file]], device=dev), dim=-1)
ti = [DGT.index(t) for t in types]; cm = dg.crop_a.to_numpy()[:, None] == np.array([ALIAS.get(t.split("|")[0], t.split("|")[0]) for t in types])[None]; y = np.array([types.index(t) for t in dg.type])
TD = {"blind_long": torch.tensor(TL["dg_long"], device=dev)[ti], "blind_tags": torch.tensor(TL["dg_tags"], device=dev)[ti], "name": torch.tensor(TL["dg_name"], device=dev)[ti]}; TD["paraphrases"] = para(types, False)
TD["paraphrases+tags"] = F.normalize(TD["paraphrases"] + TD["blind_tags"], dim=-1); TD["paraphrases+name"] = F.normalize(TD["paraphrases"] + TD["name"], dim=-1); TD["paraphrases+tags+name"] = F.normalize(TD["paraphrases"] + TD["blind_tags"] + TD["name"], dim=-1)
dgres = {}
for k, T in TD.items():
    S = np.where(cm, (scale * Xd @ T.T).cpu().numpy(), -1e18); per = pd.Series(S.argmax(1) == y).groupby(dg.type.to_numpy()).mean(); dgres[k] = per.to_dict(); print("DG", k, round(per.mean(), 3), flush=True)
base = np.array([dgres["blind_long"][t] for t in types])
for k in TD:
    d = np.array([dgres[k][t] for t in types]) - base; b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; print("DG minus blind_long", k, f"{d.mean():+.3f} [{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]")
(ROOT / "results/cv").mkdir(parents=True, exist_ok=True); json.dump(dict(cv=res, cv_summary=summ, dg_per_type=dgres), open(ROOT / "results/cv/cupl.json", "w"), indent=1)
