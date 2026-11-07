"""Per-attribute image->symptom evaluation of the VISION-ONLY LoRA checkpoints (G0_fold{k}.pt, seed 0, 10 folds; text tower unadapted), same protocol as v6_attr_eval.py:
hit = top-1 value (generic value prompts scored with the ADAPTED model) is among the group's annotated values; unit-macro over test units; compared with prior / same-crop prior / zero-shot (B=0).
  python v6_joint_attr.py -> results/cv/vision_attr.json   (run where checkpoints/v6/J{s}_fold{k}_s{s}.pt and images exist, e.g. a GPU machine)"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, peft
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as TV
from transformers import AutoModel, AutoProcessor
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; CK = ROOT / "checkpoints/v6"; dev = "cuda"; RES = 256
ATTS = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]
TPL = {"affected_part": "a photo of a plant {v}", "color": "a photo of a plant with {v} lesions", "texture": "a photo of a plant with {v} lesions", "shape": "a photo of a plant with {v} lesions", "pattern": "a photo of a plant with {v} symptoms", "lesion_density": "a photo of a plant with {v} lesions"}
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
TL = np.load(ROOT / "results/cache/v6_text_llm.npz", allow_pickle=True); AM = [str(m).split("|") for m in TL["attr_meta"]]
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); crop = A.crop.to_numpy(); ucrop = pd.Series([u.split("|")[0] for u in UN]).value_counts(); multi = np.array([ucrop.get(u.split("|")[0], 0) >= 2 for u in unit_arr])
def unit_macro(hit, idx): return float(pd.Series(hit).groupby(unit_arr[idx]).mean().mean())
sn = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(sn); model = AutoModel.from_pretrained(sn, dtype=torch.float32).to(dev).eval()
peft.inject_adapter_in_model(peft.LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=r".*vision_model.*(q_proj|k_proj|v_proj|out_proj)"), model)
named = dict(model.named_parameters()); [setattr(p, "requires_grad", False) for p in model.parameters()]
def zero_lora():
    with torch.no_grad():
        for n, p in named.items():
            if "lora_B" in n: p.zero_()
def tenc(ts):
    tk = proc(text=[t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_text_features(**tk)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
tf = TV.Compose([TV.Resize((RES, RES), antialias=True), TV.ToTensor(), TV.Normalize([.5] * 3, [.5] * 3)])
class DS(Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): im = Image.open(DATA / A.img_path.iloc[int(s.idx[i])]); im.draft("RGB", (RES * 2, RES * 2)); return tf(im.convert("RGB"))
def img_feats(idx):
    H = []
    with torch.no_grad():
        for x in DataLoader(DS(idx), batch_size=64, num_workers=4):
            with torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=x.to(dev))
            f = f if isinstance(f, torch.Tensor) else f.pooler_output; H.append(F.normalize(f.float(), dim=-1))
    return torch.cat(H)
prompts = [TPL[a].format(v=v.replace("_", " ")) for a, v in AM]
out = {a: {m: [] for m in ("prior", "cropprior", "zs", "vision")} for a in ATTS}; seedv = {a: [] for a in ATTS}
for fold in range(10):
    role = A[f"cv{fold}"].to_numpy(); tr = np.where((role == "train") & nh)[0]; te = np.where((role == "test") & nh & multi)[0]
    Ys = {}
    for a in ATTS:
        ix = [i for i, (aa, v) in enumerate(AM) if aa == a]; names = [AM[i][1] for i in ix]; ni = {n: k for k, n in enumerate(names)}; V = len(names)
        Y = np.zeros((len(A), V), np.float32); ok = np.zeros(len(A), bool)
        for k, s in enumerate(A[a].to_numpy()):
            vs = [v for v in s.split(";") if v in ni]
            if vs: ok[k] = True; Y[k, [ni[v] for v in vs]] = 1 / len(vs)
        Ys[a] = (ix, Y, ok)
    def evalm():
        H = img_feats(te); E = tenc(prompts); S = (H @ E.T).cpu().numpy(); r = {}
        for a in ATTS:
            ix, Y, ok = Ys[a]; tem = te[ok[te]]
            if len(tem) == 0: continue
            sub = S[np.where(ok[te])[0]][:, ix]; r[a] = unit_macro(Y[tem][np.arange(len(tem)), sub.argmax(1)] > 0, tem)
        return r
    zero_lora(); zs = evalm(); js = []
    for s in range(1):
        ck = torch.load(CK / f"G0_fold{fold}.pt", map_location="cpu"); zero_lora()
        with torch.no_grad():
            for n, v in ck["lora"].items(): named[n].copy_(v.to(dev))
        js.append(evalm())
    for a in ATTS:
        ix, Y, ok = Ys[a]; tem = te[ok[te]]; trm = tr[ok[tr]]
        if len(tem) == 0 or len(trm) == 0: continue
        V = Y.shape[1]; prior = np.bincount(Y[trm].argmax(1), minlength=V).argmax(); out[a]["prior"].append(unit_macro(Y[tem][:, prior] > 0, tem))
        cp = {c: np.bincount(Y[trm[crop[trm] == c]].argmax(1), minlength=V).argmax() for c in set(crop[trm])}
        out[a]["cropprior"].append(unit_macro(np.array([Y[k, cp.get(crop[k], prior)] > 0 for k in tem]), tem)); out[a]["zs"].append(zs[a]); out[a]["vision"].append(float(np.mean([j[a] for j in js]))); seedv[a].append([j[a] for j in js])
    print("fold", fold, flush=True)
rng = np.random.default_rng(0); res = {}
for a in ATTS:
    res[a] = {m: float(np.mean(out[a][m])) for m in out[a]}
    for base in ("prior", "cropprior", "zs"):
        d = np.array(out[a]["vision"]) - np.array(out[a][base]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        res[a][f"vision_minus_{base}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((d > 0).sum()), int((d < 0).sum())]
    print(a, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in res[a].items()}, flush=True)
(ROOT / "results/cv").mkdir(parents=True, exist_ok=True); json.dump(dict(per_fold=out, per_seed=seedv, summary=res), open(ROOT / "results/cv/vision_attr.json", "w"), indent=1)
