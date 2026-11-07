"""Per-image predictions of the JOINT LoRA checkpoints (J0,J1,J2 x 10 folds) for the ambiguity analysis: top-3 predicted units (same-crop candidates) for image-derived (A) and blind (L) long descriptions.
Run on a GPU machine: python scripts/v6_joint_unitpred.py -> results/cv/joint_unitpred.csv (img_id,fold,seed,unit,src,pred1,pred2,pred3)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, peft
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as TV
from transformers import AutoModel, AutoProcessor
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; CK = ROOT / "checkpoints/v6"; dev = "cuda"; RES = 256
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
P = json.load(open(DATA / "profiles/llm_profiles.json")); Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]; UI = {u: i for i, u in enumerate(UN)}
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); crop = A.crop.to_numpy(); unit_crop = np.array([u.split("|")[0] for u in UN]); CM = crop[:, None] == unit_crop[None]
ncu = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncu[UI[u]] >= 2 if u in UI else False for u in unit_arr])
def prof_img(gdf, share=0.25):
    out = {}
    for a in ATT:
        cnt = {}
        for s in gdf[a]:
            for t in [t for t in s.split(";") if t]: cnt[t] = cnt.get(t, 0) + 1
        out[a] = [t.replace("_", " ") for t, k in sorted(cnt.items(), key=lambda x: -x[1]) if k / len(gdf) >= share and t != "none"]
    return out
PA = {u: prof_img(A[(A.disease_unit == u) & nh]) for u in UN}; PL = {u: {a: [t.replace("_", " ") for t in P["profiles"][u.lower()][a]] for a in ATT} for u in UN}
j = lambda l: " and ".join(l)
def long_texts(c, p): return [f"a photo of a {c} plant with {j(p['color'])} {j(p['shape'])} {j(p['texture'])} {j(p['pattern'])} lesions on the {j(p['affected_part'])}, {j(p['lesion_density'])} spots.", f"{c} {j(p['affected_part'])} with {j(p['color'])} {j(p['pattern'])} symptoms."]
sn = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(sn); model = AutoModel.from_pretrained(sn, dtype=torch.float32).to(dev).eval()
peft.inject_adapter_in_model(peft.LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=r".*(vision_model|text_model).*(q_proj|k_proj|v_proj|out_proj)"), model)
named = dict(model.named_parameters()); [setattr(p, "requires_grad", False) for p in model.parameters()]
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
def unit_embs(pf):
    ts = []; n = []
    for u in UN: c = u.split("|")[0]; t = long_texts(c, pf[u]); ts += t; n.append(len(t))
    E = torch.cat([tenc(ts[i:i + 128]) for i in range(0, len(ts), 128)]); out = []; k0 = 0
    for m in n: out.append(F.normalize(E[k0:k0 + m].mean(0), dim=0)); k0 += m
    return torch.stack(out)
rows = []
for fold in range(10):
    te = np.where((A[f"cv{fold}"].to_numpy() == "test") & nh & multi)[0]; imgs = A.img_id.to_numpy()[te]
    for s in range(3):
        ck = torch.load(CK / f"J{s}_fold{fold}_s{s}.pt", map_location="cpu")
        with torch.no_grad():
            for n, p in named.items():
                if "lora_B" in n: p.zero_()
            for n, v in ck["lora"].items(): named[n].copy_(v.to(dev))
        H = img_feats(te)
        for src, pf in (("A", PA), ("L", PL)):
            S = np.exp(ck["logit_scale"]) * (H @ unit_embs(pf).T).cpu().numpy() + ck["logit_bias"]
            S = np.where(CM[te], S, -1e18); top = np.argsort(-S, 1)[:, :3]
            for k in range(len(te)): rows.append((imgs[k], fold, s, unit_arr[te[k]], src, UN[top[k, 0]], UN[top[k, 1]] if top.shape[1] > 1 else "", UN[top[k, 2]] if top.shape[1] > 2 else ""))
    print("fold", fold, flush=True)
(ROOT / "results/cv").mkdir(parents=True, exist_ok=True); pd.DataFrame(rows, columns=["img_id", "fold", "seed", "unit", "src", "pred1", "pred2", "pred3"]).to_csv(ROOT / "results/cv/joint_unitpred.csv", index=False)
