"""v6 text embeddings (SigLIP2-L-256, run where the model is cached): per disease unit and per cluster, several prompt styles. -> results/cache/v6_text.npz"""
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
A = pd.read_csv(ROOT / "data/annotation/annotations_v6.csv", keep_default_na=False); nh = A[A.disease_type != "healthy"]
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]
def prof(g, share):
    out = {}
    for a in ATT:
        cnt = {}
        for s in g[a]:
            for t in [t for t in s.split(";") if t]: cnt[t] = cnt.get(t, 0) + 1
        out[a] = [t.replace("_", " ") for t, k in sorted(cnt.items(), key=lambda x: -x[1]) if k / len(g) >= share and t != "none"]
    return out
j = lambda l: " and ".join(l)
def texts(c, disease, p):
    return {"long": [f"a photo of a {c} plant with {j(p['color'])} {j(p['shape'])} {j(p['texture'])} {j(p['pattern'])} lesions on the {j(p['affected_part'])}, {j(p['lesion_density'])} spots.", f"{c} {j(p['affected_part'])} with {j(p['color'])} {j(p['pattern'])} symptoms."],
            "short": [f"a {c} {j(p['affected_part'])} with {j(p['color'])} {j(p['shape'])} lesions."],
            "tags": [f"a photo of a {c} {j(p['affected_part'])}: " + ", ".join(j(v) for v in p.values() if v) + "."],
            "name": [t.format(c=c, d=disease) for t in ["a photo of a {c} plant with {d}.", "a photo of {c} leaves affected by {d}.", "{c} {d}.", "a {c} plant showing symptoms of {d}.", "a close-up photo of {d} on {c}."]],
            "crop": [f"a photo of a {c} plant.", f"{c} leaves."]}
from transformers import AutoModel, AutoProcessor
name = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name, dtype=torch.float32).cuda().eval()
def emb(t):
    with torch.no_grad():
        tok = proc(text=[x.lower() for x in t], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda"); f = model.get_text_features(**tok); f = f if isinstance(f, torch.Tensor) else f.pooler_output; return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
def me(t): e = emb(t).mean(0); return e / np.linalg.norm(e)
out = {"scale": float(model.logit_scale.exp()), "bias": float(model.logit_bias)}
UN = sorted(nh.disease_unit.unique()); out["units"] = np.array(UN)
for st in ("long", "short", "tags", "name", "crop"): out["unit_" + st] = np.stack([me(texts(nh[nh.disease_unit == u].crop.iloc[0], nh[nh.disease_unit == u].disease_name.iloc[0], prof(nh[nh.disease_unit == u], 0.25))[st]) for u in UN])
CL = sorted(nh.group_id.unique()); out["clusters"] = np.array(CL); out["cluster_unit"] = np.array([nh[nh.group_id == c].disease_unit.iloc[0] for c in CL])
for st in ("long", "short", "tags"): out["cluster_" + st] = np.stack([me(texts(nh[nh.group_id == c].crop.iloc[0], "", prof(nh[nh.group_id == c], 0.5))[st]) for c in CL])
np.savez(ROOT / "results/cache/v6_text.npz", **out); print({k: getattr(v, "shape", v) for k, v in out.items()})
