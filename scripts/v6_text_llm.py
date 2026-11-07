"""v6 text embeddings for the BLIND (LLM-written, name-only) profiles + attribute-value probes. SigLIP2-L-256 text tower, same templates as v6_text.py.
Input data/profiles/llm_profiles.json (v6_llm_profiles.py). Output results/cache/v6_text_llm.npz:
 pw_units (order = v6_text.npz units), pw_long/pw_tags/pw_name ; dg_types, dg_long/dg_tags/dg_name ; attr_meta, attr_emb (one prompt per attribute value).   python v6_text_llm.py"""
import json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
P = json.load(open(ROOT / "data/profiles/llm_profiles.json")); prof = P["profiles"]; vals = P["vocab"]; ATT = list(vals)
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]; dg_types = P["dg_types"]
j = lambda l: " and ".join(l)
def texts(c, d, p):
    p = {a: [t.replace("_", " ") for t in p[a]] for a in ATT}
    return {"long": [f"a photo of a {c} plant with {j(p['color'])} {j(p['shape'])} {j(p['texture'])} {j(p['pattern'])} lesions on the {j(p['affected_part'])}, {j(p['lesion_density'])} spots.", f"{c} {j(p['affected_part'])} with {j(p['color'])} {j(p['pattern'])} symptoms."],
            "tags": [f"a photo of a {c} {j(p['affected_part'])}: " + ", ".join(j(v) for v in p.values() if v) + "."],
            "name": [t.format(c=c, d=d) for t in ["a photo of a {c} plant with {d}.", "a photo of {c} leaves affected by {d}.", "{c} {d}.", "a {c} plant showing symptoms of {d}.", "a close-up photo of {d} on {c}."]]}
from transformers import AutoModel, AutoProcessor
name = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name, dtype=torch.float32).cuda().eval()
def emb(t):
    with torch.no_grad():
        tok = proc(text=[x.lower() for x in t], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda"); f = model.get_text_features(**tok); f = f if isinstance(f, torch.Tensor) else f.pooler_output; return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
def me(t): e = emb(t).mean(0); return e / np.linalg.norm(e)
out = {"pw_units": np.array(UN), "dg_types": np.array(dg_types)}
for pre, keys in (("pw", UN), ("dg", dg_types)):
    T = [texts(k.split("|")[0], k.split("|", 1)[1], prof[k.lower()]) for k in keys]
    for st in ("long", "tags", "name"): out[f"{pre}_{st}"] = np.stack([me(t[st]) for t in T])
TPL = {"affected_part": "a photo of a plant {v}", "color": "a photo of a plant with {v} lesions", "texture": "a photo of a plant with {v} lesions", "shape": "a photo of a plant with {v} lesions", "pattern": "a photo of a plant with {v} symptoms", "lesion_density": "a photo of a plant with {v} lesions"}
meta = [(a, v) for a in ATT for v in vals[a]]; out["attr_meta"] = np.array([f"{a}|{v}" for a, v in meta]); out["attr_emb"] = np.stack([me([TPL[a].format(v=v.replace("_", " "))]) for a, v in meta])
np.savez(ROOT / "results/cache/v6_text_llm.npz", **out); print({k: getattr(v, "shape", v) for k, v in out.items()})
