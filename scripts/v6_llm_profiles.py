"""v6 blind symptom profiles: an open LLM (Qwen2.5-VL-7B-Instruct used text-only) writes the typical visible symptoms of a disease from the CROP and DISEASE NAME ONLY, using our tag vocabulary.
It never sees images or our annotations, so the profile is what a user could supply for a new disease. Keys are 'crop|disease' (lower case).
Inputs: the 120 PlantWild units (annotations_v6.csv) and all single-diagnosis non-healthy DigiGreen types (dg_annotations.csv).  Output: data/profiles/llm_profiles.json
  python v6_llm_profiles.py"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False); nh = A[A.disease_type != "healthy"]
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; vals = {a: sorted({v for s in A[a] for v in s.split(";") if v and v != "none"}) for a in ATT}
keys = sorted({f"{c}|{d}".lower() for c, d in zip(nh.crop, nh.disease_name)})
DGF = DATA / "external/digigreen/annotations.csv"; dg = pd.read_csv(DGF)
dg = dg[dg.diagnosis.fillna("").str.count(";") == 0]; dg = dg[dg.diagnosis.str.strip().str.lower() != "healthy"]
dgkeys = sorted({f"{c.strip().lower()}|{d.strip().lower()}" for c, d in zip(dg.crop, dg.diagnosis)}); allkeys = sorted(set(keys) | set(dgkeys)); print("units", len(keys), "dg types", len(dgkeys), "total", len(allkeys), flush=True)
HINT = {"affected_part": "plant part(s) where the symptoms appear", "color": "colour(s) of the damaged tissue", "texture": "surface texture of the damaged tissue", "shape": "shape of lesions or deformation", "pattern": "spatial pattern of the symptoms", "lesion_density": "how many lesions"}
def prompt(c, d):
    lines = ",\n".join(f'"{a}": [...]  // {HINT[a]}; choose 1-3 values ONLY from {vals[a]}' for a in ATT)
    return (f"You are an expert plant pathologist. Crop: {c}. Disease or problem: {d}.\nUsing only your agronomic knowledge of this disease (no photo is given), state the typical VISIBLE symptoms in a photograph of an affected plant. "
            f"Answer with a single JSON object with exactly these keys; use only the listed values; use an empty list if an attribute does not apply:\n{{\n{lines}\n}}")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
mn = "Qwen/Qwen2.5-VL-7B-Instruct"; proc = AutoProcessor.from_pretrained(mn, use_fast=False); model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mn, torch_dtype=torch.bfloat16).cuda().eval()
out = {}; raw = {}
for i, k in enumerate(allkeys):
    c, d = k.split("|", 1); msgs = [{"role": "user", "content": [{"type": "text", "text": prompt(c, d)}]}]; t = proc.apply_chat_template(msgs, add_generation_prompt=True) + "{\n"
    inp = proc(text=[t], return_tensors="pt").to("cuda")
    with torch.no_grad(): o = model.generate(**inp, max_new_tokens=260, do_sample=False)
    txt = "{\n" + proc.batch_decode(o[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]; raw[k] = txt
    try: js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception: js = {}
    out[k] = {a: [v for v in (js.get(a) or []) if v in vals[a]][:3] for a in ATT}
    if i % 20 == 0: print(i, "/", len(allkeys), k, out[k], flush=True)
(DATA / "profiles").mkdir(exist_ok=True); json.dump(dict(source="Qwen2.5-VL-7B-Instruct text-only, crop+disease name only", vocab=vals, profiles=out, pw_units=keys, dg_types=dgkeys), open(DATA / "profiles/llm_profiles.json", "w"), indent=1); json.dump(raw, open(DATA / "profiles/llm_profiles_raw.json", "w"), indent=1)
print("empty profiles:", sum(1 for k in out if not any(out[k].values())), "of", len(out))
