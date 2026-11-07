"""v6 VLM baseline: Qwen2.5-VL-7B-Instruct zero-shot on a fixed sample of v6-CV test images (PER per test unit, seed 0) + the 69 DigiGreen main images. Crop given.
 (a) TAGS: asked for our symptom vocabulary as JSON without naming the disease -> cosine of the tag vector with each candidate unit's tag profile (symptom route, no names).
 (b) MC-NAME: multiple choice among the crop's disease names (name route, shuffled order).
Output: results/v6/vlm_raw.jsonl (resumable) and results/v6/vlm.json (unit-macro top-1).   python v6_vlm.py --per 6"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=6); ap.add_argument("--limit", type=int, default=0); g = ap.parse_args()
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; vals = {a: sorted({v for s in A[a] for v in s.split(";") if v and v != "none"}) for a in ATT}
cols = [(a, v) for a in ATT for v in vals[a]]; nh = (A.disease_type != "healthy").to_numpy(); uarr = A.disease_unit.to_numpy(); UN = sorted(set(uarr[nh])); UI = {u: i for i, u in enumerate(UN)}
Y = np.stack([A[a].map(lambda s, v=v: v in s.split(";")).to_numpy() for a, v in cols], 1).astype(np.float32); PROF = np.stack([Y[uarr == u].mean(0) for u in UN]); PN = PROF / np.linalg.norm(PROF, axis=1, keepdims=True)
unit_crop = np.array([A[A.disease_unit == u].crop.iloc[0] for u in UN]); unit_name = [A[A.disease_unit == u].disease_name.iloc[0] for u in UN]; ncrop = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy()
rng = np.random.default_rng(0); items = []  # (key, path, crop, true_unit)
for k in range(10):
    te = np.where((A[f"cv{k}"] == "test").to_numpy() & nh)[0]
    for u, gi in pd.Series(te).groupby(uarr[te]):
        if ncrop[UI[u]] < 2: continue
        for i in rng.choice(gi.values, min(g.per, len(gi)), replace=False): items.append((f"cv|{A.img_id.iloc[i]}", str(DATA / A.img_path.iloc[i]), A.crop.iloc[i], u))
Mp = pd.read_csv(ROOT / "results/external/digigreen_mapping_used.csv"); DN = {"leaf curl virus": "yellow leaf curl virus", "turcicum leaf blight": "northern leaf blight", "tobacco mosaic virus": "mosaic virus", "panama wilt": "panama disease", "citrus canker": "canker", "sigatoka leaf spot": "black leaf streak"}
Mp["unit"] = [f"{c}|{DN.get(d, d)}" for c, d in zip(Mp.crop, Mp.dg)]; Mp = Mp[(Mp.cos_nn <= 0.95) & Mp.unit.isin(UN) & (Mp.tier == "main")]
for _, r in Mp.iterrows(): items.append((f"dg|{r.image_file}", str(DATA / "external/digigreen" / r.image_file), r.crop, r.unit))
items = items[:g.limit] if g.limit else items; print("images", len(items), flush=True)
HINT = {"affected_part": "which plant part shows the symptoms", "color": "colour(s) of the damaged tissue", "texture": "surface texture", "shape": "shape of lesions or deformation", "pattern": "spatial pattern", "lesion_density": "how many lesions"}
PROMPT = "You are a plant pathologist. Describe ONLY the visible symptoms in this photo (do not name the disease). Answer with a single JSON object with exactly these keys, each a list of one or more of the listed values (or [] if not applicable):\n{\n" + ",\n".join(f'"{a}": {vals[a]}  // {HINT[a]}' for a in ATT) + "\n}"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
mn = "Qwen/Qwen2.5-VL-7B-Instruct"; proc = AutoProcessor.from_pretrained(mn, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28, use_fast=False)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mn, torch_dtype=torch.bfloat16).cuda().eval()
def ask(img, text, n):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]; t = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[t], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad(): o = model.generate(**inp, max_new_tokens=n, do_sample=False)
    return proc.batch_decode(o[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
fn = ROOT / "results/v6/vlm_raw.jsonl"; (ROOT / "results/v6").mkdir(exist_ok=True); done = {json.loads(l)["key"] for l in open(fn)} if fn.exists() else set()
with open(fn, "a") as f:
    for n, (key, path, crop, u) in enumerate(items):
        if key in done: continue
        img = Image.open(path).convert("RGB"); cand = [i for i in range(len(UN)) if unit_crop[i] == crop]; order = list(np.random.default_rng(abs(hash(key)) % 2**32).permutation(cand))
        mc = f"This is a photo of a {crop} plant. Which disease or problem does it show? Options:\n" + "\n".join(f"{j + 1}. {unit_name[c]}" for j, c in enumerate(order)) + "\nAnswer with the option number only."
        f.write(json.dumps(dict(key=key, unit=u, crop=crop, tags_raw=ask(img, PROMPT, 220), mc_raw=ask(img, mc, 6), order=[int(c) for c in order])) + "\n"); f.flush()
        if n % 20 == 0: print(n, "/", len(items), flush=True)
def score():
    res = {"tags": {}, "mc_name": {}}
    for l in open(fn):
        r = json.loads(l); sp = "dg" if r["key"].startswith("dg") else "cv"; tu = UI[r["unit"]]; cand = np.array([i for i in range(len(UN)) if unit_crop[i] == r["crop"]])
        m = re.search(r"\{.*\}", "{" + r["tags_raw"] if not r["tags_raw"].lstrip().startswith("{") else r["tags_raw"], re.S); vec = np.zeros(len(cols), np.float32)
        try:
            d = json.loads(m.group(0)); vec = np.array([1.0 if v in (d.get(a) or []) else 0.0 for a, v in cols], np.float32)
        except Exception: pass
        s = PN[cand] @ (vec / max(np.linalg.norm(vec), 1e-9)); hit_t = float(cand[int(np.argmax(s))] == tu) if vec.any() else float(1 / len(cand))
        mm = re.search(r"\d+", r["mc_raw"]); pick = r["order"][int(mm.group(0)) - 1] if mm and 0 < int(mm.group(0)) <= len(r["order"]) else -1
        res["tags"].setdefault((sp, r["unit"]), []).append(hit_t); res["mc_name"].setdefault((sp, r["unit"]), []).append(float(pick == tu))
    out = {}
    for m, d in res.items():
        for sp in ("cv", "dg"): v = [np.mean(x) for (s, u), x in d.items() if s == sp]; out[f"{m}_{sp}"] = dict(top1_unit_macro=float(np.mean(v)) if v else None, n_units=len(v))
    return out
o = score(); json.dump(o, open(ROOT / "results/v6/vlm.json", "w"), indent=1); print(json.dumps(o, indent=1))
