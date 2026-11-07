"""v6 CuPL-style blind paraphrases: Qwen2.5-VL-7B (text only) writes 6 short, different visual descriptions per (crop, disease) from the names alone (no images, no annotations).
Output data/profiles/llm_paraphrases.json {key: [sentences]}.  python v6_cupl.py"""
import json, re
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
P = json.load(open(DATA / "profiles/llm_profiles.json")); keys = sorted(P["profiles"])
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
mn = "Qwen/Qwen2.5-VL-7B-Instruct"; proc = AutoProcessor.from_pretrained(mn, use_fast=False); model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mn, torch_dtype=torch.bfloat16).cuda().eval()
out = {}; OUT = DATA / "profiles/llm_paraphrases.json"
for i, k in enumerate(keys):
    c, d = k.split("|", 1)
    q = (f"You are an expert plant pathologist. Crop: {c}. Disease or problem: {d}.\nWrite exactly 6 different one-sentence descriptions of what an affected {c} plant LOOKS LIKE in a photograph "
         f"(colour, shape, texture, pattern and number of lesions, which plant part). Do not mention the disease name or any pathogen. Cover different aspects and different stages. One sentence per line, no numbering.")
    t = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": q}]}], add_generation_prompt=True)
    inp = proc(text=[t], return_tensors="pt").to("cuda")
    with torch.no_grad(): o = model.generate(**inp, max_new_tokens=300, do_sample=False)
    txt = proc.batch_decode(o[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
    L = [re.sub(r"^[\-\*\d\.\)\s]+", "", s).strip() for s in txt.split("\n") if len(s.strip()) > 20]; out[k] = L[:6]
    if i % 25 == 0: print(i, "/", len(keys), k, out[k][:2], flush=True); json.dump(out, open(OUT, "w"), indent=1)
json.dump(out, open(OUT, "w"), indent=1); print("done", len(out), "short:", sum(1 for v in out.values() if len(v) < 4))
