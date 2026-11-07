"""Inference with a joint (vision+text LoRA) SigLIP2-L-256 checkpoint from v6_joint.py.
Given image(s), the crop, and candidate diseases described by symptoms (and/or names), print the ranked candidates.
Candidates JSON: a list of objects, each {"name": "...", "affected_part": [...], "color": [...], "texture": [...], "shape": [...], "pattern": [...], "lesion_density": [...]}
  (any attribute may be missing/empty; values should come from data/annotation/vocab.json; "name" is only used with --use_name). Free text: {"name": "...", "text": "a photo of ..."}.
  python v6_infer.py --ckpt checkpoints/v6/J0_fold0_s0.pt --image leaf.jpg --crop tomato --candidates cands.json [--use_name] [--zero_shot]"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--image", nargs="+", required=True); ap.add_argument("--crop", required=True); ap.add_argument("--candidates", required=True)
ap.add_argument("--use_name", action="store_true"); ap.add_argument("--zero_shot", action="store_true"); ap.add_argument("--topk", type=int, default=5); g = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]
j = lambda l: " and ".join(l)
def descr(c, p):
    p = {a: [x.replace("_", " ") for x in p.get(a, [])] for a in ATT}
    return [f"a photo of a {c} plant with {j(p['color'])} {j(p['shape'])} {j(p['texture'])} {j(p['pattern'])} lesions on the {j(p['affected_part'])}, {j(p['lesion_density'])} spots.", f"{c} {j(p['affected_part'])} with {j(p['color'])} {j(p['pattern'])} symptoms."]
def names(c, d): return [t.format(c=c, d=d) for t in ["a photo of a {c} plant with {d}.", "a photo of {c} leaves affected by {d}.", "{c} {d}.", "a {c} plant showing symptoms of {d}.", "a close-up photo of {d} on {c}."]]
from transformers import AutoModel, AutoProcessor
import peft
sn = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(sn); model = AutoModel.from_pretrained(sn, dtype=torch.float32).to(dev).eval()
ck = torch.load(g.ckpt, map_location="cpu")
peft.inject_adapter_in_model(peft.LoraConfig(r=ck["args"]["rank"], lora_alpha=ck["args"]["rank"], lora_dropout=0.0, target_modules=r".*(vision_model|text_model).*(q_proj|k_proj|v_proj|out_proj)"), model); model.to(dev)
named = dict(model.named_parameters())
with torch.no_grad():
    for n, v in ck["lora"].items(): named[n].copy_(v.to(dev))
    if g.zero_shot:
        for n, p in named.items():
            if "lora_B" in n: p.zero_()
scale = float(np.exp(ck["logit_scale"])) if not g.zero_shot else None
C = json.load(open(g.candidates)); texts = []; counts = []
for c in C:
    t = [c["text"]] if "text" in c else descr(g.crop, c)
    if g.use_name: t = t + names(g.crop, c["name"]) if "text" not in c else t
    texts.append(t); counts.append(len(t))
def tenc(ts):
    tk = proc(text=[t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): f = model.get_text_features(**tk)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
E_ = tenc([t for ts in texts for t in ts]); k0 = 0; T = []
for m in counts: T.append(F.normalize(E_[k0:k0 + m].mean(0), dim=0)); k0 += m
T = torch.stack(T)
import torchvision.transforms as TV
tf = TV.Compose([TV.Resize((256, 256), antialias=True), TV.ToTensor(), TV.Normalize([.5] * 3, [.5] * 3)])
x = torch.stack([tf(Image.open(p).convert("RGB")) for p in g.image]).to(dev)
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): f = model.get_image_features(pixel_values=x)
f = f if isinstance(f, torch.Tensor) else f.pooler_output; H = F.normalize(f.float(), dim=-1)
if scale is None: scale = float(model.logit_scale.exp())
P = F.softmax(scale * H @ T.T, dim=-1).cpu().numpy()
for i, p in enumerate(g.image):
    print(p); order = np.argsort(-P[i])[: g.topk]
    for r in order: print(f"  {P[i][r]:.3f}  {C[r].get('name', '?')}")
