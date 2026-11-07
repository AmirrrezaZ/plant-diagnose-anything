"""v6 JOINT model: SigLIP2-L-256 with LoRA on BOTH the vision tower and the text tower, trained end to end (images are encoded with gradients; descriptions are re-sampled every step from the unit profiles, image-derived A + blind LLM L).
Loss: image vs descriptions of same-crop TRAIN units (restricted softmax). Both LoRA B matrices start at 0, so step 0 == frozen zero-shot. Selection on val units; test units scored with canonical A_long / L_long / L_tags / L_name text at best val step and step 0.
  python v6_joint.py --fold 0 --tag J0 --seed 0            CV fold
  python v6_joint.py --fold -1 --steps 400 --tag JDG       train on all PlantWild units not matching DigiGreen types; evaluate on DG-full (zero-shot via B=0 in the same pipeline)
Saves checkpoints/v6/{tag}_fold{fold}_s{seed}.pt (vision+text LoRA, logit scale/bias, args)."""
import argparse, re, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as TV
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--steps", type=int, default=600); ap.add_argument("--eval_every", type=int, default=100); ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--lr_img", type=float, default=2e-5); ap.add_argument("--lr_txt", type=float, default=1e-4); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--tag", default="J0"); ap.add_argument("--train_text", default="M"); ap.add_argument("--patience", type=int, default=3); g = ap.parse_args()
dev = "cuda"; torch.manual_seed(g.seed); rng = np.random.default_rng(g.seed); RES = 256
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; SW = ATT[1:]
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]; UI = {u: i for i, u in enumerate(UN)}; scale0, bias0 = float(Tz["scale"]), float(Tz["bias"])
TL = np.load(ROOT / "results/cache/v6_text_llm.npz", allow_pickle=True); DGT = [str(x) for x in TL["dg_types"]]; P = json.load(open(DATA / "profiles/llm_profiles.json"))
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr]); crop = A.crop.to_numpy(); unit_crop = np.array([u.split("|")[0] for u in UN]); CM = crop[:, None] == unit_crop[None]
ncu = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncu[UI[u]] >= 2 if u in UI else False for u in unit_arr])
ALIAS = {"maize": "corn", "brinjal": "eggplant", "paddy": "rice", "capsicum": "bell pepper", "peas": "pea", "arhar": "pigeon pea"}
norm = lambda s: re.sub(r"\b(leaf|disease|virus|of|the)\b", "", s.lower()).replace(" ", "").replace("-", "")
def matches(pwkey, dgkey):
    c1, d1 = pwkey.split("|", 1); c2, d2 = dgkey.split("|", 1); c2 = ALIAS.get(c2, c2)
    return c1 == c2 and (norm(d1) == norm(d2) or (len(norm(d2)) > 3 and (norm(d2) in norm(d1) or norm(d1) in norm(d2))))
if g.fold >= 0:
    role = A[f"cv{g.fold}"].to_numpy(); tr = np.where((role == "train") & nh)[0]; va = np.where((role == "val") & nh & multi)[0]; te = np.where((role == "test") & nh & multi)[0]
else:
    DGM = {u for u in UN for d in DGT if matches(u, d)}; tr = np.where(nh & ~np.isin(unit_arr, list(DGM)))[0]; va = np.array([], int); te = np.array([], int); print("train imgs", len(tr), "excluded units", len(DGM), flush=True)
train_units = sorted(set(tgt[tr])); tu = np.zeros(len(UN), bool); tu[train_units] = True
# ---- profiles / texts (same as v6_text_ft.py)
def prof_img(gdf, share=0.25):
    out = {}
    for a in ATT:
        cnt = {}
        for s in gdf[a]:
            for t in [t for t in s.split(";") if t]: cnt[t] = cnt.get(t, 0) + 1
        out[a] = [t.replace("_", " ") for t, k in sorted(cnt.items(), key=lambda x: -x[1]) if k / len(gdf) >= share and t != "none"]
    return out
PA = {u: prof_img(A[(A.disease_unit == u) & nh]) for u in UN}; PL = {u: {a: [t.replace("_", " ") for t in P["profiles"][u.lower()][a]] for a in ATT} for u in UN}; PDG = {k: {a: [t.replace("_", " ") for t in P["profiles"][k][a]] for a in ATT} for k in DGT}
j = lambda l: " and ".join(l)
def canon(c, d, p):
    return {"long": [f"a photo of a {c} plant with {j(p['color'])} {j(p['shape'])} {j(p['texture'])} {j(p['pattern'])} lesions on the {j(p['affected_part'])}, {j(p['lesion_density'])} spots.", f"{c} {j(p['affected_part'])} with {j(p['color'])} {j(p['pattern'])} symptoms."],
            "tags": [f"a photo of a {c} {j(p['affected_part'])}: " + ", ".join(j(v) for v in p.values() if v) + "."],
            "name": [t.format(c=c, d=d) for t in ["a photo of a {c} plant with {d}.", "a photo of {c} leaves affected by {d}.", "{c} {d}.", "a {c} plant showing symptoms of {d}.", "a close-up photo of {d} on {c}."]]}
def render(c, q, t):
    part = j(q["affected_part"]) or "plant"; at = [j(q[a]) for a in ("color", "shape", "texture", "pattern") if q[a]]; den = j(q["lesion_density"]); aj = " ".join(at) or "visible"
    if t == 0: return f"a photo of a {c} plant with {aj} lesions on the {part}" + (f", {den} spots." if den else ".")
    if t == 1: return f"{c} {part} with {', '.join(at) or 'visible'} symptoms" + (f", {den}." if den else ".")
    if t == 2: return f"a photo of a {c} {part}: " + ", ".join(x for x in at + ([den] if den else []) if x) + "."
    if t == 3: return f"a close-up of {aj} lesions on a {c} {part}" + (f" ({den})." if den else ".")
    return f"{c} plant, {part} showing {', '.join(at) or 'symptoms'}" + (f", {den}." if den else ".")
def sample_state(p, r):
    q = {a: list(p[a]) for a in ATT}
    for a in SW:
        if q[a]:
            if r.random() < 0.25: q[a] = []
            else: q[a] = [str(x) for x in r.permutation(q[a])[: int(r.integers(1, len(q[a]) + 1))]]
    if not any(q[a] for a in SW): q = {a: list(p[a]) for a in ATT}
    return q
# ---- model: vision + text LoRA
from transformers import AutoModel, AutoProcessor
sn = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(sn); model = AutoModel.from_pretrained(sn, dtype=torch.float32).to(dev)
for p_ in model.parameters(): p_.requires_grad = False
import peft
peft.inject_adapter_in_model(peft.LoraConfig(r=g.rank, lora_alpha=g.rank, lora_dropout=0.0, target_modules=r".*(vision_model|text_model).*(q_proj|k_proj|v_proj|out_proj)"), model)
for n, p_ in model.named_parameters(): p_.requires_grad = "lora_" in n
pv = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and "vision_model" in n]; pt = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and "text_model" in n]; print("trainable vision", sum(p_.numel() for p_ in pv), "text", sum(p_.numel() for p_ in pt), flush=True)
ls = torch.nn.Parameter(torch.tensor(float(np.log(scale0)), device=dev)); bb = torch.nn.Parameter(torch.tensor(bias0, device=dev))
def feats(x, grad=False):
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=x)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
def tenc(texts, grad=False):
    tk = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_text_features(**tk)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
def unit_emb(keys, kind, profs):
    ts = []; n = []
    for k in keys:
        c, d = k.split("|", 1); cs = canon(c, d, profs[k])[kind]; ts += cs; n.append(len(cs))
    with torch.no_grad(): E = torch.cat([tenc(ts[i:i + 128]) for i in range(0, len(ts), 128)])
    out = []; k0 = 0
    for m in n: out.append(F.normalize(E[k0:k0 + m].mean(0), dim=0)); k0 += m
    return torch.stack(out)
mean_, std_ = [.5] * 3, [.5] * 3
class DS(Dataset):
    def __init__(s, idx, train):
        s.idx = idx; s.tf = TV.Compose([TV.Resize((RES, RES), antialias=True), TV.ToTensor(), TV.Normalize(mean_, std_)]) if not train else TV.Compose([TV.RandomResizedCrop(RES, scale=(0.6, 1.0), antialias=True), TV.RandomHorizontalFlip(), TV.ColorJitter(brightness=0.1, contrast=0.1), TV.ToTensor(), TV.RandomErasing(p=0.25, scale=(0.02, 0.1)), TV.Normalize(mean_, std_)])
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): jx = int(s.idx[i]); im = Image.open(DATA / A.img_path.iloc[jx]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), jx
class PathDS(Dataset):
    def __init__(s, files): s.files = files; s.tf = TV.Compose([TV.Resize((RES, RES), antialias=True), TV.ToTensor(), TV.Normalize(mean_, std_)])
    def __len__(s): return len(s.files)
    def __getitem__(s, i): im = Image.open(s.files[i]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), i
def enc_ds(ds):
    model.eval(); H = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=96, num_workers=g.workers): H.append(feats(x.to(dev)))
    model.train(); return torch.cat(H)
KINDS = [("A", "long"), ("L", "long"), ("L", "tags"), ("L", "name")]
def eval_set(idx):
    H = enc_ds(DS(idx, False)); out = {}
    for src, kind in KINDS:
        T = unit_emb(UN, kind, PA if src == "A" else PL); S = (ls.exp() * H @ T.T + bb).detach().cpu().numpy(); S = np.where(CM[idx], S, -1e18); out[f"{src}_{kind}"] = float(pd.Series(S.argmax(1) == tgt[idx]).groupby(unit_arr[idx]).mean().mean())
    return out
sel = lambda r: float(np.mean([r["A_long"], r["L_long"]]))
cnt = pd.Series(unit_arr[tr]).value_counts(); w = 1 / np.sqrt(pd.Series(unit_arr[tr]).map(cnt).to_numpy(float)); w /= w.sum()
class TrainStream(Dataset):
    def __init__(s): s.draw = rng.choice(tr, g.steps * g.bs, p=w); s.tf = DS(tr, True).tf
    def __len__(s): return len(s.draw)
    def __getitem__(s, i): jx = int(s.draw[i]); im = Image.open(DATA / A.img_path.iloc[jx]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), jx
opt = torch.optim.AdamW([dict(params=pv, lr=g.lr_img, weight_decay=0.01), dict(params=pt, lr=g.lr_txt, weight_decay=0.01), dict(params=[ls, bb], lr=1e-4, weight_decay=0)])
wu = max(1, int(0.1 * g.steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / wu if t < wu else 0.5 * (1 + np.cos(np.pi * (t - wu) / max(1, g.steps - wu))))
snap = lambda: ({n: p_.detach().cpu().clone() for n, p_ in model.named_parameters() if "lora_" in n}, float(ls), float(bb))
def load_state(st):
    sd, l, b = st; named = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd.items(): named[n].copy_(v.to(dev))
        ls.copy_(torch.tensor(l)); bb.copy_(torch.tensor(b))
hist = []; best = (-1, 0, None)
if len(te):
    model.eval(); r0v = eval_set(va); r0t = eval_set(te); hist.append(dict(step=0, val=sel(r0v), test=r0t)); best = (sel(r0v), 0, r0t); print("step 0", hist[-1], flush=True)
best_state = snap(); bad = 0; model.train(); t0 = time.time(); lossm = []
loader = iter(DataLoader(TrainStream(), batch_size=g.bs, num_workers=g.workers, shuffle=False, drop_last=True)); params = pv + pt
for step in range(1, g.steps + 1):
    x, jb = next(loader); x = x.to(dev); jb = jb.numpy(); h = feats(x, True)
    cs = set(crop[jb]); units_b = [i for i in train_units if unit_crop[i] in cs]; texts = []
    for u in units_b:
        uname = UN[u]; c, d = uname.split("|", 1); src = "A" if (g.train_text == "A" or (g.train_text == "M" and rng.random() < 0.5)) else "L"
        texts.append(render(c, sample_state(PA[uname] if src == "A" else PL[uname], rng), int(rng.integers(5))))
    E = tenc(texts, True); ub = np.array(units_b); lg = ls.exp() * h @ E.T + bb
    ok = torch.tensor(CM[jb][:, ub], device=dev); lg = lg.masked_fill(~ok, -1e4); pos_ = {u: k for k, u in enumerate(units_b)}; ti = torch.tensor([pos_[int(t)] for t in tgt[jb]], device=dev)
    loss = F.cross_entropy(lg, ti); lossm.append(float(loss)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % g.eval_every == 0 or step == g.steps:
        if len(te):
            model.eval(); rv = eval_set(va); rt = eval_set(te); model.train(); hist.append(dict(step=step, loss=float(np.mean(lossm[-g.eval_every:])), val=sel(rv), test=rt)); print(hist[-1], f"{time.time() - t0:.0f}s", flush=True)
            if sel(rv) > best[0] + 1e-4: best = (sel(rv), step, rt); best_state = snap(); bad = 0
            else:
                bad += 1
                if bad >= g.patience: break
        else: print({"step": step, "loss": float(np.mean(lossm[-g.eval_every:]))}, f"{time.time() - t0:.0f}s", flush=True)
if not len(te): best_state = snap()
CK = ROOT / "checkpoints/v6"; CK.mkdir(parents=True, exist_ok=True); res = ROOT / "results/v3"; res.mkdir(exist_ok=True)
torch.save(dict(lora=best_state[0], logit_scale=best_state[1], logit_bias=best_state[2], args=vars(g), best_step=best[1]), CK / f"{g.tag}_fold{g.fold}_s{g.seed}.pt")
out = dict(args=vars(g), hist=hist, best_step=best[1], best_val=best[0])
if len(te): out.update(test_zs=hist[0]["test"], test_best=best[2]); print(f"FOLD {g.fold} step0 {hist[0]['test']} | best step {best[1]} val {best[0]:.3f} test {best[2]}", flush=True); json.dump(out, open(res / f"v6joint_{g.tag}_fold{g.fold}_s{g.seed}.json", "w"))
else:
    dg = pd.read_csv(DATA / "external/digigreen/annotations.csv"); dg = dg[dg.diagnosis.fillna("").str.count(";") == 0].copy(); dg["diag"] = dg.diagnosis.str.strip().str.lower(); dg = dg[dg.diag != "healthy"]
    dg["crop_l"] = dg.crop.str.strip().str.lower(); dg["type"] = dg.crop_l + "|" + dg.diag; dg = dg[dg.type.isin(DGT)].copy(); dg["crop_a"] = dg.crop_l.map(lambda c: ALIAS.get(c, c))
    cnt_t = dg.type.value_counts(); dg = dg[dg.type.map(cnt_t) >= 3]; nt = dg.groupby("crop_a").type.nunique(); dg = dg[dg.crop_a.map(nt) >= 2].reset_index(drop=True); types = sorted(dg.type.unique())
    files = [str(DATA / "external/digigreen" / f) for f in dg.image_file]; cm = dg.crop_a.to_numpy()[:, None] == np.array([ALIAS.get(t.split("|")[0], t.split("|")[0]) for t in types])[None]; y = np.array([types.index(t) for t in dg.type])
    def dg_eval():
        Hd = enc_ds(PathDS(files)); r = {}
        for kind in ("long", "tags", "name"):
            T = unit_emb(types, kind, PDG); S = np.where(cm, (ls.exp() * Hd @ T.T + bb).detach().cpu().numpy(), -1e18); per = pd.Series(S.argmax(1) == y).groupby(dg.type.to_numpy()).mean(); r[kind] = dict(type_macro=float(per.mean()), img=float(np.mean(S.argmax(1) == y)), per_type=per.to_dict())
        return r
    load_state(best_state); out["dg_tuned"] = dg_eval(); print("DG tuned", {k: round(v["type_macro"], 4) for k, v in out["dg_tuned"].items()}, flush=True)
    saved = {n: p_.detach().clone() for n, p_ in model.named_parameters() if "lora_B" in n}
    with torch.no_grad():
        for n, p_ in model.named_parameters():
            if "lora_B" in n: p_.zero_()
        ls.copy_(torch.tensor(float(np.log(scale0)))); bb.copy_(torch.tensor(bias0))
    out["dg_zs"] = dg_eval(); print("DG zero-shot", {k: round(v["type_macro"], 4) for k, v in out["dg_zs"].items()}, flush=True)
    json.dump(out, open(res / f"v6joint_{g.tag}_fold{g.fold}_s{g.seed}.json", "w"))
