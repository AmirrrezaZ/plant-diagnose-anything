"""v6: LoRA on the SigLIP2-L-256 VISION tower, trained with the zero-shot objective (image vs. description text of a disease, softmax over same-crop TRAIN units; also over same-crop train clusters), text embeddings fixed (cached v6_text.npz).
LoRA B=0 at start => step 0 == frozen zero-shot. Selection on val units (unseen), test units of the CV fold reported at the best val step and at step 0. Run on arvan.
  python v6_lora.py --fold 0 --steps 600 --lr 5e-5 --rank 16"""
import argparse, re, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as TV
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--steps", type=int, default=600); ap.add_argument("--eval_every", type=int, default=100); ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--tag", default="lora"); ap.add_argument("--cluster_w", type=float, default=0.5); ap.add_argument("--styles", default="long,tags"); ap.add_argument("--patience", type=int, default=3); ap.add_argument("--targets", default="qkvo"); ap.add_argument("--ens", type=int, default=0)
ap.add_argument("--shuffle", type=int, default=0); ap.add_argument("--dg_full", type=int, default=0); ap.add_argument("--save", type=int, default=1); ap.add_argument("--exclude_dg", type=int, default=0); ap.add_argument("--train_text", default="A"); g = ap.parse_args()
dev = "cuda"; torch.manual_seed(g.seed); rng = np.random.default_rng(g.seed); RES = 256
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False); A = A.merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = list(Tz["units"]); UI = {u: i for i, u in enumerate(UN)}; scale0, bias0 = float(Tz["scale"]), float(Tz["bias"])
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr]); unit_crop = np.array([A[A.disease_unit == u].crop.iloc[0] for u in UN]); crop = A.crop.to_numpy(); CM = crop[:, None] == unit_crop[None]
ncrop = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncrop[UI[u]] >= 2 if u in UI else False for u in unit_arr])
CLU = list(Tz["clusters"]); CI = {c: i for i, c in enumerate(CLU)}; clus_of = np.array([CI.get(c, -1) for c in A.group_id]); clus_unit = np.array([UI[u] for u in Tz["cluster_unit"]])
TL = np.load(ROOT / "results/cache/v6_text_llm.npz", allow_pickle=True); DGT = [str(x) for x in TL["dg_types"]]
ALIAS = {"maize": "corn", "brinjal": "eggplant", "paddy": "rice", "capsicum": "bell pepper", "peas": "pea", "arhar": "pigeon pea"}
norm = lambda s: re.sub(r"\b(leaf|disease|virus|of|the)\b", "", s.lower()).replace(" ", "").replace("-", "")
def matches(pwkey, dgkey):
    c1, d1 = pwkey.split("|", 1); c2, d2 = dgkey.split("|", 1); c2 = ALIAS.get(c2, c2)
    return c1 == c2 and (norm(d1) == norm(d2) or (len(norm(d2)) > 3 and (norm(d2) in norm(d1) or norm(d1) in norm(d2))))
DG_MATCHED_PW = {u for u in UN for d in DGT if matches(u, d)}
if g.fold >= 0:
    role = A[f"cv{g.fold}"].to_numpy(); tr = np.where((role == "train") & nh)[0]; va = np.where((role == "val") & nh & multi)[0]; te = np.where((role == "test") & nh & multi)[0]
else:  # all-data training for the DigiGreen-full test: exclude every PlantWild unit that matches any DigiGreen disease
    tr = np.where(nh & ~np.isin(unit_arr, list(DG_MATCHED_PW)))[0]; va = np.array([], int); te = np.array([], int); print("train imgs", len(tr), "excluded matched units", len(DG_MATCHED_PW), flush=True)
train_units = sorted(set(tgt[tr])); tu_mask = np.zeros(len(UN), bool); tu_mask[train_units] = True
TXT = {k: torch.tensor(Tz["unit_" + k].astype(np.float32), device=dev) for k in ("long", "short", "tags", "name")}; CLT = {k: torch.tensor(Tz["cluster_" + k].astype(np.float32), device=dev) for k in ("long", "short", "tags")}
if g.train_text == "L":  # train on the BLIND (LLM-written, name-only) profiles of the training units instead of image-derived ones
    for k in ("long", "tags"): TXT[k] = torch.tensor(TL["pw_" + k].astype(np.float32), device=dev)
    print("training text = blind LLM profiles", flush=True)
CMt = torch.tensor(CM, device=dev); tgt_t = torch.tensor(tgt, device=dev); cl_t = torch.tensor(clus_of, device=dev); tu_t = torch.tensor(tu_mask, device=dev); cu_t = torch.tensor(clus_unit, device=dev); crop_unit_t = CMt
paths = A.img_path.to_numpy(); mean_, std_ = [.5] * 3, [.5] * 3
class DS(Dataset):
    def __init__(s, idx, train):
        s.idx = idx; s.tf = TV.Compose([TV.Resize((RES, RES), antialias=True), TV.ToTensor(), TV.Normalize(mean_, std_)]) if not train else TV.Compose([TV.RandomResizedCrop(RES, scale=(0.6, 1.0), antialias=True), TV.RandomHorizontalFlip(), TV.ColorJitter(brightness=0.1, contrast=0.1), TV.ToTensor(), TV.RandomErasing(p=0.25, scale=(0.02, 0.1)), TV.Normalize(mean_, std_)])
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): j = int(s.idx[i]); im = Image.open(DATA / paths[j]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), j
from transformers import AutoModel
model = AutoModel.from_pretrained("google/siglip2-large-patch16-256", dtype=torch.float32).to(dev); model.text_model = None
for p in model.parameters(): p.requires_grad = False
import peft
tm = {"qkvo": "q_proj|k_proj|v_proj|out_proj", "all": "q_proj|k_proj|v_proj|out_proj|fc1|fc2"}[g.targets]
peft.inject_adapter_in_model(peft.LoraConfig(r=g.rank, lora_alpha=g.rank, lora_dropout=0.0, target_modules=rf".*vision_model.*({tm})"), model)
for nme, p in model.named_parameters(): p.requires_grad = "lora_" in nme
ls = torch.nn.Parameter(torch.tensor(float(np.log(scale0)), device=dev)); bb = torch.nn.Parameter(torch.tensor(bias0, device=dev))
def feats(x):
    with torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=x)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW([dict(params=params, lr=g.lr, weight_decay=0.01), dict(params=[ls, bb], lr=1e-4, weight_decay=0)])
wu = max(1, int(0.1 * g.steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / wu if t < wu else 0.5 * (1 + np.cos(np.pi * (t - wu) / max(1, g.steps - wu))))
class PathDS(Dataset):
    def __init__(s, files): s.files = files; s.tf = TV.Compose([TV.Resize((RES, RES), antialias=True), TV.ToTensor(), TV.Normalize(mean_, std_)])
    def __len__(s): return len(s.files)
    def __getitem__(s, i): im = Image.open(s.files[i]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), i
def enc_ds(ds):
    model.eval(); H = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=96, num_workers=g.workers): H.append(feats(x.to(dev)))
    model.train(); return torch.cat(H)
encode = lambda idx: enc_ds(DS(idx, False))
def evaluate(idx, H):
    S = (ls.exp() * H @ TXT["long"].T + bb).detach().cpu().numpy(); S = np.where(CM[idx], S, -1e18); hit = (S.argmax(1) == tgt[idx]); u = pd.Series(hit).groupby(unit_arr[idx]).mean(); return float(u.mean()), float(hit.mean()), hit
cnt = pd.Series(unit_arr[tr]).value_counts(); w = 1 / np.sqrt(pd.Series(unit_arr[tr]).map(cnt).to_numpy(float)); w /= w.sum()
styles = g.styles.split(",")
tgt_loss = tgt.copy()
if g.shuffle:  # control: derange the unit labels of TRAIN units within each crop (image -> wrong unit text); evaluation uses true text
    prng = np.random.default_rng(1000 + g.seed); perm = np.arange(len(UN))
    for c in set(unit_crop):
        us = [i for i in train_units if unit_crop[i] == c]
        if len(us) > 1: sh = int(prng.integers(1, len(us))); perm[us] = np.array(us)[(np.arange(len(us)) + sh) % len(us)]
    tgt_loss = np.where(tgt >= 0, perm[np.maximum(tgt, 0)], -1); print("SHUFFLE control: train labels deranged within crop", flush=True)
tgt_loss_t = torch.tensor(tgt_loss, device=dev)
class TrainStream(Dataset):
    def __init__(s): s.draw = rng.choice(tr, g.steps * g.bs, p=w); s.tf = DS(tr, True).tf
    def __len__(s): return len(s.draw)
    def __getitem__(s, i): j = int(s.draw[i]); im = Image.open(DATA / paths[j]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), j
have_eval = len(te) > 0
if have_eval:
    Hva0, Hte0 = encode(va), encode(te); v0 = evaluate(va, Hva0); t0 = evaluate(te, Hte0); hist = [dict(step=0, val_unit=v0[0], val_img=v0[1], test_unit=t0[0], test_img=t0[1])]; print("step 0 (zero-shot)", hist[-1], flush=True); best = (v0[0], 0, t0, t0[2])
else: hist = []; best = (0, 0, None, None)
snap = lambda: ({n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}, float(ls), float(bb))
best_state = snap(); bad = 0; model.train(); t_start = time.time(); loader = iter(DataLoader(TrainStream(), batch_size=g.bs, num_workers=g.workers, shuffle=False, drop_last=True)); lossm = []
for step in range(1, g.steps + 1):
    x, j = next(loader); x = x.to(dev); j = j.to(dev); h = feats(x); loss = 0
    for st in styles:
        lg = ls.exp() * h @ TXT[st].T + bb; lg = lg.masked_fill(~(CMt[j] & tu_t[None]), -1e4); loss = loss + F.cross_entropy(lg, tgt_loss_t[j])
        if g.cluster_w > 0 and st != "name" and not g.shuffle:
            lc = ls.exp() * h @ CLT[st].T + bb; ok = CMt[j][:, cu_t] & tu_t[cu_t][None]; lc = lc.masked_fill(~ok, -1e4); loss = loss + g.cluster_w * F.cross_entropy(lc, cl_t[j])
    loss = loss / len(styles); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); lossm.append(float(loss))
    if step % g.eval_every == 0 or step == g.steps:
        if have_eval:
            Hv = encode(va); v = evaluate(va, Hv); Ht = encode(te); t = evaluate(te, Ht); hist.append(dict(step=step, loss=float(np.mean(lossm[-g.eval_every:])), val_unit=v[0], val_img=v[1], test_unit=t[0], test_img=t[1])); print(hist[-1], f"{time.time() - t_start:.0f}s", flush=True)
            if v[0] > best[0] + 1e-4: best = (v[0], step, t, t[2]); best_state = snap(); bad = 0
            else:
                bad += 1
                if bad >= g.patience: break
        else: print({"step": step, "loss": float(np.mean(lossm[-g.eval_every:]))}, f"{time.time() - t_start:.0f}s", flush=True)
if not have_eval: best_state = snap()
def load_state(st):
    sd, l, b = st; named = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd.items(): named[n].copy_(v.to(dev))
        ls.copy_(torch.tensor(l)); bb.copy_(torch.tensor(b))
ext = {}; res = ROOT / "results/v3"; res.mkdir(exist_ok=True); CK = ROOT / "checkpoints/v6"; CK.mkdir(parents=True, exist_ok=True)
load_state(best_state)
if g.save: torch.save(dict(lora=best_state[0], logit_scale=best_state[1], logit_bias=best_state[2], args=vars(g), best_step=best[1]), CK / f"{g.tag}_fold{g.fold}.pt")
sc = lambda H, T: (ls.exp() * H @ torch.tensor(T, device=dev).T + bb).detach().cpu().numpy()
if have_eval:
    Hte1 = encode(te); LTX = {"A_long": Tz["unit_long"], "L_long": TL["pw_long"], "L_tags": TL["pw_tags"], "L_name": TL["pw_name"]}
    for nm, T in LTX.items():
        for lab, H in (("zs", Hte0), ("tuned", Hte1)):
            S = np.where(CM[te], sc(H, T), -1e18); hit = S.argmax(1) == tgt[te]; ext[f"{nm}_{lab}"] = float(pd.Series(hit).groupby(unit_arr[te]).mean().mean())
    # per-attribute probe: top-1 value among the attribute's values (cosine to value prompts); hit if it is among the annotated values of the image's group
    AM = [str(m).split("|") for m in TL["attr_meta"]]; AE = torch.tensor(TL["attr_emb"], device=dev); attr_res = {}
    trn = tr
    for a in ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]:
        ix = [i for i, (aa, v) in enumerate(AM) if aa == a]; names = [AM[i][1] for i in ix]; sets = [set(s for s in A[a].iloc[k].split(";") if s and s != "none") for k in te]; ok = np.array([len(s) > 0 for s in sets])
        if ok.sum() == 0: continue
        r = dict(n=int(ok.sum()))
        for lab, H in (("zs", Hte0), ("tuned", Hte1)):
            pred = (H @ AE[ix].T).argmax(1).cpu().numpy(); r[lab] = float(np.mean([names[p] in sets[k] for k, p in enumerate(pred) if ok[k]]))
        cntv = pd.Series([v for k in trn for v in A[a].iloc[k].split(";") if v and v != "none"]).value_counts(); r["prior"] = float(np.mean([cntv.index[0] in sets[k] for k in range(len(te)) if ok[k]])); r["chance"] = float(np.mean([len(sets[k]) / len(names) for k in range(len(te)) if ok[k]])); attr_res[a] = r
    ext["attr_probe"] = attr_res
    print(f"FOLD {g.fold} zero-shot test unit/img {t0[0]:.3f}/{t0[1]:.3f} | best val step {best[1]} val {best[0]:.3f} -> test unit/img {best[2][0]:.3f}/{best[2][1]:.3f}"); print("EXT", json.dumps({k: v for k, v in ext.items() if k != "attr_probe"}), flush=True)
    json.dump(dict(args=vars(g), hist=hist, best_step=best[1], test_zs=t0[:2], test_best=best[2][:2], ext=ext), open(res / f"v6lora3_{g.tag}_fold{g.fold}.json", "w"))
if g.dg_full or g.fold < 0:
    dg = pd.read_csv(DATA / "external/digigreen/annotations.csv"); dg = dg[dg.diagnosis.fillna("").str.count(";") == 0].copy(); dg["diag"] = dg.diagnosis.str.strip().str.lower(); dg = dg[dg.diag != "healthy"]
    dg["crop_l"] = dg.crop.str.strip().str.lower(); dg["type"] = dg.crop_l + "|" + dg.diag; dg = dg[dg.type.isin(DGT)].copy(); dg["crop_a"] = dg.crop_l.map(lambda c: ALIAS.get(c, c))
    cnt_t = dg.type.value_counts(); dg = dg[dg.type.map(cnt_t) >= 3]; nt = dg.groupby("crop_a").type.nunique(); dg = dg[dg.crop_a.map(nt) >= 2].reset_index(drop=True); types = sorted(dg.type.unique()); print("DG-full images", len(dg), "types", len(types), "crops", dg.crop_a.nunique(), flush=True)
    files = [str(DATA / "external/digigreen" / f) for f in dg.image_file]; ti = {t: DGT.index(t) for t in types}; matched = {t: any(matches(u, t) for u in UN) for t in types}
    Hg1 = enc_ds(PathDS(files)); load_state(snap()) ; Hg0 = None
    # zero-shot features: re-create a zero-adapter pass by temporarily zeroing LoRA B (B=0 => frozen model)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.zero_()
    Hg0 = enc_ds(PathDS(files)); ls0 = float(np.log(scale0)); b0 = bias0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.copy_(saved[n])
    dgcrop = dg.crop_a.to_numpy(); dgtype = dg.type.to_numpy(); pw_crop = unit_crop; dg_tcrop = np.array([t.split("|")[0] for t in DGT]); dg_tcrop = np.array([ALIAS.get(c, c) for c in dg_tcrop])
    def dg_scores(H, st, scale, bias, variant):
        Tp = TL[f"pw_{st}"]; Td = TL[f"dg_{st}"]; Sd = scale * (H @ torch.tensor(Td, device=dev).T).cpu().numpy() + bias; out = np.full((len(dg), len(DGT)), -1e18)
        mk = np.zeros((len(dg), len(DGT)), bool)
        for i, c in enumerate(dgcrop): mk[i] = np.array([(dg_tcrop[k] == c) and (DGT[k] in types) for k in range(len(DGT))])
        res_ = {"dg_only": (np.where(mk, Sd, -1e18), None)}
        Sp = scale * (H @ torch.tensor(Tp, device=dev).T).cpu().numpy() + bias; mp = np.array([[unit_crop[k] == c for k in range(len(UN))] for c in dgcrop])
        res_["dg+pw"] = (np.where(mk, Sd, -1e18), np.where(mp, Sp, -1e18)); return res_[variant]
    tab = {}
    for st in ("long", "tags", "name"):
        for lab, H, sc_, bi in (("zs", Hg0, np.exp(ls0), b0), ("tuned", Hg1, float(ls.exp()), float(bb))):
            for variant in ("dg_only", "dg+pw"):
                Sd, Sp = dg_scores(H, st, sc_, bi, variant); S = Sd if Sp is None else np.concatenate([Sd, Sp], 1); true_col = np.array([DGT.index(t) for t in dgtype]); rank = (S > S[np.arange(len(dg)), true_col][:, None]).sum(1)
                per = pd.DataFrame(dict(type=dgtype, top1=(rank < 1), top3=(rank < 3), m=[matched[t] for t in dgtype]))
                tab[f"{st}_{lab}_{variant}"] = dict(types_macro_top1=float(per.groupby("type").top1.mean().mean()), types_macro_top3=float(per.groupby("type").top3.mean().mean()), img_top1=float(per.top1.mean()),
                                                      unmatched_types_macro_top1=float(per[~per.m].groupby("type").top1.mean().mean()) if (~per.m).any() else None, matched_types_macro_top1=float(per[per.m].groupby("type").top1.mean().mean()) if per.m.any() else None)
                if st == "long" and variant == "dg_only": per_t = per.groupby("type").agg(n=("top1", "size"), top1=("top1", "mean"), matched=("m", "first")); per_t.to_csv(res / f"v6lora3_{g.tag}_dgfull_{lab}_pertype.csv")
    dgres = dict(n_images=len(dg), n_types=len(types), n_crops=int(dg.crop_a.nunique()), n_unmatched_types=int(sum(not v for v in matched.values())), table=tab, args=vars(g)); json.dump(dgres, open(res / f"v6lora3_{g.tag}_dgfull.json", "w"), indent=1)
    print("DGFULL", json.dumps({k: (round(v["types_macro_top1"], 3), round(v["img_top1"], 3)) for k, v in tab.items()}), flush=True)
