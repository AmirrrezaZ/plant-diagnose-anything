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
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--tag", default="lora"); ap.add_argument("--cluster_w", type=float, default=0.5); ap.add_argument("--styles", default="long,tags"); ap.add_argument("--patience", type=int, default=3); ap.add_argument("--targets", default="qkvo"); ap.add_argument("--ens", type=int, default=0); g = ap.parse_args()
dev = "cuda"; torch.manual_seed(g.seed); rng = np.random.default_rng(g.seed); RES = 256
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False); A = A.merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = list(Tz["units"]); UI = {u: i for i, u in enumerate(UN)}; scale0, bias0 = float(Tz["scale"]), float(Tz["bias"])
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr]); unit_crop = np.array([A[A.disease_unit == u].crop.iloc[0] for u in UN]); crop = A.crop.to_numpy(); CM = crop[:, None] == unit_crop[None]
ncrop = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncrop[UI[u]] >= 2 if u in UI else False for u in unit_arr])
CLU = list(Tz["clusters"]); CI = {c: i for i, c in enumerate(CLU)}; clus_of = np.array([CI.get(c, -1) for c in A.group_id]); clus_unit = np.array([UI[u] for u in Tz["cluster_unit"]])
role = A[f"cv{g.fold}"].to_numpy(); tr = np.where((role == "train") & nh)[0]; va = np.where((role == "val") & nh & multi)[0]; te = np.where((role == "test") & nh & multi)[0]
train_units = sorted(set(tgt[tr])); tu_mask = np.zeros(len(UN), bool); tu_mask[train_units] = True
TXT = {k: torch.tensor(Tz["unit_" + k].astype(np.float32), device=dev) for k in ("long", "short", "tags", "name")}; CLT = {k: torch.tensor(Tz["cluster_" + k].astype(np.float32), device=dev) for k in ("long", "short", "tags")}
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
def encode(idx):
    model.eval(); H = []
    with torch.no_grad():
        for x, _ in DataLoader(DS(idx, False), batch_size=96, num_workers=g.workers): H.append(feats(x.to(dev)))
    model.train(); return torch.cat(H)
def evaluate(idx, H):
    S = (ls.exp() * H @ TXT["long"].T + bb).detach().cpu().numpy()
    if g.ens: S = S + (ls.exp() * H @ TXT["name"].T + bb).detach().cpu().numpy()
    S = np.where(CM[idx], S, -1e18); hit = (S.argmax(1) == tgt[idx]); u = pd.Series(hit).groupby(unit_arr[idx]).mean(); return float(u.mean()), float(hit.mean()), hit
# balanced sampling over train units (sqrt-inverse)
cnt = pd.Series(unit_arr[tr]).value_counts(); w = 1 / np.sqrt(pd.Series(unit_arr[tr]).map(cnt).to_numpy(float)); w /= w.sum()
styles = g.styles.split(",")
class TrainStream(Dataset):
    def __init__(s): s.draw = rng.choice(tr, g.steps * g.bs, p=w); s.tf = DS(tr, True).tf
    def __len__(s): return len(s.draw)
    def __getitem__(s, i): j = int(s.draw[i]); im = Image.open(DATA / paths[j]); im.draft("RGB", (RES * 2, RES * 2)); return s.tf(im.convert("RGB")), j
Hva0, Hte0 = encode(va), encode(te); v0 = evaluate(va, Hva0); t0 = evaluate(te, Hte0); hist = [dict(step=0, val_unit=v0[0], val_img=v0[1], test_unit=t0[0], test_img=t0[1])]; print("step 0 (zero-shot)", hist[-1], flush=True)
best = (v0[0], 0, t0, t0[2]); bad = 0; model.train(); t_start = time.time(); loader = iter(DataLoader(TrainStream(), batch_size=g.bs, num_workers=g.workers, shuffle=False, drop_last=True)); lossm = []
for step in range(1, g.steps + 1):
    x, j = next(loader); x = x.to(dev); j = j.to(dev); h = feats(x); loss = 0
    for st in styles:
        lg = ls.exp() * h @ TXT[st].T + bb; lg = lg.masked_fill(~(CMt[j] & tu_t[None]), -1e4); loss = loss + F.cross_entropy(lg, tgt_t[j])
        if g.cluster_w > 0 and st != "name":
            lc = ls.exp() * h @ CLT[st].T + bb; ok = CMt[j][:, cu_t] & tu_t[cu_t][None]; lc = lc.masked_fill(~ok, -1e4); loss = loss + g.cluster_w * F.cross_entropy(lc, cl_t[j])
    loss = loss / len(styles); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); lossm.append(float(loss))
    if step % g.eval_every == 0 or step == g.steps:
        Hv = encode(va); v = evaluate(va, Hv); Ht = encode(te); t = evaluate(te, Ht); hist.append(dict(step=step, loss=float(np.mean(lossm[-g.eval_every:])), val_unit=v[0], val_img=v[1], test_unit=t[0], test_img=t[1])); print(hist[-1], f"{time.time() - t_start:.0f}s", flush=True)
        if v[0] > best[0] + 1e-4: best = (v[0], step, t, t[2]); bad = 0
        else:
            bad += 1
            if bad >= g.patience: break
print(f"FOLD {g.fold} zero-shot test unit/img {t0[0]:.3f}/{t0[1]:.3f} | best val step {best[1]} val {best[0]:.3f} -> test unit/img {best[2][0]:.3f}/{best[2][1]:.3f}")
(ROOT / "results/v3").mkdir(exist_ok=True); json.dump(dict(args=vars(g), hist=hist, best_step=best[1], test_zs=t0[:2], test_best=best[2][:2]), open(ROOT / f"results/v3/v6lora_{g.tag}_fold{g.fold}.json", "w"))
pd.DataFrame(dict(unit=unit_arr[te], zs=t0[2].astype(float), tuned=best[3].astype(float), final=t[2].astype(float))).to_csv(ROOT / f"results/v3/v6lora_{g.tag}_fold{g.fold}_units.csv", index=False)
