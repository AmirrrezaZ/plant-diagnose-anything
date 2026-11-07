"""v6 text-side adaptation on CACHED frozen SigLIP2-L-256 image features (no image encoder training).
The text encoder (SigLIP2 text tower with LoRA / last layers unfrozen, or a sentence-transformer + projection) is trained so that an image's feature matches the description of its disease unit, softmax over same-crop TRAIN units.
Descriptions are re-sampled every step from the unit profile (attribute dropout, value subsets, template choice) so the encoder cannot memorise 100 fixed strings. Optional attribute-swap HARD NEGATIVES:
a copy of the description with one attribute value replaced by a value the unit does not have; it is a candidate only for the images of its own unit and must score below the true description.
Selection on val units; test units of the CV fold are scored with canonical image-derived (A) and blind LLM (L) profiles at the best val step and at step 0.
  python v6_text_ft.py --fold 0 --mode lora --hardneg 2 --train_text M --tag T1        (fold -1: train on all PW units not matching DigiGreen types, evaluate on DG-full with fixed --steps)"""
import argparse, re, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--mode", default="lora", help="frozen|lora|last|st_proj|st_last"); ap.add_argument("--st_model", default="all-mpnet-base-v2")
ap.add_argument("--steps", type=int, default=400); ap.add_argument("--eval_every", type=int, default=25); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--hardneg", type=int, default=0); ap.add_argument("--hn_w", type=float, default=1.0)
ap.add_argument("--train_text", default="M", help="A image-derived | L blind LLM | M both (random per sample)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="T"); ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--ncrop", type=int, default=6); ap.add_argument("--per_crop", type=int, default=8); ap.add_argument("--noaug", type=int, default=0); ap.add_argument("--init", default="distill"); ap.add_argument("--save", type=int, default=0); ap.add_argument("--nlast", type=int, default=4); g = ap.parse_args()
dev = "cuda"; torch.manual_seed(g.seed); rng = np.random.default_rng(g.seed)
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; SW = ATT[1:]
A = pd.read_csv(DATA / "annotation/annotations_v6.csv", keep_default_na=False).merge(pd.read_csv(DATA / "split/splits_cv.csv"), on="img_id")
Z = np.load(ROOT / "results/cache/embeddings_siglip2l256_all.npz", allow_pickle=True); pos = {i: k for k, i in enumerate(Z["img_id"])}; X = torch.tensor(Z["feats"][[pos[i] for i in A.img_id]], device=dev); X = F.normalize(X, dim=-1)
Tz = np.load(ROOT / "results/cache/v6_text.npz", allow_pickle=True); UN = [str(u) for u in Tz["units"]]; UI = {u: i for i, u in enumerate(UN)}; scale0, bias0 = float(Tz["scale"]), float(Tz["bias"])
TL = np.load(ROOT / "results/cache/v6_text_llm.npz", allow_pickle=True); DGT = [str(x) for x in TL["dg_types"]]; P = json.load(open(DATA / "profiles/llm_profiles.json")); VOC = P["vocab"]
nh = (A.disease_type != "healthy").to_numpy(); unit_arr = A.disease_unit.to_numpy(); tgt = np.array([UI.get(u, -1) for u in unit_arr]); crop = A.crop.to_numpy()
unit_crop = np.array([u.split("|")[0] for u in UN]); CM = crop[:, None] == unit_crop[None]; ncrop_u = pd.Series(unit_crop).map(pd.Series(unit_crop).value_counts()).to_numpy(); multi = np.array([ncrop_u[UI[u]] >= 2 if u in UI else False for u in unit_arr])
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
# ---------- profiles and texts
def prof_img(gdf, share=0.25):
    out = {}
    for a in ATT:
        cnt = {}
        for s in gdf[a]:
            for t in [t for t in s.split(";") if t]: cnt[t] = cnt.get(t, 0) + 1
        out[a] = [t.replace("_", " ") for t, k in sorted(cnt.items(), key=lambda x: -x[1]) if k / len(gdf) >= share and t != "none"]
    return out
PA = {u: prof_img(A[(A.disease_unit == u) & nh]) for u in UN}; PL = {u: {a: [t.replace("_", " ") for t in P["profiles"][u.lower()][a]] for a in ATT} for u in UN}
PDG = {k: {a: [t.replace("_", " ") for t in P["profiles"][k][a]] for a in ATT} for k in DGT}
VOCs = {a: [v.replace("_", " ") for v in VOC[a]] for a in ATT}
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
def swap_state(q, r):
    cand = [a for a in SW if q[a]]
    if not cand: return None
    a = cand[int(r.integers(len(cand)))]; alt = [v for v in VOCs[a] if v not in q[a]]
    if not alt: return None
    q2 = {k: list(v) for k, v in q.items()}; q2[a] = [alt[int(r.integers(len(alt)))]]; return q2
def unit_profile(u, src, r):
    return PA[u] if (src == "A" or (src == "M" and r.random() < 0.5)) else PL[u]
# ---------- text encoders
tsl = None
if g.mode.startswith("st"):
    from transformers import AutoModel, AutoTokenizer
    SP = ROOT / "models" / g.st_model
    stok = AutoTokenizer.from_pretrained(SP); stm = AutoModel.from_pretrained(SP, dtype=torch.float32).to(dev); pool = "cls" if "bge" in g.st_model else "mean"; pre = "query: " if "e5" in g.st_model else ""
    proj = nn.Linear(stm.config.hidden_size, 1024).to(dev)
    for p_ in stm.parameters(): p_.requires_grad = False
    if g.mode == "st_last":
        layers = stm.encoder.layer
        for l in layers[-2:]:
            for p_ in l.parameters(): p_.requires_grad = True
    def enc_raw(texts, grad):
        tk = stok([pre + t.lower() for t in texts], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            h = stm(**tk).last_hidden_state
        h = h.float(); m = tk["attention_mask"].unsqueeze(-1).float(); return h[:, 0] if pool == "cls" else (h * m).sum(1) / m.sum(1)
    def enc(texts, grad=False): return F.normalize(proj(enc_raw(texts, grad)), dim=-1)
    params = [p_ for p_ in stm.parameters() if p_.requires_grad] + list(proj.parameters())
if g.mode in ("frozen", "lora", "last") or g.mode.startswith("st"):
    from transformers import AutoModel, AutoProcessor
    sn = "google/siglip2-large-patch16-256"; proc = AutoProcessor.from_pretrained(sn); sm = AutoModel.from_pretrained(sn, dtype=torch.float32); sm.vision_model = None; sm = sm.to(dev)
    for p_ in sm.parameters(): p_.requires_grad = False
    def sig_raw(texts, grad):
        tk = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16): f = sm.get_text_features(**tk)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output; return F.normalize(f.float(), dim=-1)
    if g.mode == "lora":
        import peft; peft.inject_adapter_in_model(peft.LoraConfig(r=g.rank, lora_alpha=g.rank, lora_dropout=0.0, target_modules=r".*text_model.*(q_proj|k_proj|v_proj|out_proj)"), sm)
        for n, p_ in sm.named_parameters(): p_.requires_grad = "lora_" in n
    if g.mode == "last":
        for n, p_ in sm.named_parameters():
            if "text_model" in n and (re.search(rf"encoder\.layers\.({'|'.join(str(24 - k) for k in range(1, g.nlast + 1))})\.", n) or "final_layer_norm" in n or ".head." in n): p_.requires_grad = True
            if n.startswith("text_projection") or "text_projection" in n: p_.requires_grad = True
    if not g.mode.startswith("st"):
        enc = lambda texts, grad=False: sig_raw(texts, grad); params = [p_ for p_ in sm.parameters() if p_.requires_grad]
    print("trainable params", sum(p_.numel() for p_ in params), flush=True)
def unit_emb(u_list, kind, src, grad=False, extra=None):
    """canonical embedding (mean of the style's prompts, renormalised) for units; src 'A' image-derived, 'L' blind"""
    ts = []; n = []
    for u in u_list:
        c, d = u.split("|", 1); cs = canon(c, d, PA[u] if src == "A" else PL[u])[kind]; ts += cs; n.append(len(cs))
    E = enc(ts, grad); out = []; k = 0
    for m in n: e = E[k:k + m].mean(0); out.append(F.normalize(e, dim=0)); k += m
    return torch.stack(out)
def enc_batched(f, items, bs=256): return torch.cat([f(items[i:i + bs]) for i in range(0, len(items), bs)])
ls = nn.Parameter(torch.tensor(float(np.log(scale0)), device=dev)); bb = nn.Parameter(torch.tensor(bias0, device=dev))
# ---------- ST distillation init (text -> frozen SigLIP text space), train units' texts only
if g.mode.startswith("st") and g.init == "distill":
    ts = []
    for u in [UN[i] for i in train_units]:
        c = u.split("|")[0]
        for _ in range(30):
            src = "A" if rng.random() < 0.5 else "L"; ts.append((c, sample_state(PA[u] if src == "A" else PL[u], rng), int(rng.integers(5))))
    texts = [render(c, q, t) for c, q, t in ts]
    with torch.no_grad(): Hs = torch.cat([sig_raw(texts[i:i + 128], False) for i in range(0, len(texts), 128)]); Hr = torch.cat([enc_raw(texts[i:i + 128], False) for i in range(0, len(texts), 128)])
    Hr1 = torch.cat([Hr, torch.ones(len(Hr), 1, device=dev)], 1); W = torch.linalg.solve(Hr1.T @ Hr1 + 1.0 * torch.eye(Hr1.shape[1], device=dev), Hr1.T @ Hs)
    with torch.no_grad(): proj.weight.copy_(W[:-1].T); proj.bias.copy_(W[-1])
    print("distill cos", float(F.cosine_similarity(F.normalize(Hr1 @ W, dim=-1), Hs).mean()), flush=True)
# ---------- evaluation
def unit_macro(S, idx): S = np.where(CM[idx], S, -1e18); hit = S.argmax(1) == tgt[idx]; return float(pd.Series(hit).groupby(unit_arr[idx]).mean().mean())
def score(idx, T): return (ls.exp() * X[idx] @ T.T + bb).detach().cpu().numpy()
def eval_kinds(idx, kinds):
    out = {}
    with torch.no_grad():
        for (src, kind) in kinds:
            T = enc_batched(lambda us: unit_emb(us, kind, src), UN, 64); out[f"{src}_{kind}"] = unit_macro(score(idx, T), idx)
    return out
KINDS = [("A", "long"), ("L", "long"), ("L", "tags"), ("L", "name"), ("A", "name")]
selk = {"A": [("A", "long")], "L": [("L", "long")], "M": [("A", "long"), ("L", "long")]}[g.train_text]
sel = lambda r: float(np.mean([r[f"{s}_{k}"] for s, k in selk]))
# ---------- training
cnt = pd.Series(unit_arr[tr]).value_counts(); wimg = 1 / np.sqrt(pd.Series(unit_arr[tr]).map(cnt).to_numpy(float))
crops_tr = sorted({c for c in unit_crop[train_units] if (unit_crop[train_units] == c).sum() >= 2}); tr_by_crop = {c: np.where(crop[tr] == c)[0] for c in crops_tr}
snap = lambda: ({n: p_.detach().cpu().clone() for n, p_ in (list(sm.named_parameters()) if not g.mode.startswith("st") else list(stm.named_parameters()) + list(proj.named_parameters())) if p_.requires_grad}, float(ls), float(bb))
def load(st):
    sd, l, b = st; named = dict(sm.named_parameters()) if not g.mode.startswith("st") else {**dict(stm.named_parameters()), **dict(proj.named_parameters())}
    with torch.no_grad():
        for n, v in sd.items(): named[n].copy_(v.to(dev))
        ls.copy_(torch.tensor(l)); bb.copy_(torch.tensor(b))
hist = []; best = (-1, 0, None); res = ROOT / "results/v3"; res.mkdir(exist_ok=True)
if len(te):
    r0 = eval_kinds(va, selk); r0t = eval_kinds(te, KINDS); hist.append(dict(step=0, val=sel(r0), test=r0t)); best = (sel(r0), 0, r0t); print("step 0", hist[-1], flush=True)
best_state = snap(); bad = 0; opt = torch.optim.AdamW([dict(params=params, lr=g.lr, weight_decay=0.01), dict(params=[ls, bb], lr=1e-4, weight_decay=0)])
if g.mode == "frozen": g.steps = 0
wu = max(1, int(0.1 * g.steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / wu if t < wu else 0.5 * (1 + np.cos(np.pi * (t - wu) / max(1, g.steps - wu))))
t0 = time.time(); lossm = []
for step in range(1, g.steps + 1):
    cs = rng.choice(crops_tr, size=min(g.ncrop, len(crops_tr)), replace=False); bi = []
    for c in cs:
        ii = tr_by_crop[c]; p = wimg[ii] / wimg[ii].sum(); bi += list(tr[rng.choice(ii, size=g.per_crop, p=p)])
    bi = np.array(bi); units_b = sorted({int(u) for u in tgt[bi]} | {i for i in train_units if unit_crop[i] in set(cs)}); texts = []; owner = []; kind = []
    for u in units_b:
        uname = UN[u]; c = uname.split("|")[0]
        if g.noaug: src = "A" if g.train_text == "A" else "L" if g.train_text == "L" else ("A" if rng.random() < .5 else "L"); texts.append(canon(c, uname.split("|", 1)[1], PA[uname] if src == "A" else PL[uname])["long"][int(rng.integers(2))]); owner.append(u); kind.append(0); continue
        pr = unit_profile(uname, g.train_text, rng); q = sample_state(pr, rng); t = int(rng.integers(5)); texts.append(render(c, q, t)); owner.append(u); kind.append(0)
        for _ in range(g.hardneg):
            q2 = swap_state(q, rng)
            if q2 is not None: texts.append(render(c, q2, t)); owner.append(u); kind.append(1)
    E = enc(texts, True); owner = np.array(owner); kind = np.array(kind); h = X[bi]
    lg = ls.exp() * h @ E.T + bb; ow = torch.tensor(owner, device=dev); tg = torch.tensor(tgt[bi], device=dev); bc = torch.tensor(crop[bi] == "", device=dev)
    ok_pos = (torch.tensor(kind == 0, device=dev)[None] & torch.tensor(CM[bi][:, owner], device=dev)); ok_neg = (torch.tensor(kind == 1, device=dev)[None] & (ow[None] == tg[:, None]))
    lg = lg.masked_fill(~(ok_pos | ok_neg), -1e4); ti = torch.tensor([int(np.where((owner == t_) & (kind == 0))[0][0]) for t_ in tgt[bi]], device=dev)
    loss = F.cross_entropy(lg, ti); lossm.append(float(loss))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % g.eval_every == 0 or step == g.steps:
        if len(te):
            rv = eval_kinds(va, selk); rt = eval_kinds(te, KINDS); hist.append(dict(step=step, loss=float(np.mean(lossm[-g.eval_every:])), val=sel(rv), test=rt)); print(hist[-1], f"{time.time() - t0:.0f}s", flush=True)
            if sel(rv) > best[0] + 1e-4: best = (sel(rv), step, rt); best_state = snap(); bad = 0
            else:
                bad += 1
                if bad >= g.patience: break
        else: print({"step": step, "loss": float(np.mean(lossm[-g.eval_every:]))}, f"{time.time() - t0:.0f}s", flush=True)
if not len(te): best_state = snap()
out = dict(args=vars(g), hist=hist, best_step=best[1], best_val=best[0])
if len(te):
    print(f"FOLD {g.fold} step0 test {hist[0]['test']} | best step {best[1]} val {best[0]:.3f} test {best[2]}", flush=True); out.update(test_zs=hist[0]["test"], test_best=best[2])
    if g.mode.startswith("st"):  # score-level fusion with the frozen SigLIP text embedding; weight chosen on val
        load(best_state); fus = {}
        for src, kind in (("A", "long"), ("L", "long"), ("L", "tags"), ("L", "name")):
            cz = {"A_long": Tz["unit_long"], "L_long": TL["pw_long"], "L_tags": TL["pw_tags"], "L_name": TL["pw_name"]}[f"{src}_{kind}"]; Tz_ = torch.tensor(cz.astype(np.float32), device=dev)
            with torch.no_grad(): Ts = enc_batched(lambda us: unit_emb(us, kind, src), UN, 64)
            zs_v = (scale0 * X[va] @ Tz_.T).cpu().numpy(); st_v = score(va, Ts); zs_t = (scale0 * X[te] @ Tz_.T).cpu().numpy(); st_t = score(te, Ts)
            ws = [0, .25, .5, 1, 2, 4]; vs = [unit_macro(zs_v + w * st_v, va) for w in ws]; w = ws[int(np.argmax(vs))]; fus[f"{src}_{kind}"] = dict(w=w, val=max(vs), test=unit_macro(zs_t + w * st_t, te), zs_test=unit_macro(zs_t, te))
        out["fused"] = fus; print("FUSED", json.dumps(fus), flush=True)
    json.dump(out, open(res / f"v6tft_{g.tag}_fold{g.fold}.json", "w"))
else:  # DigiGreen-full evaluation
    load(best_state); dg = pd.read_csv(DATA / "external/digigreen/annotations.csv"); dg = dg[dg.diagnosis.fillna("").str.count(";") == 0].copy(); dg["diag"] = dg.diagnosis.str.strip().str.lower(); dg = dg[dg.diag != "healthy"]
    dg["crop_l"] = dg.crop.str.strip().str.lower(); dg["type"] = dg.crop_l + "|" + dg.diag; dg = dg[dg.type.isin(DGT)].copy(); dg["crop_a"] = dg.crop_l.map(lambda c: ALIAS.get(c, c))
    cnt_t = dg.type.value_counts(); dg = dg[dg.type.map(cnt_t) >= 3]; nt = dg.groupby("crop_a").type.nunique(); dg = dg[dg.crop_a.map(nt) >= 2].reset_index(drop=True); types = sorted(dg.type.unique())
    Dz = np.load(ROOT / "results/cache/emb_digigreen_l256.npz", allow_pickle=True); dp = {str(f): k for k, f in enumerate(Dz["files"])}; Xd = F.normalize(torch.tensor(Dz["feats"][[dp[Path(f).name] if Path(f).name in dp else dp[f] for f in dg.image_file]], device=dev), dim=-1)
    ti_ = {t: DGT.index(t) for t in types}; cm = (dg.crop_a.to_numpy()[:, None] == np.array([ALIAS.get(t.split("|")[0], t.split("|")[0]) for t in types])[None]); y = np.array([types.index(t) for t in dg.type]); matched = {t: any(matches(u, t) for u in UN) for t in types}
    def dg_eval(f_enc):
        r = {}
        for kind in ("long", "tags", "name"):
            with torch.no_grad():
                ts = []; n = []
                for t in types: c, d = t.split("|", 1); cs = canon(c, d, PDG[t])[kind]; ts += cs; n.append(len(cs))
                E = enc_batched(f_enc, ts, 128); T = torch.stack([F.normalize(E[sum(n[:i]):sum(n[:i]) + m].mean(0), dim=0) for i, m in enumerate(n)])
            S = np.where(cm, (ls.exp() * Xd @ T.T + bb).detach().cpu().numpy(), -1e18); hit = S.argmax(1) == y; per = pd.Series(hit).groupby(dg.type.to_numpy()).mean(); r[kind] = dict(type_macro=float(per.mean()), img=float(hit.mean()), per_type=per.to_dict())
        return r
    out["dg_tuned"] = dg_eval(lambda t: enc(t, False)); print("DG tuned", {k: round(v["type_macro"], 4) for k, v in out["dg_tuned"].items()}, flush=True)
    json.dump(out, open(res / f"v6tft_{g.tag}_fold{g.fold}_s{g.seed}.json", "w"))
if g.save and len(te) == 0 or g.save and len(te): torch.save(dict(state=best_state[0], args=vars(g), best_step=best[1]), ROOT / "checkpoints/v6" / f"{g.tag}_fold{g.fold}_s{g.seed}.pt")
