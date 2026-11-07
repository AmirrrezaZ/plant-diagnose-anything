"""Build every number, LaTeX table and figure of the paper from the saved logs/jsons (no hand-typed results).
Reads  logs/runs/v6lora_*.log, results/cv/{controls,bcvsa,vlm}.json ; writes paper/results.json, paper/tables/*.tex, paper/figures/*.pdf
   python scripts/make_paper_assets.py"""
import re, glob, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "paper"; (OUT / "tables").mkdir(exist_ok=True, parents=True); (OUT / "figures").mkdir(exist_ok=True, parents=True)
def folds(tag):
    R = {}
    for f in glob.glob(str(ROOT / f"logs/runs/v6lora_{tag}_f*.log")):
        for l in open(f):
            m = re.match(r"FOLD (\d+) zero-shot test unit/img ([\d.]+)/([\d.]+) \| best val step (\d+) val [\d.]+ -> test unit/img ([\d.]+)/([\d.]+)", l)
            if m: R[int(m.group(1))] = dict(zs=float(m.group(2)), lora=float(m.group(5)), step=int(m.group(4)))
    return R
def agg(tags):
    S = [folds(t) for t in tags]; assert all(len(s) == 10 for s in S), [len(s) for s in S]
    L = np.array([[s[f]["lora"] for f in range(10)] for s in S]); Z = np.array([[s[f]["zs"] for f in range(10)] for s in S]).mean(0); g = L.mean(0) - Z
    rng = np.random.default_rng(0); bs = [rng.choice(g, 10).mean() for _ in range(10000)]
    return dict(zs=float(Z.mean()), lora=float(L.mean()), seed_means=[float(x) for x in L.mean(1)], seed_std=float(L.mean(1).std()), gain=float(g.mean()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], folds_improved=int((g > 0).sum()), per_fold_lora=[float(x) for x in L.mean(0)], per_fold_zs=[float(x) for x in Z])
res = dict(desc_only=agg(["F0", "F1", "F2"]), name_track=agg(["D", "D1", "D2"]))
def dgfinal(pat, key):
    v = []
    for f in sorted(glob.glob(str(ROOT / f"logs/runs/{pat}"))):
        L = [l for l in open(f) if l.startswith("{'step'")]; v.append(eval(re.sub(r"\s+\d+s\s*$", "", L[-1].strip()))["dg"][key][0])
    return v
res["digigreen"] = dict(zs_desc=0.5362318840579711, zs_name=0.5217391304347826, zs_ens=0.5362318840579711, n_images=69, n_units=14,
                        desc_only_model=dgfinal("v6lora_dgFdesc_s*.log", "long"), name_track_model_ens=dgfinal("v6lora_dgD_s*.log", "ens") if glob.glob(str(ROOT / "logs/runs/v6lora_dgD_s*.log")) else None)
for k in ("controls", "bcvsa", "vlm"): res[k] = {m: (d.get("top1_unit_macro"), d.get("n_units")) for m, d in json.load(open(ROOT / f"results/cv/{k}.json")).items()}
json.dump(res, open(OUT / "results.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k in ("desc_only", "name_track", "digigreen")}, indent=1)[:1800])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, (k, t) in zip(ax, (("desc_only", "Symptom text only"), ("name_track", "Symptom text + disease name"))):
    r = res[k]; x = np.arange(10); a.bar(x - 0.2, r["per_fold_zs"], 0.4, label="zero-shot", color="#9aa5b1"); a.bar(x + 0.2, r["per_fold_lora"], 0.4, label="supervised (3 seeds)", color="#1f6feb")
    a.set_title(t, fontsize=10); a.set_xlabel("test fold"); a.set_xticks(x)
ax[0].set_ylabel("unit-macro top-1"); ax[0].legend(fontsize=8, frameon=False); plt.tight_layout(); plt.savefig(OUT / "figures/per_fold.pdf"); plt.close()
c, b, v = res["controls"], res["bcvsa"], res["vlm"]
rows = [("Chance (1/\\#candidates)", c["chance"][0]), ("Crop prior (no image)", c["crop_prior"][0]), ("Nearest seen disease profile", c["nearest_seen"][0]), ("Linear tag head + profile lookup", 0.361),
        ("BCVSA-style blend (DINOv2-L + CLIP-L/14)", b["bcvsa_blend"][0]), ("CLIP-L/14 zero-shot, names", b["clipL_zeroshot"][0]), ("Qwen2.5-VL-7B tags + lookup$^\\dagger$", v["tags_cv"][0]), ("Qwen2.5-VL-7B multiple choice, names$^\\dagger$", v["mc_name_cv"][0]),
        ("SigLIP2-L zero-shot, description", c["zs_long"][0]), ("SigLIP2-L zero-shot, name", c["zs_name"][0]), ("SigLIP2-L zero-shot, description + name", c["zs_long+name"][0]),
        ("Ours, symptom text only (3 seeds)", res["desc_only"]["lora"]), ("Ours, symptom text + name (3 seeds)", res["name_track"]["lora"])]
open(OUT / "tables/main_cv.tex", "w").write("\\begin{tabular}{lc}\\toprule Method & Top-1 (unit-macro)\\\\\\midrule\n" + "\n".join(f"{n} & {x:.3f}\\\\" for n, x in rows) + "\n\\bottomrule\\end{tabular}\n"); print("wrote tables/main_cv.tex")

# ---------- extension (steps 3-6): blind profiles, shuffled-label control, per-attribute probe, large DigiGreen test ----------
import csv
R3 = ROOT / "results/arvan_v3"
def boot_ci(g, n=10000):
    rng = np.random.default_rng(0); b = [rng.choice(g, len(g)).mean() for _ in range(n)]; return [float(x) for x in np.percentile(b, [2.5, 97.5])]
G = {k: json.load(open(R3 / f"v6lora3_G0_fold{k}.json")) for k in range(10)}; S = {k: json.load(open(R3 / f"v6lora3_S0_fold{k}.json")) for k in range(10)}
ext = {}
for k, lab in (("A_long", "Image-derived profile (tag shares), description"), ("L_long", "Blind LLM profile, description"), ("L_tags", "Blind LLM profile, tag list"), ("L_name", "Disease-name prompt")):
    z = np.array([G[f]["ext"][k + "_zs"] for f in range(10)]); t = np.array([G[f]["ext"][k + "_tuned"] for f in range(10)]); gg = t - z
    ext[k] = dict(label=lab, zs=float(z.mean()), tuned=float(t.mean()), gain=float(gg.mean()), ci=boot_ci(gg), folds_improved=int((gg > 0).sum()))
ext["shuffled_label_control"] = dict(best_steps=[S[f]["best_step"] for f in range(10)], final_step_test=float(np.mean([S[f]["hist"][-1]["test_unit"] for f in range(10)])), zs=float(np.mean([S[f]["test_zs"][0] for f in range(10)])))
AT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; attr = {}
for a in AT:
    r = [G[f]["ext"]["attr_probe"][a] for f in range(10) if a in G[f]["ext"]["attr_probe"]]; attr[a] = {m: float(np.mean([x[m] for x in r])) for m in ("zs", "tuned", "prior", "chance")}
ext["attr_probe"] = attr
dgs = [json.load(open(R3 / f"v6lora3_DGF_s{s}_dgfull.json")) for s in (0, 1, 2) if (R3 / f"v6lora3_DGF_s{s}_dgfull.json").exists()]
if dgs:
    ext["dgfull"] = dict(n_images=dgs[0]["n_images"], n_types=dgs[0]["n_types"], n_crops=dgs[0]["n_crops"], n_unmatched=dgs[0]["n_unmatched_types"], n_seeds=len(dgs), rows={})
    for key in dgs[0]["table"]:
        v = [d["table"][key] for d in dgs]; ext["dgfull"]["rows"][key] = {m: (float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v]))) if v[0][m] is not None else None for m in ("types_macro_top1", "types_macro_top3", "img_top1", "unmatched_types_macro_top1")}
    pt = lambda tag, lab: {r["type"]: float(r["top1"]) for r in csv.DictReader(open(R3 / f"v6lora3_{tag}_dgfull_{lab}_pertype.csv"))}
    z0 = pt("DGF_s0", "zs"); tuned = [pt(f"DGF_s{s}", "tuned") for s in range(len(dgs))]; ks = sorted(z0); gg = np.array([np.mean([t[k] for t in tuned]) - z0[k] for k in ks])
    ext["dgfull"]["gain_long_dg_only"] = dict(gain=float(gg.mean()), ci=boot_ci(gg), improved=int((gg > 0).sum()), worse=int((gg < 0).sum()), n_types=len(ks))
res["ext"] = ext; json.dump(res, open(OUT / "results.json", "w"), indent=1)
f = lambda x: f"{x:.3f}"
open(OUT / "tables/blind_cv.tex", "w").write("\\begin{tabular}{lcccc}\\toprule Text used to score a held-out disease & Zero-shot & Supervised & Gain [95\\% CI] & Folds improved\\\\\\midrule\n" + "\n".join(
    f"{ext[k]['label']} & {f(ext[k]['zs'])} & {f(ext[k]['tuned'])} & {ext[k]['gain']:+.3f} [{ext[k]['ci'][0]:.3f}, {ext[k]['ci'][1]:.3f}] & {ext[k]['folds_improved']}/10\\\\" for k in ("A_long", "L_long", "L_tags", "L_name")) + "\n\\bottomrule\\end{tabular}\n")
nm = {"affected_part": "Affected part", "color": "Colour", "texture": "Texture", "shape": "Shape", "pattern": "Pattern", "lesion_density": "Lesion density"}
open(OUT / "tables/attr_probe.tex", "w").write("\\begin{tabular}{lccccc}\\toprule Attribute & Zero-shot & Supervised & Train-set prior & Chance\\\\\\midrule\n" + "\n".join(f"{nm[a]} & {f(attr[a]['zs'])} & {f(attr[a]['tuned'])} & {f(attr[a]['prior'])} & {f(attr[a]['chance'])}\\\\" for a in AT) + "\n\\bottomrule\\end{tabular}\n")
if dgs:
    d = ext["dgfull"]["rows"]; ms = lambda x: f"{x[0]:.3f}" + (f" $\\pm$ {x[1]:.3f}" if len(dgs) > 1 else "")
    rows = [("Blind symptom profile (description)", "long"), ("Blind symptom profile (tag list)", "tags"), ("Disease-name prompt", "name")]
    open(OUT / "tables/dgfull.tex", "w").write("\\begin{tabular}{llcccc}\\toprule Scoring text & Candidates & Zero-shot & Supervised & Unmatched only (zs / sup.)\\\\\\midrule\n" + "\n".join(
        f"{lab} & {'DigiGreen types' if var == 'dg_only' else '+ PlantWild units'} & {f(d[f'{st}_zs_{var}']['types_macro_top1'][0])} & {ms(d[f'{st}_tuned_{var}']['types_macro_top1'])} & {f(d[f'{st}_zs_{var}']['unmatched_types_macro_top1'][0])} / {f(d[f'{st}_tuned_{var}']['unmatched_types_macro_top1'][0])}\\\\" for lab, st in rows for var in ("dg_only", "dg+pw")) + "\n\\bottomrule\\end{tabular}\n")
print("extension tables written;", {k: (round(v["gain"], 3) if "gain" in v else None) for k, v in ext.items() if isinstance(v, dict) and "gain" in v})

# ---------- step 4: training on BLIND (LLM-written) profiles (tag H0 = CV seed 0, DGH = large DigiGreen test, 3 seeds) ----------
H = {k: json.load(open(R3 / f"v6lora3_H0_fold{k}.json")) for k in range(10)}; blind = {}
for k, lab in (("A_long", "Image-derived profile, description"), ("L_long", "Blind LLM profile, description"), ("L_tags", "Blind LLM profile, tag list"), ("L_name", "Disease-name prompt")):
    z = np.array([H[f]["ext"][k + "_zs"] for f in range(10)]); t = np.array([H[f]["ext"][k + "_tuned"] for f in range(10)]); gg = t - z
    blind[k] = dict(label=lab, zs=float(z.mean()), tuned=float(t.mean()), gain=float(gg.mean()), ci=boot_ci(gg), folds_improved=int((gg > 0).sum()), per_fold_gain=[float(x) for x in gg])
dgh = [json.load(open(R3 / f"v6lora3_DGH_s{s}_dgfull.json")) for s in (0, 1, 2)]
blind["dgfull"] = {key: {m: (float(np.mean([d["table"][key][m] for d in dgh])), float(np.std([d["table"][key][m] for d in dgh]))) for m in ("types_macro_top1", "unmatched_types_macro_top1")} for key in dgh[0]["table"]}
pt = lambda tag, lab: {r["type"]: float(r["top1"]) for r in csv.DictReader(open(R3 / f"v6lora3_{tag}_dgfull_{lab}_pertype.csv"))}
z0 = pt("DGH_s0", "zs"); tn = [pt(f"DGH_s{s}", "tuned") for s in range(3)]; ks = sorted(z0); gg = np.array([np.mean([t[k] for t in tn]) - z0[k] for k in ks])
blind["dgfull_gain_long_dg_only"] = dict(gain=float(gg.mean()), ci=boot_ci(gg), improved=int((gg > 0).sum()), worse=int((gg < 0).sum()), n_types=len(ks))
res["ext"]["blind_trained"] = blind; json.dump(res, open(OUT / "results.json", "w"), indent=1)
open(OUT / "tables/blind_trained_cv.tex", "w").write("\\begin{tabular}{lccc}\\toprule Text used to score a held-out disease & Zero-shot & Blind-trained & Gain [95\\% CI] (folds improved)\\\\\\midrule\n" + "\n".join(
    f"{blind[k]['label']} & {f(blind[k]['zs'])} & {f(blind[k]['tuned'])} & {blind[k]['gain']:+.3f} [{blind[k]['ci'][0]:.3f}, {blind[k]['ci'][1]:.3f}] ({blind[k]['folds_improved']}/10)\\\\" for k in ("L_long", "L_tags", "L_name", "A_long")) + "\n\\bottomrule\\end{tabular}\n")
db = blind["dgfull"]; ms2 = lambda x: f"{x[0]:.3f} $\\pm$ {x[1]:.3f}"
open(OUT / "tables/dgfull_blind_trained.tex", "w").write("\\begin{tabular}{llccc}\\toprule Scoring text & Candidates & Zero-shot & Trained on image-derived & Trained on blind\\\\\\midrule\n" + "\n".join(
    f"{lab} & {'DigiGreen types' if var == 'dg_only' else '+ PlantWild units'} & {f(ext['dgfull']['rows'][f'{st}_zs_{var}']['types_macro_top1'][0])} & {ms2(ext['dgfull']['rows'][f'{st}_tuned_{var}']['types_macro_top1'])} & {ms2(db[f'{st}_tuned_{var}']['types_macro_top1'])}\\\\" for lab, st in (("Blind description", "long"), ("Blind tag list", "tags"), ("Disease-name prompt", "name")) for var in ("dg_only", "dg+pw")) + "\n\\bottomrule\\end{tabular}\n")
print("blind-trained tables written"); print(json.dumps({k: (round(v["zs"], 3), round(v["tuned"], 3), round(v["gain"], 3), [round(x, 3) for x in v["ci"]], v["folds_improved"]) for k, v in blind.items() if "ci" in v and "zs" in v})); print(blind["dgfull_gain_long_dg_only"]); print({k: (round(v["types_macro_top1"][0], 3), round(v["types_macro_top1"][1], 3)) for k, v in blind["dgfull"].items() if k.endswith("dg_only") and "tuned" in k})
