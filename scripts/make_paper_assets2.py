"""Tables from saved results: attr_transfer.tex (image->symptom head), text_side.tex (text-tower runs), dg_textlora.tex. Reads results/cv/attr_transfer.json and results/text_side/*.json."""
import json, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; T = ROOT / "paper/tables"; rng = np.random.default_rng(0)
S = json.load(open(ROOT / "results/cv/attr_transfer.json"))["summary"]
names = {"affected_part": "Affected part", "color": "Colour", "texture": "Texture", "shape": "Shape", "pattern": "Pattern", "lesion_density": "Lesion density"}
f = lambda ci: f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]"
rows = ["\\begin{tabular}{lccccccc}\\toprule Attribute & Zero-shot prompts & Most common & Same-crop prior & Trained head & Head $-$ crop prior & Head $-$ most common\\\\\\midrule".replace("lccccccc", "lcccccc")]
for a, n in names.items():
    r = S[a]; rows.append(f"{n} & {r['zs']:.3f} & {r['prior']:.3f} & {r['cropprior']:.3f} & {r['scratch']:.3f} & {f(r['scratch_minus_cropprior'])} & {f(r['scratch_minus_prior'])}\\\\")
(T / "attr_transfer.tex").write_text("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
def ci(d): d = np.array(d); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def load(tags): return [[json.load(open(ROOT / f"results/text_side/v6tft_{t}_fold{k}.json")) for k in range(10)] for t in tags]
runs = [("Text LoRA, no hard negatives (3 seeds)", ["T1_lora_M_hn0", "T1s1", "T1s2"]), ("Text LoRA + hard negatives", ["T2_lora_M_hn2"]), ("Text LoRA, blind profiles only + hard neg.", ["T3_lora_L_hn2"]),
        ("Last 4 text layers + hard negatives", ["T4_last_M_hn2"]), ("Last 4 text layers, no hard negatives", ["T5_last_M_hn0"]), ("mpnet + projection", ["S1_mpnet_proj"]), ("mpnet last 2 layers + projection", ["S2_mpnet_last"])]
rows = ["\\begin{tabular}{lcccc}\\toprule Text-side model & Image-derived description & Blind description & Blind tag list & Name prompt\\\\\\midrule"]
for lab, tags in runs:
    J = load(tags); cells = []
    for k in ("A_long", "L_long", "L_tags", "L_name"):
        z = np.mean([[j["test_zs"][k] for j in s] for s in J], 0); b = np.mean([[j["test_best"][k] for j in s] for s in J], 0); m, lo, hi = ci(b - z); cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
(T / "text_side.tex").write_text("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
z = json.load(open(ROOT / "results/text_side/v6tft_DGT0_fold-1_s0.json"))["dg_tuned"]; J = [json.load(open(ROOT / f"results/text_side/v6tft_DGT1_fold-1_s{s}.json"))["dg_tuned"] for s in range(3)]
rows = ["\\begin{tabular}{lcccc}\\toprule Text (39 types) & Zero-shot & Text LoRA (3 seeds) & Gain over types [95\\% CI] & Better / worse\\\\\\midrule"]
for k, lab in (("long", "Blind description"), ("tags", "Blind tag list"), ("name", "Disease name")):
    ts = sorted(z[k]["per_type"]); zs = np.array([z[k]["per_type"][t] for t in ts]); tu = np.mean([[j[k]["per_type"][t] for t in ts] for j in J], 0); m, lo, hi = ci(tu - zs)
    rows.append(f"{lab} & {zs.mean():.3f} & {tu.mean():.3f} & {m:+.3f} [{lo:+.3f}, {hi:+.3f}] & {int((tu - zs > 1e-9).sum())} / {int((tu - zs < -1e-9).sum())}\\\\")
(T / "dg_textlora.tex").write_text("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n"); print("ok")

# ---- joint vision+text LoRA (v6_joint.py): CV table (seeds averaged per fold) and DigiGreen table
import os
def jload(tag):
    fs = sorted(glob.glob(str(ROOT / f"results/joint/v6joint_{tag}_fold[0-9]_s*.json"))); return [json.load(open(f)) for f in fs]
seeds = [jload(t) for t in ("J0", "J1", "J2")]; seeds = [s for s in seeds if len(s) == 10]
if seeds:
    rows = [f"\\begin{{tabular}}{{lcccc}}\\toprule Joint model ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}) & Image-derived description & Blind description & Blind tag list & Name prompt\\\\\\midrule"]
    cells = []; zs_cells = []
    for k in ("A_long", "L_long", "L_tags", "L_name"):
        z = np.mean([[j["test_zs"][k] for j in s] for s in seeds], 0); b = np.mean([[j["test_best"][k] for j in s] for s in seeds], 0); m, lo, hi = ci(b - z); zs_cells.append(f"{z.mean():.3f}"); cells.append(f"{b.mean():.3f} ({m:+.3f} [{lo:+.3f}, {hi:+.3f}])")
    rows.append("Zero-shot & " + " & ".join(zs_cells) + "\\\\"); rows.append("Vision + text LoRA & " + " & ".join(cells) + "\\\\")
    (T / "joint_cv.tex").write_text("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
dgf = sorted(glob.glob(str(ROOT / "results/joint/v6joint_JDG_fold-1_s*.json")))
if dgf:
    D = [json.load(open(f)) for f in dgf]; rows = [f"\\begin{{tabular}}{{lcccc}}\\toprule Text (39 types, {len(D)} seed{'s' if len(D) > 1 else ''}) & Zero-shot & Joint LoRA & Gain over types [95\\% CI] & Better / worse\\\\\\midrule"]
    for k, lab in (("long", "Blind description"), ("tags", "Blind tag list"), ("name", "Disease name")):
        ts = sorted(D[0]["dg_zs"][k]["per_type"]); zs = np.array([D[0]["dg_zs"][k]["per_type"][t] for t in ts]); tu = np.mean([[d["dg_tuned"][k]["per_type"][t] for t in ts] for d in D], 0); m, lo, hi = ci(tu - zs)
        rows.append(f"{lab} & {zs.mean():.3f} & {tu.mean():.3f} & {m:+.3f} [{lo:+.3f}, {hi:+.3f}] & {int((tu - zs > 1e-9).sum())} / {int((tu - zs < -1e-9).sum())}\\\\")
    (T / "joint_dg.tex").write_text("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print("joint tables:", len(seeds), "cv seeds;", len(dgf), "dg seeds")

# ---- per-attribute eval of the joint and vision-only models (v6_joint_attr.py, v6_vision_attr.py -> results/cv/*_attr.json)
_ja = ROOT / "results/cv/joint_attr.json"; _va = ROOT / "results/cv/vision_attr.json"
if _ja.exists() and _va.exists():
    R = json.load(open(_ja))["summary"]; V = json.load(open(_va))["summary"]; rows = ["\\begin{tabular}{lccccccc}\\toprule Attribute & Global prior & Same-crop prior & Zero-shot prompts & Vision-only (1 seed) & Joint (3 seeds) & Vision-only $-$ prior & Joint $-$ prior\\\\\\midrule"]
    for a, nm in [("affected_part", "Affected part"), ("color", "Colour"), ("texture", "Texture"), ("shape", "Shape"), ("pattern", "Pattern"), ("lesion_density", "Lesion density")]:
        r = R[a]; v = V[a]; d = r["joint_minus_prior"]; e = v["vision_minus_prior"]
        rows.append(f"{nm} & {r['prior']:.3f} & {r['cropprior']:.3f} & {r['zs']:.3f} & {v['vision']:.3f} & {r['joint']:.3f} & {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] & {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]\\\\")
    (T / "joint_attr.tex").write_text("\n".join(rows) + "\n\\bottomrule\\end{tabular}\n"); print("joint_attr table")
