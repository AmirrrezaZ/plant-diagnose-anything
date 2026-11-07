"""Idea 3: disease-profile ambiguity analysis. Profiles = image-derived unit profiles (tags covering >=25% of the unit's images, as in v6_joint.py).
Per-unit accuracy = per-image correctness of the symptom-only vision LoRA (F1,F2 = seeds 1,2, 10 folds) and zero-shot, from results/arvan_v3/*_units.csv.
Outputs results/cv/ambiguity.json, paper/tables/ambiguity.tex, paper/figures/ambiguity.pdf. Run with geo_env python."""
import json, numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; V5 = Path(__import__("os").environ.get("PW_DATA", ROOT / "data"))  # folder that contains the PlantWild image folders
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]; rng = np.random.default_rng(0)
A = pd.read_csv(V5 / "annotation/annotations_v6.csv", keep_default_na=False); A = A[A.disease_type != "healthy"]
def prof(g, share=.25):
    o = {}
    for a in ATT:
        c = {}
        for s in g[a]:
            for t in [t for t in s.split(";") if t]: c[t] = c.get(t, 0) + 1
        o[a] = {t for t, k in c.items() if k / len(g) >= share and t != "none"}
    return o
P = {u: prof(g) for u, g in A.groupby("disease_unit")}
def jac(x, y): return 1.0 if not x and not y else len(x & y) / len(x | y)
def sim(u, v): return float(np.mean([jac(P[u][a], P[v][a]) for a in ATT]))
def diff_att(u, v): return [a for a in ATT if P[u][a] != P[v][a]]
# per-unit accuracy
U = pd.concat([pd.read_csv(ROOT / f"results/arvan_v3/v6lora_{t}_fold{k}_units.csv").assign(run=t) for t in ("F1", "F2") for k in range(10)])
acc = U.groupby(["run", "unit"])[["zs", "tuned"]].mean().groupby("unit").mean(); n_img = U.groupby("unit").size() / 2
rows = []
for u in acc.index:
    crop = u.split("|")[0]; sib = [v for v in P if v != u and v.split("|")[0] == crop]
    s = sorted(((sim(u, v), v) for v in sib), reverse=True); best = s[0]
    n_identical_part_col_pat = sum(all(P[u][a] == P[v][a] for a in ("affected_part", "color", "pattern")) for v in sib)
    rows.append(dict(unit=u, crop=crop, n_sib=len(sib), nn=best[1], nn_sim=best[0], nn_diff=",".join(diff_att(u, best[1])), n_diff=len(diff_att(u, best[1])),
                     n_sib_ge_0_8=sum(x >= .8 for x, _ in s), zs=acc.loc[u, "zs"], tuned=acc.loc[u, "tuned"], chance=1 / (len(sib) + 1)))
R = pd.DataFrame(rows)
bins = [(0, .5, "distinct (<0.5)"), (.5, .7, "similar (0.5-0.7)"), (.7, 1.01, "near-identical (>=0.7)")]
def boot(x): x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]; return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
out = {"n_units": len(R), "bins": {}}
for lo, hi, name in bins:
    m = R[(R.nn_sim >= lo) & (R.nn_sim < hi)]
    if len(m): out["bins"][name] = dict(n=len(m), zs=boot(m.zs), tuned=boot(m.tuned), chance=float(m.chance.mean()), mean_n_diff=float(m.n_diff.mean()))
from scipy.stats import spearmanr
out["spearman_nn_sim_vs_tuned"] = [float(x) for x in spearmanr(R.nn_sim, R.tuned)]; out["spearman_nn_sim_vs_zs"] = [float(x) for x in spearmanr(R.nn_sim, R.zs)]
# which attribute distinguishes the nearest sibling, per bin
att_count = {}
for lo, hi, name in bins:
    m = R[(R.nn_sim >= lo) & (R.nn_sim < hi)]; c = {a: 0 for a in ATT}
    for d in m.nn_diff: 
        for a in d.split(","): 
            if a: c[a] += 1
    att_count[name] = {a: c[a] / max(1, len(m)) for a in ATT}
out["distinguishing_attribute_share"] = att_count
out["pairs_identical_on_all_six"] = int(sum(1 for u, v in combinations(P, 2) if u.split("|")[0] == v.split("|")[0] and not diff_att(u, v)))
out["worst_units"] = R.sort_values("tuned").head(12)[["unit", "nn", "nn_sim", "nn_diff", "tuned", "zs"]].round(3).to_dict("records")
json.dump(out, open(ROOT / "results/cv/ambiguity.json", "w"), indent=1)
R.round(4).to_csv(ROOT / "results/cv/ambiguity_units.csv", index=False)
tex = ["\\begin{tabular}{lcccc}\\toprule Nearest same-crop sibling profile & Units & Zero-shot & Symptom LoRA & Chance\\\\\\midrule"]
for name, v in out["bins"].items(): tex.append(f"{name} & {v['n']} & {v['zs'][0]:.3f} & {v['tuned'][0]:.3f} [{v['tuned'][1]:.3f}, {v['tuned'][2]:.3f}] & {v['chance']:.3f}\\\\")
(ROOT / "paper/tables/ambiguity.tex").write_text("\n".join(tex) + "\n\\bottomrule\\end{tabular}\n")
fig, ax = plt.subplots(figsize=(3.6, 2.9)); plt.rcParams.update({"font.size": 8})
ax.scatter(R.nn_sim, R.tuned, s=14, c="#2b6cb0", alpha=.8); ax.set_xlabel("Profile similarity to nearest same-crop sibling"); ax.set_ylabel("Unit top-1 (symptom LoRA, 2 seeds)")
z = np.polyfit(R.nn_sim, R.tuned, 1); xs = np.linspace(R.nn_sim.min(), R.nn_sim.max(), 20); ax.plot(xs, np.polyval(z, xs), "k--", lw=.8)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(ROOT / "paper/figures/ambiguity.pdf"); fig.savefig(ROOT / "paper/figures/ambiguity.png", dpi=200)
print(json.dumps({k: v for k, v in out.items() if k != "worst_units"}, indent=1)); print(R.nn_sim.describe().round(3).to_string()); print(pd.DataFrame(out["worst_units"]).to_string())
