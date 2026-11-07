"""Top-1/top-3 (unit-macro) and sibling-confusion analysis of the joint LoRA model from results/cv/joint_unitpred.csv (made by v6_joint_unitpred.py). Run with geo_env python."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; V5 = Path(__import__("os").environ.get("PW_DATA", ROOT / "data"))  # folder that contains the PlantWild image folders; rng = np.random.default_rng(0)
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]
U = pd.read_csv(ROOT / "results/cv/joint_unitpred.csv", keep_default_na=False)
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
jac = lambda x, y: 1.0 if not x and not y else len(x & y) / len(x | y)
sim = lambda u, v: float(np.mean([jac(P[u][a], P[v][a]) for a in ATT]))
nn = {u: max((v for v in P if v != u and v.split("|")[0] == u.split("|")[0]), key=lambda v: sim(u, v), default=None) for u in P}
U["t1"] = U.unit == U.pred1; U["t3"] = (U.unit == U.pred1) | (U.unit == U.pred2) | (U.unit == U.pred3)
U["nn_conf"] = (~U.t1) & (U.pred1 == U.unit.map(nn))
def boot(x):
    x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]; return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {}
for src in ("A", "L"):
    u = U[U.src == src]; per = u.groupby(["seed", "unit"])[["t1", "t3"]].mean().groupby("unit").mean()
    ncand = u.groupby("unit").apply(lambda g: 1).index  # placeholder
    err = u[~u.t1]; out[src] = dict(top1=boot(per.t1), top3=boot(per.t3), n_units=len(per), err_is_nearest_sibling=float(err.nn_conf.mean()), n_err=len(err))
json.dump(out, open(ROOT / "results/cv/top3.json", "w"), indent=1); print(json.dumps(out, indent=1))
