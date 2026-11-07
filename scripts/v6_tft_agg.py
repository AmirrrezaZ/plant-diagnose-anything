import json, glob, sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results/v3"; rng = np.random.default_rng(0)
def ci(d):
    d = np.array(d); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]; return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
for tag in sys.argv[1:]:
    fs = sorted(glob.glob(str(R / f"v6tft_{tag}_fold[0-9].json")));
    if not fs: continue
    J = [json.load(open(f)) for f in fs]; print(f"== {tag}  folds={len(J)}")
    for k in ("A_long", "L_long", "L_tags", "L_name"):
        z = [j["test_zs"][k] for j in J]; b = [j["test_best"][k] for j in J]; m, lo, hi = ci(np.array(b) - np.array(z)); print(f"  {k:7s} zs {np.mean(z):.3f} -> tuned {np.mean(b):.3f}  gain {m:+.3f} [{lo:+.3f},{hi:+.3f}]  better {sum(x>y+1e-9 for x,y in zip(b,z))}/{len(J)}  best steps {[j['best_step'] for j in J]}")
    if "fused" in J[0]:
        for k in J[0]["fused"]: print(f"  fused {k}: zs {np.mean([j['fused'][k]['zs_test'] for j in J]):.3f} -> {np.mean([j['fused'][k]['test'] for j in J]):.3f}")
