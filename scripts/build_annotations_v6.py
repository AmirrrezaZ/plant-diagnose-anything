"""v6 label table = v2 labels + v3 image cleaning (kept ids) + reviewed v3 fixes mapped back to the FINE vocabulary + drop of at-chance values.
Fine vocabulary kept (powdery/fuzzy, pustules/corky, ...) because merging cost description matching ~0.016 top-1. Dropped values: color olive, shape holes, pattern marginal_tip / vein_associated, texture sticky_exudate; attribute lesion_size removed.
Writes data/annotation/annotations_v6.csv and vocab_v6.json. Reads only; v1/v2/v3 and splits untouched."""
import json
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[1]; AD = ROOT / "data/annotation"; W = AD / "v3_work"
A = pd.read_csv(AD / "annotations_v2.csv", keep_default_na=False); V3 = pd.read_csv(AD / "annotations_v3.csv", keep_default_na=False)
A = A[A.img_id.isin(set(V3.img_id))].copy()
DROP = {"color": {"olive"}, "shape": {"holes"}, "pattern": {"marginal_tip", "vein_associated"}, "texture": {"sticky_exudate"}}
for a, vs in DROP.items(): A[a] = A[a].map(lambda s: ";".join(v for v in s.split(";") if v and v not in vs))
BACK = {"texture": {"coating": ["powdery", "fuzzy_moldy"], "raised": ["pustules", "raised_corky", "sticky_exudate"], "sunken_soaked": ["sunken", "water_soaked"]}, "shape": {"distorted": ["curled_distorted", "swollen_galls"], "wilted": ["wilted"], "concentric_target": ["concentric_target"]}, "pattern": {"mottle_interveinal": ["interveinal", "mosaic_mottle"]}}
F = pd.read_csv(W / "fixes.csv", keep_default_na=False); log = []
for r in F[F.attr != "cluster"].itertuples():
    m = (A.group_id == r.cluster).to_numpy()
    if not m.any(): continue
    if r.action == "remove":
        rm = set(BACK.get(r.attr, {}).get(r.value, [r.value])); A.loc[m, r.attr] = A.loc[m, r.attr].map(lambda s: ";".join(v for v in s.split(";") if v and v not in rm))
    else:
        val = {"fruit_seed": "fruit"}.get(r.value, r.value); A.loc[m, r.attr] = val
    log.append((r.cluster, r.attr, r.action, r.value, int(m.sum())))
# density from v3 (carries the diffuse-symptom rule); lesion_size removed; gate = density != none
A = A.merge(V3[["img_id", "lesion_density", "v3_conf", "v3_clean_core", "v3_flags", "v3_junk_p", "v3_p_disease"]].rename(columns={"lesion_density": "_d3"}), on="img_id"); A["lesion_density"] = A["_d3"]; A = A.drop(columns=["_d3", "lesion_size"])
A["lesion_applicable"] = (A.lesion_density != "none") & (A.disease_type != "healthy")
A.to_csv(AD / "annotations_v6.csv", index=False)
ATT = ["affected_part", "color", "texture", "shape", "pattern", "lesion_density"]
vocab = {a: sorted({v for s in A[a] for v in s.split(";") if v}) for a in ATT}; json.dump(vocab, open(AD / "vocab_v6.json", "w"), indent=1)
nh = A[A.disease_type != "healthy"]
print("rows", len(A), "units", nh.disease_unit.nunique(), "clusters", nh.group_id.nunique(), "fixes applied", len(log)); print({a: len(v) for a, v in vocab.items()}, sum(len(v) for v in vocab.values()), "values")
for a in ["texture", "shape", "pattern", "color", "affected_part"]: print(a, nh[a].str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict())
print("blank texture", int((nh.texture == "").sum()), "blank pattern", int((nh.pattern == "").sum()), "blank shape", int((nh["shape"] == "").sum()))
