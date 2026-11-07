"""Dataset loader: reads data/annotation/annotations.csv and data/split/splits.csv, nothing else. Semantics: configs/attribute_semantics.yaml.

Empty cells (per attribute, see the yaml)
  color, shape, affected_part   unlabeled     -> masked
  texture, pattern              no_change in attr_reliable rows (labeled, all-zero target), unlabeled (masked) in attr_reliable=False rows
  healthy photos                color/texture/shape/pattern are not applicable -> masked
  `mask_empty=False` ignores the yaml and treats every empty cell as "no tag" (labeled, all-zero).
Lesion size / density = GATE + VALUE
  gate  = lesion_applicable (lesion_size != none; healthy -> False). Image has discrete lesions or not.
  value = size (large|small) / density (dense|sparse|targeted), a conditional target: masked wherever the gate is false (or the value is `none`).
  Gate-false rows keep their stored values in the dataframe; only the TARGETS are masked, nothing is dropped or rewritten.
Metrics: gate F1 on non-healthy images; value F1 only where the TRUE gate is true.
"""
import numpy as np, pandas as pd
from utils import ANN_CSV, SPLIT_CSV, ATTRS, MULTI, SINGLE, FOLDS, split_tags, empty_policy, truthy, SEM

LESION = tuple(SINGLE)
HEALTHY_NA = tuple(SEM["healthy_images"]["not_applicable_attributes"])

def load_annotations(path=ANN_CSV):
    A = pd.read_csv(path, keep_default_na=False, dtype=str)
    A["attr_reliable"] = A.attr_reliable.map(truthy); A["lesion_applicable"] = A.lesion_applicable.map(truthy)
    for c in ("annotation_confidence", "annotation_confidence_refined"):
        if c in A: A[c] = A[c].astype(float)
    return A

def loss_weights(df, floor=None, gamma=None):
    """per-image loss weight w = max(floor, confidence ** gamma) with confidence = annotation_confidence_refined (defaults: configs/annotation_confidence.yaml). Multiply with the target masks; do not stack with pos_weight in the same run."""
    import yaml
    from utils import CFG
    c = yaml.safe_load(open(CFG / "annotation_confidence.yaml"))["loss"]; col = c.get("column", "annotation_confidence_refined"); floor = c["floor"] if floor is None else floor; gamma = c["gamma"] if gamma is None else gamma
    return np.maximum(floor, df[col].to_numpy(float) ** gamma)

def load_splits(path=SPLIT_CSV):
    return pd.read_csv(path, keep_default_na=False)

def vocab(A=None):
    """sorted values per attribute over the whole file (stable across folds). `none` is not a value of lesion_size / lesion_density (it is the gate)."""
    A = load_annotations() if A is None else A
    return {a: sorted({t for s in A[a] for t in split_tags(s) if not (a in LESION and t == "none")}) for a in ATTRS}

def load_fold(fold, role, reliable_only=False, healthy=True):
    """annotation rows of one role of one fold. reliable_only drops attr_reliable=False rows (use it for attribute metrics)."""
    A = load_annotations(); S = load_splits()
    df = A.merge(S[["img_id", fold]].rename(columns={fold: "role"}), on="img_id"); df = df[df.role == role]
    if reliable_only: df = df[df.attr_reliable]
    if not healthy: df = df[df.disease_type != "healthy"]
    return df.reset_index(drop=True)

def gate(df):
    """lesion gate (bool array): the image shows discrete lesions. False for healthy images and whenever lesion_size is none/empty."""
    g = df.lesion_applicable.to_numpy(bool) if "lesion_applicable" in df else (df.lesion_size.str.len().gt(0) & df.lesion_size.ne("none")).to_numpy()
    return g & (df.disease_type != "healthy").to_numpy()

def labeled_mask(df, attr, mask_empty=True):
    """True where a non-lesion attribute is labeled (see the module doc)."""
    n = len(df)
    if not mask_empty: return np.ones(n, bool)
    empty = (df[attr].str.len() == 0).to_numpy(); rel = df.attr_reliable.to_numpy() if "attr_reliable" in df else np.ones(n, bool)
    pol = np.array([empty_policy(attr, bool(r)) for r in rel]); M = ~empty | (pol == "no_change")
    if attr in HEALTHY_NA: M &= ~(df.disease_type == "healthy").to_numpy()
    return M

def targets(df, attr, values=None, mask_empty=True):
    """(Y, M, values): Y multi-hot (n x V), M mask of rows that carry a target (rows with M False must be skipped by loss and metrics; their Y is zeros).
    For lesion_size / lesion_density this is the conditional VALUE target: M = gate & value present."""
    values = list(values) if values else vocab()[attr]; vi = {v: i for i, v in enumerate(values)}
    Y = np.zeros((len(df), len(values)), bool)
    for r, s in enumerate(df[attr]):
        for t in split_tags(s):
            if t in vi: Y[r, vi[t]] = True
    if attr in LESION:
        M = gate(df) & (df[attr].str.len() > 0).to_numpy() & (df[attr] != "none").to_numpy()
    else:
        M = labeled_mask(df, attr, mask_empty)
    Y[~M] = False
    return Y, M, values

def lesion_targets(df, attr, values=None):
    """(G, Y, M, values) for lesion_size / lesion_density: gate G (n,), conditional value Y, value mask M (= G & value labeled)."""
    Y, M, values = targets(df, attr, values)
    return gate(df), Y, M, values

# ---------------- metrics ----------------
def f1_multilabel(Y, P, M=None):
    """micro and macro (over values with support) F1 of multi-hot Y vs P on rows where M is True."""
    M = np.ones(len(Y), bool) if M is None else M; y, p = Y[M], P[M]
    tp, fp, fn = (y & p).sum(0), (~y & p).sum(0), (y & ~p).sum(0); d = 2 * tp + fp + fn; sup = y.sum(0) > 0
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1); f = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    return float(micro), (float(f[sup].mean()) if sup.any() else float("nan"))

def gate_metrics(df, pred_gate):
    """gate F1 on NON-HEALTHY images: F1 of the 'applicable' class, macro F1 over both classes, accuracy."""
    nh = (df.disease_type != "healthy").to_numpy(); g = gate(df)[nh]; p = np.asarray(pred_gate, bool)[nh]
    def f1(t, q): tp = (t & q).sum(); return 2 * tp / max(2 * tp + (~t & q).sum() + (t & ~q).sum(), 1)
    return dict(f1_applicable=float(f1(g, p)), f1_macro=float((f1(g, p) + f1(~g, ~p)) / 2), accuracy=float((g == p).mean()), n=int(nh.sum()), share_applicable=float(g.mean()) if nh.any() else float("nan"))

def value_metrics(df, attr, pred_values, values=None):
    """value F1 for lesion_size / lesion_density ONLY where the true gate is true (and the value is labeled)."""
    G, Y, M, vals = lesion_targets(df, attr, values); micro, macro = f1_multilabel(Y, np.asarray(pred_values, bool), M)
    return dict(f1_micro=micro, f1_macro=macro, n=int(M.sum()))

def not_applicable_counts(df):
    """per attribute, non-healthy images by what an empty cell means: masked (unlabeled), no_change (labeled all-zero), not_applicable (gate false)."""
    nh = df[df.disease_type != "healthy"]; out = {}
    for a in ATTRS:
        if a in LESION: out[a] = dict(not_applicable=int((~gate(nh)).sum()), masked=0, no_change=0); continue
        e = nh[a] == ""; pol = np.array([empty_policy(a, bool(r)) for r in nh.attr_reliable]); out[a] = dict(masked=int((e & (pol == "unlabeled")).sum()), no_change=int((e & (pol == "no_change")).sum()), not_applicable=0)
    return out

if __name__ == "__main__":
    d = load_fold("fold2", "val", reliable_only=True); print(len(d), not_applicable_counts(d))
