"""Fig 2 (annotation examples from annotations_v6.csv + version5 images) and Fig 5 (results summary from paper tables). Run with geo_env python."""
import pandas as pd, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; V5 = Path(__import__("os").environ.get("PW_DATA", ROOT / "data"))  # folder that contains the PlantWild image folders; OUT = ROOT / "paper/figures"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
# ---------- Fig 2
d = pd.read_csv(V5 / "annotation/annotations_v6.csv")
ids = {"strawberry / leaf scorch": "pw_250cd054ed8be811", "maple / tar spot": "pw_946a7f89ca7cbea9", "squash / powdery mildew": "pw_5b395931b1b34f7a", "banana / cordana leaf spot": "pw_2864abfb4575cb06", "wheat / stripe rust": "pw_489438d8fbbc8626", "tomato / late blight": "pw_86963560f35cd8d8"}  # hand-checked: tags visibly match the photo, symptoms differ
cols = [("affected_part", "Part"), ("color", "Colour"), ("texture", "Texture"), ("shape", "Shape"), ("pattern", "Pattern"), ("lesion_density", "Density")]
fig, axs = plt.subplots(4, 3, figsize=(7.2, 7.6), gridspec_kw={"height_ratios": [1.25, .85, 1.25, .85]})
for j, (u, i) in enumerate(ids.items()):
    r = d[d.img_id == i].iloc[0]; rr, cc = (j // 3) * 2, j % 3
    im = Image.open(V5 / "images" / f"{i}.jpg").convert("RGB"); im.thumbnail((400, 400))
    axs[rr, cc].imshow(im); axs[rr, cc].axis("off"); axs[rr, cc].set_title(u, fontsize=8)
    axs[rr + 1, cc].axis("off")
    lines = [f"{lab}: {str(r[c]).replace('_', ' ').replace(';', ', ')}" for c, lab in cols]
    axs[rr + 1, cc].text(0, 1, "\n".join("\n".join(textwrap.wrap(l, 34, subsequent_indent="  ")) for l in lines), va="top", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "annotation_examples.pdf"); fig.savefig(OUT / "annotation_examples.png", dpi=200); plt.close(fig)
# ---------- Fig 5 (numbers copied from paper/tables/{joint_cv,joint_attr,joint_dg}.tex)
BL, OR, GY = "#2b6cb0", "#dd6b20", "#a0aec0"
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]})
# a: zero-shot vs joint, CV
ax = axs[0]; lab = ["Image\ndesc.", "Blind\ndesc.", "Blind\ntags", "Name"]; zs = [.625, .561, .490, .638]; jt = [.684, .609, .582, .656]
lo = [.023, .011, .053, -.008]; hi = [.092, .083, .136, .043]
x = range(4)
ax.bar([i - .2 for i in x], zs, .38, color=GY, label="Zero-shot"); ax.bar([i + .2 for i in x], jt, .38, color=BL, label="Joint LoRA")
ax.errorbar([i + .2 for i in x], jt, yerr=[[j - z - l for j, z, l in zip(jt, zs, lo)], [h - (j - z) for j, z, h in zip(jt, zs, hi)]], fmt="none", ecolor="k", lw=.8, capsize=2)
ax.set_xticks(list(x)); ax.set_xticklabels(lab, fontsize=6.5); ax.set_ylim(.4, .75); ax.set_ylabel("Unit-macro top-1"); ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.set_title("(a) Held-out diseases (CV)", fontsize=8, loc="left")
# b: attributes, joint minus global prior
ax = axs[1]; at = ["Part", "Colour", "Texture", "Shape", "Pattern", "Density"]; m = [.063, -.059, -.340, .041, .281, -.312]
l = [.012, -.140, -.466, -.028, .200, -.412]; h = [.112, .012, -.195, .125, .350, -.196]
ax.barh(range(6), m, color=[BL if (a > 0 and b > 0) else GY for a, b in zip(l, h)])
ax.errorbar(m, range(6), xerr=[[a - b for a, b in zip(m, l)], [b - a for a, b in zip(m, h)]], fmt="none", ecolor="k", lw=.8, capsize=2)
ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(6)); ax.set_yticklabels(at); ax.invert_yaxis(); ax.set_xlabel("Joint model minus global prior (hit rate)")
ax.set_title("(b) Per attribute", fontsize=8, loc="left")
# c: internal vs external gain
ax = axs[2]; lab = ["Blind\ndesc.", "Blind\ntags", "Name"]; cv = [.048, .092, .018]; cl = [.011, .053, -.008]; ch = [.083, .136, .043]
dg = [.026, .018, .028]; dl = [-.041, -.064, -.031]; dh = [.092, .100, .088]
x = range(3)
for off, v, a, b, c, n in [(-.2, cv, cl, ch, BL, "PlantWild CV"), (.2, dg, dl, dh, OR, "DigiGreen (farmer)")]:
    ax.bar([i + off for i in x], v, .38, color=c, label=n)
    ax.errorbar([i + off for i in x], v, yerr=[[p - q for p, q in zip(v, a)], [q - p for p, q in zip(v, b)]], fmt="none", ecolor="k", lw=.8, capsize=2)
ax.axhline(0, color="k", lw=.8); ax.set_xticks(list(x)); ax.set_xticklabels(lab, fontsize=6.5); ax.set_ylabel("Gain over zero-shot"); ax.set_ylim(-.08,.19); ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.set_title("(c) Internal vs external", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "results_summary.pdf"); fig.savefig(OUT / "results_summary.png", dpi=200)
