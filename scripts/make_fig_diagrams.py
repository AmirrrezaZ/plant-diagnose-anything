"""Fig 1 (overview), Fig 3 (architecture), Fig 4 (disease-disjoint split) as matplotlib diagrams (no TikZ). Run with geo_env python."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "paper/figures"
C = dict(data="#edf2f7", frozen="#dbeafe", train="#fed7aa", good="#d1fae5", bad="#fee2e2", name="#ede9fe", llm="#fef9c3", test="#fb923c", val="#93c5fd", trn="#e2e8f0")
def box(ax, x, y, w, h, t, fc, fs=7.5, ec="#334155", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05", fc=fc, ec=ec, lw=.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)
def arr(ax, x0, y0, x1, y1, st="-|>"): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle=st, lw=1, color="#334155", shrinkA=0, shrinkB=0))
def canvas(w, h, xl, yl):
    f, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off"); return f, ax
# ---- Fig 1
f, ax = canvas(7.2, 2.7, 14.4, 5.4)
ax.text(.1, 5.1, "A. Name classification", fontsize=8.5, fontweight="bold", va="center")
box(ax, .1, 3.7, 1.7, 1, "Leaf\nimage", C["data"]); box(ax, 2.6, 3.7, 2.4, 1, "Disease\nclassifier", C["frozen"]); box(ax, 5.8, 3.7, 2.6, 1, "Name from a\nclosed list", C["name"])
box(ax, 9.2, 3.55, 5.0, 1.3, "Unseen disease: no output,\nor a wrong known name", C["bad"])
for a, b in [(1.8, 2.6), (5.0, 5.8), (8.4, 9.2)]: arr(ax, a, 4.2, b, 4.2)
ax.text(.1, 2.6, "B. Symptom-based matching (this paper)", fontsize=8.5, fontweight="bold", va="center")
box(ax, .1, 1.1, 1.7, 1, "Leaf\nimage", C["data"]); box(ax, 2.6, 1.1, 2.4, 1, "Visual symptom\nrepresentation", C["train"]); box(ax, 5.8, 1.1, 2.6, 1, "Match to symptom\ndescriptions", C["train"])
box(ax, 9.2, .95, 5.0, 1.3, "Hypothesis: shared symptoms\ntransfer to unseen diseases", C["good"])
for a, b in [(1.8, 2.6), (5.0, 5.8), (8.4, 9.2)]: arr(ax, a, 1.6, b, 1.6)
f.savefig(OUT / "fig_overview.pdf", bbox_inches="tight"); f.savefig(OUT / "fig_overview.png", dpi=200, bbox_inches="tight"); plt.close(f)
# ---- Fig 3
f, ax = canvas(7.2, 3.6, 14.4, 7.2)
box(ax, .1, 5.4, 2.2, 1.1, "Photo x", C["data"]); box(ax, 3.1, 5.4, 3.0, 1.1, "SigLIP2-L\nimage tower (frozen)", C["frozen"]); box(ax, 6.5, 5.4, 2.0, 1.1, "LoRA\n(vision)", C["train"])
box(ax, .1, 3.3, 2.2, 1.1, "Unit\ndescription t_u", C["data"]); box(ax, 3.1, 3.3, 3.0, 1.1, "SigLIP2-L\ntext tower (frozen)", C["frozen"]); box(ax, 6.5, 3.3, 2.0, 1.1, "LoRA\n(text)", C["train"])
arr(ax, 2.3, 5.95, 3.1, 5.95); arr(ax, 6.1, 5.95, 6.5, 5.95); arr(ax, 2.3, 3.85, 3.1, 3.85); arr(ax, 6.1, 3.85, 6.5, 3.85)
box(ax, 10.0, 4.0, 4.3, 1.7, "score s(x,u) = <f(x), g(t_u)>\nsoftmax over same-crop units\n(+ cluster loss, weight 0.5)", "white")
arr(ax, 8.5, 5.95, 10.0, 5.2); arr(ax, 8.5, 3.85, 10.0, 4.5)
ax.text(.1, 2.55, "Text sources for t_u (re-sampled every step):", fontsize=7.5, fontweight="bold", va="center")
box(ax, .1, 1.1, 3.6, 1.0, "Image-derived\ndescriptions / tags", C["good"], fs=7); box(ax, 4.0, 1.1, 3.6, 1.0, "Blind LLM descriptions\n(from disease name only)", C["llm"], fs=7); box(ax, 7.9, 1.1, 3.0, 1.0, "Disease-name\nprompts", C["name"], fs=7)
ax.add_patch(plt.Rectangle((11.4, 1.5), .5, .4, fc=C["frozen"], ec="#334155")); ax.text(12.05, 1.7, "frozen", fontsize=7, va="center")
ax.add_patch(plt.Rectangle((11.4, .8), .5, .4, fc=C["train"], ec="#334155")); ax.text(12.05, 1.0, "trainable", fontsize=7, va="center")
ax.text(.1, .3, "Attribute probe (linear head on frozen image features) is separate and not part of this retrieval model.", fontsize=6.5, style="italic", va="center")
f.savefig(OUT / "fig_architecture.pdf", bbox_inches="tight"); f.savefig(OUT / "fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close(f)
# ---- Fig 4
f, ax = canvas(7.2, 2.6, 14.4, 5.2)
ax.text(.1, 4.8, "Disease-disjoint 10-fold evaluation (fold 3 as example)", fontsize=8.5, fontweight="bold", va="center")
for i in range(10):
    k = "test" if i == 3 else "val" if i == 4 else "trn"; lab = {"test": "TEST\nunseen", "val": "val", "trn": "train"}[k]
    box(ax, .1 + i * 1.4, 3.3, 1.3, 1.0, f"fold {i}\n{lab}" if k != "trn" else f"fold {i}\ntrain", C[k], fs=6.5)
ax.text(.1, 2.55, "Each fold: 12 disease units. Every unit is test exactly once. All images of a test unit are unseen in training.", fontsize=7, va="center")
box(ax, .1, .5, 6.8, 1.4, "Used: split by disease unit\n(a test disease never appears in training)", C["good"], fs=7.5)
box(ax, 7.3, .5, 6.8, 1.4, "Not used: random image split\n(images of one disease on both sides)", C["bad"], fs=7.5)
f.savefig(OUT / "fig_split.pdf", bbox_inches="tight"); f.savefig(OUT / "fig_split.png", dpi=200, bbox_inches="tight"); plt.close(f)
