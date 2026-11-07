# Version 6 (2026-11-03): clean data, new splits, supervised vs zero-shot

All earlier conclusions were on v2 labels and the old frozen folds; this re-runs the key comparison on v6.

## Data and splits (new files only; v1/v2/v3 and splits.csv untouched)
* `data/annotation/annotations_v6.csv` (+ `vocab_v6.json`, `scripts/build_annotations_v6.py`): v2 labels restricted to v3's cleaned image set (20,942 imgs, 120 disease units, 470 clusters), reviewed v3 fixes mapped back to the fine vocabulary, at-chance values dropped (color olive, shape holes, pattern marginal_tip / vein_associated, texture sticky_exudate), `lesion_size` removed. 43 values.
* `data/split/splits_cv.csv`, `unit_folds.csv` (`scripts/make_splits_cv.py`): 10-fold unit-grouped CV over all 120 disease units (12 per fold; test fold k, val fold k+1, rest train); balanced by tag novelty, crop spread, image count. Healthy photos fixed 60/20/20 per crop. 4 single-unit crops (maple, squash, basil, onion) are excluded from the headline (crop-given candidate set of one): 116 test units.
* Text embeddings: `results/cache/v6_text.npz` (`scripts/v6_text.py`): per unit and per cluster, prompt styles long/short/tags/name/crop.

## Baselines on the new CV (unit-macro top-1, crop given, 116 units, `scripts/v6_cv.py`)
| method | top-1 |
|---|---|
| zero-shot description ("long") | 0.627 |
| zero-shot disease-name prompts | 0.639 |
| zero-shot description + name (sum of logits) | **0.683** |
| zero-shot crop-only | 0.269 |
| linear tag head + profile lookup | 0.361 |
| score fusion head + zero-shot (weight on val) | 0.599 (no gain) |
| profile-matching adapter on frozen features | ~= zero-shot |
Same picture as on the old folds: trained tag heads are far below zero-shot matching; frozen-feature adapters add nothing.

## Supervised LoRA vs zero-shot (SigLIP2-L-256 vision tower, zero-shot objective, text fixed; scripts `v6_lora2.py`, `v6_lora_dg2.py`; arvan logs `v6lora_*`)
Step 0 == frozen zero-shot (LoRA B=0). Step chosen on val units (unseen). 400 steps, lr 5e-5, rank 16, qkvo.

### PlantWild 10-fold unit-grouped CV (unit-macro top-1, crop given)
| variant | folds | zero-shot | LoRA | gain | bootstrap 95% CI over folds | folds improved |
|---|---|---|---|---|---|---|
| t1: long+tags training, scored on description | 10 | 0.625 | 0.648 | +0.023 | [-0.007, 0.053] | 5/10 |
| **D: long+tags+name training, scored desc+name** | 10 | 0.680 | **0.708** | +0.028 | [0.004, 0.054] | 6/10 |
| C: all layers, rank 32 | 2 (folds 5, 9 OOM) | 0.614 | 0.655 | +0.041 | [0.016, 0.066] | 2/2 |
* D chosen as final setup: best absolute score and only full-10-fold gain whose CI excludes zero. Beats the strongest zero-shot baseline (0.683) by ~+0.025.
* C is not comparable (2 folds; OOM caused by a stray 14.7 GB process on GPU 0 of arvan, PID 2133474). Not ranked.
* Gain is modest; fold 3 selected step 0 (no gain) and several folds are flat or lower (fold 9: 0.636 -> 0.627).

### DigiGreen external test (69 main images, 14 held-out diseases never trained on; top-1)
| | zero-shot | D, seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|---|
| description + name ensemble | 0.536 | 0.696 | 0.696 | 0.667 | **0.686** |
| description only | 0.536 | 0.667 | 0.652 | 0.667 | 0.662 |
| name only | 0.522 | 0.609 | 0.580 | 0.623 | 0.604 |
* Gain ~+0.15 top-1 on real farm photos; seeds agree. Only 69 images (CI about +-0.11), so strongly suggestive, not conclusive. Selected at final step (no val set here).


### Reading
Supervised fine-tuning beats zero-shot on both benchmarks, but by a small margin on PlantWild CV (+0.03) and a larger one on DigiGreen (+0.15). Report both, with the CI caveats. Synthetic edits not used in v6.

## 2026-11-04 Description-only track (no disease name in training or scoring)
Recipe sweep on folds 0,2,5,9 (unit-macro test top-1; baseline t1 = 0.610): E1 hotter lr 1e-4/600 steps 0.612; E2 LoRA on all layers 0.601; E3 +short prompt, cluster weight 1.0 0.618; E4 lr 2e-5/800 steps 0.616. All within fold noise (~+-0.01): recipe tuning does not help. Final config = E4 (long+tags prompts, rank 16, qkvo, lr 2e-5, 800 steps, step chosen on val units). Logs: arvan `logs/v6lora_{E1..E4}_f*.log`.

**Final description-only, 3 seeds x 10 folds** (`logs/v6lora_F{0,1,2}_f*.log`): zero-shot 0.625; LoRA seed0 0.646, seed1 0.633, seed2 0.655; 3-seed mean **0.645** (std over seeds 0.009). Gain +0.019, bootstrap-over-folds 95% CI [-0.010, 0.049], gain>0 on 5/10 folds => **not significant on PlantWild CV**. Seed-mean per fold: .681 .710 .549 .609 .777 .736 .619 .568 .701 .497.

**DigiGreen, description-only model** (69 imgs, 14 unseen diseases, scored with description text; `logs/v6lora_dgFdesc_s*.log`): zero-shot 0.536 -> 0.667 / 0.652 / 0.667 (mean **0.662**). Top-3 0.870. n=69, CI about +-0.11; selected at final step (no val set).

## Baselines re-run on v6 CV (unit-macro top-1, crop given, 116 units)
| method | top-1 | file |
|---|---|---|
| chance | 0.267 | results/v6/controls.json |
| CLIP-L/14 zero-shot, name prompts | 0.495 | results/v6/bcvsa.json |
| BCVSA-style blend (DINOv2-L + CLIP-L) | 0.456 | results/v6/bcvsa.json |
| BCVSA align head only | 0.309 | results/v6/bcvsa.json |
| linear tag head + profile lookup | 0.361 | V6 log above |
| SigLIP2-L zero-shot description | 0.627 | controls.json |
| SigLIP2-L zero-shot name | 0.639 | controls.json |
| SigLIP2-L zero-shot description+name | 0.683 | controls.json |
| controls: shuffled description / name text | 0.136 / 0.126 | controls.json |
| controls: crop prior / nearest seen class | 0.009 / 0.000 | controls.json |
| ours, description-only LoRA (3 seeds) | 0.645 | above |
| ours, name+description LoRA D (1 seed) | 0.708 | above |
Note: crop-prior / nearest-seen restricted to unseen candidates score 0.85 / 0.87 only because each crop then has ~1 unseen candidate (benchmark diagnostic, not a competitor).
Caveat: PlantWild CV profiles of test diseases come from the test images' own tags (optimistic); DigiGreen is the clean test.
