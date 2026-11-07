# Text-side experiments and image->symptom transfer (2026-11-05)

All numbers from `results/text_side/*.json`, `results/cv/attr_transfer.json`, `results/cv/cupl.json`. Cross-validation = 10 unit-grouped folds, unit-macro top-1 with the crop given; gain = tuned minus the run's own step 0; bootstrap CI over folds.
Image features are frozen SigLIP2-L-256 (cached); only the text side is trained (`scripts/v6_text_ft.py`). Descriptions are re-sampled each step (attribute dropout, value subsets, 5 templates); A = image-derived profile text, L = blind LLM profile text.

## 1. Text-tower runs (seed 0, 10 folds)
| run | A description | L description | L tags | name |
|---|---|---|---|---|
| T1 text LoRA, train A+L, no hard negatives | +0.038 [-0.003,+0.077] | +0.042 [+0.014,+0.069] | +0.065 [+0.029,+0.110] | +0.017 |
| T2 text LoRA + attribute-swap hard negatives | +0.013 | +0.005 | +0.041 [+0.006,+0.081] | -0.004 |
| T3 text LoRA, blind profiles only, hard negatives | -0.055 [-0.111,-0.006] | -0.008 | +0.025 | -0.011 |
| T4 last 4 layers unfrozen + hard negatives | -0.018 | -0.012 | +0.004 | -0.020 |
| T5 last 4 layers unfrozen, no hard negatives | +0.003 | +0.031 [-0.005,+0.068] | +0.052 | 0.000 |
| S1 mpnet + projection (distilled init) | -0.021 | -0.023 | -0.025 | -0.015 |
| S2 mpnet, last 2 layers + projection | -0.027 | 0.000 | -0.013 | -0.017 |
(S3 bge, e5, PubMedBERT: not run - bge tokenizer failed on a missing protobuf dependency; S1/S2 showed no gain so they were not pursued.)

Hard negatives hurt (T1 > T2, T5 > T4). Text-tower LoRA is the only variant with a gain.
T1 over 3 seeds (seed-averaged per fold): A description 0.624 -> 0.656 (+0.031 [-0.006,+0.063], 7/10 folds); L description 0.561 -> 0.592 (+0.031 [+0.002,+0.057], 8/10); L tags 0.493 -> 0.568 (+0.075 [+0.043,+0.117], 10/10); name 0.639 -> 0.647 (+0.008).

## 2. DigiGreen-full (423 images, 39 types), blind profiles, T1 trained on PlantWild units not matching DG types, 3 seeds, 250 steps
description 0.395 -> 0.424 (+0.029 [-0.030,+0.086]; 14 better / 12 worse); tags 0.391 -> 0.420 (+0.029 [-0.048,+0.108]); name 0.476 -> 0.463 (-0.013). Not significant.

## 3. LLM paraphrases (training-free), CV
6 blind LLM paraphrases per disease averaged: 0.409 vs blind-profile description 0.562 (-0.154 [-0.233,-0.076]); +tags 0.514; +tags+name 0.618 (+0.056 over blind description, but name alone is 0.639). DG-full: no combination differs from blind description in a CI. Negative result: free-text paraphrases are worse than the structured profile.

## 4. Image -> symptom transfer to held-out diseases (the paper's claim)
Linear head on frozen image features trained on seen diseases; hit = top-1 value among the group's annotated values; unit-macro; group-level labels.
| attribute | most-common value | same-crop prior | head | head - crop prior | head - most-common |
|---|---|---|---|---|---|
| affected part | 0.715 | 0.619 | 0.931 | +0.312 [0.250,0.373] | +0.216 [0.166,0.272] |
| colour | 0.696 | 0.554 | 0.700 | +0.146 [0.055,0.235] | +0.004 [-0.059,0.067] |
| texture | 0.653 | 0.434 | 0.660 | +0.226 [0.128,0.322] | +0.007 [-0.067,0.085] |
| shape | 0.424 | 0.268 | 0.580 | +0.312 [0.229,0.387] | +0.156 [0.077,0.225] |
| pattern | 0.248 | 0.256 | 0.552 | +0.297 [0.232,0.360] | +0.304 [0.214,0.385] |
| lesion density | 0.483 | 0.471 | 0.550 | +0.079 [0.024,0.129] | +0.067 [-0.058,0.189] |
Zero-shot attribute prompts are below the prior for colour/texture/shape/density (e.g. texture 0.344, density 0.163). Caveats: group-level labels, top-1-in-set hit, single seed per fold, a trained head (not the text-matching model).

## 3. Joint vision+text LoRA (scripts/v6_joint.py), 3 seeds (J0,J1,J2), 10-fold CV  [2026-11-05]
Gain over own zero-shot (seed-avg per fold, fold bootstrap): A description 0.625 -> 0.684 (+0.059 [+0.023,+0.092], 8 better/1 worse); L description 0.561 -> 0.609 (+0.048 [+0.011,+0.083], 7/2); L tags 0.490 -> 0.582 (+0.092 [+0.053,+0.136], 9/0); name 0.638 -> 0.656 (+0.018 [-0.008,+0.043], 6/3). Per-seed A gains 0.056/0.065/0.057. Fold 3 selects step 0 in all seeds.
DigiGreen-full (423 imgs, 39 types, 3 seeds, 400 steps): desc 0.395 -> 0.421 (+0.026 [-0.041,+0.092]); tags 0.392 -> 0.410 (+0.018 [-0.064,+0.100]); name 0.479 -> 0.507 (+0.028 [-0.031,+0.088]). Not significant.
Paper updated (abstract, contributions, new Sec. joint, limitations, discussion, conclusion); backup paper/main_before_joint.tex.bak. PDF NOT rebuilt (no tectonic on this machine).

## 4. Per-attribute probe of the joint model (scripts/v6_joint_attr.py -> results/cv/joint_attr.json), 3 seeds x 10 folds
Generic value prompts, adapted model. Joint vs global prior: part +0.063 [0.012,0.112], colour -0.059, texture -0.340, shape +0.041 (n.s.), pattern +0.281 [0.200,0.350], density -0.312. Vs same-crop prior: part +0.159, shape +0.197, pattern +0.273 (CIs > 0). Vs own zero-shot: part +0.054, colour +0.034, shape +0.073 significant.
Paper PDF rebuilt 2026-11-05 (tectonic 0.15.0 binary in the session scratchpad; re-download from GitHub releases if needed). Tables wrapped in \resizebox.

## 5. Vision-only per-attribute under disease-macro metric (scripts/v6_vision_attr.py -> results/cv/vision_attr.json, seed 0)
Part 0.818 (+0.103 vs global prior), colour 0.629, texture 0.333, shape 0.421 (-0.003), pattern 0.505 (+0.256), density 0.261. Vision-only >= joint for part and density; joint slightly higher for shape/pattern. Joint's larger matching gain is NOT from better generic-prompt attribute reading.
Number audit 2026-11-05: DG-full gains (+0.035 [-0.025,0.098] 14/8; blind-trained -0.016 15/13, CI recomputed [-0.088,0.054] vs paper [-0.091,0.055], bootstrap noise), controls, VLM, paraphrase, 3 vision seeds (0.646/0.633/0.655) all match raw files.
