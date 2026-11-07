# Annotation data card

`annotations.csv`: one row per photograph (20,942 rows, 120 disease units, 6 symptom attributes). Photographs are not included; `orig_path` / `img_path` identify the PlantWild file.

**Read this first.** Symptom tags are assigned to *groups* of visually similar images (crop, disease, affected part, cluster), so every image of a group shares one tag set and individual images may deviate. The labels are the best we could produce with limited resources and are not noise-free: the source images themselves contain watermarks, charts and unrelated photos. No inter-annotator agreement was measured.

## How labels were made
1. The affected part was annotated for the images of the dataset and reviewed by the first author.
2. Images of each (crop, disease, part) were grouped, partly by image-similarity clustering and partly by manual supervision with a cluster-review tool.
3. Symptom tags were assigned per group, for most groups by human annotators (the first author and a second annotator working under supervision), and for a smaller share by an LLM-assisted procedure. An LLM (Claude) also assisted the annotators throughout.

## Provenance columns
| column | values | meaning |
|---|---|---|
| `annotation_tier` | `human_group`, `llm_group`, `healthy`, `llm_image` | who produced the group's tags. `llm_group` (and `llm_image`) = LLM-assisted procedure, not separately human-reviewed as far as recorded; `healthy` = healthy photos |
| `review_status` | `reviewed_group`, `claude_group`, `healthy`, `claude_image` | review state matching the tier |
| `annotation_confidence` | 0.6 to 1.0 | 0.8 for human groups, 0.6 for LLM groups, 1.0 healthy, 0.9 per-image cases |
| `affected_part_source` | `human`, `model_auto_confirmed`, `gpt_model_agree`, `class_rule`, `unspecified` | how the affected part was set (`class_rule` for healthy images) |
| `cluster_source` | `user`, `pool`, `model`, `unreviewed_old_group`, `healthy_or_old` | how the image's group was formed |
| `group_quality`, `group_cohesion`, `attr_reliable` | | quality indicators for the group; filter on `attr_reliable` for cleaner subsets |
| `v3_*` columns | | image-cleaning flags (possible junk, tiny cluster, healthy-maybe-diseased) |

We recommend using `annotation_confidence` or `annotation_tier` if your use needs the cleanest subset. Licence: CC BY 4.0 (`LICENSE-DATA`).
