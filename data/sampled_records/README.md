# Original 90-video sampling records

Each benchmark has a `*_video_ids.txt` list containing 90 distinct video identifiers and a `*_nif.csv` table containing the available per-question annotation records.

| Prefix | Dataset |
| --- | --- |
| `video_mme` | Video-MME |
| `mlvu` | MLVU |
| `egoschema` | EgoSchema |
| `perception_test` | Perception Test |

CSV columns are `record_no`, `source_row`, `video_id`, `task`, `question_id`, `question`, `nif`, and `timestamp`. `nif` is the recorded number of informative frames. Multiple timestamp entries are separated by semicolons. Empty fields were unavailable in the corresponding source record and are not inferred. IDs are strings and should not be converted to numbers.

The Perception Test records refer to the original training-split sampling. These lists document the sampled diagnostic experiments and are distinct from the expanded benchmark evaluation sets.

Download benchmark videos and annotations from their original providers. This directory contains sampling and annotation records only.
