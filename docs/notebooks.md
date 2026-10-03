# Research notebook guide

The notebooks are organized as independent experiments. Running every cell sequentially may start training, call an API or write results for several datasets. Run the configuration cell and the definitions needed by the experiment you want to inspect.

Each exported cell has a `source_cell` metadata field matching its original zero-based cell index. The two new introductory cells do not have an original index.

## Main experiment notebook

| Original cell indices | Purpose |
| --- | --- |
| 1 | Shared imports |
| 3, 5, 7, 9 | Video-MME, MLVU, Perception Test and EgoSchema loaders |
| 13, 14 | MMAT training data construction |
| 16 | API client and answering helper |
| 18 | CLIP embedding wrapper |
| 20 | Frame selection using the manuscript MMR formula |
| 23, 25, 27 | Uniform-sampling API experiments |
| 30, 32, 34 | Retrieval-based API experiments |
| 37–47 | Self-supervised fine-tuning experiments |
| 50–60 | Grouped contrastive fine-tuning for BGE-M3 and CLIP |
| 62 | Load fine-tuned encoders |
| 68–92 | NIF, retrieval and ASS analysis |
| 95, 97, 99, 101 | Export selected frames for the four datasets |

Paths for raw annotations and processed frames/captions now share `RAG_DATA_ROOT/dataset`. Training JSON files and fine-tuned models are under `RAG_WORK_ROOT/finetune`. Existing training JSON files may contain absolute image paths that need remapping to the local frame tree.

The original Perception Test sampling used the training annotations. Its 90-video records should not be substituted for the validation split in the expanded evaluation. The notebooks include historical dataset splits and model settings. Check the intended split, model, input frame count and subtitle setting before running an experiment.

## Retrieval behavior

The notebook retrieves visual and caption candidates separately and sums the returned scores by frame. It does not explicitly map cosine scores to [0, 1]. MMR first selects the frame with the highest fused relevance score. For each remaining candidate, it computes image cosine similarity plus caption cosine similarity to each already selected frame. The redundancy penalty is the maximum of these sums over the selected set. Its marginal relevance is `theta * relevance - (1 - theta) * maximum_redundancy`, with `theta=0.7` by default. The candidate with the highest marginal relevance is added at each step. If fewer than K candidates are available, all available candidates are returned without duplication or padding.

On 2026-10-03, the MMR helper was corrected to match the manuscript formula. Previously, it maximized the marginal relevance across individual candidate-selected pairs, effectively using minimum redundancy for each candidate. The three calls in original cell 83 were also supplied with the retrieval encoders used by that section. This implementation correction does not establish its effect on previously reported experiment results, which have not been rerun.

## Training settings visible in the source

| Encoder | Batch size | Epochs | Temperature | Validation |
| --- | ---: | ---: | ---: | --- |
| BGE-M3 | 32 | 2 | 0.05 | Every 1,000 steps |
| CLIP ViT-L/14 | 32 | 2 | 0.05 | Every 1,000 steps and at completion |

The CLIP optimizer is AdamW with learning rate 0.00001, weight decay 0.01, betas (0.9, 0.98), epsilon 0.000001 and cosine scheduling after 10% warmup. Both encoders use grouping by the shared training video/node identifier. Full original package versions and seeds are not reconstructed from result tables.

On 2026-10-03, CLIP group labels were changed from full image paths to video IDs within each source dataset. Different sampled frames from the same source video now share a label. Source dataset names prevent unrelated videos with the same local ID from being merged. This changes the grouping used by future training runs. Existing checkpoints have not been retrained by this correction.

## Packaging changes

Notebook outputs, embedded credentials and workstation proxy configuration were removed. Absolute source paths became environment-controlled paths. The misplaced future import moved to the configuration cell. Missing imports were added to the video-sampling notebook. An endpoint-specific GPT model alias became an environment setting, and the legacy judge's maximum output length became `RAG_JUDGE_MAX_TOKENS` with a default of 4096. Original source hashes and relative filenames are listed in `source_provenance.json`.

The command-line frame and caption scripts are small adaptations for use outside the notebook. The caption script fixes the original missing-file resume check. `scripts/evaluate_mc.py` implements format matching without random fallback. On 2026-10-03, the four multiple-choice sections in `legacy_evaluation.ipynb` were changed to call this shared parser. Unparsed answers are represented by `None`, counted as incorrect, and retained in the accuracy denominator. This replaces the historical random-option fallback and broad letter-matching expressions. These code changes do not constitute a rerun or rescoring of the published or revised experiments.

The Perception Test multiple-choice scorer now keeps each parsed prediction with its question ID and matches it to the reference within the same video. File listing order no longer determines which question an answer belongs to. The Video-MME and MLVU scorers already use question-ID keys, and EgoSchema has one question per video. This correction does not rescore historical results.
