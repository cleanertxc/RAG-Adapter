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
| 20 | Archived frame selection functions |
| 23, 25, 27 | Uniform-sampling API experiments |
| 30, 32, 34 | Retrieval-based API experiments |
| 37–47 | Self-supervised fine-tuning experiments |
| 50–60 | Grouped contrastive fine-tuning for BGE-M3 and CLIP |
| 62 | Load fine-tuned encoders |
| 68–92 | NIF, retrieval and ASS analysis |
| 95, 97, 99, 101 | Export selected frames for the four datasets |

Paths for raw annotations and processed frames/captions now share `RAG_DATA_ROOT/dataset`. Training JSON files and fine-tuned models are under `RAG_WORK_ROOT/finetune`. Existing training JSON files may contain absolute image paths that need remapping to the local frame tree.

The original Perception Test sampling used the training annotations. Its 90-video records should not be substituted for the validation split in the expanded evaluation. The notebooks include historical dataset splits and model settings. Check the intended split, model, input frame count and subtitle setting before running an experiment.

## Preserved retrieval behavior

The notebook retrieves visual and caption candidates separately and sums the returned scores by frame. It does not explicitly map cosine scores to [0, 1]. The original MMR helper searches over candidate-selected pairs and retains the largest pair score. This differs from computing maximum redundancy over all selected frames for each candidate. The archived helper has been preserved rather than silently changed during packaging. A new implementation or ablation should specify which selection rule it uses.

## Training settings visible in the source

| Encoder | Batch size | Epochs | Temperature | Validation |
| --- | ---: | ---: | ---: | --- |
| BGE-M3 | 32 | 2 | 0.05 | Every 1,000 steps |
| CLIP ViT-L/14 | 32 | 2 | 0.05 | Every 1,000 steps and at completion |

The CLIP optimizer is AdamW with learning rate 0.00001, weight decay 0.01, betas (0.9, 0.98), epsilon 0.000001 and cosine scheduling after 10% warmup. Both encoders use grouping by the shared training video/node identifier. Full original package versions and seeds are not reconstructed from result tables.

## Packaging changes

Notebook outputs, embedded credentials and workstation proxy configuration were removed. Absolute source paths became environment-controlled paths. The misplaced future import moved to the configuration cell. Missing imports were added to the video-sampling notebook. An endpoint-specific GPT model alias became an environment setting, and the legacy judge's maximum output length became `RAG_JUDGE_MAX_TOKENS` with a default of 4096. Original source hashes and relative filenames are listed in `source_provenance.json`.

The command-line frame and caption scripts are small adaptations for use outside the notebook. The caption script fixes the original missing-file resume check. `scripts/evaluate_mc.py` implements format matching without random fallback. These packaging changes do not constitute a rerun of the published or revised experiments.
