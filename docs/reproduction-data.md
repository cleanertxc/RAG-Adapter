# MMAT manifests and frame captions

Use the Google Drive links below to download MMAT captions and annotations for data inspection and retriever training. Cached captions for the sampled diagnostic experiments are available in the [reproduction data release](https://github.com/cleanertxc/RAG-Adapter/releases/tag/reproduction-data-20260919). Fine-tuned encoder weights are also hosted on Google Drive, as listed in [the checkpoint guide](checkpoints.md).

| Download | Contents |
| --- | --- |
| [MMAT captions](https://drive.google.com/file/d/1c1FM8PIXsjjPUfHTGKUaZlpnn6GwtdyN/view?usp=drive_link) | MMAT caption files |
| [MMAT annotations](https://drive.google.com/file/d/1-lM2bMZwv2UcaT7ZvhCP0cUHju9WIvo2/view?usp=drive_link) | MMAT annotations |
| [Sampled frame captions](https://github.com/cleanertxc/RAG-Adapter/releases/download/reproduction-data-20260919/sampled-frame-captions.zip) | Cached candidate-frame captions for the 90 sampled videos in each of four benchmarks, with sampling and NIF records |
| [Release manifest](https://github.com/cleanertxc/RAG-Adapter/releases/download/reproduction-data-20260919/release-manifest.json) | Counts, original manifest hashes, archive sizes and SHA-256 checksums |
| [Archive checksums](https://github.com/cleanertxc/RAG-Adapter/releases/download/reproduction-data-20260919/SHA256SUMS.txt) | Reference checksums for the archived GitHub packages, including the sampled-frame-caption archive |

## MMAT records

The counts and schemas below document the archived MMAT training records. Obtain MMAT captions and annotations from the two Google Drive links above. The archived release checksums apply to the original GitHub packages.

`MMAT/train_dataset.json` contains 417,993 question records and 51,726 video groups. `MMAT/test_dataset.json` contains 109,799 question records and 10,190 video groups. The latter filename is preserved from the archive and is used for retrieval validation in the research notebook.

Both JSON files retain the original bytes and structure:

* `queries`: UUID to question text.
* `corpus`: archived video/group ID to its retained frame caption.
* `relevant_docs`: question UUID to the associated video/group ID.
* `mode`: the original text-mode marker.

`MMAT/video_groups.csv` maps these groups to MSVD-QA, MSRVTT-QA, ActivityNet-QA or TGIF-QA. Its `matching_caption_files` field locates the cached caption with text matching each corpus entry. All 61,916 corpus entries across both manifests have an exact match after trimming surrounding whitespace. The archive includes 61,508 unique caption files. `MMAT/caption_files.csv` records source, video ID, frame filename, relative path, byte count and checksum.

The archived partitions share 408 TGIF video IDs, listed in `MMAT/overlapping_video_ids.txt`. This release preserves the historical records and does not silently make the partitions video-disjoint. An experiment using corrected partitions should save a new version and retrain. These JSON files specify the caption training pairs. They do not recover unsaved random quartile-frame choices for historical CLIP training. The construction notebook supplies the visual-pair preparation procedure.

## Evaluation captions and sampling records

The sampled-frame-caption archive contains `dataset/<dataset>/captions/<video_id>/<frame_id>.txt`, with original caption bytes and filenames. Its video IDs match the four lists in [`data/sampled_records`](../data/sampled_records/). These are the sampled diagnostic sets, not caption caches for every video in the expanded main evaluation.

* Video-MME: 90 videos and 270 question records, with five videos in each duration/domain combination.
* MLVU: 90 videos and 156 question records across nine tasks. Its diagnostic sample includes the two generation tasks.
* EgoSchema: 90 videos and 90 questions.
* Perception Test: the 90 longest videos in the archived training-split sample, with 341 questions.

The sampled records also supply NIF counts and annotated evidence timestamps for oracle-frame evaluation. `metadata/*_caption_files.csv` lists every cached caption and its checksum. Frame IDs are the original filename stems. They should not be treated as exact timestamps without consulting the extraction convention or frame manifest. Caption files contain descriptions, not a new set of question-specific annotations.

The captions are archived outputs. Their generation call used one image, an empty history and the fixed prompt `Please describe the picture briefly.` The archive does not contain a complete log of later manual edits. Packaging preserves the available files without regenerating or rewriting them.

## Loading

Download MMAT captions and annotations from Google Drive and extract any archives. Place the annotation manifests at `data_local/MMAT/train_dataset.json` and `data_local/MMAT/test_dataset.json`, and retain the caption paths referenced by their records.

For the sampled diagnostic captions, download `sampled-frame-captions.zip` and `SHA256SUMS.txt` from the GitHub links above. Verify the downloaded archive and extract it into the same data directory:

```bash
sha256sum --ignore-missing -c SHA256SUMS.txt
unzip sampled-frame-captions.zip -d data_local
mkdir -p runs/finetune/video_mme
cp data_local/MMAT/train_dataset.json runs/finetune/video_mme/
cp data_local/MMAT/test_dataset.json runs/finetune/video_mme/
```

Set `RAG_DATA_ROOT` to the extracted `data_local` directory and `RAG_WORK_ROOT` to `runs` when running the research notebook. Obtain videos and images from the benchmark providers and reconstruct the expected frame tree with the extraction scripts. The archives contain text annotations and captions only. Source dataset terms continue to apply. No weights, raw media, API credentials or new benchmark predictions are included.
