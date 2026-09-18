# RAG-Adapter

Research code for **RAG-Adapter: A Plug-and-Play RAG-enhanced Framework for Long Video Understanding**.

RAG-Adapter selects question-relevant frames for a downstream video model. A visual retrieval branch and a caption retrieval branch retrieve candidates from the same video. Their scores are combined before frame selection. Frame captions are generated offline without the downstream questions and can be reused across questions about that video.

## Contents

| Path | Contents |
| --- | --- |
| `notebooks/rag_adapter_experiments.ipynb` | Original retrieval, GCL fine-tuning, frame selection, GPT answering and benchmark diagnostics |
| `notebooks/frame_extraction.ipynb` | Original per-dataset frame extraction and MMAT preparation |
| `notebooks/video_sampling.ipynb` | Original sampling code |
| `notebooks/legacy_evaluation.ipynb` | Historical evaluation code and GPT judging prompts |
| `scripts/extract_frames.py` | Command-line frame extraction with a frame manifest |
| `scripts/caption_frames.py` | Question-independent CogVLM2 caption generation |
| `scripts/evaluate_mc.py` | Multiple-choice option parsing and accuracy calculation |
| `data/sampled_records/` | The 90-video sampling records for each of four benchmarks |

The notebooks retain the original research logic and include independent experiment sections. Read [the notebook guide](docs/notebooks.md) before executing cells. The GCL retriever checkpoints are available through the Google Drive links below. A complete environment and per-run configurations for every reported experiment are not included. No paper results have been recomputed during repository preparation.

## Fine-tuned encoder checkpoints

The two GCL retrievers are hosted on Google Drive. This repository contains the code, loading instructions and checkpoint metadata.

| Retrieval branch | Checkpoint | Download |
| --- | --- | --- |
| Question to visual frame | CLIP ViT-L/14, fine-tuned image and text towers | [CLIP checkpoint file](https://drive.google.com/file/d/192JC513L0txbHmzlAnU-V8QHfbyIA7li/view?usp=sharing) |
| Question to frame caption | BGE-M3, fine-tuned dense text encoder | [BGE-M3 checkpoint folder](https://drive.google.com/drive/folders/1tnh0CbHN9jHyhqjpKh18FqFhCUdsP1Zs?usp=drive_link) |

These are the GCL retrievers used by the archived retrieval implementation. [Loading instructions](docs/checkpoints.md) describe the local file layout and model loading. [SHA-256 checksums](docs/checkpoint_metadata/SHA256SUMS.txt) and a [checkpoint manifest](docs/checkpoint_metadata/checkpoint-manifest.json) provide reference hashes for the original weight files. CogVLM2 and downstream answering models are obtained from their upstream repositories.

## Installation

Use a separate Python environment. Python 3.10 or 3.11 is suitable for inspecting the source and running the lightweight utilities.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the research notebooks, install PyTorch appropriate for your CUDA installation, then install the notebook dependencies and [OpenAI CLIP](https://github.com/openai/CLIP).

```bash
python -m pip install -r requirements-research.txt
python -m pip install git+https://github.com/openai/CLIP.git
```

`requirements-research.txt` lists imported dependencies and is not an environment lockfile. The original BGE-M3 training used SentenceTransformers 3.1.1. The notebooks depend on LlamaIndex interfaces whose compatibility must be checked against the installed version. The captioning script additionally requires a compatible CogVLM2 environment, including `accelerate` and `bitsandbytes`. Follow the [upstream CogVLM2 setup](https://github.com/zai-org/CogVLM2/tree/main/basic_demo) for its custom model implementation.

## Prepare frames and captions

Download videos and annotations from the benchmark providers. Place them outside the tracked `data/` directory. For example:

```text
data_local/dataset/Video-MME/
  videos/<video_id>.mp4
  frames/<video_id>/000001.jpg
  captions/<video_id>/000001.txt
  test-00000-of-00001.parquet
```

Extract approximately one frame per second using the original integer-frame indexing convention:

```bash
python scripts/extract_frames.py \
  --video data_local/dataset/Video-MME/videos/example.mp4 \
  --output data_local/dataset/Video-MME/frames/example
```

Frame filenames use the original one-based ordinal naming. The adjacent JSON manifest records actual source frame indices and timestamps, so the filename does not need to be interpreted as a timestamp.

Generate captions with the [CogVLM2 int4 checkpoint](https://huggingface.co/zai-org/cogvlm2-llama3-chat-19B-int4):

```bash
python scripts/caption_frames.py \
  --frames data_local/dataset/Video-MME/frames \
  --captions data_local/dataset/Video-MME/captions
```

The fixed prompt is **“Please describe the picture briefly.”** Each invocation uses one image and an empty conversation history. The script does not accept downstream questions or answers. Existing nonempty captions are reused. The checkpoint configuration and generation overrides are saved alongside the caption directory.

## Retrieval and fine-tuning

The notebook configuration cell accepts these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_ROOT` | `data_local/` | Benchmark annotations, frames, captions and results |
| `RAG_WORK_ROOT` | `runs/` | Training datasets, fine-tuned models and retrieval indices |
| `RAG_BGE_MODEL` | `BAAI/bge-m3` | Base caption retrieval encoder |

Open `notebooks/rag_adapter_experiments.ipynb` and run the relevant data loader, model definitions and experiment section. Visual retrieval uses CLIP ViT-L/14 and caption retrieval uses BGE-M3. The archived main setting retrieves up to 50 candidates from each branch, adds scores for a common frame ID and selects the requested frame budget. The grouped training functions and their configuration cells are retained in the same notebook. See [the notebook guide](docs/notebooks.md) for the cell map and implementation details.

## Multiple-choice evaluation

The shared answering template is:

```text
Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter of the correct option.
[Question]
[Labeled answer options]
The best answer is:
```

Save one JSON object per line with `question_id`, `options`, `answer`, and `response`. `answer` must be an option letter. For example:

```json
{"question_id":"example-1","options":["First option","Second option","Third option"],"answer":"B","response":"The best answer is B."}
```

```bash
python scripts/evaluate_mc.py predictions.jsonl --output runs/accuracy.json
```

The evaluator extracts a valid option letter using format matching. A missing, invalid or conflicting answer is counted as incorrect and remains in the denominator. This is the current evaluation rule. The archived evaluation notebook preserves earlier parsing behavior for inspection and is not used by this command.

For archived API experiments, set `OPENAI_API_KEY` in the environment. The SDK uses its default endpoint unless `OPENAI_BASE_URL` is explicitly configured. Keys are never stored in the source. API calls can incur charges. The historical GPT judging prompts are in `notebooks/legacy_evaluation.ipynb` and concern the MLVU generation tasks rather than the main multiple-choice comparison.

## Sampling records and validation

The [sampling record guide](data/sampled_records/README.md) describes the four CSV files and video lists. Google Drive links for the fine-tuned retrievers are listed above. Benchmark videos and other model weights are obtained from their original providers.

```bash
python -m unittest discover -s tests -v
```

Repository checks cover option parsing, missing-response handling, notebook syntax and sampling-record counts. Full GPU inference and training have not been rerun as part of packaging this release.

## Acknowledgments

The implementation builds on [CLIP](https://github.com/openai/CLIP), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [CogVLM2](https://github.com/zai-org/CogVLM2), [SentenceTransformers](https://github.com/huggingface/sentence-transformers), [LlamaIndex](https://github.com/run-llama/llama_index), and [Qdrant](https://github.com/qdrant/qdrant). See [third-party notices](THIRD_PARTY_NOTICES.md) for copied or adapted components and their licenses.
