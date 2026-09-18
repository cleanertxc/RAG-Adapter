# GCL retriever checkpoints

The two existing GCL retrievers referenced by the archived RAG-Adapter implementation are prepared for distribution through Google Drive. Their download links will be added to the [repository README](../README.md#fine-tuned-encoder-checkpoints) after upload. Weight files are not distributed through GitHub Releases.

| Archive | Contents | Initialization |
| --- | --- | --- |
| `clip-vit-l14-gcl.tar.gz` | `clip_vit_l14_gcl/clip_best_finetuned_gc.pth`, weight checksum and license | OpenAI CLIP ViT-L/14 |
| `bge-m3-gcl.tar.gz` | `bge_m3_finetuned_gc/`, including `model.safetensors`, tokenizer, pooling and normalization configuration, weight checksum and license | BAAI/bge-m3 |

CLIP includes both its image and text towers. The BGE-M3 checkpoint supplies dense embeddings for question-caption retrieval. It does not include separately trained sparse or ColBERT heads. The archives do not include CogVLM2 or downstream answering models.

## Download and extract

Once the Google Drive links are available, download both archives into a local `checkpoints` directory. From the repository directory, verify and extract them with:

```bash
cd checkpoints
sha256sum -c ../docs/checkpoint_metadata/SHA256SUMS.txt
tar -xzf clip-vit-l14-gcl.tar.gz
tar -xzf bge-m3-gcl.tar.gz
cd ..
```

The combined extracted weight files occupy approximately 3.2 GB. Allow additional space for the downloaded archives, tokenizer and configurations. The [checkpoint manifest](checkpoint_metadata/checkpoint-manifest.json) records exact file sizes and SHA-256 hashes of both archives and original weight files.

## Load CLIP

Use the [OpenAI CLIP package](https://github.com/openai/CLIP):

```python
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
visual_encoder, preprocess = clip.load("ViT-L/14", device=device, jit=False)
state = torch.load(
    "checkpoints/clip_vit_l14_gcl/clip_best_finetuned_gc.pth",
    map_location="cpu",
    weights_only=True,
)
visual_encoder.load_state_dict(state, strict=True)
visual_encoder.eval()

# preprocess(PIL_image) prepares an RGB frame.
# visual_encoder.encode_image(...) embeds frames.
# visual_encoder.encode_text(clip.tokenize(...).to(device)) embeds questions.
```

The initial `clip.load` call obtains the upstream model if it is not cached. Loading the released state dictionary then replaces the model parameters with the GCL checkpoint.

## Load BGE-M3

```python
from sentence_transformers import SentenceTransformer

caption_encoder = SentenceTransformer(
    "checkpoints/bge_m3_finetuned_gc",
    device="cpu",  # Use "cuda" for GPU inference.
)
embeddings = caption_encoder.encode(
    ["What is the person doing?", "A person is cooking."],
    normalize_embeddings=True,
)
```

The saved model contains its tokenizer and SentenceTransformers modules. Its original training metadata records SentenceTransformers 3.1.1, Transformers 4.40.2 and PyTorch 2.2.2+cu121. The model's configured maximum sequence length is 8192.

The notebook's embedding wrappers can load the extracted paths directly, after their definitions and imports have been run:

```python
clip_gc = ClipEmbedding(
    model_name="ViT-L/14",
    ck_path="checkpoints/clip_vit_l14_gcl/clip_best_finetuned_gc.pth",
)
bge_m3_gc = resolve_embed_model("local:checkpoints/bge_m3_finetuned_gc")
```

These lines replace the notebook's fine-tuned model-loading cell. Loading the released retrievers does not require running the training cells.

## Training configuration and provenance

Both retrieval branches use batch size 32, two epochs, GCL temperature 0.05 and cosine scheduling with 10% warmup. CLIP updates both towers using AdamW with learning rate 1e-5, weight decay 0.01, betas (0.9, 0.98) and epsilon 1e-6. The lowest validation-loss checkpoint is retained.

The archived BGE-M3 call uses SentenceTransformers 3.1.1 `fit` defaults for AdamW with learning rate 2e-5 and weight decay 0.01, excluding bias and LayerNorm parameters from decay. Retrieval validation is performed every 1,000 steps and at epoch completion, with the best checkpoint selected using the evaluator's primary MAP@100 metric. See the training functions in the experiment notebook and [notebook guide](notebooks.md).

The release preserves the original weight files without retraining or changing their precision. BGE-M3's configuration initialization name is normalized from a workstation path to `BAAI/bge-m3`. The archived automatically generated model card is replaced with a short loading description because its generic Trainer defaults do not describe the externally supplied optimizer and scheduler. Packaging and loading checks do not rerun benchmark results or establish a complete per-run provenance record.

The upstream checkpoints are [OpenAI CLIP](https://github.com/openai/CLIP) and [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Their MIT license notices accompany the archives and are retained under `third_party/` in the repository.
