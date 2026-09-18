# Third-party components

The captioning script adapts the [CogVLM2 CLI demo](https://github.com/zai-org/CogVLM2), distributed under Apache-2.0. Its license is included in `third_party/CogVLM2-Apache-2.0.txt`. The release adds command-line paths, resume support and run metadata to the frame-captioning loop.

The CLIP embedding wrapper and fine-tuning engine interfaces in the experiment notebook build on [LlamaIndex](https://github.com/run-llama/llama_index), distributed under the MIT license. The license notice is included in `third_party/LlamaIndex-MIT.txt`.

The GCL retriever archives prepared for Google Drive distribution derive from [OpenAI CLIP ViT-L/14](https://github.com/openai/CLIP) and [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Their upstream MIT license notices are included as `third_party/CLIP-MIT.txt` and `third_party/FlagEmbedding-MIT.txt` and accompany the corresponding archives. The BGE-M3 model card identifies its license as MIT and refers to FlagEmbedding for the project implementation.

Other dependencies retain their upstream licenses. Source benchmark videos, CogVLM2 and downstream answering model weights are obtained separately from their providers.
