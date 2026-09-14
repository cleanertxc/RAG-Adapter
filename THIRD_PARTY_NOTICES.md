# Third-party components

The captioning script adapts the [CogVLM2 CLI demo](https://github.com/zai-org/CogVLM2), distributed under Apache-2.0. Its license is included in `third_party/CogVLM2-Apache-2.0.txt`. The release adds command-line paths, resume support and run metadata to the frame-captioning loop.

The CLIP embedding wrapper and fine-tuning engine interfaces in the experiment notebook build on [LlamaIndex](https://github.com/run-llama/llama_index), distributed under the MIT license. The license notice is included in `third_party/LlamaIndex-MIT.txt`.

Other dependencies are installed from their upstream projects and retain their own licenses. Model checkpoints and videos are downloaded separately under their upstream terms. This repository does not redistribute model weights or source videos.
