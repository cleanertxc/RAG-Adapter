"""Generate question-independent captions with the original CogVLM2 prompt.

Adapted from the CogVLM2 basic CLI demo, Apache-2.0.
See third_party/CogVLM2-Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
"""
import argparse
import json
from pathlib import Path

PROMPT = 'Please describe the picture briefly.'
CHECKPOINT = 'zai-org/cogvlm2-llama3-chat-19B-int4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--captions', type=Path, required=True)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not args.frames.is_dir():
        parser.error('--frames must be an existing directory')
    paths = sorted(p for p in args.frames.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not paths:
        parser.error('No input images found')
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available():
        parser.error('This int4 CogVLM2 configuration requires a CUDA GPU')
    set_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=dtype, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True), low_cpu_mem_usage=True,
    ).eval()
    generation = {'max_new_tokens': 2048, 'pad_token_id': 128002, 'top_k': 1}
    args.captions.mkdir(parents=True, exist_ok=True)
    metadata = {'checkpoint': args.checkpoint, 'prompt': PROMPT, 'seed': args.seed,
                'generation_overrides': generation,
                'checkpoint_generation_config': model.generation_config.to_dict()}
    config_path = args.captions.with_name(args.captions.name + '_config.json')
    config_path.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    for image_path in paths:
        output = args.captions / image_path.relative_to(args.frames).with_suffix('.txt')
        if output.exists() and output.read_text(encoding='utf-8').strip():
            continue
        with Image.open(image_path) as image:
            conversation = model.build_conversation_input_ids(
                tokenizer, query=PROMPT, history=[], images=[image.convert('RGB')], template_version='chat')
        inputs = {key: conversation[key].unsqueeze(0).to('cuda')
                  for key in ('input_ids', 'token_type_ids', 'attention_mask')}
        inputs['images'] = [[conversation['images'][0].to('cuda').to(dtype)]]
        with torch.inference_mode():
            tokens = model.generate(**inputs, **generation)
        caption = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(caption, encoding='utf-8')
        print(image_path.relative_to(args.frames))


if __name__ == '__main__':
    main()
