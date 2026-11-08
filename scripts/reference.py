#!/usr/bin/env python3
"""Generate reproducible text-only Qwen reference tokens and logits on CPU."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model')
parser.add_argument('--prompt', default='Hello! Introduce yourself briefly.')
parser.add_argument('--tokens', type=int, default=8)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--output', default='reference.json')
args = parser.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
model_path = Path(args.model)
if not (model_path / 'model.safetensors-00001-of-00001.safetensors').exists():
    raise SystemExit('Model weights are missing or still downloading; run scripts/download_model.sh first.')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=True,
    attn_implementation='eager',
).eval()
messages = [{'role': 'user', 'content': args.prompt}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_tensors='pt', return_dict=True)
start = time.perf_counter()
with torch.inference_mode():
    result = model.generate(**inputs, do_sample=False, max_new_tokens=args.tokens,
                            return_dict_in_generate=True, output_scores=True)
elapsed = time.perf_counter() - start
prompt_len = inputs['input_ids'].shape[-1]
generated = result.sequences[0, prompt_len:].tolist()
data = {'prompt': args.prompt, 'input_ids': inputs['input_ids'][0].tolist(),
        'generated_ids': generated, 'output': tokenizer.decode(generated),
        'max_new_tokens': args.tokens, 'seconds': elapsed, 'threads': args.threads,
        'top5': [{'ids': score[0].topk(5).indices.tolist(),
                  'logits': score[0].topk(5).values.tolist()} for score in result.scores],
        'torch_version': torch.__version__}
Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(data, ensure_ascii=False, indent=2))
