#!/usr/bin/env python3
"""Text interface for the standalone assembly inference executable.

The tokenizer and chat template are provided by Transformers. Inference runs
in build/qwen-ids, which is handwritten x86-64 assembly with Linux syscalls.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('prompt', nargs='?', default='Hello! Introduce yourself briefly.')
p.add_argument('--tokens', type=int, default=32)
args = p.parse_args()
if args.tokens < 1:
    p.error('--tokens must be positive')
tokenizer = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
ids = tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],
                                   tokenize=True, add_generation_prompt=True,
                                   enable_thinking=False)['input_ids']
if len(ids)+args.tokens > 262144:
    p.error('the current assembly runner supports at most 262144 prompt + output tokens')
command = [str(root/'build/qwen-ids'), str(root/'model/model.safetensors-00001-of-00001.safetensors'),
           ','.join(map(str,ids)), str(args.tokens)]
t0 = time.perf_counter()
try:
    result = subprocess.run(command, text=True, capture_output=True, check=True)
except FileNotFoundError:
    raise SystemExit('Build the runner first: make build/qwen-ids')
except subprocess.CalledProcessError as e:
    raise SystemExit(e.stderr or f'Runner failed with status {e.returncode}')
tokens = [int(i) for i in result.stdout.split()]
print(tokenizer.decode(tokens, skip_special_tokens=True))
print(f'[{len(ids)} input tokens, {len(tokens)} output tokens, {time.perf_counter()-t0:.2f}s total]', file=sys.stderr)
