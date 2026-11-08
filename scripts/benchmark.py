#!/usr/bin/env python3
"""Measure standalone single-thread CPU prefill and decode; model stays assembly."""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs', type=int, default=2)
p.add_argument('--tokens', type=int, default=32)
p.add_argument('--output', default='benchmark-results.json')
args = p.parse_args()
if args.runs < 1 or args.tokens < 2:
    p.error('--runs must be at least 1 and --tokens at least 2')
fixture = json.loads((root/'tests/fixtures/reference.json').read_text())[0]
ids = fixture['input_ids']
if len(ids) + args.tokens > 262144:
    p.error('prompt and requested output exceed 262144 tokens')
cmd = [str(root/'build/qwen-ids'), str(root/'model/model.safetensors-00001-of-00001.safetensors'),
       ','.join(map(str, ids)), str(args.tokens)]
runs=[]
for i in range(args.runs):
    start=time.perf_counter()
    proc=subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    tokens=[]; stamps=[]
    for line in proc.stdout:
        tokens.append(int(line)); stamps.append(time.perf_counter())
    stderr=proc.stderr.read().decode(errors='replace')
    code=proc.wait()
    if code:
        raise SystemExit(f'benchmark run {i+1} failed ({code}): {stderr}')
    if len(stamps)<2:
        raise SystemExit('model emitted fewer than 2 tokens, so decode speed cannot be measured')
    first=stamps[0]-start
    decode=stamps[-1]-stamps[0]
    record={'run':i+1,'prompt_tokens':len(ids),'generated_tokens':len(tokens),
            'time_to_first_token_s':first,'decode_time_s':decode,
            'decode_tokens_per_s':(len(tokens)-1)/decode,'output_token_ids':tokens}
    runs.append(record)
    print(f"Run {i+1}: {first:.3f}s to first token; {record['decode_tokens_per_s']:.2f} tokens/s "
          f"({len(tokens)-1} decode tokens in {decode:.3f}s)", flush=True)
cpu='unknown'
for line in Path('/proc/cpuinfo').read_text().splitlines():
    if line.startswith('model name'):
        cpu=line.split(':',1)[1].strip(); break
result={'cpu':cpu,'model':'Qwen/Qwen3.5-0.8B','weight_dtype':'BF16',
        'runtime':'build/qwen-ids (standalone x86-64 assembly)',
        'thread_count':1,'decoding':'greedy','prompt':fixture['prompt'],
        'prompt_token_ids':ids,'requested_output_tokens':args.tokens,
        'measurement':'Wall clock to first token includes process startup and prefill; decode rate is subsequent tokens divided by elapsed time from first to last token.',
        'runs':runs,'median_decode_tokens_per_s':statistics.median(x['decode_tokens_per_s'] for x in runs)}
(root/args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(f"Median decode: {result['median_decode_tokens_per_s']:.2f} tokens/s; saved to {args.output}")
