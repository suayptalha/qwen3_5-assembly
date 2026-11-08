"""End-to-end greedy regression against saved CPU Transformers reference."""
import json
import os
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
fixtures=json.loads((root/'tests/fixtures/reference.json').read_text())
env=dict(os.environ, OPENBLAS_NUM_THREADS='1')
for case in fixtures:
    p=subprocess.run([sys.executable,str(root/'scripts/prototype.py'), '--prompt',case['prompt'],
                      '--tokens',str(len(case['generated_ids']))],cwd=root,env=env,
                     text=True,capture_output=True,check=True)
    got=json.loads(p.stdout)
    assert got['input_ids']==case['input_ids'], (got['input_ids'],case['input_ids'])
    assert got['generated_ids']==case['generated_ids'], (got['generated_ids'],case['generated_ids'])
    print(f"{case['prompt']!r}: {len(got['generated_ids'])} tokens match: {got['text']!r}")
