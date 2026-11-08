"""End-to-end text tests for the freestanding assembly executable."""
import json
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
exe=root/'build/qwen'
for case in json.loads((root/'tests/fixtures/reference.json').read_text()):
    result=subprocess.run([exe,case['prompt'],str(len(case['generated_ids']))],cwd=root,
                          capture_output=True,check=True)
    got=result.stdout.decode('utf-8')
    assert got==case['output']+'\n',(got,case['output'])
    print(f"Assembly text runner: {case['prompt']!r}: {len(case['generated_ids'])} reference tokens match")
for args in ([],['test','0'],['test','-1'],['test','x'],['test','262144']):
    assert subprocess.run([exe,*args],cwd=root,capture_output=True).returncode != 0,args
print('Text runner argument checks passed')
