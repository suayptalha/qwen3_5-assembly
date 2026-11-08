"""The static assembly-only executable must reproduce Transformers fixtures."""
import json
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
exe=root/'build/qwen-ids'
weights=root/'model/model.safetensors-00001-of-00001.safetensors'
for case in json.loads((root/'tests/fixtures/reference.json').read_text()):
    p=subprocess.run([exe,weights,','.join(map(str,case['input_ids'])),str(len(case['generated_ids']))],
                     capture_output=True,text=True,check=True)
    got=list(map(int,p.stdout.split()))
    assert got==case['generated_ids'], (got,case['generated_ids'])
    print(f"Standalone assembly: {case['prompt']!r}: {len(got)} tokens match")
for args in ([],[weights,'', '1'],[weights,'248320','1'],[weights,'42','0'],[weights,'x','1']):
    assert subprocess.run([exe,*map(str,args)],capture_output=True).returncode != 0
print('Standalone assembly argument checks: passed')
