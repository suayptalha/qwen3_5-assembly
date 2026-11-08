"""Multi-turn state vs a CPU Transformers reference, plus sampling CLI."""
import json
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
exe=root/'build/qwen'
case=json.loads((root/'tests/fixtures/chat_reference.json').read_text())
conversation='\n'.join(case['questions']+['/exit'])+'\n'
p=subprocess.run([exe,'--chat',str(case['max_tokens']),'0','40'],cwd=root,
                 input=conversation,text=True,capture_output=True,check=True,timeout=180)
expected=''.join(o['text']+'\n' for o in case['outputs'])
assert p.stdout==expected,(p.stdout,expected,p.stderr)
assert p.stderr.count('You: ') == len(case['questions']) + 1 and p.stderr.count('Qwen: ') == len(case['questions']), p.stderr
print('Two-turn greedy chat matches the PyTorch reference for both turns')
case2=json.loads((root/'tests/fixtures/reference.json').read_text())[1]
p=subprocess.run([exe,'--sample',case2['prompt'],'8','70','1'],cwd=root,
                 text=True,capture_output=True,check=True,timeout=120)
# Reference fixture contains 12 tokens; compare against its first 8 decoded tokens.
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
expected=t.decode(case2['generated_ids'][:8],skip_special_tokens=True)+'\n'
assert p.stdout==expected,(p.stdout,expected)
print('Sampling with top-k=1 matches greedy generation')
for args in (['--chat','0'],['--chat','10','201'],['--chat','10','70','0'],
             ['--chat','10','70','101'],['--sample'],['--sample','x','10','-1']):
    p=subprocess.run([exe,*args],cwd=root,text=True,capture_output=True,timeout=10)
    assert p.returncode != 0,(args,p.stdout,p.stderr)
print('Chat and sampling argument validation passed')
