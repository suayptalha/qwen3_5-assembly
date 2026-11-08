"""Check the assembly output stream decoder across token boundaries."""
import subprocess
import tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
chunks=[b'ASCII ',b'\xf0\x9f',b'\x91\x8b',b' ',b'\xe2',b'\x28',b'\xa1',b'\xed\xa0\x80',b'\xf0\x9f']
lines=['.intel_syntax noprefix','.text','.globl _start','_start:']
for i,c in enumerate(chunks):
    lines+= [f' lea rdi, [rip+chunk{i}]',f' mov esi, {len(c)}',' call emit_utf8']
lines+=[' call flush_utf8',' xor edi, edi',' mov eax, 60',' syscall','.section .rodata']
for i,c in enumerate(chunks): lines+=[f'chunk{i}: .byte '+','.join(str(x) for x in c)]
lines+=['.section .note.GNU-stack,"",@progbits']
with tempfile.TemporaryDirectory() as d:
    d=Path(d); driver=d/'driver.S'; runner=d/'runner.o'; exe=d/'test'
    driver.write_text('\n'.join(lines)+'\n')
    size=(root/'model/tokenizer-asm.bin').stat().st_size
    subprocess.run(['cc','-c','-no-pie',f'-Wa,--defsym,TEXT_MODE=1,--defsym,TOKEN_TABLE_BYTES={size}',str(root/'src/runner.S'),'-o',str(runner)],check=True)
    subprocess.run(['objcopy','--redefine-sym','_start=qwen_unused_start',str(runner)],check=True)
    sources=[root/'src'/n for n in ('tokenizer.S','kernels.S','deltanet.S','attention.S','decoder.S','model.S','sampling.S')]
    subprocess.run(['cc','-nostdlib','-static','-no-pie','-mavx2','-mfma','-o',str(exe),str(driver),str(runner),*(str(s) for s in sources),str(root/'build/model_offsets.S')],check=True)
    actual=subprocess.check_output([exe])
expected=b''.join(chunks).decode('utf-8','replace').encode('utf-8')
assert actual==expected,(actual,expected)
print('Streaming UTF-8: split, malformed and truncated sequences passed')
