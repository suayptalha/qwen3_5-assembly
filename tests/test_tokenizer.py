import ctypes
import json
import unicodedata
from pathlib import Path
import random
from transformers import AutoTokenizer
root=Path(__file__).resolve().parents[1]
lib=ctypes.CDLL(str(root/'build/libtokenizer.so'))
P=ctypes.c_void_p
lib.tokenize.argtypes=[P,ctypes.c_size_t,P,ctypes.c_size_t,P]
lib.tokenize.restype=ctypes.c_int64
lib.token_bytes.argtypes=[ctypes.c_uint32,P,ctypes.POINTER(P)]
lib.token_bytes.restype=ctypes.c_int64
data=ctypes.create_string_buffer((root/'model/tokenizer-asm.bin').read_bytes())
reference=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
cases=['', 'Hello, world!', 'Merhaba! Kısaca kendini tanıt.', '2 + 2 kaç eder?',
       'İstanbul’da yağmur yağıyor. Çığ, şüphe ve öğrenme! 😃',
       "I'm testing we're it'll DON'T", 'hello   world  ', '\n\n x  \r\n y',
       '１２３4.5', '中文日本語 한국어 العربية', 'x<think>hi</think>y', '<|im_start|>user\n',
       'a\tb\u00a0c', '👋🏽 🌍✨', 'ä e\u0301 x', 'a  \nb']
cases += [item['content'] for item in json.loads((root/'model/tokenizer.json').read_text())['added_tokens']]
rng=random.Random(3)
chars=list("a zABCçğşüéä中文1０9!?.-'\"@#$\n\t\r \u00a0👋")+list('\u0300\u0301\u0327\u0315\u0345각각ÅÅḊạ')
cases += [''.join(rng.choice(chars) for _ in range(rng.randrange(1,75))) for _ in range(300)]
for text in cases:
    raw=text.encode(); inp=ctypes.create_string_buffer(raw); out=(ctypes.c_uint32*32768)()
    n=lib.tokenize(inp,len(raw),out,len(out),data)
    assert n>=0,(text,n)
    got=list(out[:n]); expected=reference.encode(text,add_special_tokens=False)
    assert got==expected,(repr(text),got,expected)
    decoded=b''
    for id in got:
        ptr=P(); size=lib.token_bytes(id,data,ctypes.byref(ptr)); assert size>=0
        decoded+=ctypes.string_at(ptr,size)
    assert decoded==unicodedata.normalize("NFC",text).encode(),(text,decoded)
for raw in (b'\xc0\x80', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xe2\x82'):
    inp=ctypes.create_string_buffer(raw); out=(ctypes.c_uint32*16)()
    assert lib.tokenize(inp,len(raw),out,len(out),data)==-1,raw
raw=b'one two three'; inp=ctypes.create_string_buffer(raw); out=(ctypes.c_uint32*1)()
assert lib.tokenize(inp,len(raw),out,1,data)==-1
raw=b'1'*40000
inp=ctypes.create_string_buffer(raw); out=(ctypes.c_uint32*40000)()
n=lib.tokenize(inp,len(raw),out,len(out),data)
digit_id=reference.encode('1',add_special_tokens=False)[0]
assert n == 40000 and all(x == digit_id for x in out), n
print(f'{len(cases)} tokenizer cases, long input, UTF-8 validation, and capacity checks passed')
