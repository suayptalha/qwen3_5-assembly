#!/usr/bin/env python3
"""Build deterministic data tables for the assembly Qwen byte-level BPE tokenizer."""
import json
import struct
import unicodedata
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tok = json.loads((root/'model/tokenizer.json').read_text())
vocab=tok['model']['vocab']; merges=tok['model']['merges']
# GPT-2 byte-to-Unicode bijection.
bs=list(range(ord('!'),ord('~')+1))+list(range(0xa1,0xad))+list(range(0xae,0x100))
cs=bs.copy(); n=0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256+n); n+=1
byte_to_char={b:chr(c) for b,c in zip(bs,cs)}
char_to_byte={chr(c):b for b,c in zip(bs,cs)}
byte_ids=[vocab[byte_to_char[b]] for b in range(256)]
size=1<<19; mask=size-1
entries=bytearray(b'\xff'*(size*16))
for rank,merge in enumerate(merges):
    left,right=merge.split(' ')
    a,b=vocab[left],vocab[right]; result=vocab[left+right]
    idx=((a*0x9e3779b1)^b)&mask
    while struct.unpack_from('<I',entries,idx*16+12)[0]!=0xffffffff:
        idx=(idx+1)&mask
    struct.pack_into('<IIII',entries,idx*16,a,b,result,rank)
# ID -> byte sequence, including special tokens for lossless decode.
all_tokens=[None]*max(max(vocab.values()),max(x['id'] for x in tok['added_tokens']))
all_tokens.append(None)
for token,id_ in vocab.items():
    all_tokens[id_]=bytes(char_to_byte[c] for c in token)
for item in tok['added_tokens']:
    all_tokens[item['id']]=item['content'].encode('utf-8')
blob=bytearray(); index=bytearray()
for token in all_tokens:
    token=token or b''
    index+=struct.pack('<II',len(blob),len(token)); blob+=token
# Categories: 1 letter, 2 mark, 3 number, 4 whitespace, 5 CR/LF, 0 other.
categories=bytearray(0x110000)
for cp in range(len(categories)):
    ch=chr(cp); category=unicodedata.category(ch)[0]
    categories[cp]=5 if ch in '\r\n' else (4 if ch.isspace() else {'L':1,'M':2,'N':3}.get(category,0))
# Canonical decompositions (fully recursive), combining classes and canonical
# composition pairs. Hangul syllables are included in the same tables.
N=0x110000
decomp=bytearray(N*4); pool=[]; ccc=bytearray(N)
for cp in range(N):
    ch=chr(cp); d=unicodedata.normalize('NFD',ch)
    ccc[cp]=unicodedata.combining(ch)
    if d!=ch:
        seq=[ord(c) for c in d]
        assert len(seq)<32
        struct.pack_into('<I',decomp,cp*4,((len(pool)+1)<<5)|len(seq))
        pool.extend(seq)
compose={}
for cp in range(N):
    d=unicodedata.decomposition(chr(cp))
    if d and not d.startswith('<'):
        pieces=d.split()
        if len(pieces)==2:
            a,b=(int(x,16) for x in pieces)
            if unicodedata.normalize('NFC',chr(a)+chr(b))==chr(cp): compose[a,b]=cp
# Algorithmic Hangul compositions.
for l in range(19):
    for v in range(21):
        lv=0xac00+(l*21+v)*28
        compose[0x1100+l,0x1161+v]=lv
        for t in range(1,28): compose[lv,0x11a7+t]=lv+t
hash_size=1<<15; comp=bytearray(b'\xff'*(hash_size*12))
for (a,b),value in compose.items():
    slot=((a*0x9e3779b1)^b)&(hash_size-1)
    while struct.unpack_from('<I',comp,slot*12)[0]!=0xffffffff: slot=(slot+1)&(hash_size-1)
    struct.pack_into('<III',comp,slot*12,a,b,value)
pool_bytes=struct.pack(f'<{len(pool)}I',*pool)
header_size=64
byte_off=header_size
hash_off=byte_off+1024
idx_off=hash_off+len(entries)
blob_off=idx_off+len(index)
cat_off=blob_off+len(blob)
nfc_off=cat_off+len(categories)
decomp_off=nfc_off+40
pool_off=decomp_off+len(decomp)
ccc_off=pool_off+len(pool_bytes)
comp_off=ccc_off+len(ccc)
nfc_header=struct.pack('<5Q',decomp_off,pool_off,ccc_off,comp_off,hash_size-1)
header=struct.pack('<8Q',byte_off,hash_off,idx_off,blob_off,cat_off,mask,len(all_tokens),nfc_off)
out=root/'model/tokenizer-asm.bin'
with out.open('wb') as f:
    f.write(header); f.write(struct.pack('<256I',*byte_ids)); f.write(entries); f.write(index); f.write(blob); f.write(categories)
    f.write(nfc_header); f.write(decomp); f.write(pool_bytes); f.write(ccc); f.write(comp)
print(f'{out}: {out.stat().st_size} bytes, {len(merges)} merges, {len(all_tokens)} tokens, {len(compose)} compositions')
