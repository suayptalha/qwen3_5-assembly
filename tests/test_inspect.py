import json
import struct
import subprocess
import tempfile
from pathlib import Path

exe = Path(__file__).resolve().parents[1] / 'build/inspect-safetensors'
header = b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'test.safetensors'
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0'*4)
    assert json.loads(subprocess.check_output([exe, path]))['a']['shape'] == [1]
    path.write_bytes(struct.pack('<Q', 1000000) + b'{}')
    assert subprocess.run([exe, path], capture_output=True).returncode != 0
    path.write_bytes(b'abc')
    assert subprocess.run([exe, path], capture_output=True).returncode != 0
    assert subprocess.run([exe, str(path) + '.missing'], capture_output=True).returncode != 0
    assert subprocess.run([exe], capture_output=True).returncode != 0
print('Freestanding syscall/mmap safetensors header inspector: all tests passed')
