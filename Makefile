CC ?= cc
CFLAGS ?= -O2
.PHONY: all test test-model clean
all: build/qwen build/qwen-ids build/libkernels.so build/inspect-safetensors
build/libkernels.so: src/kernels.S src/deltanet.S src/attention.S src/decoder.S src/model.S src/sampling.S
	mkdir -p build
	$(CC) -nostdlib -shared -fPIC -mavx2 -mfma -Wl,-z,defs -o $@ $^
build/inspect-safetensors: src/inspect.S
	mkdir -p build
	$(CC) -nostdlib -static -Wl,--build-id=none -o $@ $<
test: all
	python3 tests/test_kernels.py
	python3 tests/test_sampling.py
	python3 tests/test_inspect.py
test-model: all build/qwen-ids build/qwen build/libtokenizer.so
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_real_weights.py
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_deltanet_reference.py
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_real_deltanet.py
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_real_attention.py
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_e2e.py
	python3 tests/test_runner.py
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_tokenizer.py
	python3 tests/test_text_runner.py
	python3 tests/test_utf8_stream.py
	OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_chat.py
clean:
	rm -rf build
build/model_offsets.S: scripts/generate_offsets.py model/model.safetensors-00001-of-00001.safetensors
	mkdir -p build
	python3 scripts/generate_offsets.py

build/qwen-ids: src/runner.S src/kernels.S src/deltanet.S src/attention.S src/decoder.S src/model.S src/sampling.S build/model_offsets.S
	$(CC) -nostdlib -static -no-pie -mavx2 -mfma -Wl,--build-id=none -o $@ $^

model/tokenizer-asm.bin: scripts/build_tokenizer_data.py model/tokenizer.json
	python3 scripts/build_tokenizer_data.py

build/libtokenizer.so: src/tokenizer.S
	mkdir -p build
	$(CC) -nostdlib -shared -fPIC -Wl,-z,defs -o $@ $<

build/qwen: src/runner.S src/tokenizer.S src/kernels.S src/deltanet.S src/attention.S src/decoder.S src/model.S src/sampling.S build/model_offsets.S model/tokenizer-asm.bin
	$(CC) -nostdlib -static -no-pie -mavx2 -mfma -Wa,--defsym,TEXT_MODE=1,--defsym,TOKEN_TABLE_BYTES=$$(stat -c %s model/tokenizer-asm.bin) -Wl,--build-id=none -o $@ $(filter %.S,$^)
