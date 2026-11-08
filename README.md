# Qwen3.5-0.8B in x86-64 assembly

A CPU inference engine for [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), written in x86-64 assembly for Linux. The tokenizer, Unicode normalization, chat template, model forward pass, sampling and text decoding all run in assembly. The executable is statically linked and uses Linux syscalls directly: **no Python, libc, GPU or ML framework at runtime**.

Inspired by [Arnesh Banerjee's gpt2.s](https://github.com/ArneshBanerjee/gpt2.s), which runs GPT-2 in hand-written AArch64 assembly.

## Benchmark

On a **Google Colab CPU runtime with an AMD EPYC 7B12**, this implementation produced tokens approximately **1.50× faster** than the tested Hugging Face Transformers/PyTorch configuration:

| Implementation | Run 1 | Run 2 | Median |
|---|---:|---:|---:|
| x86-64 assembly | 6.42 tok/s | 6.60 tok/s | **6.51 tok/s** |
| Transformers / PyTorch | 4.39 tok/s | 4.30 tok/s | **4.34 tok/s** |

Both used one CPU thread, the same 19-token prompt, greedy decoding and 32 output tokens. The generated token IDs matched in both runs. Decode speed is `(output tokens − 1) / time from the first output token to the last`.

The assembly runner reads BF16 weights and computes in FP32; the Transformers reference loaded the BF16 checkpoint as FP32, using Transformers 5.22.2 and PyTorch 2.14.1+cpu. Transformers used its PyTorch reference fallbacks for the gated delta and causal convolution operations. The implementations therefore use different weight storage formats, and cache warmth and shared runtime load may affect the result. This is a two-run measurement on one prompt, not a general speed claim. First-token times are excluded from the comparison because their measurement boundaries differed.

## Quick start

Requirements: Linux x86-64 with **AVX2 and FMA**, GNU assembler/GCC, `make`, `curl`, `sha256sum`, and Python 3.11 or newer for build-time data preparation. The model download is approximately 1.75 GB.

```bash
# Download the checkpoint and tokenizer, then verify the checkpoint SHA-256.
bash scripts/download_model.sh

# Generate tokenizer tables and checkpoint offsets, then build the executable.
make

# Start an interactive session.
./build/qwen --chat
```

Python is used to prepare the checkpoint offsets and tokenizer tables during the build. The resulting executable does not use Python at runtime. Unicode tables are generated using the build machine's Python Unicode database. The model checkpoint and tokenizer tables are separate files mapped from disk; neither is included in the executable.

## Usage

In chat mode, enter another message after each response. Type `/exit` or press Ctrl+D to quit. The attention cache and DeltaNet state are retained during the session.

```bash
./build/qwen --chat                # 64 output tokens, temperature 0.70, top-k 40
./build/qwen --chat 128 70 40      # max tokens, temperature percent, top-k
```

Temperature is specified as a percentage: `70` means 0.70. The valid range is 0–200; use `0` for greedy decoding. Top-k must be between 1 and 100.

For a single prompt:

```bash
# Greedy decoding; the output limit defaults to 32 tokens.
./build/qwen 'Hello! Introduce yourself briefly.' 32

# Sampling: prompt, max tokens, temperature percent, top-k.
./build/qwen --sample 'Tell me a short story.' 64 80 40
```

Output streams to the terminal. A single-prompt sampling invocation defaults to 32 output tokens, temperature 0.70 and top-k 40. The prompt must be valid UTF-8.

By default, the program reads the files under `model/` relative to the current working directory. For a single prompt, paths can be supplied explicitly:

```bash
./build/qwen model/model.safetensors-00001-of-00001.safetensors \
  model/tokenizer-asm.bin 'What is 2 + 2?' 32
```

For debugging, the token-ID runner accepts comma-separated input IDs and prints one generated ID per line:

```bash
./build/qwen-ids model/model.safetensors-00001-of-00001.safetensors 42 1
```

## Validation

```bash
make test        # assembly kernels, sampling and file reader
make test-model  # real checkpoint, reference layers, tokenizer and end-to-end tests
```

The model tests use CPU PyTorch and Transformers from a local `.venv` as the reference; the kernel tests require NumPy. They compare real-weight DeltaNet and full-attention layer outputs, greedy generation for three prompts, and both responses in a two-turn conversation. They also test sampling, tokenizer behavior, Unicode normalization and UTF-8 output. Exact token agreement is established only for the inputs tested.

For an assembly-only local timing run:

```bash
python3 scripts/benchmark.py --runs 2 --tokens 32
```

This measures the standalone token-ID runner on one CPU thread. It writes `benchmark-results.json`; its results depend on the prompt, CPU, context length, cache state and system load.

## Limitations

- Text-only inference; the checkpoint's vision encoder is not implemented.
- The total conversation and requested output must fit within **262,144 tokens**, including room for turn-ending tokens. Old messages are not automatically dropped.
- Conversation state is kept in memory only and is lost when the process exits.
- The runner targets the exact Qwen3.5-0.8B checkpoint verified by the build; it is not a general-purpose model loader.
- The executable size excludes the approximately 1.75 GB checkpoint, the approximately 19.4 MB tokenizer table, and runtime memory used for caches and scratch space.

## Source layout

- `src/tokenizer.S` — UTF-8, NFC, pre-tokenization, byte-level BPE and decoding.
- `src/kernels.S` — AVX2/FMA operations, normalization, activations and attention primitives.
- `src/deltanet.S`, `src/attention.S`, `src/decoder.S`, `src/model.S` — model layers and forward pass.
- `src/runner.S` — memory mapping, chat sessions and syscall-based entry point.
- `src/sampling.S` — temperature and top-k sampling.
- `scripts/build_tokenizer_data.py`, `scripts/generate_offsets.py` — build-time data preparation.

## License

This project is licensed under the [Apache License 2.0](LICENSE). The Qwen3.5-0.8B model is distributed under its own Apache 2.0 license; see `model/LICENSE` after downloading.
