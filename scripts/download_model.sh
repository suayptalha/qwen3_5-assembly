#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/model"
cd "$root/model"
base=https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main
for file in config.json tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors.index.json LICENSE; do
  test -s "$file" || curl --fail --location --retry 5 -o "$file" "$base/$file"
done
file=model.safetensors-00001-of-00001.safetensors
if test -f "$file"; then
  printf '%s  %s\n' 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696 "$file" | sha256sum -c -
  exit 0
fi
for attempt in $(seq 1 20); do
  if test -f "$file.part" && test "$(stat -c %s "$file.part")" -eq 1746942600; then
    break
  fi
  if curl --http1.1 --fail --location --retry 3 --speed-limit 10240 --speed-time 45 --continue-at - -o "$file.part" "$base/$file"; then
    break
  fi
  if test "$attempt" -eq 20; then
    echo 'Download interrupted after 20 attempts; rerun to resume.' >&2
    exit 1
  fi
  echo "Download interrupted; resuming (attempt $((attempt+1))/20)..." >&2
done
printf '%s  %s\n' 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696 "$file.part" | sha256sum -c -
mv "$file.part" "$file"
