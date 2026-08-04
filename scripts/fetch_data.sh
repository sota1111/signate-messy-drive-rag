#!/usr/bin/env bash
# Reproducibly fetch the competition data from SIGNATE into ./data (gitignored).
# Requires: signate CLI configured (~/.signate/signate.json) with a valid token.
set -euo pipefail

export LANG=C.UTF-8 LC_ALL=C.UTF-8
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
RAW="$DATA/_raw"
TASK="${SIGNATE_TASK_KEY:-90c19eddfdc746be88534b7e45fe1dd0}"
mkdir -p "$RAW"

# file_key -> filename  (from `signate file-list --task_key <TASK>`)
declare -A FILES=(
  ["d5e2eca8ea2d4ff4b2845c1e9aff5eff"]="share.zip"
  ["7a420ae6fb3f431fb881c5144a261629"]="evaluation.zip"
  ["97ee44a4a95c40fa9d6525736ecd8555"]="sample_submit.zip"
  ["0b3c641d0ee44dc4a72ff7b801d104fe"]="opening.pdf"
)

for key in "${!FILES[@]}"; do
  name="${FILES[$key]}"
  if [ ! -f "$RAW/$name" ]; then
    echo ">> downloading $name"
    signate download --task_key "$TASK" --file_key "$key" --path "$RAW/$name"
  fi
done

echo ">> unpacking"
rm -rf "$DATA/share_drive" "$DATA/questions" "$DATA/evaluation"
mkdir -p "$DATA/questions" "$DATA/_share"
unzip -o -q "$RAW/share.zip" -d "$DATA/_share"
unzip -o -q "$RAW/evaluation.zip" -d "$DATA/_eval"

DRIVE="$(find "$DATA/_share" -maxdepth 3 -type d -name '共有*' -print -quit)"
QA="$(find "$DATA/_share" -maxdepth 3 -type d -name '質問回答' -print -quit)"
cp -r "$DRIVE" "$DATA/share_drive"
cp "$QA/questions_valid.csv" "$QA/questions_test.csv" "$DATA/questions/"
cp -r "$DATA/_eval/evaluation" "$DATA/evaluation"
cp "$DATA/evaluation/data/valid_txt.csv" "$DATA/questions/valid_txt.csv"
rm -rf "$DATA/_share" "$DATA/_eval"

echo ">> done. corpus files: $(find "$DATA/share_drive" -type f | wc -l)"
