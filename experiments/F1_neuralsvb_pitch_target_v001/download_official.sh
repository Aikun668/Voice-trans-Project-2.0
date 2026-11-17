#!/bin/bash
# Download official demo audio (GT amateur / GT professional / NSVB) for local research. Data dir is separate from scripts.
set -u
cd "$(dirname "$0")"
for lang in Chinese English; do
  for d in gt_a gt_p NSVB; do
    mkdir -p "official_all/$lang/$d"
    for f in $(gh api "repos/neuralsvb/neuralsvb.github.io/contents/resources/audio/$lang/$d" --jq '.[].name'); do
      [ -s "official_all/$lang/$d/$f" ] && continue
      gh api "repos/neuralsvb/neuralsvb.github.io/contents/resources/audio/$lang/$d/$f" --jq .content | base64 -d > "official_all/$lang/$d/$f"
    done
  done
done
echo DONE
