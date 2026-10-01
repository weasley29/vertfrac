#!/usr/bin/env bash
# Downloads the six VerSe archives listed in your dataset_link.pages, then unzips them.
# Needs ~30-40 GB free. Run from the repo root:  bash scripts/01_download_verse.sh
set -euo pipefail
DEST=data/verse; mkdir -p "$DEST/zips"
BASE=https://s3.bonescreen.de/public/VerSe-complete
for f in dataset-verse19training dataset-verse19validation dataset-verse19test \
         dataset-verse20training dataset-verse20validation dataset-verse20test; do
  [ -f "$DEST/zips/$f.zip" ] || curl -L -C - -o "$DEST/zips/$f.zip" "$BASE/$f.zip"
  mkdir -p "$DEST/$f" && unzip -q -n "$DEST/zips/$f.zip" -d "$DEST/$f"
done
echo "Done. Next: python scripts/02_inspect_dataset.py"
