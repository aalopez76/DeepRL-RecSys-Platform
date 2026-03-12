#!/usr/bin/env bash
# Download MovieLens dataset
set -euo pipefail

DATA_DIR="${1:-data/movielens}"
mkdir -p "$DATA_DIR"

echo "Downloading MovieLens 100K..."
curl -fsSL https://files.grouplens.org/datasets/movielens/ml-100k.zip -o "$DATA_DIR/ml-100k.zip"
unzip -o "$DATA_DIR/ml-100k.zip" -d "$DATA_DIR"
echo "Done. Files in $DATA_DIR/ml-100k/"
