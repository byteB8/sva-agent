#!/bin/bash
# Fetch the FVEval benchmark (NVIDIA, Apache-2.0) at the commit the results use.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/FVEval ]; then
  git clone -q https://github.com/NVlabs/FVEval.git data/FVEval
fi
git -C data/FVEval checkout -q 141afe7
echo "FVEval at $(git -C data/FVEval rev-parse --short HEAD)"
