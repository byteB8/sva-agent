#!/bin/bash
# Fetch the CodeV-SVA authors' corrected FVEval tasks, used only to calibrate
# this scorer against their published Jasper scores. Data only: none of the
# repository's code is run. The repository has no license file, so the data
# is fetched for local evaluation and not redistributed.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/CodeV-SVA ]; then
  git clone -q https://github.com/wyt2000/CodeV-SVA.git data/CodeV-SVA
fi
git -C data/CodeV-SVA checkout -q 764cb3d5746e
echo "CodeV-SVA at $(git -C data/CodeV-SVA rev-parse --short HEAD)"
