#!/bin/bash
# Generates api.md file
export PATH="$HOME/miniconda3/bin:$PATH"
source activate testenv
python $(pwd)/build_tools/circle/api.py
mkdocs build
cp -r ./site ${CIRCLE_ARTIFACTS}
