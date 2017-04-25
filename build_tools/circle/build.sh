#!/bin/bash
# Generates api.md file
python $(pwd)/build_tools/circle/api.py
mkdocs build
