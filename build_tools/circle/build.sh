#!/bin/bash
# Generates api.md file
python3 $(pwd)/build_tools/circle/api.py
mkdocs build
