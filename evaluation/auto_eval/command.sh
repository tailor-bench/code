#!/usr/bin/env bash
set -euo pipefail

# all image models
python run_generation_models.py --models image --json prompts.json

# all video models
python run_generation_models.py --models video --json prompts.json

# print commands without running
python run_generation_models.py --models all --json prompts.json --dry-run

# run specific models
python run_generation_models.py \
  --models qwen wan veo3 \
  --json prompts.json \
  --results-dir results \
  --skip-existing
