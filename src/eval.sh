#!/bin/bash
python evaluate.py \
  --model_name early_fusion \
  --model_path best_early_fusion_final.pth \
  --threshold 0.4
