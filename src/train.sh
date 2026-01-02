#!/bin/bash

MODEL=$1

if [ -z "$MODEL" ]; then
  echo "Usage: ./train.sh [early_fusion|mid_fusion|late_fusion|image_only|text_only]"
  exit 1
fi

python train.py \
  --model $MODEL
