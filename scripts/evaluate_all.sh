#!/usr/bin/env bash
# Evaluate the released checkpoints for every setting reported in the paper.
set -e
cd "$(dirname "$0")/.."

for setting in gdscv1:newDRUG gdscv1:newCELL gdscv1:newPAIR \
               gdscv2:newSCAFFOLD gdscv2:newDRUG gdscv2:newCELL \
               ccle:newSCAFFOLD; do
    dataset=${setting%%:*}
    split=${setting##*:}
    python src/main.py --dataset "$dataset" --split_mode "$split" --eval_only \
        --ckpt_dir "checkpoint/${dataset}_${split}"
done
