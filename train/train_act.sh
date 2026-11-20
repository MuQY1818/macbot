#!/usr/bin/env bash
# Train ACT on a dataset exported by `macbot export` (CUDA by default; DEVICE=mps is untested).
#
#   DATASET=~/macbot-data/datasets/cubes_to_box_bin ./train_act.sh my_act
#
# Environment: DATASET (required), DEVICE (cuda), STEPS (40000), BS (8), SAVE (10000).
set -euo pipefail
NAME=${1:?usage: train_act.sh RUN_NAME [lerobot-train args]}; shift
: "${DATASET:?set DATASET to the exported dataset folder}"
lerobot-train --policy.type=act --policy.device="${DEVICE:-cuda}" --policy.push_to_hub=false \
  --dataset.repo_id="local/$(basename "$DATASET")" --dataset.root="$DATASET" \
  --batch_size="${BS:-8}" --steps="${STEPS:-40000}" --save_freq="${SAVE:-10000}" \
  --output_dir="runs/$NAME" --job_name="$NAME" "$@"
if [ -f "$DATASET/meta/camera_framing.json" ]; then
  find "runs/$NAME/checkpoints" -type d -name pretrained_model \
    -exec cp "$DATASET/meta/camera_framing.json" '{}/camera_framing.json' \;
fi
