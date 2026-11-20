#!/usr/bin/env bash
# Fine-tune pi0.5 on a dataset exported by `macbot export`, on a Linux machine with NVIDIA GPUs.
#
#   DATASET=~/data/cubes_to_box_bin GPUS=0,1,2,3 ./train_pi05.sh my_run
#
# Copy the dataset folder (including gripper.json for binary-gripper datasets) to the GPU machine first.
# Environment: DATASET (required), GPUS (default 0), BASE (default lerobot/pi05_base), STEPS (3000),
# SAVE (1000), BS (32 per GPU), LR (2.5e-5), WARM (300). Extra arguments go to lerobot-train.
# The rename map lines up macbot's camera names with the pi0.5 base checkpoint's image slots.
set -euo pipefail
NAME=${1:?usage: train_pi05.sh RUN_NAME [lerobot-train args]}; shift
: "${DATASET:?set DATASET to the exported dataset folder}"
GPUS=${GPUS:-0}; STEPS=${STEPS:-3000}
NG=$(echo "$GPUS" | tr "," "\n" | wc -l | tr -d " ")
HERE=$(cd "$(dirname "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false
MULTI=(); [ "$NG" -gt 1 ] && MULTI=(--multi_gpu)
accelerate launch --num_processes="$NG" --mixed_precision=bf16 "${MULTI[@]}" "$HERE/train_augmented.py" \
  --dataset.repo_id="local/$(basename "$DATASET")" --dataset.root="$DATASET" \
  --policy.path="${BASE:-lerobot/pi05_base}" --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --policy.optimizer_lr="${LR:-2.5e-5}" --policy.scheduler_warmup_steps="${WARM:-300}" \
  --policy.scheduler_decay_steps="$STEPS" --policy.scheduler_decay_lr=2.5e-6 \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --batch_size="${BS:-32}" --steps="$STEPS" --save_freq="${SAVE:-1000}" --log_freq=25 --num_workers=8 --seed=1000 \
  --output_dir="runs/$NAME" --job_name="$NAME" "$@"
python "$HERE/copy_camera_framing.py" --output_dir="runs/$NAME" "$@"
