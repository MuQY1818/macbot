# LeRobot policies on macOS

[简体中文](policies_zh.md)

`macbot policies` shows the LeRobot 0.7.0 policy registry and the available test evidence.
ACT and π0.5 use their original wrappers and defaults. Other policy types use the generic
PyTorch adapter; pass `--policy-impl generic` to opt into that adapter for ACT or π0.5.

## Bring a checkpoint

Use a local LeRobot `pretrained_model` directory with `config.json`, weights,
`policy_preprocessor.json`, `policy_postprocessor.json`, and their referenced state files.
Install the matching LeRobot extra if the model needs it, for example
`pip install 'lerobot[diffusion]==0.7.0'` or `pip install 'lerobot[smolvla]==0.7.0'`.
Some models also need their original backbone and tokenizer files. Only use checkpoints and
dependencies you trust.

For a six-axis layout with two cameras and a language instruction, adapt the following to the
checkpoint's actual features. `save_size` is `[width, height]`, the reverse of the last two
dimensions in a checkpoint's RGB tensor. If camera names already match the checkpoint suffixes,
omit `camera_map`. Use a full key for nonstandard names such as `observation.image`.

```yaml
robot: {type: lerobot, follower: {type: so101_follower}}
task: {name: demo, instruction: "put the cube in the box"}
cameras:
  top: {type: uvc, save_size: [256, 256]}
  wrist: {type: uvc, save_size: [256, 256]}
policy:
  camera_map: {top: camera1, wrist: camera2}
```

```bash
macbot policies
macbot --config robot.yaml bench --policy path/to/pretrained_model --profile
macbot --config robot.yaml run --policy path/to/pretrained_model --fake --autostart 3
```

`bench` measures observation conversion through action output. Use `--device cpu` for a
CPU comparison. `--profile` adds a separate synchronized pass for observation conversion,
preprocessing, model inference, postprocessing and copying the result. In `run`, `--mode auto`
plays synchronous chunks when LeRobot declares direct chunk inference (`--k` limits playback),
or calls `select_action` once per policy tick (`record.fps × speed`) when LeRobot needs history
or owns the action queue. Each start resets policy and processor state. Generic inference
does not use async, temporal ensembling or RTC.

The checkpoint's processed actions must be **absolute joint positions**, in the same order
and units as the robot. A matching width does not establish those semantics. For real hardware,
set `policy.action_space: joint_position` after confirming them. macbot does not do inverse
kinematics or translate between joint conventions. Known end-effector layouts, missing required
cameras, extra sensor modalities and non-finite outputs are rejected. Explicit model padding
cameras are left to the policy to create and mask. The original ACT/π0.5 paths are unaffected.

## MPS measurements

MacBook Pro, Apple M5 Pro (48 GiB), macOS 27.0.1, PyTorch 2.11.0, LeRobot 0.7.0,
Python 3.12. All generic timings include saved processors and transfer to CPU; no hardware was
connected. The paired sweep uses five warm-ups and 30 calls on deterministic synthetic RGB and
state. CPU uses fp32 and four threads. For the error pass, both devices use the same CPU normal
draws and categorical sampling; the latency pass uses native sampling. Memory is peak process
RSS, not a GPU allocator peak. Step timings mix queue hits and the more expensive replans.

| Policy / checkpoint | Execution | MPS median / p90 (ms) | CPU median / p90 (ms) | Max absolute action difference | MPS peak RSS (GiB) |
|---|---|---:|---:|---:|---:|
| ACT, tiny | Generic, 3-step chunk | 7.21 / 7.93 | 6.02 / 6.16 | 1.31e-6 | 0.66 |
| Diffusion, tiny DDIM | Step, replan every 3 calls | 1.44 / 22.24 | 0.11 / 12.43 | 3.10e-6 | 0.80 |
| VQ-BeT, tiny | Step, replan every 3 calls | 1.18 / 7.26 | 0.07 / 4.71 | 2.38e-7 | 0.69 |
| TDMPC, tiny | Step, replan every 2 calls | 5.30 / 9.42 | 1.06 / 2.23 | 2.38e-7 | 0.64 |

These random-weight runs verified the data path, not task performance. VQ-BeT and TDMPC each
require one camera; TDMPC uses `observation.image`. VQ-BeT needs a trained Residual VQ for
meaningful actions. Existing stock π0.5 numbers are in [inference benchmarks](acceleration.md).

To repeat a tiny checkpoint sweep from the source checkout:

```bash
python -m tools.tiny_policies diffusion /tmp/diffusion
# Use two 64 × 64 cameras named top and wrist in robot.yaml.
for device in cpu mps; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.policy_sweep /tmp/diffusion \
    --config robot.yaml --device "$device" --out "/tmp/diffusion-$device.npy"
done
python -c 'import numpy as n; print(abs(n.load("/tmp/diffusion-cpu.npy")-n.load("/tmp/diffusion-mps.npy")).max())'
```
