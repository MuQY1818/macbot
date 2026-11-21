# LeRobot policies on macOS

[简体中文](policies_zh.md)

`macbot policies` shows the LeRobot 0.7.0 policy registry and the available test evidence.
ACT and π0.5 use their original wrappers and defaults. Other policy types use the generic
PyTorch adapter; pass `--policy-impl generic` to opt into that adapter for ACT or π0.5.

`run` and `bench` accept `--backend auto` or `policy: {backend: auto}` in YAML.
`policies` lists implemented backends independently of installed extras and
[lossless qualification](acceleration.md#automatic-backend-selection).

## Bring a checkpoint

Use a local LeRobot `pretrained_model` directory with `config.json`, weights,
`policy_preprocessor.json`, `policy_postprocessor.json`, and their referenced state files.
For adapter-only PEFT exports, merge the adapter into the base policy before exporting.
Install the matching LeRobot extra if the model needs it, for example
`pip install 'lerobot[diffusion]==0.7.0'`, `pip install 'lerobot[smolvla]==0.7.0'` or
`pip install 'lerobot[xvla]==0.7.0'`.
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

For `lerobot/smolvla_base`, include a third 256 × 256 camera mapped to `camera3`.
For the tested two-camera X-VLA checkpoint, map `top` and `wrist` to `image` and `image2`.

The experimental π0 `--backend mlx` path accepts a stock 224px PaliGemma 2B checkpoint
and uses `policy.camera_map` and the saved processors. It provides `--mode chunk` and
guided `--mode rtc`, and defaults to chunks. Trained checkpoint fidelity and speed are
not verified. Use offline checks and `--fake` to evaluate it.

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

## Coverage

There are three levels. “Verified” means a public trained checkpoint ran on MPS and produced a
fake-arm rollout, or the policy uses a stock wrapper with separate verification. “Runs” means the
data path was exercised on MPS with a tiny or random-weight checkpoint, so latency and accuracy
for a trained model of that type are still unmeasured. “Not verified” means nothing was executed.
None of this means other checkpoints of the same type are interchangeable. The LeRobot factory resolved model classes for all 29 registered types
on this installation. Loading a class does not load or execute weights.

| Policy types | Status and scope |
|---|---|
| `act`, `pi05` | Verified stock wrappers. Generic ACT also tested. |
| `diffusion`, `vqbet`, `tdmpc`, `gaussian_actor` | Runs with the tiny offline checkpoints below. Gaussian actor was tested with continuous actions. |
| `pi0` | Runs with a six-joint random checkpoint using the 2B PaliGemma prefix. The 300M prefix fails on both devices because vision and text embedding widths differ. |
| `pi0_fast` | Loaded and generated tokens on CPU and MPS with a local tokenizer bundle. Both produced random tokens that failed FAST action decoding; no compatible trained checkpoint was tested. |
| `smolvla` | Verified with `lerobot/smolvla_base`: three RGB inputs, six state/action values, 50-step chunks. |
| `xvla` | Verified with `whosricky/xvla-so101-pick-red-cube-2cam`. The adapter passes six state values to the model's own padding and leaves the masked third slot absent. The `ee6d` base is unsupported. |
| `groot` | Not verified. GR00T N1.7 needs native base assets plus a compatible embodiment and saved processors. |
| `wall_x`, `wall_oss_05` | Not verified. Separate model families; require a compatible checkpoint and its matching backbone and processors. |
| `cosmos3_edge`, `cosmos3_nano`, `dm05`, `eo1`, `evo1`, `fineart_vla`, `g05`, `lawam`, `lingbot_vla_v2`, `molmoact2`, `vla_jepa` | Factory import only. No robot-specific checkpoint and backbone were exercised. |
| `fastwam`, `flux3`, `lingbot_va`, `longwam` | Factory import only. Video or action history and checkpoint-specific processor behavior need separate validation. |
| `multi_task_dit` | Runs with a tiny DDIM checkpoint and a local random CLIP model and tokenizer. |

The measurements shown below did not require `PYTORCH_ENABLE_MPS_FALLBACK`. If an untested
checkpoint hits an unsupported PyTorch MPS operator, try `--device cpu`. To diagnose a possible
CPU fallback separately, set `PYTORCH_ENABLE_MPS_FALLBACK=1` **before** launching macbot
and check the operator warnings. Fallback cannot fix a missing model asset, shape mismatch,
or float64 created and used inside a model. The adapter overrides standard device-processor
steps and converts their explicit float64 setting to float32 on MPS.

## MPS measurements

MacBook Pro, Apple M5 Pro (48 GiB), macOS 27.0.1, PyTorch 2.11.0, LeRobot 0.7.0,
Python 3.12. All generic timings include saved processors and transfer to CPU; no hardware was
connected. The paired sweep uses five warm-ups, then 30 calls for tiny policies, 15 for
SmolVLA and X-VLA, or five for π0, on deterministic synthetic RGB and state. CPU uses fp32 and four threads.
For the error pass, both devices use the same CPU normal
draws and categorical sampling; the latency pass uses native sampling. Memory is peak process
RSS, not a GPU allocator peak. Step timings mix queue hits and the more expensive replans.

| Policy / checkpoint | Execution | MPS median / p90 (ms) | CPU median / p90 (ms) | Max absolute action difference | MPS peak RSS (GiB) |
|---|---|---:|---:|---:|---:|
| ACT, tiny | Generic, 3-step chunk | 7.21 / 7.93 | 6.02 / 6.16 | 1.31e-6 | 0.66 |
| Diffusion, tiny DDIM | Step, replan every 3 calls | 1.44 / 22.24 | 0.11 / 12.43 | 3.10e-6 | 0.80 |
| VQ-BeT, tiny | Step, replan every 3 calls | 1.18 / 7.26 | 0.07 / 4.71 | 2.38e-7 | 0.69 |
| TDMPC, tiny | Step, replan every 2 calls | 5.30 / 9.42 | 1.06 / 2.23 | 2.38e-7 | 0.64 |
| Gaussian actor, tiny | Step, continuous | 3.70 / 3.97 | 0.60 / 0.67 | 4.77e-7 | 0.62 |
| Multi-Task DiT, tiny DDIM | Step, replan every 3 calls | 1.74 / 7.63 | 0.12 / 1.47 | 4.77e-7 | 0.64 |
| π0, random 2B prefix, fp32, two denoising steps | 3-step chunk | 490.01 / 492.80 | 2150.15 / 2219.61 | 2.68e-6 | 16.74 |
| SmolVLA, official base | 50-step chunk | 297.80 / 309.43 | 1126.30 / 1164.04 | 0.0316 | 3.38 |
| X-VLA, SO-101 two-camera fine-tune | 30-step chunk | 181.89 / 205.54 | 6226.21 / 7688.11 | 1.22 | 4.93 |

These random-weight runs verified the data path, not task performance. VQ-BeT and TDMPC each
require one camera; TDMPC uses `observation.image`. VQ-BeT needs a trained Residual VQ for
meaningful actions. Existing stock π0.5 numbers are in [inference benchmarks](acceleration.md).
SmolVLA uses bf16 in its vision-language backbone, and X-VLA uses its saved bf16 configuration
on MPS. These paired differences against fp32 are in the output units of each checkpoint.
Compare them to your application's tolerance before using the Mac backend on hardware.

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
