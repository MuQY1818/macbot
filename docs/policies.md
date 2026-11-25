# Policies

[简体中文](policies.zh.md)

macbot runs LeRobot 0.7.0 checkpoints. Every policy runs on PyTorch with the Apple GPU (MPS). Some also have an
MLX backend, which is usually faster. This page lists what works today and how to bring your own checkpoint.
[Acceleration](acceleration.md) explains how the MLX backends are checked and has the full measurements.

## Support at a glance

Latency is the median for one policy call on an Apple M5 Pro, from the observation to the action chunk. The two
numbers are PyTorch MPS and MLX.

| Policy | MLX | `auto` uses MLX | MPS → MLX (ms) | Checked on |
|---|---|---|---:|---|
| ACT | yes | chunk | 16 → 8 | ALOHA, public |
| π0.5 | yes | chunk, RTC | 270 → 165 | SO-101, public |
| π0 | yes | chunk, RTC | 368 → 160 | SO-101, public |
| SmolVLA | yes | chunk | 296 → 53 | base model |
| X-VLA | yes | chunk | 178 → 138 | SO-101, public |
| VQ-BeT | yes | no | 7.7 → 4.2 | PushT, public |
| Diffusion | yes, slower | no | 819 → 1960 | PushT, public |
| π0-FAST | yes | no | — → 885 | LIBERO, offline |
| TD-MPC | yes | no | — | tiny, random |

`auto` picks MLX only where MLX has matched PyTorch on a trained checkpoint of that architecture, in the listed
modes. Everywhere else it stays on PyTorch, and you can still ask for MLX with `--backend mlx`.

- **VQ-BeT** on MLX matches PyTorch to 0.0002 pixels when both run on the CPU. On the GPU, small float32
  differences in the convolutions change about 0.3 pixels of a 512-pixel action on average, so `auto` leaves it on
  PyTorch.
- **Diffusion** on MLX is correct but slower than PyTorch for the 100-step public checkpoint. Use PyTorch.
- **π0-FAST** on MLX decodes actions, but its action tokens differ from PyTorch's. The test checkpoint outputs
  end-effector actions, so it ran in an offline harness, not on an arm. PyTorch latency was not measured.
- **TD-MPC** has no public checkpoint in the LeRobot 0.7 format, so only the data path is tested.

Gaussian actor and Multi-Task DiT run on PyTorch and were tested with tiny random checkpoints. GR00T, WALL-X and
WALL-OSS need checkpoints and backbone files that have not been tried. The remaining LeRobot types (`cosmos3_edge`, `cosmos3_nano`, `dm05`, `eo1`, `evo1`, `fastwam`, `fineart_vla`, `flux3`,
`g05`, `lawam`, `lingbot_va`, `lingbot_vla_v2`, `longwam`, `molmoact2`, `vla_jepa`) load through LeRobot's factory,
but no checkpoint of theirs has been run. `macbot policies` prints the same table for your installation.

A result on one checkpoint does not carry over to every checkpoint of the same type. Benchmark yours with
`macbot bench` before you rely on it.

## Bring your checkpoint

You need a LeRobot `pretrained_model` folder: `config.json`, the weights, `policy_preprocessor.json`,
`policy_postprocessor.json` and the state files they refer to. Merge PEFT adapters into the base policy before you
export. Install the LeRobot extra the model needs, for example `pip install 'lerobot[smolvla]==0.7.0'`. Only use
checkpoints and dependencies you trust.

1. **Check the actions.** The checkpoint must output absolute joint positions, in the same order and units as your
   arm. macbot rejects known end-effector layouts and a state or action width that does not match the arm, but a
   matching width does not prove the meaning. Once you have confirmed it, set `policy.action_space: joint_position`.

2. **Map the cameras.** Name each camera in your config and map it to the checkpoint's image key. `save_size` is
   `[width, height]`. You can leave out `camera_map` when the names already match.

    ```yaml
    robot: {type: lerobot, follower: {type: so101_follower}}
    task: {name: demo, instruction: "put the cube in the box"}
    cameras:
      top: {type: uvc, save_size: [256, 256]}
      wrist: {type: uvc, save_size: [256, 256]}
    policy:
      camera_map: {top: camera1, wrist: camera2}
    ```

3. **Measure, then try it without the arm.**

    ```bash
    CKPT=path/to/pretrained_model
    macbot --config robot.yaml bench --policy "$CKPT" --backend auto --profile
    macbot --config robot.yaml run --policy "$CKPT" --backend auto --fake --autostart 3
    ```

    `bench` feeds blank images and never opens a camera or the arm. `--profile` adds a separate pass that times
    preprocessing, inference and postprocessing. `--fake` runs the whole loop against a simulated arm.

`--policy` also takes a Hub ID such as `owner/model@revision`.

### Notes for some policies

| Policy | What to know |
|---|---|
| ACT | Runs with temporal ensembling by default, which stays on PyTorch. `--mode chunk` lets `auto` use MLX. |
| π0.5 | Needs at least 32 GB of memory. Supports RTC with `--mode rtc` on both backends. |
| π0 | Supports RTC with `--mode rtc` on MLX only. |
| π0, π0-FAST | Use the gated PaliGemma tokenizer. If your account has no access, set `MACBOT_PALIGEMMA_TOKENIZER` to a local PaliGemma `tokenizer.model`. |
| SmolVLA | `lerobot/smolvla_base` expects three 256 × 256 cameras, mapped to `camera1`, `camera2` and `camera3`. |
| X-VLA | The tested checkpoint uses two cameras mapped to `image` and `image2`. RTC and the `ee6d` base model are not supported. |
| VQ-BeT, TD-MPC | Take one camera. TD-MPC names it `observation.image`, so map it with the full key. VQ-BeT needs a trained residual VQ for useful actions. |

ACT and π0.5 use their original LeRobot wrappers. Other types go through a generic adapter that calls LeRobot's
`select_action` or chunk inference; `--policy-impl generic` uses the adapter for ACT and π0.5 too. The generic
adapter does not do async execution, temporal ensembling or RTC.

## If a model fails on MPS

Try `--device cpu` first to see whether the problem is specific to MPS. To let PyTorch run unsupported operators on
the CPU, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before you start macbot and watch for operator warnings. The fallback
cannot fix a missing model file, a shape mismatch or float64 math inside the model. None of the measurements on
these pages needed the fallback.

??? note "Generic adapter: MPS against CPU"

    MacBook Pro with an Apple M5 Pro (48 GiB), macOS 27.0.1, PyTorch 2.11.0, LeRobot 0.7.0 and Python 3.12. Each
    timing includes the saved processors and the copy back to the CPU. Five warm-ups, then 30 calls for tiny
    policies, 15 for SmolVLA and X-VLA and five for π0, on fixed synthetic images and state. The CPU runs float32
    on four threads. For the action difference, both devices use the same random draws. Step timings mix calls
    served from the action queue with calls that replan.

    | Policy / checkpoint | Execution | MPS median / p90 (ms) | CPU median / p90 (ms) | Max action difference | Peak RSS (GiB) |
    |---|---|---:|---:|---:|---:|
    | ACT, tiny | 3-step chunk | 7.21 / 7.93 | 6.02 / 6.16 | 1.31e-6 | 0.66 |
    | Diffusion, tiny DDIM | replan every 3 calls | 1.44 / 22.24 | 0.11 / 12.43 | 3.10e-6 | 0.80 |
    | VQ-BeT, tiny | replan every 3 calls | 1.18 / 7.26 | 0.07 / 4.71 | 2.38e-7 | 0.69 |
    | TD-MPC, tiny | replan every 2 calls | 5.30 / 9.42 | 1.06 / 2.23 | 2.38e-7 | 0.64 |
    | Gaussian actor, tiny | continuous | 3.70 / 3.97 | 0.60 / 0.67 | 4.77e-7 | 0.62 |
    | Multi-Task DiT, tiny DDIM | replan every 3 calls | 1.74 / 7.63 | 0.12 / 1.47 | 4.77e-7 | 0.64 |
    | π0, random 2B prefix, two denoising steps | 3-step chunk | 490.01 / 492.80 | 2150.15 / 2219.61 | 2.68e-6 | 16.74 |
    | SmolVLA, official base | 50-step chunk | 297.80 / 309.43 | 1126.30 / 1164.04 | 0.0316 | 3.38 |
    | X-VLA, SO-101 two-camera | 30-step chunk | 181.89 / 205.54 | 6226.21 / 7688.11 | 1.22 | 4.93 |

    The tiny and random-weight runs check the data path, not task performance. SmolVLA and X-VLA keep their saved
    bf16 settings on MPS, so their differences from float32 are larger. Differences are in each checkpoint's action
    units.

    To repeat a tiny sweep from a source checkout:

    ```bash
    python -m tools.tiny_policies diffusion /tmp/diffusion
    # Use two 64 × 64 cameras named top and wrist in robot.yaml.
    for device in cpu mps; do
      PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.policy_sweep /tmp/diffusion \
        --config robot.yaml --device "$device" --out "/tmp/diffusion-$device.npy"
    done
    python -c 'import numpy as n; print(abs(n.load("/tmp/diffusion-cpu.npy")-n.load("/tmp/diffusion-mps.npy")).max())'
    ```
