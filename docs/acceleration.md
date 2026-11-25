# Acceleration

[简体中文](acceleration.zh.md)

PyTorch on the Apple GPU (MPS) is the default backend. For several policies macbot also has an MLX backend that
reads the same LeRobot checkpoint, uses its saved tokenizer and processors, and is usually faster. This page shows
how to use MLX, how a backend is checked before `auto` picks it, and the measurements behind each row of the
[support table](policies.md#support-at-a-glance).

## Use MLX

Install the extra on an Apple silicon Mac, then compare the two backends on your checkpoint:

```bash
pip install 'macbot[mlx]'
CKPT=path/to/pretrained_model
macbot --config robot.yaml bench --policy "$CKPT" --backend torch -n 30
macbot --config robot.yaml bench --policy "$CKPT" --backend mlx -n 30
```

`bench` feeds blank images and never opens a camera or the arm. `--profile` adds a separate timed pass per stage,
so the stages do not add up to the end-to-end time. `--no-compile` turns off the graph compiler.

To deploy, pass `--backend auto` or `--backend mlx` to `macbot run`, or set `policy: {backend: auto}` in the config.

- `--backend auto` uses MLX only for the architectures and modes in the support table and prints the checkpoint
  that qualified it. Otherwise it prints one line saying why it stays on PyTorch.
- `--backend mlx` uses MLX whenever the policy has an MLX port, whether or not it qualified.
- With MLX, `--mode auto` runs chunks asynchronously without guidance. `--mode rtc` runs guided RTC on MLX for
  π0 and π0.5.

The default backend and the default execution modes do not change when MLX is installed.

## What "lossless" means

A faster backend must give the same actions as the reference for the same weights, observations and starting
noise. We compare actions after the postprocessor, in degrees for arm joints and in the checkpoint's units for the
gripper, and report the mean and the 99th percentile.

- For a deterministic policy such as ACT, the joint error must stay within 0.01° for both statistics, and the
  gripper error within 0.1% of its range.
- For a policy that samples noise, the errors must also be smaller than the difference between two reference runs
  with different seeds.
- For π0 and π0.5, the reference is a float32 run. MLX must be faster and either meet the absolute limits above or
  be at least as close to float32 as the bf16 PyTorch path that macbot deploys.

Matching on these frames does not prove that every rollout is the same, and none of these comparisons measures task
success on a robot.

## How `auto` decides

`auto` checks the policy type, the saved processor layout, the execution mode, the compile setting and whether MLX
is installed. It picks MLX only when all of these match a configuration that passed the comparison above on a public
or recorded checkpoint. It does not check the machine, so the timings here describe an M5 Pro and nothing more.

## Faster but lossy

These options trade accuracy for speed. Check them against your own checkpoint before you use them.

| Option | Effect |
|---|---|
| `--steps N` | Fewer denoising steps for π0.5, and for π0 on MLX. |
| `--mlx-time-grid quadratic` | Spends the denoising steps near the clean end. Better than uniform steps at the same count. |
| `--mlx-bits 8` or `4`, with `--mlx-quantize prefix`, `expert` or `both` | Quantized weights. 8-bit saved memory but was slower; 4-bit raised the error a lot. |
| `--torch-autocast fp16` or `bf16` | Lower precision for ACT and Diffusion on MPS. |

Two more PyTorch MPS experiments change how the model runs, not its precision: `--torch-channels-last` changes the
Conv2d weight layout, and `--torch-single-readback` drops a redundant device barrier in ACT. `--torch-compile` compiles Diffusion's denoiser
as an experiment; ACT compilation is disabled because the tested MPS build failed to generate its kernels.
`auto` never turns any of these on.

## Measurements

Unless a section says otherwise: Mac17,9 (Apple M5 Pro, 20 GPU cores, 48 GiB), macOS 27.0.1, MLX 0.32.4 and
LeRobot 0.7.0, one benchmark process at a time, at least five warm-ups and 50 timed calls. Timings run from the
observation to the action on the CPU. Errors are mean / p99.

??? note "π0.5 on recorded observations (qualifies chunk and RTC)"

    25 frames from episodes 0, 12, 24, 36 and 49 of
    `zz4321/so101_cube@a34feb2f19718483fd6db7fac1d47927c84a5b4e`, with
    `zz4321/so101_pi05@dbd6d5754bbf78318808aea1706ff23beeb8b663`, PyTorch 2.11.0. Recorded state, images and task go
    through the saved processors. RTC reuses a float32 plan from four frames earlier, shifted four steps, with a
    20-step guidance horizon and linear guidance capped at 10.

    The reference is MPS float32 eager with seed 941 (1057.14 / 1068.51 ms for chunks, 1093.42 / 1111.33 ms for RTC).
    The bf16 rows use macbot's compiled path with the checkpoint's `max-autotune` wrapper disabled, because it fails
    on this MPS build. MLX runs compiled float16. Joint angles come from the normalized outputs with a per-joint
    scale of `span × 360 / (4095 × 200)` degrees, using spans of 2337, 2431, 2273, 2182 and 3887 encoder ticks. The
    gripper is in its 0–100 units.

    | Mode | Backend | Median / p90 ms | Joint ° | Gripper |
    |---|---|---:|---:|---:|
    | Chunk | Torch bf16, compiled | 330.30 / 335.94 | 0.401 / 3.961 | 0.111 / 0.790 |
    | Chunk | MLX fp16, compiled | 222.81 / 229.14 | 0.074 / 0.886 | 0.017 / 0.158 |
    | Chunk | Torch fp32, seed 942 | 1072.99 / 1087.34 | 11.425 / 104.201 | 4.034 / 23.821 |
    | RTC | Torch bf16, compiled | 346.87 / 367.02 | 0.342 / 4.532 | 0.069 / 0.637 |
    | RTC | MLX fp16, compiled | 226.67 / 277.24 | 0.056 / 0.841 | 0.009 / 0.113 |
    | RTC | Torch fp32, seed 942 | 1087.41 / 1102.96 | 8.798 / 100.508 | 2.552 / 22.653 |

    Both modes qualify through the bf16 comparison: MLX is closer to float32 than bf16 on all four statistics. The
    joint p99 is above the absolute 0.50° cap. Measured directly against bf16, MLX differs by 0.393 / 3.965° for
    chunks and 0.330 / 4.257° for RTC.

??? note "π0.5 on our own checkpoint, and where the time goes"

    An earlier test used our own SO-101 checkpoint, 50 recorded observations from 10 episodes, 30 timed calls and
    PyTorch 2.11.0, with bf16 as the reference. A different starting noise measures the policy's own variation.

    | Configuration | Median / p90 ms | Peak RSS / GPU GiB | Arm, all ° | Arm, first 10 ° | Gripper, all | Gripper, first 10 |
    | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
    | MPS bf16 | 297.71 / 311.91 | 12.38 / 9.16 | reference | reference | reference | reference |
    | MPS bf16, same noise again | 276.83 / 291.30 | 15.01 / 9.17 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
    | MPS bf16, other noise | 279.03 / 284.38 | 15.19 / 9.17 | 0.626 / 8.037 | 0.274 / 1.179 | 0.0060 / 0.0071 | 0.0057 / 0.0064 |
    | MPS fp32, eager | 923.30 / 980.37 | 15.07 / 21.16 | 0.081 / 0.431 | 0.063 / 0.265 | 0.0007 / 0.0024 | 0.0007 / 0.0022 |
    | MLX fp16, deployed | **168.53 / 188.62** | 8.44 / 6.89 | 0.082 / 0.416 | 0.061 / 0.263 | 0.0007 / 0.0023 | 0.0007 / 0.0022 |
    | MLX fp16, `--steps 2`, lossy | 122.40 / 131.40 | 8.49 / 6.89 | 0.447 / 2.466 | 0.316 / 1.373 | 0.0050 / 0.0273 | 0.0056 / 0.0122 |
    | MLX fp16, pooled vision and two steps, lossy | 70.96 / 71.81 | 8.42 / 6.69 | 4.173 / 26.491 | 1.807 / 7.609 | 0.0304 / 0.9806 | 0.0347 / 0.9848 |

    Every error of the deployed MLX setting is at most 34% of the matching seed-spread statistic, and the median is
    39% lower than MPS. RSS is the process peak including loading and compiling.

    The time goes mostly to the language prefix. On MPS, the two-camera SigLIP batch took 36.45 ms, the Gemma prefix
    and KV cache 126.41 ms, and each of the ten action-expert steps 13.5–13.9 ms. On MLX the split was 24.32 ms,
    98.58 ms and 5.2–5.7 ms. The prefix holds about 2.2 TFLOP of dense projections, 2.03 TFLOP of it from the image
    tokens, and MLX already runs these matrix sizes at 21–28 TFLOP/s, close to the 30 TFLOP/s the chip reaches on a
    large square matrix. Fewer steps cannot remove that cost, so lossless latency below 100 ms was not reached.
    Padding, a different weight layout, compiling the whole denoising loop and a custom AdaRMS kernel gave no
    repeatable gain.

    Shorter schedules and image-token pruning are all lossy:

    | Full replay, matched noise | Median / p90 ms | Arm ° | Worst error / seed spread |
    | :-- | --: | --: | --: |
    | Default, ten steps | 160.74 / 162.85 | 0.082 / 0.416 | 34.0% |
    | Uniform Euler, three steps | 159.42 / 181.83 | 0.320 / 1.587 | 134.4% |
    | Quadratic Euler, three steps | 132.94 / 135.83 | 0.145 / 1.350 | 50.7% |
    | Half the image patches, three uniform steps | 95.17 / 99.24 | 0.818 / 5.788 | 933.6% |
    | A quarter of the patches, three uniform steps | 82.62 / 88.07 | 1.530 / 11.938 | 13847.9% |

    Patch pruning exists only in the offline tools (`--keep-visual-fraction`). Explicit MLX RTC, checked on three
    observations with a delay of 4 and a horizon of 16, differed from guided PyTorch by 0.071 / 0.350° at
    163.17 / 175.12 ms.

??? note "π0.5 public checkpoint on synthetic observations (did not qualify)"

    `zz4321/so101_pi05@dbd6d5754bbf78318808aea1706ff23beeb8b663`, three 640 × 480 synthetic views, PyTorch 2.11.0,
    bf16 eager as the reference with seed 941 and seed 942 for the spread.

    | Mode | Backend | Median / p90 ms | Joint ° | Gripper |
    |---|---|---:|---:|---:|
    | Chunk | Torch bf16, eager | 460.22 / 466.64 | reference | reference |
    | Chunk | Torch bf16, other seed | 460.32 / 467.00 | 20.926 / 102.412 | 3.268 / 10.543 |
    | Chunk | MLX fp16, compiled | 207.25 / 210.26 | 0.878 / 4.550 | 0.150 / 0.520 |
    | RTC | Torch bf16, eager | 475.17 / 487.15 | reference | reference |
    | RTC | Torch bf16, other seed | 483.18 / 492.75 | 15.690 / 100.621 | 3.275 / 13.470 |
    | RTC | MLX fp16, compiled | 225.43 / 231.51 | 0.604 / 4.315 | 0.109 / 0.511 |

    MLX was faster and below the seed spread, but above the absolute caps of 0.10 / 0.50° and 0.10 / 0.50 gripper
    units set for this test. The recorded float32 comparison replaced it.

??? note "π0 on recorded observations (qualifies chunk and RTC)"

    `zz4321/so101_pi0_cube@ec54d7a73d8959a7640edd7dad2690f460e64af2`, 25 frames from episodes 0, 7, 14, 21 and 29 of
    `zz4321/so101_stick@06f6856ea2a7dd43bfb72696a99e1087649a268b`, two recorded views, PyTorch 2.11.0, the same
    protocol as π0.5. The float32 reference ran at 692.58 / 718.63 ms for chunks and 704.17 / 718.71 ms for RTC. The
    RTC reference calls LeRobot's guided sampler directly.

    | Mode | Backend | Median / p90 ms | Joint ° | Gripper |
    |---|---|---:|---:|---:|
    | Chunk | Torch bf16, eager | 368.02 / 374.15 | 0.151 / 0.800 | 0.050 / 0.840 |
    | Chunk | MLX fp16, compiled | 160.28 / 163.59 | 0.027 / 0.135 | 0.012 / 0.159 |
    | Chunk | Torch fp32, seed 942 | 695.33 / 738.39 | 6.162 / 40.905 | 2.401 / 22.928 |
    | RTC | Torch bf16, eager | 383.45 / 388.20 | 0.093 / 0.832 | 0.039 / 0.704 |
    | RTC | MLX fp16, compiled | 169.65 / 175.58 | 0.015 / 0.101 | 0.007 / 0.114 |
    | RTC | Torch fp32, seed 942 | 727.07 / 749.78 | 4.461 / 40.704 | 1.809 / 23.799 |

    Both modes meet the absolute caps and beat the bf16 path.

??? note "SmolVLA (qualifies chunk mode)"

    `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, PyTorch 2.11.0, one 50-action chunk with 10
    denoising steps, three 256 × 256 synthetic views. The float32 CPU reference loads every weight from the
    checkpoint. The caps are 0.01° for the joints and 0.01 gripper units, on top of the seed spread.

    | Backend | Median / p90 ms | Joint ° | Gripper |
    |---|---:|---:|---:|
    | Torch CPU fp32 | 1195.22 / 1411.72 | reference | reference |
    | Torch CPU fp32, other seed | 1067.91 / 1095.52 | 0.31330 / 1.47535 | 0.41570 / 2.23537 |
    | Torch MPS, bf16 backbone | 295.70 / 307.56 | 0.00387 / 0.01744 | 0.00478 / 0.03623 |
    | MLX fp16, compiled | 52.76 / 53.24 | 0.00062 / 0.00258 | 0.00059 / 0.00315 |

??? note "X-VLA (qualifies chunk mode)"

    `whosricky/xvla-so101-pick-red-cube-2cam@7a007f64dd5910dd5cde76c8d31d9d7f7e6c0778`, 25 frames from episodes 0,
    54, 108, 147 and 181 of `whosricky/pick-red-cube-v3@f77f31614b10b9cf4ceb4c5f4c4edb503b12b74c`, PyTorch 2.11.0,
    CPU references on four threads. Both views are resized to 256 × 256; 30-action chunks with 10 denoising steps.
    Joint spans are 2116, 2343, 2203, 2294 and 4095 ticks; the gripper is in 0–100 units. LeRobot's X-VLA sampler
    has no RTC guidance.

    | Backend | Median / p90 ms | Joint ° | Gripper |
    |---|---:|---:|---:|
    | Torch CPU fp32, seed 941 | 4875.08 / 5619.00 | reference | reference |
    | Torch CPU fp32, seed 942 | 4786.63 / 5063.67 | 0.13363 / 0.81821 | 0.34578 / 7.50583 |
    | Torch MPS bf16 | 178.19 / 205.14 | 0.13311 / 0.65928 | 0.06051 / 0.32293 |
    | MLX fp16, compiled | 137.86 / 140.33 | 0.01014 / 0.04526 | 0.00556 / 0.03549 |

    MLX meets the absolute caps of 0.10 / 0.50° and 0.10 / 0.50 gripper units, stays below the seed spread, and is
    both faster and closer than MPS bf16.

??? note "ACT (public layout qualifies chunk mode)"

    MLX runs the full 100-action prediction in float32, with batch normalization folded into the ResNet-18
    convolutions and a zero latent as in LeRobot inference. Exact float32 needs `MLX_ENABLE_TF32=0`, which macbot
    sets when it loads ACT on MLX. Start a fresh process to compare precision settings.

    The public checkpoint is
    `lerobot/act_aloha_sim_transfer_cube_human@ba73b2766f1371cdc133ca4efb97eb090d744625`, with its normalization
    moved into LeRobot 0.7 processors by `lerobot.processor.migrate_policy_normalization`. We replayed episodes 0,
    12, 24, 36 and 49 of `lerobot/aloha_sim_transfer_cube_human@6a43d500f101255823a9d2b9dc244eeb01a2cd31`: one
    640 × 480 view and 14 actions, of which 12 are arm joints in radians. Our own checkpoint uses two views and a
    0–1 gripper score. PyTorch 2.11.0, CPU references on four threads.

    | Checkpoint and backend | Median / p90 ms | Joint ° | Gripper |
    |---|---:|---:|---:|
    | Public, Torch CPU fp32 | 56.25 / 63.35 | reference | reference |
    | Public, Torch MPS fp32 | 16.42 / 17.20 | 3.39e-6 / 2.05e-5 | 1.00e-7 / 5.07e-7 |
    | Public, MLX fp32, TF32 off | 8.32 / 8.57 | 4.18e-6 / 2.73e-5 | 1.24e-7 / 9.54e-7 |
    | Ours, Torch CPU fp32 | 71.98 / 76.33 | reference | reference |
    | Ours, Torch MPS fp32 | 22.80 / 23.47 | 6.93e-6 / 4.58e-5 | 1.76e-7 / 1.43e-6 |
    | Ours, MLX fp32, TF32 off | 12.28 / 12.58 | 7.29e-6 / 4.58e-5 | 1.71e-7 / 1.28e-6 |
    | Ours, MLX fp32, TF32 on, lossy | 9.95 / 10.11 | 0.02227 / 0.08749 | 0.000399 / 0.004394 |
    | Ours, MLX fp16, lossy | 8.38 / 8.55 | 0.00706 / 0.03525 | 0.000153 / 0.001858 |

    `auto` qualifies the public ALOHA layout. Other layouts run on MLX with `--backend mlx`.

??? note "VQ-BeT, Diffusion and TD-MPC (explicit MLX only)"

    PyTorch 2.15.0 and Python 3.13, three warm-ups on MPS. 25 real frames from five episodes of
    `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19`. PushT actions are pixel positions from 0 to 512.

    | Checkpoint and backend | Median / p90 ms | Action error, mean / p99 / max (px) |
    |---|---:|---:|
    | VQ-BeT, Torch MPS fp32, argmax | 7.69 / 8.52 | reference |
    | VQ-BeT, MLX fp32, compiled, argmax | 4.24 / 4.66 | 0.261 / 0.766 / 0.816 |
    | Diffusion, Torch MPS fp32, DDPM 100 steps | 818.79 / 978.45 | reference |
    | Diffusion, MLX fp32, compiled | 1960.36 / 2039.27 | 0.034 / 0.170 / 0.211 |

    The checkpoints are `lerobot/vqbet_pusht@15c2d0af889c401c7e5db7a07499d3b498afc276` and
    `lerobot/diffusion_pusht@84a7c23178445c6bbf7e1a884ff497017910f653`. VQ-BeT picks the most likely code in both
    backends, and Diffusion shares the starting and per-step noise.

    The ports themselves match PyTorch. Run on the CPU, MLX VQ-BeT differs from PyTorch by 0.0002 px on a test frame,
    and the ResNet-18 image features agree to 2e-6. The gap on the GPU comes from MLX float32 convolutions, which
    differ from PyTorch by a few parts in 10,000. PyTorch on the CPU and on MPS agree to 0.0001 px on all 25 frames. VQ-BeT's discrete code choice and
    Diffusion's 100 denoising steps amplify that gap, so neither qualifies for `auto`. Diffusion is also slower.

    TD-MPC has no public checkpoint in the LeRobot 0.7 format. Its MLX planner runs on a tiny random checkpoint and
    returns finite actions of the right shape. The unit tests in `tests/test_mlx_small.py` compare the VQ-BeT and
    Diffusion ports with PyTorch and run the TD-MPC port, all on tiny random checkpoints on the CPU.

??? note "π0-FAST and GR00T (not qualified)"

    | Policy | Checkpoint | Result |
    |---|---|---|
    | π0-FAST | `lerobot/pi0fast-libero@840f4b503f4c09110421c33c810a85b6684fd658` | MLX greedy decoding, bf16, up to 256 tokens: 885.22 / 1281.60 ms. The action tokens diverge from PyTorch's from the first action token on. |
    | GR00T | `nvidia/gr00t17-lerobot-libero_10-640@5ee08ab09fac5c5ef2388a14c882ea825ac861db` | Not run. It needs a 12.6 GB checkpoint plus the N1.7 base and Qwen3-VL assets. |

    PyTorch 2.15.0. The π0-FAST checkpoint outputs end-effector actions, so it ran in the offline harness only.

## Reproduce

The offline tools live in `tools/accel` in a source checkout. They need a LeRobot 0.7.0 environment and never open a
camera or the arm. Keep observation archives and traces outside the repository.

```bash
python -m tools.accel.recorded_observations --help   # extract frames from a dataset
python -m tools.accel.compare_backend --help         # time backends and compare actions on those frames
python -m tools.accel.eval_fidelity --help           # π0.5 fidelity runs against a reference
python -m tools.accel.bench_mlx --help               # π0.5 stage-by-stage MLX benchmark
```

`compare_backend --pi-fp32` sets up the float32 PyTorch reference for π0 and π0.5, and `--save-rtc-leftovers` with
`--rtc-leftovers` shares the RTC history between backends. `--act-mlx-tf32` and `--act-mlx-fp16` reproduce the lossy
ACT rows.

## References

The ports follow LeRobot's own implementations. We also consulted `tokimoa/pi0-mlx` and `mlx-smolvla` (Apache-2.0),
`mlx-lm` (MIT) and the PaliGemma and SigLIP code in `mlx-vlm` (MIT). No outside code or converted weights were copied.
