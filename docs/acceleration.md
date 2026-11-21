# π0.5 inference on Apple silicon

[简体中文](acceleration_zh.md)

The standard macbot backend is PyTorch. The optional MLX backend reads stock LeRobot 0.7.0 `pi05` safetensors in place and uses the checkpoint's original LeRobot tokenizer, state normalizer and action postprocessor. It supports single observations with one or more configured RGB views. Memory variants, checkpoint-configured RTC and adapters are outside this backend's scope.

## Automatic backend selection

An optimization is **lossless** for a tested configuration when, using the same weights and
starting noise or seed, its actions stay within documented tolerances against an FP32 CPU
reference or the stated existing BF16/FP16 reference. Mean and p99 joint error in degrees
and gripper error must each be no larger than the same statistic between two reference
runs with different seeds. If a policy is deterministic and the measured seed spread is
zero, use an absolute tolerance instead: mean and p99 joint error ≤ 0.01° and mean and
p99 gripper error ≤ 0.1% of the documented gripper range. Weight
quantization, fewer denoising steps and visual token dropping are lossy options.

`--backend auto` is opt-in. It checks the policy architecture, saved processor layout,
execution mode, compile setting and availability of a supported MLX runtime. It prints
the public checkpoint used to qualify the implementation or a one-line reason for falling
back to PyTorch. Compatible SmolVLA checkpoints use MLX for synchronous chunks.
The measurements below describe the tested Apple silicon machine and dependency versions;
they do not restrict selection to that machine. Use `--backend mlx` to request it explicitly.
The default backend and the default execution modes for explicit backends are unchanged.
`bench --mode rtc` measures guided calls with a fixed leftover chunk and no robot.

SmolVLA qualification: `lerobot/smolvla_base` revision
`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, Mac17,9 (Apple M5 Pro),
macOS 27.0.1, PyTorch 2.11.0, MLX 0.32.4, LeRobot 0.7.0. One 50-action chunk
with 10 denoising steps, three 256 × 256 RGB views, saved processors, five warm-ups
and 50 timed calls. The synthetic observations exercise the complete inference path;
no robot is involved. The FP32 CPU reference strictly loads all weights from the checkpoint.
The absolute caps for this comparison are 0.01° joint error and 0.01 gripper units for
each statistic, in addition to the measured seed spread.

| Backend | Median / p90 ms | Joint mean / p99 ° | Gripper mean / p99 |
|---|---:|---:|---:|
| Torch CPU FP32 | 1195.22 / 1411.72 | Reference | Reference |
| Torch CPU FP32, other seed | 1067.91 / 1095.52 | 0.31330 / 1.47535 | 0.41570 / 2.23537 |
| Torch MPS, BF16 backbone | 295.70 / 307.56 | 0.00387 / 0.01744 | 0.00478 / 0.03623 |
| MLX FP16, compiled | 52.76 / 53.24 | 0.00062 / 0.00258 | 0.00059 / 0.00315 |

## PyTorch MPS experiments

For ACT or Diffusion on MPS, `--torch-channels-last` changes the layout of Conv2d
weights, and `--torch-autocast fp16` or `bf16` changes inference precision and can
be lossy. `--torch-single-readback` drops ACT's redundant explicit device barrier.
`--torch-compile` experiments with compiling Diffusion's denoiser. ACT compilation
is disabled after the tested MPS build failed to generate its Metal kernels.
These flags are opt-in and excluded from `auto`. The Diffusion action queue and scheduler
are unchanged. The offline runner `python -m tools.accel.compare_backend --help` can
compare matched-input action chunks without opening robot or camera devices.

## Try it offline

Install the optional dependencies on Apple silicon, then benchmark with your own checkpoint and configured camera sizes:

```sh
python -m pip install '.[mlx]'
macbot --config "$CONFIG" bench --policy "$CHECKPOINT" --backend mlx -n 30
macbot --config "$CONFIG" bench --policy "$CHECKPOINT" --backend torch --profile -n 30
```

`bench` uses generated blank RGB arrays; it never opens a camera or a robot. Synchronized stage timings are a separate pass and will not add up to the end-to-end latency. `--no-compile` disables the backend's graph compiler. To select MLX for deployment, pass `--backend mlx` to `macbot run` or add `policy: {backend: mlx}` to your YAML. With MLX, `--mode auto` uses unguided asynchronous execution. Explicit `--mode rtc` uses the selected backend for guidance.

Weight quantization is an experimental option: `--mlx-bits 8` or `--mlx-bits 4`, with `--mlx-quantize prefix`, `expert` or `both`. Reducing `--steps` changes the flow integration. Evaluate each of these options against your checkpoint before using it.

## Reproduce latency and fidelity

Run the commands below from the macbot source directory with a LeRobot 0.7.0 environment. Set `DATA` to your raw `episode_*.npz` directory and `EVAL` to a writable directory outside the repo. Store observation archives, action arrays and intermediate traces privately.

```sh
mkdir -p "$EVAL"
python -m tools.accel.eval_fidelity prepare --data "$DATA" --out "$EVAL/observations.npz"
python -m tools.accel.eval_fidelity run --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/bf16.json" --profile
python -m tools.accel.eval_fidelity run --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/seed2.json" \
  --reference "$EVAL/bf16.npy" --seed 942
python -m tools.accel.eval_fidelity run --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/fp32.json" \
  --reference "$EVAL/bf16.npy" --dtype fp32
python -m tools.accel.bench_mlx --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/torch_stages.npz" --capture-torch
python -m tools.accel.bench_mlx --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/mlx.json" \
  --trace "$EVAL/torch_stages.npz" --reference "$EVAL/bf16.npy" \
  --trim --compile --precompute-adarms --fused-rope --pack-linears --precision fp16 --profile
```

Use only one benchmark process at a time. The default fidelity sample is 50 observations across 10 quality-checked episodes. Timing uses 30 calls after five warm-up calls and regular random noise. Fidelity uses the same saved-equivalent FP32 starting noise for each observation in every run (seed 941, except the seed-spread control). Arm errors are computed after unnormalization in degrees. A binary gripper uses the unnormalized close score, reported separately. Both populations report mean and p99 over the full chunk and the first ten action positions. Agreement on this replay does not establish real robot task success.

The stage check compares projected SigLIP features, each layer's valid prefix K/V, velocity at time 1, and the final actions before benchmarking MLX. The baseline profile and results below come from real recorded observations; CLI blank-input timing is a separate check.

## M5 Pro measurements

Apple M5 Pro, 20 GPU cores and 48 GiB unified memory, with macOS 27.0.1, PyTorch 2.11.0 and MLX 0.32.4. Each benchmark used one GPU process at a time. Desktop background activity remained, so small differences between runs should not be attributed to individual settings.

The synchronized MPS baseline spent 36.45 ms in the two-view SigLIP batch (isolated views took 19.25 and 19.23 ms), 126.41 ms in Gemma prefix/KV, and 13.54–13.87 ms per expert step. RGB conversion, host-to-device transfer and image resize took 0.67, 1.27 and 0.56 ms respectively. Action unnormalization and raw readback took 0.54 and 0.22 ms. These are medians from separate runs or instrumented stages.

The prefix has roughly 2.2 TFLOP of dense projections even after trimming the text padding; the visual tokens alone account for about 2.03 TFLOP. SigLIP has about 0.42 TFLOP of projections per two-view call. The expert performs about 33 GFLOP of projections for each step and accesses roughly 0.63 GB of dense projection weights. These counts exclude attention, normalization and small projections. The prefix and vision are high arithmetic intensity; the expert has less work per byte and is more sensitive to data movement and dispatch.

FP32 MPS calibrates the size of numerical differences; a second noise draw calibrates the policy's inherent variation. For this checkpoint, we use “lossless in practice” when the configuration keeps all visual tokens and all ten steps, uses no weight quantization, and each of the eight action statistics is below half of its corresponding seed-spread statistic. This is an empirical test, not equivalence of arbitrary rollouts.

| Configuration | Median / p90 (ms) | Peak RSS / GPU (GiB) | Arm all (°) | Arm first 10 (°) | Gripper all | Gripper first 10 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| MPS BF16, initial profile | 297.71 / 311.91 | 12.38 / 9.16 | Reference | Reference | Reference | Reference |
| MPS BF16, same-noise repeat | 276.83 / 291.30 | 15.01 / 9.17 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| MPS BF16, different starting noise | 279.03 / 284.38 | 15.19 / 9.17 | 0.626 / 8.037 | 0.274 / 1.179 | 0.0060 / 0.0071 | 0.0057 / 0.0064 |
| MPS FP32, eager control | 923.30 / 980.37 | 15.07 / 21.16 | 0.081 / 0.431 | 0.063 / 0.265 | 0.0007 / 0.0024 | 0.0007 / 0.0022 |
| MLX FP16, deployed configuration | **168.53 / 188.62** | 8.44 / 6.89 | 0.082 / 0.416 | 0.061 / 0.263 | 0.0007 / 0.0023 | 0.0007 / 0.0022 |
| MLX FP16, all visual tokens and `--steps 2`, lossy | 122.40 / 131.40 | 8.49 / 6.89 | 0.447 / 2.466 | 0.316 / 1.373 | 0.0050 / 0.0273 | 0.0056 / 0.0122 |
| MLX FP16, pooled vision and two steps, lossy | **70.96 / 71.81** | 8.42 / 6.69 | 4.173 / 26.491 | 1.807 / 7.609 | 0.0304 / 0.9806 | 0.0347 / 0.9848 |

Each error cell is mean / p99. RSS is the process lifetime peak, including load and compile. The GPU figure is the maximum sampled driver allocation for MPS and peak timed allocation for MLX. They overlap with RSS on unified memory.

**Below 100 ms with fidelity was not reached.** Each of the eight error statistics for the deployed MLX setting was at most 34% of the corresponding seed-spread statistic, and median latency was 39% lower than the repeated MPS baseline. Its synchronized split was 24.32 ms for vision, 98.58 ms for prefix/KV, and 5.20–5.65 ms per expert step. Fewer steps alone cannot fix the prefix cost. Removing padding, MLX compilation, and precomputing AdaRMS helped. Affine 8-bit quantization saved GPU allocations but was slower; 4-bit prefix quantization caused much larger KV and action errors. The sub-100 ms result above uses a 2×2 mean over projected visual tokens and a two-step Euler schedule. It is available only in the offline harness (`--pool-images --steps 2`), not as a deployment setting. Start from the FP16 command above and omit `--pack-linears` to reproduce that result.

An additional speed gain at the same fidelity would require substantially faster prefix projection and attention kernels, or a model with a smaller prefix. A reduced-step model designed or distilled for that schedule might also lower expert cost. These alternatives were not trained or tested here.

## Round 2: where the time goes

The standard ten-step MLX output remained bit-for-bit identical to v0.3.0 on all 50 observations. It took **164.00/167.32 ms** (median/p90); the final integration check took **160.74/162.85 ms**. AC power, an active desktop, one benchmark at a time, at least five warm-ups and 30 timed calls. The difference between these repeats is not an optimization.

Metal capture of synthetic matrices at the real dimensions confirmed that MLX 0.32.4 already dispatches gate/up to `steel_gemm_fused_nax` and down to `steel_gemm_splitk_nax`. The replay had 553–558 tokens. At 557 rows, FP16 gate/up (2048→32768) delivered **27.7 TFLOP/s**, and down (16384→2048) delivered **21.1 TFLOP/s**. The same device delivered **30.0 TFLOP/s** on a 4096-square FP16 control. Padding to 576 rows changed gate/up from 2.696 to 2.690 ms and down from 1.771 to 1.872 ms. More FLOP/s on padded work does not necessarily save time. Reproduce the dtype/layout/alignment sweep with `python -m tools.accel.bench_mlx_shapes --out "$EVAL/shapes.json"`.

The initial synchronized split was 1.93 ms for preparation, 23.60 ms for both cameras, 92.54 ms for prefix/KV and 5.21–5.52 ms per expert evaluation. In a separate, slower run, device-ready inputs through both cameras and complete KV creation took **146.51/174.19 ms** without any denoising or intermediate barrier. These timings describe the tested implementation, not a theoretical hardware limit. Padding, changing the weight layout, compiling the entire denoising loop, a custom AdaRMS kernel and pre-slicing modulation did not establish a repeatable end-to-end improvement sufficient to change the default.

| Full replay, matched noise | Median/p90 ms | Arm mean/p99 ° | Worst of eight errors / seed spread |
| :-- | --: | --: | --: |
| Integrated default, ten steps | 160.74/162.85 | 0.082/0.416 | 34.0% |
| Uniform Euler, three evaluations | 159.42/181.83 | 0.320/1.587 | 134.4% |
| Quadratic Euler, three evaluations | 132.94/135.83 | 0.145/1.350 | 50.7% |
| Keep half the patches, three uniform evaluations | 95.17/99.24 | 0.818/5.788 | 933.6% |
| Keep one quarter, three uniform evaluations | 82.62/88.07 | 1.530/11.938 | 13847.9% |

All shortened schedules and all patch pruning are **lossy**. For the quadratic grid, set `--backend mlx --steps 3 --mlx-time-grid quadratic` in `bench` or `run`. It spends more evaluations near the clean end of the flow. Compare fidelity at equal evaluation counts; absolute timings in different desktop-load windows do not isolate solver overhead. To reproduce in the offline command above, add `--steps 3 --solver quadratic`; `ab2` and `midpoint` are also available for comparison (midpoint requires an even budget). Patch selection is offline only: `--steps 3 --keep-visual-fraction 0.5` keeps the highest attention patches per camera after block two, retains the text, and preserves absolute token positions. With half the patches the full-chunk gripper p99 error was 0.06643, against a seed spread of 0.00712.

Explicit MLX RTC was tested separately on three observations with matched noise, model-space leftovers, delay 4, horizon 16 and maximum guidance weight 10. Against guided PyTorch, arm mean/p99 was **0.071/0.350°**; timed guided MLX calls took **163.17/175.12 ms**. LeRobot 0.7.0 evaluates the velocity before enabling gradients on its input, so the reference correction is the weighted residual. Differentiating through the entire denoiser instead gave 0.183/1.670° and is not the deployed rule. No hardware rollout was performed.

## Implementation references

The port follows LeRobot's `pi05` implementation and uses MLX's fused attention, normalization and quantized matmul APIs. We consulted the `tokimoa/pi0-mlx` model card and implementation (Apache-2.0), `mlx-smolvla` (Apache-2.0), `mlx-lm` (MIT), and `mlx-vlm`'s PaliGemma/SigLIP (MIT). No external source code or converted weights were imported.
