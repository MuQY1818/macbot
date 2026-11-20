# π0.5 inference on Apple silicon

[简体中文](acceleration_zh.md)

The standard macbot backend is PyTorch. The optional MLX backend reads stock LeRobot 0.7.0 `pi05` safetensors in place and uses the checkpoint's original LeRobot tokenizer, state normalizer and action postprocessor. It supports single observations with one or more configured RGB views. Memory variants, checkpoint-configured RTC and adapters are outside this backend's scope.

## Try it offline

Install the optional dependencies on Apple silicon, then benchmark with your own checkpoint and configured camera sizes:

```sh
python -m pip install '.[mlx]'
macbot --config "$CONFIG" bench --policy "$CHECKPOINT" --backend mlx -n 30
macbot --config "$CONFIG" bench --policy "$CHECKPOINT" --backend torch --profile -n 30
```

`bench` uses generated blank RGB arrays; it never opens a camera or a robot. Synchronized stage timings are a separate pass and will not add up to the end-to-end latency. `--no-compile` disables the backend's graph compiler. To select MLX for deployment, pass `--backend mlx` to `macbot run` or add `policy: {backend: mlx}` to your YAML. With MLX, `--mode auto` uses unguided asynchronous execution. Explicit `--mode rtc` selects PyTorch for guided RTC.

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

## Implementation references

The port follows LeRobot's `pi05` implementation and uses MLX's fused attention, normalization and quantized matmul APIs. We consulted the `tokimoa/pi0-mlx` model card and implementation (Apache-2.0), `mlx-smolvla` (Apache-2.0), `mlx-lm` (MIT), and `mlx-vlm`'s PaliGemma/SigLIP (MIT). No external source code or converted weights were imported.
