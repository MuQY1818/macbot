# SO-101 simulation

[简体中文](simulation_zh.md)

Install on an Apple silicon Mac with Python 3.12 or later. Start from the
[SO-101 example config](../configs/so101_sim.yaml). First use downloads and
runs the pinned EnvHub Python code and MuJoCo assets; review them before enabling
`trust_remote_code`. The scene puts an SO-101, four cubes, and a box on the table.
On macOS the simulator prints the offscreen renderer it selected. Hardware CGL
is the default; if it cannot create that context, it warns and tries software rendering.
Software rendering can make teleoperation much slower than the 30 Hz scene clock.

```bash
pip install 'macbot[sim]'               # or: pip install -e '.[sim]'
cp configs/so101_sim.yaml ./tabletop.yaml
macbot --config tabletop.yaml doctor
```

The YAML sets up the simulator; the USB setup wizard is for physical arms.
The example keeps the existing sine input for pipeline smoke tests. Choose
`teleop.type: keyboard` in your copy and retain `teleop.hz: 30`. Run
`macbot --config tabletop.yaml collect` and start with **S**. Focus the
collection page and open **Keyboard controls** for the key map:

| Keys | Motion |
| --- | --- |
| I / K | Farther / closer |
| J / L | Left / right |
| U / O | Up / down |
| N / M | Roll the wrist |
| F / G | Open / close |
| Shift | Fine translation while held |

Lift before crossing the table. Align the jaws with a cube, lower around it,
close, lift, then open above the box. Enter saves, Left Arrow discards, and the
scene saves automatically on success or timeout. A synthetic keyboard episode
finished the full task through the collection console. Human trials are pending.

An optional calibrated physical SO-101 leader can drive the simulated follower.
Use `teleop: {type: so101_leader, port: /dev/tty.usbmodemYOUR_LEADER, id: leader, hz: 30}`.
Use the same port and ID with `lerobot-calibrate --teleop.type=so101_leader
--teleop.port=... --teleop.id=leader` if you need to calibrate. macbot reads
absolute joint angles and uses the saved LeRobot calibration. This path is mock tested.

You can generate seeded demonstrations and export them as LeRobot v3.0:

```bash
macbot --config tabletop.yaml sim demos --episodes 50 --seed 0 --out ./demos
macbot --config tabletop.yaml health --data ./demos
macbot --config tabletop.yaml export --data ./demos --out ./dataset
```

`--episodes` counts attempts. The script reads scene poses and uses IK; difficult
spawns can fail. It reports every attempt in `demos_*.json` and saves successful
episodes by default. `--keep-failures` also saves the unsuccessful attempts.

Train a local ACT checkpoint with LeRobot. This is a starting budget; check
training progress and keep evaluation seeds separate from demonstration seeds.

```bash
pip install 'lerobot[training]==0.7.0'
lerobot-train --dataset.repo_id=local/so101-tabletop --dataset.root=./dataset --dataset.video_backend=pyav \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --steps=5000 --save_freq=1000 --batch_size=8 --num_workers=0 \
  --output_dir=./act-tabletop
macbot --config tabletop.yaml eval \
  --policy ./act-tabletop/checkpoints/last/pretrained_model \
  --backend torch --mode chunk --k 30 --episodes 50 --seed 1000 --out ./trials
macbot --config tabletop.yaml health --data ./trials
macbot --config tabletop.yaml replay --dir ./trials
```

`eval` takes a local checkpoint or `owner/model@revision`; add `--videos` for
side-by-side MP4s. For large SO-101 tabletop batches, `--trace-only` renders at
policy queries and saves the joint traces without camera playback; `health` and
the trace view in `replay` still work. Each option runs every simulated step. Policy
targets go directly to the simulator without the physical arm's velocity ramp.
For async/RTC, use `--delay-steps` to model inference delay; the default is zero.
Save the scene configuration, seed range, policy revision and execution options
when comparing runs.

`eval_*.json` contains per-seed outcomes, mean episode length and a 95% Wilson
interval for the success rate. The scene succeeds when all four cube centers
pass its box test; it times out at 3,600 steps by default. Simulation performance
does not establish real robot performance. Keep simulation and hardware data separate.
