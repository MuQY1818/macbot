# Simulation

[简体中文](simulation.zh.md)

Install on an Apple silicon Mac with Python 3.12 or later. Start from the
[SO-101 example config](../configs/so101_sim.yaml). First use downloads and
runs the pinned EnvHub Python code and MuJoCo assets; review them before enabling
`trust_remote_code`. The scene puts an SO-101, four cubes, and a box on the table.
On macOS the simulator prints the offscreen renderer it selected. Hardware CGL
is the default; if it cannot create that context, it warns and tries software rendering.
Software rendering can make teleoperation much slower than the 30 Hz scene clock.
Once the pinned scene is cached, it loads without contacting the Hub. If the
scene still needs a download through a SOCKS proxy, run `pip install 'httpx[socks]'`
in the same environment. `macbot doctor` checks for this.

```bash
pip install 'macbot[sim]'               # or: pip install -e '.[sim]'
cp configs/so101_sim.yaml ./tabletop.yaml
macbot --config tabletop.yaml doctor
```

The YAML sets up the simulator; the USB setup wizard is for physical arms.
The example keeps the existing sine input for pipeline smoke tests. Choose
`teleop.type: keyboard` in your copy and retain `teleop.hz: 30`. Run
`macbot --config tabletop.yaml collect`, click the video, and press **Enter**
to start. The legend below the video highlights the held keys.

| Keys | Motion |
| --- | --- |
| W / S | Forward / back along the table in the first view |
| A / D | Left / right in the first view |
| E / Q | Raise / lower the gripper |
| Z / X | Roll clockwise / counterclockwise when viewed from above |
| Space | Toggle the gripper; it stays closed while you move |
| Shift | Fine translation and roll while held |
| Enter | Start a new take or save the current take |
| Backspace | Discard the take |
| Esc | Stop the take; then save or discard it |

The gripper points down by default. Translation runs at **6 cm/s** and roll at
**45°/s** in simulation time, with a 0.10 s ramp. Shift slows both to one
quarter speed. Releasing keys or leaving the video focus stops advancing the
movement target. The preview shows the tip height and its projection onto the
table, plus a limit indicator when motion is blocked. Keep a fixed camera first
in `cameras` when using keyboard control.

Lift before crossing the table. Align the jaws with a cube, lower around it,
close, lift, then open above the box. Lower the gripper or move back toward the
arm if the top-down workspace limits your reach. The console saves automatically
on success or timeout. A synthetic key sequence completed the four-cube task
through the collection HTTP endpoint. In a short hand-driven trial, a person placed one cube in the box.

By default, simulation data goes to `~/macbot-data/sim`, with episodes under
`raw/<task>` inside that root. The console prints the actual episode directory
on startup. An explicit `data_root` or `collect --out` still takes precedence.

An optional calibrated physical SO-101 leader can drive the simulated follower.
Use `teleop: {type: so101_leader, port: /dev/tty.usbmodemYOUR_LEADER, id: leader, hz: 30}`.
Use the same port and ID with `lerobot-calibrate --teleop.type=so101_leader
--teleop.port=... --teleop.id=leader` if you need to calibrate. macbot reads
absolute joint angles and uses the saved LeRobot calibration. This path is mock tested.
The existing episode shortcuts still apply to scripted and SO-101 leader input.

You can generate seeded demonstrations and export them as LeRobot v3.0:

```bash
macbot --config tabletop.yaml sim demos --episodes 50 --seed 0 --out ./demos
macbot --config tabletop.yaml health --data ./demos
macbot --config tabletop.yaml export --data ./demos --out ./dataset
```

`--episodes` counts attempts. The script reads scene poses and uses IK; difficult
spawns can fail. It reports every attempt in `demos_*.json` and saves successful
episodes by default. `--keep-failures` also saves the unsuccessful attempts.

Transfer the complete exported `dataset` folder to a CUDA machine and train
an ACT checkpoint there with LeRobot. Use `export --images` if you want to
preserve the decoded recording pixels without another lossy encode.

```bash
pip install 'lerobot[training]==0.7.0'
lerobot-train --dataset.repo_id=local/so101-tabletop --dataset.root=./dataset --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --steps=5000 --save_freq=1000 --batch_size=8 --num_workers=0 \
  --output_dir=./act-tabletop
```

Bring `act-tabletop/checkpoints/last/pretrained_model` back to the Mac as
`./pretrained_model`. Evaluate it on seeds outside the demonstration set.

```bash
macbot --config tabletop.yaml eval \
  --policy ./pretrained_model \
  --backend torch --mode chunk --k 30 --episodes 50 --seed 1000 --trace-only --out ./trials
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
The workflow has run end to end. It does not establish that a trained policy
solves the four-cube task.
