# Simulation

[简体中文](simulation.zh.md)

The simulator puts an SO-101, four cubes and a box on a table in MuJoCo. You can drive the arm with the keyboard,
generate scripted demonstrations, train a policy on a CUDA machine and evaluate it on your Mac, all with the same
commands as on a real arm.

!!! note "What has been shown so far"

    The whole workflow runs end to end. A trained policy that solves the four-cube task has not been shown yet, and
    results in simulation say nothing about a real robot. Keep simulated and hardware data apart.

## 1. Install

You need an Apple silicon Mac with Python 3.12 or later.

```bash
pip install 'macbot[sim]'      # from a source checkout: pip install -e '.[sim]'
cp configs/so101_sim.yaml ./tabletop.yaml
macbot --config tabletop.yaml doctor
```

The example config is [configs/so101_sim.yaml](../configs/so101_sim.yaml). The scene is set up by this file; the
`macbot init` wizard is only for physical arms.

!!! warning "The scene runs downloaded code"

    The first run downloads the pinned EnvHub scene, including Python code and MuJoCo assets, and runs it with
    `trust_remote_code`. Review that code before you enable it. Once cached, the scene loads without the Hub.

??? note "Proxies and rendering"

    If the scene still has to be downloaded through a SOCKS proxy, run `pip install 'httpx[socks]'` in the same
    environment. `macbot doctor` checks for this.

    On macOS the simulator prints which offscreen renderer it uses. Hardware CGL is the default. If it cannot create
    that context, it warns and falls back to software rendering, which can make teleoperation much slower than the
    30 Hz scene clock.

## 2. Drive the arm with the keyboard

The example config moves the arm with a sine wave, which is useful for smoke tests. For keyboard control, set
`teleop.type: keyboard` in your copy and keep `teleop.hz: 30`. Then run:

```bash
macbot --config tabletop.yaml collect
```

Click the video and press ++enter++ to start. The legend below the video highlights the keys you hold.

| Keys | Motion |
|---|---|
| ++w++ / ++s++ | Forward and back along the table, as seen in the first view |
| ++a++ / ++d++ | Left and right in the first view |
| ++e++ / ++q++ | Raise and lower the gripper |
| ++z++ / ++x++ | Roll clockwise and counterclockwise, seen from above |
| ++space++ | Open or close the gripper; it stays closed while you move |
| ++shift++ | Hold for fine translation and roll |
| ++enter++ | Start a new take, or save the current one |
| ++backspace++ | Discard the take |
| ++escape++ | Stop the take, then save or discard it |

The gripper points down. It moves at 6 cm/s and rolls at 45°/s in simulation time, with a 0.10 s ramp; ++shift++
slows both to a quarter. Motion stops when you release the keys or click outside the video. The preview shows the
tip height, its projection onto the table, and a marker when a limit blocks the motion. Keep a fixed camera first in
`cameras` when you use the keyboard.

!!! tip "Picking up a cube"

    Lift before you cross the table. Line the jaws up with a cube, lower around it, close, lift, and open above the
    box. If the top-down workspace limits your reach, lower the gripper or move back toward the arm.

The console saves on its own when the task succeeds or times out. A scripted key sequence completed the four-cube
task through the collection endpoint, and in a short trial by hand a person placed one cube in the box.

Simulated episodes go to `~/macbot-data/sim/raw/<task>`, and the console prints the exact folder at startup. An
explicit `data_root` or `collect --out` takes precedence.

??? note "Drive the simulated arm with a real SO-101 leader"

    A calibrated SO-101 leader can drive the simulated follower:

    ```yaml
    teleop: {type: so101_leader, port: /dev/tty.usbmodemYOUR_LEADER, id: leader, hz: 30}
    ```

    To calibrate it, run `lerobot-calibrate --teleop.type=so101_leader --teleop.port=... --teleop.id=leader` with the
    same port and ID. macbot reads absolute joint angles through the saved LeRobot calibration. This path has only
    been tested with a mocked leader. The episode keys work the same with the leader and the scripted input.

## 3. Generate scripted demonstrations

```bash
macbot --config tabletop.yaml sim demos --episodes 50 --seed 0 --out ./demos
macbot --config tabletop.yaml health --data ./demos
macbot --config tabletop.yaml export --data ./demos --out ./dataset
```

The script reads the cube poses from the scene and plans with inverse kinematics. `--episodes` counts attempts, and
hard spawns can fail. Every attempt is listed in `demos_*.json`; only successful ones are saved unless you pass
`--keep-failures`.

## 4. Train on a CUDA machine

Copy the whole exported `dataset` folder to a CUDA machine and train ACT with LeRobot. Export with `--images` if you
want the decoded pixels without another lossy video encode.

```bash
pip install 'lerobot[training]==0.7.0'
lerobot-train --dataset.repo_id=local/so101-tabletop --dataset.root=./dataset --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --steps=5000 --save_freq=1000 --batch_size=8 --num_workers=0 \
  --output_dir=./act-tabletop
```

Copy `act-tabletop/checkpoints/last/pretrained_model` back to the Mac as `./pretrained_model`.

## 5. Evaluate on the Mac

Evaluate on seeds that were not used for the demonstrations:

```bash
macbot --config tabletop.yaml eval \
  --policy ./pretrained_model \
  --backend torch --mode chunk --k 30 --episodes 50 --seed 1000 --trace-only --out ./trials
macbot --config tabletop.yaml health --data ./trials
macbot --config tabletop.yaml replay --dir ./trials
```

`eval_*.json` lists the outcome of every seed, the mean episode length and a 95% Wilson interval for the success
rate. An episode succeeds when all four cube centers pass the scene's box test, and times out after 3,600 steps by
default.

- `--policy` takes a local checkpoint or `owner/model@revision`.
- `--videos` writes side-by-side MP4s. `--trace-only` renders only at policy queries and saves joint traces without
  camera playback, which is much faster for large batches; `health` and the trace view in `replay` still work.
- Every simulated step is run either way. Policy targets go straight to the simulator, without the velocity ramp of
  the physical arm.
- For `async` and `rtc`, `--delay-steps` models the inference delay. The default is zero.

When you compare runs, record the scene config, seed range, policy revision and execution options.
