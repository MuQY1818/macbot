# Quickstart

[简体中文](quickstart.zh.md)

This page takes you from an assembled arm pair to your first saved demonstration.

!!! note "Before you start"

    - An Apple silicon Mac.
    - An assembled leader and follower arm: SO-100/101, Koch, OMX or reBot B601-RS. Finish the kit's assembly and
      motor ID setup first. macbot recognizes the USB adapter but does not set motor IDs. Tested arms are listed
      under [Robots](robots.md).
    - At least one camera.
    - Using a reBot or an Orbbec camera? Do the extra steps under [Special hardware](#special-hardware) first.

## 1. Install

Install Python 3.12, for example with the installer from python.org. If you use a UVC webcam, also run
`xcode-select --install` once. Then create an environment and install macbot:

```bash
python3.12 -m venv ~/.venvs/macbot
source ~/.venvs/macbot/bin/activate
pip install macbot
```

## 2. Run the setup wizard

Connect and power both arms, plug in the cameras and start the wizard:

```bash
macbot init
```

The wizard opens in your browser. Confirm the arm model, then tell the two arms apart: click
**Identify by movement**, then gently move the leader by hand and leave the follower still. The leader's
torque is disabled, so it moves freely.

??? tip "The wizard cannot tell the two arms apart"

    Unplug one USB cable and rescan. The adapter that remains is the other arm. Plug the cable back in and assign
    the roles by hand.

    Adapters with a unique USB serial number keep their role in any socket. Adapters without one are tied to the
    socket you chose, so keep them there or run the wizard again after rewiring.

## 3. Assign the cameras

Pick a role for each camera from the live previews, and allow camera access when macOS asks. The wizard skips
iPhone and Continuity Camera and chooses a capture mode that each camera supports.

![Camera role selection with two simulated previews](img/init_en.png)

If a UVC camera gets a new ID after you move it to another socket, run the wizard again.

## 4. Calibrate, if the wizard asks

macbot reads LeRobot calibration files but does not calibrate the arms itself. When a file is missing, the wizard
shows a LeRobot command for each arm.

1. Open a second Terminal, activate the same environment and run the command. If it reports missing packages,
   run `pip install 'lerobot[core-scripts]==0.7.0'` first.
2. Follow LeRobot's prompts and move the joints by hand. The files go to LeRobot's usual calibration folder.
3. Back in the wizard, click **Check again**, then **Save and run doctor**.

## 5. Record your first episode

Support the follower, clear the workspace and bring both arms into the same pose. Click **Start collection** in
the wizard, or press ++ctrl+c++ and run:

```bash
macbot collect
```

The collection console opens in the browser. Wait until it shows that teleoperation is running, then move the
leader and check that the follower tracks it.

| Key | Action |
|---|---|
| ++s++ | Start recording |
| ++enter++ | Save the episode |
| ++arrow-left++ | Discard the episode |

Record for at least two seconds. The saved episode appears in the console and under
`~/macbot-data/raw/cubes_to_box/`. Quit from the console before you unplug the arms.

No hardware at hand? `macbot collect --fake` opens the same console with simulated arms and cameras.

## Next steps

- Check, export and train on your episodes: [Workflow guide](guide.md).
- Run the trained policy on the robot: [Run a policy](guide.md#5-run-a-policy).
- Make the policy faster on your Mac: [Acceleration](acceleration.md).

## Keep the camera framing

!!! warning "A policy needs the framing it was trained with"

    Each camera role must keep the same capture mode, crop and saved image size between recording and running.

Exports store the framing in `meta/camera_framing.json`, and the training scripts copy it into the checkpoint.
`macbot run` stops before it connects to the robot when the framing differs. Pass `--allow-framing-mismatch` only
when you know the difference does not matter. Older checkpoints have no framing record, so compare the framing
that `macbot run` prints at startup with the one you trained with.

When you move an existing setup to macbot, run `macbot init --from existing.yaml` and keep
**Keep framing from the existing config** selected.

## Special hardware

??? note "reBot B601-RS (PCAN-USB)"

    Install the PCBUSB driver as described in [Setup](setup.md#2-can-driver-pcan-usb). The wizard recognizes the
    standard PCAN-USB adapter, uses `can0` and keeps the reference calibration IDs.

??? note "Orbbec cameras"

    Install the Orbbec SDK before you run the wizard. Its current PyPI package pins a PyAV version that conflicts
    with LeRobot 0.7.0 on Python 3.12 and 3.13. For color streams, install macbot first and then run, in the same
    environment:

    ```bash
    pip install --no-deps pyorbbecsdk2==2.1.3
    ```

    If the SDK needs root, run the wizard and collection with sudo and point them at your own calibration folder:

    ```bash
    export HF_LEROBOT_CALIBRATION="$(python -c 'from lerobot.utils.constants import HF_LEROBOT_CALIBRATION; print(HF_LEROBOT_CALIBRATION)')"
    sudo -E "$(command -v macbot)" init
    sudo -E "$(command -v macbot)" collect
    ```

    Run the calibration commands from your normal account.

## Reference

??? note "Configuration file"

    The wizard saves the setup to `~/.config/macbot/config.yaml`. `macbot --config other.yaml collect` uses the
    given file instead.

??? note "Browser tabs and ports"

    Each console opens in your default browser, also under sudo. A console tab left open from an earlier session
    reloads instead of opening a new one, so repeated `macbot run` sessions share one tab. Pass `--no-browser` or
    set `MACBOT_NO_BROWSER=1` to print the URL only.

    The default ports are 8765 for init, 8766 for collect, 8767 for run and 8768 for replay. When one is busy,
    macbot tries up to nine higher ports. To fix a port, pass `--port` or set `init.port`, `teleop.port`,
    `run.port` or `replay.port` in the config. An explicit port must be free.

??? note "Setup without the wizard"

    For a calibrated setup that already has a config, `macbot init --yes` reuses the saved roles and runs doctor.
    For a new serial arm pair, scripts can confirm the setup with
    `--family so101 --leader-serial … --follower-serial …`, and a single external camera becomes `top`. When the
    model or a role is ambiguous, or calibration is missing, use the wizard.
