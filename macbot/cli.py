"""Command line entry point: ``macbot <command>``.

    macbot doctor                 check dependencies, the configured robot and cameras
    macbot init [--yes]           detect and configure arms and cameras
    macbot cameras                list video devices with their uniqueIDs
    macbot joints                 read leader and follower joint angles without powering anything
    macbot robots                 list the LeRobot arm types macbot can drive (robot.type: lerobot)
    macbot policies               list LeRobot policy types and their macOS support
    macbot collect [--fake]       teleoperate and record episodes (web page)
    macbot export                 convert recorded episodes to a LeRobot dataset
    macbot health                 per-episode tracking and timing report of recorded data
    macbot run --policy DIR       run a trained LeRobot policy on the robot (web page)
    macbot replay                 browse recorded rollouts in the browser
    macbot video ROLLOUT          export a rollout as a side-by-side MP4
    macbot bench --policy DIR     time policy inference on this Mac

On macOS the Orbbec SDK requires ``sudo -E``. Simulation, fake mode and UVC-only setups run
without sudo when the robot driver permits it.
"""
from __future__ import annotations

import argparse
import sys


def _cfg(args, *, resolve=None):
    from .config import load_cli

    try:
        return load_cli(args.config, resolve=resolve)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


def cmd_doctor(args):
    from .doctor import run
    return run(_cfg(args, resolve=True))


def cmd_init(args):
    import os

    from .init_wizard import Setup
    from .system import expand

    if not args.yes and any((args.family, args.leader_serial, args.follower_serial)):
        raise SystemExit("--family and --*-serial are for macbot init --yes")
    setup = Setup(expand(args.config).absolute() if args.config else None)
    try:
        if args.yes:
            setup.refresh()
            status, output = setup.automatic(family=args.family, leader=args.leader_serial, follower=args.follower_serial)
            print(f"Configuration: {setup.path}\n{output}", end="")
            return status
        from .web.init_wizard import serve

        launch = serve(setup, args.port, open_browser=not args.no_browser)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if launch is not None:
        with launch:
            # Keep the chosen URL reserved until collect takes ownership.
            launch.socket.set_inheritable(True)
            os.execv(sys.executable, [sys.executable, "-m", "macbot.cli", "--config", str(setup.path),
                                      "collect", "--listen-fd", str(launch.fileno())])


def cmd_cameras(args):
    from .cameras import list_cameras
    cams = list_cameras()
    if not cams:
        print("no video devices found")
    for c in cams:
        print(f"{c['name']:<36s} {c['unique_id']:<24s} {c['model']}")
    print("\nPut the uniqueID of your scene camera into cameras.<name>.unique_id.")


def cmd_joints(args):
    cfg = _cfg(args, resolve=True)
    rcfg = cfg["robot"]
    if rcfg.get("type") == "sim":
        raise SystemExit("use collect or run to see simulated joints")
    if rcfg.get("type", "rebot_b601") == "rebot_b601":
        _preload(cfg)
        from .robot.rebot import JOINTS, make_leader, read_joints_passive
        leader = make_leader(port=rcfg.get("leader_port"), leader_id=rcfg.get("leader_id", "rebot_arm_102_leader"))
        leader.connect(calibrate=False)    # leader: unlock (torque off) and reset turn counters; it does not move
        act = leader.get_action()
        leader.disconnect()
        fol = read_joints_passive(rcfg)
        print(f"{'joint':14s} {'leader':>8s} {'follower':>9s} {'diff':>8s}")
        for n in JOINTS:
            a, f = act[f"{n}.pos"], fol[n]
            print(f"{n:14s} {a:8.1f} {f:9.1f} {a - f:+8.1f}")
        return
    # Generic LeRobot arms: only the leader is read. Connecting a follower enables its torque, and
    # disconnecting it again would let the arm drop, so it is not touched here.
    from .robot.lerobot_generic import joint_names, make_leader, require_calibration
    if not rcfg.get("leader"):
        raise SystemExit("robot.leader is not set")
    leader = make_leader(rcfg["leader"])
    require_calibration(leader)
    try:
        leader.connect(calibrate=False)
        act = leader.get_action()
    finally:
        if leader.is_connected:
            leader.disconnect()
    for n in joint_names(leader):
        print(f"{n:16s} {act[f'{n}.pos']:8.1f}")


def cmd_robots(args):
    from .robot.lerobot_generic import known_types
    robots, teleops = known_types()
    print("robot.type: rebot_b601 (reference profile), sim (local simulation), or lerobot:\n")
    print("registered follower types:", ", ".join(r for r in robots if r != "sim"))
    print("leaders:  ", ", ".join(teleops))
    print("\nInstall the driver extra for your robot; e.g. pip install 'lerobot[feetech]==0.7.0' for SO-100/101.")


def cmd_policies(args):
    from .policy.support import run
    run()


def _preload(cfg):
    """reBot talks CAN through PCBUSB, which must be preloaded before anything else (may re-exec)."""
    if cfg["robot"].get("type", "rebot_b601") == "rebot_b601":
        from .system import preload_pcan
        preload_pcan()


def cmd_collect(args):
    cfg = _cfg(args, resolve=None if args.fake else True)
    if not args.fake:
        _preload(cfg)
    from .collect.server import run
    run(cfg, fake=args.fake, port=args.port, out=args.out, listen_fd=args.listen_fd)


def cmd_export(args):
    from .data.export import run
    run(_cfg(args), args)


def cmd_health(args):
    from .data.health import run
    run(_cfg(args), args)


def cmd_run(args):
    cfg = _cfg(args, resolve=None if args.fake else False)
    if not args.fake:
        _preload(cfg)
    from .policy.runner import run
    run(cfg, args)


def cmd_replay(args):
    from .viz.replay import run
    run(_cfg(args), args)


def cmd_video(args):
    from .viz.video import run
    run(args)


def cmd_bench(args):
    from .policy.bench import run
    run(_cfg(args), args)


def _backend_args(parser):
    parser.add_argument("--backend", choices=["torch", "mlx"], help="policy backend (default: torch or YAML setting)")
    parser.add_argument("--policy-impl", choices=["auto", "generic"], default="auto",
                        help="use the generic LeRobot adapter even for ACT or pi0.5 (torch only)")
    parser.add_argument("--mlx-bits", type=int, choices=[4, 8], help="experimental lossy MLX weight quantization")
    parser.add_argument("--mlx-time-grid", choices=["uniform", "quadratic"], default="uniform",
                        help="MLX Euler schedule; quadratic is an opt-in lossy experiment")
    parser.add_argument("--mlx-quantize", choices=["prefix", "expert", "both"], default="prefix",
                        help="weight groups to quantize with --mlx-bits")


def main(argv=None):
    from .config import ConsolePortError
    from .robot.usb import FAMILIES

    ap = argparse.ArgumentParser(prog="macbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML file overriding the built-in defaults")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    p = sub.add_parser("init", help="detect arms and cameras and create a configuration")
    p.add_argument("--yes", action="store_true", help="configure without a browser when roles are unambiguous")
    p.add_argument("--family", choices=FAMILIES, help="confirm a robot model for scripted setup")
    p.add_argument("--leader-serial", help="confirm the leader's USB serial for scripted setup")
    p.add_argument("--follower-serial", help="confirm the follower's USB serial for scripted setup")
    p.add_argument("--port", type=int, help="local setup console port (default: 8765 or init.port)")
    p.add_argument("--no-browser", action="store_true", help="print the local setup URL without opening it")
    p.set_defaults(fn=cmd_init)

    sub.add_parser("doctor", help="check the configured robot and cameras without connecting").set_defaults(fn=cmd_doctor)
    sub.add_parser("cameras", help="list video devices and uniqueIDs").set_defaults(fn=cmd_cameras)
    sub.add_parser("joints", help="read joint angles without powering motors").set_defaults(fn=cmd_joints)
    sub.add_parser("robots", help="list the LeRobot robot and leader types available here").set_defaults(fn=cmd_robots)
    sub.add_parser("policies", help="list LeRobot policy types and their macOS support").set_defaults(fn=cmd_policies)

    p = sub.add_parser("collect", help="teleoperate and record (web page)")
    p.add_argument("--fake", action="store_true", help="simulate robot, leader and cameras")
    port_args = p.add_mutually_exclusive_group()
    port_args.add_argument("--port", type=int, help="collection console port (default: 8766 or teleop.port)")
    port_args.add_argument("--listen-fd", type=int, help=argparse.SUPPRESS)
    p.add_argument("--out", help="episode folder (default <data_root>/raw/<task>)")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("export", help="recorded episodes -> LeRobot dataset")
    p.add_argument("--data", help="episode folder (default <data_root>/raw/<task>)")
    p.add_argument("--out", help="dataset folder (default <data_root>/datasets/<task>[_bin])")
    p.add_argument("--repo-id")
    p.add_argument("--gripper", choices=["raw", "binary"], default="raw", help="gripper action: leader angle, or 0/1")
    p.add_argument("--close-above", type=float, help="binary: leader value above this = closed (default: export.gripper_close_above)")
    p.add_argument("--open-below", type=float, help="binary: once closed, below this = open (default: export.gripper_open_below)")
    p.add_argument("--open-deg", type=float, help="binary: gripper value commanded for 'open' (default: export.gripper_open_deg)")
    p.add_argument("--close-deg", type=float, help="binary: gripper value commanded for 'closed' (default: export.gripper_close_deg)")
    p.add_argument("--skip-warnings", action="store_true", help="also skip episodes with warnings")
    p.add_argument("--images", action="store_true", help="store PNG frames instead of video (debugging)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("health", help="tracking and timing report of recorded episodes")
    p.add_argument("--data", help="episode folder (default <data_root>/raw/<task>)")
    p.add_argument("--split", type=int, help="compare episodes before / from this index (e.g. after a repair)")
    p.set_defaults(fn=cmd_health)

    p = sub.add_parser("run", help="run a policy on the robot (web page)")
    p.add_argument("--policy", required=True, help="LeRobot pretrained_model folder")
    p.add_argument("--mode", choices=["auto", "chunk", "ensemble", "async", "rtc", "step"], default="auto",
                   help="auto: ACT -> ensemble, torch pi0.5 -> rtc, MLX pi0.5 -> async, generic -> chunk or step")
    p.add_argument("--k", type=int, default=30, help="chunk mode: steps executed per chunk")
    p.add_argument("--te-m", type=float, default=0.01, help="ensemble: weight exp(m * age), as in ACT")
    p.add_argument("--te-hz", type=float, default=30.0, help="ensemble: max inference rate")
    p.add_argument("--rtc-beta", type=float, default=10.0, help="rtc: max guidance weight")
    p.add_argument("--rtc-horizon", type=int, default=20, help="rtc: guided prefix length in policy steps")
    p.add_argument("--gripper", default="auto", help="auto (find gripper.json), raw, or a path to gripper.json")
    p.add_argument("--grip-debounce", type=float, default=0.6, help="binary gripper: seconds of 'open' before opening")
    p.add_argument("--steps", type=int, help="pi0.5 denoising steps (default from checkpoint)")
    p.add_argument("--no-compile", action="store_true", help="pi0.5: skip backend compilation")
    _backend_args(p)
    p.add_argument("--speed", type=float, help="0.1-1.0, default from config")
    p.add_argument("--device", default="mps")
    p.add_argument("--port", type=int, help="policy console port (default: 8767 or run.port)")
    p.add_argument("--fake", action="store_true", help="simulate robot and cameras (policy runs for real)")
    p.add_argument("--autostart", type=float, default=0, help="in fake or sim mode: run this many seconds, save, quit")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("replay", help="browse recorded rollouts")
    p.add_argument("--dir", help="rollout folder (default <data_root>/rollouts/<task>)")
    p.add_argument("--port", type=int, help="replay console port (default: 8768 or replay.port)")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("video", help="rollout -> side-by-side MP4")
    p.add_argument("rollout")
    p.add_argument("--out")
    p.add_argument("--height", type=int, default=540)
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser("bench", help="time policy inference")
    p.add_argument("--policy", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("-n", type=int, default=30)
    p.add_argument("--steps", type=int, help="pi0.5 denoising steps (default from checkpoint)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--profile", action="store_true", help="also report synchronized generic or pi0.5 stage timings")
    _backend_args(p)
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except ConsolePortError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    sys.exit(main())
