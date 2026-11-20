"""Carry the dataset framing into each portable LeRobot checkpoint."""
import argparse
import json
import shutil
from pathlib import Path


def copy_framing(output_dir):
    for config in (output_dir / "checkpoints").glob("*/pretrained_model/train_config.json"):
        if config.parent.parent.is_symlink():
            continue
        training = json.loads(config.read_text())
        if root := (training.get("dataset") or {}).get("root"):
            path = Path(root).expanduser() / "meta" / "camera_framing.json"
            if path.is_file():
                shutil.copyfile(path, config.parent / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output_dir", required=True, type=Path)
    args, _ = parser.parse_known_args()
    copy_framing(args.output_dir)
