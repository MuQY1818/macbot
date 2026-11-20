"""Emit action fingerprints; run from both source trees with the same Python and checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from macbot.policy.models import load_policy, raw_state_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(124)
    path = args.checkpoint
    cfg = json.loads((path / "config.json").read_text())
    p = load_policy(path, args.device, "put the cube in the box", backend=args.backend)
    q = np.linspace(0, 1, raw_state_shape(path, cfg)[0], dtype=np.float32)
    sizes = p.shapes if p.kind == "act" else {"top": (3, 470, 492), "wrist": (3, 360, 640)}
    obs = {cam: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for cam, (_, h, w) in sizes.items()}

    def digest(value):
        value = np.ascontiguousarray(value)
        assert np.isfinite(value).all()
        return {"shape": value.shape, "sha256": hashlib.sha256(value.tobytes()).hexdigest()}

    if p.kind == "act":
        print(json.dumps({"plain": digest(p(obs, q))}))
        return

    if not args.no_compile:
        p.enable_compile()
    config = p.config if args.backend == "mlx" else p.pol.config
    noise = rng.standard_normal((1, config.chunk_size, config.max_action_dim), dtype=np.float32)

    def call(**kwargs):
        if args.backend == "mlx":
            return p(obs, q, noise=noise, return_raw=True, **kwargs)
        t = p.torch.from_numpy(noise).to(args.device)
        with patch.object(p.pol.model, "sample_noise", lambda shape, device: t):
            return p(obs, q, return_raw=True, **kwargs)

    plain, raw = call()
    p.enable_rtc(10.0)
    guided, guided_raw = call(leftover=raw[5:], delay=8, horizon=20)
    print(json.dumps({name: digest(a) for name, a in
                      zip(("plain", "raw", "guided", "guided_raw"), (plain, raw, guided, guided_raw), strict=True)}))


if __name__ == "__main__":
    main()
