"""Explicit MPS experiments for ACT and Diffusion."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import wraps


@dataclass(frozen=True)
class TorchOptions:
    autocast: str | None = None
    channels_last: bool = False
    compile: bool = False
    single_readback: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(*(getattr(args, f"torch_{name}", default) for name, default in (
            ("autocast", None), ("channels_last", False), ("compile", False), ("single_readback", False),
        )))

    def validate(self, backend, kind, device):
        if self == TorchOptions():
            return
        if backend != "torch" or device != "mps" or kind not in {"act", "diffusion"}:
            sys.exit("Torch experiments require ACT or Diffusion on --backend torch --device mps; excluded from auto")
        if self.compile and kind == "act":
            sys.exit("ACT torch.compile is disabled: the tested MPS build fails to compile its Metal kernels")
        if self.single_readback and kind != "act":
            sys.exit("--torch-single-readback is only supported for ACT")

    def install(self, policy):
        if self == TorchOptions():
            return
        torch = policy.torch
        if self.channels_last:
            torch.nn.utils.convert_conv2d_weight_memory_format(policy.pol, torch.channels_last)
        if policy.kind == "diffusion":
            # Leave scheduler arithmetic, queueing, and processor dtypes unchanged.
            owner, name = policy.pol.diffusion.unet, "forward"
        else:
            owner, name = policy.pol, "predict_action_chunk"
        predict = getattr(owner, name)
        if self.compile:
            predict = torch.compile(predict, dynamic=False)
        if self.autocast:
            dtype = torch.float16 if self.autocast == "fp16" else torch.bfloat16
            original = predict

            @wraps(original)
            def mixed(*args, **kwargs):
                with torch.autocast("mps", dtype=dtype):
                    result = original(*args, **kwargs)
                return result.float()

            predict = mixed
        setattr(owner, name, predict)
        policy.single_readback = self.single_readback
