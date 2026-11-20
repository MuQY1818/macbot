"""Synchronized PI05 stage timers for offline benchmarks."""

from __future__ import annotations

from collections import defaultdict
from functools import wraps
from time import perf_counter
from unittest.mock import patch


class TorchStages:
    def __init__(self, sync):
        self.sync = sync
        self.samples = defaultdict(list)
        self.counts = defaultdict(int)

    def wrap(self, fn, name, indexed=False):
        @wraps(fn)
        def timed(*args, **kwargs):
            key = name
            if indexed:
                key = f"{name}_{self.counts[name]:02d}"
                self.counts[name] += 1
            self.sync()
            start = perf_counter()
            value = fn(*args, **kwargs)
            self.sync()
            self.samples[key].append((perf_counter() - start) * 1e3)
            return value

        return timed

    def instrument(self, policy, stack):
        from . import models

        m = policy.pol.model
        v = m.paligemma_with_expert
        prefix = v.paligemma.model.language_model
        for obj, attr, name, indexed in (
            (models, "_to_obs", "rgb_to_host_tensors", False),
            (policy.pol, "_preprocess_images", "image_resize_normalize", False),
            (v, "embed_image", "vision_and_projection", True),
            (prefix, "forward", "prefix_and_kv", False),
            (m, "denoise_step", "denoise", True),
            (m, "sample_noise", "noise", False),
            (policy.torch.Tensor, "cpu", "explicit_cpu_copy", True),
            (policy.torch.mps, "synchronize", "explicit_mps_sync", True),
        ):
            stack.enter_context(patch.object(obj, attr, self.wrap(getattr(obj, attr), name, indexed)))
        for stem, pipeline in (("pre", policy.pre), ("post", policy.post)):
            steps = [self.wrap(step, f"{stem}_{type(step).__name__}") for step in pipeline.steps]
            stack.enter_context(patch.object(pipeline, "steps", steps))


class MLXStages:
    def __init__(self, mx):
        self.mx = mx
        self.samples = defaultdict(list)

    def __call__(self, name, fn, *args):
        self.mx.synchronize()
        start = perf_counter()
        value = fn(*args)
        self.mx.eval(value)
        self.mx.synchronize()
        self.samples[name].append((perf_counter() - start) * 1e3)
        return value
