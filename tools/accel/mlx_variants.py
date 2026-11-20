"""Training-free, opt-in replay experiments. None are production defaults."""

import math

import mlx.core as mx

from macbot.policy.mlx_pi05 import _PREFIX


def flow_times(n, solver):
    if n < 1:
        raise ValueError("Expected a positive evaluation budget")
    if solver == "midpoint":
        if n % 2:
            raise ValueError("Midpoint needs an even number of velocity evaluations")
    elif solver not in ("euler", "ab2", "quadratic"):
        raise ValueError(f"Unknown flow solver: {solver}")
    power = 2 if solver == "quadratic" else 1
    return [(1 - i / n) ** power for i in range(n)] + [0.0]


def integrate(x, velocity, times, solver):
    """Times descend from noise to action. Keep the state and increments in FP32."""
    previous = None
    for i, (t, end) in enumerate(zip(times[:-1], times[1:], strict=True)):
        v = velocity(i, x).astype(mx.float32)
        if solver == "midpoint":
            if i % 2 == 0:
                previous = x
                x = x + (end - t) * v
            else:
                x = previous + (end - times[i - 1]) * v
        elif solver == "ab2":
            x = x + (end - t) * (v if previous is None else 1.5 * v - 0.5 * previous)
            previous = v
        else:
            x = x + (end - t) * v
    return x


def install_solver(policy, solver):
    times = flow_times(policy.steps, solver)
    config, net = policy.config, policy.network
    policy.cond = net.time_condition(mx.array(times[:-1], dtype=mx.float32), config.min_period, config.max_period)
    policy.modulation = None if policy.modulation is None else net.adarms(policy.cond)
    mx.eval(policy.cond, policy.modulation)

    def stages(images, tokens, mask, all_valid, noise, *, stage=None):
        run = (lambda name, fn, *args: fn(*args)) if stage is None else stage
        vision = run("vision", policy.vision, images)
        valid, kv = run("prefix_and_kv", policy.prefix, policy.pool(vision), tokens, mask, not all_valid)
        pos, keep = net.expert_inputs(valid, noise.shape[1], masked=not all_valid)

        def speed(i, x):
            cond, mod = policy.velocity_args(i)
            return run(f"denoise_{i:02d}", policy.velocity, x, cond, kv, pos, keep, mod)

        return integrate(noise, speed, times, solver)

    policy.stages = stages


def install_pruning(policy, fraction):
    """Keep the highest attention patches per camera, in original spatial order."""
    net = policy.network
    if not 0 < fraction < 1 or net.g.text_depth < 3 or not net.fused_rope or not policy.trim:
        raise ValueError("Visual pruning needs a fraction in (0, 1), three layers, trim and fused RoPE")
    if policy.pool_images:
        raise ValueError("Pooling and attention pruning are separate experiments")
    original_rotary = net.rotary

    def rotary(x, pos):
        if not isinstance(pos, tuple):
            return original_rotary(x, pos)
        cos, sin = pos
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate((-b, a), axis=-1) * sin

    net.rotary = rotary

    def prefix(vision, tokens, mask, full_mask=True):
        if full_mask or tokens.shape[1] == 0:
            raise ValueError("Attention pruning requires a nonempty fully trimmed single prompt")
        views, patches, _ = vision.shape
        n = views * patches
        count = max(1, math.ceil(patches * fraction))
        x = mx.concatenate((vision.flatten(0, 1)[None], net.embed_text(tokens)), axis=1)
        length = x.shape[1]
        pos = mx.array([0], dtype=mx.int32)
        cache = []
        for i in range(net.g.text_depth):
            root = f"{_PREFIX}layers.{i}."
            if i == 1:
                # Use the final text query in block two; the returned K already includes RoPE.
                h, _ = net.norm(x[:, -1:], root + "input_layernorm", None)
                names = tuple(root + "self_attn." + s for s in ("q_proj", "k_proj", "v_proj"))
                q = net.heads(net.project(h, names)[0], net.g.heads)
                q = original_rotary(q, mx.array([length - 1], dtype=mx.int32))
            x, kv = net.text_layer(x, root, pos, None, kv_only=i == net.g.text_depth - 1)
            cache.append(kv)
            if i == 1:
                logits = q.astype(mx.float32) @ kv[0].astype(mx.float32).swapaxes(-1, -2)
                score = mx.softmax(logits * net.g.head_dim ** -0.5, axis=-1).mean(axis=1)
                score = score[..., :n].reshape(views, patches)
                top = mx.argsort(score, axis=-1)[:, -count:] + mx.arange(views)[:, None] * patches
                keep = mx.concatenate((mx.sort(top.flatten()), mx.arange(n, length)))
                x = x[:, keep]
                cache = [(k[:, :, keep], v[:, :, keep]) for k, v in cache]
                # Build exact fused-RoPE sin/cos at original positions, then gather.
                d = net.g.head_dim
                basis = mx.broadcast_to(mx.concatenate((mx.ones(d // 2), mx.zeros(d // 2))), (1, 1, length, d))
                rot = original_rotary(basis.astype(x.dtype), pos)[:, :, keep]
                cos, sin = mx.split(rot, 2, axis=-1)
                pos = mx.concatenate((cos, cos), axis=-1), mx.concatenate((sin, sin), axis=-1)
        # Expert positions start after the ORIGINAL sequence; its unmasked keys can be shorter.
        return mx.ones((1, length), dtype=mx.bool_), tuple(cache)

    policy.prefix = prefix
