"""MLX inference for LeRobot ACT with a frozen ResNet-18 backbone."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .mlx_chunk import DenseWeights, MLXChunkIO, attention, stage_call


class ACTNetwork(DenseWeights):
    def __init__(self, weights, config, dtype):
        weights = {k.removeprefix("model."): v for k, v in weights.items()
                   if not k.removeprefix("model.").startswith("vae_encoder")}
        self.config = config
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self._fold_batch_norm(weights, "backbone.conv1", "backbone.bn1")
        for i in range(1, 5):
            for j in range(2):
                root = f"backbone.layer{i}.{j}."
                for k in (1, 2):
                    self._fold_batch_norm(weights, root + f"conv{k}", root + f"bn{k}")
                if i > 1 and j == 0:
                    self._fold_batch_norm(weights, root + "downsample.0", root + "downsample.1")
        for stack, count in (("encoder", config.n_encoder_layers), ("decoder", config.n_decoder_layers)):
            for i in range(count):
                names = ("self_attn",) if stack == "encoder" else ("self_attn", "multihead_attn")
                for name in names:
                    root = f"{stack}.layers.{i}.{name}."
                    for param in ("weight", "bias"):
                        packed = weights.pop(root + "in_proj_" + param)
                        q, k, v = mx.split(packed, 3, axis=0)
                        for part, value in zip(("q", "k", "v", "qk"), (q, k, v, packed[:q.shape[0] * 2]), strict=True):
                            weights[f"{root}{part}.{param}"] = value
        super().__init__(weights, dtype)
        mx.eval(self.w)

    @staticmethod
    def _fold_batch_norm(weights, conv, bn):
        def pop(name):
            return weights.pop(bn + "." + name).astype(mx.float32)

        scale = pop("weight") * mx.rsqrt(pop("running_var") + 1e-5)
        weights[conv + ".weight"] = weights[conv + ".weight"].astype(mx.float32) * scale[:, None, None, None]
        weights[conv + ".bias"] = pop("bias") - pop("running_mean") * scale

    def vision(self, x):
        x = self.pool(nn.relu(self.conv(x, "backbone.conv1", stride=2, padding=3)))
        for i in range(1, 5):
            for j in range(2):
                root = f"backbone.layer{i}.{j}."
                stride = 2 if i > 1 and j == 0 else 1
                skip = self.conv(x, root + "downsample.0", stride=stride) if stride == 2 else x
                y = nn.relu(self.conv(x, root + "conv1", stride=stride, padding=1))
                x = nn.relu(self.conv(y, root + "conv2", padding=1) + skip)
        return self.conv(x, "encoder_img_feat_input_proj")

    def attend(self, query, key, value, root, *, self_attention=False):
        if self_attention:
            q, k = mx.split(self.linear(query, root + "qk"), 2, axis=-1)
        else:
            q, k = self.linear(query, root + "q"), self.linear(key, root + "k")
        v = self.linear(value, root + "v")
        return self.linear(attention(q, k, v, self.config.n_heads), root + "out_proj")

    def transformer(self, state, image_tokens, image_positions):
        conf = self.config
        latent = self.w["encoder_latent_input_proj.bias"][None, None]
        state = self.linear(state, "encoder_robot_state_input_proj")[:, None]
        x = mx.concatenate((latent, state, image_tokens), axis=1)
        pos = mx.concatenate((self.w["encoder_1d_feature_pos_embed.weight"][None], image_positions), axis=1)
        activation = nn.relu if conf.feedforward_activation == "relu" else nn.gelu

        def residual(x, name, fn):
            if conf.pre_norm:
                return x + fn(self.norm(x, name))
            return self.norm(x + fn(x), name)

        for i in range(conf.n_encoder_layers):
            root = f"encoder.layers.{i}."
            attn_root = root + "self_attn."
            x = residual(x, root + "norm1", lambda h, attn_root=attn_root: self.attend(
                h + pos, None, h, attn_root, self_attention=True,
            ))
            linear1, linear2 = root + "linear1", root + "linear2"
            x = residual(x, root + "norm2",
                         lambda h, linear1=linear1, linear2=linear2:
                         self.linear(activation(self.linear(h, linear1)), linear2))
        memory = self.norm(x, "encoder.norm") if conf.pre_norm else x
        query_pos = self.w["decoder_pos_embed.weight"][None]
        x = mx.zeros_like(query_pos)
        for i in range(conf.n_decoder_layers):
            root = f"decoder.layers.{i}."
            self_root = root + "self_attn."
            x = residual(x, root + "norm1", lambda h, self_root=self_root: self.attend(
                h + query_pos, None, h, self_root, self_attention=True,
            ))
            cross_root = root + "multihead_attn."
            x = residual(x, root + "norm2",
                         lambda h, cross_root=cross_root:
                         self.attend(h + query_pos, memory + pos, memory, cross_root))
            linear1, linear2 = root + "linear1", root + "linear2"
            x = residual(x, root + "norm3",
                         lambda h, linear1=linear1, linear2=linear2:
                         self.linear(activation(self.linear(h, linear1)), linear2))
        return self.linear(self.norm(x, "decoder.norm"), "action_head")


class ACTMLX(MLXChunkIO):
    kind = "act"
    requires_task = False

    def __init__(self, path, task, *, dtype=mx.float32, **io):
        from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d

        super().__init__(path, task, **io)
        c = self.config
        if (c.vision_backbone != "resnet18" or c.replace_final_stride_with_dilation
                or c.env_state_feature is not None or c.feedforward_activation not in {"relu", "gelu"}
                or c.temporal_ensemble_coeff is not None or c.dim_model % 4):
            raise ValueError("MLX ACT requires the ResNet-18 joint-state model with no checkpoint temporal ensemble")
        self.H = c.chunk_size  # Match the built-in ACT wrapper's full prediction horizon.
        self.dtype = dtype
        try:
            net = ACTNetwork(mx.load(str(path / "model.safetensors")), c, dtype)
        except KeyError as exc:
            raise ValueError(f"Missing ACT weight: {exc}") from exc
        self.vision, self.transformer = net.vision, net.transformer
        # The position grid depends only on the configured image sizes.
        position = ACTSinusoidalPositionEmbedding2d(c.dim_model // 2)
        grids = []
        for feature in c.image_features.values():
            _, height, width = feature.shape
            grid = position(self.torch.zeros(1, 1, (height + 31) // 32, (width + 31) // 32))
            grids.append(grid.permute(0, 2, 3, 1).flatten(1, 2))
        self.positions = mx.array(self.torch.cat(grids, dim=1).numpy()).astype(dtype)
        mx.eval(self.positions)

    def enable_compile(self):
        self.vision, self.transformer = map(mx.compile, (self.vision, self.transformer))

    def prepare(self, imgs, q):
        from lerobot.utils.constants import OBS_STATE

        batch = self.prepare_batch(imgs, q)
        images = tuple(mx.array(batch[k].permute(0, 2, 3, 1).float().numpy()).astype(self.dtype)
                       for k in self.config.image_features)
        return images, mx.array(batch[OBS_STATE].float().numpy()).astype(self.dtype)

    def __call__(self, imgs, q, *, stage=None):
        with self.torch.inference_mode():
            images, state = stage_call(stage, "prepare", self.prepare, imgs, q)
            views = [stage_call(stage, f"vision_{i}", self.vision, img).flatten(1, 2) for i, img in enumerate(images)]
            x = stage_call(stage, "transformer", self.transformer, state, mx.concatenate(views, axis=1),
                           self.positions)
            return stage_call(stage, "readback_unnormalise", self.finish, x)
