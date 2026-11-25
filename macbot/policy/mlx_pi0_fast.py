"""MLX inference for trained LeRobot PI0-FAST checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.fft import idct

from .mlx_chunk import MLXChunkIO, stage_call
from .mlx_pi05 import _PREFIX, _VISION, _VLM, PaliGemmaNetwork
from .pali_tokenizer import resolve_pali_tokenizer


class FastPaliGemmaNetwork(PaliGemmaNetwork):
    """PaliGemma vision and language prefix without the continuous action expert."""

    adaptive_expert = False

    def flow_shapes(self, a, d):
        return ()

    def validate_weights(self):
        g, w = self.g, self.w
        shapes = {}

        def add(name, shape, bias=False):
            shapes[name + ".weight"] = shape
            if bias:
                shapes[name + ".bias"] = (shape[0],)

        def width(key, axis=0):
            if key not in w or w[key].ndim <= axis:
                raise ValueError(f"Missing or invalid PaliGemma FAST weight: {key}")
            return w[key].shape[axis]

        v = width(_VISION + "embeddings.patch_embedding.weight")
        p = width(_PREFIX + "embed_tokens.weight", 1)
        add(_VISION + "embeddings.patch_embedding", (v, 3, g.patch, g.patch), True)
        add(_VISION + "embeddings.position_embedding", (width(_VISION + "embeddings.position_embedding.weight"), v))
        add(_VISION + "post_layernorm", (v,), True)
        add(_VLM + "multi_modal_projector.linear", (p, v), True)
        add(_PREFIX + "embed_tokens", (width(_PREFIX + "embed_tokens.weight"), p))
        for i in range(g.vision_depth):
            root = f"{_VISION}encoder.layers.{i}."
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                add(root + "self_attn." + name, (v, v), True)
            for name in ("layer_norm1", "layer_norm2"):
                add(root + name, (v,), True)
            h = width(root + "mlp.fc1.weight")
            add(root + "mlp.fc1", (h, v), True)
            add(root + "mlp.fc2", (v, h), True)
        for i in range(g.text_depth):
            root = f"{_PREFIX}layers.{i}."
            for name in ("input_layernorm", "post_attention_layernorm"):
                add(root + name, (p,))
            for name, shape in (
                ("q_proj", (g.heads * g.head_dim, p)),
                ("k_proj", (g.kv_heads * g.head_dim, p)),
                ("v_proj", (g.kv_heads * g.head_dim, p)),
                ("o_proj", (p, g.heads * g.head_dim)),
            ):
                add(root + "self_attn." + name, shape)
            h = width(root + "mlp.up_proj.weight")
            add(root + "mlp.up_proj", (h, p))
            add(root + "mlp.gate_proj", (h, p))
            add(root + "mlp.down_proj", (p, h))
        add(_PREFIX + "norm", (p,))
        missing = shapes.keys() - w.keys()
        unexpected = w.keys() - shapes.keys()
        malformed = [k for k in shapes.keys() & w.keys() if tuple(w[k].shape) != shapes[k]]
        if missing or unexpected or malformed:
            raise ValueError(
                f"Unsupported PaliGemma FAST weights: missing={sorted(missing)[:4]}, "
                f"unexpected={sorted(unexpected)[:4]}, shape mismatch={sorted(malformed)[:4]}"
            )

    def prefix_with_hidden(self, vision, tokens, token_mask):
        text = self.embed_text(tokens)
        x = mx.concatenate((vision.flatten(0, 1)[None], text), axis=1)
        valid = mx.concatenate((mx.ones((1, vision.shape[0] * vision.shape[1]), dtype=mx.bool_), token_mask), axis=1)
        positions = self.pos_embed(mx.cumsum(valid, axis=1) - 1)
        attention_mask = valid[:, None, None, :] & valid[:, None, :, None]
        cache = []
        for i in range(self.g.text_depth):
            x, kv = self.text_layer(
                x,
                f"{_PREFIX}layers.{i}.",
                positions,
                attention_mask,
                kv_only=False,
            )
            cache.append(kv)
        return x, valid, tuple(cache)

    def decode_step(self, token_embedding, cache, position):
        positions = self.pos_embed(mx.array([[position]], dtype=mx.int32))
        x = token_embedding
        new_cache = []
        for i, previous in enumerate(cache):
            x, kv = self.text_layer(
                x,
                f"{_PREFIX}layers.{i}.",
                positions,
                None,
                previous=previous,
            )
            new_cache.append(kv)
        hidden = self.norm(x, _PREFIX + "norm", None)[0]
        return hidden, tuple(new_cache)

    def logits(self, hidden):
        return hidden.astype(self.dtype) @ self.w[_PREFIX + "embed_tokens.weight"].T


class FastActionTokenizer:
    """Load the trained FAST BPE sidecar and perform its inverse DCT."""

    def __init__(self, repo_id):
        import json

        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        path = Path(repo_id)
        if not path.is_dir():
            path = Path(snapshot_download(repo_id=repo_id, allow_patterns=["processor_config.json", "bpe_tokenizer/*"]))
        config = json.loads((path / "processor_config.json").read_text())
        bpe = path / "bpe_tokenizer" / "tokenizer.json"
        if not bpe.is_file():
            bpe = path / "tokenizer.json"
        self.bpe = Tokenizer.from_file(str(bpe))
        self.min_token = int(config["min_token"])
        self.scale = float(config["scale"])

    def decode(self, tokens, horizon, action_dim):
        values = []
        expected = horizon * action_dim
        for token in tokens:
            try:
                text = self.bpe.decode([int(token)])
                coeff = np.fromiter((ord(char) for char in text), dtype=np.float32) + self.min_token
            except (TypeError, ValueError):
                coeff = np.zeros(0, dtype=np.float32)
            coeff = coeff[:expected]
            if coeff.size < expected:
                coeff = np.pad(coeff, (0, expected - coeff.size))
            coeff = coeff.reshape(horizon, action_dim)
            values.append(idct(coeff / self.scale, axis=0, norm="ortho"))
        return np.stack(values).astype(np.float32)


class PI0FastMLX(MLXChunkIO):
    kind = "pi0_fast"
    supports_rtc = False

    def __init__(self, path, task, *, dtype=mx.bfloat16, tokenizer_path=None, action_tokenizer_path=None, **io):
        import_module = __import__("importlib").import_module
        import_module("lerobot.policies.pi0_fast.processor_pi0_fast")
        config_path = Path(path) / "config.json"
        official = json.loads(config_path.read_text()).get("text_tokenizer_name") or "google/paligemma-3b-pt-224"
        self.text_tokenizer = resolve_pali_tokenizer(tokenizer_path or Path(path), official)
        from unittest.mock import patch

        from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep

        def skip_action_tokenizer(step):
            # Inference never supplies an action to this training-only processor step.
            step.action_tokenizer = None
            step._paligemma_tokenizer = self.text_tokenizer

        with patch.object(ActionTokenizerProcessorStep, "__post_init__", skip_action_tokenizer):
            super().__init__(path, task, tokenizer=self.text_tokenizer, **io)
        config = self.config
        if config.paligemma_variant != "gemma_2b" or tuple(config.image_resolution) != (224, 224):
            raise ValueError("MLX pi0_fast requires the stock 224px Gemma 2B prefix")
        if config.temperature != 0:
            raise ValueError("MLX pi0_fast currently supports greedy decoding only")
        self.network = FastPaliGemmaNetwork(
            mx.load(str(path / "model.safetensors")),
            dtype=dtype,
            fused_attention=True,
            fused_rope=True,
            pack_linears=True,
        )
        self.vision = self.network.vision
        self.action_tokenizer = FastActionTokenizer(action_tokenizer_path or config.action_tokenizer_name)
        self.fast_skip_tokens = int(config.fast_skip_tokens)
        self.max_decoding_steps = int(config.max_decoding_steps)
        self.action_dim = int(config.output_features["action"].shape[0])
        self.tokenizer_vocab_size = self.text_tokenizer.vocab_size
        self.action_end_token = int(self.text_tokenizer.encode("|", add_special_tokens=False)[0])

    def enable_compile(self):
        self.vision = mx.compile(self.vision)
        self.network.prefix_with_hidden = mx.compile(self.network.prefix_with_hidden)
        self.network.decode_step = mx.compile(self.network.decode_step)

    def prepare(self, imgs, q):
        from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad_torch
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = self.prepare_batch(imgs, q)
        res = tuple(self.config.image_resolution)
        images = []
        for key in self.config.image_features:
            if key not in self.cameras:
                continue
            image = batch[key].float().permute(0, 2, 3, 1)
            if tuple(image.shape[1:3]) != res:
                image = resize_with_pad_torch(image, *res)
            images.append((image * 2 - 1).numpy())
        mask = batch[OBS_LANGUAGE_ATTENTION_MASK].numpy()
        n = int(mask.sum())
        if not mask[0, :n].all() or mask[0, n:].any():
            raise ValueError("MLX pi0_fast requires right-padded text")
        tokens = batch[OBS_LANGUAGE_TOKENS].numpy()[:, :n]
        state = pad_vector(batch["observation.state"].float(), self.config.max_state_dim)
        return (
            mx.array(np.concatenate(images)),
            mx.array(tokens, dtype=mx.int32),
            mx.array(mask[:, :n], dtype=mx.bool_),
            state,
        )

    def _generate(self, images, tokens, mask):
        vision = self.vision(images)
        bos = mx.full((tokens.shape[0], 1), self.text_tokenizer.bos_token_id, dtype=mx.int32)
        tokens = mx.concatenate((tokens, bos), axis=1)
        mask = mx.concatenate((mask, mx.ones((mask.shape[0], 1), dtype=mx.bool_)), axis=1)
        hidden, valid, cache = self.network.prefix_with_hidden(vision, tokens, mask)
        next_token = mx.argmax(self.network.logits(hidden[:, -1:]), axis=-1).astype(mx.int32)
        result = []
        finished = np.zeros((tokens.shape[0],), dtype=bool)
        position = int(valid.shape[1])
        for step in range(self.max_decoding_steps):
            if step:
                token_embedding = self.network.embed_text(next_token)
                hidden, cache = self.network.decode_step(token_embedding, cache, position)
                position += 1
                next_token = mx.argmax(self.network.logits(hidden), axis=-1).astype(mx.int32)
            result.append(next_token[:, 0])
            finished |= np.array(next_token[:, 0]) == self.action_end_token
            if bool(finished.all()):
                break
        out = np.zeros((tokens.shape[0], self.max_decoding_steps), dtype=np.int32)
        if result:
            values = np.stack([np.array(item) for item in result], axis=1)
            out[:, : values.shape[1]] = values
        return out

    def __call__(self, imgs, q, *, return_raw=False, stage=None, **_):
        with self.torch.inference_mode():
            images, tokens, mask, _ = self.prepare(imgs, q)
            generated = stage_call(stage, "decode_tokens", self._generate, images, tokens, mask)
            decoded = self.action_tokenizer.decode(
                [
                    self._tokens_to_action_tokens(row)
                    for row in generated
                ],
                self.H,
                self.action_dim,
            )
            raw = np.zeros((1, self.H, self.action_dim), dtype=np.float32)
            raw[0] = decoded
            action = self.post(self.torch.from_numpy(raw)).float().cpu().numpy()
            if not np.isfinite(action).all():
                raise ValueError("MLX pi0_fast returned non-finite actions")
            return (action[0], raw[0]) if return_raw else action[0]

    def _tokens_to_action_tokens(self, tokens):
        pieces = self.text_tokenizer.convert_ids_to_tokens(tokens)
        prefix = self.text_tokenizer.convert_ids_to_tokens(
            self.text_tokenizer.encode("Action: ", add_special_tokens=False)
        )
        end = pieces.index("|") if "|" in pieces else len(pieces)
        pieces = pieces[:end]
        i = 0
        while i <= len(pieces) - len(prefix):
            if pieces[i : i + len(prefix)] == prefix:
                del pieces[i : i + len(prefix)]
            else:
                i += 1
        ids = self.text_tokenizer.convert_tokens_to_ids(pieces)
        return [self.tokenizer_vocab_size - 1 - self.fast_skip_tokens - int(value) for value in ids]
