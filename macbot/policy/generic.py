"""Run a local LeRobot checkpoint using its native processors and inference contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import raw_state_shape

STATE = "observation.state"
ACTION = "action"
IMAGE_PREFIX = "observation.images."


def _processor_overrides(path, name, device):
    """Override the device only where the saved pipeline actually has a device step."""
    config = path / f"policy_{name}processor.json"
    if not config.is_file():
        raise ValueError(f"Missing {config.name}; export the trained LeRobot processors with the checkpoint")
    steps = json.loads(config.read_text())["steps"]
    overrides = {}
    for step in steps:
        key = step.get("registry_name") or step.get("class", "").rsplit(".", 1)[-1]
        if key in {"device_processor", "DeviceProcessorStep"}:
            override = {"device": device}
            if device == "mps" and step.get("config", {}).get("float_dtype") in {"float64", "double"}:
                override["float_dtype"] = "float32"
            overrides[key] = override
    return overrides, steps


def _features(path, metadata, cameras, camera_map, joint_count):
    mode = str(metadata.get("action_mode", "")).lower()
    space = str(metadata.get("action_space", "")).lower()
    if any(x in mode or x in space for x in ("ee6d", "eef", "end_effector", "end-effector", "cartesian", "se3")):
        raise ValueError("Checkpoint uses end-effector actions; macbot requires absolute joint positions")

    outputs = metadata.get("output_features") or {}
    action = outputs.get(ACTION) or {}
    width = action.get("shape", [])
    if set(outputs) != {ACTION} or action.get("type") != "ACTION" or len(width) != 1 or width[0] <= 0:
        raise ValueError("Expected a single flat 'action' feature containing absolute joint positions")
    if joint_count is not None and width != [joint_count]:
        raise ValueError(f"Policy action width is {width[0]}, but the robot has {joint_count} joints")
    state = raw_state_shape(path, metadata)
    xvla_padding = (metadata.get("type") == "xvla" and mode == "auto"
                    and (metadata.get("normalization_mapping") or {}).get("STATE") == "IDENTITY"
                    and state and len(state) == 1 and width[0] < state[0] <= metadata.get("max_state_dim", 0))
    if state is not None and state != width and metadata.get("type") not in {"pi0", "pi0_fast"} and not xvla_padding:
        raise ValueError(f"Expected joint state shape {width}, got {state}")

    features = metadata.get("input_features") or {}
    visuals = {}
    for key, feature in features.items():
        kind = feature.get("type")
        shape = feature.get("shape", [])
        if kind == "VISUAL" and key.startswith("observation."):
            if len(shape) != 3 or shape[0] != 3 or min(shape) < 1:
                raise ValueError(f"Camera {key!r} must have RGB shape (3, height, width), got {shape}")
            visuals[key] = tuple(shape)
        elif not ((key == STATE and kind == "STATE") or (key == "task" and kind == "LANGUAGE")):
            raise ValueError(f"Unsupported policy input {key!r} ({kind}); macbot provides RGB, joint state and task text")
    if metadata.get("type") == "tdmpc" and visuals and visuals.keys() != {"observation.image"}:
        raise ValueError("LeRobot TDMPC requires the checkpoint's single visual feature to be 'observation.image'; "
                         "map your physical camera to that key in policy.camera_map")

    if not isinstance(camera_map, dict) or not all(isinstance(v, str) for v in camera_map.values()):
        raise ValueError("policy.camera_map must map configured camera names to checkpoint feature names")
    if unknown := camera_map.keys() - cameras.keys():
        raise ValueError(f"Unknown config cameras in policy.camera_map: {', '.join(sorted(unknown))}")
    mapped = {}
    for name, (w, h) in cameras.items():
        key = camera_map.get(name, name)
        key = key if key.startswith("observation.") else IMAGE_PREFIX + key
        if key not in visuals:
            if name in camera_map:
                raise ValueError(f"Camera {name!r} maps to unknown checkpoint feature {key!r}")
            continue
        if key in mapped:
            raise ValueError(f"Multiple configured cameras map to checkpoint feature {key!r}")
        if visuals[key] != (3, h, w):
            raise ValueError(f"Camera {name!r} has save_size {[w, h]}, but {key!r} expects RGB {visuals[key]}")
        mapped[key] = name
    # These entries describe padding created and masked by the policy, not physical inputs.
    empty = {f"{IMAGE_PREFIX}empty_camera_{i}" for i in range(metadata.get("empty_cameras") or 0)}
    if missing := visuals.keys() - mapped.keys() - empty:
        raise ValueError(f"Missing policy cameras: {', '.join(sorted(missing))}; set policy.camera_map and cameras")
    if visuals and not (mapped.keys() - empty):
        raise ValueError("Configure at least one physical checkpoint camera")
    if mapped.keys() & empty:
        raise ValueError("Do not map physical cameras to checkpoint padding slots")
    return width[0], state is not None, mapped


class GenericPolicy:
    supports_rtc = False
    implementation = "generic"
    shapes = None  # Camera mapping and sizes are checked against the checkpoint at load time.

    def __init__(self, path: Path, device: str, task: str, *, cameras, camera_map=None, joint_count=None):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch, self.device, self.task = torch, device, task
        metadata = json.loads((path / "config.json").read_text())
        self.kind = metadata.get("type")
        if not self.kind:
            raise ValueError("Generic LeRobot checkpoints need a 'type' in config.json")
        if self.kind not in PreTrainedConfig.get_known_choices():
            from lerobot.utils.import_utils import register_third_party_plugins

            register_third_party_plugins()
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("PyTorch MPS is unavailable; use --device cpu or install an MPS-enabled PyTorch")
        self.width, self.has_state, self.cameras = _features(path, metadata, cameras, camera_map or {}, joint_count)
        pre_overrides, steps = _processor_overrides(path, "pre", device)
        post_overrides, _ = _processor_overrides(path, "post", "cpu")
        self.task_keys = {
            step.get("config", {}).get("task_key", "task")
            for step in steps
            if step.get("registry_name") in {"tokenizer_processor", "multi_task_dit_task_tokenizer"}
            or step.get("class", "").endswith(".TokenizerProcessorStep")
        }
        language = self.task_keys or (metadata.get("input_features") or {}).get("task", {}).get("type") == "LANGUAGE"
        if language and not task.strip():
            raise ValueError("Checkpoint needs a language instruction; set task.instruction in the config")

        if not any(path.glob("model*.safetensors")) and not (path / "adapter_model.safetensors").is_file():
            raise ValueError("No model safetensors found; export the trained weights alongside config.json")
        config = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device", device])
        # Let PyTorch place the weights and apply the checkpoint's own precision/model rules.
        self.pol = get_policy_class(self.kind).from_pretrained(path, config=config).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.pol.config, pretrained_path=path, preprocessor_overrides=pre_overrides,
            postprocessor_overrides=post_overrides,
        )
        try:
            spec = self.pol.chunk_inference_spec()
        except (ValueError, NotImplementedError):
            spec = None
        if spec is not None and (not spec.current_observation_only or "chunk" not in spec.modes):
            spec = None
        self.stepwise = spec is None
        self.H = 1 if self.stepwise else spec.execution_steps
        self.reset()

    def reset(self):
        self.pol.reset()
        self.pre.reset()
        self.post.reset()

    def _observation(self, imgs, q):
        torch = self.torch
        obs = {key: [self.task] for key in self.task_keys | {"task"}}
        if self.has_state:
            obs[STATE] = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        for key, name in self.cameras.items():
            obs[key] = torch.from_numpy(imgs[name]).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        return obs

    def _result(self, action):
        shape = tuple(getattr(action, "shape", ()))
        valid = shape == (1, self.width) if self.stepwise else (
            len(shape) == 3 and shape[0] == 1 and shape[1] >= self.H and shape[2] == self.width
        )
        if not valid:
            raise ValueError(f"Unexpected {self.kind} action shape {shape}; "
                             f"expected {'[1, joints]' if self.stepwise else '[1, steps, joints]'} "
                             f"with {self.width} joints and at least {self.H} step(s)")
        out = action[0].float().cpu().numpy()
        out = out[None, :] if self.stepwise else out[:self.H]
        if not np.isfinite(out).all():
            raise ValueError("Policy returned non-finite joint commands")
        return out

    def __call__(self, imgs, q, *, stage=None, **inference_kwargs):
        def call(name, fn, *args, **kwargs):
            return fn(*args, **kwargs) if stage is None else stage.wrap(fn, name)(*args, **kwargs)

        with self.torch.inference_mode():
            predict = self.pol.select_action if self.stepwise else self.pol.predict_action_chunk
            obs = call("observation", self._observation, imgs, q)
            batch = call("preprocess", self.pre, obs)
            action = call("select_action" if self.stepwise else "predict_chunk", predict, batch, **inference_kwargs)
            action = call("postprocess", self.post, action)
            return call("joint_output", self._result, action)
