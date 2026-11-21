"""Resolve explicit Hub inputs to immutable files in the standard Hub cache."""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .system import expand

HUB_ID = re.compile(r"(?P<repo>[\w.-]+/[\w.-]+)(?:@(?P<revision>[^\s@]+))?", re.ASCII)


def hub_reference(value):
    match = HUB_ID.fullmatch(str(value))
    if match and all(part not in (".", "..") for part in match["repo"].split("/")):
        return match["repo"], match["revision"]
    return None


@dataclass(frozen=True)
class HubRepo:
    id: str
    revision: str | None
    kind: str
    sha: str
    files: frozenset[str]

    @classmethod
    def open(cls, reference, kind):
        from huggingface_hub import HfApi

        repo, revision = reference
        with hub_errors(repo, revision, kind):
            info = HfApi().repo_info(repo, revision=revision, repo_type=kind)
        if not info.sha:
            raise ValueError(f"Hub {kind} {repo} did not return a resolved commit.")
        return cls(repo, revision, kind, info.sha, frozenset(file.rfilename for file in info.siblings or []))

    def download(self, filename):
        from huggingface_hub import hf_hub_download

        with hub_errors(self.id, self.sha, self.kind):
            return Path(hf_hub_download(self.id, filename, revision=self.sha, repo_type=self.kind))

    @property
    def pinned(self):
        return f"{self.id}@{self.sha}"


@contextmanager
def hub_errors(repo, revision, kind):
    from httpx import TransportError
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    label = f"Hub {kind} {repo}" + (f"@{revision}" if revision else "")
    try:
        yield
    except RevisionNotFoundError:
        raise ValueError(f"{label}: revision not found.") from None
    except (GatedRepoError, RepositoryNotFoundError):
        raise ValueError(f"{label}: repository is missing, private or gated. Check the ID and Hub login/access.") from None
    except HfHubHTTPError as exc:
        raise ValueError(f"{label}: Hub request failed (HTTP {exc.response.status_code}).") from None
    except (LocalEntryNotFoundError, OfflineModeIsEnabled, TransportError):
        raise ValueError(f"{label}: could not reach the Hub or find the requested file in cache.") from None
    except HFValidationError as exc:
        raise ValueError(f"{label}: {exc}") from None


PROCESSORS = ("policy_preprocessor.json", "policy_postprocessor.json")
CHECKPOINT_FILES = {"config.json", "model.safetensors", "train_config.json", "gripper.json", "camera_framing.json",
                    *PROCESSORS}


def resolve_policy(value):
    path = expand(value)
    reference = hub_reference(value)
    if path.exists() or reference is None:
        return path, None
    repo = HubRepo.open(reference, "model")
    if "config.json" in repo.files:
        prefix = ""
    elif "pretrained_model/config.json" in repo.files:
        prefix = "pretrained_model/"
    else:
        candidates = sorted(f.removesuffix("config.json") for f in repo.files if f.endswith("/pretrained_model/config.json"))
        if len(candidates) != 1:
            raise ValueError(f"Hub model {repo.pinned}: expected one LeRobot pretrained_model; "
                             "pass a local checkpoint directory to select one explicitly.")
        prefix = candidates[0]
    missing = [file for file in PROCESSORS if prefix + file not in repo.files]
    if missing:
        raise ValueError(f"Hub model {repo.pinned}: missing saved processors ({', '.join(missing)}). "
                         "Export or upload the complete LeRobot pretrained_model directory.")
    if prefix + "model.safetensors" not in repo.files:
        raise ValueError(f"Hub model {repo.pinned}: missing model.safetensors; use a complete pretrained_model.")
    state_files = set()
    for filename in PROCESSORS:
        processor = repo.download(prefix + filename)
        try:
            for step in json.loads(processor.read_text())["steps"]:
                if state := step.get("state_file"):
                    if not isinstance(state, str) or "/" in state or not state.endswith(".safetensors"):
                        raise ValueError("processor state must be a local safetensors filename")
                    if prefix + state not in repo.files:
                        raise ValueError(f"missing processor state {state}")
                    state_files.add(state)
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            raise ValueError(f"Hub model {repo.pinned}: invalid saved processor {filename}: {exc}.") from exc
    path = repo.download(prefix + "config.json").parent
    for name in sorted((CHECKPOINT_FILES - {"config.json", *PROCESSORS}) | state_files):
        if prefix + name in repo.files:
            repo.download(prefix + name)
    print(f"Hub checkpoint: {repo.pinned}", flush=True)
    return path, {"repo_id": repo.id, "revision": repo.sha}
