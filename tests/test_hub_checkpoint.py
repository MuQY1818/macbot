from pathlib import Path
from types import SimpleNamespace

import pytest

from macbot.cli import main
from macbot.hub import HubRepo, resolve_policy


@pytest.fixture
def hub(monkeypatch, tmp_path):
    prefix = "pretrained_model/"
    path = tmp_path / "snapshots" / "deadbeef" / prefix
    path.mkdir(parents=True)
    files = {"config.json": "{}", "policy_preprocessor.json": '{"steps":[{"state_file":"norm.safetensors"}]}',
             "policy_postprocessor.json": '{"steps":[]}', "norm.safetensors": "", "model.safetensors": ""}
    for file, contents in files.items():
        (path / file).write_text(contents)
    calls = []

    def open_repo(reference, kind):
        assert reference == ("owner/policy", "release/1") and kind == "model"
        return HubRepo(*reference, kind, "deadbeef", frozenset(prefix + f for f in files))

    def download(self, file):
        calls.append((self.sha, file))
        return path.parent / file

    monkeypatch.setattr(HubRepo, "open", open_repo)
    monkeypatch.setattr(HubRepo, "download", download)
    return path, files, calls


@pytest.mark.parametrize("command", ["run", "bench"])
def test_checkpoint_cli_passes_cached_path_and_commit(monkeypatch, capsys, hub, command):
    import sys

    path, files, calls = hub
    module = "runner" if command == "run" else "bench"

    def run(cfg, args):
        assert args.policy == path
        assert args.policy_hub == {"repo_id": "owner/policy", "revision": "deadbeef"}

    monkeypatch.setattr("macbot.config.setup_path", lambda: path / "absent.yaml")
    monkeypatch.setitem(sys.modules, f"macbot.policy.{module}", SimpleNamespace(run=run))
    main([command, "--policy", "owner/policy@release/1", *(["--fake"] if command == "run" else [])])
    assert "owner/policy@deadbeef" in capsys.readouterr().out
    assert set(calls) == {("deadbeef", f"pretrained_model/{name}") for name in files}


def test_checkpoint_errors_before_loading_weights(hub):
    _, files, calls = hub
    files.pop("norm.safetensors")
    with pytest.raises(ValueError, match="missing processor state"):
        resolve_policy("owner/policy@release/1")
    assert not any(f.endswith("model.safetensors") for _, f in calls)
    files.pop("policy_postprocessor.json")
    with pytest.raises(ValueError, match="missing saved processors"):
        resolve_policy("owner/policy@release/1")


def test_existing_two_part_path_wins_over_hub(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("owner/policy")
    path.mkdir(parents=True)
    assert resolve_policy(str(path)) == (path, None)


def test_access_error(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    from httpx import Request, Response
    from huggingface_hub.errors import RepositoryNotFoundError

    def denied(*args, **kwargs):
        raise RepositoryNotFoundError("Unauthorized", response=Response(
            401, request=Request("GET", "https://huggingface.co/api/models/owner/private")))

    monkeypatch.setattr(hub.HfApi, "repo_info", denied)
    with pytest.raises(ValueError, match="missing, private or gated"):
        resolve_policy("owner/private")
