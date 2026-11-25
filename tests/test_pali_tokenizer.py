import sys
import types

import pytest

from macbot.policy import pali_tokenizer
from macbot.policy.pali_tokenizer import TOKENIZER_ENV, local_pali_tokenizer, resolve_pali_tokenizer


def test_no_local_tokenizer_without_files_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKENIZER_ENV, raising=False)
    assert local_pali_tokenizer(tmp_path) is None
    assert local_pali_tokenizer(None) is None


def test_local_tokenizer_comes_from_checkpoint_or_env(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(pali_tokenizer, "PaliGemmaTokenizer", lambda path: loaded.append(path) or path)
    monkeypatch.delenv(TOKENIZER_ENV, raising=False)
    (tmp_path / "tokenizer.model").write_bytes(b"")
    assert local_pali_tokenizer(tmp_path) == tmp_path / "tokenizer.model"
    elsewhere = tmp_path / "shared.model"
    elsewhere.write_bytes(b"")
    monkeypatch.setenv(TOKENIZER_ENV, str(elsewhere))
    assert local_pali_tokenizer(tmp_path / "missing") == elsewhere


def test_official_tokenizer_failure_names_the_override_and_never_uses_a_mirror(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKENIZER_ENV, raising=False)
    requested = []

    def gated(name, **_):
        requested.append(name)
        raise OSError("gated repository")

    def mirror(*_, **__):
        raise AssertionError("a substitute tokenizer must not be downloaded")

    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=gated)))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=mirror, snapshot_download=mirror))
    with pytest.raises(RuntimeError, match=TOKENIZER_ENV):
        resolve_pali_tokenizer(tmp_path, "google/paligemma-3b-pt-224")
    assert requested == ["google/paligemma-3b-pt-224"]
