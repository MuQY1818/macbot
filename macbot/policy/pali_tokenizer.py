"""Small PaliGemma tokenizer adapter for checkpoints without bundled tokenizer files."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class PaliGemmaTokenizer:
    """SentencePiece-backed tokenizer with the PaliGemma special-token contract."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    vocab_size = 257152

    def __init__(self, model_file: str | Path):
        import sentencepiece as spm

        self.model_file = Path(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_file))
        if self.sp.vocab_size() != self.vocab_size:
            raise ValueError(f"Expected a PaliGemma vocabulary of {self.vocab_size}, got {self.sp.vocab_size()}")

    def encode(self, text, *, add_special_tokens=True, add_bos_token=True, add_eos_token=False):
        values = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            if add_bos_token:
                values.insert(0, self.bos_token_id)
            if add_eos_token:
                values.append(self.eos_token_id)
        return values

    def __call__(
        self,
        text,
        *,
        max_length=512,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors=None,
        **_,
    ):
        import torch

        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode(item) for item in texts]
        if truncation:
            rows = [row[:max_length] for row in rows]
        if padding in {"max_length", True}:
            for row in rows:
                if len(row) > max_length:
                    raise ValueError("Tokenizer row exceeds max_length after truncation")
            padded = []
            masks = []
            for row in rows:
                pad = [self.pad_token_id] * (max_length - len(row))
                mask = [1] * len(row) + [0] * len(pad)
                if padding_side == "left":
                    row, mask = pad + row, [0] * len(pad) + [1] * len(row)
                padded.append(row)
                masks.append(mask)
        else:
            width = max(map(len, rows), default=0)
            padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
            masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        result = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
        if return_tensors not in (None, "pt"):
            raise ValueError("PaliGemmaTokenizer only supports return_tensors='pt'")
        return result

    def convert_ids_to_tokens(self, ids):
        result = []
        for value in ids:
            value = int(value)
            if value == self.pad_token_id:
                result.append("<pad>")
            elif value == self.eos_token_id:
                result.append("<eos>")
            elif value == self.bos_token_id:
                result.append("<bos>")
            else:
                result.append(self.sp.id_to_piece(value))
        return result

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        special = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
        return [special.get(token, self.sp.piece_to_id(token)) for token in tokens]

    def token_ids(self, text):
        return np.asarray(self.encode(text), dtype=np.int32)


def resolve_pali_tokenizer(path: Path | None = None) -> PaliGemmaTokenizer:
    """Load a local tokenizer or the public ungated PaliGemma tokenizer mirror."""
    candidates = []
    if path is not None:
        if path.is_file():
            candidates.append(path)
        else:
            candidates.extend((path / "tokenizer.model", path / "tokenizer" / "tokenizer.model"))
    if env_path := os.environ.get("MACBOT_PALIGEMMA_TOKENIZER"):
        candidates.append(Path(env_path).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return PaliGemmaTokenizer(candidate)

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id="leo009/paligemma_tokenizer.model",
        filename="paligemma_tokenizer.model",
    )
    return PaliGemmaTokenizer(downloaded)
