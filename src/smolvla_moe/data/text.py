from __future__ import annotations

import hashlib

import torch


class HashTokenizer:
    """Small deterministic tokenizer for tiny-backbone smoke runs.

    This is not used for production SmolVLM2 training. It keeps local smoke tests independent of HF downloads.
    """

    def __init__(self, vocab_size: int = 32768, max_length: int = 64) -> None:
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)

    def __call__(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(len(texts), self.max_length, dtype=torch.long)
        mask = torch.zeros(len(texts), self.max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = text.lower().strip().split()[: self.max_length]
            for col, token in enumerate(tokens):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
                ids[row, col] = int.from_bytes(digest, "little") % self.vocab_size
                mask[row, col] = 1
            if not tokens:
                mask[row, 0] = 1
        return ids, mask
