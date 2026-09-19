"""Decoding constraints: forbid the one thing the model cannot be talked out of.

A crammed note — several syllables sung on one note — is written in exactly one
way in this ABC dialect: ``~`` joins them inside a ``w:`` line (``abc.py`` writes
``"~".join(note.lyric)``). So the pathology has a single token, and banning it
while a ``w:`` line is being generated removes it by construction.

R3-A showed the model will not stop on request, and best-of-n reranking bottoms
out at 0.086 against a corpus 0.057, so this is the remaining decoding-side
lever. What it cannot do is decide where the syllable goes instead: the model
must either write a note for it or leave it out, and the second is a lyric-recall
loss. That trade is the measurement, not an implementation detail.

``~`` is also an ornament mark in standard ABC, so the ban applies only inside a
``w:`` line — a music line keeps its full vocabulary.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import LogitsProcessor

W_LINE_PREFIX = "w:"
JOIN = "~"


def joining_token_ids(tok) -> List[int]:
    """Every token whose text contains ``~``; the ban has to cover merges too."""
    ids = []
    for i, piece in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))):
        if piece and JOIN in piece:
            ids.append(i)
    return ids


class NoSyllableJoins(LogitsProcessor):
    """Forbid ``~`` while the current line is a lyric line.

    The current line is tracked incrementally — one token decoded per step per
    row — because decoding the whole suffix each step would be quadratic.
    """

    def __init__(self, tok, prompt_len: int, batch_size: int):
        self.tok = tok
        self.prompt_len = prompt_len
        self.banned = torch.tensor(joining_token_ids(tok), dtype=torch.long)
        self.line: List[str] = [""] * batch_size
        self.seen: List[int] = [prompt_len] * batch_size
        self.blocked = 0

    def _advance(self, row: int, ids: torch.Tensor) -> None:
        start = self.seen[row]
        if ids.shape[0] <= start:
            return
        text = self.tok.decode(ids[start:].tolist(), skip_special_tokens=True)
        self.seen[row] = ids.shape[0]
        for ch in text:
            if ch == "\n":
                self.line[row] = ""
            else:
                self.line[row] += ch

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.banned.numel() == 0:
            return scores
        banned = self.banned.to(scores.device)
        for row in range(input_ids.shape[0]):
            if row >= len(self.line):  # a batch larger than declared: fail open, never silently wrong
                continue
            self._advance(row, input_ids[row])
            if self.line[row].lstrip().startswith(W_LINE_PREFIX):
                scores[row, banned] = float("-inf")
                self.blocked += 1
        return scores

    def stats(self) -> Dict[str, int]:
        return {"steps_constrained": self.blocked, "banned_token_ids": int(self.banned.numel())}
