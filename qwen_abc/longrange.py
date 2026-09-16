"""Long-range tasks on ABC-v2 songs (experiment E3 and continuation evaluation).

* **Masked late-section reconstruction (training + evaluation).** The prompt
  holds the full plan and lyrics and the whole song in ABC-v2 with one
  later-half lyric section replaced by a gap; the completion is exactly that
  section. The model must use context on both sides: earlier occurrences of the
  same label (motif identity), the later sections, and the plan. This is not
  available to whole-song SFT, which only ever conditions on the past.
* **Late-section continuation (evaluation only).** The prompt is the ordinary
  whole-song prompt followed by the reference ABC of the first k sections; the
  model continues. Under teacher forcing this is already part of whole-song SFT,
  so it is used to *evaluate* every model (E0/E1/E2/E3) free-running from a
  correct prefix, not as extra training data.

Every choice is deterministic (hash of the song id).
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

from .abc_v2 import PROMPT_HEADER_V2, song_to_abc_v2, spec_to_prompt_v2
from .canonical import Song
from .prompt import COMPLETION_MARKER, split_syllables

INFILL_HEADER = "Task: one section of the lead sheet below is missing. Write only the missing section in ABC notation."
GAP_LINE = "% [missing section]"
INFILL_MARKER = "Missing section:\n"


def split_abc_sections(abc: str) -> Tuple[str, List[str]]:
    """ABC text -> (header text, [section block text]) where each block starts with its P: line."""
    lines = abc.split("\n")
    header, blocks, cur = [], [], None
    for line in lines:
        if line.startswith("P:"):
            if cur is not None:
                blocks.append("\n".join(cur) + "\n")
            cur = [line]
        elif cur is None:
            header.append(line)
        else:
            cur.append(line)
    if cur is not None:
        text = "\n".join(cur)
        blocks.append(text if text.endswith("\n") else text + "\n")
    # the writer ends with "\n": the last block keeps exactly one trailing newline
    blocks = [b.rstrip("\n") + "\n" for b in blocks]
    return "\n".join(header) + "\n", blocks


def _h(song_id: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{song_id}".encode()).digest()[:8], "big")


def infill_candidates(spec: Dict) -> List[int]:
    """Later-half lyric sections with >= 2 bars; sections whose label occurred earlier come first."""
    n = len(spec["sections"])
    start = math.ceil(n / 2)
    cands = [i for i in range(start, n)
             if spec["sections"][i]["bars"] >= 2 and any(split_syllables(l) for l in spec["sections"][i]["lines"])]
    repeated = [i for i in cands if any(spec["sections"][j]["label"] == spec["sections"][i]["label"] for j in range(i))]
    return repeated or cands


def choose_infill_target(song_id: str, spec: Dict) -> Optional[int]:
    cands = infill_candidates(spec)
    if not cands:
        return None
    return cands[_h(song_id, "infill") % len(cands)]


def infill_example(song: Song, spec: Dict, target: int) -> Dict[str, str]:
    abc = song_to_abc_v2(song)
    header, blocks = split_abc_sections(abc)
    tgt = blocks[target]
    head_lines = tgt.split("\n")[:2]  # "P:label" and "% section i/N | B bars"
    gapped = header + "".join(blocks[:target]) + "\n".join(head_lines) + "\n" + GAP_LINE + "\n" + "".join(blocks[target + 1:])
    base = spec_to_prompt_v2(spec)
    assert base.startswith(PROMPT_HEADER_V2) and base.endswith(COMPLETION_MARKER)
    body = base[len(PROMPT_HEADER_V2): -len(COMPLETION_MARKER)]
    prompt = INFILL_HEADER + body + "ABC with a gap:\n" + gapped + "\n" + INFILL_MARKER
    return {"prompt": prompt, "completion": tgt, "target_section": target}


def continuation_prompt(song: Song, spec: Dict, k: int) -> Dict[str, str]:
    """Whole-song prompt + reference ABC-v2 of sections [0, k); the model continues from section k."""
    abc = song_to_abc_v2(song)
    header, blocks = split_abc_sections(abc)
    prefix = header + "".join(blocks[:k])
    return {"prompt": spec_to_prompt_v2(spec) + prefix, "prefix": prefix, "reference_rest": "".join(blocks[k:]),
            "start_section": k}


def continuation_start(spec: Dict) -> int:
    """Continue from the first section of the last third (at least one section is given, one is left)."""
    n = len(spec["sections"])
    return min(max(math.ceil(2 * n / 3), 1), n - 1)
