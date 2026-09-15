"""Song-level splits and cross-split leakage control.

Base split: inherited verbatim from MIDI-LLM ``phoneme_leadsheet_section_v4``
(``{train,validation,test}_songs.jsonl``), i.e. sha256 of the
``musical_semantic_hash`` with version groups kept together. Inheriting keeps
the held-out songs identical in spirit to the MIDI-LLM experiments.

That split alone leaks: MIDI-LLM's own one-stage builder found 282/581 test
and 263/591 validation songs whose normalized lyrics exactly equal a train
song (re-uploads / alternate versions under a different song id). We remove
held-out songs, never train songs, that near-duplicate anything in train:

* lyric shingles: 6-syllable n-grams; containment = |H ∩ T| / |H|
* melody shingles: 8-note (interval, IOI-ticks) n-grams, key/tempo invariant

and validation songs that near-duplicate a test song (test keeps priority).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .canonical import Song

SPLITS = ("train", "validation", "test")


def load_inherited_split(section_manifest_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in SPLITS:
        with open(section_manifest_dir / f"{split}_songs.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row["song_id"] in out and out[row["song_id"]] != split:
                        raise ValueError(f"song {row['song_id']} in two inherited splits")
                    out[row["song_id"]] = split
    return out


def _h(items: Sequence) -> int:
    return int.from_bytes(hashlib.blake2b(repr(tuple(items)).encode("utf-8"), digest_size=8).digest(), "big")


def lyric_shingles(song: Song, n: int = 6) -> Set[int]:
    sylls = [s for note in song.notes if note.lyric for s in note.lyric]
    return {_h(sylls[i:i + n]) for i in range(max(len(sylls) - n + 1, 0))}


def melody_shingles(song: Song, n: int = 8) -> Set[int]:
    notes = song.notes
    feats = [(notes[i + 1].pitch - notes[i].pitch, notes[i + 1].onset - notes[i].onset)
             for i in range(len(notes) - 1)]
    return {_h(feats[i:i + n]) for i in range(max(len(feats) - n + 1, 0))}


def lyrics_exact_hash(song: Song) -> str:
    sylls = [s for note in song.notes if note.lyric for s in note.lyric]
    return hashlib.sha256("".join(sylls).encode("utf-8")).hexdigest()


class ShingleIndex:
    def __init__(self) -> None:
        self.postings: Dict[int, List[str]] = defaultdict(list)

    def add(self, song_id: str, shingles: Iterable[int]) -> None:
        for s in shingles:
            self.postings[s].append(song_id)

    def best_match(self, shingles: Set[int], max_df: int = 100) -> Tuple[Optional[str], float]:
        """Best containment |H ∩ T| / |H|. Shingles shared by more than
        ``max_df`` songs (repeated-note figures, stock phrases) are not
        evidence of duplication and are not counted, but stay in |H|."""
        if not shingles:
            return None, 0.0
        hits: Counter = Counter()
        for s in shingles:
            posting = self.postings.get(s, ())
            if len(posting) > max_df:
                continue
            for sid in posting:
                hits[sid] += 1
        if not hits:
            return None, 0.0
        sid, count = hits.most_common(1)[0]
        return sid, count / len(shingles)


def resolve_leakage(
    songs: Dict[str, Dict],
    lyric_threshold: float,
    melody_threshold: float,
) -> Tuple[Dict[str, str], List[Dict]]:
    """songs: song_id -> {"split", "lyric_sh", "melody_sh", "lyrics_hash"}.

    Returns (final split per kept song, exclusion rows). Deterministic: songs
    are visited in sorted id order and indices are built per protected split.
    """
    final: Dict[str, str] = {}
    exclusions: List[Dict] = []
    protected = {"train": ["train"], "test": ["train"], "validation": ["train", "test"]}
    indices: Dict[str, Tuple[ShingleIndex, ShingleIndex, Dict[str, str]]] = {}
    for split in ("train", "test"):
        li, mi, exact = ShingleIndex(), ShingleIndex(), {}
        for sid in sorted(songs):
            row = songs[sid]
            if row["split"] != split:
                continue
            li.add(sid, row["lyric_sh"])
            mi.add(sid, row["melody_sh"])
            exact.setdefault(row["lyrics_hash"], sid)
        indices[split] = (li, mi, exact)

    for sid in sorted(songs):
        row = songs[sid]
        split = row["split"]
        if split == "train":
            final[sid] = split
            continue
        reason = None
        for other in protected[split]:
            li, mi, exact = indices[other]
            # a test song may have been excluded already; validation is still
            # compared against every test song, which is the conservative choice
            match = exact.get(row["lyrics_hash"])
            if match is not None and row["lyric_sh"]:
                reason = {"reason": f"lyrics_exact_duplicate_of_{other}", "match": match, "score": 1.0}
                break
            m_id, m_score = li.best_match(row["lyric_sh"])
            if m_score >= lyric_threshold:
                reason = {"reason": f"lyrics_near_duplicate_of_{other}", "match": m_id, "score": round(m_score, 4)}
                break
            m_id, m_score = mi.best_match(row["melody_sh"])
            if m_score >= melody_threshold:
                reason = {"reason": f"melody_near_duplicate_of_{other}", "match": m_id, "score": round(m_score, 4)}
                break
        if reason is None:
            final[sid] = split
        else:
            exclusions.append({"song_id": sid, "split": split, **reason})
    return final, exclusions


def containment_profile(songs: Dict[str, Dict], split: str, against: str) -> Dict[str, List[float]]:
    """Best-match containment scores of every ``split`` song against ``against``."""
    li, mi = ShingleIndex(), ShingleIndex()
    for sid in sorted(songs):
        if songs[sid]["split"] == against:
            li.add(sid, songs[sid]["lyric_sh"])
            mi.add(sid, songs[sid]["melody_sh"])
    out = {"lyrics": [], "melody": []}
    for sid in sorted(songs):
        if songs[sid]["split"] == split:
            out["lyrics"].append(li.best_match(songs[sid]["lyric_sh"])[1])
            out["melody"].append(mi.best_match(songs[sid]["melody_sh"])[1])
    return out
