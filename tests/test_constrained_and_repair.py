"""The two target-side attacks on cramming: repaired training notes, and a decoding ban."""
import torch

from qwen_abc.abc import parse_abc, song_to_abc
from qwen_abc.canonical import Chord, Note, Section, Song, comparable
from qwen_abc.constrained import JOIN, NoSyllableJoins, joining_token_ids
from qwen_abc.prompt import song_to_spec
from qwen_abc.repair import check_repair, split_crammed_notes


def crammed_song(dur=8, n_bars=2, sections=None):
    notes = [Note(0, dur, 60, ("喜", "怒", "哀"), line=0), Note(16, 8, 62, ("乐",), line=0)]
    return Song("c", 4, 100, "C major", [4] * n_bars, sections or [Section("verse", 0, n_bars)],
                notes, [Chord(0, 16 * n_bars, "C")])


# --- repaired targets -------------------------------------------------------

def test_a_crammed_note_becomes_one_note_per_syllable():
    song = crammed_song()
    fixed, st = split_crammed_notes(song)
    assert [n.lyric for n in fixed.notes] == [("喜",), ("怒",), ("哀",), ("乐",)]
    assert [(n.onset, n.duration) for n in fixed.notes[:3]] == [(0, 2), (2, 2), (4, 4)]
    assert st["crammed_notes"] == 1 and st["notes_added"] == 2 and st["notes_still_crammed"] == 0
    assert all(check_repair(song, fixed, st).values())


def test_the_split_pieces_keep_the_pitch_and_the_span():
    song = crammed_song()
    fixed, _ = split_crammed_notes(song)
    assert {n.pitch for n in fixed.notes[:3]} == {60}
    assert fixed.notes[2].offset == song.notes[0].offset


def test_a_note_too_short_for_its_syllables_keeps_the_remainder():
    song = crammed_song(dur=2)  # two ticks, three syllables
    fixed, st = split_crammed_notes(song)
    assert [n.lyric for n in fixed.notes] == [("喜",), ("怒", "哀"), ("乐",)]
    assert st["notes_still_crammed"] == 1 and st["syllables_left_joined"] == 1
    assert all(check_repair(song, fixed, st).values())


def test_a_note_spanning_a_section_boundary_is_left_alone():
    # the note runs from bar 0 into bar 1, where a new section starts
    song = crammed_song(dur=20, sections=[Section("verse", 0, 1), Section("chorus", 1, 1)])
    fixed, st = split_crammed_notes(song)
    assert st["crosses_a_section_boundary"] == 1 and st["notes_added"] == 0
    assert [n.lyric for n in fixed.notes][0] == ("喜", "怒", "哀")


def test_the_repair_leaves_the_prompt_untouched():
    song = crammed_song()
    fixed, _ = split_crammed_notes(song)
    assert song_to_spec(fixed) == song_to_spec(song)


def test_the_repaired_song_still_round_trips_through_abc():
    song = crammed_song()
    fixed, _ = split_crammed_notes(song)
    res = parse_abc(song_to_abc(fixed))
    assert res.ok and comparable(res.song) == comparable(fixed)
    assert JOIN not in song_to_abc(fixed)


# --- decoding ban -----------------------------------------------------------

class FakeTok:
    """Minimal tokenizer: one id per character, plus a '~'-joined merge token."""

    def __init__(self):
        self.pieces = ["w", ":", " ", "我", "爱", "\n", "C", JOIN, f"{JOIN}你"]

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)

    def __call__(self, text, **kw):
        return {"input_ids": [self.pieces.index(ch) for ch in text]}


def test_every_token_carrying_the_join_is_banned_including_merges():
    tok = FakeTok()
    assert sorted(joining_token_ids(tok)) == [7, 8]


def test_the_ban_applies_inside_a_lyric_line_only():
    tok = FakeTok()
    banned = joining_token_ids(tok)
    proc = NoSyllableJoins(tok, prompt_len=0, batch_size=1)
    ids = torch.tensor([tok("w: 我")["input_ids"]])
    out = proc(ids, torch.zeros(1, len(tok)))
    assert all(out[0, b] == float("-inf") for b in banned)
    assert out[0, 3] == 0.0  # an ordinary syllable is untouched

    music = NoSyllableJoins(tok, prompt_len=0, batch_size=1)
    out2 = music(torch.tensor([tok("w: 我\nC")["input_ids"]]), torch.zeros(1, len(tok)))
    assert all(out2[0, b] == 0.0 for b in banned)  # '~' is an ornament in a music line


def test_the_line_state_survives_across_steps():
    tok = FakeTok()
    proc = NoSyllableJoins(tok, prompt_len=0, batch_size=1)
    seq = tok("w: 我")["input_ids"]
    proc(torch.tensor([seq]), torch.zeros(1, len(tok)))          # step 1: inside w:
    seq2 = seq + tok("\n")["input_ids"]
    out = proc(torch.tensor([seq2]), torch.zeros(1, len(tok)))   # step 2: newline ends it
    assert all(out[0, b] == 0.0 for b in joining_token_ids(tok))
    assert proc.stats()["steps_constrained"] == 1
