"""Keys, pitch spelling and chord symbols."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

LETTERS = "CDEFGAB"
NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

_NAME_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}
# Preferred spelling for each tonic pitch class (fewest accidentals; ties go
# to the spelling pop charts use).
MAJOR_TONIC = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
MINOR_TONIC = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "Bb", 11: "B"}
# Sharps (+) / flats (-) in the signature of each major key.
MAJOR_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
                "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}


def pitch_class(name: str) -> int:
    return _NAME_PC[name]


# ------------------------------------------------------------------ keys
def parse_key_name(key: Optional[str]) -> Optional[Tuple[int, str]]:
    """'B major' / 'G# minor' / 'Bm' / 'Ebmin' -> (tonic_pc, 'major'|'minor')."""
    if not key:
        return None
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m|)\s*", key.strip())
    if not m:
        return None
    name = m.group(1).upper() + m.group(2)
    mode = "minor" if m.group(3) in ("minor", "min", "m") else "major"
    return _NAME_PC[name], mode


def canonical_key(key: Optional[str]) -> Optional[str]:
    """Normalize to a preferred spelling, e.g. 'C# major' -> 'Db major'."""
    parsed = parse_key_name(key)
    if parsed is None:
        return None
    pc, mode = parsed
    tonic = (MAJOR_TONIC if mode == "major" else MINOR_TONIC)[pc]
    return f"{tonic} {mode}"


def key_to_abc(key: Optional[str]) -> str:
    parsed = parse_key_name(key)
    if parsed is None:
        return "C"
    pc, mode = parsed
    if mode == "major":
        return MAJOR_TONIC[pc]
    return MINOR_TONIC[pc] + "m"


def abc_to_key(field: str) -> Optional[str]:
    """Parse the body of a K: field ('Bb', 'F#m', 'Amin', 'C major')."""
    text = field.strip().split("%")[0].strip()
    m = re.match(r"([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m|ion|aeo)?", text)
    if not m:
        return None
    name = m.group(1).upper() + m.group(2)
    if name not in _NAME_PC:
        return None
    mode = "minor" if (m.group(3) or "") in ("minor", "min", "m", "aeo") else "major"
    return f"{name} {mode}"


def key_fifths(key: Optional[str]) -> int:
    """Signature size of a key name, spelled as given (not re-normalized)."""
    parsed = parse_key_name(key)
    if parsed is None:
        return 0
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?).*", key.strip())
    tonic = m.group(1).upper() + m.group(2)
    pc, mode = parsed
    if mode == "minor":
        # relative major, spelled a minor third up by letter
        idx = LETTERS.index(tonic[0])
        rel_letter = LETTERS[(idx + 2) % 7]
        rel_pc = (pc + 3) % 12
        diff = (rel_pc - NATURAL_PC[rel_letter] + 6) % 12 - 6
        rel = rel_letter + {0: "", 1: "#", -1: "b"}.get(diff, "?")
        return MAJOR_FIFTHS.get(rel, 0)
    return MAJOR_FIFTHS.get(tonic, 0)


def key_signature(key: Optional[str]) -> Dict[str, int]:
    """Letter -> alteration implied by the key signature."""
    n = key_fifths(key)
    sig = {letter: 0 for letter in LETTERS}
    if n > 0:
        for letter in SHARP_ORDER[:n]:
            sig[letter] = 1
    elif n < 0:
        for letter in FLAT_ORDER[: -n]:
            sig[letter] = -1
    return sig


def spell_pitch(midi: int, key: Optional[str]) -> Tuple[str, int, int]:
    """Choose (letter, alteration, octave) for a MIDI pitch in a key.

    Diatonic pitches take the key's spelling. Others take a natural when one
    exists, otherwise a sharp in sharp/neutral keys and a flat in flat keys.
    """
    sig = key_signature(key)
    pc = midi % 12
    candidates: List[Tuple[str, int]] = []
    for letter in LETTERS:
        alter = (pc - NATURAL_PC[letter] + 6) % 12 - 6
        if abs(alter) <= 1:
            candidates.append((letter, alter))
    diatonic = [(l, a) for l, a in candidates if sig[l] == a]
    if diatonic:
        letter, alter = diatonic[0]
    else:
        naturals = [(l, a) for l, a in candidates if a == 0]
        if naturals:
            letter, alter = naturals[0]
        else:
            want = -1 if key_fifths(key) < 0 else 1
            letter, alter = [(l, a) for l, a in candidates if a == want][0]
    octave = (midi - alter - NATURAL_PC[letter]) // 12 - 1
    return letter, alter, octave


def abc_note_name(letter: str, octave: int) -> str:
    """Letter + octave -> ABC (C4 = 'C', C5 = 'c', C6 = "c'", C3 = 'C,')."""
    if octave >= 5:
        return letter.lower() + "'" * (octave - 5)
    return letter.upper() + "," * (4 - octave)


# ---------------------------------------------------------------- chords
# LVCR/Harte quality -> ABC suffix
QUALITY_TO_SUFFIX = {
    "maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "aug": "aug",
    "sus4": "sus4", "sus2": "sus2", "dim7": "dim7", "hdim7": "m7b5", "maj6": "6", "min6": "m6",
    "9": "9", "maj9": "maj9", "min9": "m9", "minmaj7": "mM7", "7sus4": "7sus4", "5": "5",
}
SUFFIX_INTERVALS = {
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6), "aug": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7), "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "9": (0, 4, 7, 10, 2),
    "maj9": (0, 4, 7, 11, 2), "m9": (0, 3, 7, 10, 2), "mM7": (0, 3, 7, 11), "7sus4": (0, 5, 7, 10),
    "5": (0, 7),
}
_DEGREE_SEMITONES = {"1": 0, "b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7,
                     "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11, "b9": 1, "9": 2}
NO_CHORD = "N.C."
_SUFFIX_RE = "|".join(sorted((re.escape(s) for s in SUFFIX_INTERVALS if s), key=len, reverse=True))
CHORD_RE = re.compile(rf"^([A-G][#b]?)({_SUFFIX_RE})?(?:/([A-G][#b]?))?$")


def spell_pc_like(pc: int, root: str) -> str:
    """Spell a pitch class with sharps or flats following the root's flavour."""
    flat = "b" in root
    names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"] if flat else \
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[pc % 12]


def corpus_chord_to_abc(symbol: str, root: Optional[str], quality: Optional[str],
                        bass: Optional[str]) -> Tuple[str, bool]:
    """Corpus chord row -> (ABC symbol, recognized)."""
    if symbol in ("N", "X") or quality in (None, "N", "X") or root is None:
        return NO_CHORD, symbol in ("N",) or quality == "N"
    suffix = QUALITY_TO_SUFFIX.get(quality)
    known = suffix is not None
    if suffix is None:
        suffix = "m" if quality.startswith("min") else ""
    text = root + suffix
    if bass and bass != "1":
        semis = _DEGREE_SEMITONES.get(bass)
        if semis is None:
            known = False
        else:
            text += "/" + spell_pc_like(pitch_class(root) + semis, root)
    return text, known


def parse_chord_symbol(text: str) -> Optional[Dict[str, object]]:
    """ABC chord symbol -> {root_pc, suffix, bass_pc, pcs}; None if invalid."""
    text = text.strip()
    if text in (NO_CHORD, "NC", "N.C", "N"):
        return {"root_pc": None, "suffix": None, "bass_pc": None, "pcs": ()}
    m = CHORD_RE.match(text)
    if not m or m.group(1) not in _NAME_PC:
        return None
    root_pc = _NAME_PC[m.group(1)]
    suffix = m.group(2) or ""
    bass = m.group(3)
    if bass is not None and bass not in _NAME_PC:
        return None
    pcs = tuple(sorted({(root_pc + i) % 12 for i in SUFFIX_INTERVALS[suffix]}))
    return {"root_pc": root_pc, "suffix": suffix, "bass_pc": _NAME_PC[bass] if bass else None, "pcs": pcs}


def key_pc_names(key: Optional[str]) -> List[str]:
    """Pitch-class names in the accidental flavour of a key (for chord roots)."""
    fifths = key_fifths(key)
    if fifths > 0:
        return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if fifths < 0:
        return ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def respell_chord_symbol(text: str, key: Optional[str]) -> str:
    return transpose_chord_symbol(text, 0, key)


def transpose_chord_symbol(text: str, semitones: int, key: Optional[str]) -> str:
    info = parse_chord_symbol(text)
    if info is None or info["root_pc"] is None:
        return text
    names = key_pc_names(key)
    out = names[(info["root_pc"] + semitones) % 12] + info["suffix"]
    if info["bass_pc"] is not None:
        out += "/" + names[(info["bass_pc"] + semitones) % 12]
    return out
