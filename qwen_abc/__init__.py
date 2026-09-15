"""Lyrics-aware ABC lead sheets for Qwen finetuning.

Pipeline: SheetSage-Pro leadsheet.json -> canonical ``Song`` -> ABC text,
and back: ABC text -> ``Song`` -> MIDI.
"""

SCHEMA_VERSION = "qwen_abc_leadsheet_v1"
