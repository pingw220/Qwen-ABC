# ABC-v2 data cleaning: section boundaries vs lyric lines

Implementation: `qwen_abc/cleaning.py` (`CLEANING_VERSION = section_boundary_snap_v2`), applied by `scripts/build_abc_v2_dataset.py`.
Every changed song is logged in `data/generated/abc_v2_20260915_120927/cleaning_log.jsonl`, with old and new sections and every decision.
Numbers come from `build_report.json` and `validation.json`. Tests: `tests/test_cleaning.py`.

## Summary

* **What changes:** only `Section.start_bar` / `num_bars`. Every note, lyric, melisma, chord, bar length, tempo, key and onset is byte-identical to ABC-v1 (checked for all 10,741 songs). Musical time is preserved exactly.
* **Songs changed:** **3,798 / 10,243 train (37.1%)**, 104 / 273 validation (38.1%), 93 / 225 test (41.3%).
* **Lyric-line cuts at section boundaries** (definitions in §1):
  * train **17,851 → 12,303 (−31%)**; songs with any cut 7,056 → 5,242;
  * validation 577 → 419;
  * test 402 → 273.
* **Deliberately not fixed:** most remaining cuts are *balanced* splits, and 6,477 of them in train were left alone. Those are usually two corpus lyric lines merged into one, so they give no evidence where the boundary belongs.
* **No song was excluded.** The split and the training examples are the same songs as ABC-v1. Clear pathologies are *flagged* (§6) and used only for the clean evaluation subset.

## 1. Definitions

A corpus lyric line (`lines[]`, carried per note as `Note.line`) **crosses** a section boundary at barline T when it has sung notes (attacks or melisma notes) both before and after T.

| kind | rule | counted as a cut? |
|---|---|---|
| **pickup** | the notes before T all lie in the bar just before T, and carry fewer syllables than after T (≤2, or less than half of the after part) | no: an anacrusis is engraved before the rehearsal mark |
| **melisma spill** | only continuation notes of the line's last syllable cross T | no: no lyric text moves |
| **tail** | fewer syllables after T than before | yes |
| **head/mid** | any other crossing | yes |

## 2. Rule (per internal boundary, left to right)

1. No cut → keep.
2. A cut is *repairable* only if its misplaced part is small: at most 3 syllables and at most ⅓ of the line. Otherwise keep the boundary (`balanced_split_kept`).
3. The direction follows the cut: a tail spill moves the boundary **later**, a head cut **earlier**. Cuts disagreeing on direction → keep (`conflicting_directions`).
4. Try the nearest existing barline in that direction, up to **2 bars** away. Accept it only if all of the following hold; otherwise keep (`no_cut_free_candidate`):
   * every repaired line lies wholly on one side (turning a cut into a "pickup" is *not* a repair);
   * no other line is cut;
   * no line that was intact at the old barline crosses the new one;
   * no intact sung line lies wholly between the old and new barline, where it would silently change section;
   * both neighbouring sections keep ≥2 bars, or their original length if already shorter.

**How the rules were revised.** The first prototype had neither the direction rule nor the "wholly on one side" rule, and it made visibly wrong moves: it pushed a whole line into the next section, or turned a spill into a fake pickup. These were found by reading 30+ decisions on validation and test before building.

A first full build (`abc_v2_20260915_113908`, rule `section_boundary_snap_v1`) still lacked the last condition. A unit test then showed that a +2-bar move could carry an intact short line across the boundary: **568 of 6,403 moves (8.9%), in 454 songs**.

What was done:

* the rule was fixed (v2);
* the dataset was rebuilt (`abc_v2_20260915_120927`);
* the E1/E1c jobs started on the flawed build were cancelled after about 20 minutes;
* the flawed directory is marked `SUPERSEDED.txt`.

No result in any report uses it.

## 3. Decisions

| | train | validation | test |
|---|---|---|---|
| boundaries with a cut | 16,868 | 519 | 380 |
| moved | 5,548 | 158 | 129 |
| kept: balanced split | 6,477 | 197 | 148 |
| kept: no valid barline within 2 bars | 4,831 | 164 | 103 |
| kept: conflicting directions | 12 | 0 | 0 |

Shift distribution (all splits, moved boundaries):

| shift | −2 bars | −1 bar | +1 bar | +2 bars |
|---|---|---|---|---|
| boundaries | 521 | 184 | **4,830** | 300 |

Most repairs move a boundary **one bar later** to pull the last syllable of a line back into its section: `…你只是个小姑 ‖ 娘` → `…你只是个小姑娘 ‖`.

Other counts:

* head cuts (−1/−2) are rarer: 705 in total;
* pickups are unchanged (train 14,948 → 14,948);
* lyric-line cuts before → after: train 17,851 → 12,303; validation 577 → 419; test 402 → 273.

## 4. Section length distribution before / after

| split | sections | mean bars | median | p95 | ≤1 bar | ≤2 bars | >32 bars | multiple of 4 |
|---|---|---|---|---|---|---|---|---|
| train before | 104,927 | 8.87 | 8 | 16 | 1,318 | 2,556 | 187 | 49.5% |
| train after | 104,927 | 8.87 | 8 | 16 | 1,316 | 2,763 | 192 | 48.6% |
| validation before / after | 2,866 | 8.83 | 8 | 16 / 16 | 29 / 29 | 73 / 79 | 1 / 1 | 49.2% / 47.9% |
| test before / after | 2,382 | 8.97 | 8 | 16 / 16 | 27 / 26 | 51 / 58 | 1 / 1 | 49.4% / 48.1% |

Train histogram, bars per section (before → after): 2: 1,238 → 1,447, 3: 1,024 → 1,363, 4: 10,003 → 9,785, 5: 5,647 → 5,520, 7: 6,846 → 6,696, 8: 31,637 → 31,049, 9: 10,157 → 10,459, 10: 6,746 → 6,821.

**A caveat that matters for interpretation.** Cleaning moves a little mass *away* from 4- and 8-bar sections: the multiple-of-4 share drops from 49.5% to 48.6%, and there are 207 more 2-bar sections. The most common repair is a single final syllable sung on the downbeat of the next section, and that is sometimes a genuine musical convention: a phrase ending on beat 1 of the next hypermeasure, while the band's new section starts on that same downbeat. All-In-One places boundaries by audio structure; the corpus lyric lines follow the text. **The cleaning therefore enforces a lyric-centric convention; it is not a proven correction.** Whether it helps a model follow structure is measured, not assumed: E1c (cleaning only) vs E0 on the same prompts.

## 5. Before / after examples (24 moved boundaries)

A deterministic sample: 16 train and 8 test songs, the first by sha256 of the song id; the first moved boundary of each is shown. Cells show `label:bars` and at most 6 lyric characters at the boundary (`/` separates prompt lines, `∅` = no lyrics). Crossing kinds give syllables before|after the old barline.

| # | split | song | boundary (section index) | shift (bars) | crossing kinds at old barline | before: …end ‖ start… | after: …end ‖ start… |
|---|---|---|---|---|---|---|---|
| 1 | train | `0391WHhfoU` | 3 | -1 | head_or_mid 3|6 | chorus:8 …不回/我恨你 ‖ instrumental:17 我恨你徒伤悲… | chorus:7 …时光一去不回 ‖ instrumental:18 我恨你我恨你… |
| 2 | train | `3EePrrOqyw` | 5 | +1 | tail 6|1 | chorus:16 …我有多少的弱 ‖ instrumental:30 点… | chorus:17 …有多少的弱点 ‖ instrumental:29 ∅… |
| 3 | train | `768GNTDGSZ` | 3 | +1 | tail 9|1 | chorus:11 …你只是个小姑 ‖ instrumental:10 娘… | chorus:12 …只是个小姑娘 ‖ instrumental:9 ∅… |
| 4 | train | `3GoxOXrST6` | 3 | +1 | tail 2|1 | chorus:8 …孩子抱/不得 ‖ chorus:13 了/女人起来… | chorus:9 …子抱/不得了 ‖ chorus:12 女人起来了/… |
| 5 | train | `1Zk2w4iTQc` | 9 | +1 | tail 8|1 | chorus:24 …找到失落的过 ‖ outro:5 去… | chorus:25 …到失落的过去 ‖ outro:4 ∅… |
| 6 | train | `0T7HfYCMtp` | 5 | +1 | tail 10|1 | chorus:5 …人要骗就骗到 ‖ instrumental:10 底… | chorus:6 …要骗就骗到底 ‖ instrumental:9 ∅… |
| 7 | train | `2eVenc1JMY` | 4 | +1 | tail 7|1 | chorus:8 …好都忙着打电 ‖ instrumental:8 话… | chorus:9 …都忙着打电话 ‖ instrumental:7 ∅… |
| 8 | train | `2IrjilmT6F` | 3 | -2 | head_or_mid 2|7 | chorus:6 …已流尽/啊啊 ‖ chorus:7 只觉得天摇地… | chorus:4 …我的泪已流尽 ‖ chorus:9 啊啊只觉得天… |
| 9 | train | `2mb6xKdlHo` | 2 | +1 | tail 7|1 | verse:7 …陌千里策马昂 ‖ verse:5 扬/壮年听雨… | verse:8 …千里策马昂扬 ‖ verse:4 壮年听雨客舟… |
| 10 | train | `5pBY7KRY8Z` | 12 | +1 | tail 8|1 | chorus:6 …我繁星熄灭之 ‖ outro:5 前… | chorus:7 …繁星熄灭之前 ‖ outro:4 ∅… |
| 11 | train | `25zESKAPu2` | 6 | +1 | tail 3|1 | chorus:10 …和意/你在哪 ‖ instrumental:8 里… | chorus:11 …意/你在哪里 ‖ instrumental:7 ∅… |
| 12 | train | `2B4n1y7ZXG` | 9 | +1 | tail 12|1 | chorus:6 …伤就是我的愿 ‖ instrumental:8 望… | chorus:7 …就是我的愿望 ‖ instrumental:7 ∅… |
| 13 | train | `7kJgYN5fGp` | 10 | +1 | tail 4|1 | chorus:12 …糊/不能做自 ‖ outro:5 己… | chorus:13 …/不能做自己 ‖ outro:4 ∅… |
| 14 | train | `7EJmDZShZj` | 1 | -2 | head_or_mid 1|6 | intro:4 …黑 ‖ verse:5 夜你独自面对… | intro:2 …∅ ‖ verse:7 黑夜你独自面… |
| 15 | train | `1ZdS7FNlrC` | 8 | +1 | tail 7|1 | chorus:5 …受这慈悲的滋 ‖ outro:6 味… | chorus:6 …这慈悲的滋味 ‖ outro:5 ∅… |
| 16 | train | `7fHZNp86Ce` | 9 | +1 | tail 8|1 | verse:10 …清这是雨还是 ‖ instrumental:11 泪… | verse:11 …这是雨还是泪 ‖ instrumental:10 ∅… |
| 17 | test | `7qDiugtLVz` | 13 | +1 | tail 17|1 | chorus:8 …的翅膀为梦翱 ‖ chorus:10 翔/就在今天… | chorus:9 …翅膀为梦翱翔 ‖ chorus:9 就在今天为您… |
| 18 | test | `7fnhHkS67S` | 8 | +1 | tail 4|1 | verse:9 …绪/独自空叹 ‖ chorus:10 息/既然有缘… | verse:10 …/独自空叹息 ‖ chorus:10 既然有缘来相… |
| 19 | test | `6I3A7q18EJ` | 5 | +1 | tail 14|1 | chorus:8 …依恋万事都成 ‖ instrumental:8 空… | chorus:9 …恋万事都成空 ‖ instrumental:7 ∅… |
| 20 | test | `0FadVuelB7` | 2 | +1 | tail 7|1 | verse:10 …后还有谁会在 ‖ chorus:11 乎/谁在哭/… | verse:11 …还有谁会在乎 ‖ chorus:10 谁在哭/让我… |
| 21 | test | `6URmER9Cyp` | 11 | +1 | tail 9|1 | chorus:13 …真的忘了告诉 ‖ chorus:9 你/午夜的那… | chorus:14 …的忘了告诉你 ‖ chorus:8 午夜的那一… |
| 22 | test | `32Wt3NCmKI` | 12 | -2 | head_or_mid 1|3 | chorus:8 …恼感受/it ‖ bridge:8 s my w… | chorus:6 …抛开烦恼感受 ‖ bridge:10 it s m… |
| 23 | test | `0FdW541km7` | 2 | +1 | tail 5|1 | verse:23 …/让温度被炸 ‖ chorus:17 开/别挣扎快… | verse:24 …让温度被炸开 ‖ chorus:16 别挣扎快去爱… |
| 24 | test | `53a8hVCKsw` | 11 | +1 | tail 8|1 | verse:7 …温柔的心口靠 ‖ outro:4 岸… | verse:8 …柔的心口靠岸 ‖ outro:3 ∅… |

Reading of the 24:

* 20 are tail spills repaired by +1 bar, e.g. `骗到 ‖ 底` → `骗到底 ‖` and `滋 ‖ 味` → `滋味 ‖`. Several land the final syllable on the first downbeat of an instrumental, which is the ambiguous case discussed in §4.
* 4 are head cuts repaired by −1/−2 bars (#1, #8, #14, #22):
  * `it ‖ s my world` becomes one line;
  * `黑 ‖ 夜你独自` moves the verse start to the line start;
  * #1 (`我恨你` repeated across chorus/instrumental) is plausible either way, and the "instrumental" label there is itself noise.

## 6. Pathological songs (flagged, not excluded)

Rules: `qwen_abc/cleaning.py:is_pathological`. **1,101 songs** across all splits carry at least one flag:

| flag | songs |
|---|---|
| melody in declared key < 70% of duration (key/tuning instability) | 432 |
| double-time suspect (≥180 BPM and < 0.55 notes per beat) | 334 |
| pitch range > 36 semitones (octave errors) | 211 |
| a section > 32 bars | 152 |
| aligner cramming (>10% of notes carry several syllables) | 86 |
| implausible note density (<1 or >9 notes per bar) | 66 |
| >25% irregular bars | 10 |

They are **not** removed from training. E1 must keep the same training examples as E0, and automatic exclusion would confound the representation ablation. The flags feed the clean test subset (`reports/CLEAN_TEST_SUBSET.md`).

What was *not* attempted automatically:

* halving double-time tempi, because it needs re-barring and cannot be verified without audio;
* re-estimating keys;
* relabelling sections.

Flagged held-out songs:

| split | song | flags | tempo |
|---|---|---|---|
| validation | `0F2wdLvHxNfLOSwtoqEzcD` | pitch_range_over_3_octaves | 91 |
| validation | `0QRKr17G0HixECRrFCYsmQ` | double_time_suspect | 214 |
| validation | `0XfOeS8zCar3E4JlmeYTrB` | double_time_suspect, section_over_32_bars | 231 |
| validation | `0iUtK3yewHsYrYR7UJrMBk` | double_time_suspect | 231 |
| validation | `0wVz36NeNRL8i4K3HBfjti` | melody_in_key_under_70pct | 91 |
| validation | `0xpRkrFwNQCI5W4duu9JFS` | melody_in_key_under_70pct | 143 |
| validation | `1EwrJpVcaME7CAC6Ebe2qT` | melody_in_key_under_70pct | 188 |
| validation | `1FcSmgcPjiorFSADzbucA6` | pitch_range_over_3_octaves | 130 |
| validation | `1Ja39aUZ16XgxR4w3QLYer` | pitch_range_over_3_octaves | 97 |
| validation | `1j4tsy5kylMSQJwCodAZS8` | double_time_suspect | 200 |
| validation | `2OyDhGtl7tCggvIhzjuW17` | pitch_range_over_3_octaves | 111 |
| validation | `2XVD8xyPoWgsTvKILF5HRZ` | aligner_cramming_over_10pct | 79 |
| validation | `2iFKTjAbYtaotesjdeAjRo` | melody_in_key_under_70pct | 103 |
| validation | `3WkQZjmz2NUSmlRcgH55zS` | pitch_range_over_3_octaves | 115 |
| validation | `3mnFlaLQG7DNozZ14tNxep` | melody_in_key_under_70pct | 79 |
| validation | `4159DU8FP3vuVJ6hYwdcpQ` | aligner_cramming_over_10pct, implausible_note_density | 143 |
| validation | `425umKmKb0r8RrrSZ6DnQD` | double_time_suspect | 231 |
| validation | `44vUhvwTEtTpCGzkiGAOXX` | double_time_suspect, melody_in_key_under_70pct | 214 |
| validation | `4qv2tJVrbzx4CifS2oUSOO` | aligner_cramming_over_10pct, implausible_note_density | 70 |
| validation | `5BYgbZJdfhBpmnRHlNqveY` | melody_in_key_under_70pct | 94 |
| validation | `5Kru2AufQBibFZgwGS4d7R` | melody_in_key_under_70pct | 120 |
| validation | `5gpjcV8YLhQ6k5n8JCUAai` | aligner_cramming_over_10pct, implausible_note_density | 125 |
| validation | `5u4xEeD5bsH7itWodhHDeE` | pitch_range_over_3_octaves | 136 |
| validation | `5vbZ3B9vokxmzNudD5ovzC` | melody_in_key_under_70pct | 83 |
| validation | `61T4BtQAApFVsAMzSGIySn` | melody_in_key_under_70pct | 94 |
| validation | `6GAtdtjuxBGbetqQEVaHba` | melody_in_key_under_70pct | 86 |
| validation | `6OYk2qyY5rwp5BWJHOm3Cs` | melody_in_key_under_70pct | 71 |
| validation | `6khTRk1UpsuiVEFJ8gyXnG` | melody_in_key_under_70pct | 81 |
| validation | `7GJxG9P2SW3peeqg29dJK0` | pitch_range_over_3_octaves | 103 |
| validation | `7hCoYs6BF8HbZx8fBtaXru` | aligner_cramming_over_10pct, implausible_note_density | 111 |
| validation | `7pnatxi4bvzSgRr5e6l7gA` | melody_in_key_under_70pct | 136 |
| validation | `7x44fWUgsH9Y1JtPe8bNFr` | melody_in_key_under_70pct | 136 |
| test | `05Yqjul6jLsld0Vh8K36KH` | section_over_32_bars | 136 |
| test | `1bIiaFMdH3udqWyXyeQMoH` | aligner_cramming_over_10pct, implausible_note_density | 70 |
| test | `1qcgbHATg4tVyVqudr2pxK` | double_time_suspect | 214 |
| test | `20WNhehqO6xfLVktkDefs2` | pitch_range_over_3_octaves | 107 |
| test | `27o7ErWiozSj47o3dnPvVf` | aligner_cramming_over_10pct | 68 |
| test | `2Wcz0tFIoJn6xEzVQkOlDR` | melody_in_key_under_70pct | 79 |
| test | `3KEgUftkAZzQeugm2kVwbT` | pitch_range_over_3_octaves | 103 |
| test | `3Wlox2t540dXpZpm20Zjyb` | double_time_suspect | 200 |
| test | `3xqFOwya52XoVYyO2HphmG` | melody_in_key_under_70pct | 120 |
| test | `40RF2zemUhNk0aPkIJNWxM` | melody_in_key_under_70pct | 100 |
| test | `4T4TeoIBft7oIW5WOte5Au` | pitch_range_over_3_octaves | 120 |
| test | `5bVKIpODZbjn82L68A80hv` | melody_in_key_under_70pct | 115 |
| test | `5fTRjBPvXscWyOOukAGtW5` | double_time_suspect | 231 |
| test | `5igSw7TE7vW1iKMCU9ACMJ` | pitch_range_over_3_octaves | 120 |
| test | `5mh8LL52KOZXeb80QpcUVG` | pitch_range_over_3_octaves | 100 |
| test | `5rVNJGnGulbxFwEgUeqAPg` | pitch_range_over_3_octaves | 130 |
| test | `6H1qeLID2gslZMgxpwfACS` | melody_in_key_under_70pct, aligner_cramming_over_10pct | 83 |
| test | `6NgFmG5lumvnyBEpjye3Rd` | double_time_suspect, melody_in_key_under_70pct | 188 |
| test | `7DeY1fKszqaJKtXUk1NiiH` | double_time_suspect | 188 |
| test | `7q5MNn6t2YpWawZvWDpAhv` | double_time_suspect | 231 |
