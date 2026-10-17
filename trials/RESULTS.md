# Rule trials - results (2026-10-17)

Everything below was measured against the production config of that moment: single words unchanged from the golden snapshot;
sentence profile = 130 ms floor for "Needs practice" + the two tone-4 rules (`t4_by_duration`, `t4_before_t4`). The left-context
rule for tone 2 (best variant, "either 2.0 st") was added afterwards: sentences now 91.8% Good / 7 false alarms.
Teacher audio is all correct, so every "Needs practice" on it is a **false alarm**. Reproduce with
`python compare_configs.py trials/<file>.yaml` (alignments are cached after the first run).

## Where production stands
| | before the tone-4 rules + 130 ms | now |
|---|---|---|
| 100 teacher sentences, 1329 syllables: Good | 89.8% | **91.3%** (91.8% with the left-context rule) |
| false alarms (syllables) | 15 | **8** (7 with the left-context rule) |
| sentences with a "Needs practice" phrase | 13% | **8%** (7%) |
| single words (237 syllables): Good / false alarms | 93.7% / 0 | 93.7% / 0 (golden gate: 237 identical) |

## Sentence practice, each trial vs production (Good % / false alarms of 1329 syllables)
| Trial | Good | False alarms | Notes |
|---|---|---|---|
| production | 91.3 | 8 | |
| rule 7: tail guard for clause-final T1/2/3 | 91.3 | 7 | no effect |
| rule 9: ignore the first 15% (<= 40 ms) of a syllable | 90.1 | 11 | **worse** (3 Good -> Needs practice, 15 Good -> Okay) |
| rule 10: soften repeated characters (看看, 謝謝) | 91.3 | 8 | no effect on teacher audio |
| left context, T2 after T1/T2 needs 2.0 st | 91.7 | 7 | +6 Good |
| left context, T2 after T1/T2 needs 1.5 st | 91.9 | 7 | +8 Good |
| left context, T2 after a high-ending syllable (>= 3 st) 2.0 st | 91.5 | 7 | +3 Good |
| left context, either condition 2.0 st | 91.8 | 7 | +7 Good |
| owner's word/phrase table (trial C), phrase floor 130 | 93.3 | 6 | see wrong-tone cost |
| trial C as pasted, phrase floor 100 | 93.3 | **12** | the 100 ms floor brings back six short function words |
| rule set B (words + sentences) | 93.8 | 6 | see wrong-tone cost |
| config A (earlier paste) | 94.1 | 7 | 2 new false alarms (這, 條) |

## Single words (237 teacher syllables: Good %, 0 false alarms in every case) and the wrong-tone stress test (520 synthetic clips)
| Trial | Words Good | Stress: wrong tones caught | Stress: wrong tones shown Good |
|---|---|---|---|
| production | 93.7 | 49.8% | 20.0% |
| half-third also at the end of a word | 94.1 | 47.0% | 22.1% |
| tone-4 length rule on words too | 93.2 | 48.5% | 20.8% |
| trial C (word column) | 93.2 | 45.3% | 22.3% |
| rule set B | 94.1 | 45.0% | 24.2% |
| config A | 94.5 | 44.8% | 26.3% |

Words are already 93.7% Good with 0 false alarms, so loosening them gains 1-2 syllables and costs 3-5 points of wrong-tone detection.
Worst single cases: direction-reversed pitch caught 50% -> 29-31% (B, C, A); wide pitch range shown Good 11.5% -> 23.1% (B, C, A).

## Wrong-tone cost of the PHRASE-mode settings (native tone t scored as if tone e were expected; counts of 234/198/336/387 native T1/T2/T3/T4)
The stress set only exercises word-mode settings, so the phrase column is measured on native sentence audio instead.
| said -> expected | today | trial C phrase column |
|---|---|---|
| T4 -> T1 | 9% | **44%** |
| T3 -> T1 | 25% | 40% |
| T2 -> T1 | 51% | 71% |
| T1 -> T2 | 43% | 61% |
| T4 -> T2 | 6% | 10% |
Total extra wrong-tone acceptances: trial C +327, rule set B / config A +203, left-context T2 (2.0 st) +12, (1.5 st) +26, (end >= 3 st) +4,
out of 957 native non-target syllables. The T1 loosening (delta 3.0 st, range 5.5 st) is what lets falling tones pass as level tone 1.

## Left context (tone.left_context): why tone 2 and why the previous END level
Tone 2 after: clause start 95% Good, T3 95%, T4 88%, T1 82%, T2 68%. After a syllable that ended high (> 3 st) 68% Good, rise 2.5 st;
ended mid 88%; ended low 90% (rise 4.8 st). The rule helps several speakers (友美, 旁白, 元真, 媽媽), not one.

## Not done (no evidence of need)
Rule 11 (T3 chains): 3 of 88 chain syllables are not Good. Rule 12 (declination): median drift 0.04 st per 300 ms syllable vs 2-3 st thresholds.

## The 12 rules: status
| # | Rule | Status |
|---|---|---|
| 1 | T3 + T3 -> T2 + T3 | applied (long-standing; also accepts T2 across word ends) |
| 2 | T3 before T1/T2/T4 may be a half third | applied (long-standing) |
| 3 | T3 at the end of a phrase may be low | applied in sentences; NOT for single words (+1 syllable Good, stress-test recall -2.8 points) |
| 4 | 不 / 一 sandhi in the expected-tone layer | applied (long-standing) |
| 5 | Neutral tone not scored by shape, never "Needs practice" | applied (long-standing) |
| 6 | T4 before T4 may fall less | applied in sentences (`t4_before_t4`, 2.0 st), plus the length rule `t4_by_duration` |
| 7 | Phrase-final / declination guard (judge the core, not the tail) | implemented (`final_tail_guard`), OFF: no effect (false alarms 8 -> 7) |
| 8 | Short syllables cannot be accused | applied: 130 ms floor in sentences (false alarms 15 -> 8) |
| 9 | Do not trust the start F0 too much | trimming the start was tried (`carryover_trim`): harmful, OFF; done properly as left-context conditioning (below): applied |
| 10 | Second character of a repeated pair is softer | implemented (`soften_reduplication`), OFF: no effect on teacher audio |
| 11 | Complex T3 chains: prosodic grouping | not done: 3 of 88 chain syllables are not Good |
| 12 | Declination / focus: local normalization | not done: median drift 0.04 st per syllable vs 2-3 st thresholds |
| - | Left context (previous tone / previous F0 endpoint / phrase-final) | applied for tone 2 after a high tone (`left_context`); engine also takes `next`, `phrase_final` |

## Fit for non-native learners (2026-10-17)

Teacher audio cannot show what suits learners: teachers score ~1.0 almost everywhere, so a threshold between 0.60 and 0.70 changes nothing for
them. The archived learner attempts (`user data/`: 101 attempts, 3 speakers, 887 syllables; 55 of one speaker's 79 attempts are the SAME sentence
5-1-1, so this is suggestive, not a measurement; no truth labels) were replayed through the real assessment path, one lever at a time.

Learners vs teachers, sentence mode: Good 57.9% vs 91.8%, Okay 39.1% vs 7.7%, Needs practice 3.0% vs 0.5%. Of the learners' 347 Okay syllables
only ~110 have a match score under 0.70; most are `unstable_pitch_track` (~180), `bad_alignment` (~90), `alignment_failed` (21). Learner match
scores are bimodal (78% >= 0.90, 17% <= 0.20, 0.5% between 0.60 and 0.70), so the Good-match bar is nearly irrelevant to them.

| lever (on top of production) | learner sentence Good (of 855) | wrong tones shown Good, native replay | wrong tones shown Good, 520 clips |
|---|---|---|---|
| production | 57.9% | 22.7% | 20.0% |
| Good match 0.65 | +3 syllables | +0.8 pt | +0.8 pt |
| Good match 0.60 | +4 | +1.4 pt | |
| Good quality 0.60 | +26 | +0.4 pt | +0.3 pt |
| Good quality 0.55 / 0.50 | +41 / +44 | +0.6 / +0.7 pt | not run (the guard against bad tracks is what it weakens) |
| tone 1 shape (2.5 / 5.0) | +2 -1 | +3.6 pt | |
| tone 2 shape / tone 3 shape | +4 / +6 -2 | +0.7 / +1.1 pt | |
| `unstable_scope: syllable` | **+35** (no new Needs practice) | **0.0 pt** | **0.0 pt** (recall 48.0%, false alarms unchanged) |

Why: `unstable_pitch_quality_factor` (x0.75) was applied to EVERY syllable of a recording as soon as one syllable had a pitch jump left in its
segment; 21 of the 26 syllables a quality-0.60 bar would rescue carried that flag. With `unstable_scope: syllable` only the syllables that
contain a jump keep the penalty (they are gated as `octave_jump_in_segment` anyway), a track unreliable as a whole still penalises everyone,
and a recording with a jump anywhere cannot produce a Needs practice (`held_back`; without that, 36 new accusations appeared on learner clips).
Teachers: sentences Good 91.8 -> 93.0%, words unchanged, false alarms 7 -> 7. Production adopts it; Good thresholds stay 0.65 quality / 0.70 match.
Not adopted: lower Good quality or match bars, the tone 1/2/3 shape loosening (small gain, tone 3 lost two correct syllables at the half-third boundary).
Listening list: `outputs/config_trials/learner_flips_unstable_scope.csv` (the syllables the adopted setting changes; file + from/to ms).

## Learners, second pass: LATIC corpus, the dictionary, tail jumps, difficulty levels (2026-10-17)

Data found under `Recorded Data/audio test` and `mandarin-speaking data` (none has tone-error labels): `user data/` (4 learners: Hau tran, Huu nghia, M, Kien;
104 attempts), the LATIC corpus (4 learners of Chinese reading mainland texts: Korean M30, Chadian M26, Russian F22 and F19; 637 sentences each; 40
short sentences each were used = 160 utterances, 1838 syllables) and `Audio/` (TTS and recordings of unknown origin, not used).

1. **Dictionary.** 103 of the 160 LATIC sentences were refused (`alignment_failed`): the dictionary held 1,828 single characters, so 70% of the 5,401 most
   common characters were missing and one of them in a sentence refuses the whole sentence (this also broke custom practice). `build_mfa_dictionary.py
   --big5` adds them (borrowed phones; 133 syllables composed from an initial and a final). Refused sentences 103 -> 1. The 237 single-word teacher syllables
   re-aligned from scratch are identical to the golden snapshot.
2. **LATIC under production** (Good / Okay / Needs practice): Korean 78.8 / 12.3 / 9.0, Chadian 63.4 / 25.3 / 11.2, Russian F22 53.9 / 24.8 / 21.3,
   Russian F19 61.1 / 23.5 / 15.3. Tone 1 is Good 85-100% for all four; the Needs-practice syllables are tone 4 (121 look like tone 1) and tone 2 (68 look like
   tone 1): flattened tones, as in monotone text reading. These look like real deviations, so making the system "easier" here is a choice, not a bug fix.
3. **Tail jumps.** 49 syllables of the archived attempts were gated by a pitch jump (`octave_jump_in_segment`): 63% have the shorter side of the jump <= 30 ms,
   92% <= 60 ms, 29 are tone 4 (a creaky end of a fall). `sanity.jump_fragment_trim_ms: 40` drops that fragment and judges the rest; a recording that needed a cut
   can be Good but never Needs practice (without that guard Needs practice went 2.9% -> 5.9% on the archived attempts). Words Good 93.7 -> 95.4%, teacher
   sentences 93.0 -> 94.1%, archived attempts +18 Good of 888, 520 clips identical.
4. **Levels** (`levels:` in the config; the learner's group picks one unless the request names one: foreigner -> easy). easy = every contour-size demand of tones 2/3/4
   x0.8 (tone 1 and position limits unchanged). Cost/benefit versus standard:

| | LATIC Good | LATIC Needs practice | archived sentences Good | 520 clips: wrong flagged | wrong shown Good | native replay: wrong tones accepted |
|---|---|---|---|---|---|---|
| standard | 64.3% | 14.3% | 63.0% | 48.0% | 20.0% | 22.7% |
| easy (x0.8) | 69.0% | 11.9% | 65.1% | 46.1% | 21.3% | 24.4% |
| x0.65 (`trials/easy_65.yaml`) | 72.5% | 9.8% | 66.6% | 44.6% | 21.5% | 26.0% |

Teacher sentences Good 93.0 -> 94.0% (x0.8), false alarms 7 -> 8; single words unchanged. The price of any easier level is the same: a weak or flat
tone is accepted as that tone. Not measurable with this data: how many of the extra Good syllables are really right (no labels).
