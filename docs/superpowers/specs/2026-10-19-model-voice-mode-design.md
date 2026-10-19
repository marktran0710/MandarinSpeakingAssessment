# Second assessment mode: follow the model voice (design)

Date: 2026-10-19 · Status: approved by the owner ("làm luôn"), implemented 2026-10-19

## Goal

Add a second way to judge a syllable's tone, next to the rule-based verdict the app has today:

| Mode | API value | Page label (en) | Verdict comes from |
|---|---|---|---|
| Rules (default, today) | `rules` | Rules | `tone_pipeline` rules (start/end/min/max features, half-third, sandhi lists, guards) |
| Model voice | `model_voice` | Follow the model voice (vi: *Theo giọng mẫu*) | DTW of the judged pitch window against contours taken from the teachers' recordings (the "model voice") |

The owner wants to *see* the new mode's results; it does not count for points yet.

## Decisions already taken

1. A `model_voice` attempt is **not** counted on the Points or Accuracy boards, nor in the per-lesson "Latest result" / ✓ marks. It is saved and shown, labelled with its mode.
2. The switch sits **on the practice page** and is remembered per browser (localStorage). Default: Rules.
3. The accepted tones of a syllable are the ones the rules already use (`[expected, *also_accept_tones]` in `tone_pipeline.verify_syllable`), not a separate list. Today the only alternative is T2 for a third tone before another third tone across a word end; 不 and 一 take one sandhi tone and need nothing extra (measured: 不→T2 96%, 一→T4 100% of the Good syllables also pass under the new formula).

## Behaviour

- Same recording path as today: audio preparation, MFA alignment, calibrated Praat F0, the judged window (`_segment_for`). Only the per-syllable verdict differs.
- Available for book sentences and Chapter 5 words (they have teacher material). Own text (`practice_mode=custom`) has no teachers: it always uses Rules; the selector is disabled there and the result says which mode was used.
- A syllable the rules cannot judge (neutral tone, `_gated`: bad alignment, too short, too few voiced frames) keeps its rule verdict (Okay).
- For the others the model-voice score gives the verdict:
  - **Good**: score >= `good_at` (0.60) and acoustic quality >= `confidence.likely_correct_quality`.
  - **Needs practice**: score <= `practice_at` (0.30) **and** the rule system's accusation guards allow it (quality >= `practice_quality`, window not shorter than the floor, alignment not doubtful, no frames trimmed behind a tracking jump, `accuse_ok`). The pure DTW method has no such guards (a failed alignment gives Good or Practice at random); this keeps them.
  - otherwise **Okay**.
- Each syllable keeps its rule verdict as `rule_status`, and gets `model_score`, `model_closest_tone` (only when not Good: the tone whose references match best, "sounds closest to tone N").
- Word cards, summary counts and the result card are built from the final statuses, exactly as now.

## Scoring method (measured on the 5 most active learners, teachers, and the synthetic wrong-tone set)

Per syllable, on the semitone values of the judged window (finite F0 frames, every second 5 ms step = 10 ms):

1. Normalise the syllable by itself: `z = (x - mean(x)) / max(std(x), 0.2)`. (The old recording-wide z-score is what drove many scores to the 0.0025 floor: two level contours on opposite sides of the sentence mean give cosine -1.)
2. DTW against each reference of the accepted tone: cost `|a_i - b_j|`, Sakoe-Chiba band of 15 % of the longer length (never narrower than the length difference).
3. Moving average, width 5, on both aligned sequences.
4. Similarity `exp(-MSE / 0.8^2)`; median over 60 references per tone.
5. Per-tone sigmoid `1 / (1 + exp(-k (s - m)))`, fitted so that the score is 0.30 at the 1.5 % quantile and 0.70 at the 7 % quantile of the correct syllables of that tone:

   | Tone | midpoint m | steepness k |
   |---|---|---|
   | 1 | 0.160 | 21.9 |
   | 2 | 0.345 | 6.1 |
   | 3 | 0.092 | 17.0 |
   | 4 | 0.457 | 7.4 |

6. Several accepted tones: take the best score, but a **non-expected** tone counts only if its own score >= 0.80 (at 0.60 a wrong T1 passes as T2 65 % of the time; at 0.80, 48.7 %; at 0.90 the owner's sandhi example is lost).

Expected effect (experiment, no guards, so to be re-measured): AUC 0.892 (Good vs rule-Practice, learners) and 0.814 (right vs wrong tone, synthetic), against 0.831 and 0.750 for the old DTW+cosine. With Good at 0.60: 95 % of the rule-Good syllables pass, 20 % of the rule-Practice ones, 33 % of the known wrong synthetic syllables (rules: 19 %), 93.5 % of the teachers.

## Data

- `model_voice_contours.json` in the repo: per tone 1-4, 60 contours (own-z, 10 ms), picked with `Random(20261019)` from the 100 teacher sentence recordings through the normal pipeline. Plus the sigmoid parameters. Small (~50 KB). No teacher audio is needed at run time.
- `tools/build_model_voice_contours.py` rebuilds it (and re-fits the four sigmoids) from the teacher recordings and the learners' attempts, and prints the measurements above, so it can be repeated when more learners have practised.
- Parameters live in `tone_v1_config.yaml` under `model_voice:` (`good_at`, `practice_at`, `alternative_tone_min`, `band`, `smooth`, `tau`, `std_floor`, `references_per_tone`).

## Where it plugs in

- API: `POST /api/assess-pronunciation` gets `scoring_mode` (form, `rules` by default or `model_voice`; anything else is a 400). The payload carries `scoring_mode` (the mode actually used).
- `tone_pipeline`: a new module `model_voice.py` (load contours, score, verdict). `verify_syllable` already holds the judged window and the guard flags; it computes the model-voice fields when asked, and a final pass before the payload is built sets `status`/`feedback` from them and keeps `rule_status`. Variant selection (`best_variant`) keeps using rule verdicts, so the chosen reading does not depend on the mode.
- DTW needs speed: ~60 references per syllable. Use `numba` (installed in this environment but not in `requirements.txt`; add it, `cache=True`, with a slower pure-Python fallback if it cannot be imported). The closest-tone lookup runs only for syllables that are not Good.
- `practice_store`: `summary.scoring_mode` is saved with the attempt; `_leaderboard` and `progress` skip `model_voice` attempts. `user_archive` records the mode.

## Page

- A select next to the practice-mode controls: "Scoring: Rules / Model voice", with a short hint "Attempts scored by the model voice are shown but not counted on the leaderboards."
- Names of the second mode (the owner chose the Vietnamese one; the others are proposals for a native check):

  | en | zh-TW | vi | id | hi |
  |---|---|---|---|---|
  | Follow the model voice | 對照範例發音 | Theo giọng mẫu | Sesuai suara contoh | आदर्श आवाज़ के अनुसार |

  "Rules" stays: Rules / 規則 / Quy tắc / Aturan / नियम.
- Result card: a small badge with the mode; in model-voice mode the syllable note may show "Closest to tone N".
- Strings in the five languages (en, zh-TW, vi, id, hi).

## Verification

- Unit tests: the scoring numbers on fixed contours; band, smoothing, floor; accepted-tone maximum and the 0.80 bar; guards (short window, doubtful alignment, trimmed frames never accuse); neutral and gated syllables keep the rule verdict; payload field; invalid mode is a 400; own text ignores the mode.
- Rules mode unchanged: the golden snapshot and the whole suite stay green.
- Boards: a `model_voice` attempt changes neither Points, Accuracy, nor progress.
- Page tests: the select, its default, localStorage, the form field, disabled for own text, the badge; five-language key count.
- Measurement: run the builder's report with the guards on and compare with the numbers above.

## Risks and open items

- The cut points were fitted to the rule verdicts of five learners (no human labels); tone 3 is the weakest (AUC 0.70 on synthetic data); the learner data has no rule-Practice third tone at all. Re-fit when more learners have practised.
- The guards have not been tried together with the DTW verdicts yet (measure first, before the page ships).
- Files 02 and 08 of the listening set (rule: Needs practice; DTW: Good) still need the owner's ear: if the rules are right there, the DTW mode is the more lenient one on flat/falling confusions.
- Chapter 5 words are one or two syllables: a single syllable gets a verdict from a short window; the guards decide.

## Not in scope

Counting model-voice attempts on the boards; changing the Rules verdicts; a per-learner default; showing both verdicts side by side.
