# Second assessment mode: follow the model voice (design)

Date: 2026-10-19 · Status: approved by the owner ("làm luôn"), implemented 2026-10-19

## Goal

Add a second way to judge a syllable's tone, next to the rule-based verdict the app has today:

| Mode | API value | Page label (en) | Verdict comes from |
|---|---|---|---|
| Rules | `rules` | Rules | `tone_pipeline` rules (start/end/min/max features, half-third, sandhi lists, guards) |
| Model voice | `model_voice` | Follow the model voice (vi: *Theo giọng mẫu*) | DTW alignment and tone-specific sigmoid score against teacher F0 contours, with same-syllable book-audio fallback |
| Fusion (default) | `fusion` | Fusion | Combines the Rules and Model Voice verdicts |

The owner wants to *see* the new mode's results; it does not count for points yet.

## Decisions already taken

1. A `model_voice` attempt is **not** counted on the Points or Accuracy boards, but it **does** update the per-lesson "Latest result" and the ✓ marks (changed on the owner's request, 2026-10-19: the newest attempt is the latest result, whichever mode judged it). It is saved with its mode.
2. The switch sits **on the practice page**. Each page visit and API request that omits `scoring_mode` defaults to **Fusion**. A selector choice lasts until the page reloads; custom text always uses Rules.
3. Expected-tone paths are mode-specific. Rules uses the current surface-tone rules and accepted sandhi alternatives. Model Voice uses the tone sequence printed in the lesson or Chapter 5 book Pinyin and does not inherit Rules-only alternates. Its result exposes the book expected tone and keeps the Rules expected tone/status as separate diagnostics.

## Behaviour

- Expected tones depend on the selected mode. Rules uses its rule-derived surface tones and accepted sandhi alternatives. Model Voice uses the Pinyin tones printed in the lesson or Chapter 5 book and does not inherit Rules-only alternates. Phrase cards and pitch-chart guides use the active mode's expected sequence; Model Voice keeps Rules expected tones and status as separate diagnostics.
- Same recording path as today: audio preparation, MFA alignment, calibrated Praat F0, the judged window (`_segment_for`). Only the per-syllable verdict differs.
- Available for book sentences and Chapter 5 words (they have teacher material). Own text (`practice_mode=custom`) has no teachers: it always uses Rules; the selector is disabled there and the result says which mode was used.
- A syllable the rules cannot judge (neutral tone, `_gated`: bad alignment, too short, too few voiced frames) keeps its rule verdict (Okay).
- For the others the model-voice score gives the verdict:
  - **Good**: sigmoid score >= `good_at` (0.60) and acoustic quality reaches `confidence.likely_correct_quality`.
  - **Needs practice**: score <= `practice_at` (0.30) **and** the rule system's accusation guards allow it (quality >= `practice_quality`, window not shorter than the floor, alignment not doubtful, no frames trimmed behind a tracking jump, `accuse_ok`). These guards still prevent low-quality or uncertain contours from causing an accusation.
  - otherwise **Okay**.
- Each syllable keeps its Rules verdict and expected tone in `rule_status` / `rule_expected_surface_tone`, and gets `model_score`, `model_closest_tone` (only when not Good: the tone whose references match best, "sounds closest to tone N").
- Word cards, summary counts and the result card are built from the final statuses, exactly as now.

## Scoring method (measured on the 5 most active learners, teachers, and the synthetic wrong-tone set)

Per syllable, on the semitone values of the judged window (finite F0 frames, every second 5 ms step = 10 ms):

1. Normalize the syllable by itself: `z = (x - mean(x)) / max(std(x), std_floor * recording_spread)`.
2. Align it to each teacher contour with DTW using absolute semitone difference and a Sakoe-Chiba band of `band` times the longer contour (at least wide enough for both endpoints).
3. Apply a five-point moving average to each pair of aligned sequences.
4. Calculate cosine similarity on the smoothed, aligned sequences; take the median across the 60 references for that tone.
5. Apply the tone's fitted sigmoid, `1 / (1 + exp(-k (s - m)))`. The fitted `midpoint` and `steepness` are stored per tone in `model_voice_contours.json`.
6. The score is used for Good (`good_at: 0.60`) and Needs practice (`practice_at: 0.30`), with the existing acoustic-quality and accusation guards.
7. Several accepted tones: take the best score, but a **non-expected** tone counts only if its own score >= 0.80.

## Data

- `model_voice_contours.json` in the repo: per book-Pinyin tone 1-4, 60 normalized contours and fitted sigmoid parameters, picked with `Random(20261019)` from teacher sentence recordings through the normal pipeline. No teacher audio is needed at run time.
- `tools/build_model_voice_contours.py` rebuilds the contours and fits a sigmoid for each tone from the collected teacher and learner data.
- Parameters live in `tone_v1_config.yaml` under `model_voice:` (`good_at`, `practice_at`, `alternative_tone_min`, `max_competing_cosine_margin`, `reference_sample_good_at`, `reference_sample_min_quality`, `band`, `smooth`, `std_floor`, `references_per_tone`).

## Where it plugs in

- API: `POST /api/assess-pronunciation` gets `scoring_mode` (form; defaults to `fusion`; `rules`, `model_voice`, and `fusion` are accepted). The payload carries the mode actually used.
- `tone_pipeline`: a new module `model_voice.py` (load contours, score, verdict). `verify_syllable` already holds the judged window and the guard flags; it computes the model-voice fields when asked, and a final pass before the payload is built sets `status`/`feedback` from them and keeps `rule_status`. Variant selection (`best_variant`) keeps using rule verdicts, so the chosen reading does not depend on the mode.
- The comparison runs against ~60 references per syllable; closest-tone lookup runs only for syllables that are not Good.
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

- Unit tests: DTW path and band, five-point smoothing, tone-specific sigmoid midpoint and thresholds; accepted-tone maximum and the 0.80 bar; guards (short window, doubtful alignment, trimmed frames never accuse); neutral and gated syllables keep the rule verdict; payload field; invalid mode is a 400; own text ignores the mode.
- Rules mode unchanged: the golden snapshot and the whole suite stay green.
- Boards: a `model_voice` attempt changes neither Points, Accuracy, nor progress.
- Page tests: the Fusion default, selector reset on page reload, the form field, disabled for own text, the badge; five-language key count.
- Measurement: run the builder report and compare its contour-only scores with archived rules verdicts and synthetic wrong-tone examples; the report does not apply pipeline accusation guards.

## Risks and open items

- The sigmoid fits use Rules-labeled learner syllables and teacher recordings; they are not human-verified calibration.
- Do not interpret the sigmoid score as a calibrated probability.
- Chapter 5 words are one or two syllables: a single syllable gets a verdict from a short window; the guards decide.

## Not in scope

Counting pure Model Voice attempts on the boards; changing the Rules verdicts; a per-learner default; showing both verdicts side by side.

## Implementation update: Fusion default (2026-10-20)

Fusion is now a third scoring choice and the practice page's default. It runs the existing Rules and Model Voice paths with their own expected-tone contexts, then combines their statuses without changing either scorer:

- Either `likely_correct` verdict makes the fused syllable `likely_correct`.
- `practice` requires both paths to return `practice`; if one path is uncertain or has no verdict and neither is Good, Fusion returns `insufficient_evidence`.
- A Rules-only acceptance displays the Rules expected tone. Otherwise Fusion displays Model Voice's book-Pinyin target. Both original statuses, targets, and the fusion basis remain in the result diagnostics.
- Each page visit and API request that omits `scoring_mode` now defaults to Fusion, so returning learners start in Fusion even if an older browser saved another choice. An explicit selector choice lasts until that page is reloaded. Pure Model Voice attempts remain excluded, Fusion attempts count on the leaderboards, and custom text always uses Rules.
- Fusion adds no second audio preparation, alignment, or F0 extraction; it combines the two results from the same recording.
