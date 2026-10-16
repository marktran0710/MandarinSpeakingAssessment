import json
import unittest
from unittest import mock

import numpy as np

import capt_core
import tone_pipeline as tp

CONFIG = tp.load_config()
SR = 16000
STEP = 0.005


def harmonic(f0_by_time, duration, sample_rate=SR):
    t = np.arange(int(duration * sample_rate)) / sample_rate
    f0 = f0_by_time(t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    audio = sum(np.sin(h * phase) / h for h in range(1, 7)) * 0.18
    return audio.astype(np.float32)


def tone_audio(contours, syllable_sec=0.35, gap_sec=0.05):
    """Concatenate syllables whose F0 follows anchor lists; an item may be (anchors, duration_sec)."""
    pieces, spans, cursor = [np.zeros(int(0.2 * SR), dtype=np.float32)], [], 0.2
    silence = np.zeros(int(gap_sec * SR), dtype=np.float32)
    for item in contours:
        anchors, duration = item if isinstance(item, tuple) else (item, syllable_sec)
        grid = np.linspace(0, duration, len(anchors))
        pieces.append(harmonic(lambda t, g=grid, a=anchors: np.interp(t, g, a), duration))
        spans.append((cursor, cursor + duration))
        pieces.append(silence)
        cursor += duration + gap_sec
    return np.concatenate(pieces), spans


def make_track(st_values, baseline=200.0, disagree_st=0.0, start=0.0):
    """Fake extraction + cleaned contour from a semitone sequence (feature-level fixtures)."""
    st = np.asarray(st_values, dtype=float)
    times = start + np.arange(len(st)) * STEP
    hz = baseline * 2 ** (st / 12)
    extraction = {"ac": hz.copy(), "cc": hz * 2 ** (disagree_st / 12), "support_tracks": [], "times": times}
    clean = {"times": times, "st": st, "octave_corrected": np.zeros(len(st), dtype=bool), "step": STEP}
    return extraction, clean


SPEAKER = {"baseline_hz": 200.0, "pitch_floor": 90.0, "pitch_ceiling": 450.0}


def verify_one(char, st_values, *, stable=True, disagree_st=0.0, end=None, alignment_ok=True, nucleus=True):
    extraction, clean = make_track(st_values, disagree_st=disagree_st)
    end = end if end is not None else len(st_values) * STEP
    syllable = {"char": char, "start": 0.0, "end": end, "phones": [], "alignment_ok": alignment_ok,
                "nucleus": (0.0, end) if nucleus else None}
    context = capt_core.expected_tone_contexts([char])[0]
    return tp.verify_syllable(syllable, context, extraction, clean, CONFIG, SPEAKER, stable)


class FrozenConfigTests(unittest.TestCase):
    def test_praat_values_match_the_spec(self):
        self.assertEqual(CONFIG["pitch"], {
            "time_step": 0.005, "max_candidates": 15, "silence_threshold": 0.03, "voicing_threshold": 0.35,
            "octave_cost": 0.01, "octave_jump_cost": 0.35, "voiced_unvoiced_cost": 0.14, "very_accurate": False})

    def test_thresholds_match_the_spec(self):
        self.assertEqual(CONFIG["sanity"]["ac_cc_tolerance_st"], 1.0)
        self.assertEqual(CONFIG["sanity"]["octave_jump_st"], 7.0)
        self.assertEqual(CONFIG["sanity"]["boundary_hit_rate"], 0.05)
        self.assertEqual(CONFIG["cleanup"]["median_filter_frames"], 5)
        self.assertEqual(CONFIG["cleanup"]["interpolate_gap_ms"], 30)
        self.assertEqual(CONFIG["alignment"]["min_syllable_ms"], 70)
        self.assertEqual(CONFIG["alignment"]["min_voiced_frames"], 6)
        self.assertFalse(CONFIG["alignment"]["allow_unk"])
        self.assertEqual(CONFIG["tone"]["t4_min_fall_st"], 3.0)
        self.assertEqual(CONFIG["confidence"]["practice_quality"], 0.80)
        self.assertEqual(CONFIG["confidence"]["practice_mismatch"], 0.80)
        self.assertEqual(CONFIG["confidence"]["likely_correct_match"], 0.70)
        self.assertEqual(CONFIG["calibration_version"], "speaker_f0_v1")


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        anchors = np.array([230, 230, 300, 190, 260, 175, 320, 200, 280, 230.0])
        grid = np.linspace(0, 6.0, len(anchors))
        self.audio = harmonic(lambda t: np.interp(t, grid, anchors), 6.0)

    def test_the_profile_can_be_stored_as_json(self):
        # practice_store saves the calibration with json.dumps: numpy scalars (np.bool_) in it break the save
        result = tp.calibrate_voice(self.audio, SR, CONFIG)
        self.assertEqual(json.loads(json.dumps(result))["status"], "calibrated")

    def test_floor_and_ceiling_follow_the_formulas_and_the_distribution_is_kept(self):
        result = tp.calibrate_voice(self.audio, SR, CONFIG)
        self.assertEqual(result["pitch_floor"], float(np.clip(round(0.4 * result["f0_median"] / 10) * 10, 50, 120)))
        self.assertEqual(result["pitch_ceiling"],
                         float(np.clip(np.ceil(1.25 * result["f0_p95"] / 50) * 50, 350, 550)))
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["calibration_version"], "speaker_f0_v1")
        for key in ("f0_p10", "f0_median", "f0_p90", "f0_p95"):
            self.assertIn(key, result)
        self.assertLess(result["f0_p10"], result["f0_median"])
        self.assertLess(result["f0_median"], result["f0_p95"])
        self.assertEqual(result["pitch_floor_hz"], result["pitch_floor"])
        self.assertIn(result["acoustic_profile"], ("low", "mid", "high", "very_high"))
        self.assertTrue(result["reliable"])

    def test_clamps_apply(self):
        result = tp.calibrate_voice(harmonic(lambda t: np.full_like(t, 330.0), 6.0), SR, CONFIG)
        self.assertLessEqual(result["pitch_floor"], 120)
        self.assertGreaterEqual(result["pitch_ceiling"], 350)

    def test_rounding_examples_from_the_spec(self):
        self.assertEqual(tp._round_to(230 * 0.40, 10, "nearest"), 90)
        self.assertEqual(tp._round_to(346 * 1.25, 50, "ceil"), 450)

    def test_rejects_short_recording(self):
        with self.assertRaises(tp.CalibrationError) as caught:
            tp.calibrate_voice(self.audio[: SR * 2], SR, CONFIG)
        self.assertEqual(caught.exception.reason, "too_short")

    def test_rejects_clipped_recording(self):
        with self.assertRaises(tp.CalibrationError) as caught:
            tp.calibrate_voice(np.clip(self.audio * 10, -1, 1), SR, CONFIG)
        self.assertEqual(caught.exception.reason, "clipped")

    def test_rejects_silence_and_noise_instead_of_creating_a_poor_profile(self):
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(SR * 6) * 0.05).astype(np.float32)
        with self.assertRaises(tp.CalibrationError):
            tp.calibrate_voice(noise, SR, CONFIG)
        with self.assertRaises(tp.CalibrationError):
            tp.calibrate_voice(np.zeros(SR * 6, dtype=np.float32), SR, CONFIG)

    def test_rejects_too_little_voiced_speech(self):
        audio = np.zeros(SR * 6, dtype=np.float32)
        audio[: SR] = self.audio[: SR]          # one second of voice in six
        with self.assertRaises(tp.CalibrationError) as caught:
            tp.calibrate_voice(audio, SR, CONFIG)
        self.assertEqual(caught.exception.reason, "too_little_voicing")

    def test_octave_candidates_are_rejected_before_percentiles(self):
        frames = np.r_[np.full(100, 200.0), np.full(5, 400.0), np.full(5, 100.0)]
        kept, rejected = tp.reject_octave_candidates(frames, CONFIG)
        self.assertEqual(rejected, 10)
        self.assertEqual(set(kept), {200.0})

    def test_acoustic_profile_labels(self):
        self.assertEqual([tp.acoustic_profile(m, CONFIG) for m in (120, 170, 230, 320)],
                         ["low", "mid", "high", "very_high"])

    def test_token_report_warns_when_tokens_are_missing(self):
        voiced = np.zeros(400, dtype=bool)
        voiced[10:100] = True
        report = tp.analyze_calibration_tokens(np.arange(400) * STEP, voiced, CONFIG)
        self.assertTrue(report["warnings"])
        self.assertEqual(len(report["tokens"]), 1)


class SemitoneTests(unittest.TestCase):
    def test_hz_to_semitone(self):
        self.assertAlmostEqual(float(tp.hz_to_st(225.0, 150.0)), 7.02, places=2)
        self.assertAlmostEqual(float(tp.hz_to_st(375.0, 250.0)), 7.02, places=2)
        self.assertAlmostEqual(float(tp.hz_to_st(300.0, 150.0)), 12.0, places=6)
        self.assertAlmostEqual(float(tp.hz_to_st(150.0, 150.0)), 0.0, places=6)


class SanityTests(unittest.TestCase):
    def test_ac_cc_agreement_is_counted_within_one_semitone(self):
        ac = np.full(10, 200.0)
        cc = ac.copy()
        cc[:3] = 200.0 * 2 ** (2.0 / 12)               # 3 of 10 frames off by 2 st
        merged, _ = tp.merge_ac_cc(ac, cc, 1.0)
        metrics = tp.sanity_metrics(merged, ac, cc, 90, 450, CONFIG)
        self.assertAlmostEqual(metrics["ac_cc_agreement"], 0.7)
        self.assertIn("ac_cc_disagree", metrics["suspicious"])

    def test_boundary_hits_are_detected_at_ceiling_and_floor(self):
        track = np.r_[np.full(20, 200.0), np.full(5, 449.0)]
        top = tp.sanity_metrics(track, track, track, 90, 450, CONFIG)
        self.assertIn("ceiling_hit", top["suspicious"])
        low = tp.sanity_metrics(np.r_[np.full(20, 200.0), np.full(5, 90.5)], track, track, 90, 450, CONFIG)
        self.assertIn("floor_hit", low["suspicious"])
        clean = tp.sanity_metrics(np.full(25, 200.0), track, track, 90, 450, CONFIG)
        self.assertNotIn("ceiling_hit", clean["suspicious"])
        self.assertNotIn("floor_hit", clean["suspicious"])

    def test_octave_jump_and_coverage_collapse_are_flagged(self):
        jumpy = np.r_[np.full(10, 200.0), np.full(10, 400.0)]
        self.assertIn("octave_jump", tp.sanity_metrics(jumpy, jumpy, jumpy, 90, 450, CONFIG)["suspicious"])
        merged = np.full(40, np.nan)
        merged[:10] = 200.0
        ac = np.full(40, 200.0)
        self.assertIn("coverage_collapse", tp.sanity_metrics(merged, ac, ac, 90, 450, CONFIG)["suspicious"])

    def test_ceiling_retry_is_directional_and_clamped(self):
        self.assertEqual(tp._retry_candidates(90, 450, ["ceiling_hit"], CONFIG), [(90, 550)])
        self.assertEqual(tp._retry_candidates(90, 550, ["ceiling_hit"], CONFIG), [(90, 600)])
        self.assertEqual(tp._retry_candidates(90, 600, ["ceiling_hit"], CONFIG), [])
        self.assertEqual(tp._retry_candidates(90, 450, ["octave_jump"], CONFIG), [(90, 350), (110, 450)])
        self.assertEqual(tp._retry_candidates(90, 320, ["octave_jump"], CONFIG)[0], (90, 300))

    def test_floor_retry_is_directional_and_clamped(self):
        self.assertEqual(tp._retry_candidates(90, 450, ["floor_hit"], CONFIG), [(70, 450)])
        self.assertEqual(tp._retry_candidates(60, 450, ["floor_hit"], CONFIG), [(50, 450)])
        self.assertEqual(tp._retry_candidates(50, 450, ["floor_hit"], CONFIG), [])

    def test_pass_selection_prefers_the_pass_that_is_not_an_octave_above_the_others(self):
        def make(f0, floor, ceiling):
            f0 = np.asarray(f0, dtype=float)
            return {"floor": floor, "ceiling": ceiling, "f0": f0, "times": np.arange(len(f0)) * STEP,
                    "metrics": tp.sanity_metrics(f0, f0, f0, floor, ceiling, CONFIG)}
        base = make(np.full(40, 185.0), 90, 450)
        doubled = make(np.full(40, 370.0), 90, 550)
        self.assertIs(tp.select_pass([doubled, base], CONFIG), base)

    def test_pass_selection_handles_passes_with_different_frame_counts(self):
        def make(n, floor):
            f0 = np.full(n, 200.0)
            return {"floor": floor, "ceiling": 450, "f0": f0, "times": 0.01 + np.arange(n) * STEP,
                    "metrics": tp.sanity_metrics(f0, f0, f0, floor, 450, CONFIG)}
        first, retry = make(916, 90), make(918, 70)
        self.assertIn(tp.select_pass([first, retry], CONFIG), (first, retry))   # must not raise on 916 vs 918 frames

    def test_clean_voice_passes_without_retry(self):
        audio, _ = tone_audio([[200, 200, 200]] * 3)
        result = tp.extract_f0(audio, SR, 90, 450, CONFIG)
        self.assertFalse(result["diagnostics"]["retried"])
        self.assertTrue(result["diagnostics"]["stable"])
        self.assertGreater(result["diagnostics"]["ac_cc_agreement"], 0.8)
        voiced = result["f0"][np.isfinite(result["f0"])]
        self.assertAlmostEqual(float(np.median(voiced)), 200.0, delta=5.0)


class CleanupTests(unittest.TestCase):
    def test_octave_up_is_corrected_only_with_supporting_evidence(self):
        f0 = np.full(30, 185.0)
        f0[15] = 370.0
        fixed, mask = tp.correct_octaves(f0, [np.full(30, np.nan)], CONFIG)
        self.assertEqual(fixed[15], 370.0)
        self.assertFalse(mask.any())
        support = np.full(30, np.nan)
        support[15] = 186.0
        fixed, mask = tp.correct_octaves(f0, [support], CONFIG)
        self.assertAlmostEqual(fixed[15], 185.0)
        self.assertTrue(mask[15])

    def test_octave_down_subharmonic_is_doubled_only_with_supporting_evidence(self):
        f0 = np.full(30, 350.0)
        f0[12:15] = 175.0
        fixed, mask = tp.correct_octaves(f0, [np.full(30, np.nan)], CONFIG)
        np.testing.assert_allclose(fixed[12:15], 175.0)
        support = np.full(30, np.nan)
        support[12:15] = 349.0
        fixed, mask = tp.correct_octaves(f0, [support], CONFIG)
        np.testing.assert_allclose(fixed[12:15], 350.0)
        self.assertEqual(int(mask.sum()), 3)

    def test_real_high_pitch_that_matches_context_is_untouched(self):
        f0 = np.full(30, 370.0)
        fixed, mask = tp.correct_octaves(f0, [np.full(30, 185.0)], CONFIG)
        np.testing.assert_allclose(fixed, f0)
        self.assertFalse(mask.any())

    def test_median_filter_removes_a_spike(self):
        st = np.zeros(20)
        st[10] = 9.0
        self.assertAlmostEqual(tp.smooth_and_interpolate(st, STEP, CONFIG)[10], 0.0)

    def test_gaps_up_to_30ms_are_bridged_longer_gaps_are_kept(self):
        short = np.r_[np.zeros(10), np.full(6, np.nan), np.ones(10) * 2]
        long = np.r_[np.zeros(10), np.full(7, np.nan), np.ones(10) * 2]
        self.assertTrue(np.isfinite(tp.smooth_and_interpolate(short, STEP, CONFIG)).all())
        self.assertTrue(np.isnan(tp.smooth_and_interpolate(long, STEP, CONFIG)[12]))


class AlignmentTests(unittest.TestCase):
    TIERS = {
        "words": {"entries": [[0.0, 0.84, "<eps>"], [0.84, 1.12, "要"], [1.12, 1.43, "在"], [2.62, 4.6, "<eps>"]]},
        "phones": {"entries": [[0.0, 0.84, "sil"], [0.84, 0.93, "j"], [0.93, 1.12, "aw˨˩˦"],
                               [1.12, 1.25, "ts"], [1.25, 1.43, "aj˥˩"], [2.62, 4.6, "sil"]]},
    }

    def test_nucleus_is_the_tone_marked_phone_not_the_onset(self):
        syllables = tp.syllables_from_tiers(self.TIERS, list("要在"))
        self.assertEqual(syllables[0]["nucleus"], (0.93, 1.12))
        self.assertEqual(syllables[1]["nucleus"], (1.25, 1.43))

    def test_unk_spn_and_word_mismatch_are_rejected(self):
        unk = {"words": {"entries": [[0, 1, "<unk>"]]}, "phones": {"entries": [[0, 1, "spn"]]}}
        with self.assertRaises(tp.AlignmentError):
            tp.syllables_from_tiers(unk, ["淇"])
        spn = {"words": {"entries": [[0, 1, "房"]]}, "phones": {"entries": [[0, 1, "spn"]]}}
        with self.assertRaises(tp.AlignmentError):
            tp.syllables_from_tiers(spn, ["房"])
        with self.assertRaises(tp.AlignmentError):
            tp.syllables_from_tiers(self.TIERS, list("要他"))
        with self.assertRaises(tp.AlignmentError):
            tp.syllables_from_tiers(self.TIERS, list("要"))

    def test_overlapping_or_empty_syllables_are_marked_bad(self):
        marked = tp._mark_alignment_sanity([
            {"start": 0.0, "end": 0.3}, {"start": 0.1, "end": 0.35}, {"start": 0.5, "end": 0.5}])
        self.assertEqual([s["alignment_ok"] for s in marked], [True, False, False])


def contour_features(anchors, n=60, duration=0.3):
    times = np.linspace(0, duration, n)
    st = np.interp(times, np.linspace(0, duration, len(anchors)), anchors)
    return tp.syllable_features(times, st, 0.0, duration)


class ToneRuleTests(unittest.TestCase):
    def test_features_cover_the_twelve_required_values(self):
        feats = contour_features([1.0, -3.0, 0.0])
        for key in ("duration", "voiced_ratio", "start_ST", "end_ST", "min_ST", "max_ST", "range_ST",
                    "min_position", "max_position", "early_to_late_delta", "fall_before_min", "rise_after_min"):
            self.assertIn(key, feats)

    def test_tone_one_level_even_with_slight_drift(self):
        scores = tp.tone_scores(contour_features([1.0, 1.4, 1.2]), False, CONFIG)
        self.assertGreaterEqual(scores[1], 0.9)
        self.assertLess(scores[4], 0.3)

    def test_tone_two_uses_rise_after_low_not_start_vs_end(self):
        st = [float(tp.hz_to_st(h, 230.0)) for h in (268, 183, 237)]   # 房
        feats = contour_features(st)
        self.assertLess(feats["end_ST"], feats["start_ST"])
        self.assertGreaterEqual(tp.tone_scores(feats, False, CONFIG)[2], 0.9)

    def test_full_tone_three(self):
        self.assertGreaterEqual(tp.tone_scores(contour_features([1.0, -3.0, -3.0, 0.0]), False, CONFIG)[3], 0.9)

    def test_half_tone_three_only_where_the_context_allows_it(self):
        half = contour_features([1.0, -3.0, -3.5])
        self.assertLess(tp.tone_scores(half, False, CONFIG)[3], 0.5)
        self.assertGreaterEqual(tp.tone_scores(half, True, CONFIG)[3], 0.7)

    def test_tone_four_needs_a_strong_fall(self):
        self.assertGreaterEqual(tp.tone_scores(contour_features([5.0, 0.0, -4.0]), False, CONFIG)[4], 0.9)
        self.assertLess(tp.tone_scores(contour_features([1.0, 0.0, -0.5]), False, CONFIG)[4], 0.3)


class ToneFourByDurationTests(unittest.TestCase):
    """tone.t4_by_duration: a short syllable needs less fall. Off unless configured; sentence trials only."""
    RULE = {"short_ms": 180, "medium_ms": 300, "min_fall_short_st": 1.5, "min_fall_medium_st": 2.0,
            "min_fall_long_st": 2.5, "require_end_below_start": True}

    def config(self, **rule):
        import copy
        config = copy.deepcopy(CONFIG)
        config["tone"]["t4_by_duration"] = {**self.RULE, **rule}
        return config

    @staticmethod
    def t4(anchors, syllable_ms, config):
        times = np.linspace(0, 0.2, 40)
        st = np.interp(times, np.linspace(0, 0.2, len(anchors)), anchors)
        return tp.tone_scores(tp.syllable_features(times, st, 0.0, 0.2, syllable_ms=syllable_ms), False, config)[4]

    # early-to-late falls of these contours (first-third mean minus last-third mean): 1.8 st and 2.1 st
    SMALL_FALL = [1.3, 0.0, -1.3]
    MEDIUM_FALL = [1.5, 0.0, -1.5]

    def test_the_same_small_fall_passes_in_a_short_syllable_and_not_in_a_long_one(self):
        config = self.config()
        self.assertGreaterEqual(self.t4(self.SMALL_FALL, 150, config), 0.9)      # < 180 ms needs 1.5 st: 1.8 is plenty
        self.assertLess(self.t4(self.SMALL_FALL, 250, config), 0.9)              # 180-300 ms needs 2.0 st
        self.assertLess(self.t4(self.SMALL_FALL, 350, config), 0.6)              # >= 300 ms needs 2.5 st
        self.assertGreater(self.t4(self.SMALL_FALL, 150, config), self.t4(self.SMALL_FALL, 250, config))
        self.assertGreater(self.t4(self.SMALL_FALL, 250, config), self.t4(self.SMALL_FALL, 350, config))

    def test_the_three_bands_use_their_own_numbers(self):
        config = self.config()
        self.assertGreaterEqual(self.t4(self.MEDIUM_FALL, 250, config), 0.9)     # 2.1 st is enough for a medium syllable
        self.assertLess(self.t4(self.MEDIUM_FALL, 400, config), 0.9)             # but not for a long one (needs 2.5)
        self.assertEqual(tp.t4_min_fall({"syllable_ms": 179.9}, config["tone"]), 1.5)
        self.assertEqual(tp.t4_min_fall({"syllable_ms": 180.0}, config["tone"]), 2.0)
        self.assertEqual(tp.t4_min_fall({"syllable_ms": 299.9}, config["tone"]), 2.0)
        self.assertEqual(tp.t4_min_fall({"syllable_ms": 300.0}, config["tone"]), 2.5)

    def test_a_tone_four_must_end_lower_than_it_starts(self):
        # high in the middle but the edges end above where they start: the means fall by about 1.1 st, the ends do not
        recovers = [-5.0, 4.0, 3.0, -1.0, -1.0]
        self.assertEqual(self.t4(recovers, 150, self.config()), 0.0)
        self.assertGreater(self.t4(recovers, 150, self.config(require_end_below_start=False)), 0.4)
        self.assertGreaterEqual(self.t4(self.SMALL_FALL, 150, self.config()), 0.9)    # a normal fall is not affected

    def test_without_the_setting_nothing_changes(self):
        for anchors in ([5.0, 0.0, -4.0], [1.0, 0.0, -0.5], [1.0, 0.5, 0.0, -0.8], [0.0, -2.0, -3.0, 0.5]):
            plain = self.t4(anchors, None, CONFIG)
            for ms in (100, 250, 500):
                self.assertEqual(self.t4(anchors, ms, CONFIG), plain)             # the length is ignored, as before
        self.assertEqual(tp.t4_min_fall({}, self.config()["tone"]), 3.0)         # rule set but no length known: single value

    def test_production_config_has_no_rule_in_either_profile(self):
        import tone_assessment as ta
        self.assertNotIn("t4_by_duration", ta.get_config()["tone"])
        self.assertNotIn("t4_by_duration", ta.get_config("sentence")["tone"])

    def test_the_syllable_length_reaches_the_scorer(self):
        feats = tp.syllable_features(np.linspace(0, 0.1, 20), np.linspace(1, -1, 20), 0.0, 0.1, syllable_ms=250)
        self.assertEqual(feats["syllable_ms"], 250.0)
        self.assertNotIn("syllable_ms", tp.syllable_features(np.linspace(0, 0.1, 20), np.linspace(1, -1, 20), 0.0, 0.1))


class SurfaceToneTests(unittest.TestCase):
    def tones(self, text):
        contexts = capt_core.expected_tone_contexts([text])
        return [c["lexical_tone"] for c in contexts], [c["target_tone"] for c in contexts]

    def test_bu_sandhi_before_tone_four(self):
        self.assertEqual(self.tones("不錯"), ([4, 4], [2, 4]))

    def test_third_tone_sandhi(self):
        self.assertEqual(self.tones("你好"), ([3, 3], [2, 3]))
        self.assertEqual(self.tones("旅館"), ([3, 3], [2, 3]))

    def test_neutral_tone_comes_from_the_lexicon_not_the_audio(self):
        self.assertEqual(self.tones("好嗎")[1][-1], 5)


class GatingTests(unittest.TestCase):
    def test_decide_is_asymmetric(self):
        self.assertEqual(tp.decide(0.55, 0.42, CONFIG), "insufficient_evidence")      # weak evidence never accuses
        self.assertEqual(tp.decide(0.87, 0.75, CONFIG), "likely_correct")
        self.assertEqual(tp.decide(0.70, 0.10, CONFIG), "insufficient_evidence")      # mismatch high, quality too low
        self.assertEqual(tp.decide(0.85, 0.25, CONFIG), "insufficient_evidence")      # mismatch 0.75 < 0.80
        self.assertEqual(tp.decide(0.85, 0.10, CONFIG), "practice")
        self.assertEqual(tp.decide(0.85, 0.10, CONFIG, allow_practice=False), "insufficient_evidence")

    def test_quality_weights(self):
        self.assertAlmostEqual(sum(CONFIG["confidence"]["quality_weights"].values()), 1.0)
        full = {"ac_cc_agreement": 1, "voiced_coverage": 1, "octave_stability": 1, "alignment_quality": 1}
        self.assertAlmostEqual(tp.acoustic_quality(full, CONFIG), 1.0)

    def test_tone_mismatch_is_one_minus_match(self):
        result = verify_one("罵", np.linspace(5, -4, 60))
        self.assertAlmostEqual(result["tone_match"] + result["tone_mismatch"], 1.0, places=2)

    def test_good_tone_four_is_accepted(self):
        result = verify_one("罵", np.linspace(5, -4, 60))
        self.assertEqual(result["status"], "likely_correct")
        self.assertEqual(result["pinyin"], "ma4")
        self.assertEqual(result["feedback"], CONFIG["feedback"]["likely_correct"])

    def test_high_confidence_strong_mismatch_triggers_practice(self):
        result = verify_one("罵", np.zeros(60))                 # level contour for an expected falling tone
        self.assertEqual(result["status"], "practice")
        self.assertEqual(result["detected_alternative"], 1)
        self.assertEqual(result["feedback"], CONFIG["feedback"]["practice"][4])

    def test_low_quality_evidence_cannot_trigger_practice(self):
        poor = verify_one("罵", np.zeros(60), disagree_st=4.0)  # AC and CC disagree by 4 st everywhere
        self.assertLess(poor["acoustic_quality"], CONFIG["confidence"]["practice_quality"])
        self.assertEqual(poor["status"], "insufficient_evidence")

    def test_poor_ac_cc_agreement_lowers_quality(self):
        good = verify_one("罵", np.linspace(5, -4, 60))
        bad = verify_one("罵", np.linspace(5, -4, 60), disagree_st=4.0)
        self.assertLess(bad["acoustic_quality"], good["acoustic_quality"])
        self.assertEqual(bad["quality_components"]["ac_cc_agreement"], 0.0)

    def test_unstable_pitch_track_caps_quality_below_the_practice_gate(self):
        result = verify_one("罵", np.zeros(60), stable=False)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("unstable_pitch_track", result["reasons"])

    def test_too_few_voiced_frames_is_insufficient_not_wrong(self):
        st = np.full(60, np.nan)
        st[:4] = 0.0
        result = verify_one("罵", st)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("too_few_voiced_frames", result["reasons"])

    def test_too_short_syllable_is_insufficient(self):
        result = verify_one("罵", np.zeros(10), end=0.05)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("syllable_too_short", result["reasons"])

    def test_bad_alignment_is_insufficient(self):
        result = verify_one("罵", np.zeros(60), alignment_ok=False)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("bad_alignment", result["reasons"])

    def test_octave_jump_inside_a_segment_blocks_a_tone_decision(self):
        st = np.r_[np.full(30, 0.0), np.full(30, -10.0)]     # an apparent 10 st "fall" in one frame
        result = verify_one("罵", st)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("octave_jump_in_segment", result["reasons"])

    def test_word_final_third_tone_inside_a_clause_may_be_a_half_third(self):
        # 我(3) | 愛(4): a word boundary after 我 stops sandhi but 我 is not clause-final.
        contexts = capt_core.expected_tone_contexts(["我愛"], phrase_end_indices=[1])
        self.assertFalse(contexts[0]["allow_half_t3"])
        self.assertTrue(tp.allows_half_third(contexts[0]))
        half = np.r_[np.linspace(2.0, -3.5, 40), np.full(20, -3.8)]      # fall -> low, no rise
        extraction, clean = make_track(half)
        syllable = {"char": "我", "start": 0.0, "end": 0.3, "phones": [], "nucleus": (0.0, 0.3)}
        result = tp.verify_syllable(syllable, contexts[0], extraction, clean, CONFIG, SPEAKER, True)
        self.assertEqual(result["status"], "likely_correct")

    def test_clause_final_third_tone_still_needs_the_full_shape(self):
        contexts = capt_core.expected_tone_contexts(["我"])
        self.assertFalse(tp.allows_half_third(contexts[0]))

    def test_fallback_trim_is_used_without_a_nucleus(self):
        result = verify_one("罵", np.linspace(5, -4, 70), nucleus=False)
        self.assertEqual(result["analysis_segment"]["source"], "syllable_trimmed")

    def test_diagnostics_include_the_documented_fields(self):
        result = verify_one("罵", np.linspace(5, -4, 60))
        for key in ("character", "pinyin", "lexical_tone", "expected_surface_tone", "start_ms", "end_ms",
                    "speaker_f0_median", "pitch_floor", "pitch_ceiling", "ac_cc_agreement", "voiced_ratio",
                    "start_st", "end_st", "min_st", "max_st", "min_position", "rise_after_min_st",
                    "tone_match", "acoustic_quality", "status"):
            self.assertIn(key, result)


class EndToEndTests(unittest.TestCase):
    def run_utterance(self, contours, script):
        audio, spans = tone_audio(contours)
        syllables = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s + 0.04, e)}
                     for c, (s, e) in zip(script, spans)]
        calibration = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
        return tp.verify_utterance(audio, SR, script, calibration, alignment=syllables, config=CONFIG)

    def test_correct_tones_are_never_flagged_for_practice(self):
        result = self.run_utterance(
            [[200, 200, 200, 200], [170, 165, 200, 235], [260, 230, 190, 150]], "媽麻罵")
        self.assertEqual([s["status"] for s in result["syllables"]], ["likely_correct"] * 3)
        self.assertEqual(result["feedback"], CONFIG["feedback"]["likely_correct"])
        self.assertNotIn("_feats", result["syllables"][0])

    def test_clear_wrong_tone_is_flagged(self):
        result = self.run_utterance([[200, 200, 200, 200]], "罵")
        self.assertEqual(result["syllables"][0]["status"], "practice")

    def test_third_tone_sandhi_is_the_expected_surface_form(self):
        result = self.run_utterance([[170, 175, 190, 210], [190, 150, 140, 175]], "你好")
        self.assertEqual([s["lexical_tone"] for s in result["syllables"]], [3, 3])
        self.assertEqual([s["expected_surface_tone"] for s in result["syllables"]], [2, 3])

    def test_failed_alignment_gives_insufficient_evidence_for_every_syllable(self):
        audio, _ = tone_audio([[200, 200, 200]] * 2)
        result = tp.verify_utterance(audio, SR, "媽罵", {"status": "calibrated", "pitch_floor": 90,
                                     "pitch_ceiling": 450, "f0_median": 200}, alignment=[], config=CONFIG)
        self.assertTrue(result["alignment_error"])
        self.assertEqual({s["status"] for s in result["syllables"]}, {"insufficient_evidence"})
        self.assertEqual(result["feedback"], CONFIG["feedback"]["insufficient_evidence"])

    CAL = {"status": "calibrated", "pitch_floor": 90, "pitch_ceiling": 450, "f0_median": 200}

    def test_overall_feedback_does_not_praise_a_partly_unchecked_utterance(self):
        audio, spans = tone_audio([[200] * 4])
        audio = np.concatenate([audio, np.zeros(int(0.4 * SR), dtype=np.float32)])   # second syllable is silent
        second = (spans[0][1] + 0.05, spans[0][1] + 0.35)
        syllables = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None},
                     {"char": "媽", "start": second[0], "end": second[1], "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽媽", self.CAL, alignment=syllables, config=CONFIG)
        self.assertEqual([s["status"] for s in result["syllables"]], ["likely_correct", "insufficient_evidence"])
        self.assertEqual(result["feedback"], CONFIG["feedback"]["insufficient_evidence"])

    def test_speech_outside_every_aligned_syllable_means_the_alignment_is_not_trusted(self):
        # Two voiced syllables were spoken but the script/alignment only accounts for the first one.
        audio, spans = tone_audio([[200] * 4, [200] * 4])
        syllables = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽", self.CAL, alignment=syllables, config=CONFIG)
        self.assertTrue(result["f0_extraction"]["speech_outside_alignment"])
        self.assertGreater(result["f0_extraction"]["voiced_outside_syllables"], 0.25)
        self.assertEqual(result["syllables"][0]["status"], "insufficient_evidence")
        self.assertIn("bad_alignment", result["syllables"][0]["reasons"])

    def test_fully_explained_speech_is_not_flagged(self):
        audio, spans = tone_audio([[200] * 4])
        syllables = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽", self.CAL, alignment=syllables, config=CONFIG)
        self.assertFalse(result["f0_extraction"]["speech_outside_alignment"])

    def test_pitch_range_override_keeps_the_speaker_median(self):
        audio, spans = tone_audio([[200, 200, 200, 200]])
        syllables = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽", {"status": "calibrated", "pitch_floor": 90, "pitch_ceiling": 450,
                                     "f0_median": 200}, alignment=syllables, config=CONFIG, pitch_range=(75, 500))
        self.assertEqual((result["f0_extraction"]["pitch_floor"], result["f0_extraction"]["pitch_ceiling"]), (75, 500))
        self.assertEqual(result["f0_extraction"]["speaker_f0_median"], 200.0)


class NeutralToneTests(unittest.TestCase):
    """v1: neutral tone is not scored. Usable audio -> likely_correct; poor pitch / alignment -> insufficient_evidence."""
    CAL = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
    SHAPES = {"flat": [190, 190, 190], "falling": [280, 200, 140], "rising": [140, 200, 280],
              "dipping": [220, 150, 230], "wide": [330, 130, 330]}

    def neutral(self, contour, neutral_sec, alignment_ok=True):
        audio, spans = tone_audio([([200, 200, 200, 200], 0.35), (contour, neutral_sec)])
        syllables = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s + 0.02, e), "alignment_ok": alignment_ok}
                     for c, (s, e) in zip("媽吧", spans)]
        return tp.verify_utterance(audio, SR, "媽吧", self.CAL, alignment=syllables, config=CONFIG)["syllables"][1]

    def test_neutral_tone_is_never_classified_as_t1_to_t4(self):
        for contour in self.SHAPES.values():
            result = self.neutral(contour, 0.30)
            self.assertEqual(result["expected_surface_tone"], 5)
            self.assertNotIn("tone_scores", result)
            self.assertIsNone(result["detected_alternative"])

    def test_usable_audio_is_accepted_whatever_the_pitch_shape_or_duration(self):
        for name, contour in self.SHAPES.items():
            for seconds in (0.15, 0.30, 0.50):        # no duration rule, no pitch-range rule
                result = self.neutral(contour, seconds)
                self.assertEqual(result["status"], "likely_correct", (name, seconds))

    def test_neutral_tone_is_never_practice(self):
        for contour in self.SHAPES.values():
            self.assertNotEqual(self.neutral(contour, 0.30)["status"], "practice")

    def test_the_tone_itself_is_reported_as_not_scored(self):
        result = self.neutral(self.SHAPES["falling"], 0.30)
        self.assertIsNone(result["tone_match"])
        self.assertIsNone(result["tone_mismatch"])
        for removed in ("neutral_checks", "duration_ratio_vs_utterance", "previous_surface_tone"):
            self.assertNotIn(removed, result)
        self.assertNotIn("neutral_tone_weak_check", result["reasons"])

    def test_a_missing_neutral_syllable_is_insufficient_not_accepted(self):
        audio, spans = tone_audio([[200, 200, 200, 200]])
        audio = np.concatenate([audio, np.zeros(int(0.4 * SR), dtype=np.float32)])        # the 吧 is silent
        end = spans[0][1] + 0.05
        syllables = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None},
                     {"char": "吧", "start": end, "end": end + 0.3, "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽吧", self.CAL, alignment=syllables, config=CONFIG)["syllables"][1]
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("too_few_voiced_frames", result["reasons"])

    def test_a_bad_alignment_is_insufficient(self):
        result = self.neutral(self.SHAPES["flat"], 0.30, alignment_ok=False)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("bad_alignment", result["reasons"])

    def test_poor_pitch_evidence_is_insufficient(self):
        # AC and CC disagree by 4 st everywhere AND every other frame has no pitch: too unreliable to call even a
        # neutral syllable usable (quality 0.53, below the 0.65 needed).
        result = verify_one("吧", np.where(np.arange(60) % 2 == 0, 0.0, np.nan), disagree_st=4.0)
        tp.finalize_neutral([result], CONFIG)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("low_acoustic_quality", result["reasons"])


class SingleSyllableAlignmentTests(unittest.TestCase):
    CAL = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}

    def run_single(self, contour, script="罵"):
        audio, _ = tone_audio([contour])
        # Any attempt to start MFA fails the test: a one-syllable script must not need it.
        with mock.patch.object(tp, "mfa_align", side_effect=AssertionError("MFA must not run")):
            return tp.verify_utterance(audio, SR, script, self.CAL, config=CONFIG)

    def test_mfa_is_only_needed_for_two_or_more_syllables(self):
        self.assertFalse(tp.needs_mfa(1, CONFIG))
        self.assertTrue(tp.needs_mfa(2, CONFIG))
        self.assertTrue(tp.needs_mfa(1, {**CONFIG, "alignment": {**CONFIG["alignment"], "skip_mfa_single_syllable": False}}))
        self.assertEqual(tp.syllable_count("你好，再見"), 4)

    def test_force_mfa_overrides_the_voiced_run_shortcut(self):
        audio, _ = tone_audio([[260, 230, 190, 150]])
        tiers = {"words": {"entries": [[0.2, 0.55, "罵"]]},
                 "phones": {"entries": [[0.2, 0.25, "m"], [0.25, 0.55, "a˥˩"]]}}
        with mock.patch.object(tp, "mfa_align", return_value=tiers) as aligner:
            default = tp.verify_utterance(audio, SR, "罵", self.CAL, config=CONFIG)
            forced = tp.verify_utterance(audio, SR, "罵", self.CAL, config=CONFIG, force_mfa=True)
        aligner.assert_called_once()
        self.assertEqual((default["aligner"], forced["aligner"]), ("voiced_run", "mfa"))

    def test_good_single_syllable_is_checked_without_mfa(self):
        result = self.run_single([260, 230, 190, 150])
        self.assertEqual(result["aligner"], "voiced_run")
        self.assertEqual(result["syllables"][0]["status"], "likely_correct")
        self.assertEqual(result["syllables"][0]["analysis_segment"]["source"], "voiced_run")

    def test_wrong_single_syllable_is_still_flagged_without_mfa(self):
        result = self.run_single([200, 200, 200, 200])
        self.assertEqual(result["syllables"][0]["status"], "practice")

    def test_no_voiced_speech_is_an_alignment_failure_not_a_crash(self):
        with mock.patch.object(tp, "mfa_align", side_effect=AssertionError("MFA must not run")):
            result = tp.verify_utterance(np.zeros(SR, dtype=np.float32), SR, "罵", self.CAL, config=CONFIG)
        self.assertIn("voiced speech", result["alignment_error"])
        self.assertEqual(result["syllables"][0]["status"], "insufficient_evidence")

    def test_a_second_spoken_syllable_is_not_silently_ignored(self):
        audio, _ = tone_audio([[260, 230, 190, 150], [200, 200, 200, 200]], gap_sec=0.2)   # two syllables, one-syllable script
        with mock.patch.object(tp, "mfa_align", side_effect=AssertionError("MFA must not run")):
            result = tp.verify_utterance(audio, SR, "罵", self.CAL, config=CONFIG)
        self.assertTrue(result["f0_extraction"]["speech_outside_alignment"])
        self.assertEqual(result["syllables"][0]["status"], "insufficient_evidence")

    def extraction(self, voiced_pattern):
        f0 = np.array([200.0 if v else np.nan for v in voiced_pattern])
        return {"times": np.arange(len(f0)) * STEP, "f0": f0}

    def test_the_longest_voiced_run_is_the_syllable_and_short_gaps_are_bridged(self):
        pattern = [1] * 20 + [0] * 10 + [1] * 30 + [0] * 30 + [1] * 15       # gaps of 50 ms and 150 ms
        syllable = tp.single_syllable_alignment(self.extraction(pattern), "罵", CONFIG)[0]
        self.assertAlmostEqual(syllable["start"], 0.0)
        self.assertAlmostEqual(syllable["end"], 60 * STEP)                   # runs 1+2 joined across the 50 ms gap
        self.assertEqual(tp.single_syllable_alignment(self.extraction([0] * 20), "罵", CONFIG), [])


class OctaveRunCorrectionTests(unittest.TestCase):
    BASELINE = 200.0

    @staticmethod
    def tracks(f0, ac, cc):
        return np.array(f0, dtype=float), np.array(ac, dtype=float), np.array(cc, dtype=float)

    def correct(self, f0, ac, cc, config=CONFIG):
        f0, ac, cc = self.tracks(f0, ac, cc)
        return tp.correct_octave_runs(f0, ac, cc, [], self.BASELINE, config)

    def test_a_sustained_octave_down_error_from_a_single_tracker_is_fixed(self):
        nan = np.nan
        f0 = [260] * 10 + [130] * 10 + [206] * 20             # CC alone dropped an octave for 10 frames
        ac = [260] * 10 + [nan] * 10 + [206] * 20
        cc = [260] * 10 + [130] * 10 + [206] * 20
        fixed, mask = self.correct(f0, ac, cc)
        np.testing.assert_allclose(fixed[10:20], 260.0)
        self.assertTrue(mask[10:20].all())
        self.assertFalse(mask[:10].any() or mask[20:].any())

    def test_a_sustained_octave_up_error_is_fixed_too(self):
        nan = np.nan
        f0 = [180] * 20 + [360] * 8 + [180] * 12
        ac = [180] * 20 + [nan] * 8 + [180] * 12
        cc = [180] * 20 + [360] * 8 + [180] * 12
        fixed, mask = self.correct(f0, ac, cc)
        np.testing.assert_allclose(fixed[20:28], 180.0)
        self.assertEqual(int(mask.sum()), 8)

    def test_a_real_recording_pattern_is_fixed(self):
        # Seen in a learner recording: CC alone tracked 129 Hz (AC unvoiced) before the true 262 -> 206 fall.
        nan = np.nan
        f0 = [129] * 4 + [262] + [206, 202, 202, 204, 204, 205, 203, 202, 204, 203]
        ac = [nan] * 5 + f0[5:]
        cc = [129] * 4 + [262] + f0[5:]
        fixed, mask = self.correct(f0, ac, cc)
        np.testing.assert_allclose(fixed[:4], 258.0)
        self.assertEqual(int(mask.sum()), 4)
        self.assertLess(float(np.max(np.abs(12 * np.log2(fixed[1:] / fixed[:-1])))), 7.0)   # no jump left

    def test_two_equally_trusted_runs_an_octave_apart_are_left_alone(self):
        # AC and CC agree on both and both are inside the speaker's range: a real leap, not a tracking error.
        f0 = [150] * 12 + [300] * 12
        fixed, mask = self.correct(f0, f0, f0)
        np.testing.assert_allclose(fixed, f0)
        self.assertFalse(mask.any())

    def test_a_run_far_outside_the_speakers_range_is_treated_as_the_doubled_one(self):
        f0 = [200] * 12 + [400] * 12                          # +12 st: beyond the 9 st range limit
        fixed, mask = self.correct(f0, f0, f0)
        np.testing.assert_allclose(fixed[12:], 200.0)
        self.assertTrue(mask[12:].all())

    def test_jumps_that_are_not_an_octave_are_never_corrected(self):
        nan = np.nan
        for low, high in ((103, 249), (193, 108), (129, 204)):        # x2.4, x0.56, x1.6 seen in real recordings
            f0 = [high] * 12 + [low] * 8
            ac = [high] * 12 + [nan] * 8
            fixed, mask = self.correct(f0, ac, f0)
            np.testing.assert_allclose(fixed, f0)
            self.assertFalse(mask.any(), (low, high))

    def test_the_move_is_dropped_when_it_does_not_restore_a_smooth_contour(self):
        nan = np.nan
        f0 = list(np.linspace(330, 250, 12)) + [155] * 8       # x2 would land at 310, 3.8 st above the junction
        ac = f0[:12] + [nan] * 8
        fixed, mask = self.correct(f0, ac, f0)
        self.assertFalse(mask.any())

    def test_it_can_be_switched_off(self):
        nan = np.nan
        f0 = [260] * 10 + [130] * 10
        off = {**CONFIG, "cleanup": {**CONFIG["cleanup"], "octave_run_correction": False}}
        fixed, mask = self.correct(f0, [260] * 10 + [nan] * 10, f0, config=off)
        np.testing.assert_allclose(fixed, f0)

    def test_clean_contour_uses_it_and_reports_the_corrected_frames(self):
        n = 60
        times = 0.005 * np.arange(n)
        f0 = np.array([260.0] * 20 + [130.0] * 20 + [250.0] * 20)
        ac = f0.copy()
        ac[20:40] = np.nan
        extraction = {"times": times, "f0": f0, "ac": ac, "cc": f0.copy(), "support_tracks": []}
        clean = tp.clean_contour(extraction, self.BASELINE, CONFIG)
        self.assertTrue(clean["octave_corrected"][20:40].all())
        self.assertLess(np.nanmax(np.abs(np.diff(clean["st"]))), 3.0)       # no jump left in the contour


class JumpsCountOnlyInsideSyllablesTests(unittest.TestCase):
    CAL = {"status": "calibrated", "pitch_floor": 75.0, "pitch_ceiling": 450.0, "f0_median": 200.0}

    @staticmethod
    def joined(contours):
        pieces = [harmonic(lambda t, g=np.linspace(0, 0.3, len(a)), a=a: np.interp(t, g, a), 0.3) for a in contours]
        return np.concatenate([np.zeros(int(0.2 * SR), dtype=np.float32), *pieces, np.zeros(int(0.1 * SR), dtype=np.float32)])

    def test_a_step_between_two_syllables_is_not_a_tracking_error(self):
        audio = self.joined([[260] * 3, [105] * 3])           # 15 st apart, with no gap between the syllables
        syllables = [{"char": "媽", "start": 0.2, "end": 0.5, "phones": [], "nucleus": None},
                     {"char": "媽", "start": 0.5, "end": 0.8, "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽媽", self.CAL, alignment=syllables, config=CONFIG)
        diagnostics = result["f0_extraction"]
        self.assertEqual(diagnostics["octave_jumps_in_segments"], 0)
        self.assertTrue(diagnostics["stable"])
        self.assertNotIn("unstable_pitch_track", result["syllables"][0]["reasons"])

    def test_a_jump_inside_one_syllable_still_makes_the_track_unstable(self):
        audio = self.joined([[260] * 3, [105] * 3])
        one_syllable = [{"char": "媽", "start": 0.2, "end": 0.8, "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽", self.CAL, alignment=one_syllable, config=CONFIG)
        self.assertGreaterEqual(result["f0_extraction"]["octave_jumps_in_segments"], 1)
        self.assertFalse(result["f0_extraction"]["stable"])
        self.assertEqual(result["syllables"][0]["status"], "insufficient_evidence")


class EdgePitchStepTests(unittest.TestCase):
    """A few frames after an implausible one-frame step must not decide between accusing and accepting."""

    @staticmethod
    def level_then_step(drift_from=0.44, drift_to=3.3, jump_to=8.3, tail_frames=6, n=53):
        # A T1 that drifts slowly up (0.4 -> 3.3 st) whose last 30 ms are tracked 5 st higher in one frame (the shape of
        # a real recording of 車): without the step it is a level tone, with it the contour looks like a late rise.
        return list(np.linspace(drift_from, drift_to, n - tail_frames)) + [jump_to] * tail_frames

    def test_a_short_plateau_after_a_big_step_is_flagged(self):
        times = np.arange(53) * STEP
        drop = tp.edge_fragment_mask(times, np.array(self.level_then_step()), 3.5, 0.040)
        self.assertEqual(int(drop.sum()), 6)
        self.assertTrue(drop[-6:].all())

    def test_a_smooth_rise_and_a_long_plateau_are_not_fragments(self):
        times = np.arange(60) * STEP
        self.assertIsNone(tp.edge_fragment_mask(times, np.linspace(-2, 6, 60), 3.5, 0.040))
        long_plateau = np.array(list(np.linspace(0, 1, 30)) + [6.0] * 30)        # 150 ms is a real movement
        self.assertIsNone(tp.edge_fragment_mask(times, long_plateau, 3.5, 0.040))

    def test_a_t1_that_only_looks_rising_because_of_the_step_is_not_practice(self):
        flagged = verify_one("媽", self.level_then_step())
        self.assertEqual(flagged["status"], "insufficient_evidence")
        self.assertIn("edge_pitch_step", flagged["reasons"])
        # ...and it really would have been an accusation without the guard
        config = {**CONFIG, "cleanup": {**CONFIG["cleanup"], "edge_step_st": 99.0}}
        extraction, clean = make_track(self.level_then_step())
        syllable = {"char": "媽", "start": 0.0, "end": 0.265, "phones": [], "alignment_ok": True, "nucleus": (0.0, 0.265)}
        context = capt_core.expected_tone_contexts(["媽"])[0]
        unguarded = tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)
        self.assertEqual(unguarded["status"], "practice")

    def test_without_the_step_the_same_t1_is_accepted(self):
        self.assertEqual(verify_one("媽", list(np.linspace(1.1, 3.3, 60)))["status"], "likely_correct")

    def test_a_clear_rising_tone_is_unaffected(self):
        result = verify_one("麻", list(np.linspace(-4, 6, 60)))
        self.assertEqual(result["status"], "likely_correct")
        self.assertNotIn("edge_pitch_step", result["reasons"])

    def test_a_step_that_does_not_change_the_verdict_is_ignored(self):
        fall = list(np.linspace(6, -6, 60))
        result = verify_one("罵", [11.0] * 4 + fall[4:])                         # a 4-frame blip at the very start
        self.assertEqual(result["status"], "likely_correct")
        self.assertNotIn("edge_pitch_step", result["reasons"])

    def test_it_can_only_turn_a_decision_into_an_abstention(self):
        # an accusation resting on the artefact becomes an abstention; nothing becomes a new accusation or acceptance
        times, st = np.arange(53) * STEP, np.array(self.level_then_step())
        expected_changes = tp.edge_step_changes_verdict(times, st, 0.0, 0.265, 1, False, "practice", 0.95, CONFIG)
        self.assertTrue(expected_changes)
        self.assertFalse(tp.edge_step_changes_verdict(times, np.linspace(-4, 6, 60), 0.0, 0.3, 2, False,
                                                      "likely_correct", 0.95, CONFIG))


class BoundarySearchTests(unittest.TestCase):
    TONES = capt_core.expected_tone_contexts(["媽罵"])            # T1 then T4

    @staticmethod
    def contour():
        # T1 level at 0 for 150 ms, then a T4 that starts high (+8) and falls to -6.
        return np.r_[np.zeros(30), np.linspace(8, -6, 30), np.full(30, -6.0)]

    def syllables(self, boundary_frame):
        boundary = boundary_frame * STEP
        return [{"char": "媽", "start": 0.0, "end": boundary, "phones": [], "nucleus": (0.0, boundary)},
                {"char": "罵", "start": boundary, "end": 90 * STEP, "phones": [], "nucleus": (boundary, 90 * STEP)}]

    def verify(self, st, syllables, config):
        extraction, clean = make_track(st)
        found, info = tp.search_boundaries(syllables, self.TONES, clean, config)
        results = [tp.verify_syllable(s, c, extraction, clean, config, SPEAKER, True) for s, c in zip(found, self.TONES)]
        return results, info

    @staticmethod
    def with_search(enabled, **changes):
        return {**CONFIG, "boundary_search": {**CONFIG["boundary_search"], "enabled": enabled, **changes}}

    def test_a_boundary_that_is_60_ms_late_is_pulled_back(self):
        late = self.syllables(42)                                  # true boundary is at frame 30
        before, _ = self.verify(self.contour(), late, self.with_search(False))
        after, info = self.verify(self.contour(), late, self.with_search(True))
        self.assertNotEqual(before[0]["status"], "likely_correct")
        self.assertEqual(after[0]["status"], "likely_correct")
        self.assertEqual(after[1]["status"], "likely_correct")
        self.assertTrue(info["applied"])
        self.assertLess(info["shifts_ms"][0], 0)
        self.assertGreater(info["mean_match_after"], info["mean_match_before"])
        self.assertIn("boundary_shift_ms", after[0])

    def test_search_never_moves_a_boundary_further_than_the_window(self):
        _, info = self.verify(self.contour(), self.syllables(42), self.with_search(True))
        self.assertTrue(all(abs(v) <= CONFIG["boundary_search"]["max_shift_ms"] for v in info["shifts_ms"]))
        _, narrow = self.verify(self.contour(), self.syllables(42), self.with_search(True, max_shift_ms=20, step_ms=20))
        self.assertTrue(all(abs(v) <= 20 for v in narrow["shifts_ms"]))

    def test_a_correct_alignment_is_left_alone(self):
        _, info = self.verify(self.contour(), self.syllables(30), self.with_search(True))
        self.assertFalse(info["applied"])
        self.assertEqual(info["shifts_ms"], [0])

    def test_no_segmentation_can_rescue_a_genuinely_wrong_tone(self):
        # Both syllables are expected level (T1) but the speaker falls steadily all the way through.
        falling = np.linspace(8, -8, 90)
        contexts = capt_core.expected_tone_contexts(["媽媽"])
        extraction, clean = make_track(falling)
        found, info = tp.search_boundaries(self.syllables(45), contexts, clean, self.with_search(True))
        results = [tp.verify_syllable(s, c, extraction, clean, self.with_search(True), SPEAKER, True)
                   for s, c in zip(found, contexts)]
        self.assertEqual([r["status"] for r in results], ["practice", "practice"])

    def test_outer_boundaries_stay_where_the_aligner_put_them(self):
        extraction, clean = make_track(self.contour())
        found, _ = tp.search_boundaries(self.syllables(42), self.TONES, clean, self.with_search(True))
        self.assertEqual(found[0]["start"], 0.0)
        self.assertAlmostEqual(found[-1]["end"], 90 * STEP)

    def test_search_is_skipped_when_the_alignment_is_not_trusted_or_disabled(self):
        extraction, clean = make_track(self.contour())
        bad = self.syllables(42)
        bad[1]["alignment_ok"] = False
        found, info = tp.search_boundaries(bad, self.TONES, clean, self.with_search(True))
        self.assertIs(found, bad)
        self.assertFalse(info["applied"])
        found, info = tp.search_boundaries(self.syllables(42), self.TONES, clean, self.with_search(False))
        self.assertFalse(info["enabled"])
        self.assertEqual(found[0]["end"], 42 * STEP)

    def test_a_single_syllable_has_no_boundary_to_search(self):
        extraction, clean = make_track(self.contour())
        found, info = tp.search_boundaries(self.syllables(42)[:1], self.TONES[:1], clean, self.with_search(True))
        self.assertEqual(info["shifts_ms"], [])
        self.assertFalse(info["applied"])


class SentenceRulesTests(unittest.TestCase):
    """Sentence-only behaviour: an accepted sandhi alternative and a minimum analysis window for `practice`."""

    @staticmethod
    def verify_with(config, char, st_values, **context_patch):
        extraction, clean = make_track(st_values)
        end = len(st_values) * STEP
        syllable = {"char": char, "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
        context = {**capt_core.expected_tone_contexts([char])[0], **context_patch}
        return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)

    RISING = list(np.linspace(-4, 4, 60))

    def test_a_t2_where_t3_is_expected_is_practice_unless_the_context_also_accepts_t2(self):
        plain = self.verify_with(CONFIG, "你", self.RISING)
        self.assertEqual(plain["status"], "practice")
        accepted = self.verify_with(CONFIG, "你", self.RISING, also_accept_tones=(2,))
        self.assertEqual(accepted["status"], "likely_correct")
        self.assertEqual(accepted["also_accepted_tones"], [2])
        self.assertEqual(accepted["expected_surface_tone"], 3)

    def test_an_alternative_does_not_excuse_a_third_wrong_tone(self):
        falling = list(np.linspace(6, -6, 60))                        # a clear T4 where T3 / T2 are accepted
        self.assertEqual(self.verify_with(CONFIG, "你", falling, also_accept_tones=(2,))["status"], "practice")

    FALLING_WRONG = list(np.linspace(4, -4, 16))                      # 80 ms of falling pitch where T2 is expected

    def test_words_are_untouched_the_window_rule_is_off_by_default(self):
        self.assertEqual(CONFIG["confidence"]["practice_min_window_ms"], 0)
        self.assertEqual(self.verify_with(CONFIG, "麻", self.FALLING_WRONG)["status"], "practice")

    def test_a_short_window_cannot_be_practice_when_the_rule_is_on(self):
        config = tp.load_config(overrides={"confidence": {"practice_min_window_ms": 100}})
        short = self.verify_with(config, "麻", self.FALLING_WRONG)
        self.assertEqual(short["status"], "insufficient_evidence")
        self.assertIn("short_analysis_window", short["reasons"])
        longer = self.verify_with(config, "麻", list(np.linspace(4, -4, 24)))            # 120 ms
        self.assertEqual(longer["status"], "practice")

    def test_the_window_rule_never_hides_an_accepted_syllable_or_creates_a_practice(self):
        config = tp.load_config(overrides={"confidence": {"practice_min_window_ms": 100}})
        ok = self.verify_with(config, "麻", list(np.linspace(-4, 4, 16)))                # correct T2, 80 ms
        self.assertEqual(ok["status"], "likely_correct")


class ClauseFinalHalfThirdTests(unittest.TestCase):
    """Candidate C: sentence practice lets a clause-final T3 be a low fall. Words keep the full-T3 requirement."""
    SENTENCE = tp.load_config(overrides={"tone": {"clause_final_half_t3": True}})
    LOW_FALL = list(np.linspace(2, -12, 60))                           # falls to the lowest point and stops there
    FULL_T3 = list(np.concatenate([np.linspace(0, -9, 30), np.linspace(-9, 3, 30)]))

    @staticmethod
    def verify_with(config, st_values, char="買"):
        extraction, clean = make_track(st_values)
        end = len(st_values) * STEP
        syllable = {"char": char, "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
        context = capt_core.expected_tone_contexts([char])[0]            # a one-character clause: clause-final T3
        return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True), context

    def test_the_default_is_unchanged_a_clause_final_t3_needs_the_full_dip_and_rise(self):
        self.assertFalse(CONFIG["tone"]["clause_final_half_t3"])
        result, context = self.verify_with(CONFIG, self.LOW_FALL)
        self.assertTrue(context["is_clause_final"])
        self.assertFalse(tp.allows_half_third(context, CONFIG))
        self.assertEqual(result["status"], "practice")
        self.assertNotIn("clause_final_half_t3", result)

    def test_with_the_flag_a_low_fall_is_accepted_and_the_verdict_says_why(self):
        result, context = self.verify_with(self.SENTENCE, self.LOW_FALL)
        self.assertTrue(tp.allows_half_third(context, self.SENTENCE))
        self.assertEqual(result["status"], "likely_correct")
        self.assertTrue(result["clause_final_half_t3"])

    def test_a_full_dip_and_rise_is_still_accepted_with_the_flag(self):
        self.assertEqual(self.verify_with(self.SENTENCE, self.FULL_T3)[0]["status"], "likely_correct")

    def test_other_tones_are_still_caught_with_the_flag(self):
        rising = list(np.linspace(-4, 6, 60))                          # T2 where a T3 is expected
        level_high = [6.0] * 60                                       # T1 where a T3 is expected
        for contour in (rising, level_high):
            self.assertEqual(self.verify_with(self.SENTENCE, contour)[0]["status"], "practice")

    def test_known_trade_off_a_t4_falls_to_low_too_and_is_not_told_apart_from_a_half_third(self):
        t4 = list(np.linspace(8, -8, 60))
        self.assertEqual(self.verify_with(self.SENTENCE, t4)[0]["status"], "likely_correct")
        self.assertEqual(self.verify_with(CONFIG, t4)[0]["status"], "practice")       # the word level does tell them apart

    def test_the_flag_changes_nothing_for_other_tones(self):
        for char in ("媽", "麻", "罵"):
            contour = list(np.linspace(-4, 6, 60))
            plain, _ = self.verify_with(CONFIG, contour, char)
            flagged, _ = self.verify_with(self.SENTENCE, contour, char)
            self.assertEqual((plain["status"], plain["tone_match"]), (flagged["status"], flagged["tone_match"]))

    def test_the_boundary_search_ignores_the_flag_so_it_changes_verdicts_and_never_boundaries(self):
        tones = capt_core.expected_tone_contexts(["媽買"])
        st = np.r_[np.zeros(30), np.linspace(2, -12, 30), np.full(30, -12.0)]
        extraction, clean = make_track(st)
        late = 42 * STEP
        syllables = [{"char": "媽", "start": 0.0, "end": late, "phones": [], "nucleus": (0.0, late)},
                     {"char": "買", "start": late, "end": 90 * STEP, "phones": [], "nucleus": (late, 90 * STEP)}]
        plain = tp.search_boundaries(syllables, tones, clean, CONFIG)[1]["shifts_ms"]
        flagged = tp.search_boundaries(syllables, tones, clean, self.SENTENCE)[1]["shifts_ms"]
        self.assertEqual(plain, flagged)


class DurationRuleAndBoundariesTests(unittest.TestCase):
    """A verdict-only rule must change verdicts, never where the boundary search puts the syllable borders."""

    def setUp(self):
        self.tones = capt_core.expected_tone_contexts(["媽罵"])
        self.st = np.r_[np.zeros(30), np.linspace(2, -2, 30), np.full(30, -2.0)]       # T1, then a small-fall T4
        late = 42 * STEP
        self.syllables = [{"char": "媽", "start": 0.0, "end": late, "phones": [], "nucleus": (0.0, late)},
                          {"char": "罵", "start": late, "end": 90 * STEP, "phones": [], "nucleus": (late, 90 * STEP)}]
        rule = ToneFourByDurationTests.RULE
        self.with_rule = {**CONFIG, "tone": {**CONFIG["tone"], "t4_by_duration": rule}}

    def test_the_search_never_sees_the_rule(self):
        seen = []
        original = tp.tone_scores

        def spy(feats, allow_half, config):
            seen.append("t4_by_duration" in config["tone"])
            return original(feats, allow_half, config)

        extraction, clean = make_track(self.st)
        with mock.patch.object(tp, "tone_scores", spy):
            tp.search_boundaries(self.syllables, self.tones, clean, self.with_rule)
        self.assertTrue(seen)
        self.assertFalse(any(seen))

    def test_the_same_boundaries_are_chosen_with_and_without_the_rule(self):
        extraction, clean = make_track(self.st)
        plain = tp.search_boundaries(self.syllables, self.tones, clean, CONFIG)[1]["shifts_ms"]
        ruled = tp.search_boundaries(self.syllables, self.tones, clean, self.with_rule)[1]["shifts_ms"]
        self.assertEqual(plain, ruled)


if __name__ == "__main__":
    unittest.main()
