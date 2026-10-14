import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capt_core as core


class TestStripErhua(unittest.TestCase):
    def test_drops_erhua_suffix(self):
        self.assertEqual(core.strip_erhua("我有一點兒累"), "我有一點累")
        self.assertEqual(core.strip_erhua("一會兒"), "一會")
        self.assertEqual(core.strip_erhua("玩兒"), "玩")

    def test_keeps_er_when_it_is_a_real_syllable(self):
        for word in ["兒子", "女兒", "兒童", "嬰兒"]:
            self.assertEqual(core.strip_erhua(word), word)


class TestPrepareTarget(unittest.TestCase):
    def test_splits_into_clauses_on_punctuation(self):
        self.assertEqual(
            core.prepare_target("你不要常常在家上網、看電腦，應該去運動。"),
            ["你不要常常在家上網", "看電腦", "應該去運動"],
        )

    def test_drops_erhua_and_non_chinese(self):
        self.assertEqual(core.prepare_target("我覺得有一點兒累，OK！"), ["我覺得有一點累"])


class TestExpectedTones(unittest.TestCase):
    def tones(self, *clauses):
        return core.expected_tones(list(clauses))

    def test_third_tone_sandhi_pair(self):
        self.assertEqual(self.tones("友美"), [2, 3])
        self.assertEqual(self.tones("你想去哪裡"), [2, 3, 4, 2, 3])

    def test_third_tone_run_of_three_becomes_2_2_3(self):
        self.assertEqual(self.tones("我很好"), [2, 2, 3])

    def test_sandhi_is_blocked_by_clause_break(self):
        self.assertEqual(self.tones("好", "我們"), [3, 3, 5])

    def test_no_sandhi_before_neutral_tone(self):
        self.assertEqual(self.tones("我們"), [3, 5])

    def test_bu_before_fourth_tone_is_second(self):
        self.assertEqual(self.tones("不錯"), [2, 4])
        self.assertEqual(self.tones("不是"), [2, 4])

    def test_bu_otherwise_stays_fourth(self):
        self.assertEqual(self.tones("不好"), [4, 3])

    def test_bu_in_verb_not_verb_is_neutral(self):
        self.assertEqual(self.tones("好不好"), [3, 5, 3])

    def test_yi_rules(self):
        self.assertEqual(self.tones("一家"), [4, 1])    # before 1st tone -> 4
        self.assertEqual(self.tones("一起"), [4, 3])    # before 3rd tone -> 4
        self.assertEqual(self.tones("一個"), [2, 4])    # before 4th tone -> 2
        self.assertEqual(self.tones("第一"), [4, 1])    # ordinal / clause-final -> 1

    def test_taiwan_readings_override_pypinyin(self):
        self.assertEqual(self.tones("星期六"), [1, 2, 4])

    def test_neutral_particles_come_from_pypinyin(self):
        self.assertEqual(self.tones("什麼"), [2, 5])

    def test_taiwan_neutral_second_syllables(self):
        self.assertEqual(self.tones("覺得"), [2, 5])
        self.assertEqual(self.tones("衣服"), [1, 5])

    def test_compound_function_words_keep_third_tone_before_a_following_third_tone(self):
        # 所以 / 哪裡 are followed by a prosodic break, so the last syllable does not sandhi into the next word
        self.assertEqual(self.tones("所以我"), [2, 3, 3])
        self.assertEqual(self.tones("哪裡有"), [2, 3, 3])
        self.assertEqual(self.tones("所以"), [2, 3])


class TestMatchSpokenChars(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(core.match_spoken_chars("我們去", "我們去"), ["我", "們", "去"])

    def test_missing_leading_char_does_not_shift_the_rest(self):
        self.assertEqual(core.match_spoken_chars("媽我跟友美", "我跟遊美"), [None, "我", "跟", "遊", "美"])

    def test_missing_tail(self):
        self.assertEqual(core.match_spoken_chars("我想回家休息", "我想回家"), ["我", "想", "回", "家", None, None])

    def test_extra_spoken_char_is_ignored(self):
        self.assertEqual(core.match_spoken_chars("我們去", "我們都去"), ["我", "們", "去"])

    def test_simplified_whisper_output_still_pairs_by_pinyin(self):
        self.assertEqual(core.match_spoken_chars("我們去", "我们去"), ["我", "们", "去"])

    def test_empty_spoken_text(self):
        self.assertEqual(core.match_spoken_chars("我們", ""), [None, None])


class TestFillMissingTimestamps(unittest.TestCase):
    @staticmethod
    def mk(start, end):
        return {"start": start, "end": end, "interpolated": False}

    def test_fills_runs_evenly_between_known_neighbours(self):
        items = [self.mk(None, None), self.mk(0.5, 0.8), self.mk(None, None), self.mk(None, None), self.mk(1.4, 1.7)]
        core.fill_missing_timestamps(items, audio_duration_sec=3.0)
        r = lambda it: (round(it["start"], 3), round(it["end"], 3))
        self.assertEqual(r(items[0]), (0.0, 0.5))
        self.assertEqual(r(items[2]), (0.8, 1.1))
        self.assertEqual(r(items[3]), (1.1, 1.4))
        self.assertEqual([it["interpolated"] for it in items], [True, False, True, True, False])

    def test_trailing_gap_uses_default_duration_and_zero_length_counts_as_missing(self):
        items = [self.mk(1.0, 1.3), self.mk(1.5, 1.5), self.mk(None, None)]
        core.fill_missing_timestamps(items, audio_duration_sec=3.0)
        self.assertAlmostEqual(items[1]["start"], 1.3)
        self.assertAlmostEqual(items[2]["end"], 1.3 + 2 * core.DEFAULT_CHAR_DURATION_SEC)

    def test_noop_when_nothing_is_missing(self):
        items = [self.mk(0.1, 0.4)]
        core.fill_missing_timestamps(items, audio_duration_sec=3.0)
        self.assertEqual(items[0], {"start": 0.1, "end": 0.4, "interpolated": False})


class TestAlignmentCache(unittest.TestCase):
    def test_legacy_cache_is_stale_when_audio_is_newer(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            audio_path = os.path.join(directory, "audio.wav")
            cache_path = os.path.join(directory, "cache.json")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write("{}")

            os.utime(audio_path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
            os.utime(cache_path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
            legacy_entry = {"script": "你好"}
            self.assertTrue(core.alignment_cache_is_current(legacy_entry, "你好", audio_path, cache_path))

            os.utime(audio_path, ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000))
            self.assertFalse(core.alignment_cache_is_current(legacy_entry, "你好", audio_path, cache_path))

    def test_audio_metadata_mismatch_invalidates_new_cache_entry(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            audio_path = os.path.join(directory, "audio.wav")
            cache_path = os.path.join(directory, "cache.json")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write("{}")

            audio_stat = os.stat(audio_path)
            entry = {
                "script": "你好",
                "audio_mtime_ns": audio_stat.st_mtime_ns,
                "audio_size_bytes": audio_stat.st_size,
            }
            self.assertTrue(core.alignment_cache_is_current(entry, "你好", audio_path, cache_path))

            with open(audio_path, "ab") as audio_file:
                audio_file.write(b" updated")
            self.assertFalse(core.alignment_cache_is_current(entry, "你好", audio_path, cache_path))


    def test_malformed_audio_metadata_invalidates_cache_without_raising(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            audio_path = os.path.join(directory, "audio.wav")
            cache_path = os.path.join(directory, "cache.json")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write("{}")

            entry = {"script": "雿末", "audio_mtime_ns": "invalid"}
            self.assertFalse(core.alignment_cache_is_current(entry, "雿末", audio_path, cache_path))


class TestLowConfidence(unittest.TestCase):
    @staticmethod
    def feats(voiced_dur):
        f = [0.0] * core.FEATURE_DIM
        f[core.F_VOICED_DUR] = voiced_dur
        return f

    def test_no_features_is_low_confidence(self):
        self.assertTrue(core.is_low_confidence(None, 0.3))

    def test_very_short_voiced_stretch_is_low_confidence(self):
        self.assertTrue(core.is_low_confidence(self.feats(0.05), 0.3))
        self.assertFalse(core.is_low_confidence(self.feats(0.20), 0.3))

    def test_collapsed_alignment_is_low_confidence(self):
        self.assertTrue(core.is_low_confidence(self.feats(0.20), 0.08))
        self.assertFalse(core.is_low_confidence(self.feats(0.20), 0.30))

    def test_last_char_has_no_alignment_duration_to_judge(self):
        # the last char of an utterance has no following onset, so its CTC span is always tiny
        self.assertFalse(core.is_low_confidence(self.feats(0.20), None))


def synth_utterance(segments, sr=16000):
    """segments: list of (duration, f0_start, f0_end); f0 None = silence. Harmonic tone with 6 harmonics."""
    import numpy as np
    out = []
    for dur, f_a, f_b in segments:
        n = int(dur * sr)
        if f_a is None:
            out.append(np.zeros(n))
            continue
        f = np.linspace(f_a, f_b, n)
        phase = 2 * np.pi * np.cumsum(f) / sr
        out.append(sum(np.sin(h * phase) / h for h in range(1, 7)) * 0.3)
    return np.concatenate(out).astype(np.float32), sr


class TestPitchFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #  0.0-0.2 silence | 0.2-0.6 rising 150->250 | 0.6-0.8 silence | 0.8-1.2 falling 250->150
        #  1.2-1.4 silence | 1.4-1.8 flat high 260   | 1.8-2.0 silence | 2.0-2.4 flat low 140
        x, sr = synth_utterance([(0.2, None, None), (0.4, 150, 250), (0.2, None, None), (0.4, 250, 150),
                                 (0.2, None, None), (0.4, 260, 260), (0.2, None, None), (0.4, 140, 140), (0.2, None, None)])
        cls.track = core.pitch_track(x, sr)

    def feats(self, w0, w1):
        return core.char_features(self.track, w0, w1, [0.0] * core.CONTEXT_DIM)

    def test_rising_tone_has_positive_slope_and_falling_negative(self):
        up, down = self.feats(0.2, 0.6), self.feats(0.8, 1.2)
        self.assertGreater(up[core.F_SLOPE], 0.5)
        self.assertLess(down[core.F_SLOPE], -0.5)

    def test_flat_tones_are_flat_and_register_separates_high_from_low(self):
        high, low = self.feats(1.4, 1.8), self.feats(2.0, 2.4)
        self.assertLess(abs(high[core.F_SLOPE]), 0.3)
        self.assertLess(abs(low[core.F_SLOPE]), 0.3)
        self.assertGreater(high[core.F_LEVEL], low[core.F_LEVEL] + 1.0)

    def test_window_over_silence_has_no_features(self):
        self.assertIsNone(self.feats(0.62, 0.78))

    def test_feature_vector_has_documented_length(self):
        self.assertEqual(len(self.feats(0.2, 0.6)), core.FEATURE_DIM)

    def test_voiced_fraction_is_lower_when_window_includes_silence(self):
        tight, loose = self.feats(0.2, 0.6), self.feats(0.0, 0.8)
        self.assertLess(loose[core.F_VOICED_FRAC], tight[core.F_VOICED_FRAC])


class TestCharWindows(unittest.TestCase):
    def test_window_extends_back_a_bit_and_forward_to_next_onset(self):
        w = core.char_windows([1.0, 1.3, 1.6])
        self.assertAlmostEqual(w[0][0], 1.0 - 0.3 * 0.3)       # first char: median gap stands in for the missing neighbour
        self.assertAlmostEqual(w[0][1], 1.3)
        self.assertAlmostEqual(w[1][0], 1.3 - 0.3 * 0.3)
        self.assertAlmostEqual(w[2][1], 1.9)                   # last char: median gap

    def test_extension_across_a_long_pause_is_capped(self):
        w = core.char_windows([1.0, 3.0])
        self.assertAlmostEqual(w[0][1], 1.0 + core.WINDOW_POST_MAX)
        self.assertAlmostEqual(w[1][0], 3.0 - core.WINDOW_PRE_MAX)


class TestUtteranceFeatures(unittest.TestCase):
    def test_features_per_char_follow_the_aligned_onsets(self):
        # rising tone at 0.2-0.6, falling tone at 0.8-1.2 (see TestPitchFeatures)
        x, sr = synth_utterance([(0.2, None, None), (0.4, 150, 250), (0.2, None, None), (0.4, 250, 150), (0.2, None, None)])
        feats, windows = core.utterance_features(x, sr, [0.2, 0.8], [2, 4], [2])
        self.assertEqual(len(feats), 2)
        self.assertGreater(feats[0][core.F_SLOPE], 0.5)
        self.assertLess(feats[1][core.F_SLOPE], -0.5)
        self.assertEqual(len(windows), 2)

    def test_chars_without_an_onset_get_no_features(self):
        x, sr = synth_utterance([(0.2, None, None), (0.4, 150, 250), (0.2, None, None)])
        feats, windows = core.utterance_features(x, sr, [0.2, None], [2, 4], [2])
        self.assertIsNotNone(feats[0])
        self.assertEqual((feats[1], windows[1]), (None, None))

    def test_optional_pitch_stats_reuses_utterance_track(self):
        x, sr = synth_utterance([(0.2, None, None), (0.4, 150, 250), (0.2, None, None)])
        feats, windows, stats = core.utterance_features(
            x, sr, [0.2], [2], [1], return_pitch_stats=True
        )
        _, _, expected_mean, expected_std = core.pitch_track(x, sr)
        self.assertIsNotNone(feats[0])
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(stats[0], expected_mean)
        self.assertAlmostEqual(stats[1], expected_std)


class TestClauseReconciliation(unittest.TestCase):
    def test_all_unassessable_clause_has_no_accuracy_or_pass_result(self):
        items = [
            {"raw_is_correct_tone": None, "is_correct_tone": None, "low_confidence": True}
            for _ in range(3)
        ]
        result = core.reconcile_clause_assessment(items)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["raw_accuracy"])
        self.assertIsNone(result["is_passed"])
        self.assertEqual(result["n_assessable"], 0)

    def test_passed_clause_only_reconciles_low_confidence_errors(self):
        items = [
            {"raw_is_correct_tone": True, "is_correct_tone": True, "low_confidence": False}
            for _ in range(7)
        ]
        items.extend([
            {"raw_is_correct_tone": False, "is_correct_tone": False, "low_confidence": True},
            {"raw_is_correct_tone": False, "is_correct_tone": False, "low_confidence": True},
            {"raw_is_correct_tone": False, "is_correct_tone": False, "low_confidence": False},
        ])
        result = core.reconcile_clause_assessment(items)
        self.assertEqual(result["raw_accuracy"], 70.0)
        self.assertEqual(result["accuracy"], 90.0)
        self.assertTrue(result["is_passed"])
        self.assertEqual(result["n_reconciled"], 2)
        self.assertTrue(items[7]["is_correct_tone"])
        self.assertTrue(items[7]["reconciled"])
        self.assertFalse(items[9]["is_correct_tone"])


class TestWindowEnergy(unittest.TestCase):
    def test_loud_window_vs_silent_window(self):
        x, sr = synth_utterance([(0.5, None, None), (0.5, 200, 200)])
        self.assertGreater(core.window_energy_ratio(x, sr, 0.5, 1.0), 1.0)    # louder than the utterance average
        self.assertLess(core.window_energy_ratio(x, sr, 0.0, 0.5), 0.01)

    def test_empty_window_is_zero(self):
        x, sr = synth_utterance([(0.5, 200, 200)])
        self.assertEqual(core.window_energy_ratio(x, sr, 5.0, 6.0), 0.0)


class TestLoadAudio(unittest.TestCase):
    def test_returns_mono_float_samples_and_rate(self):
        import tempfile
        from pydub import AudioSegment
        seg = AudioSegment.silent(duration=500, frame_rate=22050).set_channels(2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.wav")
            seg.export(path, format="wav")
            x, sr = core.load_audio_samples(path)
        self.assertEqual(sr, 22050)
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(len(x) / sr, 0.5, places=2)
        self.assertEqual(x.dtype, "float32")


class TestContextFeatures(unittest.TestCase):
    def test_flags_and_neighbour_tones(self):
        tones, lens = [2, 3, 4, 1], [2, 2]
        self.assertEqual(core.context_features(tones, lens, 1), [1.0, 0.0, 0.0, 2.0, 0.0])
        self.assertEqual(core.context_features(tones, lens, 2), [0.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(core.context_features(tones, lens, 3), [1.0, 1.0, 0.0, 4.0, 0.0])
        self.assertEqual(len(core.context_features(tones, lens, 0)), core.CONTEXT_DIM)


if __name__ == "__main__":
    unittest.main()
