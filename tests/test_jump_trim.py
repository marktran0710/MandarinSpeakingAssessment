import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import capt_core  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SPEAKER, STEP, make_track  # noqa: E402

JUMP = 7.0


def trim(st, ms=40):
    return tp.trim_jump_fragments(np.array(st, dtype=float), STEP, JUMP, ms)


class TrimJumpFragmentsTests(unittest.TestCase):
    def test_a_short_tail_behind_a_jump_is_dropped(self):
        st = np.r_[np.linspace(4, -2, 40), np.full(4, -14.0)]                # 4 frames (20 ms) an octave down
        out, dropped = trim(st)
        self.assertEqual(dropped, 4)
        self.assertTrue(np.isnan(out[-4:]).all())
        self.assertEqual(tp.count_octave_jumps(out, JUMP, in_semitones=True), 0)
        np.testing.assert_allclose(out[:40], st[:40])

    def test_a_short_head_before_a_jump_is_dropped(self):
        st = np.r_[np.full(3, 10.0), np.linspace(0, -6, 40)]
        out, dropped = trim(st)
        self.assertEqual(dropped, 3)
        self.assertTrue(np.isnan(out[:3]).all())

    def test_a_fragment_longer_than_the_limit_is_kept_and_the_jump_still_counts(self):
        st = np.r_[np.linspace(4, -2, 40), np.full(12, -14.0)]               # 12 frames = 60 ms
        out, dropped = trim(st, ms=40)
        self.assertEqual(dropped, 0)
        self.assertEqual(tp.count_octave_jumps(out, JUMP, in_semitones=True), 1)

    def test_a_jump_in_the_middle_is_never_trimmed(self):
        st = np.r_[np.full(20, 0.0), np.full(20, -10.0)]
        out, dropped = trim(st)
        self.assertEqual(dropped, 0)
        self.assertEqual(tp.count_octave_jumps(out, JUMP, in_semitones=True), 1)

    def test_a_smooth_fall_is_untouched(self):
        st = np.linspace(6, -8, 50)                                          # 14 st over 250 ms: 0.28 st per frame
        out, dropped = trim(st)
        self.assertEqual(dropped, 0)
        np.testing.assert_allclose(out, st)

    def test_two_flickers_at_the_end_are_both_removed(self):
        st = np.r_[np.linspace(3, -3, 40), np.full(2, -15.0), np.full(2, -3.0), np.full(2, -15.0)]
        out, dropped = trim(st)
        self.assertGreaterEqual(dropped, 2)
        self.assertEqual(tp.count_octave_jumps(out, JUMP, in_semitones=True), 0)

    def test_the_input_array_is_not_modified(self):
        st = np.r_[np.linspace(4, -2, 40), np.full(4, -14.0)]
        before = st.copy()
        trim(st)
        np.testing.assert_allclose(st, before)


def verify_falling_with_a_creaky_tail(trim_ms):
    st = np.r_[np.linspace(5, -4, 36), np.full(4, -16.0)]                    # a clean tone-4 fall, then a 20 ms octave slip
    extraction, clean = make_track(st)
    end = len(st) * STEP
    syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
    context = capt_core.expected_tone_contexts(["罵"])[0]
    config = copy.deepcopy(CONFIG)
    config["sanity"]["jump_fragment_trim_ms"] = trim_ms
    return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)


class VerifySyllableTests(unittest.TestCase):
    def test_without_the_trim_the_jump_gates_the_syllable(self):
        result = verify_falling_with_a_creaky_tail(0)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("octave_jump_in_segment", result["reasons"])

    def test_with_the_trim_the_good_fall_is_judged(self):
        result = verify_falling_with_a_creaky_tail(40)
        self.assertNotIn("octave_jump_in_segment", result["reasons"])
        self.assertEqual(result["status"], "likely_correct")

    def test_the_trim_is_on_in_the_config_file(self):
        import tone_assessment as ta
        self.assertEqual(ta.get_config()["sanity"]["jump_fragment_trim_ms"], 60)           # 40 until 2026-10-19


def verify_wrong_rise_with_a_creaky_tail(trim_ms):
    st = np.r_[np.linspace(-3, 4, 36), np.full(4, -16.0)]                    # a clearly RISING contour read as tone 4, plus a tail jump
    extraction, clean = make_track(st)
    end = len(st) * STEP
    syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
    context = capt_core.expected_tone_contexts(["罵"])[0]
    config = copy.deepcopy(CONFIG)
    config["sanity"]["jump_fragment_trim_ms"] = trim_ms
    return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)


class AcceptButNeverAccuseTests(unittest.TestCase):
    def test_a_trimmed_syllable_that_looks_wrong_is_okay_not_needs_practice(self):
        result = verify_wrong_rise_with_a_creaky_tail(40)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("octave_jump_in_segment", result["reasons"])
        self.assertEqual(result["tone_match"], 0.0)
        self.assertEqual(result["jump_fragment_trimmed_ms"], 20)

    def test_the_same_clean_wrong_syllable_without_a_jump_is_still_needs_practice(self):
        extraction, clean = make_track(np.linspace(-3, 4, 36))
        end = 36 * STEP
        syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
        context = capt_core.expected_tone_contexts(["罵"])[0]
        config = copy.deepcopy(CONFIG)
        config["sanity"]["jump_fragment_trim_ms"] = 40
        self.assertEqual(tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)["status"], "practice")


class SegmentPitchTests(unittest.TestCase):
    def test_segment_pitch_windows_the_track_and_trims_only_when_asked(self):
        st = np.r_[np.linspace(4, -2, 40), np.full(4, -14.0)]
        _, clean = make_track(st)
        config = copy.deepcopy(CONFIG)
        config["sanity"]["jump_fragment_trim_ms"] = 0
        times, plain, dropped = tp.segment_pitch(clean, 0.0, 1.0, config)
        self.assertEqual((int(np.isfinite(plain).sum()), dropped), (44, 0))
        config["sanity"]["jump_fragment_trim_ms"] = 40
        _, trimmed, dropped = tp.segment_pitch(clean, 0.0, 1.0, config)
        self.assertEqual((int(np.isfinite(trimmed).sum()), dropped), (40, 4))


if __name__ == "__main__":
    unittest.main()
