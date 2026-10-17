import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import capt_core  # noqa: E402
import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SPEAKER, STEP, make_track  # noqa: E402


class TrackStabilityTests(unittest.TestCase):
    def test_by_default_one_jump_makes_every_syllable_unstable(self):
        self.assertEqual(tp.track_stability(True, [0, 1, 0]), (False, [False, False, False]))
        self.assertEqual(tp.track_stability(True, [0, 1, 0], "utterance"), (False, [False, False, False]))

    def test_with_syllable_scope_only_the_syllable_with_the_jump_is_unstable(self):
        stable, per_syllable = tp.track_stability(True, [0, 1, 0], "syllable")
        self.assertFalse(stable)                                         # the recording as a whole still counts as unstable
        self.assertEqual(per_syllable, [True, False, True])

    def test_a_track_that_is_unreliable_as_a_whole_makes_every_syllable_unstable_in_both_scopes(self):
        for scope in ("utterance", "syllable"):
            self.assertEqual(tp.track_stability(False, [0, 0, 0], scope), (False, [False, False, False]), scope)

    def test_no_jumps_means_everything_is_stable_in_both_scopes(self):
        for scope in ("utterance", "syllable"):
            self.assertEqual(tp.track_stability(True, [0, 0], scope), (True, [True, True]), scope)


class ProductionTests(unittest.TestCase):
    def test_production_spares_the_syllables_that_have_no_jump_of_their_own(self):
        self.assertEqual(ta.get_config()["confidence"]["unstable_scope"], "syllable")
        self.assertEqual(ta.get_config("sentence")["confidence"]["unstable_scope"], "syllable")


def verify_wrong_tone_four(*, accuse_ok):
    """A clean, perfectly measured RISING contour read where a falling tone 4 is expected: a clear 'Needs practice'."""
    st = np.linspace(-3.0, 4.0, 60)
    extraction, clean = make_track(st)
    end = len(st) * STEP
    syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
    context = capt_core.expected_tone_contexts(["罵"])[0]
    return tp.verify_syllable(syllable, context, extraction, clean, CONFIG, SPEAKER, True, accuse_ok=accuse_ok)


class AccusationNeedsAStableRecordingTests(unittest.TestCase):
    def test_a_clear_wrong_tone_is_needs_practice_when_the_recording_is_stable(self):
        result = verify_wrong_tone_four(accuse_ok=True)
        self.assertEqual(result["status"], "practice")

    def test_the_same_syllable_is_only_okay_when_the_recording_is_not_stable_enough_to_accuse(self):
        result = verify_wrong_tone_four(accuse_ok=False)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("unstable_pitch_track", result["reasons"])
        self.assertEqual(result["tone_match"], 0.0)                      # the shape was still measured and is still reported

    def test_accusing_is_allowed_by_default(self):
        extraction, clean = make_track(np.linspace(-3.0, 4.0, 60))
        end = 60 * STEP
        syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
        context = capt_core.expected_tone_contexts(["罵"])[0]
        self.assertEqual(tp.verify_syllable(syllable, context, extraction, clean, CONFIG, SPEAKER, True)["status"], "practice")


if __name__ == "__main__":
    unittest.main()
