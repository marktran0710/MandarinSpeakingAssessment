import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SR, tone_audio  # noqa: E402

CAL = {"status": "calibrated", "pitch_floor": 90, "pitch_ceiling": 450, "f0_median": 200}


def three_spoken_two_in_the_script(chars, config):
    """Three flat syllables were spoken; the script (and so the alignment) accounts for the first two: a third of the speech is outside."""
    audio, spans = tone_audio([[200] * 4] * 3)
    syllables = [{"char": c, "start": spans[i][0], "end": spans[i][1], "phones": [], "nucleus": None} for i, c in enumerate(chars)]
    return tp.verify_utterance(audio, SR, chars, CAL, alignment=syllables, config=config)


class DoubtfulAlignmentTests(unittest.TestCase):
    def test_production_has_a_doubtful_band_between_the_two_limits(self):
        alignment = ta.get_config()["alignment"]
        self.assertEqual((alignment["max_unaligned_voiced_fraction"], alignment["hard_unaligned_voiced_fraction"]), (0.25, 0.40))

    def test_a_third_of_the_speech_outside_is_doubtful_not_untrusted(self):
        result = three_spoken_two_in_the_script("媽媽", CONFIG)
        diag = result["f0_extraction"]
        self.assertTrue(0.25 < diag["voiced_outside_syllables"] <= 0.40, diag["voiced_outside_syllables"])
        self.assertTrue(diag["alignment_doubtful"])
        self.assertFalse(diag["speech_outside_alignment"])
        self.assertEqual([s["status"] for s in result["syllables"]], ["likely_correct", "likely_correct"])     # judged as usual
        self.assertNotIn("bad_alignment", result["syllables"][0]["reasons"])

    def test_with_only_the_old_single_limit_the_same_recording_is_not_judged_at_all(self):
        config = copy.deepcopy(CONFIG)
        config["alignment"]["hard_unaligned_voiced_fraction"] = config["alignment"]["max_unaligned_voiced_fraction"]
        result = three_spoken_two_in_the_script("媽媽", config)
        self.assertTrue(result["f0_extraction"]["speech_outside_alignment"])
        self.assertEqual([s["status"] for s in result["syllables"]], ["insufficient_evidence"] * 2)
        self.assertIn("bad_alignment", result["syllables"][0]["reasons"])

    def test_in_the_doubtful_band_nobody_is_accused(self):
        # flat syllables read where a tone 4 is expected are a clear mistake, but the recording is doubtful: Okay, with the reason
        result = three_spoken_two_in_the_script("罵罵", CONFIG)
        self.assertTrue(result["f0_extraction"]["alignment_doubtful"])
        self.assertEqual({s["status"] for s in result["syllables"]}, {"insufficient_evidence"})
        self.assertTrue(all("bad_alignment" in s["reasons"] for s in result["syllables"]))
        self.assertTrue(all(s["tone_match"] == 0.0 for s in result["syllables"]))                              # the shape is still measured

    def test_the_same_mistake_in_a_fully_explained_recording_is_still_needs_practice(self):
        audio, spans = tone_audio([[200] * 4, [200] * 4])
        syllables = [{"char": "罵", "start": a, "end": b, "phones": [], "nucleus": None} for a, b in spans]
        result = tp.verify_utterance(audio, SR, "罵罵", CAL, alignment=syllables, config=CONFIG)
        self.assertFalse(result["f0_extraction"]["alignment_doubtful"])
        self.assertEqual({s["status"] for s in result["syllables"]}, {"practice"})

    def test_more_than_the_hard_limit_outside_is_still_not_trusted(self):
        audio, spans = tone_audio([[200] * 4, [200] * 4])                                                     # half of the speech is outside
        syllables = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None}]
        result = tp.verify_utterance(audio, SR, "媽", CAL, alignment=syllables, config=CONFIG)
        self.assertTrue(result["f0_extraction"]["speech_outside_alignment"])
        self.assertEqual(result["syllables"][0]["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
