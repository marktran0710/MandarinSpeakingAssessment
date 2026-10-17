import re
import sys
import unittest
from pathlib import Path

import numpy as np
import parselmouth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_phrase_feedback import syllable  # noqa: E402

SOUNDS = ROOT / "assets" / "calibration"


class ToneFourIsNeverAskedToBeDrawnOutTests(unittest.TestCase):
    """Tone 4 is the shortest tone (teachers' single words: syllable 355 ms against 410-480 ms for the others)."""

    def test_a_tone_4_that_was_too_short_to_judge_gets_the_short_and_sharp_advice_not_slower(self):
        for reason in ("short_analysis_window", "syllable_too_short", "too_few_voiced_frames"):
            items = ta.phrase_feedback([syllable("你", "likely_correct", 3), syllable("要", "insufficient_evidence", 4, [reason])])
            self.assertEqual(items, [{"type": "retry", "code": "short_fall", "char": "要"}], reason)

    def test_a_single_syllable_tone_4_gets_it_too(self):
        items = ta.phrase_feedback([syllable("要", "insufficient_evidence", 4, ["short_analysis_window"])])
        self.assertEqual(items, [{"type": "retry", "code": "short_fall"}])

    def test_the_other_tones_still_get_slower(self):
        for tone in (1, 2, 3):
            items = ta.phrase_feedback([syllable("你", "likely_correct", 3), syllable("好", "insufficient_evidence", tone, ["short_analysis_window"])])
            self.assertEqual(items[0]["code"], "slower", tone)

    def test_the_advice_follows_the_doubtful_syllable_not_just_any_tone_4_in_the_phrase(self):
        group = [syllable("要", "likely_correct", 4), syllable("好", "insufficient_evidence", 2, ["short_analysis_window"])]
        self.assertEqual(ta.phrase_feedback(group)[0]["code"], "slower")

    def test_the_page_words_the_new_advice_in_every_language_without_asking_to_stretch(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for code, stretch in (("en", "draw it out"), ("zh-TW", "不要拉長"), ("vi", "đừng kéo dài"), ("id", "jangan dipanjangkan"), ("hi", "खींचें नहीं")):
            chunk = re.search(r'\n {16}(?:%s|"%s"): \{.*?retryShortFall: "([^"]*)"' % (code, code), source, re.S).group(1)
            self.assertTrue(stretch in chunk or stretch.replace(" ", " ") in chunk or ("don" in chunk and stretch in chunk), (code, chunk))
        self.assertIn('short_fall: "retryShortFall"', source)


class PaceIgnoresShortTonesTests(unittest.TestCase):
    def test_tone_4_and_light_syllables_never_make_a_reading_too_fast(self):
        def syl(tone, ms, reasons=()):
            return {"duration_ms": ms, "expected_surface_tone": tone, "reasons": list(reasons)}
        quick_fours = [syl(4, 150), syl(4, 160), syl(5, 120), syl(1, 300), syl(2, 310), syl(3, 290), syl(2, 305)]
        self.assertFalse(ta.pace_summary(quick_fours)["too_fast"])                       # without the short tones the median is 300 ms
        self.assertTrue(ta.pace_summary([syl(1, 150), syl(2, 160), syl(3, 170), syl(2, 180)])["too_fast"])
        short_fours = [syl(4, 300, ["short_analysis_window"]) for _ in range(3)] + [syl(1, 300), syl(2, 300), syl(3, 300)]
        self.assertFalse(ta.pace_summary(short_fours)["too_fast"])                       # three too-short tone 4s of six: not "read slower"
        short_ones = [syl(1, 300, ["short_analysis_window"]) for _ in range(3)] + [syl(2, 300), syl(3, 300), syl(2, 300)]
        self.assertTrue(ta.pace_summary(short_ones)["too_fast"])

    def test_production_excludes_tones_4_and_5(self):
        self.assertEqual(ta.get_config()["pace"]["exclude_tones"], [4, 5])


class CalibrationPaceIgnoresTheShortSoundsTests(unittest.TestCase):
    def test_the_natural_shortness_of_ma4_and_ma5_is_not_a_rushed_reading(self):
        config = ta.get_config()
        natural = {"quality": {"tokens": [{"meets_minimum": True}] * 3 + [{"meets_minimum": False}] * 2}}
        self.assertEqual(tp.calibration_pace_hint(natural, config), "")
        rushed = {"quality": {"tokens": [{"meets_minimum": False}] * 2 + [{"meets_minimum": True}] * 3}}
        self.assertEqual(tp.calibration_pace_hint(rushed, config), "slower")                # two of the first three sounds were short


class ReferenceSoundTests(unittest.TestCase):
    def test_the_tone_4_reference_is_the_shortest_of_the_four_tones(self):
        lengths = {tone: parselmouth.Sound(str(SOUNDS / f"ma{tone}.wav")).duration for tone in (1, 2, 3, 4)}
        self.assertEqual(min(lengths, key=lengths.get), 4)
        self.assertLess(lengths[4], 0.45)

    def test_the_calibration_step_says_tone_4_stays_short_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for code in ("en", "zh-TW", "vi", "id", "hi"):
            hint = re.search(r'\n {16}(?:%s|"%s"): \{.*?calStep1Hint: "([^"]*)"' % (code, code), source, re.S).group(1)
            self.assertTrue("m\\u00e0" in hint or "mà" in hint, code)


if __name__ == "__main__":
    unittest.main()
