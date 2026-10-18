import copy
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
from test_tone_pipeline import SPEAKER, STEP, make_track  # noqa: E402


def verify_first(text, fall, config, ms=250):
    """The first syllable of `text` (a tone 4) is read as a straight fall whose first-third / last-third difference (what the scorer measures) is `fall` semitones."""
    frames = int(ms / 1000 / STEP)
    extraction, clean = make_track(np.linspace(fall * 0.75, -fall * 0.75, frames))
    end = frames * STEP
    syllable = {"char": text[0], "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
    context = capt_core.expected_tone_contexts([text])[0]
    return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)


class ToneFourBeforeAHighOnsetTests(unittest.TestCase):
    def setUp(self):
        self.sentence = ta.get_config("sentence")
        self.easy = ta.get_config("sentence", "easy")

    def bars(self, config, ms, followed):
        feats = tp.syllable_features(np.linspace(0, ms / 1000, 40), np.linspace(1, -1, 40), 0.0, ms / 1000, syllable_ms=ms, followed_by_t4=followed)
        return tp.t4_min_fall(feats, config["tone"])

    def test_the_bars_before_a_high_onset_are_half_a_semitone_lower_and_never_below_the_floor(self):
        self.assertEqual([round(self.bars(self.sentence, ms, False), 3) for ms in (150, 220, 350)], [1.2, 1.7, 2.2])
        self.assertEqual([round(self.bars(self.sentence, ms, True), 3) for ms in (150, 220, 350)], [0.8, 1.2, 1.7])       # 0.7 -> the 0.8 floor
        self.assertEqual([round(self.bars(self.easy, ms, False), 3) for ms in (150, 220, 350)], [0.38, 0.54, 0.71])
        self.assertEqual([round(self.bars(self.easy, ms, True), 3) for ms in (150, 220, 350)], [0.38, 0.54, 0.71])         # the 0.8 floor holds up a lowered bar, it never raises the bar itself

    def test_the_floor_is_a_setting(self):
        config = copy.deepcopy(self.sentence)
        config["tone"]["t4_before_t4"]["floor_st"] = 1.0
        self.assertEqual(self.bars(config, 150, True), 1.0)                            # 1.2 - 0.5 = 0.7, held up to the floor

    def test_the_relaxation_is_never_stricter_than_the_bar_it_relaxes(self):
        config = copy.deepcopy(self.sentence)
        config["tone"]["t4_before_t4"]["floor_st"] = 1.5                               # above the 1.2 bar of a 150 ms syllable
        self.assertEqual(self.bars(config, 150, True), 1.2)
        easy = copy.deepcopy(self.easy)
        self.assertEqual(self.bars(easy, 120, False), 0.25)                            # the very short class of the easy level ...
        self.assertEqual(self.bars(easy, 120, True), 0.25)                             # ... stays 0.25 before a high onset: the 0.8 floor must not raise it

    def test_a_really_clipped_tone_four_has_its_own_lower_bar(self):
        self.assertEqual([self.bars(self.sentence, ms, False) for ms in (100, 139.9, 140, 179.9)], [0.9, 0.9, 1.2, 1.2])
        self.assertEqual([self.bars(self.easy, ms, False) for ms in (100, 139.9, 140, 179.9)], [0.25, 0.25, 0.38, 0.38])
        self.assertEqual(self.bars(self.sentence, 120, True), 0.8)                     # 0.9 - 0.5 = 0.4 -> the 0.8 floor

    def test_a_tone_four_before_tone_one_or_tone_four_gets_the_lower_bar_one_before_tone_two_does_not(self):
        fall = 1.3                                                                  # between the two bars of a 250 ms syllable (1.2 and 1.7)
        for following, expected in (("罵媽", True), ("罵罵", True), ("罵麻", False), ("罵馬", False)):
            result = verify_first(following, fall, self.sentence)
            self.assertEqual(result["status"] == "likely_correct", expected, (following, result["status"], result["tone_match"]))

    def test_the_old_behaviour_is_still_there_without_next_tones(self):
        config = copy.deepcopy(self.sentence)
        config["tone"]["t4_before_t4"]["next_tones"] = [4]                           # only a following tone 4, as before
        self.assertNotEqual(verify_first("罵媽", 1.3, config)["status"], "likely_correct")
        self.assertEqual(verify_first("罵罵", 1.3, config)["status"], "likely_correct")

    def test_a_flat_line_never_passes_before_a_high_onset(self):
        for text in ("罵媽", "罵罵"):
            self.assertLess(verify_first(text, 0.05, self.easy)["tone_match"], 0.7, text)          # 0.3 st passes now that the easy bar is 0.25; a really flat line (0.05 st) must not

    def test_single_words_are_untouched(self):
        words = ta.get_config()
        self.assertEqual(words["tone"]["t4_min_fall_st"], 3.0)
        self.assertNotIn("t4_before_t4", words["tone"])
        self.assertNotIn("t4_by_duration", words["tone"])
        self.assertEqual(ta.get_config(None, "easy")["tone"]["t4_min_fall_st"], 1.8)


if __name__ == "__main__":
    unittest.main()
