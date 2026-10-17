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


def flat_syllable(char, window_ms, config):
    """A perfectly measured FLAT contour on a window of `window_ms` (the whole syllable is the window)."""
    frames = int(round(window_ms / 1000 / STEP))
    extraction, clean = make_track(np.zeros(frames))
    end = frames * STEP
    syllable = {"char": char, "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
    context = capt_core.expected_tone_contexts([char])[0]
    return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True)


class ToneFourFloorTests(unittest.TestCase):
    def setUp(self):
        self.sentence = ta.get_config("sentence")

    def test_production_gives_tone_4_its_own_shorter_floor_in_sentences_only(self):
        self.assertEqual(self.sentence["confidence"]["practice_min_window_ms"], 130)
        self.assertEqual(self.sentence["confidence"]["practice_min_window_ms_by_tone"], {4: 100})
        self.assertNotIn("practice_min_window_ms_by_tone", ta.get_config()["confidence"])             # single words
        self.assertEqual(ta.get_config("sentence", "easy")["confidence"]["practice_min_window_ms_by_tone"], {4: 100})

    def test_a_flat_tone_4_is_flagged_from_100_ms_of_window(self):
        for ms in (100, 110, 125):
            result = flat_syllable("罵", ms, self.sentence)
            self.assertEqual(result["status"], "practice", ms)
        shorter = flat_syllable("罵", 90, self.sentence)
        self.assertEqual(shorter["status"], "insufficient_evidence")
        self.assertIn("short_analysis_window", shorter["reasons"])

    def test_the_other_tones_keep_the_130_ms_floor(self):
        for char in ("麻", "馬"):                                                       # tone 2, tone 3 read flat
            self.assertEqual(flat_syllable(char, 110, self.sentence)["status"], "insufficient_evidence", char)

    def test_a_rising_tone_4_is_no_longer_hidden_by_being_short(self):
        frames = int(round(0.11 / STEP))
        extraction, clean = make_track(np.linspace(-1.5, 1.5, frames))
        end = frames * STEP
        syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
        context = capt_core.expected_tone_contexts(["罵"])[0]
        self.assertEqual(tp.verify_syllable(syllable, context, extraction, clean, self.sentence, SPEAKER, True)["status"], "practice")

    def test_without_the_setting_the_old_floor_applies(self):
        config = copy.deepcopy(self.sentence)
        del config["confidence"]["practice_min_window_ms_by_tone"]
        self.assertEqual(flat_syllable("罵", 110, config)["status"], "insufficient_evidence")

    def test_a_short_flat_tone_4_still_cannot_accuse_in_a_doubtful_recording(self):
        extraction, clean = make_track(np.zeros(22))
        end = 22 * STEP
        syllable = {"char": "罵", "start": 0.0, "end": end, "phones": [], "alignment_ok": True, "nucleus": (0.0, end)}
        context = capt_core.expected_tone_contexts(["罵"])[0]
        result = tp.verify_syllable(syllable, context, extraction, clean, self.sentence, SPEAKER, True, accuse_ok=False)
        self.assertEqual(result["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
