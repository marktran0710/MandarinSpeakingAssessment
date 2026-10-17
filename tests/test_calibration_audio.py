import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import parselmouth  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import audio_prep  # noqa: E402
import capt_core  # noqa: E402
import main  # noqa: E402
import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402

SOUND_DIR = ROOT / "assets" / "calibration"
SCRIPTS = {1: "媽", 2: "麻", 3: "馬", 4: "罵", 5: "嗎"}
VOICE = {"status": "calibrated", "pitch_floor": 70.0, "pitch_ceiling": 450.0, "f0_median": 206.0}


class ReferenceSoundTests(unittest.TestCase):
    def test_all_five_sounds_exist_and_last_long_enough_to_be_recognised(self):
        for tone in range(1, 6):
            sound = parselmouth.Sound(str(SOUND_DIR / f"ma{tone}.wav"))
            self.assertGreater(sound.duration, 0.25 if tone == 5 else 0.5, tone)       # the full tones are drawn out; the light one is short
            self.assertLess(sound.duration, 1.0, tone)

    def test_the_system_hears_each_sound_as_the_tone_it_is_meant_to_teach(self):
        # the same judgement a learner gets: if the reference were a wrong tone, the page would teach the wrong thing
        config = ta.get_config()
        for tone in (1, 2, 3, 4, 5):
            prepared = audio_prep.prepare_recording(str(SOUND_DIR / f"ma{tone}.wav"), config)
            contexts = capt_core.expected_tone_contexts([SCRIPTS[tone]])
            result = tp.verify_utterance(prepared.samples, prepared.sample_rate, SCRIPTS[tone], VOICE, contexts=contexts, config=config)
            syllable = result["syllables"][0]
            self.assertEqual(syllable["status"], "likely_correct", (tone, syllable["reasons"]))
            if tone != 5:
                self.assertEqual(syllable["tone_match"], 1.0, tone)

    def test_the_sounds_are_not_each_others_tone(self):
        config = ta.get_config()
        for tone, wrong in ((1, 4), (2, 4), (3, 1), (4, 1)):                         # a rise is not a fall, a fall is not level, ...
            prepared = audio_prep.prepare_recording(str(SOUND_DIR / f"ma{tone}.wav"), config)
            contexts = capt_core.expected_tone_contexts([SCRIPTS[wrong]])
            result = tp.verify_utterance(prepared.samples, prepared.sample_rate, SCRIPTS[wrong], VOICE, contexts=contexts, config=config)
            self.assertNotEqual(result["syllables"][0]["status"], "likely_correct", (tone, wrong))


class SoundRouteTests(unittest.TestCase):
    def test_the_route_serves_exactly_the_five_sounds(self):
        for tone in range(1, 6):
            response = main.calibration_sound(tone)
            self.assertEqual(Path(response.path), SOUND_DIR / f"ma{tone}.wav")
            self.assertEqual(response.media_type, "audio/wav")
        for tone in (0, 6, -1):
            with self.assertRaises(HTTPException) as caught:
                main.calibration_sound(tone)
            self.assertEqual(caught.exception.status_code, 404)


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_every_calibration_tile_has_a_play_button_and_there_is_one_for_all_five(self):
        for tone, char in SCRIPTS.items():
            self.assertRegex(self.source, r'class="cal-tile"><button type="button" class="cal-play" data-tone="%d" onclick="playCalibrationSound\(%d\)"' % (tone, tone))
            self.assertIn(f'<span class="han" lang="zh-Hant">{char}</span>', self.source)
        self.assertIn('onclick="playCalibrationSound(0)"', self.source)

    def test_the_step_tells_the_learner_to_go_slowly_and_stretch_each_sound_in_every_language(self):
        for code in ("en", "zh-TW", "vi", "id", "hi"):
            chunk = re.search(r'\n {16}(?:%s|"%s"): \{.*?calStep1Hint: "([^"]*)"' % (code, code), self.source, re.S).group(1)
            self.assertTrue(chr(0x2011) in chunk or '\\u2011' in chunk, code)    # the stretched-sound example 'ma-a-a' with non-breaking hyphens (the source holds the escape text)

    def test_a_rushed_recording_gets_a_tip_in_the_dialog_and_in_the_results(self):
        self.assertIn('calibrationPaceHint ? "calibrationSavedSlower" : "calibrationSaved"', self.source)
        self.assertIn('data.pace?.too_fast && !systemFailure', self.source)


class PaceTests(unittest.TestCase):
    def test_a_calibration_reading_that_was_rushed_is_told_to_go_slower(self):
        config = ta.get_config()
        good = {"quality": {"tokens": [{"meets_minimum": True}] * 5}}
        merged = {"quality": {"tokens": [{"meets_minimum": True}] * 3}}                 # fewer separate sounds than expected
        short = {"quality": {"tokens": [{"meets_minimum": False}] * 2 + [{"meets_minimum": True}] * 3}}
        self.assertEqual(tp.calibration_pace_hint(good, config), "")
        self.assertEqual(tp.calibration_pace_hint(merged, config), "slower")
        self.assertEqual(tp.calibration_pace_hint(short, config), "slower")
        self.assertEqual(tp.calibration_pace_hint({"quality": {"tokens": [{"meets_minimum": False}] + [{"meets_minimum": True}] * 4}}, config), "")

    def test_a_practice_recording_is_fast_when_syllables_are_short_or_many_cannot_be_judged(self):
        def run(durations, reasons=None):
            return ta.pace_summary([{"duration_ms": d, "reasons": (reasons or {}).get(i, [])} for i, d in enumerate(durations)])
        self.assertFalse(run([300, 320, 280, 310])["too_fast"])
        self.assertFalse(run([240, 235, 260, 250])["too_fast"])                         # the fastest teacher sentence is 230 ms per syllable
        self.assertTrue(run([180, 190, 170, 200])["too_fast"])
        too_short = run([300, 300, 300, 300], {1: ["short_analysis_window"], 2: ["syllable_too_short"]})
        self.assertTrue(too_short["too_fast"])
        self.assertEqual(too_short["too_short_share"], 0.5)
        self.assertFalse(run([300, 300, 300, 300], {1: ["short_analysis_window"]})["too_fast"])    # one of four: normal

    def test_a_very_short_text_is_never_called_fast(self):
        self.assertEqual(ta.pace_summary([{"duration_ms": 120, "reasons": []}, {"duration_ms": 110, "reasons": []}]), {"too_fast": False})

    def test_recordings_with_no_timing_are_not_called_fast(self):
        self.assertEqual(ta.pace_summary([{"reasons": ["alignment_failed"]}] * 6), {"too_fast": False})

    def test_the_slower_advice_also_asks_to_stretch_the_sound(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for code, word in (("en", "stretch"), ("zh-TW", "拉長"), ("vi", "kéo dài"), ("id", "panjangkan"), ("hi", "खींच")):
            chunk = re.search(r'\n {16}(?:%s|"%s"): \{.*?retrySlower: "([^"]*)"' % (code, code), source, re.S).group(1)
            self.assertIn(word, chunk, code)


if __name__ == "__main__":
    unittest.main()
