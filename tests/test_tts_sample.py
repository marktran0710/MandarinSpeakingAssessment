import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import tts_sample  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")


def fake_voice(command, **kwargs):
    """Stands in for PowerShell: finds the output path in the script and writes a small 'wav' there."""
    script = command[-1]
    target = re.search(r"SetOutputToWaveFile\('([^']*)'\)", script).group(1)
    Path(target).write_bytes(b"RIFF" + b"\0" * 4000)
    return mock.Mock(returncode=0)


class CleanTextTests(unittest.TestCase):
    def test_only_chinese_text_and_pause_punctuation_are_spoken(self):
        self.assertEqual(tts_sample.clean_text("  今天天氣很好, 我認識 123 朋友! OK "), "今天天氣很好，我認識朋友！")
        self.assertEqual(tts_sample.clean_text("，，你好。"), "你好。")

    def test_nothing_to_say_or_too_long_is_refused(self):
        for text in ("", "   ", "hello 123", "，。！", None):
            with self.assertRaises(tts_sample.BadText, msg=repr(text)):
                tts_sample.clean_text(text)
        with self.assertRaises(tts_sample.BadText):
            tts_sample.clean_text("好" * (tts_sample.MAX_HAN + 1))
        self.assertEqual(len(tts_sample.clean_text("好" * tts_sample.MAX_HAN)), tts_sample.MAX_HAN)


class SynthesizeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for patcher in (mock.patch.object(tts_sample, "cache_dir", lambda: Path(self.folder.name)),
                        mock.patch.object(tts_sample, "available", lambda: True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_sample_is_made_once_and_then_taken_from_the_cache(self):
        with mock.patch.object(tts_sample.subprocess, "run", side_effect=fake_voice) as run:
            first = tts_sample.synthesize("你好嗎")
            second = tts_sample.synthesize("你好嗎")
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)
            self.assertNotEqual(tts_sample.synthesize("再見"), first)
            self.assertEqual(run.call_count, 2)

    def test_the_text_goes_through_a_file_and_the_voice_is_the_taiwan_one_at_the_configured_speed(self):
        with mock.patch.object(tts_sample.subprocess, "run", side_effect=fake_voice) as run:
            tts_sample.synthesize("你好嗎")
        script = run.call_args[0][0][-1]
        self.assertIn("SelectVoice('Microsoft Hanhan Desktop')", script)
        self.assertIn(f"$s.Rate = {tts_sample.RATE}", script)
        self.assertNotIn("你好嗎", script)                                   # never on the command line
        self.assertLess(tts_sample.RATE, 0)                                  # slower than normal speech

    def test_a_different_speed_is_a_different_sample(self):
        self.assertNotEqual(tts_sample.sample_path("你好", -2), tts_sample.sample_path("你好", 0))

    def test_without_the_voice_a_new_sample_cannot_be_made_but_a_cached_one_still_plays(self):
        with mock.patch.object(tts_sample.subprocess, "run", side_effect=fake_voice):
            cached = tts_sample.synthesize("你好嗎")
        with mock.patch.object(tts_sample, "available", lambda: False):
            self.assertEqual(tts_sample.synthesize("你好嗎"), cached)
            with self.assertRaises(tts_sample.SampleUnavailable):
                tts_sample.synthesize("再見")

    def test_a_voice_that_makes_no_sound_is_reported(self):
        with mock.patch.object(tts_sample.subprocess, "run", return_value=mock.Mock(returncode=0)):
            with self.assertRaises(tts_sample.SampleUnavailable):
                tts_sample.synthesize("你好嗎")

    def test_nothing_leaves_the_machine(self):
        source = (ROOT / "tts_sample.py").read_text(encoding="utf-8")
        for word in ("urllib", "requests", "aiohttp", "socket", "http.client", "edge_tts"):
            self.assertNotIn(word, source, word)


class EndpointTests(unittest.TestCase):
    def test_the_endpoint_answers_with_the_audio_and_maps_the_two_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / "x.wav"
            wav.write_bytes(b"RIFF" + b"\0" * 2000)
            with mock.patch.object(tts_sample, "synthesize", return_value=wav):
                response = main.tts_sample_audio(text="你好")
            self.assertEqual(response.media_type, "audio/wav")
            with mock.patch.object(tts_sample, "synthesize", side_effect=tts_sample.BadText("nothing")):
                with self.assertRaises(main.HTTPException) as caught:
                    main.tts_sample_audio(text="abc")
            self.assertEqual(caught.exception.status_code, 400)
            with mock.patch.object(tts_sample, "synthesize", side_effect=tts_sample.SampleUnavailable("no voice")):
                with self.assertRaises(main.HTTPException) as caught:
                    main.tts_sample_audio(text="你好")
            self.assertEqual(caught.exception.status_code, 503)

    def test_what_is_spoken_is_taiwan_traditional_when_that_keeps_the_text_the_same_length(self):
        with mock.patch.object(tts_sample, "synthesize", return_value=ROOT / "tts_sample.py") as synth:
            main.tts_sample_audio(text="你们好")
            self.assertEqual(synth.call_args[0][0], "你們好")
            main.tts_sample_audio(text="内存")                              # Taiwan wording is longer (記憶體): keep what the learner typed
            self.assertEqual(synth.call_args[0][0], "内存")


class PageTests(unittest.TestCase):
    def test_every_language_has_the_sample_texts(self):
        for key in ("customSampleListen", "customSampleBusy", "customSampleNote", "customSampleError", "customSampleUnavailable"):
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_the_button_is_only_in_my_own_text_and_a_changed_text_hides_the_old_sample(self):
        self.assertIn('document.getElementById("customSampleBlock").hidden = !isCustom;', SOURCE)
        self.assertIn("hideStaleCustomSample(script);", SOURCE)
        self.assertIn("/api/tts-sample?text=", SOURCE)

    @unittest.skipUnless(tts_sample.available(), "the Taiwan Mandarin voice is not installed here")
    def test_the_real_voice_makes_a_slower_sample_than_its_normal_speed(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(tts_sample, "cache_dir", lambda: Path(folder)):
            slow = sf.info(str(tts_sample.synthesize("今天天氣很好。", -2))).duration
            normal = sf.info(str(tts_sample.synthesize("今天天氣很好。", 0))).duration
        self.assertGreater(slow, 0.8)
        self.assertGreater(slow, normal)


if __name__ == "__main__":
    unittest.main()
