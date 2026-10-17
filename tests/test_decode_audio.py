import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_prep  # noqa: E402
from pydub import AudioSegment  # noqa: E402
from pydub.exceptions import CouldntDecodeError  # noqa: E402


class FakeSegment:
    sample_width = 2

    def set_channels(self, n):
        return self

    def set_frame_rate(self, rate):
        return self

    def get_array_of_samples(self):
        return [0, 16384, -16384, 0]


def fake_recording(folder):
    path = Path(folder) / "recording.webm"
    path.write_bytes(b"not really audio")
    return path


class DecodeByPathTests(unittest.TestCase):
    """FFmpeg must read the file by its path: piping it makes FFmpeg write a fixed-name temp file that two decodes at the
    same moment fight over ("File exists"), and a learner's recording is refused for no reason."""

    def test_the_file_is_given_to_ffmpeg_as_a_path_not_as_an_open_file(self):
        with mock.patch.object(AudioSegment, "from_file", return_value=FakeSegment()) as from_file:
            samples, rate = audio_prep.decode_audio(Path("some") / "recording.webm")
        argument = from_file.call_args_list[0].args[0]
        self.assertIsInstance(argument, str)
        self.assertEqual(from_file.call_count, 1)
        self.assertEqual(rate, 16000)
        np.testing.assert_allclose(samples, [0.0, 0.5, -0.5, 0.0])

    def test_when_the_path_cannot_be_opened_the_piped_way_is_tried_and_a_collision_is_retried(self):
        calls = []

        def flaky(source, *args, **kwargs):
            calls.append(type(source).__name__)
            if len(calls) <= 2:                                              # the path, then the first piped try, both fail
                raise CouldntDecodeError("File exists")
            return FakeSegment()

        with tempfile.TemporaryDirectory() as folder:
            path = fake_recording(folder)
            with mock.patch.object(AudioSegment, "from_file", side_effect=flaky), mock.patch.object(audio_prep.time, "sleep"):
                audio_prep.decode_audio(path)
        self.assertEqual(calls[0], "str")
        self.assertEqual(len(calls), 3)

    def test_a_recording_that_really_cannot_be_decoded_still_fails_after_the_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fake_recording(folder)
            with mock.patch.object(AudioSegment, "from_file", side_effect=CouldntDecodeError("broken")) as from_file, \
                    mock.patch.object(audio_prep.time, "sleep"):
                with self.assertRaises(CouldntDecodeError):
                    audio_prep.decode_audio(path)
        self.assertEqual(from_file.call_count, 5)                            # the path once, then four piped tries

    def test_a_real_non_wav_file_decodes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tone.mp3"
            AudioSegment.silent(duration=500, frame_rate=44100).export(str(path), format="mp3")
            samples, rate = audio_prep.decode_audio(path)
        self.assertEqual(rate, 16000)
        self.assertAlmostEqual(len(samples) / rate, 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
