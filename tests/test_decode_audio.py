import subprocess
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

    def test_a_decode_that_times_out_counts_as_a_failed_try_and_is_retried(self):
        calls = []

        def hangs_once(source, *args, **kwargs):
            calls.append(type(source).__name__)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("ffprobe", 60)
            return FakeSegment()

        with tempfile.TemporaryDirectory() as folder:
            path = fake_recording(folder)
            with mock.patch.object(AudioSegment, "from_file", side_effect=hangs_once), mock.patch.object(audio_prep.time, "sleep"):
                samples, rate = audio_prep.decode_audio(path)
        self.assertEqual(calls, ["str", "BufferedReader"])                  # the path hung; the piped try worked
        self.assertEqual(rate, 16000)

    def test_a_decode_that_always_hangs_ends_with_the_timeout_not_a_forever_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fake_recording(folder)
            with mock.patch.object(AudioSegment, "from_file", side_effect=subprocess.TimeoutExpired("ffprobe", 60)), \
                    mock.patch.object(audio_prep.time, "sleep"):
                with self.assertRaises(subprocess.TimeoutExpired):
                    audio_prep.decode_audio(path)


class HardenedPydubTests(unittest.TestCase):
    """pydub starts ffprobe / ffmpeg with the stdin it inherits and waits for them forever; a detached batch job once hung
    on a perfectly good file. Its child processes get an empty stdin and a time limit."""

    def setUp(self):
        audio_prep._harden_pydub()

    def test_a_child_process_of_pydub_sees_an_empty_stdin(self):
        import pydub.utils

        process = pydub.utils.Popen([sys.executable, "-c", "import sys; print(len(sys.stdin.read()))"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        self.assertEqual(out.strip(), b"0")

    def test_an_explicit_stdin_is_left_alone(self):
        import pydub.utils

        process = pydub.utils.Popen([sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate(b"hello")
        self.assertEqual(out.strip(), b"hello")

    def test_a_child_that_never_finishes_is_killed_at_the_time_limit(self):
        import pydub.utils

        process = pydub.utils.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with mock.patch.object(audio_prep, "DECODE_TIMEOUT_SEC", 0.5):
            with self.assertRaises(subprocess.TimeoutExpired):
                process.communicate()
        self.assertIsNotNone(process.poll())                                # it was killed, not left behind

    def test_hardening_does_not_touch_the_shared_subprocess_module(self):
        import pydub.audio_segment

        self.assertIsNot(pydub.audio_segment.subprocess, subprocess)
        self.assertIs(subprocess.Popen.__mro__[0], subprocess.Popen)
        self.assertNotEqual(subprocess.Popen.__name__, "SafePopen")
        self.assertIs(pydub.audio_segment.subprocess.PIPE, subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
