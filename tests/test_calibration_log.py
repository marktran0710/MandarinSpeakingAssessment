import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi import UploadFile  # noqa: E402

import main  # noqa: E402
from user_archive import UserArchive  # noqa: E402
from practice_store import PracticeStore  # noqa: E402

STUDENT = {"id": "11111111-2222-3333-4444-555555555555", "name": "Cal Log", "group_type": "foreigner"}


def five_sounds_twice():
    """The five reference sounds read twice with short pauses: what a good calibration recording is."""
    pieces = []
    for _ in range(2):
        for tone in range(1, 6):
            data, rate = sf.read(str(ROOT / "assets" / "calibration" / f"ma{tone}.wav"))
            pieces += [data, np.zeros(int(0.3 * rate))]
    return np.concatenate(pieces), rate


def upload(samples, rate, name="voice-calibration.wav"):
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return UploadFile(file=buffer, filename=name)


class CalibrationLogTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.archive = UserArchive(self.folder.name)
        # A store of its own: the accounts and calibrations these tests make must never land in the real practice_data database
        self.store = PracticeStore(Path(self.folder.name) / "practice.sqlite3", Path(self.folder.name) / "recordings")
        patcher = mock.patch.object(main, "practice_store", self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.student = self.store.create_or_get_student(f"cal-log-{id(self)}", "foreigner")

    def lines(self):
        files = list(Path(self.folder.name).glob("*/calibration_log.jsonl"))
        return [json.loads(line) for f in files for line in f.read_text(encoding="utf-8").splitlines()]

    def call(self, file):
        with mock.patch.object(main, "user_archive", self.archive):
            return main.calibrate_student_voice(self.student["id"], file)

    def test_a_good_calibration_is_logged_with_its_measurements(self):
        samples, rate = five_sounds_twice()
        result = self.call(upload(samples, rate))
        self.assertIn("voice_calibration", result)
        (line,) = self.lines()
        self.assertEqual(line["outcome"], "saved")
        self.assertGreater(line["duration_sec"], 5.0)
        self.assertGreater(line["voiced_sec"], 2.0)
        for key in ("ac_cc_agreement", "median_hnr_db", "boundary_hit_rate", "pace_hint", "created_at", "student_id"):
            self.assertIn(key, line)

    def test_a_refused_calibration_is_logged_with_the_reason(self):
        silent = np.zeros(int(6 * 16000))
        response = self.call(upload(silent, 16000))
        self.assertEqual(response.status_code, 422)
        (line,) = self.lines()
        self.assertEqual((line["outcome"], line["reason"]), ("refused", "too_little_voicing"))
        self.assertAlmostEqual(line["duration_sec"], 6.0, places=1)

    def test_a_recording_that_is_too_short_says_so_in_the_log(self):
        samples, rate = five_sounds_twice()
        self.call(upload(samples[: int(3 * rate)], rate))
        (line,) = self.lines()
        self.assertEqual(line["reason"], "too_short")

    def test_a_file_that_cannot_be_decoded_is_logged_as_unreadable(self):
        junk = UploadFile(file=io.BytesIO(b"this is not audio at all" * 50), filename="voice-calibration.webm")
        with self.assertRaises(Exception) as caught:
            self.call(junk)
        self.assertEqual(getattr(caught.exception, "status_code", None), 400)
        (line,) = self.lines()
        self.assertEqual(line["outcome"], "unreadable")
        self.assertGreater(line["bytes"], 0)
        self.assertIn("error", line)

    def test_each_try_is_one_line_in_the_users_own_folder(self):
        samples, rate = five_sounds_twice()
        self.call(upload(samples[: int(3 * rate)], rate))
        self.call(upload(samples, rate))
        self.assertEqual([line["outcome"] for line in self.lines()], ["refused", "saved"])
        self.assertEqual(len(list(Path(self.folder.name).glob("*/calibration_log.jsonl"))), 1)

    def test_a_disabled_archive_writes_nothing_and_breaks_nothing(self):
        archive = UserArchive(None)
        self.assertIsNone(archive.archive_calibration(STUDENT, {"outcome": "saved"}))

    def test_the_log_never_breaks_a_calibration_when_the_folder_is_unusable(self):
        broken = UserArchive(str(Path(self.folder.name) / "a file"))
        Path(self.folder.name, "a file").write_text("not a folder", encoding="utf-8")
        self.assertIsNone(broken.archive_calibration(STUDENT, {"outcome": "saved"}))


class RefusalMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.archive = UserArchive(self.folder.name)
        self.store = PracticeStore(Path(self.folder.name) / "practice.sqlite3", Path(self.folder.name) / "recordings")
        patcher = mock.patch.object(main, "practice_store", self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.student = self.store.create_or_get_student("measured", "foreigner")

    def logged(self, audio, rate):
        with mock.patch.object(main, "user_archive", self.archive):
            response = main.calibrate_student_voice(self.student["id"], upload(audio, rate))
        lines = [json.loads(line) for f in Path(self.folder.name).glob("*/calibration_log.jsonl") for line in f.read_text(encoding="utf-8").splitlines()]
        return response, lines[-1]

    def test_a_refusal_for_too_little_voice_says_how_much_steady_voice_there_was(self):
        samples, rate = five_sounds_twice()
        audio = np.zeros(6 * rate)
        audio[:rate] = samples[:rate]                                      # about one second of voice in six
        response, line = self.logged(audio, rate)
        self.assertEqual(response.status_code, 422)
        self.assertEqual((line["outcome"], line["reason"]), ("refused", "too_little_voicing"))
        self.assertLess(line["voiced_sec"], 1.0)
        self.assertGreater(line["stable_frames"], 0)
        self.assertIn("ac_cc_agreement", line)

    def test_a_silent_recording_logs_zero_voice(self):
        _, line = self.logged(np.zeros(6 * 16000), 16000)
        self.assertEqual((line["voiced_sec"], line["stable_frames"]), (0.0, 0))


class RejectedRequestTests(unittest.TestCase):
    """A calibration upload that arrives without its audio is rejected by FastAPI before our code runs: it must still leave a line in the user's log."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.archive = UserArchive(self.folder.name)
        self.store = PracticeStore(Path(self.folder.name) / "practice.sqlite3", Path(self.folder.name) / "recordings")
        for name, value in (("practice_store", self.store), ("user_archive", self.archive)):
            patcher = mock.patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.student = self.store.create_or_get_student("no-audio", "foreigner")

    def reject(self, path):
        import asyncio
        from fastapi.exceptions import RequestValidationError
        from starlette.requests import Request
        request = Request({"type": "http", "method": "POST", "path": path, "headers": [(b"content-length", b"187")], "query_string": b""})
        error = RequestValidationError([{"type": "missing", "loc": ("body", "file"), "msg": "Field required", "input": None}])
        return asyncio.run(main.explain_rejected_request(request, error))

    def lines(self):
        return [json.loads(line) for f in Path(self.folder.name).glob("*/calibration_log.jsonl") for line in f.read_text(encoding="utf-8").splitlines()]

    def test_the_answer_is_still_the_normal_422_and_the_log_says_no_audio_arrived(self):
        response = self.reject(f"/api/students/{self.student['id']}/voice-calibration")
        self.assertEqual(response.status_code, 422)
        (line,) = self.lines()
        self.assertEqual(line["outcome"], "no_audio_received")
        self.assertEqual(line["content_length"], "187")
        self.assertEqual(line["problems"], ["body.file: missing"])

    def test_other_routes_and_unknown_students_leave_no_calibration_line(self):
        self.assertEqual(self.reject("/api/feedback").status_code, 422)
        self.assertEqual(self.reject("/api/students/00000000000000000000000000000000/voice-calibration").status_code, 422)
        self.assertEqual(self.lines(), [])


class ShortButUsableRecordingTests(unittest.TestCase):
    """Real learners produce 2.1-2.4 s of steady voice for the five sounds twice; 1 s is enough for a profile (the gate was 2.0 s)."""

    def test_the_gate_is_one_second(self):
        cal = main.tone_assessment.get_config()["calibration"]
        self.assertEqual((cal["min_voiced_sec"], cal["preferred_voiced_sec"]), (1.0, 1.0))

    def test_a_recording_with_about_one_to_two_seconds_of_voice_is_saved(self):
        samples, rate = five_sounds_twice()
        saved = {}
        for seconds in (1.0, 1.3, 1.6, 1.9):                              # a growing part of the recording, padded with silence to 6 s
            audio = np.zeros(6 * rate)
            audio[:int(seconds * rate)] = samples[:int(seconds * rate)]
            try:
                saved[seconds] = main.tone_pipeline.calibrate_voice(audio, rate, main.tone_assessment.get_config())["quality"]["voiced_sec"]
            except main.tone_pipeline.CalibrationError:
                saved[seconds] = None
        usable = [v for v in saved.values() if v is not None and 1.0 <= v < 2.0]
        self.assertTrue(usable, saved)                                      # the band the old 2.0 s gate refused is now accepted

    def test_half_a_second_of_voice_is_still_refused(self):
        samples, rate = five_sounds_twice()
        audio = np.zeros(6 * rate)
        audio[:int(0.5 * rate)] = samples[:int(0.5 * rate)]
        with self.assertRaises(main.tone_pipeline.CalibrationError) as caught:
            main.tone_pipeline.calibrate_voice(audio, rate, main.tone_assessment.get_config())
        self.assertEqual(caught.exception.reason, "too_little_voicing")


class QuietMessageTests(unittest.TestCase):
    def test_the_message_says_loudness_is_not_the_problem_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        import re
        for code, word in (("en", "louder doesn"), ("zh-TW", "大聲說並沒有幫助"), ("vi", "nói to hơn không giúp"), ("id", "lebih keras tidak membantu"), ("hi", "ज़ोर से बोलने से मदद नहीं")):
            chunk = re.search(r'\n {16}(?:%s|"%s"): \{.*?calReasonQuiet: "([^"]*)"' % (code, code), source, re.S).group(1)
            self.assertIn(word, chunk, code)
            self.assertNotIn("Speak out loud", chunk)

    def test_the_saved_with_a_reading_tip_variant_counts_as_a_success_in_the_dialog(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('const saved = key === "calibrationSaved" || key === "calibrationSavedSlower";', source)
        self.assertIn('status.classList.toggle("success", saved);', source)
        self.assertIn('else if (saved) calibrationOutcome = "success";', source)
        self.assertNotIn('status.classList.toggle("success", key === "calibrationSaved");', source)

    def test_the_too_little_voice_message_tells_the_learner_to_say_each_sound_longer(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for code, word in (("en", "longer"), ("zh-TW", "長一點"), ("vi", "kéo dài"), ("id", "panjangkan"), ("hi", "लंबा")):
            import re
            chunk = re.search(r'\n {16}(?:%s|"%s"): \{.*?calReasonQuiet: "([^"]*)"' % (code, code), source, re.S).group(1)
            self.assertIn(word, chunk, code)


if __name__ == "__main__":
    unittest.main()
