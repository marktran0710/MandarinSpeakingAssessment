import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import user_archive as ua

STUDENT = {"id": "a" * 32, "name": "Hau tran", "group_type": "foreigner", "created_at": "2026-10-16T01:00:00+00:00",
           "voice_calibration": {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0,
                                 "calibration_version": "speaker_f0_v1", "quality": {"tokens": ["not copied"]}}}


def payload(sample_id="6-1-5", status="practice"):
    return {"practice_mode": "book", "sample_id": sample_id, "target_script": "我不喜歡跑步",
            "audio_quality": {"passed": True, "reasons": []}, "summary": {"practice": 1},
            "phrases": [{"word": "我不", "status": "uncertain", "strict_status": "practice", "expected_tone_sequence": [3, 4],
                         "feedback_items": [{"type": "tone", "char": "不", "tone": 4}]}],
            "tone_verification": {"syllables": [
                {"index": 0, "character": "我", "expected_surface_tone": 3, "status": "likely_correct", "tone_match": 1.0,
                 "acoustic_quality": 0.9, "start_ms": 0, "end_ms": 300, "duration_ms": 300, "reasons": []},
                {"index": 1, "character": "不", "expected_surface_tone": 4, "status": status, "tone_match": 0.0,
                 "acoustic_quality": 0.9, "start_ms": 300, "end_ms": 600, "duration_ms": 300, "reasons": [],
                 "also_accepted_tones": [2]}]}}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "user data"
        self.audio = Path(self.tmp.name) / "upload.webm"
        self.audio.write_bytes(b"RIFF-fake-audio")
        self.archive = ua.UserArchive(self.base)

    def attempt(self, n=1):
        return {"id": f"{n:08d}" + "b" * 24, "created_at": f"2026-10-16T0{n}:00:00+00:00"}

    def test_a_folder_named_after_the_user_holds_the_audio_the_log_and_the_profile(self):
        folder = self.archive.archive_attempt(STUDENT, self.attempt(), self.audio, "rec.webm", payload())
        self.assertEqual(folder, self.base / "Hau tran")
        audio = [p for p in folder.iterdir() if p.suffix == ".webm"]
        self.assertEqual(len(audio), 1)
        self.assertEqual(audio[0].read_bytes(), b"RIFF-fake-audio")
        self.assertIn("6-1-5", audio[0].name)
        record = json.loads((folder / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((record["student_name"], record["sample_id"], record["audio_file"]), ("Hau tran", "6-1-5", audio[0].name))
        self.assertEqual([s["status"] for s in record["syllables"]], ["likely_correct", "practice"])
        self.assertEqual(record["syllables"][1]["also_accepted_tones"], [2])
        self.assertEqual(record["phrases"][0]["strict_status"], "practice")
        profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(profile["student_id"], STUDENT["id"])
        self.assertEqual(profile["voice_calibration"]["pitch_ceiling"], 450.0)
        self.assertNotIn("quality", profile["voice_calibration"])          # statistics only

    def test_a_second_attempt_is_appended_and_the_csv_opens_in_excel(self):
        self.archive.archive_attempt(STUDENT, self.attempt(1), self.audio, "a.webm", payload())
        self.archive.archive_attempt(STUDENT, self.attempt(2), self.audio, "b.wav", payload("custom"))
        folder = self.base / "Hau tran"
        self.assertEqual(len((folder / "log.jsonl").read_text(encoding="utf-8").splitlines()), 2)
        raw = (folder / "log.csv").read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))                    # one BOM, at the start only
        self.assertEqual(raw.count(b"\xef\xbb\xbf"), 1)
        rows = list(csv.DictReader(open(folder / "log.csv", encoding="utf-8-sig", newline="")))
        self.assertEqual([r["sample_id"] for r in rows], ["6-1-5", "custom"])
        self.assertIn("不(T4):practice", rows[0]["syllables"])
        self.assertEqual(len([p for p in folder.iterdir() if p.suffix in (".webm", ".wav")]), 2)

    def test_two_profiles_with_the_same_name_do_not_share_a_folder(self):
        other = {**STUDENT, "id": "c" * 32, "group_type": "taiwanese"}
        first = self.archive.archive_attempt(STUDENT, self.attempt(1), self.audio, "a.webm", payload())
        second = self.archive.archive_attempt(other, self.attempt(2), self.audio, "a.webm", payload())
        again = self.archive.archive_attempt(STUDENT, self.attempt(3), self.audio, "a.webm", payload())
        self.assertEqual((first.name, second.name, again.name), ("Hau tran", "Hau tran (taiwanese)", "Hau tran"))

    def test_hostile_or_odd_names_stay_inside_the_archive(self):
        for name in ("../../evil", "a/b\\c", "CON", "name.", "  ", "中文 名字", "x" * 300):
            folder = self.archive.archive_attempt({**STUDENT, "name": name, "id": "d" * 32}, self.attempt(), self.audio, "a.webm", payload())
            self.assertIsNotNone(folder, name)
            self.assertEqual(folder.resolve().parent, self.base.resolve(), name)
        self.assertEqual(ua.safe_name("CON"), "_CON")
        self.assertEqual(ua.safe_name("../x"), "_x")
        self.assertEqual(ua.safe_name("中文 名字"), "中文 名字")
        self.assertLessEqual(len(ua.safe_name("x" * 300)), 80)

    def test_it_never_raises_and_can_be_switched_off(self):
        blocked = Path(self.tmp.name) / "a file"
        blocked.write_text("not a folder")
        self.assertIsNone(ua.UserArchive(blocked).archive_attempt(STUDENT, self.attempt(), self.audio, "a.webm", payload()))
        self.assertIsNone(self.archive.archive_attempt(STUDENT, self.attempt(), Path(self.tmp.name) / "missing.webm", "a.webm", payload()))
        off = ua.UserArchive(None)
        self.assertFalse(off.enabled)
        self.assertIsNone(off.archive_attempt(STUDENT, self.attempt(), self.audio, "a.webm", payload()))

    def test_a_csv_locked_by_excel_does_not_lose_the_jsonl_record(self):
        self.archive.archive_attempt(STUDENT, self.attempt(1), self.audio, "a.webm", payload())
        real_open = open

        def locked(path, *args, **kwargs):
            if str(path).endswith("log.csv"):
                raise PermissionError("locked by Excel")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", locked):
            folder = self.archive.archive_attempt(STUDENT, self.attempt(2), self.audio, "a.webm", payload())
        self.assertIsNotNone(folder)
        self.assertEqual(len((folder / "log.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_the_default_folder_and_the_off_switch_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {"MANDARIN_USER_DATA_DIR": ""}):
            self.assertFalse(ua.UserArchive.from_env().enabled)
        with mock.patch.dict(os.environ, {"MANDARIN_USER_DATA_DIR": str(self.base)}):
            self.assertEqual(ua.UserArchive.from_env().base_dir, self.base)
        env = {k: v for k, v in os.environ.items() if k != "MANDARIN_USER_DATA_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(str(ua.UserArchive.from_env().base_dir), ua.DEFAULT_DIR)


if __name__ == "__main__":
    unittest.main()
