import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from practice_store import PracticeStore


class PracticeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = PracticeStore(root / "practice.sqlite3", root / "recordings")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_profile_is_reused_for_same_name_and_group(self):
        first = self.store.create_or_get_student("  Mei   Lin ", "taiwanese")
        second = self.store.create_or_get_student("mei lin", "taiwanese")

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.store.get_student(first["id"])["name"], "Mei Lin")

    def test_a_name_that_exists_is_continued_whatever_group_is_chosen(self):
        first = self.store.create_or_get_student("Mei Lin", "taiwanese")
        again = self.store.create_or_get_student("MEI  lin", "foreigner")

        self.assertEqual(first["id"], again["id"])                       # no second profile with the same name
        self.assertEqual(again["group_type"], "taiwanese")               # the profile keeps the group it was made with
        with self.store._connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM students").fetchone()[0], 1)

    def test_the_database_itself_refuses_a_second_profile_with_the_same_name(self):
        self.store.create_or_get_student("Mei Lin", "taiwanese")
        with self.store._connection() as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO students (id, name, normalized_name, group_type, created_at, last_seen_at) VALUES ('x', 'MEI LIN', 'mei lin', 'foreigner', 'a', 'a')")

    def test_an_older_database_with_the_name_in_both_groups_still_works_and_prefers_the_chosen_group(self):
        root = Path(self.temp_dir.name)
        legacy = root / "legacy.sqlite3"
        connection = sqlite3.connect(legacy)
        connection.executescript("""
            CREATE TABLE students (id TEXT PRIMARY KEY, name TEXT NOT NULL, normalized_name TEXT NOT NULL,
                group_type TEXT NOT NULL CHECK (group_type IN ('taiwanese', 'foreigner')), created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
                UNIQUE (normalized_name, group_type));
            INSERT INTO students VALUES ('a', 'Lan', 'lan', 'taiwanese', '2026-01-01', '2026-01-01');
            INSERT INTO students VALUES ('b', 'Lan', 'lan', 'foreigner', '2026-01-02', '2026-01-02');
        """)
        connection.commit()
        connection.close()
        store = PracticeStore(legacy, root / "legacy-recordings")                 # initialising must not fail on the duplicate
        self.assertEqual(store.create_or_get_student("lan", "foreigner")["id"], "b")
        self.assertEqual(store.create_or_get_student("lan", "taiwanese")["id"], "a")
        self.assertEqual(store.create_or_get_student("new person", "taiwanese")["name"], "new person")

    def test_attempt_saves_original_audio_and_result_history(self):
        profile = self.store.create_or_get_student("Mei Lin", "taiwanese")
        root = Path(self.temp_dir.name)
        source = root / "recording.webm"
        source.write_bytes(b"original audio bytes")
        result = {
            "summary": {"n_phrases": 1, "n_assessable": 1},
            "transcript_check": {
                "status": "partial_match",
                "recognized_text": "你好",
            },
            "phrases": [{"annotation": "phrase", "expected_tone_sequence": [2, 3], "observed_tone_sequence": [2, 3], "sequence_evidence": "complete_match", "status": "likely_correct"}],
        }

        saved = self.store.save_attempt(
            student_id=profile["id"],
            sample_id="5-1-1",
            target_script="你好嗎",
            audio_source=source,
            original_filename="my-recording.webm",
            result=result,
        )
        source.unlink()
        attempts = self.store.list_attempts(profile["id"])

        self.assertTrue(saved["audio_saved"])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["recognized_text"], "你好")
        self.assertEqual(attempts[0]["summary"]["n_assessable"], 1)
        audio_path = self.store.recordings_dir / f"{saved['id']}.webm"
        self.assertEqual(audio_path.read_bytes(), b"original audio bytes")

        with self.store._connection() as connection:
            stored_json = connection.execute(
                "SELECT result_json FROM practice_attempts WHERE id = ?",
                (saved["id"],),
            ).fetchone()[0]
        self.assertEqual(json.loads(stored_json), result)

    def test_unknown_profile_and_invalid_group_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create_or_get_student("Test", "unknown")
        with self.assertRaises(ValueError):
            self.store.save_attempt(
                "missing",
                "5-1-1",
                "你好",
                Path(self.temp_dir.name) / "missing.wav",
                "missing.wav",
                {},
            )

    def test_voice_calibration_is_saved_and_returned_with_profile(self):
        profile = self.store.create_or_get_student("Mei Lin", "taiwanese")
        calibration = {
            "status": "calibrated",
            "method": "praat_ac_cc_wide_pass_v1",
            "f0_hz": {"p10": 105.0, "p50": 145.0, "p90": 190.0, "p95": 205.0, "baseline": 145.0},
            "pitch_floor_hz": 78.75,
            "pitch_ceiling_hz": 287.0,
        }

        saved = self.store.save_voice_calibration(profile["id"], calibration)
        restored = self.store.get_student(profile["id"])

        self.assertEqual(saved["status"], "calibrated")
        self.assertEqual(restored["voice_calibration"]["f0_hz"], calibration["f0_hz"])
        self.assertEqual(restored["voice_calibration"]["pitch_floor_hz"], calibration["pitch_floor_hz"])
        self.assertTrue(restored["voice_calibration"]["created_at"])

    def test_feedback_is_saved_with_student_and_attempt(self):
        profile = self.store.create_or_get_student("Mei Lin", "taiwanese")
        source = Path(self.temp_dir.name) / "recording.webm"
        source.write_bytes(b"audio")
        attempt = self.store.save_attempt(
            student_id=profile["id"],
            sample_id="5-1-1",
            target_script="hello",
            audio_source=source,
            original_filename="recording.webm",
            result={"summary": {}, "transcript_check": {}, "phrases": [{"expected_tone_sequence": [2, 3]}]},
        )

        feedback = self.store.save_feedback(
            student_id=profile["id"],
            attempt_id=attempt["id"],
            rating=4,
            comment="The pitch chart was helpful.",
        )

        self.assertTrue(feedback["id"])
        with self.store._connection() as connection:
            row = connection.execute(
                "SELECT student_id, attempt_id, rating, comment FROM user_feedback WHERE id = ?",
                (feedback["id"],),
            ).fetchone()
        self.assertEqual(row["student_id"], profile["id"])
        self.assertEqual(row["attempt_id"], attempt["id"])
        self.assertEqual(row["rating"], 4)
        self.assertEqual(row["comment"], "The pitch chart was helpful.")
        dashboard = self.store.list_admin_dashboard()
        self.assertEqual(dashboard[0]["name"], "Mei Lin")
        self.assertEqual(dashboard[0]["attempts"][0]["id"], attempt["id"])
        self.assertEqual(dashboard[0]["attempts"][0]["feedback"]["rating"], 4)
        self.assertEqual(dashboard[0]["attempts"][0]["phrases"][0]["expected_tone_sequence"], [2, 3])
        self.assertTrue(self.store.get_attempt_audio_path(attempt["id"]).is_file())

        with self.assertRaises(ValueError):
            self.store.save_feedback(profile["id"], attempt["id"], 6)

    def test_delete_student_removes_profile_attempt_feedback_calibration_and_audio(self):
        profile = self.store.create_or_get_student("Mei Lin", "taiwanese")
        source = Path(self.temp_dir.name) / "recording.webm"
        source.write_bytes(b"audio to remove")
        attempt = self.store.save_attempt(
            profile["id"], "5-1-1", "友美", source, "recording.webm",
            {"summary": {}, "transcript_check": {}, "data": []},
        )
        self.store.save_feedback(profile["id"], attempt["id"], 4, "Useful")
        self.store.save_feedback(profile["id"], attempt["id"], 3, "Another response")
        self.store.save_voice_calibration(profile["id"], {
            "status": "calibrated",
            "method": "praat_ac_cc_wide_pass_v1",
            "f0_hz": {"p10": 100, "p50": 145, "p90": 190, "p95": 205, "baseline": 145},
            "pitch_floor_hz": 75,
            "pitch_ceiling_hz": 285,
        })
        audio_path = self.store.get_attempt_audio_path(attempt["id"])

        deleted = self.store.delete_student_data(profile["id"])

        self.assertEqual(deleted["attempts_deleted"], 1)
        self.assertEqual(deleted["feedback_deleted"], 2)
        self.assertTrue(deleted["voice_calibration_deleted"])
        self.assertEqual(deleted["recordings_deleted"], 1)
        self.assertEqual(deleted["audio_cleanup_pending"], 0)
        self.assertFalse(audio_path.exists())
        self.assertIsNone(self.store.get_student(profile["id"]))
        self.assertIsNone(self.store.get_attempt_audio_path(attempt["id"]))
        self.assertEqual(self.store.list_admin_dashboard(), [])
        with self.store._connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM practice_attempts").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM user_feedback").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM student_voice_calibrations").fetchone()[0], 0)

    def test_delete_student_returns_none_for_unknown_account(self):
        self.assertIsNone(self.store.delete_student_data("missing-account"))

    def test_delete_student_rolls_back_database_and_restores_audio_on_constraint_error(self):
        profile = self.store.create_or_get_student("Mei Lin", "taiwanese")
        source = Path(self.temp_dir.name) / "recording.webm"
        source.write_bytes(b"audio to preserve")
        attempt = self.store.save_attempt(
            profile["id"], "5-1-1", "友美", source, "recording.webm",
            {"summary": {}, "transcript_check": {}, "data": []},
        )
        audio_path = self.store.get_attempt_audio_path(attempt["id"])
        with self.store._connection() as connection:
            connection.execute(
                "CREATE TABLE protected_rows (student_id TEXT REFERENCES students(id))"
            )
            connection.execute("INSERT INTO protected_rows (student_id) VALUES (?)", (profile["id"],))

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.delete_student_data(profile["id"])

        self.assertIsNotNone(self.store.get_student(profile["id"]))
        self.assertEqual(len(self.store.list_attempts(profile["id"])), 1)
        self.assertTrue(audio_path.is_file())
        self.assertEqual(audio_path.read_bytes(), b"audio to preserve")
        self.assertFalse(list(self.store.recordings_dir.glob(".account-delete-*")))

    def test_startup_restores_audio_if_process_stopped_before_database_delete(self):
        profile = self.store.create_or_get_student("Mei Lin", "taiwanese")
        source = Path(self.temp_dir.name) / "recording.webm"
        source.write_bytes(b"audio to restore")
        attempt = self.store.save_attempt(
            profile["id"], "5-1-1", "友美", source, "recording.webm",
            {"summary": {}, "transcript_check": {}, "data": []},
        )
        audio_path = self.store.get_attempt_audio_path(attempt["id"])
        staging_dir = self.store.recordings_dir / ".account-delete-interrupted"
        staging_dir.mkdir()
        staged_path = staging_dir / f"0-{audio_path.name}"
        (staging_dir / "manifest.json").write_text(json.dumps({
            "student_id": profile["id"],
            "files": [{"original": str(audio_path.resolve()), "staged": str(staged_path)}],
        }), encoding="utf-8")
        audio_path.replace(staged_path)

        restarted_store = PracticeStore(self.store.database_path, self.store.recordings_dir)

        self.assertEqual(restarted_store.get_attempt_audio_path(attempt["id"]), audio_path)
        self.assertEqual(audio_path.read_bytes(), b"audio to restore")
        self.assertFalse(staging_dir.exists())


if __name__ == "__main__":
    unittest.main()
