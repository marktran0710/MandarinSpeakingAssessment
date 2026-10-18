"""Small local SQLite store for speaker profiles and speaking attempts."""

from __future__ import annotations

import json
import re
import os
import shutil
import sqlite3
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "practice_data"
DATA_DIR = Path(os.environ.get("MANDARIN_PRACTICE_DATA_DIR", DEFAULT_DATA_DIR))
DATABASE_PATH = Path(os.environ.get("MANDARIN_PRACTICE_DB", DATA_DIR / "practice.sqlite3"))
RECORDINGS_DIR = Path(os.environ.get("MANDARIN_RECORDINGS_DIR", DATA_DIR / "recordings"))
ALLOWED_GROUPS = {"taiwanese", "foreigner"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Accounts made by tests and tools: never on the leaderboard.
ACCURACY_MIN_LESSONS = 5                      # lessons a learner needs before they appear on the accuracy board
LEADERBOARD_HIDDEN_NAMES = re.compile(r"^(mandarin-speaking|cal-log-|cal log$|lvl-api|tts-check|strict-check|no-audio|measured)", re.IGNORECASE)


def _normalize_name(name):
    return " ".join(unicodedata.normalize("NFKC", name).split()).casefold()


class PracticeStore:
    def __init__(self, database_path=DATABASE_PATH, recordings_dir=RECORDINGS_DIR):
        self.database_path = Path(database_path)
        self.recordings_dir = Path(recordings_dir)
        self.initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    group_type TEXT NOT NULL CHECK (group_type IN ('taiwanese', 'foreigner')),
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    UNIQUE (normalized_name, group_type)
                );

                CREATE TABLE IF NOT EXISTS practice_attempts (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(id),
                    sample_id TEXT NOT NULL,
                    target_script TEXT NOT NULL,
                    recognized_text TEXT,
                    transcript_status TEXT,
                    summary_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_attempts_student_created
                    ON practice_attempts(student_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS student_voice_calibrations (
                    student_id TEXT PRIMARY KEY REFERENCES students(id) ON DELETE CASCADE,
                    calibration_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(id),
                    attempt_id TEXT NOT NULL REFERENCES practice_attempts(id),
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_created
                    ON user_feedback(created_at DESC);
                """
            )
            try:
                connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_students_unique_name ON students(normalized_name)")
            except sqlite3.IntegrityError:
                pass            # an older database that has one name in both groups keeps working: create_or_get_student prefers the chosen group

        self._recover_account_deletions()

    def _recording_path(self, value):
        path = Path(value)
        resolved = path.resolve()
        root = self.recordings_dir.resolve()
        if path.is_symlink() or not resolved.is_relative_to(root) or resolved == root:
            raise ValueError("A saved recording path is outside the recordings directory.")
        return path

    def _recover_account_deletions(self):
        """Finish or roll back interrupted account deletions after a process restart."""
        root = self.recordings_dir.resolve()
        for staging_dir in root.glob(".account-delete-*"):
            if not staging_dir.is_dir() or staging_dir.resolve().parent != root:
                continue
            manifest_path = staging_dir / "manifest.json"
            if not manifest_path.is_file():
                # A crash before writing the manifest cannot have moved any audio.
                try:
                    staging_dir.rmdir()
                except OSError:
                    pass
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                with self._connection() as connection:
                    student_exists = connection.execute(
                        "SELECT 1 FROM students WHERE id = ?",
                        (manifest["student_id"],),
                    ).fetchone() is not None
                for entry in manifest.get("files", []):
                    original = self._recording_path(entry["original"])
                    staged = self._recording_path(entry["staged"])
                    if not staged.is_file():
                        continue
                    if student_exists:
                        if original.exists():
                            raise FileExistsError(f"Cannot restore saved audio to {original}.")
                        original.parent.mkdir(parents=True, exist_ok=True)
                        staged.replace(original)
                    else:
                        staged.unlink()
                manifest_path.unlink(missing_ok=True)
                staging_dir.rmdir()
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                # Leave the directory for a later retry; never discard uncertain data.
                continue

    @staticmethod
    def _student_dict(row):
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "group_type": row["group_type"],
            "created_at": row["created_at"],
            "last_seen_at": row["last_seen_at"],
        }

    def create_or_get_student(self, name, group_type):
        if not isinstance(name, str):
            raise ValueError("Enter a name.")
        name = " ".join(name.split())
        if not name or len(name) > 100:
            raise ValueError("Name must contain 1 to 100 characters.")
        if group_type not in ALLOWED_GROUPS:
            raise ValueError("Choose Taiwanese or Foreigner.")

        normalized_name = _normalize_name(name)
        now = _now()
        with self._connection() as connection:
            # A name belongs to ONE profile, whichever group is chosen: someone who types a name that already exists simply continues as that profile (its group
            # stays as it was). Profiles made before this rule, with the same name in both groups, still resolve: the one of the chosen group first.
            row = connection.execute(
                "SELECT * FROM students WHERE normalized_name = ? ORDER BY (group_type = ?) DESC, created_at LIMIT 1",
                (normalized_name, group_type),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT OR IGNORE INTO students
                       (id, name, normalized_name, group_type, created_at, last_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (uuid.uuid4().hex, name, normalized_name, group_type, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM students WHERE normalized_name = ? ORDER BY (group_type = ?) DESC, created_at LIMIT 1",
                    (normalized_name, group_type),
                ).fetchone()
            connection.execute("UPDATE students SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
            row = connection.execute("SELECT * FROM students WHERE id = ?", (row["id"],)).fetchone()
        student = self._student_dict(row)
        if student is not None:
            student["voice_calibration"] = self.get_voice_calibration(student["id"])
        return student

    def leaderboard(self, viewer_id=None, limit=10):
        """The best-scoring learners. A lesson (a book sentence or a Chapter 5 word) is worth 0-100 points: the BEST attempt at it,
        100 x (Good syllables + half of the Okay ones) / all syllables; the points of all lessons add up. Own text does not count (a single easy
        character would give full points), nor does an attempt without a single Good syllable (silence, a failed alignment: all Okay).
        Returns {"top": [{rank, name, points, lessons, is_me}], "me": {rank, name, points, lessons} or None, "players": n}; `me` is only
        filled for the viewer, and ids are never given out."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT s.id, s.name, a.sample_id, a.summary_json FROM practice_attempts a JOIN students s ON s.id = a.student_id
                   WHERE a.sample_id <> 'custom'"""
            ).fetchall()
            created = {r["id"]: r["created_at"] for r in connection.execute("SELECT id, created_at FROM students")}
        best = {}                                              # (student, lesson) -> best points
        names = {}
        for row in rows:
            if LEADERBOARD_HIDDEN_NAMES.match(row["name"] or ""):
                continue
            try:
                counts = json.loads(row["summary_json"] or "{}").get("syllables") or {}
            except (ValueError, AttributeError):
                continue
            good, okay, practice = (int(counts.get(k) or 0) for k in ("likely_correct", "insufficient_evidence", "practice"))
            total = good + okay + practice
            if good < 1 or total < 1:
                continue
            points = 100.0 * (good + 0.5 * okay) / total
            key = (row["id"], row["sample_id"])
            best[key] = max(best.get(key, 0.0), points)
            names[row["id"]] = row["name"]
        totals = {}
        for (student, _lesson), points in best.items():
            entry = totals.setdefault(student, [0.0, 0])
            entry[0] += points
            entry[1] += 1
        ranked = sorted(totals.items(), key=lambda kv: (-round(kv[1][0]), -kv[1][1], created.get(kv[0], ""), names[kv[0]]))
        results = []
        for student, (points, lessons) in ranked:
            shown = int(round(points))
            rank = 1 + sum(1 for other in results if other["points"] > shown)              # equal points share a rank
            results.append({"student": student, "rank": rank, "name": names[student], "points": shown, "lessons": lessons})
        top = [{"rank": r["rank"], "name": r["name"], "points": r["points"], "lessons": r["lessons"], "is_me": r["student"] == viewer_id}
               for r in results[:limit]]
        me = next(({"rank": r["rank"], "name": r["name"], "points": r["points"], "lessons": r["lessons"]} for r in results if r["student"] == viewer_id), None)
        return {"top": top, "me": me, "players": len(results), "accuracy": self._accuracy_board(totals, names, created, viewer_id, limit)}

    @staticmethod
    def _accuracy_board(totals, names, created, viewer_id, limit):
        """The second board: the average points of a learner's lessons (their best attempt of each, 0-100), so a learner who did many lessons
        and one who did a few compare fairly. It needs ACCURACY_MIN_LESSONS lessons (one perfect sentence must not win); below that the learner
        sees how many more are needed. Equal averages (to one decimal) share a rank."""
        averages = {student: (round(points / lessons, 1), lessons) for student, (points, lessons) in totals.items()}
        eligible = sorted(((student, accuracy, lessons) for student, (accuracy, lessons) in averages.items() if lessons >= ACCURACY_MIN_LESSONS),
                          key=lambda row: (-row[1], -row[2], created.get(row[0], ""), names[row[0]]))
        rows = []
        for student, accuracy, lessons in eligible:
            rank = 1 + sum(1 for other in rows if other["accuracy"] > accuracy)
            rows.append({"student": student, "rank": rank, "name": names[student], "accuracy": accuracy, "lessons": lessons})
        top = [{"rank": r["rank"], "name": r["name"], "accuracy": r["accuracy"], "lessons": r["lessons"], "is_me": r["student"] == viewer_id}
               for r in rows[:limit]]
        me = next(({"rank": r["rank"], "name": r["name"], "accuracy": r["accuracy"], "lessons": r["lessons"]} for r in rows if r["student"] == viewer_id), None)
        if me is None and viewer_id in averages:                                            # has lessons, but not enough of them yet
            accuracy, lessons = averages[viewer_id]
            me = {"rank": None, "name": names[viewer_id], "accuracy": accuracy, "lessons": lessons, "needed": ACCURACY_MIN_LESSONS - lessons}
        return {"top": top, "me": me, "players": len(rows), "min_lessons": ACCURACY_MIN_LESSONS}

    def get_student(self, student_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM students WHERE id = ?", (student_id,)
            ).fetchone()
        student = self._student_dict(row)
        if student is not None:
            student["voice_calibration"] = self.get_voice_calibration(student_id)
        return student

    def get_voice_calibration(self, student_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT calibration_json, created_at FROM student_voice_calibrations WHERE student_id = ?",
                (student_id,),
            ).fetchone()
        if row is None:
            return None
        calibration = json.loads(row["calibration_json"])
        calibration["created_at"] = row["created_at"]
        return calibration

    def save_voice_calibration(self, student_id, calibration):
        if self.get_student(student_id) is None:
            raise ValueError("The selected speaker profile was not found.")
        if not isinstance(calibration, dict) or calibration.get("status") != "calibrated":
            raise ValueError("A valid voice calibration is required.")
        floor = float(calibration.get("pitch_floor_hz", 0))
        ceiling = float(calibration.get("pitch_ceiling_hz", 0))
        if not 40 <= floor < ceiling <= 600:
            raise ValueError("The calibrated pitch range is invalid.")
        created_at = _now()
        saved = {**calibration, "created_at": created_at}
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO student_voice_calibrations (student_id, calibration_json, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(student_id) DO UPDATE SET
                     calibration_json = excluded.calibration_json,
                     created_at = excluded.created_at""",
                (student_id, json.dumps(calibration, ensure_ascii=False), created_at),
            )
        return saved

    def save_attempt(
        self, student_id, sample_id, target_script, audio_source,
        original_filename, result,
    ):
        if self.get_student(student_id) is None:
            raise ValueError("The selected speaker profile was not found.")

        attempt_id = uuid.uuid4().hex
        suffix = Path(original_filename).suffix.lower()
        audio_path = self.recordings_dir / f"{attempt_id}{suffix}"
        transcript = result.get("transcript_check") or {}
        created_at = _now()
        saved_attempt = {
            "id": attempt_id,
            "created_at": created_at,
            "audio_saved": True,
        }
        result["saved_attempt"] = saved_attempt
        try:
            shutil.copyfile(audio_source, audio_path)
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO practice_attempts
                       (id, student_id, sample_id, target_script, recognized_text,
                        transcript_status, summary_json, result_json, audio_path,
                        original_filename, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        attempt_id,
                        student_id,
                        sample_id,
                        target_script,
                        transcript.get("recognized_text"),
                        transcript.get("status"),
                        json.dumps(result.get("summary", {}), ensure_ascii=False),
                        json.dumps(result, ensure_ascii=False),
                        str(audio_path.resolve()),
                        Path(original_filename).name,
                        created_at,
                    ),
                )
        except Exception:
            audio_path.unlink(missing_ok=True)
            raise

        return saved_attempt

    def delete_student_data(self, student_id):
        """Delete a profile and every linked record, with recoverable audio cleanup."""
        connection = self._connect()
        staging_dir = None
        manifest_path = None
        staged_files = []
        committed = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            student = connection.execute(
                "SELECT id, name FROM students WHERE id = ?", (student_id,)
            ).fetchone()
            if student is None:
                connection.rollback()
                return None

            attempts = connection.execute(
                "SELECT audio_path FROM practice_attempts WHERE student_id = ?",
                (student_id,),
            ).fetchall()
            audio_files = []
            seen_paths = set()
            for row in attempts:
                audio_path = self._recording_path(row["audio_path"])
                canonical_path = str(audio_path.resolve())
                if canonical_path in seen_paths:
                    continue
                seen_paths.add(canonical_path)
                if audio_path.is_file():
                    audio_files.append(audio_path)

            if audio_files:
                staging_dir = Path(tempfile.mkdtemp(
                    prefix=".account-delete-", dir=str(self.recordings_dir.resolve())
                ))
                staged_files = [
                    (original, staging_dir / f"{index}-{original.name}")
                    for index, original in enumerate(audio_files)
                ]
                manifest_path = staging_dir / "manifest.json"
                manifest = {
                    "student_id": student_id,
                    "files": [
                        {"original": str(original.resolve()), "staged": str(staged)}
                        for original, staged in staged_files
                    ],
                }
                manifest_tmp = staging_dir / "manifest.tmp"
                manifest_tmp.write_text(json.dumps(manifest), encoding="utf-8")
                manifest_tmp.replace(manifest_path)
                for original, staged in staged_files:
                    original.replace(staged)

            feedback_deleted = connection.execute(
                """DELETE FROM user_feedback
                   WHERE student_id = ? OR attempt_id IN (
                       SELECT id FROM practice_attempts WHERE student_id = ?
                   )""",
                (student_id, student_id),
            ).rowcount
            attempts_deleted = connection.execute(
                "DELETE FROM practice_attempts WHERE student_id = ?", (student_id,)
            ).rowcount
            calibration_deleted = connection.execute(
                "DELETE FROM student_voice_calibrations WHERE student_id = ?",
                (student_id,),
            ).rowcount
            student_deleted = connection.execute(
                "DELETE FROM students WHERE id = ?", (student_id,)
            ).rowcount
            if student_deleted != 1:
                raise RuntimeError("The account changed while it was being deleted.")
            connection.commit()
            committed = True
        except Exception:
            if not committed:
                connection.rollback()
                restore_errors = []
                for original, staged in reversed(staged_files):
                    try:
                        if staged.exists():
                            if original.exists():
                                raise FileExistsError(f"Cannot restore recording to {original}.")
                            staged.replace(original)
                    except OSError as exc:
                        restore_errors.append(exc)
                if staging_dir is not None:
                    try:
                        if manifest_path is not None:
                            manifest_path.unlink(missing_ok=True)
                        (staging_dir / "manifest.tmp").unlink(missing_ok=True)
                        staging_dir.rmdir()
                    except OSError as exc:
                        restore_errors.append(exc)
                if restore_errors:
                    raise RuntimeError(
                        "Account deletion rolled back, but saved audio could not be fully restored."
                    ) from restore_errors[0]
            raise
        finally:
            connection.close()

        pending_audio_cleanup = 0
        cleanup_recovery_pending = False
        if staging_dir is not None:
            for _, staged in staged_files:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    pending_audio_cleanup += 1
            if pending_audio_cleanup == 0:
                try:
                    manifest_path.unlink(missing_ok=True)
                    staging_dir.rmdir()
                except OSError:
                    cleanup_recovery_pending = True

        return {
            "student_id": student_id,
            "name": student["name"],
            "attempts_deleted": attempts_deleted,
            "feedback_deleted": feedback_deleted,
            "voice_calibration_deleted": bool(calibration_deleted),
            "recordings_deleted": len(staged_files) - pending_audio_cleanup,
            "audio_cleanup_pending": pending_audio_cleanup,
            "cleanup_recovery_pending": cleanup_recovery_pending or pending_audio_cleanup > 0,
        }

    def list_attempts(self, student_id, limit=20):
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, sample_id, target_script, recognized_text,
                          transcript_status, summary_json, original_filename, created_at
                   FROM practice_attempts
                   WHERE student_id = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                (student_id, max(1, min(int(limit), 100))),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "sample_id": row["sample_id"],
                "target_script": row["target_script"],
                "recognized_text": row["recognized_text"],
                "transcript_status": row["transcript_status"],
                "summary": json.loads(row["summary_json"]),
                "original_filename": row["original_filename"],
                "created_at": row["created_at"],
                "audio_saved": True,
            }
            for row in rows
        ]

    def save_feedback(self, student_id, attempt_id, rating, comment=""):
        if type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError("Rating must be an integer from 1 to 5.")
        if not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError("Feedback comment must be 2,000 characters or fewer.")

        feedback_id = uuid.uuid4().hex
        created_at = _now()
        with self._connection() as connection:
            attempt = connection.execute(
                "SELECT id FROM practice_attempts WHERE id = ? AND student_id = ?",
                (attempt_id, student_id),
            ).fetchone()
            if attempt is None:
                raise ValueError("The selected practice attempt was not found.")
            connection.execute(
                """INSERT INTO user_feedback
                   (id, student_id, attempt_id, rating, comment, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (feedback_id, student_id, attempt_id, rating, comment.strip(), created_at),
            )
        return {"id": feedback_id, "created_at": created_at}

    def list_admin_dashboard(self):
        with self._connection() as connection:
            students = connection.execute(
                """SELECT id, name, group_type, last_seen_at
                   FROM students ORDER BY last_seen_at DESC, name COLLATE NOCASE"""
            ).fetchall()
            dashboard = []
            for student in students:
                rows = connection.execute(
                    """SELECT a.id, a.sample_id, a.target_script, a.recognized_text,
                              a.transcript_status, a.summary_json, a.result_json,
                              a.original_filename, a.created_at,
                              f.rating AS feedback_rating, f.comment AS feedback_comment,
                              f.created_at AS feedback_created_at
                       FROM practice_attempts a
                       LEFT JOIN user_feedback f ON f.attempt_id = a.id
                       WHERE a.student_id = ?
                       ORDER BY a.created_at DESC, a.id DESC""",
                    (student["id"],),
                ).fetchall()
                attempts = []
                for row in rows:
                    result = json.loads(row["result_json"])
                    attempts.append({
                        "id": row["id"],
                        "sample_id": row["sample_id"],
                        "practice_mode": result.get("practice_mode", "book"),
                        "practice_item": result.get("practice_item"),
                        "target_script": row["target_script"],
                        "recognized_text": row["recognized_text"],
                        "transcript_status": row["transcript_status"],
                        "summary": json.loads(row["summary_json"]),
                        "original_filename": row["original_filename"],
                        "created_at": row["created_at"],
                        "phrases": [
                            {
                                key: item.get(key)
                                for key in (
                                    "word", "annotation", "expected_tone_sequence",
                                    "observed_tone_sequence", "sequence_evidence",
                                    "assessment_note", "status",
                                )
                            }
                            for item in result.get("phrases", [])
                        ],
                        "syllables": [
                            {
                                key: item.get(key)
                                for key in (
                                    "expected_char", "spoken_char", "expected_pinyin",
                                    "spoken_pinyin", "lexical_tone", "expected_tone",
                                    "tone_rule", "tone_context", "strong_tones",
                                    "possible_tones", "tone_features", "assessment_note", "status",
                                    "transcript_status",
                                )
                            }
                            for item in result.get("data", [])
                        ],
                        "words": [
                            {
                                key: item.get(key)
                                for key in (
                                    "word", "annotation", "assessment_note", "status",
                                )
                            }
                            for item in result.get("words", [])
                        ],
                        "feedback": (
                            {
                                "rating": row["feedback_rating"],
                                "comment": row["feedback_comment"],
                                "created_at": row["feedback_created_at"],
                            }
                            if row["feedback_rating"] is not None else None
                        ),
                        "audio_url": f"/api/admin/attempts/{row['id']}/audio",
                    })
                dashboard.append({
                    "id": student["id"],
                    "name": student["name"],
                    "group_type": student["group_type"],
                    "last_seen_at": student["last_seen_at"],
                    "attempts": attempts,
                })
        return dashboard

    def get_attempt_audio_path(self, attempt_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT audio_path FROM practice_attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            return None
        audio_path = Path(row["audio_path"])
        return audio_path if audio_path.is_file() else None


practice_store = PracticeStore()
