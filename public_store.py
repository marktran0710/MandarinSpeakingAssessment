"""Account-owned progress for the public learning site."""
import hashlib
import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PublicStore:
    def __init__(self, practice_store):
        self.practice = practice_store
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS public_accounts (
                    id TEXT PRIMARY KEY, google_sub TEXT NOT NULL UNIQUE,
                    student_id TEXT NOT NULL UNIQUE REFERENCES students(id) ON DELETE CASCADE,
                    email TEXT NOT NULL DEFAULT '', display_name TEXT NOT NULL,
                    avatar_url TEXT NOT NULL DEFAULT '', locale TEXT NOT NULL DEFAULT 'vi',
                    study_goal TEXT NOT NULL DEFAULT '', weekly_days INTEGER NOT NULL DEFAULT 3,
                    leaderboard_opt_in INTEGER NOT NULL DEFAULT 0,
                    onboarding_done INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS public_sessions (
                    token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES public_accounts(id) ON DELETE CASCADE,
                    csrf_token TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS public_speaking_profiles (
                    student_id TEXT PRIMARY KEY REFERENCES students(id) ON DELETE CASCADE,
                    speaker_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS comprehension_attempts (
                    id TEXT PRIMARY KEY, student_id TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                    story_id TEXT NOT NULL, skill TEXT NOT NULL CHECK(skill IN ('listen','read')),
                    content_version INTEGER NOT NULL, answers_json TEXT NOT NULL,
                    correct INTEGER NOT NULL, total INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_comprehension_student ON comprehension_attempts(student_id, created_at DESC);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.practice.database_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def account_for_google(self, sub, email, name, avatar):
        if not sub:
            raise ValueError("Missing Google subject")
        with self.db() as db:
            row = db.execute("SELECT * FROM public_accounts WHERE google_sub=?", (sub,)).fetchone()
            if row:
                db.execute("UPDATE public_accounts SET email=?, avatar_url=?, updated_at=? WHERE id=?", (email, avatar, now(), row['id']))
                return dict(db.execute("SELECT * FROM public_accounts WHERE id=?", (row['id'],)).fetchone())
            student_id, account_id = uuid.uuid4().hex, uuid.uuid4().hex
            internal_name = f"account-{student_id}"
            stamp = now()
            db.execute("INSERT INTO students VALUES (?,?,?,?,?,?)", (student_id, internal_name, internal_name, 'foreigner', stamp, stamp))
            db.execute("""INSERT INTO public_accounts
                (id, google_sub, student_id, email, display_name, avatar_url, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)""", (account_id, sub, student_id, email, (name or 'Người học')[:100], avatar, stamp, stamp))
            return dict(db.execute("SELECT * FROM public_accounts WHERE id=?", (account_id,)).fetchone())

    def create_session(self, account_id):
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        expiry = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat(timespec='seconds')
        with self.db() as db:
            db.execute("DELETE FROM public_sessions WHERE expires_at < ?", (now(),))
            db.execute("INSERT INTO public_sessions VALUES (?,?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), account_id, csrf, expiry))
        return token

    def session(self, token):
        if not token:
            return None
        with self.db() as db:
            row = db.execute("""SELECT a.*, s.csrf_token FROM public_sessions s
                JOIN public_accounts a ON a.id=s.account_id WHERE s.token_hash=? AND s.expires_at>?""",
                (hashlib.sha256(token.encode()).hexdigest(), now())).fetchone()
            return dict(row) if row else None

    def revoke(self, token):
        if token:
            with self.db() as db:
                db.execute("DELETE FROM public_sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def speaking_profile(self, student_id):
        with self.db() as db:
            row = db.execute("SELECT speaker_id FROM public_speaking_profiles WHERE student_id=?", (student_id,)).fetchone()
            return row['speaker_id'] if row else None

    def bind_speaking_profile(self, student_id, speaker_id):
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO public_speaking_profiles VALUES (?,?,?)",
                       (student_id, speaker_id, now()))
            row = db.execute("SELECT speaker_id FROM public_speaking_profiles WHERE student_id=?", (student_id,)).fetchone()
            if row is None or row['speaker_id'] != speaker_id:
                raise ValueError("A different speaking profile is already linked to this learner.")
            return row['speaker_id']

    def update_profile(self, student_id, values):
        allowed = ('display_name', 'locale', 'study_goal', 'weekly_days', 'leaderboard_opt_in', 'onboarding_done')
        updates = {k: values[k] for k in allowed if k in values}
        if updates:
            updates['updated_at'] = now()
            with self.db() as db:
                db.execute(f"UPDATE public_accounts SET {', '.join(k+'=?' for k in updates)} WHERE student_id=?", (*updates.values(), student_id))
        with self.db() as db:
            return dict(db.execute("SELECT * FROM public_accounts WHERE student_id=?", (student_id,)).fetchone())

    def save_quiz(self, student_id, story_id, skill, answers, correct, total):
        attempt_id = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO comprehension_attempts VALUES (?,?,?,?,?,?,?,?,?)",
                (attempt_id, student_id, story_id, skill, 1, json.dumps(answers), correct, total, now()))
        return attempt_id

    def quiz_rows(self, student_id):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM comprehension_attempts WHERE student_id=? ORDER BY created_at DESC, rowid DESC", (student_id,))]

    def speaking_rows(self, student_id):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id,sample_id,summary_json,created_at FROM practice_attempts WHERE student_id=? ORDER BY created_at DESC", (student_id,))]

    def owns_audio(self, student_id, attempt_id):
        with self.db() as db:
            return db.execute("SELECT 1 FROM practice_attempts WHERE id=? AND student_id=?", (attempt_id, student_id)).fetchone() is not None

    def delete_account(self, account_id):
        with self.db() as db:
            db.execute("DELETE FROM public_accounts WHERE id=?", (account_id,))
