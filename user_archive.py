"""
A second copy of every practice attempt, organised by user, for building an evaluation dataset.

    <base>/<user name>/
        profile.json     name, group, student id, the calibrated pitch range (no calibration audio is kept)
        log.jsonl        one line per attempt, with the verdict of every syllable (machine readable)
        log.csv          the same attempts as a table (opens in Excel; written on a best-effort basis)
        <time>_<sample>_<attempt>.webm / .wav ...   the recording of each attempt

The base folder comes from MANDARIN_USER_DATA_DIR (an empty value switches the archive off). Writing here is a
convenience copy: it never raises, so a locked file or a missing drive cannot break an assessment. The learner's
calibration recording is not archived, because the page promises it is discarded after the pitch statistics are taken.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\audio test\user data"
CSV_COLUMNS = ["created_at", "attempt_id", "practice_mode", "sample_id", "target_script", "audio_file", "audio_quality",
               "phrases", "syllables"]
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_name(text, fallback="user", limit=80):
    """A folder / file name that is valid on Windows and cannot leave its parent folder."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")[:limit].strip(" .")
    if not cleaned or set(cleaned) <= {"_"}:
        return fallback
    if cleaned.split(".")[0].casefold() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


def _syllable_record(s):
    keys = ("index", "character", "expected_surface_tone", "status", "tone_match", "acoustic_quality", "start_ms", "end_ms",
            "duration_ms")
    record = {k: s.get(k) for k in keys}
    record["reasons"] = list(s.get("reasons") or [])
    if s.get("also_accepted_tones"):
        record["also_accepted_tones"] = list(s["also_accepted_tones"])
    return record


def build_record(student, attempt, audio_name, original_filename, payload):
    verification = payload.get("tone_verification") or {}
    quality = payload.get("audio_quality") or {}
    return {
        "created_at": attempt.get("created_at"), "attempt_id": attempt.get("id"),
        "student_id": student.get("id"), "student_name": student.get("name"), "group_type": student.get("group_type"),
        "practice_mode": payload.get("practice_mode"), "sample_id": payload.get("sample_id"),
        "target_script": payload.get("target_script"), "audio_file": audio_name, "original_filename": original_filename,
        "audio_quality": {"passed": quality.get("passed"), "reasons": list(quality.get("reasons") or [])},
        "alignment_error": (verification.get("alignment") or {}).get("error"),      # why no pitch chart: kept so it can be diagnosed
        "summary": payload.get("summary"),
        "phrases": [{"word": p.get("word"), "status": p.get("status"), "strict_status": p.get("strict_status"),
                     "expected_tones": p.get("expected_tone_sequence"), "advice": p.get("feedback_items")}
                    for p in payload.get("phrases") or []],
        "syllables": [_syllable_record(s) for s in verification.get("syllables") or []],
    }


def _csv_row(record):
    quality = record["audio_quality"]
    return {
        "created_at": record["created_at"], "attempt_id": record["attempt_id"], "practice_mode": record["practice_mode"],
        "sample_id": record["sample_id"], "target_script": record["target_script"], "audio_file": record["audio_file"],
        "audio_quality": "ok" if quality["passed"] else "failed: " + ",".join(quality["reasons"]),
        "phrases": " | ".join(f"{p['word']}: {p['status']}" for p in record["phrases"]),
        "syllables": " ".join(f"{s['character']}(T{s['expected_surface_tone']}):{s['status']}" for s in record["syllables"]),
    }


class UserArchive:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir) if base_dir else None
        self._write_lock = Lock()

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("MANDARIN_USER_DATA_DIR", DEFAULT_DIR).strip() or None)

    @property
    def enabled(self):
        return self.base_dir is not None

    def user_folder(self, student):
        """The folder named after the user. A second profile with the same name (another group) gets its own folder."""
        name = safe_name(student.get("name"))
        for candidate in (name, f"{name} ({student.get('group_type')})", f"{name} ({str(student.get('id'))[:8]})"):
            folder = self.base_dir / safe_name(candidate)
            if not folder.exists():
                return folder
            try:
                owner = json.loads((folder / "profile.json").read_text(encoding="utf-8")).get("student_id")
            except (OSError, ValueError):
                owner = None
            if owner == student.get("id"):
                return folder
        return self.base_dir / safe_name(f"{name} ({student.get('id')})")

    def archive_attempt(self, student, attempt, audio_source, original_filename, payload):
        """Copy the recording and log the attempt. Returns the user's folder, or None if nothing was written."""
        with self._write_lock:
            return self._archive_attempt(student, attempt, audio_source, original_filename, payload)

    def _archive_attempt(self, student, attempt, audio_source, original_filename, payload):
        if not self.enabled:
            return None
        try:
            folder = self.user_folder(student)
            if folder.resolve().parent != self.base_dir.resolve():
                raise ValueError("the user folder would be outside the archive")
            folder.mkdir(parents=True, exist_ok=True)
            stamp = re.sub(r"[^0-9TZ]", "", str(attempt.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")))
            suffix = Path(original_filename or "").suffix.lower() or ".webm"
            audio_name = f"{stamp}_{safe_name(payload.get('sample_id'), 'sample', 40)}_{str(attempt.get('id'))[:8]}{suffix}"
            shutil.copyfile(audio_source, folder / audio_name)
            record = build_record(student, attempt, audio_name, original_filename, payload)
            with open(folder / "log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._write_profile(folder, student)
            self._append_csv(folder / "log.csv", record)
            return folder
        except Exception as exc:                                     # never let the copy break an assessment
            print(f"Warning: could not archive the attempt for {student.get('name')!r}: {exc}", file=sys.stderr)
            return None

    def archive_calibration(self, student, outcome):
        """One line per voice-calibration try, no audio: when, what came out and why (calibration_log.jsonl in the user's folder). Calibration
        keeps nothing else, so without this a failure cannot be explained afterwards. Returns the folder, or None if nothing was written."""
        with self._write_lock:
            return self._archive_calibration(student, outcome)

    def _archive_calibration(self, student, outcome):
        if not self.enabled:
            return None
        try:
            folder = self.user_folder(student)
            if folder.resolve().parent != self.base_dir.resolve():
                raise ValueError("the user folder would be outside the archive")
            folder.mkdir(parents=True, exist_ok=True)
            self._write_profile(folder, student)                     # the folder is found again by this file, also for a user with no attempt yet
            record = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "student_id": student.get("id"), **outcome}
            with open(folder / "calibration_log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return folder
        except Exception as exc:                                     # never let the log break a calibration
            print(f"Warning: could not log the calibration of {student.get('name')!r}: {exc}", file=sys.stderr)
            return None

    @staticmethod
    def _write_profile(folder, student):
        calibration = student.get("voice_calibration") or {}
        profile = {"student_id": student.get("id"), "name": student.get("name"), "group_type": student.get("group_type"),
                   "created_at": student.get("created_at"),
                   "voice_calibration": {k: calibration.get(k) for k in
                                         ("status", "pitch_floor", "pitch_ceiling", "f0_median", "calibration_version")}}
        (folder / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def _append_csv(path, record):
        try:                                                           # a copy open in Excel is locked: skip it, log.jsonl has it
            new = not path.exists()
            with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow(_csv_row(record))
        except OSError as exc:
            print(f"Warning: could not append to {path.name}: {exc}", file=sys.stderr)
