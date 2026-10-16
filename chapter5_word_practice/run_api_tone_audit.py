"""Run the existing pronunciation API over the Chapter 5 word audio set."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import practice_store


MANIFEST = ROOT / "items.json"
OUTPUT_DIR = Path(os.environ.get("CHAPTER5_TONE_AUDIT_OUTPUT_DIR", ROOT / "tone_audit_results")).resolve()
API_ROOT = os.environ.get("CHAPTER5_TONE_AUDIT_API_ROOT", "http://127.0.0.1:8000")
RAW_JSONL = OUTPUT_DIR / "api_responses.jsonl"
DETAIL_CSV = OUTPUT_DIR / "item_summary.csv"
REPORT_MD = OUTPUT_DIR / "report.md"


def persist_report(rows, request_started):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "word_key", "word_id", "section", "item_type", "script", "pinyin",
        "english_meaning", "audio_file", "http_status", "result_status",
        "phrase_statuses", "expected_tone_sequences", "observed_tone_sequences",
        "transcript_status", "recognized_text", "matched_syllables", "total_syllables",
        "aligned_script_variant", "alignment_status", "alignment_method",
        "aligned_characters", "alignment_candidates", "phrase_window_sec",
        "trimmed_voiced_coverage", "maximum_internal_unvoiced_gap_sec",
        "n_phrases", "n_with_audio_evidence", "error",
    ]
    with DETAIL_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row.get("http_status") == 200 and not row.get("error")]
    normalize_status = lambda value: "good" if value == "likely_correct" else value
    status_counts = {}
    phrase_status_counts = {}
    item_type_counts = {}
    for row in completed:
        statuses = [normalize_status(value) for value in json.loads(row.get("phrase_statuses") or "[]")]
        for status in statuses:
            phrase_status_counts[status] = phrase_status_counts.get(status, 0) + 1
        if not statuses:
            clip_status = row.get("result_status") or "no_phrase_result"
        elif all(status == "good" for status in statuses):
            clip_status = "good"
        elif any(status == "practice" for status in statuses):
            clip_status = "practice"
        elif any(status in {"text_mismatch", "transcript_unchecked"} for status in statuses):
            clip_status = "not_fully_checked"
        else:
            clip_status = "uncertain"
        status_counts[clip_status] = status_counts.get(clip_status, 0) + 1
        kind = row.get("item_type") or "unknown"
        type_counts = item_type_counts.setdefault(kind, {})
        type_counts[clip_status] = type_counts.get(clip_status, 0) + 1

    total = len(rows)
    observed = sum(status_counts.values())
    all_good = status_counts.get("good", 0) == total and observed == total
    good_count = status_counts.get("good", 0)
    good_percent = 100.0 * good_count / total if total else 0.0
    lines = [
        "# Kiểm tra thanh điệu audio Chapter 5",
        "",
        f"- Thời gian chạy (UTC): {request_started}",
        f"- Nguồn: 53 audio Chapter 5 đã tách từ ZIP; gồm 49 mục từ vựng và 4 cụm từ.",
        f"- Pipeline: `POST /api/assess-pronunciation`, mode `praat_phrase_tone_direct_v1`; Praat lấy F0 toàn clip; WhisperX căn transcript cho từng chữ, giữ nguyên ngưỡng chấm.",
        f"- Kết quả API: {len(completed)}/{total} thành công.",
        f"- Trạng thái theo mục: `{json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- Tách theo loại mục: `{json.dumps(item_type_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- Có thể xác nhận 100% mục đúng theo hệ thống: **{'có' if all_good else 'không'}**; hệ thống trả `good` cho {good_count}/{total} mục ({good_percent:.1f}%).",
        "",
        "`good` là kết quả theo luật F0 hiện tại, không phải xác suất đã hiệu chỉnh hay nhãn kiểm chứng độc lập. `practice` là contour lệch rõ theo luật; `uncertain` được giữ riêng và không tính là đúng.",
        "Các audio trong bộ này là giọng giáo viên đọc mẫu. Kết quả chỉ kiểm tra cách hệ thống xử lý các clip mẫu này, chưa đo độ chính xác trên học sinh. Audio không được dùng làm reference để chấm (`reference_audio_used: false`).",
        "Cửa sổ cụm lấy từ âm tiết đầu đến cuối theo forced alignment. F0 mất trong khoảng nội bộ không được nội suy. Alignment yếu hoặc F0 không đủ vẫn trả `uncertain`.",
        "",
        f"Bảng từng mục: `{DETAIL_CSV.name}`. JSON response đầy đủ: `{RAW_JSONL.name}`.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_student(student_id: str) -> None:
    db_path = practice_store.DATABASE_PATH
    recordings = practice_store.RECORDINGS_DIR.resolve()
    audio_paths = []
    with sqlite3.connect(db_path, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        student = connection.execute("SELECT id FROM students WHERE id = ?", (student_id,)).fetchone()
        if not student:
            return
        attempts = connection.execute(
            "SELECT id, audio_path FROM practice_attempts WHERE student_id = ?", (student_id,)
        ).fetchall()
        audio_paths = [Path(row[1]) for row in attempts]
        connection.execute("DELETE FROM user_feedback WHERE student_id = ?", (student_id,))
        connection.execute("DELETE FROM practice_attempts WHERE student_id = ?", (student_id,))
        connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
    for audio_path in audio_paths:
        try:
            resolved = audio_path.resolve()
            if resolved.is_relative_to(recordings):
                resolved.unlink(missing_ok=True)
        except (OSError, RuntimeError):
            pass


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    if manifest.get("chapter") != 5 or len(items) != 53:
        raise RuntimeError(f"Unexpected Chapter 5 input set: chapter={manifest.get('chapter')}, count={len(items)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_JSONL.write_text("", encoding="utf-8")
    request_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session = requests.Session()
    session.trust_env = False
    nonce = uuid.uuid4().hex[:10]
    profile_name = f"Temporary Chapter 5 tone audit {nonce}"
    created = session.post(
        f"{API_ROOT}/api/students",
        json={"name": profile_name, "group_type": "foreigner"},
        timeout=20,
    )
    created.raise_for_status()
    student_id = created.json()["id"]

    # Guard against writing this temporary profile into a different database than cleanup uses.
    with sqlite3.connect(practice_store.DATABASE_PATH, timeout=30) as connection:
        found = connection.execute("SELECT id FROM students WHERE id = ?", (student_id,)).fetchone()
    if not found:
        cleanup_student(student_id)
        raise RuntimeError(
            f"Temporary profile {student_id} is not in {practice_store.DATABASE_PATH}; stopping before audio uploads."
        )

    rows = []
    try:
        with RAW_JSONL.open("a", encoding="utf-8") as raw_file:
            for index, item in enumerate(items, start=1):
                audio_path = ROOT / "audio" / item["audio_file"]
                record = {
                    "word_key": item["word_key"],
                    "word_id": item["word_id"],
                    "section": item["section"],
                    "item_type": item["item_type"],
                    "script": item["script"],
                    "pinyin": item["pinyin"],
                    "english_meaning": item["english_meaning"],
                    "audio_file": item["audio_file"],
                    "http_status": "",
                    "result_status": "",
                    "phrase_statuses": "[]",
                    "expected_tone_sequences": "[]",
                    "observed_tone_sequences": "[]",
                    "transcript_status": "",
                    "recognized_text": "",
                    "matched_syllables": "",
                    "total_syllables": "",
                    "n_phrases": "",
                    "n_with_audio_evidence": "",
                    "error": "",
                }
                payload = None
                try:
                    with audio_path.open("rb") as audio_file:
                        response = session.post(
                            f"{API_ROOT}/api/assess-pronunciation",
                            data={
                                "script": item["script"],
                                "sample_id": item["word_key"],
                                "student_id": student_id,
                                "include_transcript": "true",
                                "practice_mode": "chapter5_word",
                            },
                            files={"file": (audio_path.name, audio_file, "audio/mpeg")},
                            timeout=(15, 240),
                        )
                    record["http_status"] = response.status_code
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    if response.ok and isinstance(payload, dict):
                        phrases = payload.get("phrases") or []
                        statuses = [phrase.get("phrase_status", phrase.get("status", "unknown")) for phrase in phrases]
                        record.update({
                            "result_status": ",".join(statuses),
                            "phrase_statuses": json.dumps(statuses, ensure_ascii=False),
                            "expected_tone_sequences": json.dumps([p.get("expected_tone_sequence") for p in phrases], ensure_ascii=False),
                            "observed_tone_sequences": json.dumps([p.get("observed_tone_sequence") for p in phrases], ensure_ascii=False),
                            "transcript_status": (payload.get("transcript_check") or {}).get("status", ""),
                            "recognized_text": (payload.get("transcript_check") or {}).get("recognized_text", ""),
                            "matched_syllables": (payload.get("transcript_check") or {}).get("matched_syllables", ""),
                            "total_syllables": (payload.get("transcript_check") or {}).get("total_syllables", ""),
                            "n_phrases": (payload.get("summary") or {}).get("n_phrases", len(phrases)),
                            "n_with_audio_evidence": (payload.get("summary") or {}).get("n_with_audio_evidence", ""),
                        })
                        phrase = phrases[0] if phrases else {}
                        phrase_analysis = phrase.get("phrase_analysis") or {}
                        alignment = (payload.get("alignment_quality") or {}).get("learner") or {}
                        record.update({
                            "aligned_script_variant": payload.get("aligned_script_variant", ""),
                            "alignment_status": alignment.get("status", ""),
                            "alignment_method": alignment.get("method", ""),
                            "aligned_characters": f"{alignment.get('reliable_character_count', '')}/{alignment.get('character_count', '')}",
                            "alignment_candidates": json.dumps(alignment.get("candidates", []), ensure_ascii=False),
                            "phrase_window_sec": json.dumps(phrase_analysis.get("phrase_window_sec"), ensure_ascii=False),
                            "trimmed_voiced_coverage": phrase_analysis.get("trimmed_voiced_coverage", ""),
                            "maximum_internal_unvoiced_gap_sec": phrase_analysis.get("maximum_internal_unvoiced_gap_sec", ""),
                        })
                    else:
                        record["error"] = (payload or {}).get("detail", response.text[:1000]) if isinstance(payload, dict) else response.text[:1000]
                except Exception as exc:  # Keep each failed request visible in the audit.
                    record["error"] = f"{type(exc).__name__}: {exc}"
                raw_file.write(json.dumps({"index": index, "item": item, "http_status": record["http_status"], "response": payload, "error": record["error"]}, ensure_ascii=False) + "\n")
                raw_file.flush()
                rows.append(record)
                persist_report(rows, request_started)
                print(
                    f"[{index:02d}/53] {item['word_key']} {item['script']} "
                    f"HTTP {record['http_status']} {record['result_status'] or record['error']}",
                    flush=True,
                )
    finally:
        cleanup_student(student_id)
        session.close()

    persist_report(rows, request_started)
    print(f"REPORT={REPORT_MD}", flush=True)
    print(f"CSV={DETAIL_CSV}", flush=True)
    print(f"JSONL={RAW_JSONL}", flush=True)
    return 0 if len(rows) == len(items) else 2


if __name__ == "__main__":
    raise SystemExit(main())
