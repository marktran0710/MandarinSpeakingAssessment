import csv
import hashlib
import hmac
import json
import os
import re
import sys
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from threading import Lock

import opencc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Depends
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from pypinyin import lazy_pinyin, Style

import audio_prep
import capt_core as core
import tts_sample
import tone_assessment
import tone_pipeline
from practice_store import practice_store
from user_archive import UserArchive

# A second copy of every attempt (audio + log), one folder per user. MANDARIN_USER_DATA_DIR="" switches it off.
user_archive = UserArchive.from_env()

# Initialize the FastAPI app
app = FastAPI(title="Taiwan Mandarin CAPT API")


@app.exception_handler(RequestValidationError)
async def explain_rejected_request(request: Request, exc: RequestValidationError):
    """A 422 from FastAPI itself (the request did not even reach our code) used to leave one bare line in the console. Say what was missing, and for a
    voice calibration also write it into the user's calibration log: the usual cause is an upload that arrived without its audio (a connection or
    proxy that cut the body, a browser that stopped before the recording was ready)."""
    problems = [f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('type')}" for error in exc.errors()]
    length = request.headers.get("content-length")
    print(f"[422] {request.method} {request.url.path} rejected before it reached the handler: {'; '.join(problems)} (content-length {length})", file=sys.stderr)
    match = re.fullmatch(r"/api/students/([0-9a-f]+)/voice-calibration", request.url.path)
    if match:
        try:
            student = practice_store.get_student(match.group(1))
            if student is not None:
                user_archive.archive_calibration(student, {"outcome": "no_audio_received", "problems": problems, "content_length": length})
        except Exception as error:                      # explaining must never break the answer
            print(f"Warning: could not log the rejected calibration request: {error}", file=sys.stderr)
    return await request_validation_exception_handler(request, exc)
ASSESSMENT_LOCK = Lock()


@app.on_event("startup")
def warm_up_alignment():
    """With TONE_MFA_WARM=1 load the aligner now, so the first learner does not wait for `mfa align_one` to start."""
    tone_pipeline.prewarm_mfa()

ADMIN_PASSWORD = os.environ.get("MANDARIN_ADMIN_PASSWORD", "").strip()
ADMIN_SESSION_COOKIE = "mandarin_admin_session"
ADMIN_SESSION_TTL_SECONDS = 8 * 60 * 60
admin_basic_auth = HTTPBasic(auto_error=False)


class StudentProfileInput(BaseModel):
    name: str
    group_type: str


class UserFeedbackInput(BaseModel):
    student_id: str
    attempt_id: str
    rating: int
    comment: str = ""


class AdminLoginInput(BaseModel):
    password: str

# Human-recorded practice samples and their verified scripts.
SAMPLE_DATA_DIR = Path(os.environ.get(
    "MANDARIN_SAMPLE_DATA_DIR",
    r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll"
))
SAMPLE_CSV_PATH = SAMPLE_DATA_DIR / "lessons-5-8_with_character_audio_synced.csv"
WORD_LEVEL_PATH = Path(os.environ.get(
    "MANDARIN_WORD_LEVEL_PATH",
    r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_word_level.json"
))


def load_sample_catalog():
    catalog = {}
    if not SAMPLE_CSV_PATH.is_file():
        print(f"Warning: sample script CSV was not found: {SAMPLE_CSV_PATH}")
        return catalog

    with SAMPLE_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                lesson = int(row["lesson"])
                story = int(row["story"])
                scene = int(row["scene"])
            except (KeyError, TypeError, ValueError):
                continue

            audio_path = SAMPLE_DATA_DIR / f"{lesson}-{story}-{scene:02d}.mp3"
            script = (row.get("script") or "").strip()
            if not script:
                continue

            sample_id = f"{lesson}-{story}-{scene}"
            catalog[sample_id] = {
                "id": sample_id,
                "lesson": lesson,
                "story": story,
                "scene": scene,
                "character": (row.get("character") or "").strip(),
                "script": script,
                "audio_path": audio_path,
                "audio_available": audio_path.is_file(),
            }

    print(f"Loaded {len(catalog)} human voice samples from {SAMPLE_CSV_PATH.name}")
    return catalog


SAMPLE_CATALOG = load_sample_catalog()

def load_word_annotations():
    if not WORD_LEVEL_PATH.is_file():
        print(f"Warning: word annotation JSON was not found: {WORD_LEVEL_PATH}")
        return {}
    try:
        data = json.loads(WORD_LEVEL_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(f"Warning: could not load word annotation JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        print("Warning: word annotation JSON must contain an object.")
        return {}
    return {
        str(sample_id): [str(word) for word in words if str(word).strip()]
        for sample_id, words in data.items()
        if isinstance(words, list)
    }


WORD_ANNOTATIONS = load_word_annotations()

CHAPTER5_WORD_DATA_DIR = Path(__file__).resolve().parent / "chapter5_word_practice"
CHAPTER5_WORD_MANIFEST = CHAPTER5_WORD_DATA_DIR / "items.json"
CHAPTER5_CATEGORY_MANIFEST = CHAPTER5_WORD_DATA_DIR / "categories.json"
CHAPTER5_AUDIO_DIR = CHAPTER5_WORD_DATA_DIR / "audio"


def load_chapter5_taxonomy():
    if not CHAPTER5_CATEGORY_MANIFEST.is_file():
        print(f"Warning: Chapter 5 category data was not found: {CHAPTER5_CATEGORY_MANIFEST}")
        return {"categories": [], "items": {}}
    try:
        payload = json.loads(CHAPTER5_CATEGORY_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Warning: could not load Chapter 5 category data: {exc}")
        return {"categories": [], "items": {}}
    if payload.get("schema_version") != 1 or not isinstance(payload.get("categories"), list) or not isinstance(payload.get("items"), dict):
        print("Warning: Chapter 5 category data has an invalid format.")
        return {"categories": [], "items": {}}
    return payload


CHAPTER5_TAXONOMY = load_chapter5_taxonomy()


def load_chapter5_word_items():
    if not CHAPTER5_WORD_MANIFEST.is_file():
        print(f"Warning: Chapter 5 word practice data was not found: {CHAPTER5_WORD_MANIFEST}")
        return {}
    try:
        payload = json.loads(CHAPTER5_WORD_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Warning: could not load Chapter 5 word practice data: {exc}")
        return {}
    if payload.get("chapter") != 5 or not isinstance(payload.get("items"), list):
        print("Warning: Chapter 5 word practice manifest has an invalid format.")
        return {}

    items = {}
    for item in payload["items"]:
        word_key = str(item.get("word_key") or "").strip()
        audio_file = str(item.get("audio_file") or "").strip()
        audio_path = CHAPTER5_AUDIO_DIR / audio_file
        if (
            not word_key
            or not audio_file
            or Path(audio_file).name != audio_file
            or not audio_path.is_file()
            or not str(item.get("script") or "").strip()
        ):
            print(f"Warning: skipped incomplete Chapter 5 word item: {word_key or audio_file}")
            continue
        normalized_item = dict(item)
        normalized_item["audio_path"] = audio_path
        category = CHAPTER5_TAXONOMY.get("items", {}).get(str(item.get("word_id")), {})
        normalized_item["category_id"] = category.get("category_id")
        normalized_item["subcategory_id"] = category.get("subcategory_id")
        if not normalized_item["category_id"] or not normalized_item["subcategory_id"]:
            print(f"Warning: Chapter 5 item has no category assignment: {word_key}")
        items[word_key] = normalized_item
    print(f"Loaded {len(items)} Chapter 5 word/phrase recordings.")
    return items


CHAPTER5_WORD_ITEMS = load_chapter5_word_items()

converter = opencc.OpenCC('s2twp')   # Normalize scripts to Taiwan Traditional Chinese

# Words used to split text the learner types (My own text) into phrases: the teacher's annotation + the vocabulary lists.
SEGMENT_LEXICON = tone_assessment.build_lexicon(
    [word for words in WORD_ANNOTATIONS.values() for word in words]
    + [item["script"] for item in CHAPTER5_WORD_ITEMS.values()], converter.convert)


@app.get("/api/tts-sample")
def tts_sample_audio(text: str = ""):
    """A computer-voice sample of what the learner typed in "My own text" (Windows' Taiwan Mandarin voice, a little slow; cached). Not used for scoring."""
    text = text[:200]
    converted = converter.convert(text)
    try:
        path = tts_sample.synthesize(converted if len(converted) == len(text) else text)
    except tts_sample.BadText as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except tts_sample.SampleUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/pinyin")
async def get_pinyin(text: str = ""):
    """Pinyin for a text the learner is typing, one entry per character, for the page to show above it."""
    text = text[:200]
    converted = converter.convert(text)
    return {"pinyin": core.pinyin_per_character(converted if len(converted) == len(text) else text)}



@app.get("/api/sample-lessons")
async def get_sample_lessons():
    return [
        {
            "id": sample["id"],
            "lesson": sample["lesson"],
            "story": sample["story"],
            "scene": sample["scene"],
            "character": sample["character"],
            "script": sample["script"],
            "pinyin": core.display_pinyin(sample["script"]),
            "audio_available": bool(sample.get("audio_available", Path(sample["audio_path"]).is_file())),
            "audio_url": (
                f"/api/sample-lessons/{sample['id']}/audio"
                if sample.get("audio_available", Path(sample["audio_path"]).is_file())
                else None
            ),
        }
        for sample in SAMPLE_CATALOG.values()
    ]


@app.get("/api/sample-lessons/{sample_id}/audio")
async def get_sample_audio(sample_id: str):
    sample = SAMPLE_CATALOG.get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample audio was not found.")
    if not Path(sample["audio_path"]).is_file():
        raise HTTPException(status_code=404, detail="Sample audio was not found.")
    return FileResponse(sample["audio_path"], media_type="audio/mpeg")


@app.get("/api/chapter5-word-items")
async def get_chapter5_word_items():
    taxonomy = load_chapter5_taxonomy()
    category_assignments = taxonomy.get("items", {})
    return [
        {
            "word_key": item["word_key"],
            "word_id": item["word_id"],
            "chapter": 5,
            "section": item["section"],
            "item_type": item["item_type"],
            "script": item["script"],
            "traditional_chinese": item["traditional_chinese"],
            "pinyin": item["pinyin"],
            "character_pinyin": tone_assessment.chapter5_character_pinyin(item),
            "display_cells": tone_assessment.chapter5_display_cells(item),   # full form with the optional part marked, or null
            "category_id": category_assignments.get(str(item["word_id"]), {}).get("category_id"),
            "subcategory_id": category_assignments.get(str(item["word_id"]), {}).get("subcategory_id"),
            "pos": item["pos"],
            "english_meaning": item["english_meaning"],
            "audio_url": f"/api/chapter5-word-items/{item['word_key']}/audio",
        }
        for item in CHAPTER5_WORD_ITEMS.values()
    ]


@app.get("/api/chapter5-word-categories")
async def get_chapter5_word_categories():
    return load_chapter5_taxonomy().get("categories", [])


@app.get("/api/chapter5-word-items/{word_key}/audio")
async def get_chapter5_word_audio(word_key: str):
    item = CHAPTER5_WORD_ITEMS.get(word_key)
    if item is None or not Path(item["audio_path"]).is_file():
        raise HTTPException(status_code=404, detail="Chapter 5 word audio was not found.")
    return FileResponse(item["audio_path"], media_type="audio/mpeg")


@app.post("/api/students")
async def create_student_profile(profile: StudentProfileInput):
    try:
        return practice_store.create_or_get_student(profile.name, profile.group_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/leaderboard")
async def get_leaderboard(student_id: str = ""):
    """The top 10 learners by points (see PracticeStore.leaderboard); `student_id` (optional) adds the caller's own rank."""
    return practice_store.leaderboard(student_id or None)


@app.get("/api/progress")
async def get_progress(student_id: str = ""):
    """What the learner has practised (marks for the sentence and word lists) and their latest results; see PracticeStore.progress."""
    return practice_store.progress(student_id) if student_id else {"lessons": {}, "recent": []}


@app.get("/api/students/{student_id}")
async def get_student_profile(student_id: str):
    profile = practice_store.get_student(student_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Speaker profile was not found.")
    return profile


CALIBRATION_SOUND_DIR = Path(__file__).resolve().parent / "assets" / "calibration"


@app.get("/api/calibration-sound/{tone}")
def calibration_sound(tone: int):
    """The reference sound of one calibration tile: ma1 ... ma5 (tools/make_calibration_audio.py)."""
    path = CALIBRATION_SOUND_DIR / f"ma{tone}.wav"
    if tone not in (1, 2, 3, 4, 5) or not path.is_file():
        raise HTTPException(status_code=404, detail="Calibration sound not found.")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/students/{student_id}/voice-calibration")
def calibrate_student_voice(student_id: str, file: UploadFile = File(...)):
    student = practice_store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Speaker profile was not found.")
    allowed_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")
    filename = str(file.filename or "voice-calibration.webm")
    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")
    suffix = os.path.splitext(filename)[1].lower() or ".webm"
    temp_path = None
    seen = {"file": filename}                                   # what the calibration log says about this try
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        seen["bytes"] = os.path.getsize(temp_path)
        samples, sample_rate = audio_prep.decode_audio(temp_path)
        seen["duration_sec"] = round(len(samples) / float(sample_rate), 2)
        calibration = tone_pipeline.calibrate_voice(samples, sample_rate, tone_assessment.get_config())
        saved = practice_store.save_voice_calibration(student_id, calibration)
        pace_hint = tone_pipeline.calibration_pace_hint(calibration, tone_assessment.get_config())
        quality = calibration.get("quality") or {}
        user_archive.archive_calibration(student, {**seen, "outcome": "saved", "pace_hint": pace_hint, "reliable": calibration.get("reliable"),
                                                   "ac_cc_agreement": calibration.get("ac_cc_agreement"), "voiced_sec": quality.get("voiced_sec"),
                                                   "median_hnr_db": quality.get("median_hnr_db"), "boundary_hit_rate": quality.get("boundary_hit_rate"),
                                                   "warnings": calibration.get("warnings")})
        return {"voice_calibration": saved, "pace_hint": pace_hint}
    except tone_pipeline.CalibrationError as exc:          # the page shows a specific, translated tip for each reason
        user_archive.archive_calibration(student, {**seen, "outcome": "refused", "reason": exc.reason, "message": str(exc), **exc.details})
        return JSONResponse(status_code=422, content={"detail": str(exc), "reason": exc.reason})
    except ValueError as exc:
        user_archive.archive_calibration(student, {**seen, "outcome": "invalid", "message": str(exc)[:300]})
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        user_archive.archive_calibration(student, {**seen, "outcome": "unreadable", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        raise HTTPException(status_code=400, detail=f"Could not read calibration audio: {exc}")
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.get("/api/students/{student_id}/attempts")
async def get_student_attempts(student_id: str, limit: int = 20):
    if practice_store.get_student(student_id) is None:
        raise HTTPException(status_code=404, detail="Speaker profile was not found.")
    return {"attempts": practice_store.list_attempts(student_id, limit)}


@app.post("/api/feedback")
async def submit_user_feedback(feedback: UserFeedbackInput):
    try:
        return practice_store.save_feedback(
            student_id=feedback.student_id,
            attempt_id=feedback.attempt_id,
            rating=feedback.rating,
            comment=feedback.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _admin_session_signature(expires_at: int) -> str:
    message = f"mandarin-admin:{expires_at}".encode("utf-8")
    return hmac.new(ADMIN_PASSWORD.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _has_valid_admin_session(request: Request) -> bool:
    if not ADMIN_PASSWORD:
        return False
    token = request.cookies.get(ADMIN_SESSION_COOKIE, "")
    try:
        expires_text, signature = token.split(".", 1)
        expires_at = int(expires_text)
    except (ValueError, TypeError):
        return False
    return expires_at > int(time.time()) and secrets.compare_digest(
        signature, _admin_session_signature(expires_at)
    )


def require_admin_access(request: Request, credentials: HTTPBasicCredentials | None = None):
    client_host = request.client.host if request.client else ""
    is_loopback = client_host in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}
    if not ADMIN_PASSWORD and is_loopback:
        return
    if not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="Admin sign-in is unavailable. Contact the server administrator.",
        )
    if _has_valid_admin_session(request):
        return
    valid_username = credentials is not None and secrets.compare_digest(credentials.username, "admin")
    valid_password = credentials is not None and secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=401,
            detail="Admin login required.",
        )


@app.post("/admin/login")
async def admin_login(payload: AdminLoginInput, request: Request, response: Response):
    client_host = request.client.host if request.client else ""
    is_loopback = client_host in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}
    if not ADMIN_PASSWORD:
        if is_loopback:
            return {"ok": True, "local_access": True}
        raise HTTPException(status_code=503, detail="Admin sign-in is unavailable. Contact the server administrator.")
    if not secrets.compare_digest(payload.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    expires_at = int(time.time()) + ADMIN_SESSION_TTL_SECONDS
    token = f"{expires_at}.{_admin_session_signature(expires_at)}"
    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
    response.set_cookie(
        ADMIN_SESSION_COOKIE,
        token,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        secure=is_https,
        samesite="strict",
        path="/",
    )
    return {"ok": True}


@app.post("/admin/logout")
async def admin_logout(response: Response):
    response.delete_cookie(ADMIN_SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return {"ok": True}


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    client_host = request.client.host if request.client else ""
    is_loopback = client_host in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}
    preauthenticated = not ADMIN_PASSWORD and is_loopback
    html = """<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Practice results</title>
        <style>
            body { max-width: 1040px; margin: 24px auto; padding: 0 16px 40px; font: 15px/1.5 system-ui, sans-serif; color: #243447; background: #f4f6f9; }
            header { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 20px; }
            h1 { margin: 0; font-size: 26px; }
            #loadStatus { color: #536577; }
            #users { display: grid; gap: 14px; }
            .user, .attempt { border: 1px solid #d8e1ea; border-radius: 10px; background: white; }
            .user { padding: 16px; }
            .user-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
            .user h2 { margin: 0 0 3px; font-size: 19px; }
            .muted { color: #607386; font-size: 13px; }
            .attempt { margin-top: 12px; overflow: hidden; }
            .attempt summary { cursor: pointer; padding: 11px 14px; font-weight: 650; background: #f8fafc; }
            .attempt-body { padding: 4px 14px 14px; }
            .attempt-body h3 { margin: 15px 0 5px; font-size: 15px; }
            .scores { display: flex; flex-wrap: wrap; gap: 8px; }
            .score { background: #eef4f8; border-radius: 6px; padding: 5px 8px; }
            .syllable, .word { border-top: 1px solid #edf1f5; padding: 7px 0; }
            .feedback { border-left: 3px solid #3182ce; background: #f4f9ff; padding: 9px 12px; margin-top: 8px; white-space: pre-wrap; }
            .login-panel { max-width: 420px; margin: 12vh auto 0; padding: 24px; border: 1px solid #d8e1ea; border-radius: 10px; background: white; box-shadow: 0 4px 18px rgba(25, 45, 65, 0.08); }
            .login-panel h1 { margin: 0 0 6px; font-size: 24px; }
            .login-panel p { color: #607386; }
            .login-panel label { display: block; margin: 16px 0 12px; font-weight: 600; }
            .login-panel input { display: block; width: 100%; box-sizing: border-box; padding: 11px; margin-top: 6px; border: 1px solid #cbd5e0; border-radius: 6px; font: inherit; }
            #loginStatus { min-height: 1.5em; color: #9b2c2c; }
            #logoutBtn { background: #718096; }
            .danger { background: #b42318; }
            .danger:hover { background: #912018; }
            button:disabled { opacity: 0.6; cursor: wait; }
            .admin-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin: 0 0 14px; padding: 10px 12px; background: white; border: 1px solid #d8e1ea; border-radius: 10px; }
            .view-tabs { display: flex; gap: 6px; }
            .view-tabs button { background: #edf2f7; color: #2d3748; }
            .view-tabs button.active { background: #2b6cb0; color: white; }
            #feedbackFilters { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; }
            #feedbackFilters[hidden], #feedbackList[hidden], #users[hidden] { display: none; }
            #feedbackFilters label { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; }
            #feedbackFilters input[type="search"], #feedbackFilters select { font: inherit; padding: 4px 6px; border: 1px solid #cbd5e0; border-radius: 6px; }
            #feedbackSummary { flex: 1 1 100%; }
            #feedbackList { display: grid; gap: 10px; }
            .feedback-card { border: 1px solid #d8e1ea; border-radius: 10px; background: white; padding: 12px 14px; }
            .feedback-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }
            .feedback-stars { color: #d69e2e; letter-spacing: 1px; font-size: 16px; }
            .feedback-stars .off { color: #cbd5e0; }
            .feedback-who { font-weight: 650; }
            .feedback-text { margin: 8px 0 6px; white-space: pre-wrap; }
            .feedback-card audio { width: 100%; height: 34px; margin-top: 8px; }
            audio { width: min(420px, 100%); margin-top: 8px; }
            button { border: 0; border-radius: 6px; padding: 9px 13px; background: #2b6cb0; color: white; font-weight: 600; cursor: pointer; }
            @media (max-width: 560px) { header, .user-header { align-items: flex-start; flex-direction: column; } }
        </style>
    </head>
    <body>
        <section id="loginPanel" class="login-panel">
            <h1>Admin login</h1>
            <p>Enter the administrator password to view practice records.</p>
            <form id="loginForm" onsubmit="loginAdmin(event)">
                <label for="adminPassword">Password
                    <input id="adminPassword" name="password" type="password" autocomplete="current-password" required>
                </label>
                <button id="loginBtn" type="submit">Sign in</button>
                <p id="loginStatus" role="status" aria-live="polite"></p>
            </form>
        </section>
        <div id="dashboardPanel" hidden>
        <header>
            <div><h1>Practice results</h1><div class="muted">Users, assessment output, saved audio and feedback</div></div>
            <div><button type="button" onclick="loadDashboard()">Refresh</button> <button id="logoutBtn" type="button" onclick="logoutAdmin()">Log out</button></div>
        </header>
        <div class="admin-tools">
            <div class="view-tabs" role="tablist" aria-label="View">
                <button type="button" id="tabAll" class="active" role="tab" aria-selected="true" onclick="setAdminView('all')">All results</button>
                <button type="button" id="tabFeedback" role="tab" aria-selected="false" onclick="setAdminView('feedback')">Ratings &amp; comments</button>
            </div>
            <div id="feedbackFilters" hidden>
                <label><input type="checkbox" id="onlyComments" onchange="renderAdminView()"> Only with a comment</label>
                <label>Rating
                    <select id="ratingFilter" onchange="renderAdminView()">
                        <option value="">All</option>
                        <option value="5">5 stars</option>
                        <option value="4">4 stars</option>
                        <option value="3">3 stars</option>
                        <option value="2">2 stars</option>
                        <option value="1">1 star</option>
                        <option value="low">3 stars or lower</option>
                    </select>
                </label>
                <label>User <input type="search" id="userFilter" placeholder="name" oninput="renderAdminView()"></label>
                <span id="feedbackSummary" class="muted" role="status"></span>
            </div>
        </div>
        <p id="loadStatus" role="status">Loading results...</p>
        <main id="users"></main>
        <main id="feedbackList" hidden></main>
        </div>
        <script>
            const ADMIN_PREAUTHENTICATED = __ADMIN_PREAUTHENTICATED__;
            const statusNames = {
                likely_correct: "Likely correct", practice: "Practice",
                uncertain: "Uncertain", insufficient_evidence: "Not enough evidence", unassessable: "Not scored",
                text_mismatch: "Script mismatch", transcript_unchecked: "Transcript unchecked"
            };
            const groupNames = { taiwanese: "Taiwanese", foreigner: "Foreigner" };
            function showAdminDashboard() {
                document.getElementById("loginPanel").hidden = true;
                document.getElementById("dashboardPanel").hidden = false;
                loadDashboard();
            }

            async function loginAdmin(event) {
                event.preventDefault();
                const status = document.getElementById("loginStatus");
                const button = document.getElementById("loginBtn");
                button.disabled = true;
                status.textContent = "Signing in...";
                try {
                    const response = await fetch("/admin/login", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ password: document.getElementById("adminPassword").value })
                    });
                    if (!response.ok) {
                        status.textContent = response.status === 401
                            ? "Incorrect password. Try again."
                            : "Admin sign-in is unavailable. Contact the server administrator.";
                        return;
                    }
                    document.getElementById("adminPassword").value = "";
                    showAdminDashboard();
                } catch (error) {
                    status.textContent = "Could not sign in. Check the server connection and try again.";
                } finally {
                    button.disabled = false;
                }
            }

            async function logoutAdmin() {
                try { await fetch("/admin/logout", { method: "POST" }); } catch (error) {}
                window.location.reload();
            }

            const el = (tag, text, className) => {
                const node = document.createElement(tag);
                if (text !== undefined && text !== null) node.textContent = text;
                if (className) node.className = className;
                return node;
            };
            const dateLabel = value => value ? new Date(value).toLocaleString() : "—";

            function renderAttempt(attempt) {
                const details = el("details", undefined, "attempt");
                const scored = Number.isInteger(attempt.summary.n_phrases)
                    ? `${attempt.summary.n_assessable || 0}/${attempt.summary.n_phrases} phrase decisions confident`
                    : `${attempt.summary.n_assessable || 0}/${attempt.summary.n_syllables || 0} syllables scored`;
                const sourceLabel = attempt.practice_mode === "custom"
                    ? "Custom text"
                    : attempt.practice_mode === "chapter5_word"
                        ? `Chapter 5 ${attempt.practice_item?.script || attempt.target_script}`
                        : `Sample ${attempt.sample_id}`;
                const summary = el("summary", `${sourceLabel} | ${dateLabel(attempt.created_at)} | ${scored}`);
                details.append(summary);
                const body = el("div", undefined, "attempt-body");
                body.append(el("div", `Target: ${attempt.target_script || "—"}`));
                body.append(el("div", `Heard: ${attempt.recognized_text || "No transcript"} · ${attempt.transcript_status || "transcript status unknown"}`, "muted"));
                const scores = el("div", undefined, "scores");
                for (const [label, value] of [["Likely correct", attempt.summary.likely_correct], ["Practice", attempt.summary.practice], ["Not enough evidence", attempt.summary.insufficient_evidence ?? attempt.summary.uncertain]]) {
                    scores.append(el("span", `${label}: ${value || 0}`, "score"));
                }
                body.append(scores);
                if (attempt.phrases?.length) {
                    body.append(el("h3", "Phrase tone sequence output"));
                    const toneSequence = values => Array.isArray(values)
                        ? values.map(tone => tone === null || tone === undefined ? "T?" : `T${tone}`).join(" - ")
                        : "Unavailable";
                    for (const item of attempt.phrases) {
                        const phrase = item.annotation || item.word || "Phrase";
                        const note = item.assessment_note ? ` | ${item.assessment_note}` : "";
                        body.append(el("div", `${phrase} | Expected ${toneSequence(item.expected_tone_sequence)} | ${statusNames[item.status] || item.status || "Not scored"}${note}`, "word"));
                    }
                }

                if (attempt.syllables.length) {
                    body.append(el("h3", "Syllable output"));
                    for (const item of attempt.syllables) {
                        const tone = item.lexical_tone && item.lexical_tone !== item.expected_tone
                            ? `T${item.lexical_tone} → T${item.expected_tone}`
                            : item.expected_tone ? `T${item.expected_tone}` : "tone unavailable";
                        const note = item.assessment_note ? ` · ${item.assessment_note}` : "";
                        body.append(el("div", `${item.expected_char || "?"} (${item.expected_pinyin || "—"}) · ${tone} · ${statusNames[item.status] || item.status || "Not scored"}${note}`, "syllable"));
                    }
                }
                if (attempt.words.length) {
                    body.append(el("h3", "Word output"));
                    for (const item of attempt.words) {
                        const note = item.assessment_note ? ` · ${item.assessment_note}` : "";
                        body.append(el("div", `${item.annotation || item.word || "—"} · ${statusNames[item.status] || item.status || "Not scored"}${note}`, "word"));
                    }
                }
                if (attempt.feedback) {
                    const feedbackNote = attempt.feedback.comment || "No comment";
                    const feedbackText = `User rating: ${attempt.feedback.rating}/5${String.fromCharCode(10)}${feedbackNote}`;
                    body.append(el("h3", `Feedback · ${dateLabel(attempt.feedback.created_at)}`));
                    body.append(el("div", feedbackText, "feedback"));
                } else {
                    body.append(el("h3", "Feedback"));
                    body.append(el("div", "No feedback submitted yet.", "muted"));
                }
                const audio = document.createElement("audio");
                audio.controls = true;
                audio.preload = "none";
                audio.src = attempt.audio_url;
                audio.setAttribute("aria-label", `Original recording for ${sourceLabel.toLowerCase()}`);
                body.append(audio);
                details.append(body);
                return details;
            }

            function renderUser(user) {
                const card = el("section", undefined, "user");
                const userHeader = el("div", undefined, "user-header");
                userHeader.append(el("h2", user.name));
                const deleteButton = el("button", "Delete account", "danger");
                deleteButton.type = "button";
                deleteButton.setAttribute("aria-label", `Delete account ${user.name} and all saved data`);
                deleteButton.addEventListener("click", () => deleteUser(user, deleteButton));
                userHeader.append(deleteButton);
                card.append(userHeader);
                card.append(el("div", `${groupNames[user.group_type] || user.group_type} · ${user.attempts.length} practice attempts · last seen ${dateLabel(user.last_seen_at)}`, "muted"));
                if (!user.attempts.length) card.append(el("p", "No assessment output yet.", "muted"));
                for (const attempt of user.attempts) card.append(renderAttempt(attempt));
                return card;
            }

            async function deleteUser(user, button) {
                const confirmed = window.confirm(
                    `Permanently delete the account “${user.name}” and all of its data?\n\n` +
                    "This removes practice results, feedback, voice calibration, and saved recordings."
                );
                if (!confirmed) return;
                button.disabled = true;
                const status = document.getElementById("loadStatus");
                status.textContent = `Deleting ${user.name} and saved data...`;
                try {
                    const response = await fetch(`/api/admin/students/${encodeURIComponent(user.id)}`, {
                        method: "DELETE"
                    });
                    const data = await response.json().catch(() => ({}));
                    if (response.status === 401) {
                        document.getElementById("dashboardPanel").hidden = true;
                        document.getElementById("loginPanel").hidden = false;
                        document.getElementById("loginStatus").textContent = "Your sign-in expired. Please sign in again.";
                        return;
                    }
                    if (!response.ok) throw new Error(data.detail || "Could not delete this account.");
                    await loadDashboard();
                    status.textContent = data.audio_cleanup_pending
                        ? `Deleted ${user.name}'s account and records. ${data.audio_cleanup_pending} audio file(s) still need storage cleanup; retry after restarting the server.`
                        : `Deleted ${user.name}'s account and all linked data.`;
                } catch (error) {
                    status.textContent = error.message || "Could not delete this account.";
                } finally {
                    button.disabled = false;
                }
            }

            // Every rating/comment the learners left, newest first. filters: { onlyComments, rating ("" | "1".."5" | "low"), user (part of a name) }.
            function collectFeedback(users, filters) {
                const wanted = String((filters && filters.user) || "").trim().toLowerCase();
                const rows = [];
                for (const user of users || []) {
                    if (wanted && !String(user.name || "").toLowerCase().includes(wanted)) continue;
                    for (const attempt of user.attempts || []) {
                        const feedback = attempt.feedback;
                        if (!feedback) continue;
                        const comment = String(feedback.comment || "").trim();
                        if (filters && filters.onlyComments && !comment) continue;
                        const rating = Number(feedback.rating);
                        if (filters && filters.rating === "low" && !(rating <= 3)) continue;
                        if (filters && filters.rating && filters.rating !== "low" && rating !== Number(filters.rating)) continue;
                        rows.push({ user, attempt, rating, comment, created_at: feedback.created_at || attempt.created_at });
                    }
                }
                rows.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
                const total = rows.reduce((sum, row) => sum + row.rating, 0);
                return {
                    rows,
                    summary: { count: rows.length, average: rows.length ? total / rows.length : null, withComments: rows.filter(row => row.comment).length }
                };
            }

            function renderFeedbackCard(row) {
                const card = el("article", undefined, "feedback-card");
                const head = el("div", undefined, "feedback-head");
                const stars = el("span", undefined, "feedback-stars");
                stars.setAttribute("aria-label", `${row.rating} out of 5`);
                for (let n = 1; n <= 5; n++) stars.append(el("span", "★", n <= row.rating ? "" : "off"));
                head.append(stars, el("span", row.user.name, "feedback-who"), el("span", dateLabel(row.created_at), "muted"));
                card.append(head);
                card.append(el("div", row.comment || "No comment", row.comment ? "feedback-text" : "feedback-text muted"));
                const summary = row.attempt.summary || {};
                const sourceLabel = row.attempt.practice_mode === "custom" ? "Custom text"
                    : row.attempt.practice_mode === "chapter5_word" ? `Chapter 5 ${row.attempt.practice_item?.script || row.attempt.target_script}`
                    : `Sample ${row.attempt.sample_id}`;
                card.append(el("div", `${sourceLabel} · ${row.attempt.target_script || "—"} · Good ${summary.likely_correct || 0} · Okay ${summary.insufficient_evidence ?? summary.uncertain ?? 0} · Practice ${summary.practice || 0}`, "muted"));
                if (row.attempt.audio_url) {
                    const audio = document.createElement("audio");
                    audio.controls = true;
                    audio.preload = "none";
                    audio.src = row.attempt.audio_url;
                    audio.setAttribute("aria-label", `Recording that got this rating, ${row.user.name}`);
                    card.append(audio);
                }
                return card;
            }

            let adminData = null;
            let adminView = "all";
            try { adminView = localStorage.getItem("adminView") === "feedback" ? "feedback" : "all"; } catch (error) { /* private window */ }

            function setAdminView(view) {
                adminView = view === "feedback" ? "feedback" : "all";
                try { localStorage.setItem("adminView", adminView); } catch (error) { /* private window */ }
                renderAdminView();
            }

            function renderAdminView() {
                const feedbackView = adminView === "feedback";
                for (const [id, active] of [["tabAll", !feedbackView], ["tabFeedback", feedbackView]]) {
                    document.getElementById(id).classList.toggle("active", active);
                    document.getElementById(id).setAttribute("aria-selected", String(active));
                }
                document.getElementById("feedbackFilters").hidden = !feedbackView;
                document.getElementById("feedbackList").hidden = !feedbackView;
                document.getElementById("users").hidden = feedbackView;
                const status = document.getElementById("loadStatus");
                if (!adminData) return;
                if (!feedbackView) {
                    status.textContent = adminData.users.length ? `${adminData.users.length} users` : "No users yet.";
                    return;
                }
                const found = collectFeedback(adminData.users, {
                    onlyComments: document.getElementById("onlyComments").checked,
                    rating: document.getElementById("ratingFilter").value,
                    user: document.getElementById("userFilter").value
                });
                document.getElementById("feedbackList").replaceChildren(...found.rows.map(renderFeedbackCard));
                const { count, average, withComments } = found.summary;
                document.getElementById("feedbackSummary").textContent = count
                    ? `${count} rating${count === 1 ? "" : "s"} · average ${average.toFixed(2)} · ${withComments} with a comment`
                    : "Nothing matches these filters.";
                status.textContent = "";
            }

            async function loadDashboard() {
                const status = document.getElementById("loadStatus");
                status.textContent = "Loading results...";
                try {
                    const response = await fetch("/api/admin/dashboard");
                    const data = await response.json();
                    if (response.status === 401) {
                        document.getElementById("dashboardPanel").hidden = true;
                        document.getElementById("loginPanel").hidden = false;
                        document.getElementById("loginStatus").textContent = "Your sign-in expired. Please sign in again.";
                        return;
                    }
                    if (!response.ok) throw new Error("Could not load results.");
                    const users = document.getElementById("users");
                    users.replaceChildren(...data.users.map(renderUser));
                    adminData = data;
                    renderAdminView();
                } catch (error) {
                    status.textContent = error.message || "Could not load results.";
                }
            }
            if (ADMIN_PREAUTHENTICATED) showAdminDashboard();
        </script>
    </body>
    </html>"""
    return html.replace("__ADMIN_PREAUTHENTICATED__", "true" if preauthenticated else "false")


@app.get("/api/admin/dashboard")
async def get_admin_dashboard(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(admin_basic_auth),
):
    require_admin_access(request, credentials)
    return {"users": practice_store.list_admin_dashboard()}


@app.delete("/api/admin/students/{student_id}")
def delete_admin_student(
    student_id: str,
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(admin_basic_auth),
):
    require_admin_access(request, credentials)
    # Prevent an in-flight assessment from saving a new attempt during deletion.
    with ASSESSMENT_LOCK:
        try:
            deleted = practice_store.delete_student_data(student_id)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Could not safely delete account data. Check the server and retry.",
            ) from exc
    if deleted is None:
        raise HTTPException(status_code=404, detail="Account was not found.")
    return deleted


@app.get("/api/admin/attempts/{attempt_id}/audio")
async def get_admin_attempt_audio(
    attempt_id: str,
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(admin_basic_auth),
):
    require_admin_access(request, credentials)
    audio_path = practice_store.get_attempt_audio_path(attempt_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Saved recording was not found.")
    return FileResponse(audio_path)








def comparable_script(text):
    """Normalize script to the Hanzi sequence used by pronunciation scoring."""
    normalized = converter.convert((text or "").strip())
    return "".join(core.prepare_target(normalized))






@app.get("/", response_class=HTMLResponse)
async def read_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Taiwan Mandarin CAPT - Pronunciation Assessment UI</title>
        <meta charset="utf-8">
        <link rel="icon" href="data:,">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, 'Nirmala UI', 'Noto Sans Devanagari', 'Mangal', 'Kohinoor Devanagari', 'Devanagari Sangam MN', 'Microsoft JhengHei', 'PingFang TC', 'Noto Sans TC', 'Noto Sans CJK TC', 'Heiti TC', sans-serif; max-width: 750px; margin: 30px auto; padding: 20px; background: #f4f6f9; }
            .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 20px; }
            .student-heading { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
            .student-heading h2 { margin: 0; }
            .language-picker { display: inline-flex; align-items: center; gap: 8px; color: #4a5568; font-size: 14px; font-weight: 600; }
            .language-picker select { width: auto; min-width: 150px; margin: 0; padding: 8px 10px; font-size: 14px; }
            .profile-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: end; }
            .profile-fields label { display: block; font-weight: 600; }
            .practice-mode { display: flex; flex-wrap: wrap; gap: 10px 18px; margin: 0 0 14px; padding: 0; border: 0; }
            .practice-mode legend { margin-bottom: 8px; font-weight: 600; }
            .practice-mode label { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; }
            .practice-mode input { margin: 0; }
            .chapter5-category-picker { max-width: 520px; margin: 0 0 12px; }
            .chapter5-category-picker[hidden] { display: none; }
            .chapter5-category-picker label { display: block; }
            .chapter5-category-picker select { margin-top: 4px; }
            .chapter5-word-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin: 0 0 15px; padding: 12px; border: 1px solid #d9e2ef; border-radius: 10px; background: #f8fafc; }
            .chapter5-word-controls[hidden] { display: none; }
            .chapter5-word-info { flex: 1 1 220px; min-width: 0; }
            .chapter5-word-progress { color: #64748b; font-size: 13px; margin-top: 3px; }
            .chapter5-word-category { color: #52657a; font-size: 13px; margin-top: 3px; }
            .chapter5-word-controls button { min-width: 92px; }
            #customTextHint { margin: 0 0 8px; }
            #saveProfileBtn, #changeProfileBtn { background: #2b6cb0; color: white; margin-top: 12px; }
            #speakerDialog { width: min(440px, calc(100vw - 40px)); box-sizing: border-box; border: 0; border-radius: 12px; padding: 24px; box-shadow: 0 18px 60px rgba(15, 35, 55, 0.28); }
            #speakerDialog::backdrop { background: rgba(24, 39, 56, 0.55); }
            #speakerDialog h2 { margin-top: 0; }
            #speakerDialog .profile-fields { grid-template-columns: 1fr; }
            #cancelProfileBtn { background: #edf2f7; color: #2d3748; margin-top: 12px; }
            #speakerFormError { color: #9b2c2c; min-height: 1.3em; margin: 10px 0 0; }
            #voiceCalibrationDialog { width: min(480px, calc(100vw - 24px)); box-sizing: border-box; border: 0; border-radius: 14px; padding: 20px; box-shadow: 0 18px 60px rgba(15, 35, 55, 0.28); }
            #voiceCalibrationDialog::backdrop { background: rgba(24, 39, 56, 0.55); }
            #voiceCalibrationDialog h2 { margin: 0 0 6px; padding-right: 36px; }
            .cal-close { position: absolute; top: 8px; right: 8px; width: 36px; height: 36px; padding: 0; border-radius: 50%; background: transparent; color: #4a5568; font-size: 26px; line-height: 1; font-weight: 400; }
            .cal-close:hover:not(:disabled) { background: #edf2f7; }
            .cal-close:disabled { opacity: 0.35; cursor: default; }
            .cal-why { margin: 0 0 12px; color: #4a5568; line-height: 1.5; }
            .cal-steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
            .cal-step { border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; background: #fff; }
            .cal-head { display: flex; align-items: center; gap: 10px; font-weight: 700; line-height: 1.35; }
            .cal-num { flex: none; width: 24px; height: 24px; border-radius: 50%; background: #2b6cb0; color: #fff; font-size: 13px; display: inline-flex; align-items: center; justify-content: center; }
            .cal-hint { margin: 4px 0 0 34px; color: #607386; font-size: 13px; line-height: 1.4; }
            #voiceCalibrationScript { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; margin: 10px 0 0; }
            .cal-tile { text-align: center; padding: 8px 2px; border: 1px solid #d9e2ef; border-radius: 9px; background: #f7fafc; min-width: 0; }
            .cal-tile { position: relative; }
            .cal-play { display: block; margin: 0 auto 2px; width: 28px; min-width: 28px; height: 28px; padding: 0; border-radius: 50%; font-size: 12px; line-height: 1; background: #e3edf9; color: #1a4f9c; border: 1px solid #b9cde8; cursor: pointer; }
            .cal-play[data-playing="true"] { background: #1a4f9c; color: #fff; }
            .cal-listen-all { margin: 8px 0 0 34px; padding: 6px 12px; font-size: 13px; background: #e3edf9; color: #1a4f9c; border: 1px solid #b9cde8; border-radius: 8px; width: auto; cursor: pointer; }
            .cal-tile .han { display: block; font-size: 24px; line-height: 1.25; font-weight: 600; }
            .cal-tile .py { display: block; font-size: 15px; color: #2d3748; }
            .cal-tile .tone { display: block; font-size: 11px; color: #607386; overflow-wrap: anywhere; }
            #voiceCalibrationRecordBtn { width: 100%; min-height: 48px; margin-top: 10px; background: #2b6cb0; color: white; display: flex; align-items: center; justify-content: center; gap: 9px; }
            #voiceCalibrationRecordBtn::before { content: ""; flex: none; width: 12px; height: 12px; border-radius: 50%; background: #fff; }
            #voiceCalibrationRecordBtn.recording { background: #c53030; }
            #voiceCalibrationRecordBtn.recording::before { border-radius: 2px; }
            #voiceCalibrationRecordBtn:disabled { opacity: 0.75; cursor: default; }
            .cal-live { display: none; margin-top: 10px; }
            #voiceCalibrationDialog[data-state="recording"] .cal-live { display: block; }
            .cal-live-row { display: flex; justify-content: space-between; font-size: 13px; color: #4a5568; margin-bottom: 4px; }
            .cal-meter, .cal-timer { height: 8px; border-radius: 4px; background: #e2e8f0; overflow: hidden; position: relative; }
            .cal-meter > i { display: block; height: 100%; width: 0; background: #38a169; transition: width 0.08s linear; }
            .cal-timer-row { display: flex; align-items: center; gap: 10px; margin-top: 10px; font-size: 13px; color: #4a5568; }
            .cal-timer-row .cal-timer { flex: 1; }
            #calibrationTimerText { flex: none; min-width: 3.4em; text-align: right; font-variant-numeric: tabular-nums; }
            .cal-timer > i { display: block; height: 100%; width: 0; background: #d69e2e; }
            .cal-timer > i.enough { background: #38a169; }
            .cal-timer::after { content: ""; position: absolute; left: 52%; top: 0; bottom: 0; width: 2px; background: #fff; }
            .cal-nosound, .rec-nosound { color: #975a16; font-size: 13px; margin: 6px 0 0; }
            .cal-nosound[hidden], .rec-nosound[hidden], .rec-indicator[hidden] { display: none; }
            #voiceCalibrationStatus { margin: 12px 0 0; color: #4a5568; overflow-wrap: anywhere; line-height: 1.5; }
            #voiceCalibrationStatus:empty { display: none; }
            #voiceCalibrationStatus.error { color: #9b2c2c; background: #fff5f5; border: 1px solid #feb2b2; border-radius: 10px; padding: 10px 12px; }
            #voiceCalibrationStatus.success { color: #22543d; background: #f0fff4; border: 1px solid #9ae6b4; border-radius: 10px; padding: 12px; font-weight: 600; }
            #voiceCalibrationDialog[data-state="success"] .cal-steps, #voiceCalibrationDialog[data-state="success"] .cal-why { display: none; }
            #voiceCalibrationControls { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 12px; }
            #voiceCalibrationControls button { flex: 1 1 auto; }
            #voiceCalibrationControls button[hidden] { display: none; }
            #voiceCalibrationLaterBtn { background: #edf2f7; color: #2d3748; }
            #voiceCalibrationDialog[data-state="success"] #voiceCalibrationLaterBtn { background: #2b6cb0; color: #fff; min-height: 48px; }
            .cal-privacy { margin: 12px 0 0; font-size: 12px; color: #718096; line-height: 1.4; }
            .rec-indicator { display: flex; align-items: center; gap: 10px; margin: 0 0 10px; font-size: 14px; color: #4a5568; }
            .rec-dot { flex: none; width: 10px; height: 10px; border-radius: 50%; background: #e53e3e; animation: pulse 1.2s infinite; }
            .rec-clock { flex: none; min-width: 2.6em; font-variant-numeric: tabular-nums; font-weight: 600; }
            .rec-level { flex: 1; height: 8px; border-radius: 4px; background: #e2e8f0; overflow: hidden; }
            .rec-level > i { display: block; height: 100%; width: 0; background: #38a169; transition: width 0.08s linear; }
            #activeStudentPanel { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
            #calibrateVoiceBtn { background: #edf2f7; color: #2d3748; padding: 8px 12px; font-size: 14px; }
            .results-header { border-bottom: 2px solid #edf2f7; padding-bottom: 15px; margin-bottom: 20px; }

            button { padding: 12px 24px; font-size: 16px; cursor: pointer; border: none; border-radius: 6px; font-weight: bold; transition: all 0.2s; }
            #recordBtn { background-color: #2b6cb0; color: white; }
            #recordBtn.recording { background-color: #1a4f80; animation: pulse 1.2s infinite; }
            #recordBtn:disabled { opacity: 0.6; cursor: wait; }
            #stopBtn { background-color: #718096; color: white; }
            #stopBtn:disabled { background-color: #e2e8f0; cursor: not-allowed; }
            .record-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 15px; }
            #status { font-weight: bold; color: #2b6cb0; line-height: 1.5; overflow-wrap: anywhere; }
            #status.error { color: #744210; }
            #status.is-loading { min-height: 24px; display: flex; align-items: center; }
            .loading-spinner { width: 20px; height: 20px; box-sizing: border-box; border: 3px solid #cbd5e0; border-top-color: #2b6cb0; border-radius: 50%; animation: status-spin 0.8s linear infinite; }
            .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
            @keyframes status-spin { to { transform: rotate(360deg); } }
            @media (prefers-reduced-motion: reduce) { .loading-spinner { animation-duration: 1.8s; } }
            .upload-controls { margin-top: 20px; padding-top: 16px; border-top: 1px solid #e2e8f0; }
            .upload-hint { font-size: 13px; color: #4a5568; line-height: 1.5; margin: 6px 0 10px; }
            input[type="file"] { width: 100%; box-sizing: border-box; padding: 12px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 14px; }
            #uploadBtn { background: #2b6cb0; color: white; margin-top: 12px; }
            #uploadBtn:disabled { opacity: 0.6; cursor: not-allowed; }
            @media (max-width: 480px) { .record-controls button { flex-basis: 100%; } .profile-fields { grid-template-columns: 1fr; } }

            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

            input[type="text"], textarea, select { width: 100%; padding: 12px; font-size: 16px; box-sizing: border-box; border: 1px solid #cbd5e0; border-radius: 6px; margin-top: 6px; }
            audio { width: 100%; margin-top: 10px; }
            .pinyin-toggle { display: inline-flex; align-items: center; gap: 6px; margin-top: 10px; color: #4a5568; }
            .pinyin-toggle[hidden] { display: none; }
            .pinyin-toggle input { margin: 0; }
            .transcript { min-height: 58px; padding: 10px 12px; margin-top: 6px; background: #f7fafc; border: 1px solid #cbd5e0; border-radius: 6px; line-height: 1.8; }
            .sample-audio-section { margin: 12px 0 16px; padding: 10px 12px; background: #f7fafc; border: 1px solid #cbd5e0; border-radius: 8px; }
            .sample-audio-section label { display: block; margin-bottom: 4px; }
            .sample-audio-section audio { display: block; width: 100%; margin-top: 4px; }
            .sample-audio-error { margin: 6px 0 0; color: #b42318; font-size: 13px; }
            .transcript-char { display: inline-flex; flex-direction: column; align-items: center; margin-right: 8px; }
            .transcript-pinyin { font-size: 12px; color: #718096; line-height: 1.2; }

            .pronunciation-results { margin: 0 0 18px; }
            .pronunciation-results h3 { margin: 0 0 4px; }
            .results-note { margin: 0 0 12px; color: #4a5568; font-size: 13px; line-height: 1.45; }
            .result-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; }
            .word-card { min-width: 0; min-height: 112px; display: flex; flex-direction: column; align-items: stretch; justify-content: center; gap: 5px; padding: 10px; border: 1px solid #e2e8f0; border-radius: 7px; color: #1a202c; font: inherit; cursor: pointer; }
            .word-card:hover, .word-card:focus-visible { outline: 2px solid #3182ce; outline-offset: 2px; }
            .word-card.likely-correct { border-color: transparent; background: #eaf6ee; }
            .word-card.practice { border-color: transparent; background: #fae9e9; }
            .word-card.uncertain { border-color: transparent; background: #fbf3e8; }
            .word-card-label { align-self: center; font-size: 17px; font-weight: 650; padding-bottom: 6px; }       /* room for the line under a marked character */
            .char-practice { text-decoration: underline wavy #b42318; text-decoration-thickness: 1.5px; text-underline-offset: 5px; }
            .char-doubt { text-decoration: underline dotted #9a6700; text-decoration-thickness: 2px; text-underline-offset: 5px; }
            .word-card-tone-sequence { color: #607386; font-size: 12px; letter-spacing: 0.02em; text-align: center; }
            .word-card-status { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; }
            .word-card-advice { display: block; margin-top: 4px; color: #4a5568; font-size: 12.5px; line-height: 1.45; text-align: left; }
            .results-advice { margin: 0 0 12px; padding: 9px 12px; border-left: 3px solid #d69e2e; background: #fffaf0; color: #744210; font-size: 13px; line-height: 1.45; }
            .status-icon { display: inline-grid; place-items: center; width: 20px; height: 20px; border: 1.5px solid currentColor; border-radius: 50%; font-size: 13px; line-height: 1; }
            .status-icon.half { background: linear-gradient(90deg, currentColor 50%, transparent 50%); }
            .json-output { display: none; /* temporarily hidden: remove this line to show "View JSON response" again */ margin-top: 16px; border-top: 1px solid #e2e8f0; }
            .json-output > summary { padding: 10px 2px; color: #2b6cb0; font-size: 14px; font-weight: 600; cursor: pointer; }
            .json-output pre { max-height: 420px; overflow: auto; margin: 0; padding: 12px; border: 1px solid #e2e8f0; border-radius: 6px; background: #f7fafc; color: #243447; font: 12px/1.5 Consolas, 'Courier New', monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
            .word-card.likely-correct .word-card-status { color: #147a42; }
            .word-card.practice .word-card-status { color: #b42318; }
            .word-card.uncertain .word-card-status { color: #9a6700; }
            .syllable-details { margin-top: 18px; border-top: 1px solid #e2e8f0; }
            .syllable-details > summary { min-height: 24px; padding: 12px 2px; color: #1a202c; font-weight: 650; cursor: pointer; }
            .tone-table-wrap { overflow-x: auto; }
            .tone-table { width: 100%; border-collapse: collapse; text-align: left; }
            .tone-table th, .tone-table td { padding: 9px 8px; border-bottom: 1px solid #edf1f5; vertical-align: top; }
            .tone-table th { color: #526273; font-size: 12px; font-weight: 600; }
            .tone-table td { font-size: 14px; }
            .tone-table tr.likely-correct td:last-child { color: #147a42; }
            .tone-table tr.practice td:last-child { color: #b42318; }
            .tone-table tr.uncertain td:last-child { color: #9a6700; }
            .tone-table tr.selected-row { background: #f3f7fb; }
            .tone-char { font-size: 18px; font-weight: 650; }
            .tone-pinyin, .tone-rule { display: block; margin-top: 2px; color: #607386; font-size: 12px; }
            .tone-shape-note { margin: 0 2px 8px; color: #607386; font-size: 12px; line-height: 1.45; }
            .tone-expected { display: inline-flex; align-items: center; gap: 8px; }
            .tone-shape { width: 34px; height: 22px; flex: none; overflow: visible; }
            .tone-shape path { fill: none; stroke: #607386; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
            .tone-shape .tone-shape-alt { opacity: 0.45; }
            .pitch-details { margin: 10px 0 20px; border-top: 1px solid #e2e8f0; }
            .pitch-details > summary { min-height: 24px; padding: 12px 4px; color: #2b6cb0; font-weight: 600; cursor: pointer; }
            .pitch-clause { margin: 8px 0 16px; }
            .pitch-clause h3 { font-size: 15px; margin: 8px 0; }

            .pitch-legend { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; color: #4a5568; font-size: 12px; margin: 16px 0 8px; }
            .pitch-controls { margin-bottom: 20px; }
            .pitch-controls select { max-width: 300px; }
            .pitch-note { font-size: 13px; color: #4a5568; line-height: 1.5; margin: 8px 0; }
            .pitch-key { display: inline-flex; align-items: center; gap: 5px; }
            .pitch-chart { max-width: 100%; overflow-x: auto; }
            @media (max-width: 480px) { .result-groups { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); } }
            .pitch-swatch { display: inline-block; width: 18px; border-top: 3px solid; }
            .pitch-swatch.reference { border-color: #2367b0; }
            .pitch-swatch.learner { border-color: #b85c00; border-top-style: dashed; }
            .pitch-chart { overflow-x: auto; margin: 12px 0; border: 1px solid #e2e8f0; border-radius: 6px; background: white; }
            .pitch-chart svg { display: block; width: 100%; height: 180px; }
            .pitch-chart .reference-line { fill: none; stroke: #2367b0; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
            .pitch-chart .learner-line { fill: none; stroke: #b85c00; stroke-width: 2.5; stroke-dasharray: 5 3; stroke-linecap: round; stroke-linejoin: round; }
            .pitch-chart .pitch-grid { stroke: #edf2f7; stroke-width: 1; }
            .pitch-chart .pitch-label { fill: #4a5568; font-size: 12px; }
            .pitch-chart .pitch-char { fill: #2d3748; font-size: 16px; }
            .pitch-chart.unavailable { padding: 12px; font-size: 13px; color: #718096; }
            .feedback-panel { margin-top: 20px; padding: 16px; border: 1px solid #bee3f8; border-radius: 8px; background: #f7fbff; }
            .feedback-panel h3 { margin: 0 0 6px; }
            .feedback-panel p { margin: 6px 0 12px; color: #4a5568; }
            .feedback-ratings { display: flex; flex-wrap: wrap; gap: 8px; border: 0; padding: 0; margin: 0 0 12px; }
            .feedback-ratings label { display: inline-flex; align-items: center; gap: 5px; padding: 7px 10px; background: white; border: 1px solid #cbd5e0; border-radius: 6px; cursor: pointer; }
            .feedback-panel textarea { min-height: 70px; }
            #feedbackSubmitBtn { background: #2b6cb0; color: white; margin-top: 10px; }
            #feedbackStatus { min-height: 1.4em; margin-bottom: 0; }

            .hanzi { font-size: 22px; font-weight: bold; color: #2d3748; display: block; }
            .pinyin { font-size: 12px; color: #718096; display: block; }
            .transcript-char.optional { opacity: 0.6; }
            .transcript-char.optional > span:first-child { border-bottom: 2px dotted #a0aec0; }
            #chapter5OptionalHint { margin: 6px 0 0; }
            #chapter5OptionalHint[hidden] { display: none; }
            .tone-tag { font-size: 10px; margin-top: 4px; padding: 2px 6px; border-radius: 4px; display: inline-block; }
            .tone-likely-correct { background: #c6f6d5; color: #22543d; }
            .tone-likely-incorrect { background: #fed7d7; color: #742a2a; }
            .tone-uncertain { background: #feebc8; color: #744210; }
            .word-card.not-heard { border-color: transparent; background: #eceff3; }
            .word-card.not-heard .word-card-status { color: #4a5568; }

            /* ---- compact layout, phones, and the temporarily hidden upload ---- */
            body { margin: 18px auto; padding: 0 16px 32px; }
            .card { padding: 18px 20px; margin-bottom: 14px; }
            .student-heading { gap: 8px 16px; }
            .student-heading h2 { font-size: 17px; }
            .language-picker select { min-width: 0; padding: 6px 8px; }
            #activeStudentPanel { margin-top: 8px; gap: 6px 10px; }
            #activeStudentPanel .upload-hint { margin: 0; }
            #changeProfileBtn { margin-top: 0; padding: 8px 12px; font-size: 14px; }
            #calibrateVoiceBtn { margin-left: auto; }
            h2[data-i18n="appTitle"] { font-size: 21px; margin: 0 0 6px; }
            p[data-i18n="practiceInstructions"] { margin: 0 0 10px; font-size: 14px; color: #4a5568; }
            #busyOverlay { border: 0; padding: 0; background: transparent; width: 100vw; height: 100vh; max-width: none; max-height: none; overflow: visible; }
            #busyOverlay[open] { display: flex; align-items: center; justify-content: center; }
            #busyOverlay::backdrop { background: rgba(24, 39, 56, 0.55); backdrop-filter: blur(2px); }
            .busy-card { box-sizing: border-box; background: #fff; border-radius: 14px; padding: 26px 30px; text-align: center; max-width: min(320px, calc(100vw - 48px)); box-shadow: 0 18px 60px rgba(15, 35, 55, 0.3); }
            .busy-spinner { width: 56px; height: 56px; box-sizing: border-box; margin: 0 auto 14px; border: 6px solid #dbe5f1; border-top-color: #2b6cb0; border-radius: 50%; animation: status-spin 0.9s linear infinite; }
            #busyMessage { margin: 0 0 4px; font-weight: 700; color: #1a202c; line-height: 1.4; }
            .busy-sub { margin: 0; color: #607386; font-size: 13px; }
            @media (prefers-reduced-motion: reduce) { .busy-spinner { animation-duration: 1.8s; } }
            /* Topics: one tidy card with the item, its meaning and where you are */
            .chapter5-word-controls { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 8px 10px; margin: 0 0 8px; padding: 8px 10px; }
            .chapter5-word-controls[hidden] { display: none; }
            .chapter5-word-controls button { min-width: 0; padding: 8px 12px; font-size: 14px; background: #edf2f7; color: #2d3748; }
            .chapter5-word-controls button:disabled { opacity: 0.45; cursor: default; }
            #previousChapter5WordBtn::before { content: "‹"; margin-right: 6px; font-size: 18px; line-height: 1; }
            #nextChapter5WordBtn::after { content: "›"; margin-left: 6px; font-size: 18px; line-height: 1; }
            .chapter5-word-info { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 1px 8px; min-width: 0; }
            #chapter5WordType { grid-row: 1; grid-column: 1; }
            #chapter5WordProgress { grid-row: 1; grid-column: 2; }
            #chapter5WordMeaning { grid-row: 2; }
            #chapter5WordCategory { grid-row: 3; }
            #chapter5WordType { justify-self: start; padding: 1px 9px; border-radius: 999px; background: #e3eefb; color: #2b6cb0; font-size: 12px; font-weight: 600; }
            #chapter5WordProgress { justify-self: end; margin: 0; color: #64748b; font-size: 12px; }
            #chapter5WordMeaning { grid-column: 1 / -1; color: #1a202c; font-size: 17px; font-weight: 600; line-height: 1.35; overflow-wrap: anywhere; }
            #chapter5WordCategory { grid-column: 1 / -1; margin: 0; color: #64748b; font-size: 12.5px; }
            #chapter5ReferenceLabel { order: 3; flex: 1 1 100%; margin: 0; color: #64748b; font-size: 12px; }
            /* My own text: a normal font (not monospace), grows with the text, no empty preview bar */
            #scriptInput { font-family: inherit; font-size: 18px; line-height: 1.6; min-height: 3.4em; max-height: 40vh; resize: vertical; field-sizing: content; }
            #transcriptDisplay:empty { display: none; }
            #customTextHint { margin: 0 0 4px; font-size: 13px; }
            button, input, select, textarea { font-family: inherit; }              /* form controls ignore the page font by default */
            :lang(hi) body { line-height: 1.7; }                                    /* Devanagari needs more height than Latin */
            :lang(hi) button, :lang(hi) select, :lang(hi) input { line-height: 1.5; }
            .tight { margin-bottom: 10px; }
            #bookSamplePicker label { font-size: 14px; }
            #bookSamplePicker select { margin-top: 4px; padding: 9px 10px; }
            .sample-audio-section { margin: 8px 0 10px; padding: 6px 10px; display: flex; align-items: center; flex-wrap: wrap; gap: 2px 10px; }
            .sample-audio-section[hidden] { display: none; }
            .sample-audio-section label { margin: 0; font-size: 13px; color: #4a5568; flex: 0 0 auto; }
            .sample-audio-section audio { flex: 1 1 220px; width: auto; height: 36px; margin: 0; }
                .sample-audio-section .upload-hint, .sample-audio-error { flex: 1 1 100%; margin: 0; }
            .page-grid { display: block; }
            .leaderboard h2 { margin: 0 0 6px; font-size: 1.05rem; }
            .leaderboard-list { list-style: none; margin: 8px 0 0; padding: 0; }
            .leaderboard-row { display: grid; grid-template-columns: 30px minmax(0, 1fr) auto; gap: 8px; align-items: baseline; padding: 6px 8px; border-radius: 8px; }
            .leaderboard-row:nth-child(odd) { background: #f7fafc; }
            .leaderboard-row.me, .leaderboard-me { background: #ebf8ff; font-weight: 600; }
            .leaderboard-rank { color: #718096; font-variant-numeric: tabular-nums; text-align: right; }
            .leaderboard-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .leaderboard-points { font-variant-numeric: tabular-nums; }
            .leaderboard-lessons { display: block; font-size: 11px; font-weight: 400; color: #718096; text-align: right; }
            .leaderboard-me { margin: 10px 0 0; padding: 8px; border-radius: 8px; }
            .leaderboard-me[hidden] { display: none; }
            .board-tabs { display: flex; gap: 6px; margin: 4px 0 8px; }
            .board-tabs button { flex: 1; padding: 6px 8px; font-size: 13px; background: #edf2f7; color: #2d3748; }
            .board-tabs button.active { background: #2b6cb0; color: white; }
            .recent { margin-top: 14px; padding-top: 10px; border-top: 1px solid #e2e8f0; }
            .recent[hidden] { display: none; }
            .recent h3 { margin: 0 0 6px; font-size: 0.95rem; }
            .recent-list { list-style: none; margin: 0; padding: 0; }
            .recent-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: baseline; width: 100%; padding: 5px 8px; border-radius: 8px; background: transparent; color: inherit; font-size: 14px; font-weight: 400; text-align: left; }
            .recent-row:hover:not(:disabled), .recent-row:focus-visible { background: #edf2f7; }
            .recent-row:disabled { cursor: default; opacity: 1; }
            .recent-script { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .recent-points { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }
            .recent-time { display: block; font-size: 11px; font-weight: 400; color: #718096; }
            .result-actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 4px; }
            .result-actions button { padding: 9px 16px; font-size: 15px; }
            .result-actions .retry-btn.recording { background-color: #1a4f80; animation: pulse 1.2s infinite; }
            .result-actions .next-item-btn { background: #edf2f7; color: #2d3748; }
            .result-actions .next-item-btn:disabled { opacity: 0.5; }
            .item-progress { margin: 6px 0 0; }
            .item-progress.practiced { color: #276749; font-weight: 600; }
            #practiceMode { scroll-margin-top: 12px; }
            @media (min-width: 1280px) {
                body { max-width: 1400px; }
                .page-grid { display: grid; grid-template-columns: minmax(0, 1040px) 300px; gap: 20px; align-items: start; justify-content: center; }
                .leaderboard { position: sticky; top: 18px; max-height: calc(100vh - 36px); overflow-y: auto; }
            }
            .custom-sample { margin: 8px 0 10px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px 10px; }
            .custom-sample[hidden], .custom-sample audio[hidden] { display: none; }
            .custom-sample audio { flex: 1 1 220px; width: auto; height: 36px; margin: 0; }
            .custom-sample .upload-hint, .custom-sample .sample-audio-error { flex: 1 1 100%; margin: 0; }
            #scriptBlock { position: relative; }
            #scriptBlock > label { font-size: 13px; color: #4a5568; }
            #scriptInput[readonly] { display: none; }          /* the same text is shown with pinyin right below */
            #scriptInput { margin-top: 4px; }
            #pinyinToggle { position: absolute; right: 0; top: 0; margin: 0; font-size: 13px; }
            .transcript { min-height: 0; padding: 8px 10px; margin-top: 4px; }
            .record-controls { margin-bottom: 10px; }
            .upload-controls { display: none; /* temporarily hidden: remove this line to show "Upload your audio" again */ }
            @media (max-width: 600px) {
                body { margin: 8px auto; padding: 0 10px 24px; }
                .card { padding: 14px; border-radius: 10px; margin-bottom: 10px; }
                .student-heading { flex-wrap: nowrap; }
                .language-picker > span { display: none; }
                #activeStudentLabel, #voiceCalibrationLabel { flex: 1 1 100%; }
                #activeStudentLabel { font-weight: 600; }
                #activeStudentPanel > button { flex: 1 1 calc(50% - 6px); }
                #calibrateVoiceBtn { margin-left: 0; }
                h2[data-i18n="appTitle"] { font-size: 18px; }
                button { padding: 10px 14px; font-size: 15px; }
                input[type="text"], textarea, select { padding: 10px; }
                #speakerDialog, #voiceCalibrationDialog { padding: 16px; }
                .cal-step { padding: 10px; }
                .cal-tile .han { font-size: 22px; }
                .chapter5-word-controls { padding: 8px; gap: 6px; }
                .chapter5-word-controls button { font-size: 0; padding: 8px 12px; }
                #previousChapter5WordBtn::before, #nextChapter5WordBtn::after { margin: 0; font-size: 26px; }
            }

        </style>
    </head>
    <body>
        <div class="page-grid"><div class="page-main">
        <div class="card" id="studentCard">
            <div class="student-heading">
                <h2 data-i18n="speakerProfile">Speaker profile</h2>
                <label class="language-picker" for="languageSelect"><span data-i18n="language">Language</span>
                    <select id="languageSelect" onchange="changeLanguage(this.value)" aria-label="Language">
                        <option value="en">English</option>
                        <option value="zh-TW" data-i18n="languageZh">Traditional Chinese</option>
                        <option value="vi" data-i18n="languageVi">Vietnamese</option>
                        <option value="id">Bahasa Indonesia</option>
                        <option value="hi">हिन्दी</option>
                    </select>
                </label>
            </div>
            <div id="activeStudentPanel" hidden>
                <span id="activeStudentLabel"></span>
                <span id="voiceCalibrationLabel" class="upload-hint"></span>
                <button type="button" id="calibrateVoiceBtn" onclick="openVoiceCalibrationDialog(true)" data-i18n="calibrateVoice">Calibrate voice</button>
                <button type="button" id="changeProfileBtn" onclick="changeStudentProfile()" data-i18n="changeSpeaker">Change speaker</button>
            </div>
        </div>

        <dialog id="speakerDialog" aria-labelledby="speakerDialogTitle" aria-describedby="speakerDialogDescription">
            <h2 id="speakerDialogTitle" data-i18n="speakerProfile">Speaker profile</h2>
            <p id="speakerDialogDescription" data-i18n="profilePrompt">Enter your name and participant group before speaking practice.</p>
            <p class="upload-hint" data-i18n="profileHint">Use a name that distinguishes you from other people; the same name always continues the same practice history.</p>
            <form id="studentForm" onsubmit="saveStudentProfile(event)">
                <div class="profile-fields">
                    <label for="studentName"><span data-i18n="name">Name</span>
                        <input type="text" id="studentName" maxlength="100" autocomplete="name" required>
                    </label>
                    <label for="studentGroup"><span data-i18n="participantGroup">Participant group</span>
                        <select id="studentGroup" required>
                            <option value="" disabled selected data-i18n="chooseOne">Choose one...</option>
                            <option value="taiwanese" data-i18n="taiwanese">Taiwanese</option>
                            <option value="foreigner" data-i18n="foreigner">Foreigner</option>
                        </select>
                    </label>
                </div>
                <p id="speakerFormError" role="alert" aria-live="polite"></p>
                <button id="saveProfileBtn" type="submit" data-i18n="continue">Continue</button>
                <button id="cancelProfileBtn" type="button" onclick="closeStudentDialog()" hidden data-i18n="cancel">Cancel</button>
            </form>
        </dialog>

        <dialog id="voiceCalibrationDialog" aria-labelledby="voiceCalibrationTitle" aria-describedby="voiceCalibrationInstruction" data-state="idle">
            <button type="button" id="voiceCalibrationCloseBtn" class="cal-close" onclick="deferVoiceCalibration()" aria-label="Close" data-i18n-aria-label="calClose">&times;</button>
            <h2 id="voiceCalibrationTitle" data-i18n="voiceCalibrationTitle">Set up your voice</h2>
            <p id="voiceCalibrationInstruction" class="cal-why" data-i18n="voiceCalibrationPrompt">A 10-second sample lets us judge your tones against your own voice, so the feedback is fair.</p>
            <ol class="cal-steps">
                <li class="cal-step">
                    <div class="cal-head"><span class="cal-num" aria-hidden="true">1</span><span data-i18n="calStep1">Read these five sounds aloud, twice</span></div>
                    <p class="cal-hint" data-i18n="calStep1Hint">Use your own comfortable, normal voice. No need to strain or to match the sample's pitch.</p>
                    <button type="button" id="calListenAllBtn" class="cal-listen-all" onclick="playCalibrationSound(0)" data-i18n="calListenAll">&#9654; Listen to the five sounds</button>
                    <div id="voiceCalibrationScript">
                        <div class="cal-tile"><button type="button" class="cal-play" data-tone="1" onclick="playCalibrationSound(1)" aria-label="Listen" data-i18n-aria-label="calListen">&#9654;</button><span class="han" lang="zh-Hant">媽</span><span class="py">mā</span><span class="tone" data-i18n="calToneHigh">high</span></div>
                        <div class="cal-tile"><button type="button" class="cal-play" data-tone="2" onclick="playCalibrationSound(2)" aria-label="Listen" data-i18n-aria-label="calListen">&#9654;</button><span class="han" lang="zh-Hant">麻</span><span class="py">má</span><span class="tone" data-i18n="calToneRising">rising</span></div>
                        <div class="cal-tile"><button type="button" class="cal-play" data-tone="3" onclick="playCalibrationSound(3)" aria-label="Listen" data-i18n-aria-label="calListen">&#9654;</button><span class="han" lang="zh-Hant">馬</span><span class="py">mǎ</span><span class="tone" data-i18n="calToneDip">dip</span></div>
                        <div class="cal-tile"><button type="button" class="cal-play" data-tone="4" onclick="playCalibrationSound(4)" aria-label="Listen" data-i18n-aria-label="calListen">&#9654;</button><span class="han" lang="zh-Hant">罵</span><span class="py">mà</span><span class="tone" data-i18n="calToneFalling">falling</span></div>
                        <div class="cal-tile"><button type="button" class="cal-play" data-tone="5" onclick="playCalibrationSound(5)" aria-label="Listen" data-i18n-aria-label="calListen">&#9654;</button><span class="han" lang="zh-Hant">嗎</span><span class="py">ma</span><span class="tone" data-i18n="calToneLight">light</span></div>
                    </div>
                </li>
                <li class="cal-step">
                    <div class="cal-head"><span class="cal-num" aria-hidden="true">2</span><span data-i18n="calStep2">Press the button and start reading</span></div>
                    <button id="voiceCalibrationRecordBtn" type="button" onclick="toggleVoiceCalibrationRecording()">Start recording</button>
                    <div class="cal-live">
                        <div class="cal-live-row"><span data-i18n="micLevel">Microphone level</span></div>
                        <div class="cal-meter" aria-hidden="true"><i id="calibrationLevelBar"></i></div>
                        <div class="cal-timer-row">
                            <div class="cal-timer" aria-hidden="true"><i id="calibrationTimerBar"></i></div>
                            <span id="calibrationTimerText">0 s</span>
                        </div>
                        <p id="calibrationNoSound" class="cal-nosound" data-i18n="micNoSound" hidden>We can't hear you. Speak a little louder or check your microphone.</p>
                    </div>
                </li>
            </ol>
            <p id="voiceCalibrationStatus" role="status" aria-live="polite"></p>
            <div id="voiceCalibrationControls">
                <button id="voiceCalibrationLaterBtn" type="button" onclick="deferVoiceCalibration()" data-i18n="voiceCalibrationLater">Skip for now</button>
            </div>
            <p class="cal-privacy" data-i18n="calPrivacy">Your recording is not kept. Only a few pitch numbers are saved.</p>
        </dialog>

        <dialog id="busyOverlay" aria-label="Working">
            <div class="busy-card" role="status" aria-live="polite">
                <div class="busy-spinner" aria-hidden="true"></div>
                <p id="busyMessage"></p>
                <p class="busy-sub" data-i18n="busyWait">This usually takes a few seconds.</p>
            </div>
        </dialog>

        <div class="card">
            <h2 data-i18n="appTitle">Taiwan Mandarin Pronunciation Practice</h2>
            <p data-i18n="practiceInstructions">Choose a human voice sample, listen to it, then record yourself or upload an audio file reading the matching script:</p>
            <fieldset id="practiceMode" class="practice-mode">
                <legend data-i18n="practiceSource">Practice text</legend>
                <label><input type="radio" name="practiceMode" value="book" checked onchange="changePracticeMode()"><span data-i18n="bookTextMode">Book sample</span></label>
                <label><input type="radio" name="practiceMode" value="chapter5_word" onchange="changePracticeMode()"><span data-i18n="chapter5WordMode">Topics</span></label>
                <label><input type="radio" name="practiceMode" value="custom" onchange="changePracticeMode()"><span data-i18n="customTextMode">My own text</span></label>
            </fieldset>

            <div id="chapter5CategoryPicker" class="chapter5-category-picker" hidden>
                <label for="chapter5CategorySelect"><b data-i18n="chapter5Topic">Topic</b></label>
                <select id="chapter5CategorySelect" onchange="changeChapter5CategoryFilter()">
                    <option value="" data-i18n="chapter5AllTopics">All topics</option>
                </select>
            </div>

            <div id="chapter5WordControls" class="chapter5-word-controls" hidden>
                <button id="previousChapter5WordBtn" type="button" onclick="moveChapter5Word(-1)" data-i18n="previousWord">Previous</button>
                <div class="chapter5-word-info" aria-live="polite">
                    <b id="chapter5WordType"></b>
                    <div id="chapter5WordMeaning"></div>
                    <div id="chapter5WordCategory" class="chapter5-word-category"></div>
                    <div id="chapter5WordProgress" class="chapter5-word-progress"></div>
                </div>
                <button id="nextChapter5WordBtn" type="button" onclick="moveChapter5Word(1)" data-i18n="nextWord">Next</button>
            </div>

            <div id="bookSamplePicker" class="tight">
                <label for="sampleSelect"><b data-i18n="sampleRecording">Sample recording (Lessons 5-8):</b></label>
                <select id="sampleSelect" onchange="sampleTouched = true; selectSample()" disabled>
                    <option data-i18n="loadingSamples">Loading sample recordings...</option>
                </select>
            </div>

            <div class="sample-audio-section" id="sampleAudioSection" hidden>
                <p id="chapter5ReferenceLabel" class="upload-hint" data-i18n="chapter5ReferenceAudio" hidden>Listen to the word audio as a reference. It is not used to score your recording.</p>
                <label for="sampleAudio"><b data-i18n="sampleAudioLabel">Listen to sample audio:</b></label>
                <audio id="sampleAudio" controls preload="metadata" aria-label="Sample audio" data-i18n-aria-label="sampleAudioLabel"></audio>
                <p id="sampleAudioError" class="sample-audio-error" role="status" data-i18n="sampleAudioLoadError" hidden>Sample audio could not be loaded. Try selecting the item again.</p>
            </div>

            <div id="scriptBlock" class="tight">
                <label for="scriptInput"><b id="scriptLabel" data-i18n="matchingScript">Matching script:</b></label>
                <p id="customTextHint" class="upload-hint" data-i18n="customTextHint" hidden>Enter any Mandarin text. It does not need to appear in a textbook.</p>
                <textarea id="scriptInput" rows="2" readonly oninput="handleScriptInput()" data-i18n-placeholder="customTextPlaceholder"></textarea>
                <div id="transcriptDisplay" class="transcript" aria-label="Matching script with pinyin" data-i18n-aria-label="scriptWithPinyin"></div>
                <p id="itemProgress" class="upload-hint item-progress" role="status" hidden></p>
                <p id="chapter5OptionalHint" class="upload-hint" hidden></p>
                <label id="pinyinToggle" class="pinyin-toggle">
                    <input type="checkbox" id="showPinyin" checked onchange="refreshPinyinDisplay()">
                    <span data-i18n="showPinyin">Show pinyin</span>
                </label>
                <div id="customSampleBlock" class="custom-sample" hidden>
                    <button type="button" id="customSampleBtn" onclick="playCustomSample()" data-i18n="customSampleListen">Listen to a sample</button>
                    <audio id="customSampleAudio" controls preload="none" hidden aria-label="Sample" data-i18n-aria-label="customSampleListen"></audio>
                    <p class="upload-hint" data-i18n="customSampleNote">A computer voice reading your text a little slowly. It is only a guide; a word may sound slightly different from a teacher\u2019s reading.</p>
                    <p id="customSampleError" class="sample-audio-error" role="status" hidden></p>
                </div>
            </div>

            <div class="record-controls">
                <button id="recordBtn" onclick="toggleRecording()" disabled data-i18n="startRecording">Start recording</button>
                <button id="stopBtn" onclick="stopRecording()" disabled data-i18n="stopAndAssess">Stop &amp; assess</button>
            </div>
            <div id="recordingIndicator" class="rec-indicator" hidden>
                <span class="rec-dot" aria-hidden="true"></span>
                <span id="recordingClock" class="rec-clock">0:00</span>
                <div class="rec-level" role="img" aria-label="Microphone level" data-i18n-aria-label="micLevel"><i id="recordingLevelBar"></i></div>
            </div>
            <p id="recordingNoSound" class="rec-nosound" data-i18n="micNoSound" hidden>We can't hear you. Speak a little louder or check your microphone.</p>

            <div id="recordedAudioContainer" hidden>
                <b id="recordedAudioLabel" data-i18n="latestRecording">Your latest microphone recording:</b>
                <audio id="recordedAudio" controls preload="metadata" aria-labelledby="recordedAudioLabel"></audio>
            </div>

            <div class="upload-controls">
                <label for="audioFileInput"><b data-i18n="uploadAudio">Upload your audio:</b></label>
                <p class="upload-hint" id="audioFileHint" data-i18n="uploadHint">Choose a recording of yourself reading the selected script. Supported formats: WAV, MP3, M4A, OGG, FLAC, WebM.</p>
                <input type="file" id="audioFileInput" accept=".wav,.mp3,.m4a,.ogg,.flac,.webm" onchange="selectAudioFile()" aria-describedby="audioFileHint">
                <audio id="uploadedAudio" controls preload="metadata" aria-label="Your selected audio" data-i18n-aria-label="selectedAudio" hidden></audio>
                <button type="button" id="uploadBtn" onclick="uploadAudio()" disabled data-i18n="uploadAssess">Upload &amp; assess</button>
            </div>
            <p id="status" role="status" aria-live="polite" aria-atomic="true"></p>
        </div>

        <!-- Visual results card -->
        <div id="resultCard" class="card" style="display: none;">
            <div class="results-header">
                <div>
                    <h2 data-i18n="resultsTitle">Reading practice results</h2>
                    <p id="targetScriptDisplay" style="color: #718096; margin: 0;"></p>
                </div>
                <div class="result-actions">
                    <button type="button" id="retryBtn" class="retry-btn" onclick="toggleRecording()">Read again</button>
                    <button type="button" id="nextItemBtn" class="next-item-btn" onclick="goToNextItem()" data-i18n="resultNext">Next</button>
                </div>
            </div>

            <div id="pronunciationResults"></div>
            <details id="pitchDetails" class="pitch-details">
                <summary data-i18n="viewPitch">View pitch graphs (optional)</summary>
                <div class="pitch-controls">
                    <label for="pitchScale"><b data-i18n="pitchDisplay">Pitch display:</b></label>
                    <select id="pitchScale" onchange="refreshPitchPlots()">
                        <option value="relative" data-i18n="relativePitch">Relative pitch</option>
                        <option value="hz" data-i18n="f0Hz">F0 (Hz)</option>
                        <option value="midi" data-i18n="midiNote">MIDI note</option>
                    </select>
                    <div class="pitch-legend">
                        <span class="pitch-key"><i class="pitch-swatch learner" aria-hidden="true"></i><span data-i18n="recordingLine">Your pitch contour</span></span>
                    </div>
                    <p class="pitch-note" data-i18n="pitchNote">Tone feedback uses only your pitch contour; sample audio is not scored.</p>
                </div>
                <div id="clausesContainer"></div>
            </details>
            <section id="feedbackPanel" class="feedback-panel" hidden aria-labelledby="feedbackTitle">
                <h3 id="feedbackTitle" data-i18n="quickFeedback">Quick feedback</h3>
                <p data-i18n="feedbackQuestion">How helpful was this practice? 1 = not helpful, 5 = very helpful.</p>
                <form id="feedbackForm" onsubmit="submitUserFeedback(event)">
                    <fieldset class="feedback-ratings" aria-label="How helpful was this practice?" data-i18n-aria-label="feedbackQuestion">
                        <label><input type="radio" name="feedbackRating" value="1" required> 1</label>
                        <label><input type="radio" name="feedbackRating" value="2"> 2</label>
                        <label><input type="radio" name="feedbackRating" value="3"> 3</label>
                        <label><input type="radio" name="feedbackRating" value="4"> 4</label>
                        <label><input type="radio" name="feedbackRating" value="5"> 5</label>
                    </fieldset>
                    <label for="feedbackComment" data-i18n="feedbackImprove">Anything we should improve? (optional)</label>
                    <textarea id="feedbackComment" maxlength="2000" placeholder="Share a short comment" data-i18n-placeholder="feedbackPlaceholder"></textarea>
                    <button id="feedbackSubmitBtn" type="submit" data-i18n="sendFeedback">Send feedback</button>
                </form>
                <p id="feedbackStatus" role="status" aria-live="polite"></p>
            </section>
            <div class="result-actions">
                <button type="button" class="retry-btn" onclick="toggleRecording()">Read again</button>
                <button type="button" class="next-item-btn" onclick="goToNextItem()" data-i18n="resultNext">Next</button>
            </div>
        </div>
        </div>
        <aside id="leaderboardCard" class="card leaderboard" aria-labelledby="leaderboardTitle">
            <h2 id="leaderboardTitle" data-i18n="leaderboardTitle">Top 10</h2>
            <div class="board-tabs" role="tablist" aria-label="Top 10">
                <button type="button" id="boardTabPoints" class="active" role="tab" aria-selected="true" data-i18n="boardTabPoints" onclick="setBoardView('points')">Points</button>
                <button type="button" id="boardTabAccuracy" role="tab" aria-selected="false" data-i18n="boardTabAccuracy" onclick="setBoardView('accuracy')">Accuracy</button>
            </div>
            <p id="leaderboardHelp" class="upload-hint" data-i18n="leaderboardHelp">Points: the best attempt of each lesson (a Good syllable counts 1, an Okay one &frac12;), added up. Your own text does not count.</p>
            <ol id="leaderboardList" class="leaderboard-list"></ol>
            <p id="leaderboardEmpty" class="upload-hint" hidden>No scores yet. Be the first!</p>
            <p id="leaderboardMe" class="leaderboard-me" hidden></p>
            <section id="recentCard" class="recent" aria-labelledby="recentTitle" hidden>
                <h3 id="recentTitle" data-i18n="recentTitle">My recent results</h3>
                <ol id="recentList" class="recent-list"></ol>
                <p id="recentEmpty" class="upload-hint" data-i18n="recentEmpty" hidden>Your results will appear here.</p>
            </section>
        </aside>
        </div>



        <script>
            let mediaRecorder;
            let voiceCalibrationRecorder = null;
            let voiceCalibrationStream = null;
            let voiceCalibrationChunks = [];
            let voiceCalibrationStartedAt = 0;
            let voiceCalibrationTimer = null;
            let isVoiceCalibrationRecording = false;
            let isVoiceCalibrationProcessing = false;
            let isRecording = false;
            let isProcessing = false;
            let calibrationHadError = false;   // "Do this later" only appears once calibration has failed (or is done)
            let calibrationOutcome = "";       // "", "success" or "error": what the dialog shows after an attempt
            let calibrationCanStop = false;    // the sample is long enough to stop (5 s)
            let calibrationClock = null;
            let stopCalibrationMeter = null;
            let practiceIndicatorTimer = null;
            let stopPracticeMeter = null;
            const CALIBRATION_MIN_MS = 5200;   // the server needs 5 s of speech; a little extra for the start-up lag
            const CALIBRATION_MAX_MS = 10000;
            let busyTimer = null;      // declared with the other state: the first status render happens before the helpers below
            let isSavingStudent = false;
            let sampleLessons = [];
            let progressData = { lessons: {}, recent: [] };   // what the learner has practised (from /api/progress): marks for the lists, latest results
            let sampleTouched = false;                      // the learner picked a sample themselves: the page stops choosing the starting one
            let chapter5WordItems = [];
            let chapter5Categories = [];
            let chapter5CategoryFilter = "";
            let chapter5WordIndex = 0;
            let lastAssessment = null;
            let activeStudent = null;
            let calibrationPaceHint = "";
            let customTextDraft = "";
            let customPinyin = { script: "", values: [] };
            let customPinyinTimer = null;
            let activeFeedbackAttemptId = null;
            let uploadedAudioUrl = null;
            let recordedAudioUrl = null;

            const LANGUAGE_TEXT = {
                en: {
                    language: "Language", languageEn: "English", languageZh: "中文（繁體）", languageVi: "Tiếng Việt",
                    speakerProfile: "Speaker profile", changeSpeaker: "Change speaker",
                    calListen: "Listen to this sound", calListenAll: "\u25b6 Listen to the five sounds",
                    calibrationSavedSlower: "Your voice is saved. Tip: next time read a little slower and stretch each sound \u2013 it is easier to recognise.",
                    paceTooFast: "Some sounds were too quick to analyse. Read a little slower, or stretch each sound a bit, so the tones are easier to recognise.",
                    lengthLonger: "It sounds like you said more than the sentence shown. Please read exactly the text on screen.",
                    lengthShorter: "It sounds like some words of the sentence were left out. Please read the whole text on screen.",
                    lengthDifferent: "What you said doesn\u2019t seem to match the sentence shown. Please read exactly the text on screen.",
                    profilePrompt: "Enter your name and participant group before speaking practice.",
                    profileHint: "Use a name that distinguishes you from other people; the same name always continues the same practice history.",
                    name: "Name", participantGroup: "Participant group", chooseOne: "Choose one...", taiwanese: "Taiwanese", foreigner: "Foreigner",
                    continue: "Continue", cancel: "Cancel", appTitle: "Taiwan Mandarin Pronunciation Practice",
            practiceInstructions: "Choose a lesson sample or enter your own Mandarin text, then record yourself reading it:",
            practiceSource: "Practice text", bookTextMode: "Book sample", customTextMode: "My own text",
            chapter5WordMode: "Topics", previousWord: "Previous", nextWord: "Next",
            chapter5Vocabulary: "Vocabulary", chapter5Phrase: "Phrase", chapter5Meaning: "{meaning}",
                    chapter5OptionalHint: "The lighter part is optional: “{short}” and “{full}” are both accepted.",
            chapter5ReferenceAudio: "Reference only – not used for scoring.",
            chapter5Progress: "{current} / {total}", chapter5Unavailable: "Topic practice items are unavailable.",
            chapter5Ready: "Ready. Record yourself reading it.",
                    chapter5Topic: "Topic", chapter5AllTopics: "All topics", chapter5AllInTopic: "All in {topic}",
            customTextPreview: "Entered text preview",
            customScriptLabel: "Text to read:",
            customTextHint: "Enter any Mandarin text. It does not need to appear in a textbook.",
            customSampleListen: "Listen to a sample",
            customSampleBusy: "Making the sample\u2026",
            customSampleNote: "A computer voice reading your text a little slowly. It is only a guide; a word may sound slightly different from a teacher\u2019s reading.",
            customSampleError: "The sample could not be made. Try again.",
            customSampleUnavailable: "The sample voice is not available on this server.",
            leaderboardTitle: "Top 10",
            leaderboardHelp: "Points: the best attempt of each sentence (a Good syllable counts 1, an Okay one \u00bd), added up. Your own text does not count.",
            leaderboardEmpty: "No scores yet. Be the first!",
            leaderboardPoints: "{n} pts",
            leaderboardLessons: "{n} sentences",
            leaderboardMeRank: "You: #{rank} \u00b7 {points} pts",
            leaderboardMeNone: "Practice a sentence to join the board.",
            boardTabPoints: "Points",
            boardTabAccuracy: "Accuracy",
            leaderboardHelpAccuracy: "Accuracy: over all your attempts at sentences and words, the share of syllables that were Good (an Okay one counts \u00bd), out of 100. You need at least {n} sentences. Your own text does not count.",
            leaderboardEmptyAccuracy: "Nobody has done {n} sentences yet.",
            leaderboardAccuracy: "{n}%",
            leaderboardMeAccuracyRank: "You: #{rank} \u00b7 {accuracy}%",
            leaderboardMeAccuracyNeed: "Practice {n} more sentence(s) to join the accuracy board.",
            resultRetry: "Read again",
            resultNext: "Next",
            itemProgress: "Practised {n}× · best {best}% · latest {last}%",
            itemNotPracticed: "Not practised yet.",
            recentTitle: "My recent results",
            recentEmpty: "Your results will appear here.",
            recentPoints: "{n}%",
            customTextPlaceholder: "Type or paste Mandarin text here…", customTextReady: "Your text is ready. Record yourself reading it.",
            customTextRequiresHanzi: "Enter Chinese characters to assess Mandarin tones.",
                    sampleRecording: "Sample recording (Lessons 5–8):", loadingSamples: "Loading sample recordings...", matchingScript: "Matching script:",
                    scriptWithPinyin: "Matching script with pinyin", showPinyin: "Show pinyin", startRecording: "Start recording",
                    stopAndAssess: "Stop & assess", recordingStop: "Recording (click to stop)...", recording: "Recording... Speak clearly into the microphone!",
                    latestRecording: "Your latest microphone recording:", uploadAudio: "Upload your audio:",
                    uploadHint: "Choose a recording of yourself reading the selected script. Supported formats: WAV, MP3, M4A, OGG, FLAC, WebM.",
                    selectedAudio: "Your selected audio", uploadAssess: "Upload & assess", resultsTitle: "Reading practice results",
                    viewPitch: "View your pitch graph (optional)", pitchDisplay: "Pitch display:", relativePitch: "Relative pitch (semitones)", f0Hz: "F0 (Hz)",
                    recordingLine: "Your pitch contour",
                    pitchNote: "Tone feedback uses your pitch contour only. Sample audio is not scored.", quickFeedback: "Quick feedback",
                    feedbackQuestion: "How helpful was this practice? 1 = not helpful, 5 = very helpful.",
                    feedbackImprove: "Anything we should improve? (optional)", feedbackPlaceholder: "Share a short comment", sendFeedback: "Send feedback",
                    sendingFeedback: "Sending feedback...", feedbackSaved: "Thanks for your feedback!", feedbackSaveFailed: "Could not save feedback.",
                    practicingAs: "Practicing as {name} ({group})", profileReady: "Profile ready for {name}. Choose a sample to begin.",
                    voiceCalibrationTitle: "Set up your voice", voiceCalibrationPrompt: "A 10-second sample lets us judge your tones against your own voice, so the feedback is fair.",
                    calPrivacy: "Your recording is not kept. Only a few pitch numbers are saved.",
                    voiceCalibrationLater: "Skip for now", calibrationDone: "Start practicing", recordCalibration: "Start recording", stopCalibration: "I'm done",
                    calibrationRecording: "Recording… read the five sounds twice.", calibrationProcessing: "Checking your voice…", calibrationSaved: "All set! Your tones will now be judged against your own voice.",
                    calibrationTooShort: "That was too short. Keep reading until the button lets you stop.", calibrationFailed: "We couldn't use that recording. Read all five sounds twice, clearly, then try again.",
                    calStep1: "Read these five sounds aloud, twice",
                    calStep1Hint: "Listen first (\u25b6). Then say each sound slowly and stretch it a little (m\u0101\u2011\u0101\u2011\u0101), with a short pause between sounds. Use your own comfortable, normal voice: no need to match the sample\u2019s pitch or to push or strain. A relaxed voice gives the most natural tones. Tone 4 (m\u00e0) stays short and sharp.",
                    calStep2: "Press the button and start reading",
                    calToneHigh: "high",
                    calToneRising: "rising",
                    calToneDip: "dip",
                    calToneFalling: "falling",
                    calToneLight: "light",
                    calKeepReading: "Keep reading…",
                    calTryAgain: "Try again",
                    calSeconds: "{seconds} s",
                    calClose: "Close",
                    micNoSound: "We can't hear you. Speak a little louder or check your microphone.",
                    micLevel: "Microphone level",
                    calReasonClipped: "The sound was too loud and distorted. Move a little back from the microphone and try again.",
                    calReasonNoisy: "There was too much background noise. Try again in a quieter place.",
                    calReasonUnclear: "We couldn't follow your pitch clearly. Read all five sounds twice, calmly, and try again.",
                    calReasonQuiet: "We found too little steady voice, and speaking louder doesn\u2019t help. Use your normal voice (not a whisper) and say each sound a bit longer, about half a second (m\u0101\u2011\u0101\u2011\u0101), then try again.",
                    calibrateVoice: "Calibrate voice", recalibrateVoice: "Recalibrate voice", voiceCalibrated: "Voice range personalized", voiceNotCalibrated: "Using a general pitch range",
                    profileSaveFailed: "Could not save speaker profile.", needProfile: "Enter your name and participant group first.",
                    chooseAudio: "Choose an audio file first.", unsupportedAudio: "Unsupported file format. Choose WAV, MP3, M4A, OGG, FLAC, or WebM.",
                    emptyAudioFile: "This audio file is empty. Choose another recording.", uploadReady: "{file} is ready. Click Upload & assess to check it against the selected script.",
                    chooseSample: "Choose a reference sample first.", noAudio: "No audio was captured. Record again or choose another file.",
                    processing: "Processing audio and checking pronunciation...", assessmentSaved: "Assessment complete and saved!",
                    assessmentFailed: "Assessment failed: {error}", noSampleAudio: "No matching audio samples were found.",
                    sampleDataUnavailable: "Sample audio or script data is unavailable on the server.", sampleLoadFailed: "Could not load sample recordings from the server.",
                    sampleReady: "Script ready. Record yourself or upload an audio file reading it.",
                    sampleAudioUnavailable: "Sample audio is unavailable; you can still practice from the script.",
                    sampleAudioLabel: "Listen to sample audio:", sampleAudioLoadError: "Sample audio could not be loaded. Try selecting the item again.",
                    microphoneHttps: "Microphone access requires HTTPS or localhost. You can upload an audio file instead.",
                    recordingUnsupported: "Audio recording is not supported in this browser. You can upload an audio file instead.",
                    microphoneFailed: "Could not access the microphone: {error}", serverNoResponse: "The server returned HTTP {status} without an assessment response.",
                    lessonOption: "Lesson {lesson} · Story {story} · Scene {scene} · {character}",
                    likelyCorrect: "Good", practice: "Needs practice", uncertain: "Okay", insufficientEvidence: "Okay",
                    mismatch: "Not scored: script mismatch", transcriptUnchecked: "Not scored: transcript unchecked",
                    unassessable: "Not enough audio to score", notScored: "Not scored", noMatch: "Speech recognition did not match the script; phrase tone feedback was skipped.",
                    partialMatch: "Speech recognition only partly matched the script; unmatched phrases are not tone-scored.",
                    unchecked: "Speech recognition was not checked, so phrase tone feedback is unavailable.",
                    matched: "Speech recognition matched the script; tone feedback is shown by phrase.", renderPhraseScores: "Each card checks the expected tone sequence for the whole phrase.",
                    adviceTone1: "“{char}” – tone 1: keep your voice high and steady, with no rise or fall.",
                    adviceTone2: "“{char}” – tone 2: start at a middle pitch and rise clearly, like asking “huh?”.",
                    adviceTone3: "“{char}” – tone 3: let your voice sink low, like a soft “hmm…”, instead of staying high.",
                    adviceTone4: "“{char}” – tone 4: start high and drop quickly, like a firm “No!”.",
                    retrySlower: "Say this word a little slower and stretch it a little \u2013 don\u2019t rush it, so the tones are easier to hear.",
                    retryShortFall: "Tone 4 is short and sharp: don\u2019t draw it out. Start high and drop firmly in one move.",
                    retrySteady: "Keep your voice steady and a little louder. Avoid whispering or a creaky voice.",
                    retryClearer: "Make the tone a bit more distinct so the rise and fall can be heard clearly.",
                    retryReadExactly: "Read this part exactly as written, from start to finish.",
                    retryAgain: "Try saying it once more, clearly and at a natural pace.",
                    qualityClipped: "The recording is distorted. Move back from the microphone and try again.",
                    qualityTooQuiet: "The recording is too quiet. Speak louder or move closer to the microphone.",
                    qualityNoisy: "There is too much background noise. Try again somewhere quieter.",
                    qualityTooShort: "The recording is too short. Try again.",
                    qualityTooLong: "The recording is too long. Record only the practice phrase.",
                    qualityNoSpeech: "No speech was detected. Try again.",
                    qualityNoVoicing: "No clear voice was detected. Speak aloud rather than whispering.",
                    alignmentFailed: "We couldn't analyse this recording because of a temporary problem on our side, not because of your voice. Please try again.",
                    busyWait: "This usually takes a few seconds.",
                    notHeard: "Not heard", retryNotHeard: "We couldn’t hear this part. Speak a little louder, read all of it, and try again.",
                    retryFullThird: "Say this third tone in full: let your voice dip low, then rise a little at the end.",
                    expectedSequence: "Expected", detectedSequence: "Detected", sequenceUnavailable: "Not enough tone evidence",
                    noProbabilityModel: "Tone labels use experimental rules; this system has no trained or calibrated tone probabilities.",
                    pronunciationFeedback: "Tone feedback", phraseScore: "Phrase score {score}", phraseScoreUnavailable: "Phrase score unavailable", viewJsonOutput: "View JSON response", scriptProvidedOnly: "Scored against the script shown above, so please read it exactly as written.",
                    syllableDetails: "Syllable details", syllableColumn: "Syllable", toneColumn: "Tone", resultColumn: "Result", noteColumn: "Note",
                    toneShapeNote: "The small line sketches the target tone; the chart shows your measured pitch.",
                    toneShapeAria: "Schematic contour for {tone}",
                    toneT1: "T1", toneT2: "T2", toneT3: "T3", toneT4: "T4", neutralTone: "Neutral",
                    thirdToneSandhi: "third-tone change", buSandhi: "不 tone change", yiSandhi: "一 tone change",
                    taiwanReading: "Taiwan reading", phraseFinalTone: "phrase-final tone", lexicalToneRule: "lexical tone",
                    matchedTone: "Tone contour matches the target.", differentTone: "A different tone contour was detected.",
                    ambiguousTone: "The contour could match more than one tone.", neutralToneUncertain: "Neutral tone depends on context.",
                    boundaryDisagreement: "The middle and full syllable contours disagree.",
                    alignmentLow: "Syllable timing is unclear.", tooFewFrames: "Too little voiced pitch data.", sparseF0: "Pitch data has gaps.",
                    octaveJump: "Pitch tracking has a sudden jump.", incompleteContour: "Pitch data does not cover the whole syllable.",
                    pinyinMismatch: "The recognized syllable differs from the script.", transcriptNotChecked: "Speech recognition was not checked.",
                    phraseNeedsPractice: "One or more syllables need practice.", phraseUncertain: "One or more syllables are uncertain.",
                    otherSyllables: "Other syllables", phrase: "Phrase", sentence: "Sentence {number}", scriptLabel: "Script: {script}", noScores: "No pronunciation scores are available.",
                    noPitch: "No measurable pitch in this phrase.", f0Title: "F0 in Hz", relativeTitle: "Relative pitch", relativeSemitone: "Semitones from your pitch median",
                    midiNote: "MIDI note",
                    midiTitle: "Pitch as MIDI note",
                    midiLabel: "MIDI note (A4 = 440 Hz = 69)", halfThirdTone: "Half-T3", clause: "sentence {number}",
                    pitchDescription: "Your pitch contour is shown for the whole phrase. Gaps in measured pitch remain blank.",
                    reference: "Reference", yourRecording: "Your recording", lowConfidence: " (alignment or pitch has low confidence)",
                    relative: "Relative", hz: "Hz", invalidAudio: "Invalid audio file format.", emptyScript: "The input script must not be empty.",
                    invalidProfile: "Enter a valid speaker profile before practicing.", sampleMissing: "The selected prototype sample was not found.",
                    noHanScript: "The script contains no valid Han characters to score pronunciation.", emptyName: "Enter a name.",
                    invalidNameLength: "Name must contain 1 to 100 characters.", invalidGroup: "Choose Taiwanese or Foreigner.",
                    attemptMissing: "The selected practice attempt was not found.", invalidFeedbackRating: "Rating must be an integer from 1 to 5.",
                    feedbackTooLong: "Feedback comment must be 2,000 characters or fewer.",
                    scriptMismatch: "The selected sample and script do not match. Only punctuation and spacing differences are allowed. Expected: {expected} Entered: {received}"
                },
                "zh-TW": {
                    language: "語言", languageEn: "English", languageZh: "中文（繁體）", languageVi: "Tiếng Việt",
                    speakerProfile: "使用者資料", changeSpeaker: "更換使用者",
                    calListen: "聽這個音", calListenAll: "\u25b6 聽這五個音",
                    calibrationSavedSlower: "聲音已儲存。小提醒：下次可以唸慢一點、把每個音拉長一點，這樣更容易辨認。",
                    paceTooFast: "有些音太快，無法分析。請唸慢一點，或把每個音稍微拉長，聲調會更容易辨認。",
                    lengthLonger: "聽起來你唸的比畫面上的句子還多。請照畫面上的文字唸。",
                    lengthShorter: "聽起來句子裡有些字沒有唸到。請把畫面上的整句唸完。",
                    lengthDifferent: "你唸的內容似乎和畫面上的句子不一樣。請照畫面上的文字唸。",
                    profilePrompt: "請先輸入姓名和參與者類別，再開始口說練習。",
                    profileHint: "請使用能和其他人區分的姓名；輸入相同的姓名就會沿用同一份練習紀錄。",
                    name: "姓名", participantGroup: "參與者類別", chooseOne: "請選擇…", taiwanese: "台灣人", foreigner: "外國人",
                    continue: "繼續", cancel: "取消", appTitle: "台灣華語發音練習",
                    practiceInstructions: "選擇課本範例，或輸入自己的華語文本，再依照內容錄音：",
                    practiceSource: "練習文本", bookTextMode: "課本範例", customTextMode: "自選文本",
                    chapter5WordMode: "主題", previousWord: "上一個", nextWord: "下一個",
                    chapter5Vocabulary: "詞彙", chapter5Phrase: "詞組", chapter5Meaning: "{meaning}",
                    chapter5OptionalHint: "淺色的部分可說可不說：「{short}」或「{full}」都可以。",
                    chapter5ReferenceAudio: "僅供參考，不會用於評分。",
                    chapter5Progress: "{current} / {total}", chapter5Unavailable: "主題練習資料目前無法使用。",
                    chapter5Ready: "項目已準備好，請錄音朗讀。",
                    chapter5Topic: "主題", chapter5AllTopics: "所有主題", chapter5AllInTopic: "全部：{topic}",
                    customTextPreview: "輸入文本預覽",
                    customScriptLabel: "朗讀文本：",
                    customTextHint: "輸入任何華語文本，不必對應課本內容。",
                    customSampleListen: "聽範例",
                    customSampleBusy: "正在製作範例……",
                    customSampleNote: "電腦語音會稍微放慢朗讀你的文本，僅供參考，個別詞的唸法可能與老師略有不同。",
                    customSampleError: "無法製作範例，請再試一次。",
                    customSampleUnavailable: "這台伺服器沒有可用的範例語音。",
                    leaderboardTitle: "前十名",
                    leaderboardHelp: "積分：每一句取最好的一次（Good 音節算 1，Okay 算 \u00bd），全部相加。自己輸入的文本不計分。",
                    leaderboardEmpty: "還沒有成績，來當第一名吧！",
                    leaderboardPoints: "{n} 分",
                    leaderboardLessons: "{n} 句",
                    leaderboardMeRank: "你：第 {rank} 名\u30fb{points} 分",
                    leaderboardMeNone: "練習一句就能上榜。",
                    boardTabPoints: "積分",
                    boardTabAccuracy: "準確率",
                    leaderboardHelpAccuracy: "準確率：你所有句子與單字練習中，Good 音節的比例（Okay 算 \u00bd），滿分 100。至少要練 {n} 句。自己輸入的文本不計分。",
                    leaderboardEmptyAccuracy: "還沒有人練滿 {n} 句。",
                    leaderboardAccuracy: "{n}%",
                    leaderboardMeAccuracyRank: "你：第 {rank} 名\u30fb{accuracy}%",
                    leaderboardMeAccuracyNeed: "再練 {n} 句就能上準確率榜。",
                    resultRetry: "再讀一次",
                    resultNext: "下一個",
                    itemProgress: "已練習 {n} 次・最佳 {best}%・最近 {last}%",
                    itemNotPracticed: "還沒練習過。",
                    recentTitle: "我的最近成績",
                    recentEmpty: "成績會顯示在這裡。",
                    recentPoints: "{n}%",
                    customTextPlaceholder: "在此輸入或貼上華語文本…", customTextReady: "文本已準備好，請依照內容錄音。",
                    customTextRequiresHanzi: "請輸入中文字才能評估華語聲調。",
                    sampleRecording: "範例錄音（第 5–8 課）：", loadingSamples: "正在載入範例錄音…", matchingScript: "練習文本：",
                    scriptWithPinyin: "含拼音的練習文本", showPinyin: "顯示拼音", startRecording: "開始錄音",
                    stopAndAssess: "停止並評估", recordingStop: "錄音中（按一下停止）…", recording: "錄音中…請對著麥克風清楚說話！",
                    latestRecording: "最近一次麥克風錄音：", uploadAudio: "上傳音檔：",
                    uploadHint: "請選擇朗讀目前文本的錄音。支援 WAV、MP3、M4A、OGG、FLAC、WebM。",
                    selectedAudio: "已選擇的音檔", uploadAssess: "上傳並評估", resultsTitle: "朗讀練習結果",
                    viewPitch: "查看你的音高圖（選填）", pitchDisplay: "音高顯示：", relativePitch: "相對音高（半音）", f0Hz: "基頻（Hz）",
                    recordingLine: "你的音高曲線",
                    pitchNote: "評估只使用你的音高曲線；範例音訊不會用來評分。", quickFeedback: "快速回饋",
                    feedbackQuestion: "這次練習對你有幫助嗎？1 = 沒幫助，5 = 非常有幫助。",
                    feedbackImprove: "你覺得哪些地方可以改進？（選填）", feedbackPlaceholder: "請簡短留言", sendFeedback: "送出回饋",
                    sendingFeedback: "正在送出回饋…", feedbackSaved: "謝謝你的回饋！", feedbackSaveFailed: "無法儲存回饋。",
                    practicingAs: "目前使用者：{name}（{group}）", profileReady: "使用者資料已準備好，{name}，請選擇範例開始練習。",
                    voiceCalibrationTitle: "設定你的聲音", voiceCalibrationPrompt: "錄一段約 10 秒的聲音，系統就能依你自己的音高來判斷聲調，回饋會更公平。",
                    calPrivacy: "錄音不會被保存，只會儲存少量音高數據。",
                    voiceCalibrationLater: "先跳過", calibrationDone: "開始練習", recordCalibration: "開始錄音", stopCalibration: "我讀完了",
                    calibrationRecording: "錄音中……請把五個音讀兩遍。", calibrationProcessing: "正在檢查你的聲音……", calibrationSaved: "完成！之後的聲調會依你自己的聲音來判斷。",
                    calibrationTooShort: "錄音太短了。請繼續朗讀，等按鈕可以按時再停止。", calibrationFailed: "這段錄音無法使用。請清楚地把五個音讀兩遍，再試一次。",
                    calStep1: "請把這五個音大聲讀兩遍",
                    calStep1Hint: "先聽一聽（\u25b6）。再把每個音唸慢一點、拉長一點（mā\u2011ā\u2011ā），音與音之間稍微停一下。用自己最舒服的平常聲音就好：不用模仿示範音的高低，也不必用力或硬撐，放鬆的聲音聲調最自然。第四聲（mà）要保持短而有力。",
                    calStep2: "按下按鈕，開始朗讀",
                    calToneHigh: "高平",
                    calToneRising: "上升",
                    calToneDip: "降升",
                    calToneFalling: "下降",
                    calToneLight: "輕聲",
                    calKeepReading: "請繼續朗讀……",
                    calTryAgain: "再試一次",
                    calSeconds: "{seconds} 秒",
                    calClose: "關閉",
                    micNoSound: "聽不到你的聲音。請稍微大聲一點，或檢查麥克風。",
                    micLevel: "麥克風音量",
                    calReasonClipped: "聲音太大而破音了。請離麥克風遠一點，再試一次。",
                    calReasonNoisy: "背景噪音太多。請到安靜的地方再試一次。",
                    calReasonUnclear: "無法清楚追蹤你的音高。請自然地把五個音讀兩遍，再試一次。",
                    calReasonQuiet: "偵測到的穩定聲音太少，大聲說並沒有幫助。請用平常的聲音（不要用氣音），把每個音唸長一點，約半秒（m\u0101\u2011\u0101\u2011\u0101），再試一次。",
                    calibrateVoice: "校準聲音", recalibrateVoice: "重新校準聲音", voiceCalibrated: "已個人化音高範圍", voiceNotCalibrated: "目前使用一般音高範圍",
                    profileSaveFailed: "無法儲存使用者資料。", needProfile: "請先輸入姓名和參與者類別。",
                    chooseAudio: "請先選擇音檔。", unsupportedAudio: "不支援此檔案格式。請選擇 WAV、MP3、M4A、OGG、FLAC 或 WebM。",
                    emptyAudioFile: "音檔是空的，請選擇其他錄音。", uploadReady: "{file} 已準備好。按一下「上傳並評估」即可依照所選文本檢查。",
                    chooseSample: "請先選擇範例語音。", noAudio: "沒有錄到音訊，請重新錄音或選擇其他檔案。",
                    processing: "正在處理音訊並檢查發音…", assessmentSaved: "評估完成並已儲存！",
                    assessmentFailed: "評估失敗：{error}", noSampleAudio: "找不到符合條件的範例錄音。",
                    sampleDataUnavailable: "伺服器上沒有可用的範例音訊或文本資料。", sampleLoadFailed: "無法從伺服器載入範例錄音。",
                    sampleReady: "文本已準備好。請依照文本錄音或上傳音檔。",
                    sampleAudioUnavailable: "範例音訊目前無法使用，你仍可依照文本練習。",
                    sampleAudioLabel: "播放範例音訊：", sampleAudioLoadError: "無法載入範例音訊，請重新選擇此項目。",
                    microphoneHttps: "麥克風需要 HTTPS 或 localhost 才能使用。你也可以改為上傳音檔。",
                    recordingUnsupported: "此瀏覽器不支援錄音。你可以改為上傳音檔。",
                    microphoneFailed: "無法使用麥克風：{error}", serverNoResponse: "伺服器回傳 HTTP {status}，沒有提供評估結果。",
                    lessonOption: "第 {lesson} 課 · 故事 {story} · 場景 {scene} · {character}",
                    likelyCorrect: "良好", practice: "需要練習", uncertain: "還可以", insufficientEvidence: "還可以",
                    mismatch: "未評分：文本不符", transcriptUnchecked: "未評分：未檢查辨識文本",
                    unassessable: "音訊不足，無法評分", notScored: "未評分", noMatch: "語音辨識未符合文本，已略過聲調回饋。",
                    partialMatch: "語音辨識只部分符合文本；未符合的詞組不提供聲調回饋。",
                    unchecked: "尚未檢查語音辨識，無法提供詞組聲調回饋。",
                    matched: "語音辨識符合文本，以下顯示整個詞組的聲調回饋。", renderPhraseScores: "每張卡片評估整個詞組的預期聲調順序。",
                    adviceTone1: "「{char}」第一聲：保持高而平穩的聲音，不要上升或下降。",
                    adviceTone2: "「{char}」第二聲：從中等高度開始，明顯往上揚，像在問「嗯？」。",
                    adviceTone3: "「{char}」第三聲：讓聲音沉到低處，像輕輕說「嗯……」，不要停在高音。",
                    adviceTone4: "「{char}」第四聲：從高處開始，乾脆地往下降，像堅定地說「不！」。",
                    retrySlower: "這個詞請唸慢一點、稍微拉長，不要趕著唸過去，聲調才聽得清楚。",
                    retryShortFall: "第四聲本來就短而有力，不要拉長。從高處開始，一口氣乾脆地降下來。",
                    retrySteady: "請讓聲音保持穩定，並稍微大聲一點；避免氣音或沙啞的聲音。",
                    retryClearer: "請把聲調唸得更明顯，讓音高的升降聽得更清楚。",
                    retryReadExactly: "請從頭到尾照著文字完整唸出來。",
                    retryAgain: "請再唸一次，吐字清楚，用自然的速度。",
                    qualityClipped: "錄音有破音。請離麥克風遠一點，再試一次。",
                    qualityTooQuiet: "錄音太小聲。請說大聲一點，或靠近麥克風。",
                    qualityNoisy: "背景噪音太多。請在安靜的地方再試一次。",
                    qualityTooShort: "錄音太短。請再試一次。",
                    qualityTooLong: "錄音太長。請只錄練習的句子。",
                    qualityNoSpeech: "沒有偵測到說話聲。請再試一次。",
                    qualityNoVoicing: "沒有偵測到清楚的嗓音。請出聲說話，不要用氣音。",
                    alignmentFailed: "這次沒能分析這段錄音，是系統暫時出了問題，不是你的聲音有問題。請再試一次。",
                    busyWait: "通常只需要幾秒鐘。",
                    notHeard: "沒聽到", retryNotHeard: "這一段沒有聽到聲音。請稍微大聲一點、完整唸出來，再試一次。",
                    retryFullThird: "請把這個三聲讀完整：聲音先降到低處，結尾再微微上揚。",
                    expectedSequence: "預期", detectedSequence: "辨識", sequenceUnavailable: "聲調證據不足",
                    noProbabilityModel: "聲調結果使用實驗性規則；目前沒有經訓練或校準的聲調機率模型。",
                    pronunciationFeedback: "聲調回饋", phraseScore: "詞組分數 {score}", phraseScoreUnavailable: "無詞組分數", viewJsonOutput: "\u67e5\u770b JSON \u8f38\u51fa", scriptProvidedOnly: "依上方顯示的文字評分，請照著文字唸。",
                    syllableDetails: "音節明細", syllableColumn: "音節", toneColumn: "聲調", resultColumn: "結果", noteColumn: "說明",
                    toneShapeNote: "小圖示意目標聲調走向；下方圖表顯示你錄到的實際音高。",
                    toneShapeAria: "目標聲調示意：{tone}",
                    toneT1: "一聲", toneT2: "二聲", toneT3: "三聲", toneT4: "四聲", neutralTone: "輕聲",
                    thirdToneSandhi: "三聲變調", buSandhi: "「不」變調", yiSandhi: "「一」變調",
                    taiwanReading: "台灣讀音", phraseFinalTone: "詞組末聲調", lexicalToneRule: "詞彙聲調",
                    matchedTone: "聲調曲線符合目標。", differentTone: "偵測到不同的聲調曲線。",
                    ambiguousTone: "這條曲線可能屬於多種聲調。", neutralToneUncertain: "輕聲需依語境判斷。",
                    boundaryDisagreement: "音節中央與完整曲線的判斷不同。",
                    alignmentLow: "音節時間定位不清楚。", tooFewFrames: "有聲音高資料太少。", sparseF0: "音高資料有缺漏。",
                    octaveJump: "音高追蹤出現突然跳變。", incompleteContour: "音高資料未涵蓋完整音節。",
                    pinyinMismatch: "辨識出的音節與文本不同。", transcriptNotChecked: "尚未檢查語音辨識。",
                    phraseNeedsPractice: "有一個或多個音節需要練習。", phraseUncertain: "有一個或多個音節尚不確定。",
                    otherSyllables: "其他音節", phrase: "詞組", sentence: "句子 {number}", scriptLabel: "文本：{script}", noScores: "目前沒有可用的發音分數。",
                    noPitch: "這個詞組沒有可測量的音高。", f0Title: "基頻（Hz）", relativeTitle: "相對音高", relativeSemitone: "相對錄音中位數（半音）",
                    midiNote: "MIDI 音符",
                    midiTitle: "以 MIDI 音符表示的音高",
                    midiLabel: "MIDI 音符（A4 = 440 Hz = 69）", halfThirdTone: "半三聲", clause: "句子 {number}",
                    pitchDescription: "圖表顯示整個詞組中的音高曲線；未測得的 F0 區段會留白。",
                    reference: "範例語音", yourRecording: "你的錄音", lowConfidence: "（對齊或音高可信度較低）",
                    relative: "相對音高", hz: "Hz", invalidAudio: "音訊檔案格式無效。", emptyScript: "練習文本不能為空。",
                    invalidProfile: "請確認使用者資料後再開始練習。", sampleMissing: "找不到所選的範例語音。",
                    noHanScript: "文本中沒有可供發音評估的有效漢字。", emptyName: "請輸入姓名。", invalidNameLength: "姓名長度須為 1 到 100 個字元。",
                    invalidGroup: "請選擇台灣人或外國人。", attemptMissing: "找不到所選的練習紀錄。", invalidFeedbackRating: "評分必須是 1 到 5 的整數。",
                    feedbackTooLong: "回饋不得超過 2,000 個字元。",
                    scriptMismatch: "所選範例與練習文本不符。僅允許標點和空格不同。範例：{expected} 輸入：{received}"
                },
                vi: {
                    language: "Ngôn ngữ", languageEn: "English", languageZh: "中文（繁體）", languageVi: "Tiếng Việt",
                    speakerProfile: "Thông tin người nói", changeSpeaker: "Đổi người dùng",
                    calListen: "Nghe âm này", calListenAll: "\u25b6 Nghe cả năm âm",
                    calibrationSavedSlower: "Đã lưu giọng của bạn. Mẹo: lần sau hãy đọc chậm hơn một chút và kéo dài từng âm \u2013 sẽ dễ nhận diện hơn.",
                    paceTooFast: "Một số âm quá nhanh nên chưa phân tích được. Hãy đọc chậm hơn, hoặc kéo dài từng âm một chút để dễ nhận diện thanh điệu.",
                    lengthLonger: "Có vẻ bạn đã đọc dài hơn câu hiển thị. Hãy đọc đúng nội dung trên màn hình.",
                    lengthShorter: "Có vẻ bạn đã bỏ sót một số từ của câu. Hãy đọc đầy đủ nội dung trên màn hình.",
                    lengthDifferent: "Nội dung bạn đọc có vẻ không khớp với câu hiển thị. Hãy đọc đúng nội dung trên màn hình.",
                    profilePrompt: "Nhập tên và nhóm người tham gia trước khi luyện nói.",
                    profileHint: "Hãy dùng tên giúp phân biệt bạn với người khác; nhập cùng một tên thì luôn tiếp tục lịch sử luyện tập cũ.",
                    name: "Tên", participantGroup: "Nhóm người tham gia", chooseOne: "Chọn một mục...", taiwanese: "Người Đài Loan", foreigner: "Người nước ngoài",
                    continue: "Tiếp tục", cancel: "Hủy", appTitle: "Luyện phát âm tiếng Quan Thoại Đài Loan",
                    practiceInstructions: "Chọn bài mẫu hoặc nhập văn bản tiếng Trung của riêng bạn, rồi đọc và ghi âm:",
                    practiceSource: "Nội dung luyện tập", bookTextMode: "Bài trong sách", customTextMode: "Tự chọn văn bản",
                    chapter5WordMode: "Chủ đề", previousWord: "Trước", nextWord: "Tiếp theo",
                    chapter5Vocabulary: "Từ vựng", chapter5Phrase: "Cụm từ", chapter5Meaning: "{meaning}",
                    chapter5OptionalHint: "Phần màu nhạt là tùy chọn: đọc “{short}” hay “{full}” đều được.",
                    chapter5ReferenceAudio: "Chỉ để tham khảo, không dùng để chấm điểm.",
                    chapter5Progress: "{current} / {total}", chapter5Unavailable: "Không tải được dữ liệu luyện tập theo chủ đề.",
                    chapter5Ready: "Mục đã sẵn sàng. Hãy ghi âm bài đọc của bạn.",
                    chapter5Topic: "Chủ đề", chapter5AllTopics: "Tất cả chủ đề", chapter5AllInTopic: "Tất cả: {topic}",
                    customTextPreview: "Xem trước nội dung đã nhập",
                    customScriptLabel: "Nội dung cần đọc:",
                    customTextHint: "Nhập bất kỳ văn bản tiếng Trung nào; không cần có trong sách.",
                    customSampleListen: "Nghe thử mẫu",
                    customSampleBusy: "Đang tạo âm thanh mẫu…",
                    customSampleNote: "Giọng máy đọc chậm văn bản của bạn. Chỉ để tham khảo; một số từ có thể nghe hơi khác so với giáo viên đọc.",
                    customSampleError: "Không tạo được âm thanh mẫu. Hãy thử lại.",
                    customSampleUnavailable: "Máy chủ này chưa có giọng đọc mẫu.",
                    leaderboardTitle: "Top 10",
                    leaderboardHelp: "Điểm: lấy lượt tốt nhất của mỗi câu (âm Good tính 1, Okay tính \u00bd), cộng lại. Văn bản tự nhập không tính.",
                    leaderboardEmpty: "Chưa có điểm nào. Hãy là người đầu tiên!",
                    leaderboardPoints: "{n} điểm",
                    leaderboardLessons: "{n} câu",
                    leaderboardMeRank: "Bạn: hạng {rank} \u00b7 {points} điểm",
                    leaderboardMeNone: "Luyện một câu để có tên trên bảng.",
                    boardTabPoints: "Điểm",
                    boardTabAccuracy: "Độ chính xác",
                    leaderboardHelpAccuracy: "Độ chính xác: trên tất cả các lượt luyện câu và từ của bạn, tỉ lệ âm Good (Okay tính \u00bd), trên thang 100. Cần ít nhất {n} câu. Văn bản tự nhập không tính.",
                    leaderboardEmptyAccuracy: "Chưa ai luyện đủ {n} câu.",
                    leaderboardAccuracy: "{n}%",
                    leaderboardMeAccuracyRank: "Bạn: hạng {rank} \u00b7 {accuracy}%",
                    leaderboardMeAccuracyNeed: "Luyện thêm {n} câu để vào bảng độ chính xác.",
                    resultRetry: "Đọc lại",
                    resultNext: "Tiếp theo",
                    itemProgress: "Đã luyện {n} lần · tốt nhất {best}% · gần nhất {last}%",
                    itemNotPracticed: "Chưa luyện.",
                    recentTitle: "Kết quả gần đây của tôi",
                    recentEmpty: "Kết quả của bạn sẽ hiện ở đây.",
                    recentPoints: "{n}%",
                    customTextPlaceholder: "Nhập hoặc dán văn bản tiếng Trung tại đây…", customTextReady: "Đã sẵn sàng. Hãy đọc văn bản và ghi âm.",
                    customTextRequiresHanzi: "Hãy nhập chữ Hán để đánh giá thanh điệu tiếng Trung.",
                    sampleRecording: "Bản ghi mẫu (Bài 5–8):", loadingSamples: "Đang tải bản ghi mẫu...", matchingScript: "Nội dung luyện đọc:",
                    scriptWithPinyin: "Nội dung luyện đọc kèm pinyin", showPinyin: "Hiện pinyin", startRecording: "Bắt đầu ghi âm",
                    stopAndAssess: "Dừng và chấm", recordingStop: "Đang ghi âm (nhấn để dừng)...", recording: "Đang ghi âm... Hãy nói rõ vào micro!",
                    latestRecording: "Bản ghi micro gần nhất:", uploadAudio: "Tải âm thanh lên:",
                    uploadHint: "Chọn bản ghi bạn đọc nội dung đã chọn. Hỗ trợ WAV, MP3, M4A, OGG, FLAC và WebM.",
                    selectedAudio: "Âm thanh đã chọn", uploadAssess: "Tải lên và chấm", resultsTitle: "Kết quả luyện đọc",
                    viewPitch: "Xem đường cao độ của bạn (tùy chọn)", pitchDisplay: "Hiển thị cao độ:", relativePitch: "Cao độ tương đối (semitone)", f0Hz: "F0 (Hz)",
                    recordingLine: "Đường cao độ của bạn",
                    pitchNote: "Phản hồi thanh điệu chỉ dựa trên đường cao độ của bạn; âm thanh mẫu không dùng để chấm.", quickFeedback: "Phản hồi nhanh",
                    feedbackQuestion: "Buổi luyện tập này hữu ích thế nào? 1 = không hữu ích, 5 = rất hữu ích.",
                    feedbackImprove: "Bạn muốn chúng tôi cải thiện điều gì? (không bắt buộc)", feedbackPlaceholder: "Viết nhận xét ngắn", sendFeedback: "Gửi phản hồi",
                    sendingFeedback: "Đang gửi phản hồi...", feedbackSaved: "Cảm ơn phản hồi của bạn!", feedbackSaveFailed: "Không thể lưu phản hồi.",
                    practicingAs: "Đang luyện tập: {name} ({group})", profileReady: "Đã sẵn sàng, {name}. Hãy chọn mẫu để bắt đầu.",
                    voiceCalibrationTitle: "Thiết lập giọng nói của bạn", voiceCalibrationPrompt: "Một đoạn ghi khoảng 10 giây giúp hệ thống chấm thanh điệu theo chính giọng của bạn, nên nhận xét sẽ công bằng hơn.",
                    calPrivacy: "Bản ghi không được lưu lại. Hệ thống chỉ lưu vài số liệu về cao độ.",
                    voiceCalibrationLater: "Bỏ qua lúc này", calibrationDone: "Bắt đầu luyện tập", recordCalibration: "Bắt đầu ghi", stopCalibration: "Tôi đọc xong rồi",
                    calibrationRecording: "Đang ghi… hãy đọc năm âm hai lần.", calibrationProcessing: "Đang kiểm tra giọng của bạn…", calibrationSaved: "Xong rồi! Từ giờ thanh điệu sẽ được chấm theo chính giọng của bạn.",
                    calibrationTooShort: "Bản ghi quá ngắn. Hãy đọc tiếp cho đến khi bấm được nút dừng.", calibrationFailed: "Không dùng được bản ghi này. Hãy đọc rõ cả năm âm, hai lần, rồi thử lại.",
                    calStep1: "Đọc to năm âm này, hai lần",
                    calStep1Hint: "Hãy nghe trước (\u25b6). Sau đó đọc từng âm thật chậm và kéo dài một chút (m\u0101\u2011\u0101\u2011\u0101), nghỉ ngắn giữa các âm. Cứ dùng giọng bình thường, thoải mái của bạn: không cần bắt chước độ cao của âm mẫu, cũng không cần gồng hay ép giọng. Giọng thả lỏng cho thanh điệu tự nhiên nhất. Riêng thanh 4 (m\u00e0) vẫn đọc ngắn và dứt khoát.",
                    calStep2: "Bấm nút và bắt đầu đọc",
                    calToneHigh: "cao",
                    calToneRising: "lên",
                    calToneDip: "xuống-lên",
                    calToneFalling: "xuống",
                    calToneLight: "nhẹ",
                    calKeepReading: "Hãy đọc tiếp…",
                    calTryAgain: "Thử lại",
                    calSeconds: "{seconds} giây",
                    calClose: "Đóng",
                    micNoSound: "Chưa nghe thấy giọng bạn. Hãy nói to hơn một chút hoặc kiểm tra micro.",
                    micLevel: "Mức âm lượng micro",
                    calReasonClipped: "Âm thanh quá to và bị vỡ. Hãy lùi xa micro một chút rồi thử lại.",
                    calReasonNoisy: "Có quá nhiều tiếng ồn xung quanh. Hãy thử lại ở nơi yên tĩnh hơn.",
                    calReasonUnclear: "Chưa theo dõi rõ cao độ của bạn. Hãy đọc nhẹ nhàng cả năm âm hai lần rồi thử lại.",
                    calReasonQuiet: "Nghe được quá ít giọng ổn định, và nói to hơn không giúp được. Hãy dùng giọng bình thường (không thì thầm), kéo dài mỗi âm khoảng nửa giây (m\u0101\u2011\u0101\u2011\u0101) rồi thử lại.",
                    calibrateVoice: "Hiệu chuẩn giọng", recalibrateVoice: "Hiệu chuẩn lại giọng", voiceCalibrated: "Đã cá nhân hóa dải cao độ", voiceNotCalibrated: "Đang dùng dải cao độ mặc định",
                    profileSaveFailed: "Không thể lưu thông tin người nói.", needProfile: "Trước tiên hãy nhập tên và nhóm người tham gia.",
                    chooseAudio: "Trước tiên hãy chọn tệp âm thanh.", unsupportedAudio: "Định dạng tệp không được hỗ trợ. Chọn WAV, MP3, M4A, OGG, FLAC hoặc WebM.",
                    emptyAudioFile: "Tệp âm thanh trống. Hãy chọn bản ghi khác.", uploadReady: "Đã sẵn sàng tệp {file}. Nhấn “Tải lên và chấm” để đối chiếu với nội dung đã chọn.",
                    chooseSample: "Trước tiên hãy chọn giọng mẫu.", noAudio: "Không ghi được âm thanh. Hãy ghi lại hoặc chọn tệp khác.",
                    processing: "Đang xử lý âm thanh và kiểm tra phát âm...", assessmentSaved: "Đã chấm xong và lưu kết quả!",
                    assessmentFailed: "Chấm điểm thất bại: {error}", noSampleAudio: "Không tìm thấy bản ghi mẫu phù hợp.",
                    sampleDataUnavailable: "Máy chủ không có dữ liệu âm thanh mẫu hoặc nội dung.", sampleLoadFailed: "Không thể tải bản ghi mẫu từ máy chủ.",
                    sampleReady: "Nội dung đã sẵn sàng. Hãy đọc rồi ghi âm hoặc tải tệp lên.",
                    sampleAudioUnavailable: "Hiện không có âm thanh mẫu; bạn vẫn có thể luyện theo nội dung.",
                    sampleAudioLabel: "Nghe âm thanh mẫu:", sampleAudioLoadError: "Không tải được âm thanh mẫu. Hãy chọn lại mục này.",
                    microphoneHttps: "Cần HTTPS hoặc localhost để dùng micro. Bạn cũng có thể tải tệp âm thanh lên.",
                    recordingUnsupported: "Trình duyệt này không hỗ trợ ghi âm. Bạn có thể tải tệp âm thanh lên.",
                    microphoneFailed: "Không thể truy cập micro: {error}", serverNoResponse: "Máy chủ trả về HTTP {status} nhưng không có kết quả chấm.",
                    lessonOption: "Bài {lesson} · Truyện {story} · Cảnh {scene} · {character}",
                    likelyCorrect: "Tốt", practice: "Cần luyện", uncertain: "Tạm ổn", insufficientEvidence: "Tạm ổn",
                    mismatch: "Không chấm: nội dung không khớp", transcriptUnchecked: "Không chấm: chưa kiểm tra lời nói",
                    unassessable: "Không đủ âm thanh để chấm", notScored: "Không chấm", noMatch: "Nhận dạng giọng nói không khớp kịch bản; đã bỏ qua phản hồi thanh điệu.",
                    partialMatch: "Nhận dạng chỉ khớp một phần; cụm từ không khớp sẽ không được chấm thanh.",
                    unchecked: "Chưa kiểm tra nhận dạng, nên chưa có phản hồi thanh điệu theo cụm từ.",
                    matched: "Nhận dạng khớp nội dung; kết quả thanh điệu bên dưới là theo cụm từ.", renderPhraseScores: "Mỗi thẻ đánh giá cả chuỗi thanh dự kiến của một cụm từ.",
                    adviceTone1: "“{char}” – thanh 1: giữ giọng cao và đều, không lên cũng không xuống.",
                    adviceTone2: "“{char}” – thanh 2: bắt đầu ở độ cao vừa rồi lên rõ rệt, như khi hỏi “hả?”.",
                    adviceTone3: "“{char}” – thanh 3: hạ giọng xuống thấp, như khi khẽ nói “ừm…”, đừng giữ ở giọng cao.",
                    adviceTone4: "“{char}” – thanh 4: bắt đầu ở giọng cao rồi hạ nhanh, dứt khoát, như khi nói “Không!”.",
                    retrySlower: "Hãy đọc từ này chậm hơn và kéo dài âm một chút, đừng đọc lướt, để dễ nghe rõ thanh điệu.",
                    retryShortFall: "Thanh 4 vốn ngắn và dứt khoát: đừng kéo dài. Bắt đầu cao rồi hạ xuống dứt khoát trong một nhịp.",
                    retrySteady: "Hãy giữ giọng đều và to hơn một chút. Tránh thì thầm hoặc giọng khàn.",
                    retryClearer: "Hãy nhấn thanh điệu rõ hơn một chút để nghe rõ chỗ lên xuống.",
                    retryReadExactly: "Hãy đọc đúng và đủ như chữ đã cho, từ đầu đến cuối.",
                    retryAgain: "Hãy thử đọc lại một lần nữa, rõ ràng và với tốc độ tự nhiên.",
                    qualityClipped: "Bản ghi bị méo tiếng. Hãy lùi xa micro một chút rồi thử lại.",
                    qualityTooQuiet: "Bản ghi quá nhỏ. Hãy nói to hơn hoặc lại gần micro.",
                    qualityNoisy: "Có quá nhiều tiếng ồn nền. Hãy thử lại ở nơi yên tĩnh hơn.",
                    qualityTooShort: "Bản ghi quá ngắn. Hãy thử lại.",
                    qualityTooLong: "Bản ghi quá dài. Hãy chỉ ghi câu cần luyện.",
                    qualityNoSpeech: "Không phát hiện giọng nói. Hãy thử lại.",
                    qualityNoVoicing: "Không phát hiện giọng rõ. Hãy nói thành tiếng, đừng thì thầm.",
                    alignmentFailed: "Lần này hệ thống chưa phân tích được bản ghi do sự cố tạm thời phía chúng tôi, không phải do giọng của bạn. Hãy thử lại.",
                    busyWait: "Thường chỉ mất vài giây.",
                    notHeard: "Chưa nghe rõ", retryNotHeard: "Chưa nghe rõ đoạn này. Hãy nói to hơn một chút, đọc đủ cả đoạn rồi thử lại.",
                    retryFullThird: "Hãy đọc trọn thanh 3: hạ giọng xuống thấp rồi nâng nhẹ lên ở cuối.",
                    expectedSequence: "Kỳ vọng", detectedSequence: "Nhận diện", sequenceUnavailable: "Chưa đủ bằng chứng thanh điệu",
                    noProbabilityModel: "Kết quả dùng quy tắc thử nghiệm; hệ thống chưa có mô hình xác suất thanh điệu đã huấn luyện và hiệu chuẩn.",
                    pronunciationFeedback: "Phản hồi thanh điệu", phraseScore: "Điểm cụm từ {score}", phraseScoreUnavailable: "Chưa có điểm cụm từ", viewJsonOutput: "Xem JSON ph\u1ea3n h\u1ed3i", scriptProvidedOnly: "Chấm theo đoạn văn bản hiển thị ở trên, nên hãy đọc đúng như chữ đã cho.",
                    syllableDetails: "Chi tiết âm tiết", syllableColumn: "Âm tiết", toneColumn: "Thanh", resultColumn: "Kết quả", noteColumn: "Ghi chú",
                    toneShapeNote: "Đường nhỏ minh họa thanh mục tiêu; biểu đồ bên dưới là cao độ đo từ bản ghi của bạn.",
                    toneShapeAria: "Contour minh họa cho {tone}",
                    toneT1: "Thanh 1", toneT2: "Thanh 2", toneT3: "Thanh 3", toneT4: "Thanh 4", neutralTone: "Thanh nhẹ",
                    thirdToneSandhi: "biến điệu thanh 3", buSandhi: "biến điệu 不", yiSandhi: "biến điệu 一",
                    taiwanReading: "Cách đọc Đài Loan", phraseFinalTone: "thanh cuối cụm", lexicalToneRule: "thanh từ vựng",
                    matchedTone: "Đường thanh điệu khớp mục tiêu.", differentTone: "Đường thanh điệu khác đã được nhận diện.",
                    ambiguousTone: "Đường này có thể thuộc nhiều thanh điệu.", neutralToneUncertain: "Thanh nhẹ cần xét theo ngữ cảnh.",
                    boundaryDisagreement: "Đường ở giữa và toàn âm tiết cho kết quả khác nhau.",
                    alignmentLow: "Chưa căn thời gian âm tiết rõ.", tooFewFrames: "Có quá ít dữ liệu cao độ hữu thanh.", sparseF0: "Dữ liệu cao độ bị đứt quãng.",
                    octaveJump: "Theo dõi cao độ có bước nhảy đột ngột.", incompleteContour: "Dữ liệu cao độ chưa phủ hết âm tiết.",
                    pinyinMismatch: "Âm tiết nhận dạng khác nội dung mẫu.", transcriptNotChecked: "Chưa kiểm tra nhận dạng giọng nói.",
                    phraseNeedsPractice: "Có âm tiết cần luyện thêm.", phraseUncertain: "Có âm tiết chưa chắc chắn.",
                    otherSyllables: "Âm tiết khác", phrase: "Cụm", sentence: "Câu {number}", scriptLabel: "Nội dung: {script}", noScores: "Chưa có điểm phát âm.",
                    noPitch: "Không đo được cao độ trong cụm từ này.", f0Title: "F0 (Hz)", relativeTitle: "Cao độ tương đối", relativeSemitone: "Semitone so với trung vị bản ghi",
                    midiNote: "Nốt MIDI",
                    midiTitle: "Cao độ theo nốt MIDI",
                    midiLabel: "Nốt MIDI (A4 = 440 Hz = 69)", halfThirdTone: "Thanh 3 nửa", clause: "câu {number}",
                    pitchDescription: "Biểu đồ hiển thị contour cao độ của cả cụm từ; các khoảng thiếu F0 sẽ để trống.",
                    reference: "Giọng mẫu", yourRecording: "Bản ghi của bạn", lowConfidence: " (độ tin cậy căn chỉnh hoặc cao độ thấp)",
                    relative: "Tương đối", hz: "Hz", invalidAudio: "Định dạng tệp âm thanh không hợp lệ.", emptyScript: "Nội dung không được để trống.",
                    invalidProfile: "Hãy nhập thông tin người nói hợp lệ trước khi luyện tập.", sampleMissing: "Không tìm thấy giọng mẫu đã chọn.",
                    noHanScript: "Nội dung không có chữ Hán hợp lệ để chấm phát âm.", emptyName: "Hãy nhập tên.", invalidNameLength: "Tên phải có từ 1 đến 100 ký tự.",
                    invalidGroup: "Hãy chọn người Đài Loan hoặc người nước ngoài.", attemptMissing: "Không tìm thấy lượt luyện tập đã chọn.", invalidFeedbackRating: "Điểm đánh giá phải là số nguyên từ 1 đến 5.",
                    feedbackTooLong: "Nhận xét không được dài quá 2.000 ký tự.",
                    scriptMismatch: "Mẫu và nội dung đã chọn không khớp. Chỉ được khác dấu câu và khoảng trắng. Mẫu: {expected} Nội dung nhập: {received}"
                },
                id: {
                    language: "Bahasa",
                    languageEn: "English",
                    languageZh: "中文（繁體）",
                    languageVi: "Tiếng Việt",
                    speakerProfile: "Profil pembicara",
                    changeSpeaker: "Ganti pembicara",
                    calListen: "Dengarkan bunyi ini", calListenAll: "\u25b6 Dengarkan kelima bunyi",
                    calibrationSavedSlower: "Suara Anda tersimpan. Tips: lain kali bacalah sedikit lebih pelan dan panjangkan setiap bunyi \u2013 lebih mudah dikenali.",
                    paceTooFast: "Beberapa bunyi terlalu cepat untuk dianalisis. Bacalah sedikit lebih pelan, atau panjangkan setiap bunyi agar nadanya lebih mudah dikenali.",
                    lengthLonger: "Sepertinya Anda mengucapkan lebih banyak dari kalimat yang ditampilkan. Bacalah persis teks di layar.",
                    lengthShorter: "Sepertinya ada kata dalam kalimat yang terlewat. Bacalah seluruh teks di layar.",
                    lengthDifferent: "Yang Anda ucapkan tampaknya tidak sesuai dengan kalimat yang ditampilkan. Bacalah persis teks di layar.",
                    profilePrompt: "Masukkan nama dan kelompok peserta sebelum latihan berbicara.",
                    profileHint: "Gunakan nama yang membedakan Anda dari orang lain; nama yang sama akan selalu melanjutkan riwayat latihan yang sama.",
                    name: "Nama",
                    participantGroup: "Kelompok peserta",
                    chooseOne: "Pilih salah satu...",
                    taiwanese: "Orang Taiwan",
                    foreigner: "Orang asing",
                    continue: "Lanjutkan",
                    cancel: "Batal",
                    appTitle: "Latihan Pelafalan Mandarin Taiwan",
                    practiceInstructions: "Pilih contoh pelajaran atau ketik teks Mandarin Anda sendiri, lalu rekam diri Anda membacanya:",
                    practiceSource: "Teks latihan",
                    bookTextMode: "Contoh buku",
                    customTextMode: "Teks saya sendiri",
                    chapter5WordMode: "Topik",
                    previousWord: "Sebelumnya",
                    nextWord: "Berikutnya",
                    chapter5Vocabulary: "Kosakata",
                    chapter5Phrase: "Frasa",
                    chapter5Meaning: "{meaning}",
                    chapter5OptionalHint: "Bagian yang lebih pudar bersifat opsional: “{short}” maupun “{full}” sama-sama diterima.",
                    chapter5ReferenceAudio: "Hanya sebagai acuan – tidak dipakai untuk penilaian.",
                    chapter5Progress: "{current} / {total}",
                    chapter5Unavailable: "Butir latihan topik tidak tersedia.",
                    chapter5Ready: "Siap. Rekam diri Anda membacanya.",
                    chapter5Topic: "Topik",
                    chapter5AllTopics: "Semua topik",
                    chapter5AllInTopic: "Semua di {topic}",
                    customTextPreview: "Pratinjau teks yang dimasukkan",
                    customScriptLabel: "Teks yang dibaca:",
                    customTextHint: "Masukkan teks Mandarin apa saja. Tidak harus ada di buku pelajaran.",
                    customSampleListen: "Dengarkan contoh",
                    customSampleBusy: "Membuat contoh…",
                    customSampleNote: "Suara komputer membacakan teks Anda dengan agak pelan. Hanya sebagai panduan; beberapa kata bisa terdengar sedikit berbeda dari bacaan guru.",
                    customSampleError: "Contoh tidak dapat dibuat. Coba lagi.",
                    customSampleUnavailable: "Suara contoh tidak tersedia di server ini.",
                    leaderboardTitle: "10 Teratas",
                    leaderboardHelp: "Poin: percobaan terbaik tiap kalimat (suku kata Good bernilai 1, Okay \u00bd), dijumlahkan. Teks sendiri tidak dihitung.",
                    leaderboardEmpty: "Belum ada skor. Jadilah yang pertama!",
                    leaderboardPoints: "{n} poin",
                    leaderboardLessons: "{n} kalimat",
                    leaderboardMeRank: "Anda: peringkat {rank} \u00b7 {points} poin",
                    leaderboardMeNone: "Latih satu kalimat untuk masuk papan.",
                    boardTabPoints: "Poin",
                    boardTabAccuracy: "Akurasi",
                    leaderboardHelpAccuracy: "Akurasi: dari semua percobaan kalimat dan kata Anda, bagian suku kata Good (Okay dihitung \u00bd), dari 100. Perlu minimal {n} kalimat. Teks sendiri tidak dihitung.",
                    leaderboardEmptyAccuracy: "Belum ada yang menyelesaikan {n} kalimat.",
                    leaderboardAccuracy: "{n}%",
                    leaderboardMeAccuracyRank: "Anda: peringkat {rank} \u00b7 {accuracy}%",
                    leaderboardMeAccuracyNeed: "Latih {n} kalimat lagi untuk masuk papan akurasi.",
                    resultRetry: "Baca lagi",
                    resultNext: "Berikutnya",
                    itemProgress: "Sudah dilatih {n}× · terbaik {best}% · terakhir {last}%",
                    itemNotPracticed: "Belum dilatih.",
                    recentTitle: "Hasil terbaru saya",
                    recentEmpty: "Hasil Anda akan muncul di sini.",
                    recentPoints: "{n}%",
                    customTextPlaceholder: "Ketik atau tempel teks Mandarin di sini…",
                    customTextReady: "Teks Anda siap. Rekam diri Anda membacanya.",
                    customTextRequiresHanzi: "Masukkan aksara Mandarin untuk menilai nada Mandarin.",
                    sampleRecording: "Rekaman contoh (Pelajaran 5–8):",
                    loadingSamples: "Memuat rekaman contoh...",
                    matchingScript: "Teks yang cocok:",
                    scriptWithPinyin: "Teks yang cocok beserta pinyin",
                    showPinyin: "Tampilkan pinyin",
                    startRecording: "Mulai merekam",
                    stopAndAssess: "Berhenti & nilai",
                    recordingStop: "Merekam (klik untuk berhenti)...",
                    recording: "Merekam... Bicaralah dengan jelas ke mikrofon!",
                    latestRecording: "Rekaman mikrofon terbaru Anda:",
                    uploadAudio: "Unggah audio Anda:",
                    uploadHint: "Pilih rekaman diri Anda yang membaca teks terpilih. Format yang didukung: WAV, MP3, M4A, OGG, FLAC, WebM.",
                    selectedAudio: "Audio yang Anda pilih",
                    uploadAssess: "Unggah & nilai",
                    resultsTitle: "Hasil latihan membaca",
                    viewPitch: "Lihat grafik nada suara Anda (opsional)",
                    pitchDisplay: "Tampilan nada:",
                    relativePitch: "Nada relatif (semiton)",
                    f0Hz: "F0 (Hz)",
                    recordingLine: "Kontur nada suara Anda",
                    pitchNote: "Umpan balik nada hanya memakai kontur nada suara Anda. Audio contoh tidak dinilai.",
                    quickFeedback: "Masukan singkat",
                    feedbackQuestion: "Seberapa membantu latihan ini? 1 = tidak membantu, 5 = sangat membantu.",
                    feedbackImprove: "Apa yang perlu kami perbaiki? (opsional)",
                    feedbackPlaceholder: "Tulis komentar singkat",
                    sendFeedback: "Kirim masukan",
                    sendingFeedback: "Mengirim masukan...",
                    feedbackSaved: "Terima kasih atas masukan Anda!",
                    feedbackSaveFailed: "Masukan tidak dapat disimpan.",
                    practicingAs: "Berlatih sebagai {name} ({group})",
                    profileReady: "Profil {name} siap. Pilih contoh untuk memulai.",
                    voiceCalibrationTitle: "Atur suara Anda",
                    voiceCalibrationPrompt: "Contoh suara sekitar 10 detik membuat nada Anda dinilai sesuai suara Anda sendiri, sehingga umpan baliknya lebih adil.",
                    calPrivacy: "Rekaman Anda tidak disimpan. Hanya beberapa angka nada yang disimpan.",
                    voiceCalibrationLater: "Lewati dulu",
                    calibrationDone: "Mulai berlatih",
                    recordCalibration: "Mulai merekam",
                    stopCalibration: "Saya sudah selesai",
                    calibrationRecording: "Merekam… bacalah kelima bunyi dua kali.",
                    calibrationProcessing: "Memeriksa suara Anda…",
                    calibrationSaved: "Selesai! Mulai sekarang nada Anda dinilai sesuai suara Anda sendiri.",
                    calibrationTooShort: "Rekaman terlalu singkat. Teruslah membaca sampai tombol berhenti bisa ditekan.",
                    calibrationFailed: "Rekaman ini tidak bisa dipakai. Bacalah kelima bunyi dengan jelas, dua kali, lalu coba lagi.",
                    calStep1: "Bacalah lima bunyi ini dengan lantang, dua kali",
                    calStep1Hint: "Dengarkan dulu (\u25b6). Lalu ucapkan setiap bunyi dengan pelan dan panjangkan sedikit (m\u0101\u2011\u0101\u2011\u0101), dengan jeda singkat antarbunyi. Gunakan suara normal Anda yang paling nyaman: tidak perlu meniru tinggi nada contoh dan tidak perlu memaksa atau menegangkan suara. Suara yang santai menghasilkan nada paling alami. Nada 4 (m\u00e0) tetap pendek dan tegas.",
                    calStep2: "Tekan tombol dan mulai membaca",
                    calToneHigh: "tinggi",
                    calToneRising: "naik",
                    calToneDip: "turun-naik",
                    calToneFalling: "turun",
                    calToneLight: "ringan",
                    calKeepReading: "Teruskan membaca…",
                    calTryAgain: "Coba lagi",
                    calSeconds: "{seconds} dtk",
                    calClose: "Tutup",
                    micNoSound: "Suara Anda tidak terdengar. Bicaralah sedikit lebih keras atau periksa mikrofon.",
                    micLevel: "Tingkat suara mikrofon",
                    calReasonClipped: "Suara terlalu keras dan pecah. Menjauhlah sedikit dari mikrofon lalu coba lagi.",
                    calReasonNoisy: "Terlalu banyak bising di sekitar. Coba lagi di tempat yang lebih tenang.",
                    calReasonUnclear: "Nada suara Anda belum bisa diikuti dengan jelas. Bacalah kelima bunyi dua kali dengan tenang lalu coba lagi.",
                    calReasonQuiet: "Suara yang stabil terlalu sedikit, dan bicara lebih keras tidak membantu. Gunakan suara normal (bukan berbisik), panjangkan setiap bunyi sekitar setengah detik (m\u0101\u2011\u0101\u2011\u0101), lalu coba lagi.",
                    calibrateVoice: "Kalibrasi suara",
                    recalibrateVoice: "Kalibrasi ulang suara",
                    voiceCalibrated: "Rentang suara sudah disesuaikan",
                    voiceNotCalibrated: "Memakai rentang nada umum",
                    profileSaveFailed: "Profil pembicara tidak dapat disimpan.",
                    needProfile: "Masukkan nama dan kelompok peserta terlebih dahulu.",
                    chooseAudio: "Pilih berkas audio terlebih dahulu.",
                    unsupportedAudio: "Format berkas tidak didukung. Pilih WAV, MP3, M4A, OGG, FLAC, atau WebM.",
                    emptyAudioFile: "Berkas audio ini kosong. Pilih rekaman lain.",
                    uploadReady: "{file} siap. Klik Unggah & nilai untuk memeriksanya terhadap teks terpilih.",
                    chooseSample: "Pilih contoh acuan terlebih dahulu.",
                    noAudio: "Tidak ada audio yang terekam. Rekam lagi atau pilih berkas lain.",
                    processing: "Memproses audio dan memeriksa pelafalan...",
                    assessmentSaved: "Penilaian selesai dan tersimpan!",
                    assessmentFailed: "Penilaian gagal: {error}",
                    noSampleAudio: "Tidak ditemukan contoh audio yang cocok.",
                    sampleDataUnavailable: "Audio contoh atau data teks tidak tersedia di server.",
                    sampleLoadFailed: "Rekaman contoh tidak dapat dimuat dari server.",
                    sampleReady: "Teks siap. Rekam diri Anda atau unggah berkas audio yang membacanya.",
                    sampleAudioUnavailable: "Audio contoh tidak tersedia; Anda tetap bisa berlatih dari teksnya.",
                    sampleAudioLabel: "Dengarkan audio contoh:",
                    sampleAudioLoadError: "Audio contoh tidak dapat dimuat. Coba pilih butir itu lagi.",
                    microphoneHttps: "Akses mikrofon memerlukan HTTPS atau localhost. Anda dapat mengunggah berkas audio sebagai gantinya.",
                    recordingUnsupported: "Perekaman audio tidak didukung di peramban ini. Anda dapat mengunggah berkas audio sebagai gantinya.",
                    microphoneFailed: "Tidak dapat mengakses mikrofon: {error}",
                    serverNoResponse: "Server mengembalikan HTTP {status} tanpa respons penilaian.",
                    lessonOption: "Pelajaran {lesson} · Cerita {story} · Adegan {scene} · {character}",
                    likelyCorrect: "Bagus",
                    practice: "Perlu latihan",
                    uncertain: "Cukup",
                    insufficientEvidence: "Cukup",
                    mismatch: "Tidak dinilai: teks tidak cocok",
                    transcriptUnchecked: "Tidak dinilai: transkrip belum diperiksa",
                    unassessable: "Audio tidak cukup untuk dinilai",
                    notScored: "Tidak dinilai",
                    noMatch: "Pengenalan ucapan tidak cocok dengan teks; umpan balik nada per frasa dilewati.",
                    partialMatch: "Pengenalan ucapan hanya sebagian cocok dengan teks; frasa yang tidak cocok tidak dinilai nadanya.",
                    unchecked: "Pengenalan ucapan tidak diperiksa, sehingga umpan balik nada per frasa tidak tersedia.",
                    matched: "Pengenalan ucapan cocok dengan teks; umpan balik nada ditampilkan per frasa.",
                    renderPhraseScores: "Setiap kartu memeriksa urutan nada yang diharapkan untuk seluruh frasa.",
                    adviceTone1: "“{char}” – nada 1: jaga suara tetap tinggi dan datar, tanpa naik atau turun.",
                    adviceTone2: "“{char}” – nada 2: mulai dari nada sedang lalu naik dengan jelas, seperti bertanya “hah?”.",
                    adviceTone3: "“{char}” – nada 3: biarkan suara turun rendah, seperti “hmm…” pelan, jangan tetap tinggi.",
                    adviceTone4: "“{char}” – nada 4: mulai dari nada tinggi lalu turun cepat, seperti “Tidak!” yang tegas.",
                    retrySlower: "Ucapkan kata ini sedikit lebih pelan dan panjangkan sedikit \u2013 jangan terburu-buru, agar nadanya terdengar jelas.",
                    retryShortFall: "Nada 4 memang pendek dan tegas: jangan dipanjangkan. Mulai dari tinggi lalu turun tegas dalam satu gerakan.",
                    retrySteady: "Jaga suara tetap stabil dan sedikit lebih keras. Hindari berbisik atau suara serak.",
                    retryClearer: "Buat nadanya sedikit lebih jelas agar naik-turunnya terdengar.",
                    retryReadExactly: "Bacalah bagian ini persis seperti tertulis, dari awal sampai akhir.",
                    retryAgain: "Coba ucapkan sekali lagi, dengan jelas dan kecepatan yang wajar.",
                    qualityClipped: "Rekaman terdistorsi. Menjauhlah sedikit dari mikrofon lalu coba lagi.",
                    qualityTooQuiet: "Rekaman terlalu pelan. Bicaralah lebih keras atau dekatkan mikrofon.",
                    qualityNoisy: "Terlalu banyak bising latar. Coba lagi di tempat yang lebih tenang.",
                    qualityTooShort: "Rekaman terlalu pendek. Coba lagi.",
                    qualityTooLong: "Rekaman terlalu panjang. Rekam hanya frasa latihan.",
                    qualityNoSpeech: "Tidak ada ucapan terdeteksi. Coba lagi.",
                    qualityNoVoicing: "Suara yang jelas tidak terdeteksi. Bicaralah dengan suara, jangan berbisik.",
                    alignmentFailed: "Rekaman ini belum bisa dianalisis karena masalah sementara di sisi kami, bukan karena suara Anda. Silakan coba lagi.",
                    busyWait: "Biasanya hanya butuh beberapa detik.",
                    notHeard: "Tidak terdengar",
                    retryNotHeard: "Bagian ini tidak terdengar. Bicaralah sedikit lebih keras, bacalah seluruhnya, lalu coba lagi.",
                    retryFullThird: "Ucapkan nada ketiga ini secara penuh: turunkan suara ke bawah, lalu naikkan sedikit di akhir.",
                    expectedSequence: "Diharapkan",
                    detectedSequence: "Terdeteksi",
                    sequenceUnavailable: "Bukti nada tidak cukup",
                    noProbabilityModel: "Label nada memakai aturan eksperimental; sistem ini tidak memiliki probabilitas nada yang dilatih atau dikalibrasi.",
                    pronunciationFeedback: "Umpan balik nada",
                    phraseScore: "Skor frasa {score}",
                    phraseScoreUnavailable: "Skor frasa tidak tersedia",
                    viewJsonOutput: "Lihat respons JSON",
                    scriptProvidedOnly: "Dinilai terhadap teks yang ditampilkan di atas, jadi bacalah persis seperti tertulis.",
                    syllableDetails: "Rincian suku kata",
                    syllableColumn: "Suku kata",
                    toneColumn: "Nada",
                    resultColumn: "Hasil",
                    noteColumn: "Catatan",
                    toneShapeNote: "Garis kecil menggambarkan nada target; grafik menunjukkan nada suara yang terukur.",
                    toneShapeAria: "Kontur skematis untuk {tone}",
                    toneT1: "T1",
                    toneT2: "T2",
                    toneT3: "T3",
                    toneT4: "T4",
                    neutralTone: "Netral",
                    thirdToneSandhi: "perubahan nada ketiga",
                    buSandhi: "perubahan nada 不",
                    yiSandhi: "perubahan nada 一",
                    taiwanReading: "bacaan Taiwan",
                    phraseFinalTone: "nada akhir frasa",
                    lexicalToneRule: "nada kamus",
                    matchedTone: "Kontur nada sesuai dengan target.",
                    differentTone: "Terdeteksi kontur nada yang berbeda.",
                    ambiguousTone: "Kontur ini bisa cocok dengan lebih dari satu nada.",
                    neutralToneUncertain: "Nada netral bergantung pada konteks.",
                    boundaryDisagreement: "Kontur bagian tengah dan seluruh suku kata tidak sejalan.",
                    alignmentLow: "Waktu suku kata tidak jelas.",
                    tooFewFrames: "Data nada bersuara terlalu sedikit.",
                    sparseF0: "Data nada memiliki celah.",
                    octaveJump: "Pelacakan nada melompat tiba-tiba.",
                    incompleteContour: "Data nada tidak mencakup seluruh suku kata.",
                    pinyinMismatch: "Suku kata yang dikenali berbeda dari teks.",
                    transcriptNotChecked: "Pengenalan ucapan tidak diperiksa.",
                    phraseNeedsPractice: "Satu atau lebih suku kata perlu latihan.",
                    phraseUncertain: "Satu atau lebih suku kata belum pasti.",
                    otherSyllables: "Suku kata lain",
                    phrase: "Frasa",
                    sentence: "Kalimat {number}",
                    scriptLabel: "Teks: {script}",
                    noScores: "Belum ada skor pelafalan.",
                    noPitch: "Tidak ada nada yang dapat diukur pada frasa ini.",
                    f0Title: "F0 dalam Hz",
                    relativeTitle: "Nada relatif",
                    relativeSemitone: "Semiton dari median nada Anda",
                    midiNote: "Nada MIDI",
                    midiTitle: "Nada sebagai nada MIDI",
                    midiLabel: "Nada MIDI (A4 = 440 Hz = 69)",
                    halfThirdTone: "Nada 3 setengah",
                    clause: "kalimat {number}",
                    pitchDescription: "Kontur nada suara Anda ditampilkan untuk seluruh frasa. Celah pada nada yang terukur dibiarkan kosong.",
                    reference: "Acuan",
                    yourRecording: "Rekaman Anda",
                    lowConfidence: " (penyelarasan atau nada kurang yakin)",
                    relative: "Relatif",
                    hz: "Hz",
                    invalidAudio: "Format berkas audio tidak valid.",
                    emptyScript: "Teks masukan tidak boleh kosong.",
                    invalidProfile: "Masukkan profil pembicara yang valid sebelum berlatih.",
                    sampleMissing: "Contoh prototipe yang dipilih tidak ditemukan.",
                    noHanScript: "Teks tidak berisi aksara Han yang valid untuk menilai pelafalan.",
                    emptyName: "Masukkan nama.",
                    invalidNameLength: "Nama harus terdiri dari 1 sampai 100 karakter.",
                    invalidGroup: "Pilih Orang Taiwan atau Orang asing.",
                    attemptMissing: "Percobaan latihan yang dipilih tidak ditemukan.",
                    invalidFeedbackRating: "Penilaian harus berupa bilangan bulat dari 1 sampai 5.",
                    feedbackTooLong: "Komentar masukan maksimal 2.000 karakter.",
                    scriptMismatch: "Contoh terpilih dan teks tidak cocok. Hanya perbedaan tanda baca dan spasi yang diizinkan. Diharapkan: {expected} Dimasukkan: {received}",
                },
                hi: {
                    language: "भाषा",
                    languageEn: "English",
                    languageZh: "中文（繁體）",
                    languageVi: "Tiếng Việt",
                    speakerProfile: "वक्ता प्रोफ़ाइल",
                    changeSpeaker: "वक्ता बदलें",
                    calListen: "यह ध्वनि सुनें", calListenAll: "\u25b6 पाँचों ध्वनियाँ सुनें",
                    calibrationSavedSlower: "आपकी आवाज़ सहेज ली गई। सुझाव: अगली बार थोड़ा धीमे पढ़ें और हर ध्वनि को खींचकर बोलें \u2013 पहचानना आसान होगा।",
                    paceTooFast: "कुछ ध्वनियाँ विश्लेषण के लिए बहुत तेज़ थीं। थोड़ा धीमे पढ़ें, या हर ध्वनि को थोड़ा खींचें ताकि स्वर आसानी से पहचाने जा सकें।",
                    lengthLonger: "ऐसा लगता है कि आपने स्क्रीन पर दिए वाक्य से ज़्यादा बोला। कृपया स्क्रीन पर लिखा पाठ ठीक वैसा ही पढ़ें।",
                    lengthShorter: "ऐसा लगता है कि वाक्य के कुछ शब्द छूट गए। कृपया स्क्रीन पर लिखा पूरा पाठ पढ़ें।",
                    lengthDifferent: "आपने जो बोला वह स्क्रीन पर दिए वाक्य से मेल नहीं खाता लगता। कृपया स्क्रीन पर लिखा पाठ ठीक वैसा ही पढ़ें।",
                    profilePrompt: "बोलने का अभ्यास शुरू करने से पहले अपना नाम और प्रतिभागी समूह दर्ज करें।",
                    profileHint: "ऐसा नाम चुनें जो आपको दूसरे लोगों से अलग पहचान दे; वही नाम डालने पर पुराना अभ्यास-इतिहास ही जारी रहेगा।",
                    name: "नाम",
                    participantGroup: "प्रतिभागी समूह",
                    chooseOne: "कोई एक चुनें...",
                    taiwanese: "ताइवानी",
                    foreigner: "विदेशी",
                    continue: "आगे बढ़ें",
                    cancel: "रद्द करें",
                    appTitle: "ताइवानी मंदारिन उच्चारण अभ्यास",
                    practiceInstructions: "कोई पाठ-नमूना चुनें या अपना मंदारिन पाठ लिखें, फिर उसे पढ़ते हुए अपनी रिकॉर्डिंग करें:",
                    practiceSource: "अभ्यास का पाठ",
                    bookTextMode: "पुस्तक का नमूना",
                    customTextMode: "मेरा अपना पाठ",
                    chapter5WordMode: "विषय",
                    previousWord: "पिछला",
                    nextWord: "अगला",
                    chapter5Vocabulary: "शब्दावली",
                    chapter5Phrase: "वाक्यांश",
                    chapter5Meaning: "{meaning}",
                    chapter5OptionalHint: "हल्के रंग वाला हिस्सा वैकल्पिक है: “{short}” और “{full}” दोनों मान्य हैं।",
                    chapter5ReferenceAudio: "केवल संदर्भ के लिए – अंकन में इस्तेमाल नहीं होता।",
                    chapter5Progress: "{current} / {total}",
                    chapter5Unavailable: "विषय-आधारित अभ्यास सामग्री उपलब्ध नहीं है।",
                    chapter5Ready: "तैयार। इसे पढ़ते हुए अपनी रिकॉर्डिंग करें।",
                    chapter5Topic: "विषय",
                    chapter5AllTopics: "सभी विषय",
                    chapter5AllInTopic: "{topic} में सभी",
                    customTextPreview: "दर्ज किए गए पाठ का पूर्वावलोकन",
                    customScriptLabel: "पढ़ने का पाठ:",
                    customTextHint: "कोई भी मंदारिन पाठ लिखें। उसका पाठ्यपुस्तक में होना ज़रूरी नहीं है।",
                    customSampleListen: "नमूना सुनें",
                    customSampleBusy: "नमूना बन रहा है…",
                    customSampleNote: "कंप्यूटर की आवाज़ आपका पाठ थोड़ा धीमे पढ़ती है। यह केवल मार्गदर्शन है; कुछ शब्द शिक्षक के पढ़ने से थोड़ा अलग लग सकते हैं।",
                    customSampleError: "नमूना नहीं बन सका। फिर कोशिश करें।",
                    customSampleUnavailable: "इस सर्वर पर नमूना आवाज़ उपलब्ध नहीं है।",
                    leaderboardTitle: "शीर्ष 10",
                    leaderboardHelp: "अंक: हर वाक्य का सबसे अच्छा प्रयास (Good अक्षर-खंड = 1, Okay = \u00bd), जोड़कर। आपका अपना पाठ नहीं गिना जाता।",
                    leaderboardEmpty: "अभी कोई अंक नहीं। पहले बनिए!",
                    leaderboardPoints: "{n} अंक",
                    leaderboardLessons: "{n} वाक्य",
                    leaderboardMeRank: "आप: #{rank} \u00b7 {points} अंक",
                    leaderboardMeNone: "बोर्ड में आने के लिए एक वाक्य का अभ्यास करें।",
                    boardTabPoints: "अंक",
                    boardTabAccuracy: "सटीकता",
                    leaderboardHelpAccuracy: "सटीकता: आपके सभी वाक्य और शब्द प्रयासों में Good अक्षर-खंडों का अनुपात (Okay = \u00bd), 100 में से। कम से कम {n} वाक्य ज़रूरी हैं। अपना लिखा पाठ नहीं गिना जाता।",
                    leaderboardEmptyAccuracy: "अभी तक किसी ने {n} वाक्य पूरे नहीं किए।",
                    leaderboardAccuracy: "{n}%",
                    leaderboardMeAccuracyRank: "आप: #{rank} \u00b7 {accuracy}%",
                    leaderboardMeAccuracyNeed: "सटीकता बोर्ड में आने के लिए {n} और वाक्य अभ्यास करें।",
                    resultRetry: "फिर से पढ़ें",
                    resultNext: "अगला",
                    itemProgress: "{n} बार अभ्यास · सर्वश्रेष्ठ {best}% · अंतिम {last}%",
                    itemNotPracticed: "अभी अभ्यास नहीं किया।",
                    recentTitle: "मेरे हाल के परिणाम",
                    recentEmpty: "आपके परिणाम यहाँ दिखेंगे।",
                    recentPoints: "{n}%",
                    customTextPlaceholder: "यहाँ मंदारिन पाठ लिखें या चिपकाएँ…",
                    customTextReady: "आपका पाठ तैयार है। इसे पढ़ते हुए अपनी रिकॉर्डिंग करें।",
                    customTextRequiresHanzi: "मंदारिन टोन जाँचने के लिए चीनी अक्षर दर्ज करें।",
                    sampleRecording: "नमूना रिकॉर्डिंग (पाठ 5–8):",
                    loadingSamples: "नमूना रिकॉर्डिंग लोड हो रही हैं...",
                    matchingScript: "मिलान वाला पाठ:",
                    scriptWithPinyin: "पिनयिन सहित मिलान वाला पाठ",
                    showPinyin: "पिनयिन दिखाएँ",
                    startRecording: "रिकॉर्डिंग शुरू करें",
                    stopAndAssess: "रोकें और जाँचें",
                    recordingStop: "रिकॉर्ड हो रहा है (रोकने के लिए क्लिक करें)...",
                    recording: "रिकॉर्ड हो रहा है... माइक्रोफ़ोन में साफ़ बोलें!",
                    latestRecording: "आपकी ताज़ा माइक्रोफ़ोन रिकॉर्डिंग:",
                    uploadAudio: "अपनी ऑडियो अपलोड करें:",
                    uploadHint: "चुने हुए पाठ को पढ़ते हुए अपनी रिकॉर्डिंग चुनें। समर्थित फ़ॉर्मैट: WAV, MP3, M4A, OGG, FLAC, WebM।",
                    selectedAudio: "आपकी चुनी हुई ऑडियो",
                    uploadAssess: "अपलोड करें और जाँचें",
                    resultsTitle: "पठन अभ्यास के परिणाम",
                    viewPitch: "अपना पिच ग्राफ़ देखें (वैकल्पिक)",
                    pitchDisplay: "पिच प्रदर्शन:",
                    relativePitch: "सापेक्ष पिच (सेमीटोन)",
                    f0Hz: "F0 (Hz)",
                    recordingLine: "आपकी पिच रेखा",
                    pitchNote: "टोन की प्रतिक्रिया केवल आपकी पिच रेखा पर आधारित है। नमूना ऑडियो का अंकन नहीं होता।",
                    quickFeedback: "त्वरित प्रतिक्रिया",
                    feedbackQuestion: "यह अभ्यास कितना उपयोगी रहा? 1 = उपयोगी नहीं, 5 = बहुत उपयोगी।",
                    feedbackImprove: "हमें क्या सुधारना चाहिए? (वैकल्पिक)",
                    feedbackPlaceholder: "एक छोटी टिप्पणी लिखें",
                    sendFeedback: "प्रतिक्रिया भेजें",
                    sendingFeedback: "प्रतिक्रिया भेजी जा रही है...",
                    feedbackSaved: "आपकी प्रतिक्रिया के लिए धन्यवाद!",
                    feedbackSaveFailed: "प्रतिक्रिया सहेजी नहीं जा सकी।",
                    practicingAs: "{name} ({group}) के रूप में अभ्यास",
                    profileReady: "{name} की प्रोफ़ाइल तैयार है। शुरू करने के लिए कोई नमूना चुनें।",
                    voiceCalibrationTitle: "अपनी आवाज़ सेट करें",
                    voiceCalibrationPrompt: "लगभग 10 सेकंड का नमूना हमें आपके टोन को आपकी अपनी आवाज़ के हिसाब से परखने देता है, ताकि प्रतिक्रिया निष्पक्ष रहे।",
                    calPrivacy: "आपकी रिकॉर्डिंग सहेजी नहीं जाती। केवल कुछ पिच-आँकड़े सहेजे जाते हैं।",
                    voiceCalibrationLater: "अभी छोड़ें",
                    calibrationDone: "अभ्यास शुरू करें",
                    recordCalibration: "रिकॉर्डिंग शुरू करें",
                    stopCalibration: "मेरा पढ़ना हो गया",
                    calibrationRecording: "रिकॉर्ड हो रहा है… पाँचों ध्वनियाँ दो बार पढ़ें।",
                    calibrationProcessing: "आपकी आवाज़ जाँची जा रही है…",
                    calibrationSaved: "हो गया! अब आपके टोन आपकी अपनी आवाज़ के हिसाब से परखे जाएँगे।",
                    calibrationTooShort: "रिकॉर्डिंग बहुत छोटी थी। तब तक पढ़ते रहें जब तक बटन दबाने लायक न हो जाए।",
                    calibrationFailed: "यह रिकॉर्डिंग इस्तेमाल नहीं हो सकी। पाँचों ध्वनियाँ साफ़ दो बार पढ़कर फिर कोशिश करें।",
                    calStep1: "इन पाँच ध्वनियों को ज़ोर से, दो बार पढ़ें",
                    calStep1Hint: "पहले सुनें (\u25b6)। फिर हर ध्वनि को धीरे बोलें और थोड़ा खींचकर बोलें (m\u0101\u2011\u0101\u2011\u0101), ध्वनियों के बीच छोटा विराम लें। अपनी सामान्य, आरामदायक आवाज़ में बोलें: नमूने की ऊँचाई की नकल करने या आवाज़ पर ज़ोर लगाने की ज़रूरत नहीं है। सहज आवाज़ में स्वर सबसे स्वाभाविक निकलते हैं। स्वर 4 (m\u00e0) छोटा और तीखा ही रहता है।",
                    calStep2: "बटन दबाएँ और पढ़ना शुरू करें",
                    calToneHigh: "ऊँचा",
                    calToneRising: "चढ़ता",
                    calToneDip: "गिर-चढ़",
                    calToneFalling: "गिरता",
                    calToneLight: "हल्का",
                    calKeepReading: "पढ़ते रहें…",
                    calTryAgain: "फिर कोशिश करें",
                    calSeconds: "{seconds} सेकंड",
                    calClose: "बंद करें",
                    micNoSound: "आपकी आवाज़ सुनाई नहीं दे रही। थोड़ा ज़ोर से बोलें या माइक्रोफ़ोन जाँचें।",
                    micLevel: "माइक्रोफ़ोन स्तर",
                    calReasonClipped: "आवाज़ बहुत तेज़ होकर फट गई। माइक्रोफ़ोन से थोड़ा दूर हटकर फिर कोशिश करें।",
                    calReasonNoisy: "आसपास बहुत शोर था। किसी शांत जगह पर फिर कोशिश करें।",
                    calReasonUnclear: "आपकी पिच साफ़ पकड़ में नहीं आई। पाँचों ध्वनियाँ शांति से दो बार पढ़कर फिर कोशिश करें।",
                    calReasonQuiet: "स्थिर आवाज़ बहुत कम मिली, और ज़ोर से बोलने से मदद नहीं मिलती। अपनी सामान्य आवाज़ में (फुसफुसाकर नहीं) हर ध्वनि को थोड़ा लंबा, लगभग आधा सेकंड, बोलें (m\u0101\u2011\u0101\u2011\u0101), फिर कोशिश करें।",
                    calibrateVoice: "आवाज़ कैलिब्रेट करें",
                    recalibrateVoice: "आवाज़ दोबारा कैलिब्रेट करें",
                    voiceCalibrated: "आवाज़ की सीमा आपके अनुसार सेट है",
                    voiceNotCalibrated: "सामान्य पिच-सीमा का उपयोग हो रहा है",
                    profileSaveFailed: "वक्ता प्रोफ़ाइल सहेजी नहीं जा सकी।",
                    needProfile: "पहले अपना नाम और प्रतिभागी समूह दर्ज करें।",
                    chooseAudio: "पहले कोई ऑडियो फ़ाइल चुनें।",
                    unsupportedAudio: "यह फ़ाइल फ़ॉर्मैट समर्थित नहीं है। WAV, MP3, M4A, OGG, FLAC या WebM चुनें।",
                    emptyAudioFile: "यह ऑडियो फ़ाइल खाली है। कोई दूसरी रिकॉर्डिंग चुनें।",
                    uploadReady: "{file} तैयार है। चुने हुए पाठ से मिलान जाँचने के लिए “अपलोड करें और जाँचें” पर क्लिक करें।",
                    chooseSample: "पहले कोई संदर्भ नमूना चुनें।",
                    noAudio: "कोई ऑडियो रिकॉर्ड नहीं हुई। फिर से रिकॉर्ड करें या कोई दूसरी फ़ाइल चुनें।",
                    processing: "ऑडियो प्रोसेस हो रही है और उच्चारण जाँचा जा रहा है...",
                    assessmentSaved: "जाँच पूरी हुई और सहेज ली गई!",
                    assessmentFailed: "जाँच विफल रही: {error}",
                    noSampleAudio: "मेल खाते ऑडियो नमूने नहीं मिले।",
                    sampleDataUnavailable: "सर्वर पर नमूना ऑडियो या पाठ का डेटा उपलब्ध नहीं है।",
                    sampleLoadFailed: "सर्वर से नमूना रिकॉर्डिंग लोड नहीं हो सकीं।",
                    sampleReady: "पाठ तैयार है। इसे पढ़ते हुए अपनी रिकॉर्डिंग करें या ऑडियो फ़ाइल अपलोड करें।",
                    sampleAudioUnavailable: "नमूना ऑडियो उपलब्ध नहीं है; आप पाठ से अभ्यास कर सकते हैं।",
                    sampleAudioLabel: "नमूना ऑडियो सुनें:",
                    sampleAudioLoadError: "नमूना ऑडियो लोड नहीं हो सकी। वही आइटम फिर से चुनकर देखें।",
                    microphoneHttps: "माइक्रोफ़ोन के लिए HTTPS या localhost ज़रूरी है। आप इसके बजाय ऑडियो फ़ाइल अपलोड कर सकते हैं।",
                    recordingUnsupported: "इस ब्राउज़र में ऑडियो रिकॉर्डिंग समर्थित नहीं है। आप इसके बजाय ऑडियो फ़ाइल अपलोड कर सकते हैं।",
                    microphoneFailed: "माइक्रोफ़ोन तक पहुँच नहीं मिल सकी: {error}",
                    serverNoResponse: "सर्वर ने जाँच के उत्तर के बिना HTTP {status} लौटाया।",
                    lessonOption: "पाठ {lesson} · कहानी {story} · दृश्य {scene} · {character}",
                    likelyCorrect: "बढ़िया",
                    practice: "अभ्यास करें",
                    uncertain: "ठीक-ठाक",
                    insufficientEvidence: "ठीक-ठाक",
                    mismatch: "अंकन नहीं हुआ: पाठ मेल नहीं खाता",
                    transcriptUnchecked: "अंकन नहीं हुआ: ट्रांसक्रिप्ट जाँचा नहीं गया",
                    unassessable: "अंकन के लिए ऑडियो पर्याप्त नहीं",
                    notScored: "अंकन नहीं हुआ",
                    noMatch: "वाक् पहचान पाठ से मेल नहीं खाई; वाक्यांश की टोन प्रतिक्रिया छोड़ दी गई।",
                    partialMatch: "वाक् पहचान पाठ से केवल आंशिक रूप से मेल खाई; बेमेल वाक्यांशों की टोन का अंकन नहीं होता।",
                    unchecked: "वाक् पहचान की जाँच नहीं हुई, इसलिए वाक्यांश की टोन प्रतिक्रिया उपलब्ध नहीं है।",
                    matched: "वाक् पहचान पाठ से मेल खाई; टोन प्रतिक्रिया वाक्यांश के अनुसार दिखाई गई है।",
                    renderPhraseScores: "हर कार्ड पूरे वाक्यांश के अपेक्षित टोन-क्रम को जाँचता है।",
                    adviceTone1: "“{char}” – टोन 1: आवाज़ ऊँची और एक-सी रखें, न ऊपर जाए न नीचे।",
                    adviceTone2: "“{char}” – टोन 2: बीच की पिच से शुरू करें और साफ़ ऊपर उठाएँ, जैसे “हैं?” पूछ रहे हों।",
                    adviceTone3: "“{char}” – टोन 3: आवाज़ को नीचे उतरने दें, जैसे धीमे से “हम्म…” कहते हैं; ऊँची न रखें।",
                    adviceTone4: "“{char}” – टोन 4: ऊँची आवाज़ से शुरू करें और तेज़ी से नीचे गिराएँ, जैसे दृढ़ता से “नहीं!” कहते हैं।",
                    retrySlower: "इस शब्द को थोड़ा धीमे और थोड़ा खींचकर बोलें \u2013 जल्दबाज़ी न करें, ताकि स्वर साफ़ सुनाई दें।",
                    retryShortFall: "स्वर 4 छोटा और तीखा होता है: इसे खींचें नहीं। ऊँचे से शुरू करें और एक ही झटके में मज़बूती से नीचे गिराएँ।",
                    retrySteady: "आवाज़ स्थिर और थोड़ी ऊँची रखें। फुसफुसाने या खरखराती आवाज़ से बचें।",
                    retryClearer: "टोन को थोड़ा और स्पष्ट करें ताकि उतार-चढ़ाव साफ़ सुनाई दे।",
                    retryReadExactly: "इस हिस्से को जैसा लिखा है वैसा ही, शुरू से अंत तक पढ़ें।",
                    retryAgain: "एक बार फिर बोलकर देखें, साफ़ और सामान्य गति से।",
                    qualityClipped: "रिकॉर्डिंग में विकृति है। माइक्रोफ़ोन से थोड़ा दूर हटें और फिर कोशिश करें।",
                    qualityTooQuiet: "रिकॉर्डिंग बहुत धीमी है। ज़ोर से बोलें या माइक्रोफ़ोन के पास आएँ।",
                    qualityNoisy: "पृष्ठभूमि में बहुत शोर है। किसी शांत जगह पर फिर कोशिश करें।",
                    qualityTooShort: "रिकॉर्डिंग बहुत छोटी है। फिर कोशिश करें।",
                    qualityTooLong: "रिकॉर्डिंग बहुत लंबी है। केवल अभ्यास वाला वाक्यांश रिकॉर्ड करें।",
                    qualityNoSpeech: "कोई आवाज़ नहीं मिली। फिर कोशिश करें।",
                    qualityNoVoicing: "साफ़ आवाज़ नहीं मिली। फुसफुसाने के बजाय आवाज़ के साथ बोलें।",
                    alignmentFailed: "इस बार रिकॉर्डिंग का विश्लेषण नहीं हो सका। यह हमारी ओर की अस्थायी समस्या है, आपकी आवाज़ की नहीं। कृपया फिर कोशिश करें।",
                    busyWait: "इसमें आम तौर पर कुछ ही सेकंड लगते हैं।",
                    notHeard: "सुनाई नहीं दिया",
                    retryNotHeard: "यह हिस्सा सुनाई नहीं दिया। थोड़ा ज़ोर से बोलें, पूरा पढ़ें और फिर कोशिश करें।",
                    retryFullThird: "इस तीसरे टोन को पूरा बोलें: आवाज़ नीचे गिराएँ, फिर अंत में थोड़ा ऊपर उठाएँ।",
                    expectedSequence: "अपेक्षित",
                    detectedSequence: "पहचाना गया",
                    sequenceUnavailable: "टोन के पर्याप्त प्रमाण नहीं",
                    noProbabilityModel: "टोन के लेबल प्रायोगिक नियमों पर आधारित हैं; इस प्रणाली में प्रशिक्षित या कैलिब्रेट की गई टोन-संभावनाएँ नहीं हैं।",
                    pronunciationFeedback: "टोन प्रतिक्रिया",
                    phraseScore: "वाक्यांश का स्कोर {score}",
                    phraseScoreUnavailable: "वाक्यांश का स्कोर उपलब्ध नहीं",
                    viewJsonOutput: "JSON उत्तर देखें",
                    scriptProvidedOnly: "अंकन ऊपर दिखाए गए पाठ के आधार पर होता है, इसलिए इसे ठीक वैसा ही पढ़ें जैसा लिखा है।",
                    syllableDetails: "अक्षर-खंड का विवरण",
                    syllableColumn: "अक्षर-खंड",
                    toneColumn: "टोन",
                    resultColumn: "परिणाम",
                    noteColumn: "टिप्पणी",
                    toneShapeNote: "छोटी रेखा लक्ष्य टोन का आकार दिखाती है; चार्ट आपकी मापी गई पिच दिखाता है।",
                    toneShapeAria: "{tone} की योजनाबद्ध रेखा",
                    toneT1: "T1",
                    toneT2: "T2",
                    toneT3: "T3",
                    toneT4: "T4",
                    neutralTone: "तटस्थ",
                    thirdToneSandhi: "तीसरे टोन का परिवर्तन",
                    buSandhi: "不 का टोन परिवर्तन",
                    yiSandhi: "一 का टोन परिवर्तन",
                    taiwanReading: "ताइवानी उच्चारण",
                    phraseFinalTone: "वाक्यांश के अंत का टोन",
                    lexicalToneRule: "शब्दकोश टोन",
                    matchedTone: "टोन की रेखा लक्ष्य से मेल खाती है।",
                    differentTone: "टोन की एक अलग रेखा मिली।",
                    ambiguousTone: "यह रेखा एक से अधिक टोन से मेल खा सकती है।",
                    neutralToneUncertain: "तटस्थ टोन संदर्भ पर निर्भर करता है।",
                    boundaryDisagreement: "अक्षर-खंड के बीच के हिस्से और पूरी रेखा में अंतर है।",
                    alignmentLow: "अक्षर-खंड का समय अस्पष्ट है।",
                    tooFewFrames: "आवाज़ वाली पिच का डेटा बहुत कम है।",
                    sparseF0: "पिच के डेटा में अंतराल हैं।",
                    octaveJump: "पिच ट्रैकिंग में अचानक छलांग है।",
                    incompleteContour: "पिच का डेटा पूरे अक्षर-खंड को नहीं ढकता।",
                    pinyinMismatch: "पहचाना गया अक्षर-खंड पाठ से अलग है।",
                    transcriptNotChecked: "वाक् पहचान की जाँच नहीं हुई।",
                    phraseNeedsPractice: "एक या अधिक अक्षर-खंडों पर अभ्यास की ज़रूरत है।",
                    phraseUncertain: "एक या अधिक अक्षर-खंड अनिश्चित हैं।",
                    otherSyllables: "अन्य अक्षर-खंड",
                    phrase: "वाक्यांश",
                    sentence: "वाक्य {number}",
                    scriptLabel: "पाठ: {script}",
                    noScores: "उच्चारण के स्कोर उपलब्ध नहीं हैं।",
                    noPitch: "इस वाक्यांश में मापने योग्य पिच नहीं मिली।",
                    f0Title: "Hz में F0",
                    relativeTitle: "सापेक्ष पिच",
                    relativeSemitone: "आपकी पिच के माध्यिका से सेमीटोन",
                    midiNote: "MIDI नोट",
                    midiTitle: "MIDI नोट के रूप में पिच",
                    midiLabel: "MIDI नोट (A4 = 440 Hz = 69)",
                    halfThirdTone: "आधा-T3",
                    clause: "वाक्य {number}",
                    pitchDescription: "पूरे वाक्यांश के लिए आपकी पिच रेखा दिखाई गई है। जहाँ पिच नहीं मापी जा सकी वहाँ जगह खाली रहती है।",
                    reference: "संदर्भ",
                    yourRecording: "आपकी रिकॉर्डिंग",
                    lowConfidence: " (संरेखण या पिच पर भरोसा कम है)",
                    relative: "सापेक्ष",
                    hz: "Hz",
                    invalidAudio: "ऑडियो फ़ाइल का फ़ॉर्मैट अमान्य है।",
                    emptyScript: "दर्ज किया गया पाठ खाली नहीं हो सकता।",
                    invalidProfile: "अभ्यास से पहले एक मान्य वक्ता प्रोफ़ाइल दर्ज करें।",
                    sampleMissing: "चुना हुआ प्रोटोटाइप नमूना नहीं मिला।",
                    noHanScript: "उच्चारण का अंकन करने के लिए पाठ में कोई मान्य चीनी अक्षर नहीं है।",
                    emptyName: "नाम दर्ज करें।",
                    invalidNameLength: "नाम 1 से 100 अक्षरों का होना चाहिए।",
                    invalidGroup: "ताइवानी या विदेशी में से चुनें।",
                    attemptMissing: "चुना हुआ अभ्यास-प्रयास नहीं मिला।",
                    invalidFeedbackRating: "रेटिंग 1 से 5 के बीच की पूर्ण संख्या होनी चाहिए।",
                    feedbackTooLong: "प्रतिक्रिया की टिप्पणी 2,000 अक्षरों या उससे कम होनी चाहिए।",
                    scriptMismatch: "चुना हुआ नमूना और पाठ मेल नहीं खाते। केवल विराम-चिह्न और खाली स्थान का अंतर मान्य है। अपेक्षित: {expected} दर्ज किया: {received}",
                }
            };

            let currentLanguage = "en";

            function t(key, variables = {}) {
                const template = (LANGUAGE_TEXT[currentLanguage] || LANGUAGE_TEXT.en)[key] || LANGUAGE_TEXT.en[key] || key;
                return template.replace(/\\{(\\w+)\\}/g, (_, name) => String(variables[name] ?? ""));
            }

            function findTranslationDescriptor(message) {
                for (const translations of Object.values(LANGUAGE_TEXT)) {
                    for (const [key, template] of Object.entries(translations)) {
                        if (template === message) return { key, variables: {} };
                    }
                }
                return null;
            }

            function localizeServerMessage(message) {
                if (message && typeof message === "object") {
                    if (message.message === "The selected sample and script do not match. Only punctuation and spacing differences are allowed.") {
                        return t("scriptMismatch", { expected: message.expected_script || "", received: message.received_script || "" });
                    }
                    return JSON.stringify(message);
                }
                const text = String(message || "");
                const messageKeys = {
                    "Invalid audio file format.": "invalidAudio",
                    "The input script must not be empty.": "emptyScript",
                    "Enter a valid speaker profile before practicing.": "invalidProfile",
                    "The selected prototype sample was not found.": "sampleMissing",
                    "The script contains no valid Han characters to score pronunciation.": "noHanScript",
                    "Enter a name.": "emptyName",
                    "Name must contain 1 to 100 characters.": "invalidNameLength",
                    "Choose Taiwanese or Foreigner.": "invalidGroup",
                    "The selected practice attempt was not found.": "attemptMissing",
                    "Rating must be an integer from 1 to 5.": "invalidFeedbackRating",
                    "Feedback comment must be 2,000 characters or fewer.": "feedbackTooLong",
                    "Could not save speaker profile.": "profileSaveFailed",
                    "Could not save feedback.": "feedbackSaveFailed"
                };
                return messageKeys[text] ? t(messageKeys[text]) : text;
            }

            // The interface language of a first visit comes from the browser's own language settings (not from the IP address: a Vietnamese learner living in
            // Taiwan has a Taiwanese IP). The first language in the browser's list that the page has wins; anything else (or an error) is English.
            function detectBrowserLanguage() {
                try {
                    const preferred = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || ""];
                    for (const tag of preferred) {
                        const base = String(tag || "").toLowerCase().split(/[-_;]/)[0];
                        if (base === "vi") return "vi";
                        if (base === "id" || base === "in") return "id";
                        if (base === "hi") return "hi";
                        if (base === "zh") return "zh-TW";
                        if (base === "en") return "en";
                    }
                } catch (error) { /* fall through to English */ }
                return "en";
            }

            function savedLanguage() {
                try {
                    return localStorage.getItem("mandarinUiLanguage") || "";
                } catch (error) {
                    return "";
                }
            }

            function applyLanguage(language, remember = true) {
                currentLanguage = Object.prototype.hasOwnProperty.call(LANGUAGE_TEXT, language) ? language : "en";
                if (remember) {                                   // a language the learner picked is kept; the browser's guess is not stored
                    try { localStorage.setItem("mandarinUiLanguage", currentLanguage); } catch (error) { /* private window */ }
                }
                document.documentElement.lang = currentLanguage;
                document.getElementById("languageSelect").value = currentLanguage;
                document.querySelectorAll("[data-i18n]").forEach(element => {
                    element.textContent = t(element.dataset.i18n);
                });
                if (chapter5Categories.length) populateChapter5CategoryOptions();
                document.querySelectorAll("[data-i18n-placeholder]").forEach(element => {
                    element.placeholder = t(element.dataset.i18nPlaceholder);
                });
                document.querySelectorAll("[data-i18n-aria-label]").forEach(element => {
                    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
                });
                document.querySelectorAll("[data-i18n-title]").forEach(element => {
                    element.setAttribute("title", t(element.dataset.i18nTitle));
                });
                document.title = t("appTitle");
                document.getElementById("recordBtn").innerText = isRecording ? t("recordingStop") : t("startRecording");
                const calibrationStatus = document.getElementById("voiceCalibrationStatus");
                if (calibrationStatus.dataset.translation) {
                    const savedCalibrationStatus = JSON.parse(calibrationStatus.dataset.translation);
                    calibrationStatus.textContent = t(savedCalibrationStatus.key, savedCalibrationStatus.variables || {});
                }
                if (activeStudent) updateActiveStudentLabel();
                syncVoiceCalibrationUI();
                if (sampleLessons.length) refreshSampleOptions();
                syncPracticeModeUI();
                if (selectedPracticeMode() === "chapter5_word" && chapter5WordItems.length) renderChapter5WordItem();
                if (lastAssessment) renderVisualResults(lastAssessment);
                if (lastLeaderboard) renderLeaderboard(lastLeaderboard);
                renderRecent();
                syncResultButtons();
                renderItemProgress();
                const status = document.getElementById("status");
                if (status.dataset.translation) {
                    const savedStatus = JSON.parse(status.dataset.translation);
                    renderStatusContent(status, t(savedStatus.key, savedStatus.variables || {}), savedStatus.key === "processing");
                }
            }

            function changeLanguage(language) {
                applyLanguage(language);
            }

            function updateActiveStudentLabel() {
                const groupLabel = activeStudent.group_type === "taiwanese" ? t("taiwanese") : t("foreigner");
                document.getElementById("activeStudentLabel").textContent = t("practicingAs", { name: activeStudent.name, group: groupLabel });
                syncVoiceCalibrationUI();
            }

            function syncVoiceCalibrationUI() {
                const label = document.getElementById("voiceCalibrationLabel");
                const button = document.getElementById("calibrateVoiceBtn");
                const recordButton = document.getElementById("voiceCalibrationRecordBtn");
                const laterButton = document.getElementById("voiceCalibrationLaterBtn");
                if (!label || !button || !recordButton || !laterButton) return;
                const calibrated = activeStudent?.voice_calibration?.status === "calibrated";
                label.textContent = activeStudent
                    ? t(calibrated ? "voiceCalibrated" : "voiceNotCalibrated")
                    : "";
                button.textContent = t(calibrated ? "recalibrateVoice" : "calibrateVoice");
                const dialog = document.getElementById("voiceCalibrationDialog");
                dialog.dataset.state = isVoiceCalibrationProcessing ? "processing"
                    : isVoiceCalibrationRecording ? "recording" : (calibrationOutcome || "idle");
                recordButton.textContent = isVoiceCalibrationRecording
                    ? t(calibrationCanStop ? "stopCalibration" : "calKeepReading")
                    : t(calibrationOutcome === "error" ? "calTryAgain" : "recordCalibration");
                recordButton.disabled = isVoiceCalibrationProcessing || (isVoiceCalibrationRecording && !calibrationCanStop);
                recordButton.classList.toggle("recording", isVoiceCalibrationRecording);
                recordButton.hidden = calibrationOutcome === "success";
                document.getElementById("voiceCalibrationCloseBtn").disabled = isVoiceCalibrationRecording || isVoiceCalibrationProcessing;
                laterButton.disabled = isVoiceCalibrationRecording || isVoiceCalibrationProcessing;
                laterButton.hidden = !(calibrated || calibrationHadError);      // never offered to skip up front
                laterButton.textContent = t(calibrationOutcome === "success" ? "calibrationDone"
                    : calibrated ? "calClose" : "voiceCalibrationLater");
            }

            // Live microphone feedback shared by the calibration dialog and the practice recorder: a level bar that moves
            // with the voice, and a hint when nothing has been heard for two seconds.
            function startMicMeter(stream, barElement, hintElement) {
                let context = null;
                let frame = 0;
                let stopped = false;
                try {
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                    if (!AudioContextClass || !barElement) return () => {};
                    context = new AudioContextClass();
                    const analyser = context.createAnalyser();
                    analyser.fftSize = 1024;
                    context.createMediaStreamSource(stream).connect(analyser);
                    const samples = new Float32Array(analyser.fftSize);
                    const startedAt = Date.now();
                    let lastHeardAt = startedAt;
                    const tick = () => {
                        if (stopped) return;
                        analyser.getFloatTimeDomainData(samples);
                        let energy = 0;
                        for (let i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
                        const level = Math.sqrt(energy / samples.length);
                        barElement.style.width = `${Math.min(100, Math.round(Math.sqrt(level) * 160))}%`;
                        const now = Date.now();
                        if (level > 0.008) lastHeardAt = now;
                        if (hintElement) hintElement.hidden = !(now - startedAt > 2000 && now - lastHeardAt > 2000);
                        frame = requestAnimationFrame(tick);
                    };
                    tick();
                } catch (error) {
                    return () => {};                               // the meter is a nicety; recording works without it
                }
                return () => {
                    stopped = true;
                    cancelAnimationFrame(frame);
                    try { context.close(); } catch (error) { /* already closed */ }
                    barElement.style.width = "0%";
                    if (hintElement) hintElement.hidden = true;
                };
            }

            function formatClock(milliseconds) {
                const seconds = Math.floor(milliseconds / 1000);
                return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
            }

            function showRecordingIndicator(stream) {
                const clock = document.getElementById("recordingClock");
                const startedAt = Date.now();
                hideRecordingIndicator();
                document.getElementById("recordingIndicator").hidden = false;
                clock.textContent = formatClock(0);
                practiceIndicatorTimer = window.setInterval(() => { clock.textContent = formatClock(Date.now() - startedAt); }, 250);
                stopPracticeMeter = startMicMeter(stream, document.getElementById("recordingLevelBar"),
                    document.getElementById("recordingNoSound"));
            }

            function hideRecordingIndicator() {
                window.clearInterval(practiceIndicatorTimer);
                practiceIndicatorTimer = null;
                if (stopPracticeMeter) stopPracticeMeter();
                stopPracticeMeter = null;
                document.getElementById("recordingIndicator").hidden = true;
            }

            function startCalibrationLive(stream) {
                stopCalibrationLive();
                calibrationCanStop = false;
                const bar = document.getElementById("calibrationTimerBar");
                const text = document.getElementById("calibrationTimerText");
                const update = () => {
                    const elapsed = Date.now() - voiceCalibrationStartedAt;
                    bar.style.width = `${Math.min(100, elapsed / CALIBRATION_MAX_MS * 100)}%`;
                    bar.classList.toggle("enough", elapsed >= CALIBRATION_MIN_MS);
                    text.textContent = t("calSeconds", { seconds: Math.min(10, Math.floor(elapsed / 1000)) });
                    if (elapsed >= CALIBRATION_MIN_MS && !calibrationCanStop) {
                        calibrationCanStop = true;                  // the button turns into "I'm done"
                        syncVoiceCalibrationUI();
                    }
                };
                update();
                calibrationClock = window.setInterval(update, 100);
                stopCalibrationMeter = startMicMeter(stream, document.getElementById("calibrationLevelBar"),
                    document.getElementById("calibrationNoSound"));
            }

            function stopCalibrationLive() {
                window.clearInterval(calibrationClock);
                calibrationClock = null;
                if (stopCalibrationMeter) stopCalibrationMeter();
                stopCalibrationMeter = null;
            }

            // Which tip to show for each way the server can reject a calibration recording.
            function calibrationReasonKey(reason) {
                return {
                    too_short: "calibrationTooShort",
                    clipped: "calReasonClipped",
                    boundary_hits: "calReasonNoisy",
                    noisy: "calReasonNoisy",
                    ac_cc_disagree: "calReasonUnclear",
                    too_little_voicing: "calReasonQuiet",
                }[reason] || "calibrationFailed";
            }

            function refreshSampleOptions() {
                const select = document.getElementById("sampleSelect");
                const selectedId = select.value;
                select.replaceChildren();
                sampleLessons.forEach(sample => {
                    const done = progressData.lessons[sample.id];
                    const label = t("lessonOption", sample);
                    select.add(new Option(done ? `✓ ${done.best}% · ${label}` : label, sample.id));       // a practised sentence: a tick and the best points
                });
                if (sampleLessons.some(sample => sample.id === selectedId)) select.value = selectedId;
            }

            function selectedPracticeMode() {
                return document.querySelector('input[name="practiceMode"]:checked')?.value || "book";
            }

            function hasMandarinCharacters(text) {
                return /\\p{Script=Han}/u.test(String(text || ""));
            }

            async function loadChapter5WordItems() {
                const modeInput = document.querySelector('input[name="practiceMode"][value="chapter5_word"]');
                try {
                    const [itemsResponse, categoriesResponse] = await Promise.all([
                        fetch("/api/chapter5-word-items"),
                        fetch("/api/chapter5-word-categories")
                    ]);
                    if (!itemsResponse.ok || !categoriesResponse.ok) throw new Error(t("chapter5Unavailable"));
                    chapter5WordItems = await itemsResponse.json();
                    chapter5Categories = await categoriesResponse.json();
                    modeInput.disabled = chapter5WordItems.length === 0;
                    if (!chapter5WordItems.length) throw new Error(t("chapter5Unavailable"));
                    chapter5WordIndex = Math.min(chapter5WordIndex, chapter5WordItems.length - 1);
                    populateChapter5CategoryOptions();
                    if (selectedPracticeMode() === "chapter5_word") renderChapter5WordItem();
                } catch (error) {
                    chapter5WordItems = [];
                    modeInput.disabled = true;
                    if (selectedPracticeMode() === "chapter5_word") showTranslatedStatus("chapter5Unavailable", {}, true);
                }
                syncPracticeControls();
            }

            function chapter5Label(labels) {
                const key = currentLanguage === "zh-TW" ? "zh" : currentLanguage;   // the topic data says "zh"
                return labels?.[key] || labels?.en || "";
            }

            function populateChapter5CategoryOptions() {
                const select = document.getElementById("chapter5CategorySelect");
                const selected = chapter5CategoryFilter;
                select.replaceChildren(new Option(t("chapter5AllTopics"), ""));
                chapter5Categories.forEach(category => {
                    const group = document.createElement("optgroup");
                    const categoryLabel = chapter5Label(category.labels);
                    group.label = categoryLabel;
                    group.appendChild(new Option(
                        t("chapter5AllInTopic", { topic: categoryLabel }),
                        `category:${category.id}`
                    ));
                    (category.subcategories || []).forEach(subcategory => {
                        group.appendChild(new Option(
                            `  · ${chapter5Label(subcategory.labels)}`,
                            `subcategory:${subcategory.id}`
                        ));
                    });
                    select.appendChild(group);
                });
                const hasSelectedOption = Array.from(select.options).some(option => option.value === selected);
                chapter5CategoryFilter = hasSelectedOption ? selected : "";
                select.value = chapter5CategoryFilter;
            }

            function filteredChapter5WordIndexes() {
                return chapter5WordItems
                    .map((item, index) => ({ item, index }))
                    .filter(({ item }) => {
                        if (!chapter5CategoryFilter) return true;
                        if (chapter5CategoryFilter.startsWith("category:")) {
                            return item.category_id === chapter5CategoryFilter.slice("category:".length);
                        }
                        if (chapter5CategoryFilter.startsWith("subcategory:")) {
                            return item.subcategory_id === chapter5CategoryFilter.slice("subcategory:".length);
                        }
                        return true;
                    })
                    .map(({ index }) => index);
            }

            function chapter5ItemCategoryLabel(item) {
                const category = chapter5Categories.find(entry => entry.id === item.category_id);
                const subcategory = category?.subcategories?.find(entry => entry.id === item.subcategory_id);
                if (!category || !subcategory) return "";
                return `${chapter5Label(category.labels)} · ${chapter5Label(subcategory.labels)}`;
            }

            function changeChapter5CategoryFilter() {
                const select = document.getElementById("chapter5CategorySelect");
                if (isProcessing || isRecording || isSavingStudent) {
                    select.value = chapter5CategoryFilter;
                    return;
                }
                chapter5CategoryFilter = select.value;
                const indexes = filteredChapter5WordIndexes();
                chapter5WordIndex = indexes.length ? indexes[0] : 0;
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";
                showFeedbackPanel(null);
                renderChapter5WordItem();
                showTranslatedStatus("chapter5Ready");
            }

            function renderChapter5WordItem() {
                const visibleIndexes = filteredChapter5WordIndexes();
                if (!visibleIndexes.length) return;
                let visiblePosition = visibleIndexes.indexOf(chapter5WordIndex);
                if (visiblePosition < 0) {
                    chapter5WordIndex = visibleIndexes[0];
                    visiblePosition = 0;
                }
                const item = chapter5WordItems[chapter5WordIndex];
                if (!item) return;
                document.getElementById("scriptInput").value = item.script;
                const cells = Array.isArray(item.display_cells) && item.display_cells.length ? item.display_cells : null;
                renderTranscript(cells ? { cells } : { script: item.script, pinyin: item.character_pinyin || [] });
                const optionalHint = document.getElementById("chapter5OptionalHint");
                optionalHint.hidden = !cells;           // the recording says 裡面; the learner may say 裡 or 裡面
                if (cells) {
                    optionalHint.textContent = t("chapter5OptionalHint", {
                        short: cells.filter(cell => !cell.optional).map(cell => cell.char).join(""),
                        full: cells.map(cell => cell.char).join("")
                    });
                }
                const audio = document.getElementById("sampleAudio");
                audio.pause();
                document.getElementById("sampleAudioError").hidden = true;
                audio.src = item.audio_url;
                audio.load();
                document.getElementById("chapter5WordType").textContent = t(String(item.item_type).toLowerCase() === "phrase" ? "chapter5Phrase" : "chapter5Vocabulary");
                document.getElementById("chapter5WordMeaning").textContent = t("chapter5Meaning", { meaning: item.english_meaning || item.pos || "" });
                document.getElementById("chapter5WordCategory").textContent = chapter5ItemCategoryLabel(item);
                document.getElementById("chapter5WordProgress").textContent = t("chapter5Progress", { current: visiblePosition + 1, total: visibleIndexes.length });
                document.getElementById("previousChapter5WordBtn").disabled = visiblePosition <= 0 || isProcessing || isRecording || isSavingStudent;
                document.getElementById("nextChapter5WordBtn").disabled = visiblePosition >= visibleIndexes.length - 1 || isProcessing || isRecording || isSavingStudent;
                syncPracticeModeUI();
                syncPracticeControls();
            }

            function moveChapter5Word(direction) {
                if (isProcessing || isRecording || isSavingStudent || !chapter5WordItems.length) return;
                const visibleIndexes = filteredChapter5WordIndexes();
                const visiblePosition = visibleIndexes.indexOf(chapter5WordIndex);
                const nextIndex = visibleIndexes[visiblePosition + direction];
                if (nextIndex === undefined) return;
                chapter5WordIndex = nextIndex;
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";
                showFeedbackPanel(null);
                renderChapter5WordItem();
                showTranslatedStatus("chapter5Ready");
            }

            function syncPracticeModeUI() {
                const mode = selectedPracticeMode();
                const isCustom = mode === "custom";
                const isChapter5 = mode === "chapter5_word";
                const scriptInput = document.getElementById("scriptInput");
                const sampleAudio = document.getElementById("sampleAudio");
                document.getElementById("bookSamplePicker").hidden = isCustom || isChapter5;
                document.getElementById("chapter5CategoryPicker").hidden = !isChapter5;
                document.getElementById("chapter5WordControls").hidden = !isChapter5;
                document.getElementById("customTextHint").hidden = !isCustom;
                document.getElementById("customSampleBlock").hidden = !isCustom;
                if (!isCustom) document.getElementById("customSampleAudio").pause();
                document.getElementById("pinyinToggle").hidden = false;
                document.getElementById("chapter5ReferenceLabel").hidden = !isChapter5;
                if (!isChapter5) document.getElementById("chapter5OptionalHint").hidden = true;
                document.getElementById("sampleAudioSection").hidden = isCustom || !sampleAudio.hasAttribute("src");
                document.getElementById("scriptLabel").textContent = t(isCustom ? "customScriptLabel" : "matchingScript");
                document.getElementById("transcriptDisplay").setAttribute("aria-label", t(isCustom ? "customTextPreview" : "scriptWithPinyin"));
                scriptInput.readOnly = !isCustom;
                sampleAudio.hidden = isCustom || !sampleAudio.hasAttribute("src");
                if (isCustom) sampleAudio.pause();
            }

            function changePracticeMode() {
                if (isProcessing || isRecording || isSavingStudent) return;
                const scriptInput = document.getElementById("scriptInput");
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";
                showFeedbackPanel(null);
                if (selectedPracticeMode() === "custom") {
                    scriptInput.value = customTextDraft;
                    syncPracticeModeUI();
                    renderCustomTranscript(scriptInput.value);
                    requestCustomPinyin(scriptInput.value);
                    showTranslatedStatus(hasMandarinCharacters(scriptInput.value) ? "customTextReady" : "customTextHint");
                } else if (selectedPracticeMode() === "chapter5_word") {
                    syncPracticeModeUI();
                    if (chapter5WordItems.length) {
                        renderChapter5WordItem();
                        showTranslatedStatus("chapter5Ready");
                    } else {
                        showTranslatedStatus("chapter5Unavailable", {}, true);
                    }
                } else {
                    syncPracticeModeUI();
                    selectSample();
                }
                syncPracticeControls();
            }

            function renderCustomTranscript(script) {
                renderTranscript({ script, pinyin: customPinyin.script === script ? customPinyin.values : [] });
            }

            // The pinyin of what the learner types comes from the server (the same readings the scoring uses).
            function requestCustomPinyin(script) {
                clearTimeout(customPinyinTimer);
                if (!hasMandarinCharacters(script)) return;
                customPinyinTimer = setTimeout(async () => {
                    try {
                        const response = await fetch(`/api/pinyin?text=${encodeURIComponent([...script].slice(0, 200).join(""))}`);
                        if (!response.ok) return;
                        const data = await response.json();
                        customPinyin = { script, values: Array.isArray(data.pinyin) ? data.pinyin : [] };
                        if (selectedPracticeMode() === "custom" && document.getElementById("scriptInput").value === script) {
                            renderCustomTranscript(script);
                        }
                    } catch (error) { /* the preview simply stays without pinyin */ }
                }, 250);
            }

            // "My own text": a computer-voice sample of the typed text (the server's Taiwan Mandarin voice), made when the learner asks for it.
            let customSample = { script: "", url: "" };

            function hideStaleCustomSample(script) {
                if (customSample.script !== script) {
                    const audio = document.getElementById("customSampleAudio");
                    audio.pause();
                    audio.hidden = true;
                }
                document.getElementById("customSampleError").hidden = true;
            }

            async function playCustomSample() {
                const script = document.getElementById("scriptInput").value;
                const button = document.getElementById("customSampleBtn");
                const audio = document.getElementById("customSampleAudio");
                const error = document.getElementById("customSampleError");
                error.hidden = true;
                if (!hasMandarinCharacters(script)) {
                    showTranslatedStatus("customTextHint");
                    return;
                }
                if (customSample.script === script && customSample.url) {
                    audio.hidden = false;
                    audio.currentTime = 0;
                    audio.play().catch(() => {});
                    return;
                }
                button.disabled = true;
                button.textContent = t("customSampleBusy");
                try {
                    const response = await fetch(`/api/tts-sample?text=${encodeURIComponent([...script].slice(0, 200).join(""))}`);
                    if (!response.ok) throw new Error(String(response.status));
                    const blob = await response.blob();
                    if (document.getElementById("scriptInput").value !== script) return;          // the text changed while it was being made
                    if (customSample.url) URL.revokeObjectURL(customSample.url);
                    customSample = { script, url: URL.createObjectURL(blob) };
                    audio.src = customSample.url;
                    audio.hidden = false;
                    audio.play().catch(() => {});
                } catch (failure) {
                    const key = failure.message === "503" ? "customSampleUnavailable" : "customSampleError";
                    error.dataset.i18n = key;
                    error.textContent = t(key);
                    error.hidden = false;
                } finally {
                    button.disabled = false;
                    button.textContent = t("customSampleListen");
                }
            }

            // The leaderboard: two boards, "points" (the best attempt of every sentence added up) and "accuracy" (their average out of 100, from 5 sentences on).
            // The server works both out from every saved attempt; the tab the learner looks at is remembered in this browser.
            let lastLeaderboard = null;
            let boardView = "points";
            try { boardView = localStorage.getItem("leaderboardView") === "accuracy" ? "accuracy" : "points"; } catch (error) { /* private window */ }

            function setBoardView(view) {
                boardView = view === "accuracy" ? "accuracy" : "points";
                try { localStorage.setItem("leaderboardView", boardView); } catch (error) { /* private window */ }
                renderLeaderboard(lastLeaderboard);
            }

            function renderLeaderboard(data) {
                lastLeaderboard = data;
                const byAccuracy = boardView === "accuracy";
                for (const [id, active] of [["boardTabPoints", !byAccuracy], ["boardTabAccuracy", byAccuracy]]) {
                    document.getElementById(id).classList.toggle("active", active);
                    document.getElementById(id).setAttribute("aria-selected", String(active));
                }
                const board = byAccuracy ? ((data && data.accuracy) || {}) : (data || {});
                const minLessons = (data && data.accuracy && data.accuracy.min_lessons) || 5;
                document.getElementById("leaderboardHelp").textContent = byAccuracy ? t("leaderboardHelpAccuracy", { n: minLessons }) : t("leaderboardHelp");
                const list = document.getElementById("leaderboardList");
                list.replaceChildren();
                const top = board.top || [];
                const empty = document.getElementById("leaderboardEmpty");
                empty.textContent = byAccuracy ? t("leaderboardEmptyAccuracy", { n: minLessons }) : t("leaderboardEmpty");
                empty.hidden = top.length > 0;
                top.forEach(row => {
                    const item = document.createElement("li");
                    item.className = "leaderboard-row" + (row.is_me ? " me" : "");
                    const rank = document.createElement("span");
                    rank.className = "leaderboard-rank";
                    rank.textContent = String(row.rank);
                    const name = document.createElement("span");
                    name.className = "leaderboard-name";
                    name.textContent = row.name;
                    name.title = row.name;
                    const points = document.createElement("span");
                    points.className = "leaderboard-points";
                    points.textContent = byAccuracy ? t("leaderboardAccuracy", { n: row.accuracy.toFixed(1) }) : t("leaderboardPoints", { n: row.points });
                    const lessons = document.createElement("span");
                    lessons.className = "leaderboard-lessons";
                    lessons.textContent = t("leaderboardLessons", { n: row.lessons });
                    points.appendChild(lessons);
                    item.append(rank, name, points);
                    list.appendChild(item);
                });
                const me = document.getElementById("leaderboardMe");
                const mine = board.me;
                const inTop = top.some(row => row.is_me);
                if (!activeStudent) {
                    me.hidden = true;
                } else if (mine && mine.rank === null) {
                    me.textContent = t("leaderboardMeAccuracyNeed", { n: mine.needed });
                    me.hidden = false;
                } else if (mine && !inTop) {
                    me.textContent = byAccuracy ? t("leaderboardMeAccuracyRank", { rank: mine.rank, accuracy: mine.accuracy.toFixed(1) })
                                                : t("leaderboardMeRank", { rank: mine.rank, points: mine.points });
                    me.hidden = false;
                } else if (!mine) {
                    me.textContent = t("leaderboardMeNone");
                    me.hidden = false;
                } else {
                    me.hidden = true;
                }
            }

            async function loadLeaderboard() {
                try {
                    const query = activeStudent ? `?student_id=${encodeURIComponent(activeStudent.id)}` : "";
                    const response = await fetch(`/api/leaderboard${query}`);
                    if (response.ok) renderLeaderboard(await response.json());
                } catch (error) { /* the board simply stays as it was */ }
            }

            // What this learner has practised: a tick and the best points in the sentence list, a line under the script, and their latest results.
            async function loadProgress() {
                if (!activeStudent) {
                    progressData = { lessons: {}, recent: [] };
                } else {
                    try {
                        const response = await fetch(`/api/progress?student_id=${encodeURIComponent(activeStudent.id)}`);
                        if (response.ok) progressData = await response.json();
                    } catch (error) { /* the marks simply stay as they were */ }
                }
                if (sampleLessons.length && !isRecording) refreshSampleOptions();
                renderItemProgress();
                renderRecent();
            }

            function renderItemProgress() {
                const line = document.getElementById("itemProgress");
                const id = selectedPracticeMode() === "custom" ? "" : selectedPracticeSampleId();
                if (!activeStudent || !id) {
                    line.hidden = true;
                    return;
                }
                const done = progressData.lessons[id];
                line.textContent = done ? t("itemProgress", { n: done.attempts, best: done.best, last: done.last }) : t("itemNotPracticed");
                line.classList.toggle("practiced", Boolean(done));
                line.hidden = false;
            }

            function relativeTime(iso) {
                const then = Date.parse(iso);
                if (!Number.isFinite(then)) return "";
                const seconds = Math.round((then - Date.now()) / 1000);
                try {
                    const formatter = new Intl.RelativeTimeFormat(currentLanguage, { numeric: "auto" });
                    for (const [unit, size] of [["day", 86400], ["hour", 3600], ["minute", 60]]) {
                        if (Math.abs(seconds) >= size) return formatter.format(Math.round(seconds / size), unit);
                    }
                    return formatter.format(Math.min(seconds, 0), "second");
                } catch (error) {
                    return new Date(then).toLocaleString();
                }
            }

            function renderRecent() {
                const card = document.getElementById("recentCard");
                card.hidden = !activeStudent;
                if (!activeStudent) return;
                const list = document.getElementById("recentList");
                list.replaceChildren();
                const recent = progressData.recent || [];
                document.getElementById("recentEmpty").hidden = recent.length > 0;
                recent.forEach(entry => {
                    const item = document.createElement("li");
                    const row = document.createElement("button");
                    row.type = "button";
                    row.className = "recent-row";
                    const script = document.createElement("span");
                    script.className = "recent-script";
                    const text = String(entry.script || "");
                    script.textContent = text.length > 18 ? `${text.slice(0, 18)}…` : text;
                    script.title = text;
                    const points = document.createElement("span");
                    points.className = "recent-points";
                    points.textContent = entry.points === null ? "–" : t("recentPoints", { n: entry.points });
                    const when = document.createElement("span");
                    when.className = "recent-time";
                    when.textContent = relativeTime(entry.created_at);
                    points.appendChild(when);
                    row.append(script, points);
                    row.disabled = entry.mode === "custom";                  // own text is shown, but there is no list entry to go back to
                    row.addEventListener("click", () => openRecent(entry));
                    item.appendChild(row);
                    list.appendChild(item);
                });
            }

            // Back to a sentence or word from the latest results.
            function openRecent(entry) {
                if (isProcessing || isRecording || isSavingStudent || entry.mode === "custom") return;
                if (entry.mode === "book") {
                    if (!sampleLessons.some(sample => sample.id === entry.sample_id)) return;
                    document.querySelector('input[name="practiceMode"][value="book"]').checked = true;
                    document.getElementById("sampleSelect").value = entry.sample_id;
                    sampleTouched = true;
                } else {
                    const index = chapter5WordItems.findIndex(item => item.word_key === entry.sample_id);
                    if (index < 0) return;
                    document.querySelector('input[name="practiceMode"][value="chapter5_word"]').checked = true;
                    chapter5CategoryFilter = "";
                    document.getElementById("chapter5CategorySelect").value = "";
                    chapter5WordIndex = index;
                }
                changePracticeMode();
                scrollToPractice();
            }

            function scrollToPractice() {
                document.getElementById("practiceMode").scrollIntoView({ behavior: "smooth", block: "start" });
            }

            // "Read again" is the record / stop button of the top of the page, "Next" the next sentence (or word) of the list: both sit under the results,
            // so the learner neither re-selects nor scrolls back up.
            function hasNextItem() {
                const mode = selectedPracticeMode();
                if (mode === "book") {
                    const select = document.getElementById("sampleSelect");
                    return select.selectedIndex >= 0 && select.selectedIndex < select.options.length - 1 && sampleLessons.length > 0;
                }
                if (mode === "chapter5_word") {
                    const visible = filteredChapter5WordIndexes();
                    const position = visible.indexOf(chapter5WordIndex);
                    return position >= 0 && position < visible.length - 1;
                }
                return false;
            }

            function syncResultButtons() {
                const busy = isProcessing || isRecording || isSavingStudent;
                const recordDisabled = document.getElementById("recordBtn").disabled;
                document.querySelectorAll(".retry-btn").forEach(button => {
                    button.textContent = isRecording ? t("recordingStop") : t("resultRetry");
                    button.classList.toggle("recording", isRecording);
                    button.disabled = recordDisabled;
                });
                const next = hasNextItem();
                document.querySelectorAll(".next-item-btn").forEach(button => {
                    button.hidden = selectedPracticeMode() === "custom";
                    button.disabled = busy || !next;
                });
            }

            function goToNextItem() {
                if (isProcessing || isRecording || isSavingStudent || !hasNextItem()) return;
                if (selectedPracticeMode() === "chapter5_word") {
                    moveChapter5Word(1);
                } else {
                    const select = document.getElementById("sampleSelect");
                    select.selectedIndex += 1;
                    sampleTouched = true;
                    selectSample();
                }
                scrollToPractice();
            }

            function handleScriptInput() {
                if (selectedPracticeMode() !== "custom") return;
                const script = document.getElementById("scriptInput").value;
                customTextDraft = script;
                hideStaleCustomSample(script);
                renderCustomTranscript(script);
                requestCustomPinyin(script);
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";
                showFeedbackPanel(null);
                if (script.trim() && !hasMandarinCharacters(script)) {
                    showTranslatedStatus("customTextRequiresHanzi");
                } else {
                    showTranslatedStatus(hasMandarinCharacters(script) ? "customTextReady" : "customTextHint");
                }
                syncPracticeControls();
            }

            applyLanguage(savedLanguage() || detectBrowserLanguage(), Boolean(savedLanguage()));
            loadLeaderboard();

            // A full-screen "working" layer for the slow steps (assessment, calibration): a small spinner at the bottom of
            // the page goes unnoticed. It appears only if the step lasts more than a moment, and cannot be dismissed.
            function setBusy(isBusy, message = "") {
                const overlay = document.getElementById("busyOverlay");
                if (!overlay) return;
                if (isBusy) {
                    document.getElementById("busyMessage").textContent = message;
                    if (overlay.open || busyTimer) return;
                    busyTimer = setTimeout(() => {
                        busyTimer = null;
                        if (!overlay.open && typeof overlay.showModal === "function") overlay.showModal();
                    }, 200);
                } else {
                    clearTimeout(busyTimer);
                    busyTimer = null;
                    if (overlay.open) overlay.close();
                }
            }
            document.getElementById("busyOverlay")?.addEventListener("cancel", event => event.preventDefault());

            function showStatus(message, isError = false, translation = null) {
                const status = document.getElementById("status");
                translation = translation || findTranslationDescriptor(message);
                if (translation) status.dataset.translation = JSON.stringify(translation);
                else delete status.dataset.translation;
                renderStatusContent(status, message, translation?.key === "processing");
                status.classList.toggle("error", isError);
            }

            function renderStatusContent(status, message, isLoading = false) {
                status.classList.toggle("is-loading", isLoading);
                setBusy(isLoading, message);
                status.replaceChildren();
                if (!isLoading) {
                    status.textContent = message;
                    return;
                }
                const spinner = document.createElement("span");
                spinner.className = "loading-spinner";
                spinner.setAttribute("aria-hidden", "true");
                const accessibleMessage = document.createElement("span");
                accessibleMessage.className = "sr-only";
                accessibleMessage.textContent = message;
                status.append(spinner, accessibleMessage);
            }

            function showTranslatedStatus(key, variables = {}, isError = false) {
                showStatus(t(key, variables), isError, { key, variables });
            }

            function showFeedbackPanel(attemptId) {
                activeFeedbackAttemptId = attemptId || null;
                const panel = document.getElementById("feedbackPanel");
                const form = document.getElementById("feedbackForm");
                form.reset();
                document.getElementById("feedbackSubmitBtn").disabled = false;
                form.hidden = false;
                document.getElementById("feedbackStatus").textContent = "";
                panel.hidden = !activeFeedbackAttemptId;
            }

            async function submitUserFeedback(event) {
                event.preventDefault();
                if (!activeStudent || !activeFeedbackAttemptId) return;
                const form = document.getElementById("feedbackForm");
                const submitButton = document.getElementById("feedbackSubmitBtn");
                const status = document.getElementById("feedbackStatus");
                const formData = new FormData(form);
                submitButton.disabled = true;
                status.textContent = t("sendingFeedback");
                try {
                    const response = await fetch("/api/feedback", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            student_id: activeStudent.id,
                            attempt_id: activeFeedbackAttemptId,
                            rating: Number(formData.get("feedbackRating")),
                            comment: document.getElementById("feedbackComment").value
                        })
                    });
                    const data = await response.json();
                    if (!response.ok) throw new Error(localizeServerMessage(data.detail) || t("feedbackSaveFailed"));
                    form.hidden = true;
                    status.textContent = t("feedbackSaved");
                } catch (error) {
                    status.textContent = localizeServerMessage(error.message) || t("feedbackSaveFailed");
                    submitButton.disabled = false;
                }
            }

            let calibrationPlayer = null;
            let calibrationPlayToken = 0;

            function stopCalibrationSound() {
                calibrationPlayToken += 1;
                if (calibrationPlayer) calibrationPlayer.pause();
                document.querySelectorAll(".cal-play").forEach(button => { button.dataset.playing = "false"; });
            }

            function playOneCalibrationSound(tone, token) {
                return new Promise(resolve => {
                    const button = document.querySelector(`.cal-play[data-tone="${tone}"]`);
                    const audio = new Audio(`/api/calibration-sound/${tone}`);
                    calibrationPlayer = audio;
                    if (button) button.dataset.playing = "true";
                    const done = () => { if (button) button.dataset.playing = "false"; resolve(token === calibrationPlayToken); };
                    audio.addEventListener("ended", done);
                    audio.addEventListener("error", done);
                    audio.play().catch(done);
                });
            }

            async function playCalibrationSound(tone) {
                stopCalibrationSound();
                const token = calibrationPlayToken;
                const tones = tone === 0 ? [1, 2, 3, 4, 5] : [tone];
                for (const one of tones) {
                    const stillCurrent = await playOneCalibrationSound(one, token);
                    if (!stillCurrent) return;
                    if (tones.length > 1) await new Promise(resolve => setTimeout(resolve, 450));
                    if (token !== calibrationPlayToken) return;
                }
            }

            function setActiveStudent(profile) {
                activeStudent = profile;
                document.getElementById("activeStudentPanel").hidden = !profile;
                if (profile) {
                    updateActiveStudentLabel();
                    localStorage.setItem("mandarinPracticeStudent", JSON.stringify(profile));
                    document.getElementById("speakerDialog").close();
                } else {
                    localStorage.removeItem("mandarinPracticeStudent");
                }
                syncVoiceCalibrationUI();
                syncPracticeControls();
                loadLeaderboard();                                       // "You" is a different person now
                loadProgress();
                if (profile && !sampleTouched && sampleLessons.length && selectedPracticeMode() === "book") applyStartSample();
            }

            function calibrationDismissKey(studentId) {
                return `mandarinVoiceCalibrationLater:${studentId}`;
            }

            function maybePromptVoiceCalibration(profile) {
                if (!profile || profile.voice_calibration?.status === "calibrated") return;
                if (sessionStorage.getItem(calibrationDismissKey(profile.id)) === "1") return;
                window.setTimeout(() => openVoiceCalibrationDialog(false), 0);
            }

            function openVoiceCalibrationDialog(manual = true) {
                if (!activeStudent || isVoiceCalibrationProcessing) return;
                if (!manual && (
                    activeStudent.voice_calibration?.status === "calibrated"
                    || sessionStorage.getItem(calibrationDismissKey(activeStudent.id)) === "1"
                )) return;
                const dialog = document.getElementById("voiceCalibrationDialog");
                const status = document.getElementById("voiceCalibrationStatus");
                status.textContent = "";
                status.dataset.translation = "";
                status.classList.remove("error", "success");
                calibrationOutcome = "";
                calibrationCanStop = false;
                if (!dialog.open) dialog.showModal();
                syncVoiceCalibrationUI();
                document.getElementById("voiceCalibrationRecordBtn").focus();
            }

            function deferVoiceCalibration() {
                if (isVoiceCalibrationRecording || isVoiceCalibrationProcessing) return;
                if (activeStudent && activeStudent.voice_calibration?.status !== "calibrated") {
                    sessionStorage.setItem(calibrationDismissKey(activeStudent.id), "1");
                }
                document.getElementById("voiceCalibrationDialog").close();
            }

            function setVoiceCalibrationStatus(key, isError = false, variables = {}) {
                const status = document.getElementById("voiceCalibrationStatus");
                status.textContent = t(key, variables);
                status.dataset.translation = JSON.stringify({ key, variables });
                status.classList.toggle("error", isError);
                const saved = key === "calibrationSaved" || key === "calibrationSavedSlower";      // the "read slower" tip is a success too
                status.classList.toggle("success", saved);
                if (isError) calibrationOutcome = "error";
                else if (saved) calibrationOutcome = "success";
                else calibrationOutcome = "";
                setBusy(key === "calibrationProcessing", status.textContent);
                if (isError && !calibrationHadError) {          // a failure must not trap the learner in the dialog
                    calibrationHadError = true;
                    syncVoiceCalibrationUI();
                }
            }

            async function toggleVoiceCalibrationRecording() {
                if (isVoiceCalibrationProcessing || !activeStudent) return;
                if (isVoiceCalibrationRecording && voiceCalibrationRecorder) {
                    window.clearTimeout(voiceCalibrationTimer);
                    voiceCalibrationRecorder.stop();
                    return;
                }
                if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
                    setVoiceCalibrationStatus("recordingUnsupported", true);
                    return;
                }
                try {
                    voiceCalibrationStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const preferredTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
                    const mimeType = typeof MediaRecorder.isTypeSupported === "function"
                        ? preferredTypes.find(type => MediaRecorder.isTypeSupported(type)) : "";
                    voiceCalibrationRecorder = mimeType
                        ? new MediaRecorder(voiceCalibrationStream, { mimeType })
                        : new MediaRecorder(voiceCalibrationStream);
                    voiceCalibrationChunks = [];
                    voiceCalibrationStartedAt = Date.now();
                    const recordingMimeType = voiceCalibrationRecorder.mimeType || mimeType || "audio/webm";
                    voiceCalibrationRecorder.ondataavailable = event => {
                        if (event.data.size) voiceCalibrationChunks.push(event.data);
                    };
                    voiceCalibrationRecorder.onstop = async () => {
                        window.clearTimeout(voiceCalibrationTimer);
                        stopCalibrationLive();
                        if (voiceCalibrationStream) voiceCalibrationStream.getTracks().forEach(track => track.stop());
                        voiceCalibrationStream = null;
                        isVoiceCalibrationRecording = false;
                        voiceCalibrationRecorder = null;
                        const durationMs = Date.now() - voiceCalibrationStartedAt;
                        const audioBlob = new Blob(voiceCalibrationChunks, { type: recordingMimeType });
                        voiceCalibrationChunks = [];
                        if (durationMs < 4700) {
                            setVoiceCalibrationStatus("calibrationTooShort", true);
                            syncVoiceCalibrationUI();
                            return;
                        }
                        await submitVoiceCalibration(audioBlob, recordingMimeType);
                    };
                    voiceCalibrationRecorder.start();
                    isVoiceCalibrationRecording = true;
                    setVoiceCalibrationStatus("calibrationRecording");
                    startCalibrationLive(voiceCalibrationStream);
                    syncVoiceCalibrationUI();
                    voiceCalibrationTimer = window.setTimeout(() => {
                        if (isVoiceCalibrationRecording && voiceCalibrationRecorder?.state === "recording") {
                            voiceCalibrationRecorder.stop();
                        }
                    }, CALIBRATION_MAX_MS);
                } catch (error) {
                    if (voiceCalibrationStream) voiceCalibrationStream.getTracks().forEach(track => track.stop());
                    voiceCalibrationStream = null;
                    isVoiceCalibrationRecording = false;
                    stopCalibrationLive();
                    setVoiceCalibrationStatus("microphoneFailed", true, { error: error.message || error });
                    syncVoiceCalibrationUI();
                }
            }

            async function submitVoiceCalibration(audioBlob, mimeType) {
                stopCalibrationSound();
                isVoiceCalibrationProcessing = true;
                setVoiceCalibrationStatus("calibrationProcessing");
                syncVoiceCalibrationUI();
                try {
                    const extension = mimeType.includes("mp4") ? "m4a" : mimeType.includes("ogg") ? "ogg" : "webm";
                    const formData = new FormData();
                    formData.append("file", audioBlob, `voice-calibration.${extension}`);
                    const response = await fetch(`/api/students/${encodeURIComponent(activeStudent.id)}/voice-calibration`, {
                        method: "POST", body: formData
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        const failure = new Error(typeof data.detail === "string" ? data.detail : "");
                        failure.reason = data.reason || "";
                        throw failure;
                    }
                    activeStudent.voice_calibration = data.voice_calibration;
                    calibrationPaceHint = data.pace_hint || "";
                    localStorage.setItem("mandarinPracticeStudent", JSON.stringify(activeStudent));
                    updateActiveStudentLabel();
                    sessionStorage.removeItem(calibrationDismissKey(activeStudent.id));
                    setVoiceCalibrationStatus(calibrationPaceHint ? "calibrationSavedSlower" : "calibrationSaved");
                } catch (error) {
                    setVoiceCalibrationStatus(calibrationReasonKey(error.reason), true);
                } finally {
                    isVoiceCalibrationProcessing = false;
                    setBusy(false);
                    syncVoiceCalibrationUI();
                }
            }

            async function saveStudentProfile(event) {
                event.preventDefault();
                if (isSavingStudent || isProcessing || isRecording) return;
                document.getElementById("speakerFormError").textContent = "";
                isSavingStudent = true;
                syncPracticeControls();
                try {
                    const response = await fetch("/api/students", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            name: document.getElementById("studentName").value,
                            group_type: document.getElementById("studentGroup").value
                        })
                    });
                    const data = await response.json();
                    if (!response.ok) throw new Error(localizeServerMessage(data.detail) || t("profileSaveFailed"));
                    setActiveStudent(data);
                    showTranslatedStatus("profileReady", { name: data.name });
                    maybePromptVoiceCalibration(data);
                } catch (error) {
                    document.getElementById("speakerFormError").textContent = localizeServerMessage(error.message) || t("profileSaveFailed");
                    showStatus(localizeServerMessage(error.message) || t("profileSaveFailed"), true);
                } finally {
                    isSavingStudent = false;
                    syncPracticeControls();
                }
            }

            function changeStudentProfile() {
                if (isProcessing || isRecording || isSavingStudent) return;
                openStudentDialog(activeStudent);
            }

            function openStudentDialog(profile = null) {
                const dialog = document.getElementById("speakerDialog");
                const form = document.getElementById("studentForm");
                form.reset();
                document.getElementById("studentName").value = profile?.name || "";
                document.getElementById("studentGroup").value = profile?.group_type || "";
                document.getElementById("speakerFormError").textContent = "";
                document.getElementById("cancelProfileBtn").hidden = !activeStudent;
                if (!dialog.open) dialog.showModal();
                document.getElementById("studentName").focus();
            }

            function closeStudentDialog() {
                if (activeStudent) document.getElementById("speakerDialog").close();
            }

            document.getElementById("speakerDialog").addEventListener("cancel", event => {
                if (!activeStudent) event.preventDefault();
            });

            document.getElementById("voiceCalibrationDialog").addEventListener("cancel", event => {
                event.preventDefault();
                deferVoiceCalibration();
            });

            // The voice setup opens by itself for a new speaker and covers the page, so there must be an obvious way out:
            // the corner button, Esc, or a click on the dimmed area around the dialog.
            document.getElementById("voiceCalibrationDialog").addEventListener("click", event => {
                const box = event.currentTarget.getBoundingClientRect();
                const outside = event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom;
                if (outside) deferVoiceCalibration();
            });

            async function restoreStudentProfile() {
                try {
                    const stored = JSON.parse(localStorage.getItem("mandarinPracticeStudent") || "null");
                    if (!stored || !stored.id) {
                        openStudentDialog();
                        return;
                    }
                    const response = await fetch(`/api/students/${encodeURIComponent(stored.id)}`);
                    if (!response.ok) {
                        localStorage.removeItem("mandarinPracticeStudent");
                        openStudentDialog();
                        return;
                    }
                    const profile = await response.json();
                    setActiveStudent(profile);
                    maybePromptVoiceCalibration(profile);
                } catch (error) {
                    localStorage.removeItem("mandarinPracticeStudent");
                    openStudentDialog();
                }
            }

            function audioFileError(file) {
                if (!file) return t("chooseAudio");
                const formats = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"];
                if (!formats.some(extension => file.name.toLowerCase().endsWith(extension))) {
                    return t("unsupportedAudio");
                }
                if (file.size === 0) return t("emptyAudioFile");
                return "";
            }

            function renderTranscript(sample) {
                const transcript = document.getElementById("transcriptDisplay");
                transcript.replaceChildren();
                const pinyin = Array.isArray(sample.pinyin) ? sample.pinyin : [];
                // `cells` (character, pinyin, optional) is used when part of the word is optional, e.g. 裡(面)
                const cells = Array.isArray(sample.cells) ? sample.cells
                    : [...sample.script].map((character, index) => ({ char: character, pinyin: pinyin[index], optional: false }));
                cells.forEach(cell => {
                    const characterElement = document.createElement("span");
                    characterElement.className = cell.optional ? "transcript-char optional" : "transcript-char";
                    const hanziElement = document.createElement("span");
                    hanziElement.textContent = cell.char;
                    characterElement.appendChild(hanziElement);
                    if (document.getElementById("showPinyin").checked && cell.pinyin) {
                        const pinyinElement = document.createElement("span");
                        pinyinElement.className = "transcript-pinyin";
                        pinyinElement.textContent = cell.pinyin;
                        characterElement.appendChild(pinyinElement);
                    }
                    transcript.appendChild(characterElement);
                });
            }

            function refreshPinyinDisplay() {
                const mode = selectedPracticeMode();
                if (mode === "custom") {
                    renderCustomTranscript(document.getElementById("scriptInput").value);
                    return;
                }
                if (mode === "chapter5_word") {
                    renderChapter5WordItem();
                    refreshPitchPlots();
                    return;
                }
                const sample = sampleLessons.find(item => item.id === document.getElementById("sampleSelect").value);
                if (sample) renderTranscript(sample);
                refreshPitchPlots();
            }

            function syncPracticeControls() {
                const busy = isProcessing || isRecording || isSavingStudent;
                const canPractice = Boolean(activeStudent);
                const mode = selectedPracticeMode();
                const isCustom = mode === "custom";
                const isChapter5 = mode === "chapter5_word";
                const script = document.getElementById("scriptInput").value.trim();
                const visibleChapter5WordIndexes = isChapter5 ? filteredChapter5WordIndexes() : [];
                const visibleChapter5Position = isChapter5
                    ? visibleChapter5WordIndexes.indexOf(chapter5WordIndex)
                    : chapter5WordIndex;
                const hasSelection = isCustom || (isChapter5
                    ? Boolean(chapter5WordItems[chapter5WordIndex])
                    : Boolean(document.getElementById("sampleSelect").value));
                const hasScript = hasMandarinCharacters(script) && hasSelection;
                const fileInput = document.getElementById("audioFileInput");
                document.getElementById("practiceMode").disabled = busy;
                document.getElementById("saveProfileBtn").disabled = busy;
                document.getElementById("changeProfileBtn").disabled = busy;
                document.getElementById("sampleSelect").disabled = busy || !canPractice || isCustom || isChapter5 || sampleLessons.length === 0;
                document.getElementById("chapter5CategorySelect").disabled = busy || !isChapter5;
                document.getElementById("scriptInput").disabled = busy;
                document.getElementById("recordBtn").disabled = isProcessing || !canPractice || !hasScript;
                document.getElementById("stopBtn").disabled = !isRecording || isProcessing;
                document.getElementById("previousChapter5WordBtn").disabled = busy || visibleChapter5Position <= 0;
                document.getElementById("nextChapter5WordBtn").disabled = busy || visibleChapter5Position >= visibleChapter5WordIndexes.length - 1;
                document.getElementById("recordedAudio").controls = !isRecording;
                fileInput.disabled = busy || !canPractice;
                document.getElementById("uploadBtn").disabled = busy || !canPractice || !hasScript || Boolean(audioFileError(fileInput.files[0]));
                syncResultButtons();
                renderItemProgress();
            }

            function selectedPracticeSampleId(mode = selectedPracticeMode()) {
                if (mode === "custom") return "custom";
                if (mode === "chapter5_word") return chapter5WordItems[chapter5WordIndex]?.word_key || "";
                return document.getElementById("sampleSelect").value;
            }

            function previewRecording(audioBlob) {
                if (audioBlob.size === 0) return;
                const preview = document.getElementById("recordedAudio");
                preview.pause();
                if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
                recordedAudioUrl = URL.createObjectURL(audioBlob);
                preview.src = recordedAudioUrl;
                preview.load();
                document.getElementById("recordedAudioContainer").hidden = false;
            }

            function selectAudioFile() {
                const file = document.getElementById("audioFileInput").files[0];
                const preview = document.getElementById("uploadedAudio");
                preview.pause();
                if (uploadedAudioUrl) URL.revokeObjectURL(uploadedAudioUrl);
                uploadedAudioUrl = null;
                preview.removeAttribute("src");
                preview.hidden = true;
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";

                const error = audioFileError(file);
                if (error) {
                    showStatus(error, Boolean(file));
                    syncPracticeControls();
                    return;
                }

                uploadedAudioUrl = URL.createObjectURL(file);
                preview.src = uploadedAudioUrl;
                preview.hidden = false;
                preview.load();
                showTranslatedStatus("uploadReady", { file: file.name });
                syncPracticeControls();
            }

            async function uploadAudio() {
                if (isProcessing || isRecording) return;
                if (!activeStudent) {
                    showTranslatedStatus("needProfile", {}, true);
                    return;
                }
                const file = document.getElementById("audioFileInput").files[0];
                const error = audioFileError(file);
                if (error) {
                    showStatus(error, true);
                    return;
                }
                const script = document.getElementById("scriptInput").value;
                const practiceMode = selectedPracticeMode();
                const sampleId = selectedPracticeSampleId(practiceMode);
                if (!script.trim()) {
                    showTranslatedStatus("emptyScript", {}, true);
                    return;
                }
                if (!hasMandarinCharacters(script)) {
                    showTranslatedStatus("noHanScript", {}, true);
                    return;
                }
                if (practiceMode !== "custom" && !sampleId) {
                    showTranslatedStatus("chooseSample", {}, true);
                    return;
                }
                await submitAssessment(file, file.name, script, sampleId, practiceMode);
            }

            async function submitAssessment(audio, filename, script, sampleId, practiceMode = selectedPracticeMode()) {
                if (!activeStudent) {
                    showTranslatedStatus("needProfile", {}, true);
                    return;
                }
                if (audio.size === 0) {
                    isProcessing = false;
                    syncPracticeControls();
                    showTranslatedStatus("noAudio", {}, true);
                    return;
                }
                isProcessing = true;
                syncPracticeControls();
                document.getElementById("sampleAudio").pause();
                document.getElementById("uploadedAudio").pause();
                document.getElementById("recordedAudio").pause();
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";
                showFeedbackPanel(null);
                showTranslatedStatus("processing");
                const formData = new FormData();
                formData.append("file", audio, filename);
                formData.append("script", script);
                formData.append("sample_id", sampleId);
                formData.append("practice_mode", practiceMode);
                formData.append("student_id", activeStudent.id);
                formData.append("include_transcript", "true");

                try {
                    const response = await fetch("/api/assess-pronunciation", { method: "POST", body: formData });
                    const responseText = await response.text();
                    let data;
                    try {
                        data = JSON.parse(responseText);
                    } catch (error) {
                        throw new Error(t("serverNoResponse", { status: response.status }));
                    }
                    if (!response.ok || !data.success) {
                        const detail = localizeServerMessage(data.detail || "Assessment failed.");
                        throw new Error(detail);
                    }
                    renderVisualResults(data);
                    loadLeaderboard();                                   // the attempt may have moved the learner up
                    loadProgress();                                      // the attempt is now a mark and a recent result
                    showFeedbackPanel(data.saved_attempt?.id);
                    showStatus(data.reference_pitch_warning ? `${t("assessmentSaved")} ${data.reference_pitch_warning}` : t("assessmentSaved"), false, data.reference_pitch_warning ? null : { key: "assessmentSaved", variables: {} });
                } catch (error) {
                    showTranslatedStatus("assessmentFailed", { error: error.message || error }, true);
                } finally {
                    isProcessing = false;
                    setBusy(false);
                    syncPracticeControls();
                }
            }

            window.addEventListener("beforeunload", () => {
                if (uploadedAudioUrl) URL.revokeObjectURL(uploadedAudioUrl);
                if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
            });

            // Where a learner starts: the first sentence of the first story of each chapter (5-1-1, 6-1-1, 7-1-1, 8-1-1 ...), a different one for different people
            // (the same one for the same person: it comes from their profile id; before a profile exists it is random).
            function chapterStartSamples(samples) {
                const first = new Map();
                for (const sample of samples) {
                    const order = Number(sample.story) * 1000 + Number(sample.scene);
                    const kept = first.get(String(sample.lesson));
                    if (!kept || order < kept.order) first.set(String(sample.lesson), { order, sample });
                }
                return [...first.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).map(entry => entry[1].sample);
            }

            function startSampleFor(samples, studentId) {
                const starts = chapterStartSamples(samples);
                if (!starts.length) return null;
                let seed = Math.floor(Math.random() * 4294967296);
                if (studentId) {
                    seed = 2166136261;
                    for (const character of String(studentId)) seed = Math.imul(seed ^ character.charCodeAt(0), 16777619) >>> 0;
                }
                return starts[seed % starts.length];
            }

            function applyStartSample() {
                const start = startSampleFor(sampleLessons, activeStudent && activeStudent.id);
                if (!start) return;
                document.getElementById("sampleSelect").value = start.id;
                selectSample();
            }

            async function loadSampleLessons() {
                const select = document.getElementById("sampleSelect");
                try {
                    const response = await fetch("/api/sample-lessons");
                    if (!response.ok) throw new Error(t("sampleLoadFailed"));
                    sampleLessons = await response.json();
                    select.replaceChildren();

                    if (sampleLessons.length === 0) {
                        select.add(new Option(t("noSampleAudio"), ""));
                        if (selectedPracticeMode() === "book") showTranslatedStatus("sampleDataUnavailable", {}, true);
                        syncPracticeControls();
                        return;
                    }

                    refreshSampleOptions();
                    const start = startSampleFor(sampleLessons, activeStudent && activeStudent.id);
                    if (start) select.value = start.id;

                    selectSample();
                } catch (error) {
                    select.replaceChildren(new Option(t("sampleLoadFailed"), ""));
                    showTranslatedStatus("sampleLoadFailed", {}, true);
                    syncPracticeControls();
                }
            }

            function selectSample() {
                if (selectedPracticeMode() !== "book") return;
                const sample = sampleLessons.find(item => item.id === document.getElementById("sampleSelect").value);
                if (!sample) return;

                document.getElementById("scriptInput").value = sample.script;
                renderTranscript(sample);
                const audio = document.getElementById("sampleAudio");
                audio.pause();
                document.getElementById("sampleAudioError").hidden = true;
                if (sample.audio_url) {
                    audio.src = sample.audio_url;
                } else {
                    audio.removeAttribute("src");
                }
                audio.hidden = !sample.audio_url;
                document.getElementById("sampleAudioSection").hidden = !sample.audio_url;
                audio.load();
                lastAssessment = null;
                document.getElementById("resultCard").style.display = "none";
                showFeedbackPanel(null);
                showTranslatedStatus(sample.audio_url ? "sampleReady" : "sampleAudioUnavailable");
                syncPracticeControls();
            }

            const sampleAudioElement = document.getElementById("sampleAudio");
            sampleAudioElement.addEventListener("error", () => {
                if (sampleAudioElement.hasAttribute("src") && selectedPracticeMode() !== "custom") {
                    document.getElementById("sampleAudioError").hidden = false;
                }
            });
            sampleAudioElement.addEventListener("loadedmetadata", () => {
                document.getElementById("sampleAudioError").hidden = true;
            });

            restoreStudentProfile();
            loadSampleLessons();
            loadChapter5WordItems();

            async function toggleRecording() {
                if (isProcessing) return;
                if (!activeStudent) {
                    showTranslatedStatus("needProfile", {}, true);
                    return;
                }
                if (!isRecording) {
                    const recordingScript = document.getElementById("scriptInput").value;
                    const recordingPracticeMode = selectedPracticeMode();
                    const recordingSampleId = selectedPracticeSampleId(recordingPracticeMode);
                    const audioChunks = [];
                    let stream;
                    isProcessing = true;
                    syncPracticeControls();
                    document.getElementById("sampleAudio").pause();
                    document.getElementById("uploadedAudio").pause();
                    document.getElementById("recordedAudio").pause();
                    try {
                        if (!navigator.mediaDevices?.getUserMedia) {
                            throw new Error(t("microphoneHttps"));
                        }
                        if (!window.MediaRecorder) {
                            throw new Error(t("recordingUnsupported"));
                        }
                        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        const recordingMimeTypes = [
                            "audio/webm;codecs=opus",
                            "audio/webm",
                            "audio/mp4",
                            "audio/ogg;codecs=opus"
                        ];
                        const supportedMimeType = typeof MediaRecorder.isTypeSupported === "function"
                            ? recordingMimeTypes.find(type => MediaRecorder.isTypeSupported(type))
                            : "";
                        mediaRecorder = supportedMimeType
                            ? new MediaRecorder(stream, { mimeType: supportedMimeType })
                            : new MediaRecorder(stream);
                        const recordingMimeType = mediaRecorder.mimeType || supportedMimeType || "audio/webm";
                        const recordingExtension = recordingMimeType.includes("mp4")
                            ? "m4a"
                            : recordingMimeType.includes("ogg") ? "ogg" : "webm";

                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };

                        mediaRecorder.onstop = async () => {
                            isRecording = false;
                            hideRecordingIndicator();
                            document.getElementById("recordBtn").classList.remove("recording");
                            document.getElementById("recordBtn").innerText = t("startRecording");
                            const audioBlob = new Blob(audioChunks, { type: recordingMimeType });
                            previewRecording(audioBlob);
                            await submitAssessment(audioBlob, `recording.${recordingExtension}`, recordingScript, recordingSampleId, recordingPracticeMode);
                        };

                        mediaRecorder.start();
                        isRecording = true;
                        isProcessing = false;
                        showRecordingIndicator(stream);
                        syncPracticeControls();
                        document.getElementById("recordBtn").classList.add("recording");
                        document.getElementById("recordBtn").innerText = t("recordingStop");
                        showTranslatedStatus("recording");
                    } catch (err) {
                        if (stream) stream.getTracks().forEach(track => track.stop());
                        hideRecordingIndicator();
                        isProcessing = false;
                        syncPracticeControls();
                        showTranslatedStatus("microphoneFailed", { error: err.message || err }, true);
                    }
                } else {
                    stopRecording();
                }
            }

            function stopRecording() {
                if (isRecording && mediaRecorder) {
                    isProcessing = true;
                    mediaRecorder.stop();
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                    isRecording = false;
                    hideRecordingIndicator();
                    document.getElementById("recordBtn").classList.remove("recording");
                    document.getElementById("recordBtn").innerText = t("startRecording");
                    syncPracticeControls();
                }
            }

            function renderPhrasePitchChart(phrase, index) {
                const scale = document.getElementById("pitchScale").value;     // "relative" | "hz" | "midi"
                const useHz = scale === "hz";
                const useMidi = scale === "midi";
                const valueKey = useHz ? "hz" : useMidi ? "midi" : "relative_semitone";
                const contour = Array.isArray(phrase.phrase_f0?.contour) ? phrase.phrase_f0.contour : [];
                const duration = Number(phrase.phrase_f0?.duration_sec || 0);
                const values = contour.map(point => point[valueKey]).filter(Number.isFinite);
                if (!values.length || duration <= 0) return `<div class="pitch-chart unavailable">${t("noPitch")}</div>`;

                let yMin, yMax;
                if (useHz) {
                    const low = Math.min(...values), high = Math.max(...values);
                    const padding = Math.max(15, (high - low) * 0.15);
                    yMin = Math.max(0, Math.floor((low - padding) / 25) * 25);
                    yMax = Math.ceil((high + padding) / 25) * 25;
                } else if (useMidi) {                                              // absolute notes: no zero line to keep in view
                    yMin = Math.floor(Math.min(...values)) - 1;
                    yMax = yMin + Math.ceil((Math.ceil(Math.max(...values)) + 1 - yMin) / 4) * 4;    // four equal steps of whole notes
                } else {
                    yMin = Math.min(-2, Math.floor(Math.min(...values)));
                    yMax = Math.max(2, Math.ceil(Math.max(...values)));
                }
                if (yMax <= yMin) yMax = yMin + 1;
                const width = 620, height = 180, plotLeft = 48, plotRight = width - 18;
                let lines = "";
                for (let tick = 0; tick <= 4; tick++) {
                    const y = 18 + tick * 26;
                    const value = yMax - (yMax - yMin) * tick / 4;
                    lines += `<line class="pitch-grid" x1="${plotLeft}" y1="${y}" x2="${plotRight}" y2="${y}"></line>`;
                    lines += `<text class="pitch-label" x="${plotLeft - 6}" y="${y + 4}" text-anchor="end">${useHz || useMidi ? value.toFixed(0) : value.toFixed(1)}</text>`;
                }
                let segment = [];
                const segments = [];
                for (const point of contour) {
                    const value = point[valueKey];
                    if (!Number.isFinite(value)) {
                        if (segment.length) segments.push(segment);
                        segment = [];
                        continue;
                    }
                    const x = plotLeft + Math.min(1, Math.max(0, Number(point.offset_sec) / duration)) * (plotRight - plotLeft);
                    const y = 18 + (yMax - value) * 104 / (yMax - yMin);
                    segment.push(`${x.toFixed(1)},${y.toFixed(1)}`);
                }
                if (segment.length) segments.push(segment);
                for (const points of segments.filter(value => value.length > 1)) {
                    lines += `<polyline class="learner-line" points="${points.join(" ")}"><title>${t("yourRecording")}</title></polyline>`;
                }
                return `<div class="pitch-chart">
                    <svg viewBox="0 0 ${width} ${height}" role="img" tabindex="0" aria-labelledby="phrase-pitch-title-${index} phrase-pitch-description-${index}">
                        <title id="phrase-pitch-title-${index}">${useHz ? t("f0Title") : useMidi ? t("midiTitle") : t("relativeTitle")} \u2013 ${t("phrase")} ${index + 1}</title>
                        <desc id="phrase-pitch-description-${index}">${t("pitchDescription")}</desc>
                        <text class="pitch-label" x="8" y="12">${useHz ? t("hz") : useMidi ? t("midiLabel") : t("relativeSemitone")}</text>
                        ${lines}
                    </svg>
                </div>`;
            }

            function refreshPitchPlots() {
                if (!lastAssessment) return;
                const pitchDetails = document.getElementById("pitchDetails");
                const wasOpen = pitchDetails.open;
                renderVisualResults(lastAssessment);
                pitchDetails.open = wasOpen;
            }

            function openPhraseChart(index) {
                const details = document.getElementById("pitchDetails");
                details.open = true;
                document.getElementById(`phrase-chart-${index}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }

            function renderVisualResults(data) {
                lastAssessment = data;
                document.getElementById("resultCard").style.display = "block";
                document.getElementById("targetScriptDisplay").innerText = t("scriptLabel", { script: data.target_script || "" });

                const legacyWords = Array.isArray(data.words) ? data.words : [];
                const phrases = Array.isArray(data.phrases) && data.phrases.length
                    ? data.phrases
                    : legacyWords.length ? legacyWords : (Array.isArray(data.clauses) ? data.clauses.map(clause => ({
                        annotation: clause.clause_text, word: clause.clause_text, status: "uncertain"
                    })) : []);
                const statusLabels = {
                    likely_correct: t("likelyCorrect"),
                    practice: t("practice"),
                    uncertain: t("uncertain"),
                    insufficient_evidence: t("insufficientEvidence"),
                    not_heard: t("notHeard"),
                    text_mismatch: t("mismatch"),
                    transcript_unchecked: t("transcriptUnchecked"),
                    unassessable: t("unassessable")
                };
                const statusClass = status => status === "likely_correct"
                    ? "likely-correct"
                    : status === "practice" ? "practice" : status === "not_heard" ? "not-heard" : "uncertain";
                const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({
                    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
                }[char]));
                const sequenceText = sequence => Array.isArray(sequence) && sequence.length
                    && sequence.some(tone => tone !== null && tone !== undefined)
                    ? sequence.map(tone => tone === null || tone === undefined ? "T?" : `T${Number(tone)}`).join(" - ")
                    : t("sequenceUnavailable");
                const transcriptCheck = data.transcript_check || {};
                const transcriptNote = transcriptCheck.status === "script_provided"
                    ? t("scriptProvidedOnly")
                    : transcriptCheck.status === "no_match"
                    ? t("noMatch", { matched: transcriptCheck.matched_syllables || 0, total: transcriptCheck.total_syllables || 0 })
                    : transcriptCheck.status === "partial_match"
                        ? t("partialMatch", { matched: transcriptCheck.matched_syllables, total: transcriptCheck.total_syllables })
                        : transcriptCheck.status === "unchecked"
                            ? t("unchecked")
                            : t("matched", { matched: transcriptCheck.matched_syllables || 0, total: transcriptCheck.total_syllables || 0 });

                const qualityKeys = { clipped: "qualityClipped", too_quiet: "qualityTooQuiet", noisy: "qualityNoisy", too_short: "qualityTooShort", too_long: "qualityTooLong", no_speech: "qualityNoSpeech", no_voicing: "qualityNoVoicing" };
                const qualityKey = data.audio_quality && data.audio_quality.passed === false ? qualityKeys[(data.audio_quality.reasons || [])[0]] : null;
                // The aligner itself failed (not the learner's recording): say so, instead of leaving "Okay" everywhere.
                const systemFailure = data.tone_verification?.alignment?.error_kind === "system";
                const retryKeys = { short_fall: "retryShortFall", slower: "retrySlower", steady: "retrySteady", clearer: "retryClearer", read_exactly: "retryReadExactly", again: "retryAgain", not_heard: "retryNotHeard", full_third: "retryFullThird" };
                const groups = phrases.map((phrase, index) => {
                    const labelText = phrase.annotation || phrase.word || `${t("phrase")} ${index + 1}`;
                    // In a phrase of several characters, mark the one that needs work (red wavy line) or was unclear (amber dotted line).
                    const parts = Array.isArray(phrase.syllable_diagnostics) ? phrase.syllable_diagnostics : [];
                    const marked = parts.length > 1 && parts.map(part => part.character).join("") === labelText;
                    const label = marked ? parts.map(part => {
                        const mark = part.status === "practice" ? "char-practice"
                            : part.status === "insufficient_evidence" && part.expected_surface_tone !== 5 ? "char-doubt" : "";
                        return mark ? `<span class="${mark}">${escapeHtml(part.character)}</span>` : escapeHtml(part.character);
                    }).join("") : escapeHtml(labelText);
                    const expected = sequenceText(phrase.expected_tone_sequence);
                    // Good = tick, Needs practice = "again" arrow, Not heard = dash, Okay = a half-filled circle ("partly there"),
                    // drawn with CSS: a "?" read as "something is wrong with the system", and it needs no font glyph.
                    const icon = phrase.status === "likely_correct" ? "&#10003;" : phrase.status === "practice" ? "&#8635;" : phrase.status === "not_heard" ? "&#8212;" : "";
                    const iconClass = icon ? "status-icon" : "status-icon half";
                    const advice = (Array.isArray(phrase.feedback_items) ? phrase.feedback_items : []).map(item =>
                        item.type === "tone" && item.tone >= 1 && item.tone <= 4 ? t(`adviceTone${item.tone}`, { char: item.char })
                        : item.type === "retry" && retryKeys[item.code]
                            ? (item.char ? `\u201c${item.char}\u201d \u2013 ` : "") + t(retryKeys[item.code]) : "").filter(Boolean);
                    return `<button type="button" class="word-card ${statusClass(phrase.status)}" aria-controls="phrase-chart-${index}" onclick="openPhraseChart(${index})">
                        <span class="word-card-label">${label}</span>
                        <span class="word-card-tone-sequence"><b>${t("expectedSequence")}:</b> ${escapeHtml(expected)}</span>
                        <span class="word-card-status"><span class="${iconClass}" aria-hidden="true">${icon}</span>${statusLabels[phrase.status] || t("notScored")}</span>
                        ${advice.map(text => `<span class="word-card-advice">${escapeHtml(text)}</span>`).join("")}
                    </button>`;
                });

                document.getElementById("pronunciationResults").innerHTML = `
                    <section class="pronunciation-results" aria-labelledby="pronunciation-title">
                        <h3 id="pronunciation-title">${t("pronunciationFeedback")}</h3>
                        <p class="results-note">${transcriptNote} ${t("renderPhraseScores")}</p>
                        <p class="results-note">${t("noProbabilityModel")}</p>
                        ${qualityKey ? `<p class="results-advice">${escapeHtml(t(qualityKey))}</p>` : ""}
                        ${systemFailure ? `<p class="results-advice">${escapeHtml(t("alignmentFailed"))}</p>` : ""}
                        ${data.pace?.too_fast && !systemFailure ? `<p class="results-advice">${escapeHtml(t("paceTooFast"))}</p>` : ""}
                        ${(() => { const key = { longer: "lengthLonger", shorter: "lengthShorter", different: "lengthDifferent" }[data.length_check?.direction];
                                   return key && !systemFailure ? `<p class="results-advice">${escapeHtml(t(key))}</p>` : ""; })()}
                        <div class="result-groups">${groups.join("") || `<p>${t("noScores")}</p>`}</div>
                        <details class="json-output">
                            <summary>${t("viewJsonOutput")}</summary>
                            <pre id="assessmentJsonOutput"></pre>
                        </details>
                    </section>`;
                document.getElementById("assessmentJsonOutput").textContent = JSON.stringify(data, null, 2);

                document.getElementById("clausesContainer").innerHTML = phrases.map((phrase, index) => `
                    <section class="pitch-clause" id="phrase-chart-${index}">
                        <h3>${escapeHtml(phrase.annotation || phrase.word || `${t("phrase")} ${index + 1}`)} - ${escapeHtml(sequenceText(phrase.expected_tone_sequence))}</h3>
                        ${renderPhrasePitchChart(phrase, index)}
                    </section>`).join("");
            }

        </script>
    </body>
    </html>
    """


@app.post("/api/assess-pronunciation")
def assess_pronunciation(
    file: UploadFile = File(...),
    script: str = Form(...),
    sample_id: str = Form(default="custom"),
    student_id: str = Form(...),
    include_transcript: bool = Form(default=True),   # accepted for older clients; ASR is no longer used
    practice_mode: str = Form(default="book"),
    level: str = Form(default=""),          # API / tools only (the page has no switch and never sends it): "" = the default for the learner's group (easy for everybody), or "standard" / "easy"
):
    with ASSESSMENT_LOCK:
        return _assess_pronunciation(file, script, sample_id, student_id, practice_mode, level)


def _assess_pronunciation(
    file: UploadFile,
    script: str,
    sample_id: str,
    student_id: str,
    practice_mode: str = "book",
    level: str = "",
):
    """Verify the learner's tones against the known script (calibrated Praat AC+CC F0, MFA alignment)."""
    allowed_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")
    uploaded_filename = str(file.filename or "")
    if not uploaded_filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")
    if not script or not script.strip():
        raise HTTPException(status_code=400, detail="The input script must not be empty.")
    student = practice_store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=400, detail="Enter a valid speaker profile before practicing.")
    voice_calibration = practice_store.get_voice_calibration(student_id)
    try:
        assessment_level = tone_assessment.resolve_level(level, student.get("group_type"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    practice_mode = str(practice_mode or "book").strip().lower()
    if practice_mode not in {"book", "custom", "chapter5_word"}:
        raise HTTPException(status_code=400, detail="Invalid practice mode.")
    target_script = converter.convert(script.strip())
    practice_item = None
    if practice_mode == "book":
        sample = SAMPLE_CATALOG.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=400, detail="The selected practice script was not found.")
        sample_hanzi = comparable_script(sample["script"])
        received_hanzi = comparable_script(target_script)
        if sample_hanzi != received_hanzi:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "The selected sample and script do not match. Only punctuation and spacing differences are allowed.",
                    "sample_id": sample_id,
                    "expected_script": sample["script"],
                    "received_script": script.strip(),
                    "expected_hanzi": sample_hanzi,
                    "received_hanzi": received_hanzi,
                },
            )
    elif practice_mode == "chapter5_word":
        practice_item = CHAPTER5_WORD_ITEMS.get(sample_id)
        if practice_item is None:
            raise HTTPException(status_code=400, detail="The selected Chapter 5 word was not found.")
        allowed_scripts = {
            comparable_script(converter.convert(variant))
            for variant in practice_item.get("script_variants", [practice_item["script"]])
        }
        if comparable_script(target_script) not in allowed_scripts:
            raise HTTPException(
                status_code=400,
                detail="The submitted script does not match the selected Chapter 5 word.",
            )
    else:
        # Custom practice uses the learner's own transcript as its target and never
        # looks up or compares against a lesson catalog entry.
        sample_id = "custom"

    clauses = [clause for clause in core.prepare_target(target_script) if clause.strip()]
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="The script contains no valid Han characters to score pronunciation.",
        )

    suffix = os.path.splitext(uploaded_filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        prepared = audio_prep.prepare_recording(temp_path, tone_assessment.get_config())
        if len(prepared.samples) == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio is empty.")

        if practice_mode == "chapter5_word":
            payload = tone_assessment.assess_chapter5_item(
                prepared, item=practice_item, sample_id=sample_id,
                target_script=target_script, calibration=voice_calibration, normalize=converter.convert,
                level=assessment_level,
            )
        else:
            words = (tone_assessment.segment_words(clauses, SEGMENT_LEXICON) if practice_mode == "custom"
                     else WORD_ANNOTATIONS.get(sample_id, []))          # typed text has no annotation: split it into words
            groups, word_annotation_status = tone_assessment.make_groups(clauses, words, converter.convert)
            contexts = tone_assessment.sentence_contexts(clauses, groups, word_annotation_status)
            payload = tone_assessment.assess_recording(
                prepared, clauses=clauses, contexts=contexts, groups=groups,
                word_annotation_status=word_annotation_status, calibration=voice_calibration,
                practice_mode=practice_mode, sample_id=sample_id, target_script=target_script,
                level=assessment_level,
            )
        try:
            payload["saved_attempt"] = practice_store.save_attempt(
                student_id=student_id,
                sample_id=sample_id,
                target_script=target_script,
                audio_source=temp_path,
                original_filename=uploaded_filename,
                result=payload,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Assessment completed, but saving this practice attempt failed: {exc}",
            )
        user_archive.archive_attempt(                       # best effort: never affects the result the learner sees
            practice_store.get_student(student_id) or {"id": student_id, "name": student_id},
            payload["saved_attempt"], temp_path, uploaded_filename, payload)
        return payload
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
