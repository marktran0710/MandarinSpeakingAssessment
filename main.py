import csv
import hashlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from threading import Lock

import opencc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from pypinyin import lazy_pinyin, Style

import audio_prep
import capt_core as core
import tone_assessment
import tone_pipeline
from practice_store import practice_store

# Initialize the FastAPI app
app = FastAPI(title="Taiwan Mandarin CAPT API")
ASSESSMENT_LOCK = Lock()
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
            "pinyin": lazy_pinyin(sample["script"], style=Style.TONE),
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
            "character_pinyin": lazy_pinyin(item["script"], style=Style.TONE),
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


@app.get("/api/students/{student_id}")
async def get_student_profile(student_id: str):
    profile = practice_store.get_student(student_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Speaker profile was not found.")
    return profile


@app.post("/api/students/{student_id}/voice-calibration")
def calibrate_student_voice(student_id: str, file: UploadFile = File(...)):
    if practice_store.get_student(student_id) is None:
        raise HTTPException(status_code=404, detail="Speaker profile was not found.")
    allowed_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")
    filename = str(file.filename or "voice-calibration.webm")
    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")
    suffix = os.path.splitext(filename)[1].lower() or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        samples, sample_rate = audio_prep.decode_audio(temp_path)
        calibration = tone_pipeline.calibrate_voice(samples, sample_rate, tone_assessment.get_config())
        saved = practice_store.save_voice_calibration(student_id, calibration)
        return {"voice_calibration": saved}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
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
        <p id="loadStatus" role="status">Loading results...</p>
        <main id="users"></main>
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
                        body.append(el("div", `${phrase} | Expected ${toneSequence(item.expected_tone_sequence)} | Detected ${toneSequence(item.observed_tone_sequence)} | ${statusNames[item.status] || item.status || "Not scored"}${note}`, "word"));
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
                    status.textContent = `${data.users.length} users`;
                    if (!data.users.length) status.textContent = "No users yet.";
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
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 750px; margin: 30px auto; padding: 20px; background: #f4f6f9; }
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
            #voiceCalibrationDialog { width: min(520px, calc(100vw - 32px)); box-sizing: border-box; border: 0; border-radius: 14px; padding: 24px; box-shadow: 0 18px 60px rgba(15, 35, 55, 0.28); }
            #voiceCalibrationDialog::backdrop { background: rgba(24, 39, 56, 0.55); }
            #voiceCalibrationDialog h2 { margin-top: 0; }
            #voiceCalibrationScript { margin: 14px 0; padding: 14px; border: 1px solid #d9e2ef; border-radius: 9px; background: #f7fafc; font-size: 20px; line-height: 1.9; }
            #voiceCalibrationScript small { display: block; color: #607386; font-size: 14px; line-height: 1.6; }
            #voiceCalibrationControls { display: flex; flex-wrap: wrap; gap: 9px; }
            #voiceCalibrationRecordBtn { background: #2b6cb0; color: white; }
            #voiceCalibrationRecordBtn.recording { background: #1a4f80; }
            #voiceCalibrationLaterBtn { background: #edf2f7; color: #2d3748; }
            #voiceCalibrationStatus { min-height: 1.5em; color: #4a5568; overflow-wrap: anywhere; }
            #voiceCalibrationStatus.error { color: #9b2c2c; }
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
            .word-card-label { align-self: center; font-size: 17px; font-weight: 650; }
            .word-card-tone-sequence { color: #607386; font-size: 12px; letter-spacing: 0.02em; text-align: center; }
            .word-card-status { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; }
            .status-icon { display: inline-grid; place-items: center; width: 20px; height: 20px; border: 1.5px solid currentColor; border-radius: 50%; font-size: 13px; line-height: 1; }
            .json-output { margin-top: 16px; border-top: 1px solid #e2e8f0; }
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
            .tone-tag { font-size: 10px; margin-top: 4px; padding: 2px 6px; border-radius: 4px; display: inline-block; }
            .tone-likely-correct { background: #c6f6d5; color: #22543d; }
            .tone-likely-incorrect { background: #fed7d7; color: #742a2a; }
            .tone-uncertain { background: #feebc8; color: #744210; }

        </style>
    </head>
    <body>
        <div class="card" id="studentCard">
            <div class="student-heading">
                <h2 data-i18n="speakerProfile">Speaker profile</h2>
                <label class="language-picker" for="languageSelect"><span data-i18n="language">Language</span>
                    <select id="languageSelect" onchange="changeLanguage(this.value)" aria-label="Language">
                        <option value="en">English</option>
                        <option value="zh-TW" data-i18n="languageZh">Traditional Chinese</option>
                        <option value="vi" data-i18n="languageVi">Vietnamese</option>
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
            <p class="upload-hint" data-i18n="profileHint">Use a name that distinguishes you from others in the same group; matching name and group reuse the same practice history.</p>
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

        <dialog id="voiceCalibrationDialog" aria-labelledby="voiceCalibrationTitle" aria-describedby="voiceCalibrationInstruction">
            <h2 id="voiceCalibrationTitle" data-i18n="voiceCalibrationTitle">Personalize your voice</h2>
            <p data-i18n="voiceCalibrationPrompt">Record a short sample so tone feedback can use a pitch range suited to your voice.</p>
            <p id="voiceCalibrationInstruction" class="upload-hint" data-i18n="voiceCalibrationInstruction">Read the prompt twice, naturally, for about 5–10 seconds. Praat checks your pitch with two tracking methods. The calibration audio is processed and discarded; only pitch statistics are saved.</p>
            <div id="voiceCalibrationScript" lang="zh-Hant">
                媽、麻、馬、罵、嗎；你好嗎？我今天想學中文。
                <small>mā (T1) · má (T2) · mǎ (T3) · mà (T4) · ma (neutral)</small>
            </div>
            <div id="voiceCalibrationControls">
                <button id="voiceCalibrationRecordBtn" type="button" onclick="toggleVoiceCalibrationRecording()" data-i18n="recordCalibration">Record calibration</button>
                <button id="voiceCalibrationLaterBtn" type="button" onclick="deferVoiceCalibration()" data-i18n="voiceCalibrationLater">Do this later</button>
            </div>
            <p id="voiceCalibrationStatus" role="status" aria-live="polite"></p>
        </dialog>

        <div class="card">
            <h2 data-i18n="appTitle">Taiwan Mandarin Pronunciation Practice</h2>
            <p data-i18n="practiceInstructions">Choose a human voice sample, listen to it, then record yourself or upload an audio file reading the matching script:</p>
            <fieldset id="practiceMode" class="practice-mode">
                <legend data-i18n="practiceSource">Practice text</legend>
                <label><input type="radio" name="practiceMode" value="book" checked onchange="changePracticeMode()"><span data-i18n="bookTextMode">Book sample</span></label>
                <label><input type="radio" name="practiceMode" value="chapter5_word" onchange="changePracticeMode()"><span data-i18n="chapter5WordMode">Chapter 5 · one item at a time</span></label>
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

            <div id="bookSamplePicker" style="margin-bottom: 15px;">
                <label for="sampleSelect"><b data-i18n="sampleRecording">Sample recording (Lessons 5-8):</b></label>
                <select id="sampleSelect" onchange="selectSample()" disabled>
                    <option data-i18n="loadingSamples">Loading sample recordings...</option>
                </select>
            </div>

            <div class="sample-audio-section" id="sampleAudioSection" hidden>
                <p id="chapter5ReferenceLabel" class="upload-hint" data-i18n="chapter5ReferenceAudio" hidden>Listen to the word audio as a reference. It is not used to score your recording.</p>
                <label for="sampleAudio"><b data-i18n="sampleAudioLabel">Listen to sample audio:</b></label>
                <audio id="sampleAudio" controls preload="metadata" aria-label="Sample audio" data-i18n-aria-label="sampleAudioLabel"></audio>
                <p id="sampleAudioError" class="sample-audio-error" role="status" data-i18n="sampleAudioLoadError" hidden>Sample audio could not be loaded. Try selecting the item again.</p>
            </div>

            <div style="margin-bottom: 15px;">
                <label for="scriptInput"><b id="scriptLabel" data-i18n="matchingScript">Matching script:</b></label>
                <p id="customTextHint" class="upload-hint" data-i18n="customTextHint" hidden>Enter any Mandarin text. It does not need to appear in a textbook.</p>
                <textarea id="scriptInput" rows="3" readonly oninput="handleScriptInput()" data-i18n-placeholder="customTextPlaceholder"></textarea>
                <div id="transcriptDisplay" class="transcript" aria-label="Matching script with pinyin" data-i18n-aria-label="scriptWithPinyin"></div>
                <label id="pinyinToggle" class="pinyin-toggle">
                    <input type="checkbox" id="showPinyin" checked onchange="refreshPinyinDisplay()">
                    <span data-i18n="showPinyin">Show pinyin</span>
                </label>
            </div>

            <div class="record-controls">
                <button id="recordBtn" onclick="toggleRecording()" disabled data-i18n="startRecording">Start recording</button>
                <button id="stopBtn" onclick="stopRecording()" disabled data-i18n="stopAndAssess">Stop &amp; assess</button>
            </div>

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
            </div>

            <div id="pronunciationResults"></div>
            <details id="pitchDetails" class="pitch-details">
                <summary data-i18n="viewPitch">View pitch graphs (optional)</summary>
                <div class="pitch-controls">
                    <label for="pitchScale"><b data-i18n="pitchDisplay">Pitch display:</b></label>
                    <select id="pitchScale" onchange="refreshPitchPlots()">
                        <option value="relative" data-i18n="relativePitch">Relative pitch</option>
                        <option value="hz" data-i18n="f0Hz">F0 (Hz)</option>
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
            let isSavingStudent = false;
            let sampleLessons = [];
            let chapter5WordItems = [];
            let chapter5Categories = [];
            let chapter5CategoryFilter = "";
            let chapter5WordIndex = 0;
            let lastAssessment = null;
            let activeStudent = null;
            let customTextDraft = "";
            let activeFeedbackAttemptId = null;
            let uploadedAudioUrl = null;
            let recordedAudioUrl = null;

            const LANGUAGE_TEXT = {
                en: {
                    language: "Language", languageEn: "English", languageZh: "中文（繁體）", languageVi: "Tiếng Việt",
                    speakerProfile: "Speaker profile", changeSpeaker: "Change speaker",
                    profilePrompt: "Enter your name and participant group before speaking practice.",
                    profileHint: "Use a name that distinguishes you from others in the same group; matching name and group reuse the same practice history.",
                    name: "Name", participantGroup: "Participant group", chooseOne: "Choose one...", taiwanese: "Taiwanese", foreigner: "Foreigner",
                    continue: "Continue", cancel: "Cancel", appTitle: "Taiwan Mandarin Pronunciation Practice",
            practiceInstructions: "Choose a lesson sample or enter your own Mandarin text, then record yourself reading it:",
            practiceSource: "Practice text", bookTextMode: "Book sample", customTextMode: "My own text",
            chapter5WordMode: "Chapter 5 · one item at a time", previousWord: "Previous", nextWord: "Next",
            chapter5Vocabulary: "Vocabulary", chapter5Phrase: "Phrase", chapter5Meaning: "Meaning: {meaning}",
            chapter5ReferenceAudio: "Listen to the word audio as a reference. It is not used to score your recording.",
            chapter5Progress: "Item {current} of {total}", chapter5Unavailable: "Chapter 5 practice items are unavailable.",
            chapter5Ready: "Chapter 5 item ready. Record yourself or upload your reading.",
                    chapter5Topic: "Topic", chapter5AllTopics: "All topics", chapter5AllInTopic: "All in {topic}",
            customTextPreview: "Entered text preview",
            customScriptLabel: "Text to read:",
            customTextHint: "Enter any Mandarin text. It does not need to appear in a textbook.",
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
                    voiceCalibrationTitle: "Personalize your voice", voiceCalibrationPrompt: "Record a short sample so tone feedback can use a pitch range suited to your voice.",
                    voiceCalibrationInstruction: "Read the prompt twice, naturally, for about 5–10 seconds. Praat checks your pitch with two tracking methods. The calibration audio is processed and discarded; only pitch statistics are saved.",
                    voiceCalibrationLater: "Do this later", calibrationDone: "Done", recordCalibration: "Record calibration", stopCalibration: "Stop and calibrate",
                    calibrationRecording: "Recording… read the prompt twice.", calibrationProcessing: "Checking your pitch range…", calibrationSaved: "Voice calibration saved.",
                    calibrationTooShort: "Please record for at least 5 seconds.", calibrationFailed: "Could not calibrate this recording. Read the full prompt clearly and try again.",
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
                    likelyCorrect: "Good", practice: "Practice", uncertain: "Uncertain", insufficientEvidence: "Not enough evidence",
                    mismatch: "Not scored: script mismatch", transcriptUnchecked: "Not scored: transcript unchecked",
                    unassessable: "Not enough audio to score", notScored: "Not scored", noMatch: "Speech recognition did not match the script; phrase tone feedback was skipped.",
                    partialMatch: "Speech recognition only partly matched the script; unmatched phrases are not tone-scored.",
                    unchecked: "Speech recognition was not checked, so phrase tone feedback is unavailable.",
                    matched: "Speech recognition matched the script; tone feedback is shown by phrase.", renderPhraseScores: "Each card checks the expected tone sequence for the whole phrase.",
                    expectedSequence: "Expected", detectedSequence: "Detected", sequenceUnavailable: "Not enough tone evidence",
                    noProbabilityModel: "Tone labels use experimental rules; this system has no trained or calibrated tone probabilities.",
                    pronunciationFeedback: "Tone feedback", phraseScore: "Phrase score {score}", phraseScoreUnavailable: "Phrase score unavailable", viewJsonOutput: "View JSON response", scriptProvidedOnly: "Using the selected script; ASR and alignment are skipped in Chapter 5 direct mode.", diagnosticNotNeeded: "Not needed for this result",
                    scriptProvidedOnly: "Using the selected script; ASR and alignment are skipped in Chapter 5 direct mode.", diagnosticNotNeeded: "Not needed for this result",
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
                    noPitch: "No measurable pitch in this phrase.", f0Title: "F0 in Hz", relativeTitle: "Relative pitch", relativeSemitone: "Semitones from your pitch median", halfThirdTone: "Half-T3", clause: "sentence {number}",
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
                    profilePrompt: "請先輸入姓名和參與者類別，再開始口說練習。",
                    profileHint: "請使用能區分同組使用者的姓名；姓名和類別相同時會沿用同一份練習紀錄。",
                    name: "姓名", participantGroup: "參與者類別", chooseOne: "請選擇…", taiwanese: "台灣人", foreigner: "外國人",
                    continue: "繼續", cancel: "取消", appTitle: "台灣華語發音練習",
                    practiceInstructions: "選擇課本範例，或輸入自己的華語文本，再依照內容錄音：",
                    practiceSource: "練習文本", bookTextMode: "課本範例", customTextMode: "自選文本",
                    chapter5WordMode: "第 5 課・逐詞練習", previousWord: "上一個", nextWord: "下一個",
                    chapter5Vocabulary: "詞彙", chapter5Phrase: "詞組", chapter5Meaning: "意思：{meaning}",
                    chapter5ReferenceAudio: "可播放單詞音檔作為參考；評分不會使用此音檔。",
                    chapter5Progress: "第 {current} 項，共 {total} 項", chapter5Unavailable: "第 5 課練習資料目前無法使用。",
                    chapter5Ready: "第 5 課項目已準備好。請錄音或上傳朗讀音檔。",
                    chapter5Topic: "主題", chapter5AllTopics: "所有主題", chapter5AllInTopic: "全部：{topic}",
                    customTextPreview: "輸入文本預覽",
                    customScriptLabel: "朗讀文本：",
                    customTextHint: "輸入任何華語文本，不必對應課本內容。",
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
                    voiceCalibrationTitle: "個人化你的聲音", voiceCalibrationPrompt: "錄製一小段聲音，讓聲調回饋使用適合你的音高範圍。",
                    voiceCalibrationInstruction: "請自然地將提示內容讀兩次，約 5–10 秒。Praat 會用兩種追蹤方式檢查音高。校準音檔只用於即時分析後丟棄，只會儲存音高統計資料。",
                    voiceCalibrationLater: "稍後再做", calibrationDone: "完成", recordCalibration: "開始校準錄音", stopCalibration: "停止並校準",
                    calibrationRecording: "錄音中……請將提示內容讀兩次。", calibrationProcessing: "正在檢查音高範圍……", calibrationSaved: "已儲存個人聲音校準。",
                    calibrationTooShort: "請至少錄音 5 秒。", calibrationFailed: "無法校準這段錄音。請清楚讀完整段提示後再試一次。",
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
                    likelyCorrect: "良好", practice: "需要練習", uncertain: "不確定", insufficientEvidence: "證據不足",
                    mismatch: "未評分：文本不符", transcriptUnchecked: "未評分：未檢查辨識文本",
                    unassessable: "音訊不足，無法評分", notScored: "未評分", noMatch: "語音辨識未符合文本，已略過聲調回饋。",
                    partialMatch: "語音辨識只部分符合文本；未符合的詞組不提供聲調回饋。",
                    unchecked: "尚未檢查語音辨識，無法提供詞組聲調回饋。",
                    matched: "語音辨識符合文本，以下顯示整個詞組的聲調回饋。", renderPhraseScores: "每張卡片評估整個詞組的預期聲調順序。",
                    expectedSequence: "預期", detectedSequence: "辨識", sequenceUnavailable: "聲調證據不足",
                    noProbabilityModel: "聲調結果使用實驗性規則；目前沒有經訓練或校準的聲調機率模型。",
                    pronunciationFeedback: "聲調回饋", phraseScore: "詞組分數 {score}", phraseScoreUnavailable: "無詞組分數", viewJsonOutput: "\u67e5\u770b JSON \u8f38\u51fa", scriptProvidedOnly: "???????? 5 ??????????????????", diagnosticNotNeeded: "????????",
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
                    noPitch: "這個詞組沒有可測量的音高。", f0Title: "基頻（Hz）", relativeTitle: "相對音高", relativeSemitone: "相對錄音中位數（半音）", halfThirdTone: "半三聲", clause: "句子 {number}",
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
                    profilePrompt: "Nhập tên và nhóm người tham gia trước khi luyện nói.",
                    profileHint: "Hãy dùng tên giúp phân biệt bạn với người khác trong cùng nhóm; cùng tên và nhóm sẽ dùng chung lịch sử luyện tập.",
                    name: "Tên", participantGroup: "Nhóm người tham gia", chooseOne: "Chọn một mục...", taiwanese: "Người Đài Loan", foreigner: "Người nước ngoài",
                    continue: "Tiếp tục", cancel: "Hủy", appTitle: "Luyện phát âm tiếng Quan Thoại Đài Loan",
                    practiceInstructions: "Chọn bài mẫu hoặc nhập văn bản tiếng Trung của riêng bạn, rồi đọc và ghi âm:",
                    practiceSource: "Nội dung luyện tập", bookTextMode: "Bài trong sách", customTextMode: "Tự chọn văn bản",
                    chapter5WordMode: "Chương 5 · luyện từng mục", previousWord: "Trước", nextWord: "Tiếp theo",
                    chapter5Vocabulary: "Từ vựng", chapter5Phrase: "Cụm từ", chapter5Meaning: "Nghĩa: {meaning}",
                    chapter5ReferenceAudio: "Có thể nghe audio từ làm mẫu; audio này không được dùng để chấm bài ghi âm.",
                    chapter5Progress: "Mục {current}/{total}", chapter5Unavailable: "Không tải được dữ liệu luyện tập Chương 5.",
                    chapter5Ready: "Mục Chương 5 đã sẵn sàng. Hãy ghi âm hoặc tải bài đọc lên.",
                    chapter5Topic: "Chủ đề", chapter5AllTopics: "Tất cả chủ đề", chapter5AllInTopic: "Tất cả: {topic}",
                    customTextPreview: "Xem trước nội dung đã nhập",
                    customScriptLabel: "Nội dung cần đọc:",
                    customTextHint: "Nhập bất kỳ văn bản tiếng Trung nào; không cần có trong sách.",
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
                    voiceCalibrationTitle: "Cá nhân hóa giọng nói", voiceCalibrationPrompt: "Ghi một đoạn ngắn để hệ thống chọn dải cao độ phù hợp với giọng của bạn.",
                    voiceCalibrationInstruction: "Đọc tự nhiên đoạn gợi ý hai lần, trong khoảng 5–10 giây. Praat kiểm tra cao độ bằng hai cách theo dõi. Âm thanh hiệu chuẩn chỉ được xử lý rồi xóa; hệ thống chỉ lưu thống kê cao độ.",
                    voiceCalibrationLater: "Để sau", calibrationDone: "Xong", recordCalibration: "Ghi mẫu hiệu chuẩn", stopCalibration: "Dừng và hiệu chuẩn",
                    calibrationRecording: "Đang ghi… hãy đọc đoạn gợi ý hai lần.", calibrationProcessing: "Đang kiểm tra dải cao độ…", calibrationSaved: "Đã lưu hiệu chuẩn giọng nói.",
                    calibrationTooShort: "Hãy ghi ít nhất 5 giây.", calibrationFailed: "Không thể hiệu chuẩn bản ghi này. Hãy đọc rõ toàn bộ đoạn gợi ý rồi thử lại.",
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
                    likelyCorrect: "Tốt", practice: "Cần luyện", uncertain: "Chưa chắc", insufficientEvidence: "Chưa đủ bằng chứng",
                    mismatch: "Không chấm: nội dung không khớp", transcriptUnchecked: "Không chấm: chưa kiểm tra lời nói",
                    unassessable: "Không đủ âm thanh để chấm", notScored: "Không chấm", noMatch: "Nhận dạng giọng nói không khớp kịch bản; đã bỏ qua phản hồi thanh điệu.",
                    partialMatch: "Nhận dạng chỉ khớp một phần; cụm từ không khớp sẽ không được chấm thanh.",
                    unchecked: "Chưa kiểm tra nhận dạng, nên chưa có phản hồi thanh điệu theo cụm từ.",
                    matched: "Nhận dạng khớp nội dung; kết quả thanh điệu bên dưới là theo cụm từ.", renderPhraseScores: "Mỗi thẻ đánh giá cả chuỗi thanh dự kiến của một cụm từ.",
                    expectedSequence: "Kỳ vọng", detectedSequence: "Nhận diện", sequenceUnavailable: "Chưa đủ bằng chứng thanh điệu",
                    noProbabilityModel: "Kết quả dùng quy tắc thử nghiệm; hệ thống chưa có mô hình xác suất thanh điệu đã huấn luyện và hiệu chuẩn.",
                    pronunciationFeedback: "Phản hồi thanh điệu", phraseScore: "Điểm cụm từ {score}", phraseScoreUnavailable: "Chưa có điểm cụm từ", viewJsonOutput: "Xem JSON ph\u1ea3n h\u1ed3i", scriptProvidedOnly: "D?ng v?n b?n ?? ch?n; ch? ?? tr?c ti?p Ch??ng 5 b? qua nh?n d?ng v? c?n ch?nh.", diagnosticNotNeeded: "K?t qu? n?y kh?ng c?n ch?n ?o?n",
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
                    noPitch: "Không đo được cao độ trong cụm từ này.", f0Title: "F0 (Hz)", relativeTitle: "Cao độ tương đối", relativeSemitone: "Semitone so với trung vị bản ghi", halfThirdTone: "Thanh 3 nửa", clause: "câu {number}",
                    pitchDescription: "Biểu đồ hiển thị contour cao độ của cả cụm từ; các khoảng thiếu F0 sẽ để trống.",
                    reference: "Giọng mẫu", yourRecording: "Bản ghi của bạn", lowConfidence: " (độ tin cậy căn chỉnh hoặc cao độ thấp)",
                    relative: "Tương đối", hz: "Hz", invalidAudio: "Định dạng tệp âm thanh không hợp lệ.", emptyScript: "Nội dung không được để trống.",
                    invalidProfile: "Hãy nhập thông tin người nói hợp lệ trước khi luyện tập.", sampleMissing: "Không tìm thấy giọng mẫu đã chọn.",
                    noHanScript: "Nội dung không có chữ Hán hợp lệ để chấm phát âm.", emptyName: "Hãy nhập tên.", invalidNameLength: "Tên phải có từ 1 đến 100 ký tự.",
                    invalidGroup: "Hãy chọn người Đài Loan hoặc người nước ngoài.", attemptMissing: "Không tìm thấy lượt luyện tập đã chọn.", invalidFeedbackRating: "Điểm đánh giá phải là số nguyên từ 1 đến 5.",
                    feedbackTooLong: "Nhận xét không được dài quá 2.000 ký tự.",
                    scriptMismatch: "Mẫu và nội dung đã chọn không khớp. Chỉ được khác dấu câu và khoảng trắng. Mẫu: {expected} Nội dung nhập: {received}"
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

            function applyLanguage(language) {
                currentLanguage = Object.prototype.hasOwnProperty.call(LANGUAGE_TEXT, language) ? language : "en";
                localStorage.setItem("mandarinUiLanguage", currentLanguage);
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
                recordButton.textContent = isVoiceCalibrationRecording
                    ? t("stopCalibration") : t("recordCalibration");
                recordButton.disabled = isVoiceCalibrationProcessing;
                laterButton.disabled = isVoiceCalibrationRecording || isVoiceCalibrationProcessing;
                laterButton.textContent = calibrated ? t("calibrationDone") : t("voiceCalibrationLater");
            }

            function refreshSampleOptions() {
                const select = document.getElementById("sampleSelect");
                const selectedId = select.value;
                select.replaceChildren();
                sampleLessons.forEach(sample => select.add(new Option(t("lessonOption", sample), sample.id)));
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
                return labels?.[currentLanguage] || labels?.en || "";
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
                renderTranscript({ script: item.script, pinyin: item.character_pinyin || [] });
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
                document.getElementById("pinyinToggle").hidden = isCustom;
                document.getElementById("chapter5ReferenceLabel").hidden = !isChapter5;
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
                    renderTranscript({ script: scriptInput.value, pinyin: [] });
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

            function handleScriptInput() {
                if (selectedPracticeMode() !== "custom") return;
                const script = document.getElementById("scriptInput").value;
                customTextDraft = script;
                renderTranscript({ script, pinyin: [] });
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

            applyLanguage(localStorage.getItem("mandarinUiLanguage") || "en");

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
                status.classList.remove("error");
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
                    syncVoiceCalibrationUI();
                    voiceCalibrationTimer = window.setTimeout(() => {
                        if (isVoiceCalibrationRecording && voiceCalibrationRecorder?.state === "recording") {
                            voiceCalibrationRecorder.stop();
                        }
                    }, 10000);
                } catch (error) {
                    if (voiceCalibrationStream) voiceCalibrationStream.getTracks().forEach(track => track.stop());
                    voiceCalibrationStream = null;
                    isVoiceCalibrationRecording = false;
                    setVoiceCalibrationStatus("microphoneFailed", true, { error: error.message || error });
                    syncVoiceCalibrationUI();
                }
            }

            async function submitVoiceCalibration(audioBlob, mimeType) {
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
                    if (!response.ok) throw new Error(data.detail || t("calibrationFailed"));
                    activeStudent.voice_calibration = data.voice_calibration;
                    localStorage.setItem("mandarinPracticeStudent", JSON.stringify(activeStudent));
                    updateActiveStudentLabel();
                    sessionStorage.removeItem(calibrationDismissKey(activeStudent.id));
                    setVoiceCalibrationStatus("calibrationSaved");
                } catch (error) {
                    setVoiceCalibrationStatus("calibrationFailed", true);
                } finally {
                    isVoiceCalibrationProcessing = false;
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
                [...sample.script].forEach((character, index) => {
                    const characterElement = document.createElement("span");
                    characterElement.className = "transcript-char";
                    const hanziElement = document.createElement("span");
                    hanziElement.textContent = character;
                    characterElement.appendChild(hanziElement);
                    if (document.getElementById("showPinyin").checked && pinyin[index]) {
                        const pinyinElement = document.createElement("span");
                        pinyinElement.className = "transcript-pinyin";
                        pinyinElement.textContent = pinyin[index];
                        characterElement.appendChild(pinyinElement);
                    }
                    transcript.appendChild(characterElement);
                });
            }

            function refreshPinyinDisplay() {
                const mode = selectedPracticeMode();
                if (mode === "custom") return;
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
                    showFeedbackPanel(data.saved_attempt?.id);
                    showStatus(data.reference_pitch_warning ? `${t("assessmentSaved")} ${data.reference_pitch_warning}` : t("assessmentSaved"), false, data.reference_pitch_warning ? null : { key: "assessmentSaved", variables: {} });
                } catch (error) {
                    showTranslatedStatus("assessmentFailed", { error: error.message || error }, true);
                } finally {
                    isProcessing = false;
                    syncPracticeControls();
                }
            }

            window.addEventListener("beforeunload", () => {
                if (uploadedAudioUrl) URL.revokeObjectURL(uploadedAudioUrl);
                if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
            });

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
                            document.getElementById("recordBtn").classList.remove("recording");
                            document.getElementById("recordBtn").innerText = t("startRecording");
                            const audioBlob = new Blob(audioChunks, { type: recordingMimeType });
                            previewRecording(audioBlob);
                            await submitAssessment(audioBlob, `recording.${recordingExtension}`, recordingScript, recordingSampleId, recordingPracticeMode);
                        };

                        mediaRecorder.start();
                        isRecording = true;
                        isProcessing = false;
                        syncPracticeControls();
                        document.getElementById("recordBtn").classList.add("recording");
                        document.getElementById("recordBtn").innerText = t("recordingStop");
                        showTranslatedStatus("recording");
                    } catch (err) {
                        if (stream) stream.getTracks().forEach(track => track.stop());
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
                    document.getElementById("recordBtn").classList.remove("recording");
                    document.getElementById("recordBtn").innerText = t("startRecording");
                    syncPracticeControls();
                }
            }

            function renderPhrasePitchChart(phrase, index) {
                const useHz = document.getElementById("pitchScale").value === "hz";
                const valueKey = useHz ? "hz" : "relative_semitone";
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
                    lines += `<text class="pitch-label" x="${plotLeft - 6}" y="${y + 4}" text-anchor="end">${useHz ? value.toFixed(0) : value.toFixed(1)}</text>`;
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
                        <title id="phrase-pitch-title-${index}">${useHz ? t("f0Title") : t("relativeTitle")} ? ${t("phrase")} ${index + 1}</title>
                        <desc id="phrase-pitch-description-${index}">${t("pitchDescription")}</desc>
                        <text class="pitch-label" x="8" y="12">${useHz ? t("hz") : t("relativeSemitone")}</text>
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
                    text_mismatch: t("mismatch"),
                    transcript_unchecked: t("transcriptUnchecked"),
                    unassessable: t("unassessable")
                };
                const statusClass = status => status === "likely_correct"
                    ? "likely-correct"
                    : status === "practice" ? "practice" : "uncertain";
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

                const groups = phrases.map((phrase, index) => {
                    const label = escapeHtml(phrase.annotation || phrase.word || `${t("phrase")} ${index + 1}`);
                    const expected = sequenceText(phrase.expected_tone_sequence);
                    const observed = Array.isArray(phrase.observed_tone_sequence)
                        && phrase.observed_tone_sequence.some(tone => tone !== null && tone !== undefined)
                        ? sequenceText(phrase.observed_tone_sequence)
                        : t("diagnosticNotNeeded");
                    const icon = phrase.status === "likely_correct" ? "&#10003;" : phrase.status === "practice" ? "&#8635;" : "?";
                    return `<button type="button" class="word-card ${statusClass(phrase.status)}" aria-controls="phrase-chart-${index}" onclick="openPhraseChart(${index})">
                        <span class="word-card-label">${label}</span>
                        <span class="word-card-tone-sequence"><b>${t("expectedSequence")}:</b> ${escapeHtml(expected)}</span>
                        <span class="word-card-tone-sequence"><b>${t("detectedSequence")}:</b> ${escapeHtml(observed)}</span>
                        <span class="word-card-status"><span class="status-icon" aria-hidden="true">${icon}</span>${statusLabels[phrase.status] || t("notScored")}</span>
                    </button>`;
                });

                document.getElementById("pronunciationResults").innerHTML = `
                    <section class="pronunciation-results" aria-labelledby="pronunciation-title">
                        <h3 id="pronunciation-title">${t("pronunciationFeedback")}</h3>
                        <p class="results-note">${transcriptNote} ${t("renderPhraseScores")}</p>
                        <p class="results-note">${t("noProbabilityModel")}</p>
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
):
    with ASSESSMENT_LOCK:
        return _assess_pronunciation(file, script, sample_id, student_id, practice_mode)


def _assess_pronunciation(
    file: UploadFile,
    script: str,
    sample_id: str,
    student_id: str,
    practice_mode: str = "book",
):
    """Verify the learner's tones against the known script (calibrated Praat AC+CC F0, MFA alignment)."""
    allowed_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")
    uploaded_filename = str(file.filename or "")
    if not uploaded_filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")
    if not script or not script.strip():
        raise HTTPException(status_code=400, detail="The input script must not be empty.")
    if practice_store.get_student(student_id) is None:
        raise HTTPException(status_code=400, detail="Enter a valid speaker profile before practicing.")
    voice_calibration = practice_store.get_voice_calibration(student_id)

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
            )
        else:
            groups, word_annotation_status = tone_assessment.make_groups(
                clauses, WORD_ANNOTATIONS.get(sample_id, []), converter.convert)
            contexts = core.expected_tone_contexts(
                clauses,
                phrase_end_indices=[end for _, _, end in groups] if word_annotation_status == "matched" else None,
            )
            payload = tone_assessment.assess_recording(
                prepared, clauses=clauses, contexts=contexts, groups=groups,
                word_annotation_status=word_annotation_status, calibration=voice_calibration,
                practice_mode=practice_mode, sample_id=sample_id, target_script=target_script,
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
