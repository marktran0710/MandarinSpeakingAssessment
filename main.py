import os
import shutil
import tempfile

import joblib
import numpy as np
import opencc
import torch
import whisperx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pypinyin import lazy_pinyin, Style

import capt_core as core

# Initialize the FastAPI app
app = FastAPI(title="Taiwan Mandarin CAPT API")

# Load the tone model
MODEL_PATH = "tone_model_v2.joblib"
ml_models = {}
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("feature_dim") == core.FEATURE_DIM:
        ml_models["tone_classifier"] = bundle["classifier"]
        print(f"Loaded {MODEL_PATH} (held-out lesson accuracy: {bundle.get('cv_accuracy_non_neutral', 0) * 100:.1f}%, excluding neutral tone)")
    else:
        print(f"Warning: {MODEL_PATH} has {bundle.get('feature_dim')} features, but the code expects {core.FEATURE_DIM}. Run train_model.py again.")
else:
    print(f"Warning: {MODEL_PATH} was not found. Tone assessment is temporarily unavailable (assessable=False).")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading the WhisperX model on device: {device.upper()}...")
whisper_model = whisperx.load_model("small", device, compute_type="int8" if device == "cpu" else "float16")
align_model, metadata = whisperx.load_align_model(language_code="zh", device=device)

converter = opencc.OpenCC('s2twp')            # Normalize the script / Whisper transcript to Taiwan Traditional Chinese
to_simplified = opencc.OpenCC('t2s').convert  # The alignment model uses a simplified-character dictionary (one character -> one character)

UNVOICED_ENERGY_RATIO = 0.05   # Distinguish unvoiced speech from a low-energy region; neither is scored for tone when F0 is missing.

TONE_DESCRIPTIONS = {
    1: "Tone 1 (ML - Level)",
    2: "Tone 2 (ML - Rising)",
    3: "Tone 3 (ML - Dipping)",
    4: "Tone 4 (ML - Falling)",
    5: "Tone 5 (Neutral)"
}

def check_pronunciation_match(expected_char, spoken_char):
    exp_pinyin = lazy_pinyin(expected_char, style=Style.NORMAL)[0] if expected_char else ""
    spk_pinyin = lazy_pinyin(spoken_char, style=Style.NORMAL)[0] if spoken_char else ""
    is_pinyin_match = (exp_pinyin == spk_pinyin)
    return {
        "expected_pinyin": exp_pinyin,
        "spoken_pinyin": spk_pinyin,
        "is_syllable_correct": is_pinyin_match
    }

def classify_tone(features, char, energy_ratio):
    # 1. Neutral tone
    if char in core.NEUTRAL_TONE_CHARS:
        return {"detected_tone": 5, "tone_verified": False, "assessable": True, "desc": "Tone 5 (Neutral Tone)"}

    # 2. F0 cannot be measured (unvoiced / low energy) -> tone_verified = False
    if features is None:
        if energy_ratio >= UNVOICED_ENERGY_RATIO:
            return {
                "detected_tone": None,
                "assessable": False,
                "tone_verified": False,
                "desc": "Speech detected, but F0 cannot be measured (unvoiced); tone is not scored"
            }
        return {
            "detected_tone": None, 
            "assessable": False,
            "tone_verified": False,
            "desc": "Unvoiced or invalid (no F0)"
        }

    # 3. Tone model unavailable -> assessable = False, tone_verified = False
    clf = ml_models.get("tone_classifier")
    if clf is None:
        return {
            "detected_tone": None, 
            "assessable": False, 
            "tone_verified": False, 
            "desc": f"Could not load model {MODEL_PATH}"
        }

    # 4. Predict with the trained tone classifier
    predicted_tone = int(clf.predict(np.array([features]))[0])
    return {
        "detected_tone": predicted_tone, 
        "assessable": True, 
        "tone_verified": True,
        "desc": TONE_DESCRIPTIONS.get(predicted_tone, "ML Model Match")
    }

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Taiwan Mandarin CAPT Test UI</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 650px; margin: 40px auto; padding: 20px; background: #f9f9f9; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; border-radius: 4px; font-weight: bold; }
            #recordBtn { background-color: #e74c3c; color: white; }
            #recordBtn.recording { background-color: #c0392b; animation: pulse 1.2s infinite; }
            #stopBtn { background-color: #7f8c8d; color: white; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
            pre { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; overflow-x: auto; max-height: 400px; }
            input[type="text"] { width: 100%; padding: 10px; font-size: 15px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🎤 Taiwan Mandarin Pronunciation Assessment</h2>
            <p>Enter a Mandarin sentence, then record yourself with your microphone:</p>
            
            <div style="margin-bottom: 15px;">
                <label><b>Sample script (target script):</b></label>
                <input type="text" id="scriptInput" value="友美，妳這個週末要做什麼？">
            </div>

            <div style="margin-bottom: 15px;">
                <button id="recordBtn" onclick="toggleRecording()">Start recording</button>
                <button id="stopBtn" onclick="stopRecording()" disabled style="margin-left: 10px;">Stop &amp; assess</button>
                <span id="status" style="margin-left: 15px; font-weight: bold; color: #2980b9;"></span>
            </div>

            <h3>API analysis results:</h3>
            <pre id="result">No results yet...</pre>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;

            async function toggleRecording() {
                if (!isRecording) {
                    try {
                        audioChunks = [];
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                        
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };

                        mediaRecorder.onstop = async () => {
                            document.getElementById("status").innerText = "Uploading audio to the server for analysis...";
                            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            const formData = new FormData();
                            formData.append("file", audioBlob, "recording.webm");
                            formData.append("script", document.getElementById("scriptInput").value);

                            document.getElementById("result").innerText = "Processing with WhisperX & Parselmouth...";
                            
                            const response = await fetch("/api/assess-pronunciation", {
                                method: "POST",
                                body: formData
                            });
                            const data = await response.json();
                            document.getElementById("result").innerText = JSON.stringify(data, null, 2);
                            document.getElementById("status").innerText = "Complete!";
                        };

                        mediaRecorder.start();
                        isRecording = true;
                        document.getElementById("recordBtn").classList.add("recording");
                        document.getElementById("recordBtn").innerText = "Recording (click to stop)...";
                        document.getElementById("stopBtn").disabled = false;
                        document.getElementById("status").innerText = "Recording... Speak into the microphone.";
                    } catch (err) {
                        alert("Could not access the microphone: " + err);
                    }
                } else {
                    stopRecording();
                }
            }

            function stopRecording() {
                if (isRecording && mediaRecorder) {
                    mediaRecorder.stop();
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                    isRecording = false;
                    document.getElementById("recordBtn").classList.remove("recording");
                    document.getElementById("recordBtn").innerText = "Start recording";
                    document.getElementById("stopBtn").disabled = true;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/assess-pronunciation")
async def assess_pronunciation(
    file: UploadFile = File(...),
    script: str = Form(...)
):
    allowed_extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm')
    uploaded_filename = str(file.filename or "")
    if not uploaded_filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")

    # Fix #4: Validate the input script
    if not script or not script.strip():
        raise HTTPException(status_code=400, detail="The input script must not be empty.")

    target_script = converter.convert(script.strip())
    raw_clauses = core.prepare_target(target_script)
    clauses = [c for c in raw_clauses if c.strip()]
    
    target_chars = list("".join(clauses))

    # If the script contains no valid Han characters (for example, English, numbers, or punctuation only)
    if not target_chars:
        raise HTTPException(
            status_code=400, 
            detail="The script contains no valid Han characters to score pronunciation."
        )

    suffix = os.path.splitext(uploaded_filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        samples, sample_rate = core.load_audio_samples(temp_path)
        audio_duration_sec = len(samples) / sample_rate

        expected_tones = core.expected_tones(clauses)

        # 2. Align the known script to the audio with WhisperX
        audio_wx = whisperx.load_audio(temp_path)
        items = core.force_align_script(audio_wx, target_chars, align_model, metadata, device, to_simplified)
        core.fill_missing_timestamps(items, audio_duration_sec)

        # 3. Extract per-character F0 features
        features, windows = core.utterance_features(
            samples, sample_rate, [it["start"] for it in items], expected_tones, [len(c) for c in clauses]
        )

        # 4. Use Whisper to recognize the learner's speech and match syllables
        spoken_text = ""
        if target_chars:
            result = whisper_model.transcribe(audio_wx, batch_size=4, language="zh")
            spoken_text = converter.convert("".join(segment.get("text", "") for segment in result["segments"]))
        spoken_chars = core.match_spoken_chars("".join(target_chars), spoken_text)

        response_data = []
        last_idx = len(target_chars) - 1

        for i, ref_char in enumerate(target_chars):
            item, ft, window = items[i], features[i], windows[i]
            energy_ratio = core.window_energy_ratio(samples, sample_rate, *window) if ft is None and window else 0.0
            tone_result = classify_tone(ft, ref_char, energy_ratio)
            expected_tone = expected_tones[i]

            spoken_char = spoken_chars[i] or ""
            match_info = check_pronunciation_match(ref_char, spoken_char)

            if not tone_result.get("assessable", True) or expected_tone is None:
                raw_is_correct_tone = None
            else:
                raw_is_correct_tone = tone_result["detected_tone"] == expected_tone

            has_span = i != last_idx and item["start"] is not None and item["end"] is not None
            align_duration = (item["end"] - item["start"]) if has_span else None

            response_data.append({
                "expected_char": ref_char,
                "spoken_char": spoken_char,
                "expected_pinyin": match_info["expected_pinyin"],
                "spoken_pinyin": match_info["spoken_pinyin"],
                "is_syllable_correct": match_info["is_syllable_correct"],

                "expected_tone": expected_tone,
                "detected_tone": tone_result["detected_tone"],
                
                # Fix #1: Keep the raw result separate from the reconciled result
                "raw_is_correct_tone": raw_is_correct_tone,
                "is_correct_tone": raw_is_correct_tone,
                "reconciled": False,  # Whether this result was reconciled
                
                "tone_desc": tone_result["desc"],
                "tone_verified": tone_result.get("tone_verified", False),
                "low_confidence": core.is_low_confidence(ft, align_duration),

                "start": item["start"],
                "end": item["end"],
                "timestamp_interpolated": item["interpolated"],
                "pitch": {
                    "status": "ok" if ft is not None else ("unvoiced" if window else "no_timestamp")
                }
            })

        # 5. GROUP ASSESSMENTS BY CLAUSE AND RECONCILE (CLAUSE-LEVEL ASSESSMENT & RECONCILIATION)
        clause_results = []
        char_pointer = 0

        for clause in clauses:
            clause_len = len(clause)
            clause_items = response_data[char_pointer : char_pointer + clause_len]
            char_pointer += clause_len
            
            clause_metrics = core.reconcile_clause_assessment(clause_items)
            for item in clause_items:
                if item["reconciled"]:
                    item["tone_desc"] += " (Passed at clause level; low-confidence error reconciled)"

            clause_results.append({
                "clause_text": clause,
                "accuracy": round(clause_metrics["accuracy"], 2) if clause_metrics["accuracy"] is not None else None,
                "raw_accuracy": round(clause_metrics["raw_accuracy"], 2) if clause_metrics["raw_accuracy"] is not None else None,
                "is_passed": clause_metrics["is_passed"],
                "n_assessable": clause_metrics["n_assessable"],
                "n_reconciled": clause_metrics["n_reconciled"],
                "details": clause_items
            })

        return {
            "success": True,
            "mode": "guided_practice",
            "target_script": target_script,
            "data": response_data,
            "clauses": clause_results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
