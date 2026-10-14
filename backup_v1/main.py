import os
import shutil
import tempfile
import re
import numpy as np
import parselmouth
from scipy.signal import medfilt
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
import opencc
import joblib
import torch
import whisperx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

# Initialize the FastAPI app
app = FastAPI(title="Taiwan Mandarin CAPT API")

# Load global models (ensure tone_model.joblib exists)
ml_models = {}
if os.path.exists("tone_model.joblib"):
    ml_models["tone_classifier"] = joblib.load("tone_model.joblib")
    print("Successfully loaded tone_model.joblib!")
else:
    print("Warning: tone_model.joblib was not found!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading the WhisperX model on device: {device.upper()}...")
whisper_model = whisperx.load_model("small", device, compute_type="int8" if device == "cpu" else "float16")
align_model, metadata = whisperx.load_align_model(language_code="zh", device=device)

def is_chinese_char(char):
    return bool(re.match(r'^[\u4e00-\u9fa5]$', char))

def get_expected_tones(script_text):
    pinyin_results = lazy_pinyin(script_text, style=Style.TONE3, tone_sandhi=True, neutral_tone_with_five=True)
    tones = []
    for p_str in pinyin_results:
        if p_str and p_str[-1].isdigit():
            tones.append(int(p_str[-1]))
        else:
            tones.append(None)
    return tones

def check_pronunciation_match(expected_char, spoken_char):
    exp_pinyin = lazy_pinyin(expected_char, style=Style.NORMAL)[0] if expected_char else ""
    spk_pinyin = lazy_pinyin(spoken_char, style=Style.NORMAL)[0] if spoken_char else ""
    is_pinyin_match = (exp_pinyin == spk_pinyin)
    return {
        "expected_pinyin": exp_pinyin,
        "spoken_pinyin": spk_pinyin,
        "is_syllable_correct": is_pinyin_match
    }

MIN_CHAR_DURATION_SEC = 0.08
DEFAULT_CHAR_DURATION_SEC = 0.3   # Assumed duration for a final character with no timestamp
UNVOICED_ENERGY_RATIO = 0.1       # Chunk RMS / utterance RMS: at or above this threshold, treat it as speech

def slice_chunk(audio, start_sec, end_sec):
    """Cut one character's audio segment and pad it by 50 ms on both sides."""
    start_ms = max(0, int(start_sec * 1000) - 50)
    end_ms = min(len(audio), int(end_sec * 1000) + 50)
    return audio[start_ms:end_ms]

def fill_missing_timestamps(items, full_audio):
    """
    Interpolate timestamps for characters WhisperX could not align (start/end is None or end <= start):
    divide the gap between the nearest aligned characters, then trim the audio chunk.
    """
    audio_end = len(full_audio) / 1000.0

    def has_timestamp(item):
        return item["start"] is not None and item["end"] is not None and item["end"] > item["start"]

    i = 0
    while i < len(items):
        if has_timestamp(items[i]):
            i += 1
            continue

        # Group consecutive characters with missing timestamps into a span [i, j)
        j = i
        while j < len(items) and not has_timestamp(items[j]):
            j += 1
        run = j - i

        gap_start = items[i - 1]["end"] if i > 0 else 0.0
        gap_end = items[j]["start"] if j < len(items) else gap_start + DEFAULT_CHAR_DURATION_SEC * run
        gap_end = min(audio_end, max(gap_end, gap_start + MIN_CHAR_DURATION_SEC * run))

        if gap_end > gap_start:
            step = (gap_end - gap_start) / run
            for k in range(run):
                item = items[i + k]
                item["start"] = gap_start + k * step
                item["end"] = gap_start + (k + 1) * step
                item["chunk"] = slice_chunk(full_audio, item["start"], item["end"])
                item["interpolated"] = True
        i = j

def extract_features_from_chunk(chunk, utterance_semitones, utterance_rms=None):
    try:
        if len(chunk) < 25:  
            return {"status": "too_short", "features": None}
            
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
        if chunk.sample_width == 2:
            samples = samples / 32768.0
        elif chunk.sample_width == 4:
            samples = samples / 2147483648.0
        if chunk.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
            
        sound = parselmouth.Sound(samples, sampling_frequency=chunk.frame_rate)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        
        if len(pitch_values) == 0:
            rms_ratio = (chunk.rms / utterance_rms) if utterance_rms else 0.0
            return {"status": "unvoiced", "features": None, "rms_ratio": rms_ratio}
            
        semitones = 12 * np.log2(pitch_values / 16.35)
        if len(semitones) >= 3:
            semitones = medfilt(semitones, kernel_size=5 if len(semitones) >= 5 else 3)
            
        local_mean = float(np.mean(semitones))
        local_std = float(np.std(utterance_semitones)) if utterance_semitones is not None else 1.0
        if local_std == 0: local_std = 1.0
        
        normalized_s = (semitones - local_mean) / local_std
        third = len(normalized_s) // 3
        if third == 0:
            val = float(np.mean(normalized_s))
            start_v, mid_v, end_v = val, val, val
        else:
            start_v = float(np.mean(normalized_s[:third]))
            mid_v = float(np.mean(normalized_s[third:2*third]))
            end_v = float(np.mean(normalized_s[2*third:]))
            
        slope = end_v - start_v
        return {"status": "ok", "features": [start_v, mid_v, end_v, slope]}
    except Exception:
        return {"status": "error", "features": None}

NEUTRAL_TONE_CHARS = {"的", "嗎", "呢", "吧", "啊", "呀", "啦", "麼", "們"}

def classify_tone_with_ml(pitch_result, char=""):
    if char in NEUTRAL_TONE_CHARS:
        return {"detected_tone": 5, "desc": "Tone 5 (Neutral Tone)"}

    # F0 cannot be measured, but the chunk has energy: the learner spoke, but the sound is unvoiced or hoarse.
    # Do not guess the tone or mark the expected tone as correct; mark it as unscored.
    if pitch_result.get("status") == "unvoiced" and pitch_result.get("rms_ratio", 0.0) >= UNVOICED_ENERGY_RATIO:
        return {
            "detected_tone": None,
            "assessable": False,
            "desc": "Speech detected, but F0 cannot be measured (unvoiced); tone is not scored"
        }

    if pitch_result.get("status") != "ok" or pitch_result.get("features") is None:
        return {"detected_tone": None, "desc": f"Unvoiced or Invalid ({pitch_result.get('status')})"}
        
    clf = ml_models.get("tone_classifier")
    features = np.array([pitch_result["features"]])
    
    if clf is not None:
        predicted_tone = int(clf.predict(features)[0])
        descriptions = {
            1: "Tone 1 (ML - Level)",
            2: "Tone 2 (ML - Rising)",
            3: "Tone 3 (ML - Dipping)",
            4: "Tone 4 (ML - Falling)",
            5: "Tone 5 (Neutral)"
        }
        return {
            "detected_tone": predicted_tone,
            "desc": descriptions.get(predicted_tone, "ML Model Match")
        }
    else:
        return {"detected_tone": 1, "desc": "Fallback Default"}

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
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")

    # Save the file temporarily for processing
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    converter = opencc.OpenCC('s2twp')

    try:
        # 1. Read the full audio with pydub
        full_audio_pydub = AudioSegment.from_file(temp_path)
        audio_duration_ms = len(full_audio_pydub)
        utterance_rms = full_audio_pydub.rms

        # Extract the utterance semitone curve for local normalization
        samples_full = np.array(full_audio_pydub.get_array_of_samples()).astype(np.float32)
        if full_audio_pydub.sample_width == 2: samples_full = samples_full / 32768.0
        elif full_audio_pydub.sample_width == 4: samples_full = samples_full / 2147483648.0
        if full_audio_pydub.channels == 2: samples_full = samples_full.reshape((-1, 2)).mean(axis=1)
        sound_full = parselmouth.Sound(samples_full, sampling_frequency=full_audio_pydub.frame_rate)
        pitch_vals = sound_full.to_pitch().selected_array['frequency']
        utterance_semitones = 12 * np.log2(pitch_vals[pitch_vals > 0] / 16.35) if len(pitch_vals[pitch_vals > 0]) > 0 else None

        # 2. WhisperX alignment
        audio_wx = whisperx.load_audio(temp_path)
        result = whisper_model.transcribe(audio_wx, batch_size=4, language="zh")
        result = whisperx.align(result["segments"], align_model, metadata, audio_wx, device, return_char_alignments=True)

        raw_student_chars = []
        for segment in result["segments"]:
            for item in segment.get("words", []):
                raw_text = item.get("word", "")
                char_trad = converter.convert(raw_text)
                start_sec = item.get("start")
                end_sec = item.get("end")
                
                if start_sec is not None and end_sec is not None and end_sec > start_sec:
                    duration = end_sec - start_sec
                    if duration < MIN_CHAR_DURATION_SEC:
                        end_sec = min(audio_duration_ms / 1000.0, start_sec + MIN_CHAR_DURATION_SEC)
                    chunk = slice_chunk(full_audio_pydub, start_sec, end_sec)
                else:
                    chunk = None

                raw_student_chars.append({
                    "char": char_trad,
                    "chunk": chunk,
                    "start": start_sec,
                    "end": end_sec,
                    "interpolated": False
                })

        chinese_items = [item for item in raw_student_chars if is_chinese_char(item["char"])]
        # Fill timestamps for characters WhisperX could not align to avoid a missing chunk shifting the sequence
        fill_missing_timestamps(chinese_items, full_audio_pydub)
        target_script = converter.convert(script.strip())
        target_chars_only = [c for c in target_script if is_chinese_char(c)]
        expected_tones = get_expected_tones("".join(target_chars_only))

        response_data = []

        for i, ref_char in enumerate(target_chars_only):
            expected_tone = expected_tones[i] if i < len(expected_tones) else None
            
            if i < len(chinese_items):
                spk_item = chinese_items[i]
                spoken_char = spk_item.get("char", "")
                chunk = spk_item.get("chunk")
                
                if chunk is not None and len(chunk) > 0:
                    pitch_result = extract_features_from_chunk(chunk, utterance_semitones, utterance_rms)
                else:
                    pitch_result = {"status": "no_chunk", "features": None}

                tone_result = classify_tone_with_ml(pitch_result, ref_char)
                start_time = spk_item.get("start")
                end_time = spk_item.get("end")
                interpolated = spk_item.get("interpolated", False)
            else:
                spoken_char = ""
                pitch_result = {"status": "missing", "features": None}
                tone_result = {"detected_tone": None, "desc": "Missing chunk"}
                start_time = None
                end_time = None
                interpolated = False

            # If speech is detected but F0 cannot be measured, do not score the tone (None)
            if not tone_result.get("assessable", True):
                is_correct_tone = None
            else:
                is_correct_tone = (tone_result["detected_tone"] == expected_tone if expected_tone is not None else False)

            # Check whether the Pinyin syllables match
            match_info = check_pronunciation_match(ref_char, spoken_char)

            response_data.append({
                "expected_char": ref_char,
                "spoken_char": spoken_char,
                "expected_pinyin": match_info["expected_pinyin"],
                "spoken_pinyin": match_info["spoken_pinyin"],
                "is_syllable_correct": match_info["is_syllable_correct"],
                
                "expected_tone": expected_tone,
                "detected_tone": tone_result["detected_tone"],
                "is_correct_tone": is_correct_tone,
                "tone_desc": tone_result["desc"],

                "start": start_time,
                "end": end_time,
                "timestamp_interpolated": interpolated,
                "pitch": {
                    "status": pitch_result["status"]
                }
            })

        return {
            "success": True,
            "mode": "guided_practice",
            "target_script": target_script,
            "data": response_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)