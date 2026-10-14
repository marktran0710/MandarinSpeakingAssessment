import os
import pandas as pd
import numpy as np
import parselmouth
from scipy.signal import medfilt
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
import opencc
import joblib
import re
import torch
import whisperx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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

def extract_features_from_chunk(chunk, utterance_semitones):
    try:
        if len(chunk) < 25:  
            return None
            
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
            return None
            
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
        return [start_v, mid_v, end_v, slope]
    except Exception:
        return None

def train_tone_classifier():
    csv_path =  r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll\FULL_audio_content_audit.csv"
    audio_dir = r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll"

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    converter = opencc.OpenCC('s2twp')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading the WhisperX model on device: {device.upper()} to align training data...")
    model_size = "small"
    compute_type = "int8" if device == "cpu" else "float16"
    whisper_model = whisperx.load_model(model_size, device, compute_type=compute_type)
    align_model, metadata = whisperx.load_align_model(language_code="zh", device=device)

    X = []
    y = []

    print("Scanning audio files and using WhisperX alignment to extract features...")
    for idx, row in df.iterrows():
        filename = row['filename']
        script = str(row['script'])
        file_path = os.path.join(audio_dir, filename)
        
        if not os.path.exists(file_path):
            continue

        try:
            full_audio_pydub = AudioSegment.from_file(file_path)
            audio_duration_ms = len(full_audio_pydub)
            
            samples_full = np.array(full_audio_pydub.get_array_of_samples()).astype(np.float32)
            if full_audio_pydub.sample_width == 2: samples_full = samples_full / 32768.0
            elif full_audio_pydub.sample_width == 4: samples_full = samples_full / 2147483648.0
            if full_audio_pydub.channels == 2: samples_full = samples_full.reshape((-1, 2)).mean(axis=1)
            sound_full = parselmouth.Sound(samples_full, sampling_frequency=full_audio_pydub.frame_rate)
            pitch_vals = sound_full.to_pitch().selected_array['frequency']
            utterance_semitones = 12 * np.log2(pitch_vals[pitch_vals > 0] / 16.35) if len(pitch_vals[pitch_vals > 0]) > 0 else None

            audio_wx = whisperx.load_audio(file_path)
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
                        if duration < 0.08:
                            end_sec = min(audio_duration_ms / 1000.0, start_sec + 0.08)
                        start_ms = max(0, int(start_sec * 1000) - 50)
                        end_ms = min(audio_duration_ms, int(end_sec * 1000) + 50)
                        chunk = full_audio_pydub[start_ms:end_ms]
                        features_data = extract_features_from_chunk(chunk, utterance_semitones)
                    else:
                        features_data = None

                    raw_student_chars.append({
                        "char": char_trad,
                        "features": features_data
                    })

            chinese_items = [item for item in raw_student_chars if is_chinese_char(item["char"])]
            target_script = converter.convert(script.strip())
            target_chars_only = [c for c in target_script if is_chinese_char(c)]
            expected_tones = get_expected_tones("".join(target_chars_only))

            for i, ref_char in enumerate(target_chars_only):
                tone = expected_tones[i] if i < len(expected_tones) else None
                if tone is None: 
                    continue
                if i < len(chinese_items) and chinese_items[i]["features"] is not None:
                    X.append(chinese_items[i]["features"])
                    y.append(tone)
        except Exception as e:
            continue

    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        
        # Split the data into 80/20 train/test sets for the report
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        clf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_eval.fit(X_train, y_train)
        y_pred = clf_eval.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\nMODEL EVALUATION ON TEST SET (20%):")
        print(f"- Accuracy: {acc * 100:.2f}%")
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred))
        
        # Retrain on 100% of the data and save the optimized model for the API
        clf_full = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_full.fit(X, y)
        joblib.dump(clf_full, "tone_model.joblib")
        print(f"Retrained on all {len(X)} samples and successfully saved to 'tone_model.joblib'.")
    else:
        print("No data could be extracted for training.")

if __name__ == "__main__":
    train_tone_classifier()