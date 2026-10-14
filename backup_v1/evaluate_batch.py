import os
import pandas as pd
import requests

FOLDER_PATH = r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll"
CSV_PATH = os.path.join(FOLDER_PATH, "FULL_audio_content_audit.csv")
API_URL = "http://127.0.0.1:8000/api/assess-pronunciation"

def run_csv_batch_evaluation():
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Read {len(df)} sample rows from the CSV.")
    print(f"Audio folder: {FOLDER_PATH}\n")
    print("=" * 70)

    total_files = 0
    total_chars_evaluated = 0
    total_chars_correct = 0
    summary_reports = []

    for idx, row in df.iterrows():
        filename = str(row["filename"]).strip()
        script_text = str(row["script"]).strip()
        audio_path = os.path.join(FOLDER_PATH, filename)

        if not os.path.exists(audio_path):
            print(f"[{idx+1}/{len(df)}] Audio file not found: {filename} -> Skipping.")
            continue

        total_files += 1
        print(f"[{idx+1}/{len(df)}] Checking: {filename}")
        print(f"   📜 Script: {script_text}")

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (filename, f, "audio/mpeg")}
                data = {"script": script_text}
                response = requests.post(API_URL, files=files, data=data)

            if response.status_code == 200:
                res_json = response.json()
                chars_data = res_json.get("data", [])
                
                file_total = 0
                file_correct = 0

                for item in chars_data:
                    char = item.get("expected_char")
                    if not char:
                        continue
                    
                    file_total += 1
                    expected_tone = item.get("expected_tone")
                    
                    # Read detected_tone and tone_desc from the reconciled item
                    detected_tone = item.get("detected_tone")
                    tone_desc = item.get("tone_desc")
                    is_correct = item.get("is_correct_tone")

                    if is_correct:
                        file_correct += 1
                        status_str = "CORRECT"
                    else:
                        status_str = "❌ SAI"

                    print(f"      - Character [{char}] | Expected: Tone {expected_tone} -> Detected: Tone {detected_tone} ({tone_desc}) {status_str}")

                total_chars_evaluated += file_total
                total_chars_correct += file_correct
                
                file_acc = (file_correct / file_total * 100) if file_total > 0 else 0.0
                summary_reports.append({
                    "filename": filename,
                    "total": file_total,
                    "correct": file_correct,
                    "accuracy": file_acc
                })
                print(f"Results for {filename}: {file_correct}/{file_total} characters correct ({file_acc:.1f}%)")
            else:
                print(f"API error ({response.status_code}): {response.text}")

        except Exception as e:
            print(f"Processing error: {str(e)}")
        
        print("-" * 70)

    # Print the overall summary report
    print("\n" + "=" * 25 + " FULL DATASET SUMMARY " + "=" * 25)
    print(f"Total audio files checked: {total_files}")
    print(f"Total Han characters evaluated: {total_chars_evaluated}")
    print(f"Total characters with correct tones: {total_chars_correct}")
    
    overall_accuracy = (total_chars_correct / total_chars_evaluated * 100) if total_chars_evaluated > 0 else 0.0
    print(f"OVERALL ACCURACY: {overall_accuracy:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_csv_batch_evaluation()