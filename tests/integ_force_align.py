"""Integration test (loads the real wav2vec2 aligner, ~1 min): run manually, not part of unittest discover."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opencc
import whisperx
import capt_core as core

FOLDER = r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll"
CASES = {
    "5-2-01.mp3": "媽，我跟友美要去喝下午茶，可是我的錢包不在房間裡。",   # Whisper used to drop the leading 媽
    "6-2-02.mp3": "我覺得有一點兒累，我想回家休息。",                      # has 兒 that is not pronounced
}

device = "cpu"
align_model, meta = whisperx.load_align_model(language_code="zh", device=device)
t2s = opencc.OpenCC("t2s")
fails = []

for name, script in CASES.items():
    audio = whisperx.load_audio(os.path.join(FOLDER, name))
    dur = len(audio) / 16000.0
    chars = list("".join(core.prepare_target(script)))
    items = core.force_align_script(audio, chars, align_model, meta, device, t2s.convert)

    def check(label, cond, info=""):
        print(("PASS " if cond else "FAIL ") + f"{name}: {label}" + (f"  {info}" if not cond else ""))
        if not cond:
            fails.append((name, label))

    check("one item per script char", len(items) == len(chars), f"{len(items)} vs {len(chars)}")
    check("chars preserved in order", [i["char"] for i in items] == chars)
    check("every char has start<end", all(i["start"] is not None and i["end"] is not None and i["end"] > i["start"] for i in items))
    check("times are non-decreasing", all(a["start"] <= b["start"] for a, b in zip(items, items[1:])))
    check("inside the audio", items[0]["start"] >= 0 and items[-1]["end"] <= dur + 0.05, f"dur={dur:.2f} last={items[-1]['end']}")
    print("      ", " ".join(f"{i['char']}{i['start']:.2f}-{i['end']:.2f}" for i in items))

print("\nFAILED:" if fails else "\nALL PASSED", fails if fails else "")
sys.exit(1 if fails else 0)
