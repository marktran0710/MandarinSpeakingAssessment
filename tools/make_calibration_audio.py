"""Build the five reference sounds of the voice-calibration dialog (assets/calibration/ma1.wav ... ma5.wav).

The voice is the teacher's own: the syllable 媽 (mā) of lesson 5-2-1 ("媽，我跟友美要去喝下午茶..."). Its pitch is replaced, with Praat's
overlap-add resynthesis, by the textbook contour of each tone (Chao tone letters 55 / 35 / 214 / 51 and a short light tone), and the vowel is
drawn out a little so each sound is easy to recognise. Nothing else changes: same timbre, same speaker.

    python tools/make_calibration_audio.py [--check]

Source span (from the MFA alignment of that recording): the syllable runs 0.110-0.570 s, "m" until SOURCE_M_END. --check prints the measured
contour of every output so a wrong shape is caught without listening.
"""
import sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll\5-2-01.mp3")
SOURCE_START, SOURCE_END, SOURCE_M_END = 0.110, 0.570, 0.190
OUT = ROOT / "assets" / "calibration"
REFERENCE_HZ = 206.0                       # the teacher's median pitch in the source syllable
M_SEC = 0.09                               # the "m" keeps about this length in every sound

# tone -> (total seconds, loudness, [(position 0..1 along the vowel, semitones re. REFERENCE_HZ), ...])
TONES = {
    1: (0.62, 1.0, [(0.0, 4.0), (0.5, 4.5), (1.0, 4.5)]),                                   # 55  high, level
    2: (0.62, 1.0, [(0.0, 0.5), (0.18, 0.0), (0.6, 2.5), (1.0, 5.5)]),                      # 35  rising
    3: (0.78, 1.0, [(0.0, -2.0), (0.35, -5.5), (0.62, -5.0), (1.0, 3.0)]),                  # 214 dipping, ends rising
    4: (0.56, 1.0, [(0.0, 5.5), (0.45, 1.5), (1.0, -5.5)]),                                 # 51  falling
    5: (0.30, 0.55, [(0.0, -0.5), (1.0, -2.5)]),                                            # light, short, soft
}


def build(tone):
    total, loudness, points = TONES[tone]
    sound = parselmouth.Sound(str(SOURCE))
    part = sound.extract_part(SOURCE_START - 0.02, SOURCE_END + 0.02, preserve_times=False)
    t_m, t_end = SOURCE_M_END - (SOURCE_START - 0.02), part.duration
    manipulation = call(part, "To Manipulation", 0.01, 70, 450)

    pitch = call("Create PitchTier", "tone", 0, part.duration)
    call(pitch, "Add point", 0.0, REFERENCE_HZ * 2 ** (points[0][1] / 12))
    for position, st in points:
        call(pitch, "Add point", t_m + position * (t_end - t_m) * 0.98, REFERENCE_HZ * 2 ** (st / 12))
    call(pitch, "Add point", part.duration, REFERENCE_HZ * 2 ** (points[-1][1] / 12))
    call([pitch, manipulation], "Replace pitch tier")

    vowel_factor = (total - M_SEC) / (t_end - t_m)
    duration = call("Create DurationTier", "length", 0, part.duration)
    call(duration, "Add point", 0.0, M_SEC / t_m)
    call(duration, "Add point", t_m - 0.005, M_SEC / t_m)
    call(duration, "Add point", t_m + 0.005, vowel_factor)
    call(duration, "Add point", part.duration, vowel_factor)
    call([duration, manipulation], "Replace duration tier")

    out = call(manipulation, "Get resynthesis (overlap-add)")
    out.scale_peak(0.7 * loudness)
    out = call(out, "Resample", 24000, 50)
    return out


def contour(sound):
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=500)
    t, f0 = pitch.xs(), pitch.selected_array["frequency"]
    voiced = f0 > 0
    st = 12 * np.log2(f0[voiced] / REFERENCE_HZ)
    return t[voiced], st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tone in TONES:
        sound = build(tone)
        sound.save(str(OUT / f"ma{tone}.wav"), "WAV")
        if "--check" in sys.argv:
            t, st = contour(sound)
            pts = np.interp(np.linspace(t[0], t[-1], 6), t, st)
            print(f"ma{tone}: {sound.duration:.2f} s, voiced {t[-1] - t[0]:.2f} s, contour (st re. 206 Hz): " + " ".join(f"{p:5.1f}" for p in pts))
    print("written to", OUT)


if __name__ == "__main__":
    main()
