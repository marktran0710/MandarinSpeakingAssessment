"""
Logic shared by the API (main.py) and training (train_model.py), kept separate so both sides
process scripts, tone labels, and timestamps identically, and can be tested without loading a model.
"""
import os
import re
from difflib import SequenceMatcher

import numpy as np
import parselmouth
from pypinyin import lazy_pinyin, Style
from scipy.signal import medfilt

# These characters are handled by the API's explicit neutral-tone rule and do
# not need to contribute neutral-tone examples to the acoustic classifier.
NEUTRAL_TONE_CHARS = {"的", "嗎", "呢", "吧", "啊", "呀", "啦", "麼", "們"}

MIN_CHAR_DURATION_SEC = 0.08
DEFAULT_CHAR_DURATION_SEC = 0.3     # Assumed duration for a final character with no timestamp
LOW_CONF_ALIGN_SEC = 0.15           # At or below this alignment, tone accuracy is ~40% (about 8% at <= 0.085s) on held-out data
LOW_CONF_VOICED_SEC = 0.10          # Below this voiced-window duration, the F0 contour is too short to classify reliably
CLAUSE_PASS_THRESHOLD = 70.0

# ---------------------------------------------------------------- script ----

def is_chinese_char(char):
    return bool(re.match(r'^[一-龥]$', char))

# Erhua is a syllable in these words; elsewhere it is a retroflex suffix with no separate syllable.
_ER_KEEP_BEFORE = set("女嬰孤幼孩男少獨")
_ER_KEEP_AFTER = set("子童女科歌媳孫")

def strip_erhua(text):
    out = []
    for i, ch in enumerate(text):
        if ch == "兒":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if is_chinese_char(prev) and prev not in _ER_KEEP_BEFORE and nxt not in _ER_KEEP_AFTER:
                continue
        out.append(ch)
    return "".join(out)

def prepare_target(script):
    """Convert a script to clauses containing only Han characters, remove erhua, and split at punctuation."""
    clauses = re.split(r'[^一-龥]+', strip_erhua(script))
    return [c for c in clauses if c]

# ----------------------------------------------------------------- tones ----

# Taiwan Mandarin readings differ from mainland Pinyin readings.
TAIWAN_WORD_TONES = {
    "星期": [1, 2],
    "垃圾": [4, 4],
    "覺得": [2, 5],
    "衣服": [1, 5],
}
# These compounds are often followed by a pause: keep the final syllable at Tone 3 instead of applying sandhi to the next character.
PHRASE_BREAK_AFTER = ["所以", "哪裡"]
_NUMERAL_CHARS = set("第零二三四五六七八九十兩")

def _base_tones(clause):
    pinyin = lazy_pinyin(clause, style=Style.TONE3, neutral_tone_with_five=True)
    if len(pinyin) != len(clause):
        pinyin = [lazy_pinyin(c, style=Style.TONE3, neutral_tone_with_five=True)[0] for c in clause]
    tones = [int(p[-1]) if p and p[-1].isdigit() else None for p in pinyin]
    for word, word_tones in TAIWAN_WORD_TONES.items():
        pos = clause.find(word)
        while pos != -1:
            tones[pos:pos + len(word)] = word_tones
            pos = clause.find(word, pos + len(word))
    return tones

def _clause_tones(clause):
    base = _base_tones(clause)
    tones = list(base)
    n = len(clause)
    break_after = set()
    for word in PHRASE_BREAK_AFTER:
        pos = clause.find(word)
        while pos != -1:
            break_after.add(pos + len(word) - 1)
            pos = clause.find(word, pos + len(word))
    for i, ch in enumerate(clause):
        nxt = base[i + 1] if i + 1 < n else None
        prev_ch = clause[i - 1] if i > 0 else ""
        next_ch = clause[i + 1] if i + 1 < n else ""
        if ch == "不":
            if prev_ch and prev_ch == next_ch:       # V不V: 好不好, 是不是
                tones[i] = 5
            else:
                tones[i] = 2 if nxt == 4 else 4
        elif ch == "一":
            if nxt is None or prev_ch in _NUMERAL_CHARS or next_ch in _NUMERAL_CHARS:
                tones[i] = 1                          # Counting, ordinal use, or clause-final position
            else:
                tones[i] = 2 if nxt == 4 else 4
        elif base[i] == 3 and nxt == 3 and i not in break_after:   # 3-3 -> 2-3 (3-3-3 -> 2-2-3)
            tones[i] = 2
    return tones

def expected_tones(clauses):
    """Return the expected tone for each character after sandhi; sandhi does not cross punctuation."""
    result = []
    for clause in clauses:
        result.extend(_clause_tones(clause))
    return result

# ------------------------------------------------------- spoken-text match ----

def _toneless_pinyin(chars):
    text = "".join(chars)
    pinyin = lazy_pinyin(text, style=Style.NORMAL)
    if len(pinyin) != len(chars):
        pinyin = [lazy_pinyin(c, style=Style.NORMAL)[0] for c in chars]
    return pinyin

def match_spoken_chars(target_text, spoken_text):
    """
    Match each script character with the character recognized by Whisper (None if omitted), using
    tone-free Pinyin and diff so Whisper insertions or omissions do not shift later matches.
    """
    target = [c for c in target_text if is_chinese_char(c)]
    spoken = [c for c in strip_erhua(spoken_text) if is_chinese_char(c)]
    result = [None] * len(target)
    if not target or not spoken:
        return result
    matcher = SequenceMatcher(None, _toneless_pinyin(target), _toneless_pinyin(spoken), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                result[i1 + k] = spoken[j1 + k]
    return result

# ------------------------------------------------------------- timestamps ----

def _has_timestamp(item):
    return item["start"] is not None and item["end"] is not None and item["end"] > item["start"]


def alignment_cache_is_current(cache_entry, script, audio_path, cache_path):
    """Return whether a saved alignment matches both the target text and audio file.

    New entries carry the source audio's size and nanosecond mtime. For legacy
    entries, the cache file must be at least as new as the audio file.
    """
    if not isinstance(cache_entry, dict) or cache_entry.get("script") != script:
        return False
    try:
        audio_stat = os.stat(audio_path)
        cache_stat = os.stat(cache_path)
    except OSError:
        return False

    try:
        if "audio_mtime_ns" in cache_entry:
            if int(cache_entry["audio_mtime_ns"]) != audio_stat.st_mtime_ns:
                return False
        elif audio_stat.st_mtime_ns > cache_stat.st_mtime_ns:
            return False
        if "audio_size_bytes" in cache_entry:
            if int(cache_entry["audio_size_bytes"]) != audio_stat.st_size:
                return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True

def fill_missing_timestamps(items, audio_duration_sec):
    """
    Interpolate timestamps for unaligned characters (start/end is None or end <= start) by dividing the gap
    between the nearest aligned characters. Updates `items` in place and sets item["interpolated"] = True.
    """
    i = 0
    while i < len(items):
        if _has_timestamp(items[i]):
            i += 1
            continue
        j = i
        while j < len(items) and not _has_timestamp(items[j]):
            j += 1
        run = j - i

        gap_start = items[i - 1]["end"] if i > 0 else 0.0
        gap_end = items[j]["start"] if j < len(items) else gap_start + DEFAULT_CHAR_DURATION_SEC * run
        gap_end = min(audio_duration_sec, max(gap_end, gap_start + MIN_CHAR_DURATION_SEC * run))

        if gap_end > gap_start:
            step = (gap_end - gap_start) / run
            for k in range(run):
                items[i + k]["start"] = gap_start + k * step
                items[i + k]["end"] = gap_start + (k + 1) * step
                items[i + k]["interpolated"] = True
        i = j

def force_align_script(audio, chars, align_model, metadata, device, to_alignment_script):
    """
    Force-align the known script to the audio without using Whisper's transcript, so Whisper omissions or
    substitutions do not shift timestamps. `to_alignment_script` converts characters to the alignment model's
    dictionary form (simplified, one character -> one character). Returns one item per character in `chars`; unaligned items have start/end = None.
    """
    import whisperx

    items = [{"char": c, "start": None, "end": None, "score": None, "interpolated": False} for c in chars]
    if not chars:
        return items

    text = "".join(to_alignment_script(c) for c in chars)
    duration = len(audio) / 16000.0
    result = whisperx.align([{"text": text, "start": 0.0, "end": duration}], align_model, metadata,
                            audio, device, return_char_alignments=True)
    aligned = [c for segment in result["segments"] for c in (segment.get("chars") or [])]
    if len(aligned) != len(chars):
        return items
    for item, a in zip(items, aligned):
        item["start"], item["end"], item["score"] = a.get("start"), a.get("end"), a.get("score")
    return items

# ------------------------------------------------------------ pitch features ----
# These constants were selected with cross-validation grouped by conversation (see train_model.py).
# A character's CTC timestamp marks its peak, not the syllable boundary, so the F0 window is placed
# around that peak: it starts slightly earlier and extends to the next character's peak.
WINDOW_ALPHA = 0.3
WINDOW_BETA = 1.0
WINDOW_PRE_MAX = 0.2    # Seconds; prevent the window from reaching the previous character across a long pause
WINDOW_POST_MAX = 0.5

# Feature indices in the vector (F_*), followed by CONTEXT_DIM contextual features
F_START, F_MID, F_END, F_SLOPE, F_LEVEL, F_CURVATURE, F_VOICED_DUR, F_VOICED_FRAC, F_RANGE = range(9)
CONTOUR_POINTS = 5
CONTEXT_DIM = 5
FEATURE_DIM = 9 + CONTOUR_POINTS + CONTEXT_DIM

def _semitones(f0):
    return 12 * np.log2(f0 / 16.35)

def pitch_track(samples, sample_rate):
    """
    Return the utterance F0 contour (semitones, NaN for unvoiced frames) and voiced-frame mean/std for speaker normalization.
    Returns (times, semitones, mean, std).
    """
    pitch = parselmouth.Sound(np.asarray(samples, dtype=np.float64), sampling_frequency=sample_rate).to_pitch()
    times, f0 = pitch.xs(), pitch.selected_array['frequency']
    sem = np.full(len(f0), np.nan)
    voiced = f0 > 0
    sem[voiced] = _semitones(f0[voiced])

    i = 0
    while i < len(sem):                      # Apply a median filter within each continuous voiced region
        if np.isnan(sem[i]):
            i += 1
            continue
        j = i
        while j < len(sem) and not np.isnan(sem[j]):
            j += 1
        if j - i >= 3:
            sem[i:j] = medfilt(sem[i:j], kernel_size=5 if j - i >= 5 else 3)
        i = j

    values = sem[~np.isnan(sem)]
    if len(values) == 0:
        return times, sem, 0.0, 1.0
    return times, sem, float(values.mean()), float(values.std()) or 1.0

def char_windows(starts):
    """Return the per-character F0 window (w0, w1) from the complete, ordered CTC peak timestamps in `starts`."""
    starts = np.asarray(starts, dtype=float)
    gaps = np.diff(starts)
    median_gap = float(np.median(gaps)) if len(gaps) else DEFAULT_CHAR_DURATION_SEC
    next_gap = np.append(gaps, median_gap)
    prev_gap = np.insert(gaps, 0, median_gap)
    return [
        (float(s - min(WINDOW_ALPHA * gp, WINDOW_PRE_MAX)), float(s + min(WINDOW_BETA * gn, WINDOW_POST_MAX)))
        for s, gp, gn in zip(starts, prev_gap, next_gap)
    ]

def context_features(tones, clause_lengths, idx):
    """Clause/utterance start or end plus expected tones of adjacent characters within the clause."""
    clause_ends, clause_starts, pos = set(), set(), 0
    for length in clause_lengths:
        clause_starts.add(pos)
        pos += length
        clause_ends.add(pos - 1)
    prev_tone = tones[idx - 1] if idx > 0 and idx not in clause_starts else 0
    next_tone = tones[idx + 1] if idx + 1 < len(tones) and idx not in clause_ends else 0
    return [float(idx in clause_ends), float(idx == len(tones) - 1), float(idx in clause_starts),
            float(prev_tone or 0), float(next_tone or 0)]

def char_features(track, w0, w1, context):
    """Return one character's feature vector in [w0, w1], or None if the window has no F0."""
    times, sem, mean, std = track
    window = sem[(times >= w0) & (times <= w1)]
    voiced = window[~np.isnan(window)]
    if len(voiced) < 2:
        return None
    z = (voiced - mean) / std
    k = len(z) // 3
    if k == 0:
        start = mid = end = float(z.mean())
    else:
        start, mid, end = float(z[:k].mean()), float(z[k:2 * k].mean()), float(z[2 * k:].mean())
    contour = np.interp(np.linspace(0, len(z) - 1, CONTOUR_POINTS), np.arange(len(z)), z)
    frame_sec = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    return ([start, mid, end, end - start, float(z.mean()), mid - (start + end) / 2,
             len(z) * frame_sec, len(voiced) / len(window), float(z.max() - z.min())]
            + [float(c) for c in contour] + list(context))

def utterance_features(samples, sample_rate, starts, tones, clause_lengths, return_pitch_stats=False):
    """
    Compute utterance-level features from `starts` (seconds, None if a character has no timestamp):
    return feature vectors and windows; characters without timestamps or window F0 get None. Shared by training and the API.
    """
    n = len(starts)
    feats, windows = [None] * n, [None] * n
    valid = [i for i, s in enumerate(starts) if s is not None]
    track = None
    if valid:
        track = pitch_track(samples, sample_rate)
        for i, (w0, w1) in zip(valid, char_windows([starts[i] for i in valid])):
            windows[i] = (w0, w1)
            feats[i] = char_features(track, w0, w1, context_features(tones, clause_lengths, i))
    if return_pitch_stats:
        pitch_stats = (track[2], track[3]) if track is not None else (0.0, 1.0)
        return feats, windows, pitch_stats
    return feats, windows

def window_energy_ratio(samples, sample_rate, w0, w1):
    """Return window RMS / utterance RMS to tell whether the learner spoke when F0 could not be measured."""
    samples = np.asarray(samples, dtype=np.float64)
    window = samples[max(0, int(w0 * sample_rate)):max(0, int(w1 * sample_rate))]
    total_rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    if len(window) == 0 or total_rms == 0.0:
        return 0.0
    return float(np.sqrt(np.mean(window ** 2))) / total_rms

def load_audio_samples(path):
    """Read audio as mono float32 samples in [-1, 1] plus the sample rate, identically for training and the API."""
    from pydub import AudioSegment
    segment = AudioSegment.from_file(path).set_channels(1)
    samples = np.array(segment.get_array_of_samples()).astype(np.float32) / float(2 ** (8 * segment.sample_width - 1))
    return samples, segment.frame_rate

def is_low_confidence(features, align_duration):
    """
    Tone measurement is unreliable if F0 is missing, the voiced segment is too short, or alignment collapses
    to roughly one frame. `align_duration` (seconds) is None for the final character, which has no following character to define its end.
    """
    if features is None or features[F_VOICED_DUR] < LOW_CONF_VOICED_SEC:
        return True
    return align_duration is not None and align_duration <= LOW_CONF_ALIGN_SEC


def reconcile_clause_assessment(items, pass_threshold=CLAUSE_PASS_THRESHOLD):
    """Score a clause and reconcile low-confidence errors when the raw score meets the pass threshold.

    Each item must have `raw_is_correct_tone`, `is_correct_tone`, and `low_confidence`.
    Unscored items (`raw_is_correct_tone is None`) are excluded from the denominator.
    If no character in the clause is assessable, return None for the score/status instead of 100%.
    Updates `is_correct_tone` and `reconciled` on each item so the details remain
    consistent with the clause score.
    """
    if not 0.0 <= pass_threshold <= 100.0:
        raise ValueError("pass_threshold must be between 0 and 100")

    valid_items = [item for item in items if item.get("raw_is_correct_tone") is not None]
    total_valid = len(valid_items)
    if total_valid == 0:
        return {
            "accuracy": None,
            "raw_accuracy": None,
            "is_passed": None,
            "n_assessable": 0,
            "n_reconciled": 0,
        }

    raw_correct = sum(bool(item["raw_is_correct_tone"]) for item in valid_items)
    raw_accuracy = raw_correct / total_valid * 100.0
    is_passed = raw_accuracy >= pass_threshold
    reconciled_count = 0

    if is_passed:
        for item in valid_items:
            if item["raw_is_correct_tone"] is False and item.get("low_confidence", False):
                item["is_correct_tone"] = True
                item["reconciled"] = True
                reconciled_count += 1

    final_correct = sum(bool(item.get("is_correct_tone")) for item in valid_items)
    return {
        "accuracy": final_correct / total_valid * 100.0,
        "raw_accuracy": raw_accuracy,
        "is_passed": is_passed,
        "n_assessable": total_valid,
        "n_reconciled": reconciled_count,
    }
