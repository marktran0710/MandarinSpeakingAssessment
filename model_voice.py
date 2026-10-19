"""Second assessment mode, "Theo giọng mẫu" (follow the model voice).

The rule-based verdict (tone_pipeline) asks whether the judged pitch window has the shape a tone needs. This mode asks how close it is to
the way the teachers said the same tone: the window is normalised by itself, aligned by DTW to each of N teacher contours of the tone, and the
median similarity goes through a per-tone sigmoid to a 0..1 score. A syllable whose script accepts a second tone (a third tone before another
third tone: T2 too) is also scored against that tone, which counts only when it clearly matches (`alternative_tone_min`).

The contours and the sigmoids are data (`model_voice_contours.json`, rebuilt by tools/build_model_voice_contours.py); the other numbers are
the `model_voice` section of tone_v1_config.yaml. Nothing here imports tone_pipeline: the pipeline calls in.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

MODE_RULES = "rules"
MODE_MODEL_VOICE = "model_voice"
MODES = (MODE_RULES, MODE_MODEL_VOICE)
CONTOURS_PATH = Path(__file__).with_name("model_voice_contours.json")
TONES = (1, 2, 3, 4)

try:                                                   # optional: ~100x faster DTW; without it the same numbers come out of plain Python
    import numba
except Exception:                                      # pragma: no cover - depends on the environment
    numba = None


# ------------------------------------------------------------------------------------------------ the sequence ----

def judged_values(st):
    """The finite semitone values of a judged window, every second 5 ms frame (10 ms steps); None when fewer than 2 are left."""
    values = np.asarray(st, dtype=float)
    values = values[np.isfinite(values)][::2]
    return values if len(values) >= 2 else None


def recording_std(clean_st):
    """Spread (semitones) of every voiced frame of the recording: the unit in which the std floor of a syllable is expressed."""
    every = np.asarray(clean_st, dtype=float)
    every = every[np.isfinite(every)]
    return float(every.std()) if len(every) else 1.0


def normalise(values, spread, std_floor):
    """The syllable by itself: (x - mean) / max(std, std_floor x spread). A level contour keeps a small amplitude (its std is raised to the floor)
    so tracking noise is not blown up into a shape, and the level itself (high or low in the sentence) plays no part."""
    x = np.asarray(values, dtype=float)
    return (x - x.mean()) / max(float(x.std()), std_floor * (spread or 1.0))


# --------------------------------------------------------------------------------------------------------- DTW ----

def _smooth(x, window):
    n = len(x)
    if window <= 1 or n < 3:
        return x
    width = min(window, n)
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return x
    r = width // 2
    return np.convolve(np.pad(x, (r, r), mode="edge"), np.ones(width) / width, mode="valid")


def _dtw_python(ref, lr, band):
    """Aligned index paths of two sequences (cost |a - b|; steps up, left, diagonal; ties go up, then left). `band` < 0: no band."""
    n, m = len(ref), len(lr)
    width = max(band, abs(n - m)) if band >= 0 else n + m + 2
    inf = float("inf")
    acc = [[inf] * (m + 1) for _ in range(n + 1)]
    acc[0][0] = 0.0
    for i in range(1, n + 1):
        centre = i * m / n
        above, row, value = acc[i - 1], acc[i], ref[i - 1]
        for j in range(1, m + 1):
            if abs(j - centre) > width:
                continue
            best = above[j]
            if row[j - 1] < best:
                best = row[j - 1]
            if above[j - 1] < best:
                best = above[j - 1]
            row[j] = abs(value - lr[j - 1]) + best
    ia, ib = [], []
    i, j = n, m
    while i > 0 or j > 0:
        ia.append(max(i - 1, 0))
        ib.append(max(j - 1, 0))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step, value = 0, acc[i - 1][j]
            if acc[i][j - 1] < value:
                step, value = 1, acc[i][j - 1]
            if acc[i - 1][j - 1] < value:
                step = 2
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    return np.array(ia[::-1]), np.array(ib[::-1])


def _similarity_python(ref, lr, band_fraction, window, tau):
    band = int(math.ceil(band_fraction * max(len(ref), len(lr)))) if band_fraction is not None else -1
    ia, ib = _dtw_python(ref.tolist(), lr.tolist(), band)
    a, b = _smooth(ref[ia], window), _smooth(lr[ib], window)
    return math.exp(-float(np.mean((a - b) ** 2)) / (tau * tau))


if numba is not None:
    @numba.njit(cache=True)
    def _similarities_numba(flat, offsets, lr, band_fraction, window, tau2):
        out = np.empty(len(offsets) - 1)
        m = len(lr)
        for r in range(len(offsets) - 1):
            ref = flat[offsets[r]:offsets[r + 1]]
            n = len(ref)
            width = int(np.ceil(band_fraction * max(n, m))) if band_fraction >= 0 else n + m + 2
            width = max(width, abs(n - m)) if band_fraction >= 0 else width
            acc = np.full((n + 1, m + 1), np.inf)
            acc[0, 0] = 0.0
            for i in range(1, n + 1):
                centre = i * m / n
                for j in range(1, m + 1):
                    if abs(j - centre) > width:
                        continue
                    best = acc[i - 1, j]
                    if acc[i, j - 1] < best:
                        best = acc[i, j - 1]
                    if acc[i - 1, j - 1] < best:
                        best = acc[i - 1, j - 1]
                    acc[i, j] = abs(ref[i - 1] - lr[j - 1]) + best
            ia = np.empty(n + m + 2, np.int64)
            ib = np.empty(n + m + 2, np.int64)
            k = 0
            i, j = n, m
            while i > 0 or j > 0:
                ia[k] = max(i - 1, 0)
                ib[k] = max(j - 1, 0)
                k += 1
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    step = 0
                    value = acc[i - 1, j]
                    if acc[i, j - 1] < value:
                        step = 1
                        value = acc[i, j - 1]
                    if acc[i - 1, j - 1] < value:
                        step = 2
                    if step == 0:
                        i -= 1
                    elif step == 1:
                        j -= 1
                    else:
                        i -= 1
                        j -= 1
            a = ref[ia[:k][::-1]]
            b = lr[ib[:k][::-1]]
            if window > 1 and k >= 3:
                size = min(window, k)
                if size % 2 == 0:
                    size -= 1
                if size > 1:
                    radius = size // 2
                    sa = np.empty(k)
                    sb = np.empty(k)
                    for p in range(k):
                        ta = 0.0
                        tb = 0.0
                        for q in range(-radius, radius + 1):
                            idx = min(max(p + q, 0), k - 1)
                            ta += a[idx]
                            tb += b[idx]
                        sa[p] = ta / size
                        sb[p] = tb / size
                    a, b = sa, sb
            total = 0.0
            for p in range(k):
                total += (a[p] - b[p]) ** 2
            out[r] = np.exp(-(total / k) / tau2)
        return out


USE_NUMBA = numba is not None


def warm_up():
    """Compile the fast path now (first call), so the first learner does not wait for it."""
    if USE_NUMBA:
        _similarities_numba(np.array([0.0, 1.0, 2.0]), np.array([0, 3], dtype=np.int64), np.array([0.0, 1.0, 2.0, 3.0]), 0.15, 5, 0.64)


# ----------------------------------------------------------------------------------------------- the contours ----

class Contours:
    """Teacher contours per tone (already normalised) and the per-tone sigmoid that turns a median similarity into a 0..1 score."""

    def __init__(self, references, sigmoid, meta=None):
        self.references = {int(t): [np.asarray(r, dtype=float) for r in rs] for t, rs in references.items()}
        self.sigmoid = {int(t): (float(p["midpoint"]), float(p["steepness"])) for t, p in sigmoid.items()}
        self.meta = meta or {}
        self._flat = {}
        for tone, refs in self.references.items():
            offsets = np.concatenate([[0], np.cumsum([len(r) for r in refs])]).astype(np.int64)
            self._flat[tone] = (np.concatenate(refs) if refs else np.zeros(0), offsets)

    def tones(self):
        return tuple(t for t in TONES if self.references.get(t) and t in self.sigmoid)

    def similarity(self, tone, sequence, settings, count=None):
        """Median similarity of the normalised `sequence` to the first `count` (default: all) contours of `tone`."""
        refs = self.references[tone]
        count = min(count or settings["references_per_tone"], len(refs))
        band = float(settings["band"]) if settings.get("band") is not None else -1.0
        window, tau = int(settings["smooth"]), float(settings["tau"])
        if USE_NUMBA:
            flat, offsets = self._flat[tone]
            values = _similarities_numba(flat, offsets[:count + 1], np.ascontiguousarray(sequence, dtype=float), band, window, tau * tau)
        else:
            values = [_similarity_python(ref, sequence, None if band < 0 else band, window, tau) for ref in refs[:count]]
        return float(np.median(values))

    def score(self, tone, sequence, settings, count=None):
        midpoint, steepness = self.sigmoid[tone]
        exponent = max(-60.0, min(60.0, -steepness * (self.similarity(tone, sequence, settings, count) - midpoint)))
        return 1.0 / (1.0 + math.exp(exponent))


@lru_cache(maxsize=4)
def load_contours(path=None):
    data = json.loads(Path(path or CONTOURS_PATH).read_text(encoding="utf-8"))
    return Contours(data["references"], data["sigmoid"], {k: v for k, v in data.items() if k not in ("references", "sigmoid")})


def available(path=None):
    try:
        return bool(load_contours(str(path) if path else None).tones())
    except (OSError, ValueError, KeyError):
        return False


# ------------------------------------------------------------------------------------------------- the verdict ----

def judge(values, spread, accepted, settings, contours):
    """Score a judged window against the accepted tones (the expected one first).

    Returns None when the window is too short to judge, else {"score": best accepted score, "scores": {tone: score of every tone that was scored}}.
    A tone other than the expected one counts only at `alternative_tone_min` or more."""
    if values is None or len(values) < int(settings.get("min_frames", 2)):
        return None
    sequence = normalise(values, spread, float(settings["std_floor"]))
    scores = {}
    expected = accepted[0]
    if expected not in contours.tones():
        return None
    scores[expected] = contours.score(expected, sequence, settings)
    best = scores[expected]
    for tone in accepted[1:]:
        if tone in contours.tones():
            scores[tone] = contours.score(tone, sequence, settings)
            if scores[tone] >= float(settings["alternative_tone_min"]):
                best = max(best, scores[tone])
    return {"score": best, "scores": scores, "sequence": sequence}


def closest_tone(sequence, scores, settings, contours):
    """The tone whose contours the window matches best, scoring the tones not yet scored; None when it is one of `scores`' best anyway."""
    every = dict(scores)
    for tone in contours.tones():
        if tone not in every:
            every[tone] = contours.score(tone, sequence, settings)
    best = max(every, key=every.get)
    return best, every


def verdict(score, quality, can_accuse, config):
    """Good at `good_at` with decent quality; Needs practice at `practice_at` or less only when accusing is allowed and the quality is high; else Okay."""
    settings, confidence = config["model_voice"], config["confidence"]
    if score >= float(settings["good_at"]) and quality >= confidence["likely_correct_quality"]:
        return "likely_correct"
    if score <= float(settings["practice_at"]) and can_accuse and quality >= confidence["practice_quality"]:
        return "practice"
    return "insufficient_evidence"
