"""Second assessment mode, "Theo giọng mẫu" (follow the model voice).

The rule-based verdict (tone_pipeline) asks whether the judged pitch window has the shape a tone needs. This mode asks how close it is to
the way the teachers said the same tone: the window is normalised by itself, aligned by DTW to each of N teacher contours of every tone, and the
median cosine similarity goes through a per-tone sigmoid to a 0..1 score. The highest raw cosine identifies the closest tone. A syllable whose script accepts a second tone (a third tone before another
third tone: T2 too) is also scored against that tone, which counts only when it clearly matches (`alternative_tone_min`).
When that comparison does not accept the syllable, the mode can compare it with the same syllable in the selected book audio as a fallback.

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
MODE_FUSION = "fusion"
MODES = (MODE_RULES, MODE_MODEL_VOICE, MODE_FUSION)
MODES_USING_MODEL_VOICE = (MODE_MODEL_VOICE, MODE_FUSION)
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


def _similarity_python(ref, lr, band_fraction, window):
    band = int(math.ceil(band_fraction * max(len(ref), len(lr)))) if band_fraction is not None else -1
    ia, ib = _dtw_python(ref.tolist(), lr.tolist(), band)
    a, b = _smooth(ref[ia], window), _smooth(lr[ib], window)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


if numba is not None:
    @numba.njit(cache=True)
    def _similarities_numba(flat, offsets, lr, band_fraction, window):
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
            dot = 0.0
            norm_a = 0.0
            norm_b = 0.0
            for p in range(k):
                dot += a[p] * b[p]
                norm_a += a[p] * a[p]
                norm_b += b[p] * b[p]
            denominator = np.sqrt(norm_a * norm_b)
            if denominator <= 1e-12:
                out[r] = 0.0
            else:
                out[r] = min(1.0, max(-1.0, dot / denominator))
        return out


USE_NUMBA = numba is not None


def warm_up():
    """Compile the fast path now (first call), so the first learner does not wait for it."""
    if USE_NUMBA:
        _similarities_numba(np.array([0.0, 1.0, 2.0]), np.array([0, 3], dtype=np.int64), np.array([0.0, 1.0, 2.0, 3.0]), 0.15, 5)


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
        """Median DTW-aligned cosine similarity to the first `count` (default: all) contours of `tone`."""
        refs = self.references[tone]
        count = min(count or settings["references_per_tone"], len(refs))
        band = float(settings["band"]) if settings.get("band") is not None else -1.0
        window = int(settings["smooth"])
        if USE_NUMBA:
            flat, offsets = self._flat[tone]
            values = _similarities_numba(flat, offsets[:count + 1], np.ascontiguousarray(sequence, dtype=float), band, window)
        else:
            values = [_similarity_python(ref, sequence, None if band < 0 else band, window) for ref in refs[:count]]
        return float(np.median(values))

    def score_from_similarity(self, tone, similarity):
        midpoint, steepness = self.sigmoid[tone]
        exponent = max(-60.0, min(60.0, -steepness * (similarity - midpoint)))
        return 1.0 / (1.0 + math.exp(exponent))

    def score(self, tone, sequence, settings, count=None):
        return self.score_from_similarity(tone, self.similarity(tone, sequence, settings, count))


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

    Returns None when the window is too short to judge. Otherwise scores every tone: cosine chooses the closest tone,
    and the corresponding tone-specific sigmoid supplies the confidence score. A tone other than the expected one counts
    as an accepted reading only at `alternative_tone_min` or more."""
    if values is None or len(values) < int(settings.get("min_frames", 2)):
        return None
    sequence = normalise(values, spread, float(settings["std_floor"]))
    expected = accepted[0]
    if expected not in contours.tones():
        return None
    similarities = {tone: contours.similarity(tone, sequence, settings) for tone in contours.tones()}
    scores = {tone: contours.score_from_similarity(tone, similarity) for tone, similarity in similarities.items()}
    best_tone = max(similarities, key=similarities.get)
    accepted_tones = [tone for tone in accepted if tone in similarities]
    eligible = [expected, *(tone for tone in accepted_tones[1:]
                             if scores[tone] >= float(settings["alternative_tone_min"]))]
    selected = max(eligible, key=similarities.get)
    competing = [similarity for tone, similarity in similarities.items() if tone not in eligible]
    competing_lead = max(competing, default=-1.0) - similarities[selected]
    within_competing_margin = competing_lead <= float(settings.get("max_competing_cosine_margin", 0.0))
    accepted_match = within_competing_margin
    return {"score": scores[selected], "scores": scores, "similarities": similarities,
            "best_tone": best_tone, "selected_accepted_tone": selected,
            "accepted_match": accepted_match, "sequence": sequence}


def closest_tone(sequence, scores, settings, contours):
    """Return the tone with the highest raw DTW-aligned cosine, plus every tone's sigmoid score."""
    every = dict(scores)
    for tone in contours.tones():
        if tone not in every:
            every[tone] = contours.score(tone, sequence, settings)
    similarities = {tone: contours.similarity(tone, sequence, settings) for tone in contours.tones()}
    best = max(similarities, key=similarities.get)
    return best, every


def verdict(score, quality, can_accuse, config):
    """Good at `good_at` with decent quality; Needs practice at `practice_at` or less only when accusing is allowed and the quality is high; else Okay."""
    settings, confidence = config["model_voice"], config["confidence"]
    if score >= float(settings["good_at"]) and quality >= confidence["likely_correct_quality"]:
        return "likely_correct"
    if score <= float(settings["practice_at"]) and can_accuse and quality >= confidence["practice_quality"]:
        return "practice"
    return "insufficient_evidence"


def same_sample_similarity(values, spread, reference_values, reference_spread, settings):
    """DTW cosine against this exact syllable in the selected teacher audio, or None if either contour is too short."""
    learner = np.asarray(values, dtype=float)
    reference = np.asarray(reference_values, dtype=float)
    learner = learner[np.isfinite(learner)]
    reference = reference[np.isfinite(reference)]
    minimum = int(settings.get("min_frames", 2))
    if len(learner) < minimum or len(reference) < minimum:
        return None
    learner = normalise(learner, spread, float(settings["std_floor"]))
    reference = normalise(reference, reference_spread, float(settings["std_floor"]))
    band = float(settings["band"]) if settings.get("band") is not None else None
    return _similarity_python(reference, learner, band, int(settings["smooth"]))
