"""
Service layer between the API and tone_pipeline.

Runs the v1 tone-verification pipeline on one recording and returns the payload shape the web UI and
practice_store already read (`phrases`, `summary`, `transcript_check`, `phrase_f0.contour`, ...), plus the full
per-syllable diagnostics under `tone_verification`. No web framework and no model loading happens here.
"""
from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

import audio_prep
import capt_core
import model_voice
import tone_pipeline as tp

MODE = "tone_pipeline_v1"
BOOK_SENTENCE_TONES_PATH = Path(__file__).with_name("book_sentence_tones.json")
_CONFIG = None
_SENTENCE_CONFIG = None
_LEVEL_CONFIGS = {}
_MODEL_VOICE_CONFIGS = {}
STANDARD = "standard"

try:
    _BOOK_SENTENCE_TONES = json.loads(BOOK_SENTENCE_TONES_PATH.read_text(encoding="utf-8"))["samples"]
except (OSError, ValueError, KeyError, TypeError):
    _BOOK_SENTENCE_TONES = {}


def _layer(*layers):
    """Overrides for tp.load_config: section by section, later layers win key by key."""
    merged = {}
    for layer in layers:
        for section, values in (layer or {}).items():
            merged.setdefault(section, {}).update(values)
    return merged


def get_config(profile=None, level=None, scoring_mode=None):
    """The frozen configuration. profile="sentence" layers the `sentence` overrides on top (book / custom practice).

    `level` ("easy", ...) layers the matching `levels:` section of the config on top of that: the same pipeline asking for a
    smaller contour movement (see `levels` in tone_v1_config.yaml). None / "standard" is the config as it is.

    `scoring_mode="model_voice"` is the same configuration with the "Theo giọng mẫu" verdict switched on (model_voice.py); None / "rules" is as it is."""
    if scoring_mode in (None, model_voice.MODE_RULES):
        return _rules_config(profile, level)
    if scoring_mode not in model_voice.MODES_USING_MODEL_VOICE:
        raise ValueError(f"Unknown scoring mode: {scoring_mode}")
    key = (profile, level)
    if key not in _MODEL_VOICE_CONFIGS:
        config = deepcopy(_rules_config(profile, level))
        config.setdefault("model_voice", {})["enabled"] = True
        _MODEL_VOICE_CONFIGS[key] = config
    return _MODEL_VOICE_CONFIGS[key]


def _rules_config(profile=None, level=None):
    global _CONFIG, _SENTENCE_CONFIG
    if _CONFIG is None:
        _CONFIG = tp.load_config()
    if profile not in (None, "sentence"):
        raise ValueError(f"Unknown config profile: {profile}")
    if level in (None, STANDARD):
        if profile is None:
            return _CONFIG
        if _SENTENCE_CONFIG is None:
            _SENTENCE_CONFIG = tp.load_config(overrides=_CONFIG["sentence"])
        return _SENTENCE_CONFIG
    levels = _CONFIG.get("levels") or {}
    if level not in levels:
        raise ValueError(f"Unknown difficulty level: {level}")
    key = (profile, level)
    if key not in _LEVEL_CONFIGS:
        mine = levels[level]
        shared = {section: values for section, values in mine.items() if section != "sentence"}
        layers = [_CONFIG["sentence"], shared, mine.get("sentence")] if profile == "sentence" else [shared]
        _LEVEL_CONFIGS[key] = tp.load_config(overrides=_layer(*layers))
    return _LEVEL_CONFIGS[key]


def resolve_level(requested, group_type, config=None):
    """The level to assess with: the one the request names ("" / None = the learner group's default); ValueError if unknown."""
    config = config or get_config()
    requested = str(requested or "").strip().lower()
    if not requested:
        return level_for_group(group_type, config)
    if requested != STANDARD and requested not in (config.get("levels") or {}):
        raise ValueError(f"Unknown difficulty level: {requested}")
    return requested


def level_for_group(group_type, config=None):
    """The level used when the request names none ("Automatic" in the page): `levels.default_by_group[group]` if the config has one for this
    group, else `levels.default` (easy). An unknown name falls back to standard."""
    config = config or get_config()
    levels = config.get("levels") or {}
    level = (levels.get("default_by_group") or {}).get(group_type) or levels.get("default") or STANDARD
    return level if level == STANDARD or level in levels else STANDARD


# ------------------------------------------------------------------ grouping ----

def make_groups(clauses, annotated_words, normalize):
    """Phrase groups as (label, start, end) over the flattened characters, plus the annotation status.

    Word-level annotations are used when they reproduce the script exactly; otherwise punctuation clauses.
    """
    target_chars = list("".join(clauses))
    word_mapping = []
    for word in annotated_words:
        chars = list("".join(capt_core.prepare_target(normalize(word))))
        if chars:
            word_mapping.append((word, chars))
    annotation_hanzi = "".join(character for _, chars in word_mapping for character in chars)
    if not annotated_words:
        status = "missing"
    elif annotation_hanzi != "".join(target_chars):
        status = "does_not_match_script"
    else:
        status = "matched"
    groups, cursor = [], 0
    pieces = [(word, len(chars)) for word, chars in word_mapping] if status == "matched" else [
        (clause, len(clause)) for clause in clauses]
    for label, length in pieces:
        groups.append((label, cursor, cursor + length))
        cursor += length
    return groups, status


# ---------------------------------------------------------- words in typed text ----

def build_lexicon(words, normalize=lambda text: text):
    """Words of two to four characters, from the teacher's word annotation and the vocabulary lists."""
    lexicon = set()
    for word in words:
        for variant in re.split(r"[/／]", str(word)):
            han = "".join(capt_core.prepare_target(normalize(variant)))
            if 2 <= len(han) <= 4:
                lexicon.add(han)
    return lexicon


_JIEBA = None


def _jieba():
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _JIEBA = jieba
        except ImportError:
            _JIEBA = False
    return _JIEBA or None


def segment_words(clauses, lexicon, max_word=4, cap=3):
    """Words for text the learner typed, where no word annotation exists, so results can be shown per word.

    Known words (the teacher's annotation and the vocabulary lists) are taken longest first, which reproduces the
    textbook's own divisions; what is left is split by jieba on the Simplified form (held out from the lexicon, the
    boundaries agree with the teacher's about 95% of the time). Words are at most `cap` characters and never cross a
    punctuation clause. Without jieba the unknown characters simply stay single."""
    jieba = _jieba()

    def unknown(run):
        if jieba is None or len(run) < 2:
            return list(run)
        pieces, position = [], 0
        for word in jieba.lcut(capt_core.simplified(run)):
            piece = run[position:position + len(word)]
            position += len(word)
            pieces.extend([piece] if len(piece) <= cap else list(piece))
        return pieces if position == len(run) else list(run)

    words = []
    for clause in clauses:
        run, i = "", 0
        while i < len(clause):
            for n in range(min(max_word, len(clause) - i), 1, -1):
                if clause[i:i + n] in lexicon:
                    words.extend(unknown(run))
                    run = ""
                    words.append(clause[i:i + n])
                    i += n
                    break
            else:
                run += clause[i]
                i += 1
        words.extend(unknown(run))
    return words


# ------------------------------------------------------------------ chapter 5 ----

def split_optional(text):
    """'裡(面)' -> ('裡', '面', ''): the text before, inside and after one parenthesized optional span, or None."""
    for opening, closing in ((chr(0xFF08), chr(0xFF09)), ("(", ")")):
        start = text.find(opening)
        end = text.find(closing, start + 1) if start >= 0 else -1
        if start >= 0 and end > start:
            return text[:start], text[start + 1:end], text[end + 1:]
    return None


def chapter5_optional_text_variants(text):
    """Expand one parenthesized optional span into present/omitted transcript forms."""
    variants = [text]
    parts = split_optional(text)
    if parts:
        prefix, optional, suffix = parts
        variants.extend((prefix + suffix, prefix + optional + suffix))
    return variants


def split_variants(script):
    """'哪裡/哪兒' -> ['哪裡', '哪兒']: every slash alternative is its own pronunciation path."""
    return [part.strip() for part in re.split(r"[/／]", script) if part.strip()]


def expand_optional(alternative):
    """'裡(面)' -> ['裡', '裡面'] (omitted, present); text without optional parts is returned as is."""
    options = chapter5_optional_text_variants(alternative)
    return options[1:] if len(options) > 1 else options


def chapter5_candidates(item, requested_script, normalize):
    """Distinct allowed scripts to align. Each variant (a slash alternative, optional text present or omitted)
    is its own candidate path; paths are never concatenated into one longer script."""
    candidates, seen = [], set()
    for script in [requested_script, item.get("script", ""), *item.get("script_variants", [])]:
        for alternative in split_variants(normalize(str(script or "").strip())):
            for variant in expand_optional(alternative):
                clauses = [part for part in capt_core.prepare_target(variant) if part.strip()]
                chars = tuple("".join(clauses))
                if chars and chars not in seen:
                    seen.add(chars)
                    candidates.append({"script_variant": variant, "clauses": clauses, "chars": list(chars)})
    return candidates


def chapter5_contexts(clauses, item):
    """pypinyin sandhi plus the catalog's disambiguating Pinyin where the workbook gives one."""
    contexts = capt_core.expected_tone_contexts(clauses)
    supplied_pinyin = str(item.get("pinyin") or "")
    # The workbook gives kòng in 沒（有）空; pypinyin picks kōng in the longer context.
    if "".join(clauses) in {"沒空", "沒有空"} and supplied_pinyin.endswith("kòng") and contexts:
        contexts[-1] = {**contexts[-1], "lexical_tone": 4, "target_tone": 4, "tone_rule": "catalog_pinyin_disambiguation"}
    return contexts


_PINYIN_SYLLABLE = re.compile(
    r"(?:[zcs]h|[bpmfdtnlgkhjqxrzcswy])?[aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+(?:ng|n|r)?"
    r"(?![aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ])"
)
_PINYIN_MARK_TONE = {
    char: tone
    for tone, group in enumerate(("āēīōūǖ", "áéíóúǘ", "ǎěǐǒǔǚ", "àèìòùǜ"), start=1)
    for char in group
}


def book_pinyin_tone_paths(pinyin):
    """Tone-number paths (5 = unmarked/neutral) in a workbook's tone-marked Pinyin."""
    paths = []
    for variant in str(pinyin or "").split("/"):
        text = variant.replace(" ", "").replace("'", "").lower()
        for form in expand_optional(text):
            form = form.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
            paths.append([
                next((_PINYIN_MARK_TONE[ch] for ch in match.group(0) if ch in _PINYIN_MARK_TONE), 5)
                for match in _PINYIN_SYLLABLE.finditer(form)
            ])
    return paths


def _contexts_with_book_tones(contexts, tones):
    if len(contexts) != len(tones):
        raise ValueError(f"Book Pinyin has {len(tones)} syllables, but the script has {len(contexts)}.")
    out = []
    for index, (context, tone) in enumerate(zip(contexts, tones)):
        if tone not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid book tone: {tone!r}.")
        same_previous_clause = index > 0 and contexts[index - 1].get("clause_index") == context.get("clause_index")
        same_next_clause = index + 1 < len(contexts) and contexts[index + 1].get("clause_index") == context.get("clause_index")
        previous_tone = int(tones[index - 1]) if same_previous_clause else None
        next_tone = int(tones[index + 1]) if same_next_clause else None
        item = {
            **context,
            "lexical_tone": int(tone),
            "target_tone": int(tone),
            "tone_rule": "book_pinyin",
            "tone_source": "book_pinyin",
            "tone_confidence": 1.0,
            "neutral_state": "required" if tone == 5 else None,
            "previous_tone": previous_tone,
            "next_tone": next_tone,
            "adjacent_previous_tone": previous_tone,
            "adjacent_next_tone": next_tone,
            "allow_half_t3": tone == 3 and not context.get("is_clause_final") and next_tone in {1, 2, 4, 5},
        }
        # Book mode follows the printed target. Do not carry the Rules mode's optional sandhi tones over.
        item.pop("also_accept_tones", None)
        out.append(item)
    return out


def book_sentence_contexts(sample_id, clauses, contexts):
    """Book-printed expected tones for one lesson sentence, separate from surface-tone rule contexts."""
    entry = _BOOK_SENTENCE_TONES.get(str(sample_id))
    if not entry:
        raise ValueError(f"No book Pinyin tone data is available for sample {sample_id}.")
    actual_script = "".join(clauses)
    book_script = "".join(capt_core.prepare_target(entry.get("script", "")))
    if actual_script != book_script:
        raise ValueError(f"Book Pinyin tone data does not match sample {sample_id}.")
    return _contexts_with_book_tones(contexts, entry.get("tones", []))


def chapter5_book_contexts(clauses, item, script_variant, normalize=lambda text: text):
    """Book-Pinyin targets for one Chapter 5 transcript variant."""
    contexts = chapter5_contexts(clauses, item)
    paths = book_pinyin_tone_paths(item.get("pinyin"))
    if not paths:
        raise ValueError(f"No book Pinyin tone data is available for {script_variant}.")

    def script_key(text):
        return "".join(capt_core.prepare_target(normalize(text)))

    source_text = str(item.get("traditional_chinese") or item.get("script") or "")
    script_forms = [form for alternative in split_variants(source_text) for form in expand_optional(alternative)]
    tone_by_script = {}
    if len(script_forms) == len(paths):
        tone_by_script = {script_key(text): tones for text, tones in zip(script_forms, paths)}
    else:
        for text in script_forms:
            key = script_key(text)
            matching = [tones for tones in paths if len(tones) == len(key)]
            if len(matching) == 1:
                tone_by_script[key] = matching[0]

    key = script_key(script_variant)
    tones = tone_by_script.get(key)
    if tones is None:
        matching = [path for path in paths if len(path) == len(contexts)]
        if len(matching) == 1:
            tones = matching[0]
    if tones is None:
        raise ValueError(f"Book Pinyin tone data does not match Chapter 5 variant {script_variant}.")
    return _contexts_with_book_tones(contexts, tones)


def chapter5_display_cells(item):
    """Characters with pinyin for an item written with an optional part, e.g. 裡(面): [{char, pinyin, optional}, ...].

    The recording says 裡面 and both 裡 and 裡面 are accepted, so the learner is shown the full form with the optional
    part marked, not just 裡. Items without an optional part return None (their plain script and pinyin are shown)."""
    text = (split_variants(str(item.get("traditional_chinese") or item.get("script") or "")) or [""])[0]
    parts = split_optional(text)
    if not parts or not parts[1].strip():
        return None
    prefix, optional, suffix = parts
    full = prefix + optional + suffix
    clauses = [c for c in capt_core.prepare_target(full) if c.strip()]
    pinyin = capt_core.display_pinyin(full, tones=[capt_core.spoken_tone_for_print(c) for c in chapter5_contexts(clauses, item)])
    if len(pinyin) != len(full):
        return None
    flags = [False] * len(prefix) + [True] * len(optional) + [False] * len(suffix)
    return [{"char": char, "pinyin": letters, "optional": flag} for char, letters, flag in zip(full, pinyin, flags)]


def chapter5_character_pinyin(item):
    """Pinyin to show over a Chapter 5 word: the tones the word is scored on (including the catalog's own readings; 不 and 一 as spoken)."""
    clauses = [c for c in capt_core.prepare_target(item["script"]) if c.strip()]
    tones = [capt_core.spoken_tone_for_print(c) for c in chapter5_contexts(clauses, item)]
    return capt_core.display_pinyin(item["script"], tones=tones)


# -------------------------------------------------------------------- payload ----

def phrase_f0(contour, start_sec, end_sec, baseline_hz):
    """Chart contour for one phrase window: trims outer unvoiced frames, keeps inner gaps as nulls."""
    if not contour or end_sec <= start_sec or not baseline_hz:
        return None
    times = np.asarray(contour["times"], dtype=float)
    st = np.array([np.nan if v is None else v for v in contour["st"]], dtype=float)
    window = np.flatnonzero((times >= start_sec) & (times < end_sec))
    if not len(window):
        return None
    voiced = window[np.isfinite(st[window])]
    if not len(voiced):
        return None
    selected = np.arange(voiced[0], voiced[-1] + 1)
    origin, step = float(times[selected[0]]), float(contour["step_sec"])
    baseline_midi = float(tp.hz_to_midi(baseline_hz))
    points = [{
        "offset_sec": round(float(times[i]) - origin, 3),
        "relative_semitone": None if not np.isfinite(st[i]) else round(float(st[i]), 2),
        "midi": None if not np.isfinite(st[i]) else round(baseline_midi + float(st[i]), 2),     # 69 + 12*log2(F0/440)
        "hz": None if not np.isfinite(st[i]) else round(float(baseline_hz * 2 ** (st[i] / 12)), 1),
    } for i in selected]
    window_voiced = int(np.isfinite(st[window]).sum())
    return {
        "duration_sec": round(float(times[selected[-1]]) + step - origin, 3),
        "origin_sec": round(origin, 3),                       # recording time of offset 0 (the first voiced frame): lets a chart place syllables on its axis
        "aligned_window_sec": [round(start_sec, 3), round(end_sec, 3)],
        "frame_count": int(len(selected)),
        "voiced_frame_count": int(np.isfinite(st[selected]).sum()),
        "aligned_voiced_coverage": round(window_voiced / len(window), 3),
        "pitch_reference_hz": round(float(baseline_hz), 1),
        "pitch_reference_midi": round(baseline_midi, 2),
        "contour": points,
    }


def _tone_or_zero(tone):
    return 0 if tone == 5 else tone


# Why a word could not be judged -> what the learner can do about it. First match wins; codes only, the UI words them.
_RETRY_BY_REASON = (
    ("full_third", {"half_third_only"}),
    ("read_exactly", {"bad_alignment"}),
    ("slower", {"syllable_too_short", "short_analysis_window", "too_few_voiced_frames"}),
    ("steady", {"unstable_pitch_track", "octave_jump_in_segment", "edge_pitch_step", "low_acoustic_quality"}),
    ("clearer", {"ambiguous_contour"}),
)
MAX_TONE_ADVICE = 3


_SYSTEM_ALIGNMENT_ERRORS = ("MFA failed", "MFA timed out", "MFA executable not found")


def alignment_error_kind(error):
    """None (no error), "system" (the aligner itself failed or timed out: not the learner's doing) or "reading"
    (the recording could not be matched to the script: out-of-vocabulary character, words that do not line up...)."""
    if not error:
        return None
    return "system" if str(error).startswith(_SYSTEM_ALIGNMENT_ERRORS) else "reading"


def _doubtful_syllable(group, names=None):
    """The first syllable that was Okay (for the given reasons; not a neutral tone)."""
    for s in group:
        if (s["status"] == "insufficient_evidence" and s["expected_surface_tone"] != 5
                and (names is None or set(s["reasons"]) & names)):
            return s
    return None


def _doubtful_character(group, names=None):
    """In a phrase of several characters, the first character that was Okay (for the given reasons): the advice names it."""
    if len(group) < 2:
        return None
    s = _doubtful_syllable(group, names)
    return s["character"] if s else None


def phrase_feedback(group):
    """`_phrase_advice` plus, in the "Theo giọng mẫu" mode, which tone a syllable that is not Good sounds closest to."""
    items = _phrase_advice(group)
    heard = next((s for s in group if s.get("model_closest_tone") and s["status"] != "likely_correct"), None)
    if heard is not None:
        items = [*items, {"type": "heard_as", "char": heard["character"], "tone": int(heard["model_closest_tone"])}]
    return items


def _phrase_advice(group):
    """What to do to improve a word, as codes for the UI to word in the learner's language (no measurements).

    practice -> the tone to work on for each flagged syllable; not enough evidence -> how to try again;
    good -> nothing. A recording that failed the audio quality check is explained once for the whole recording,
    not under every word."""
    status = tp.aggregate_status(group)
    if heard_nothing(group):
        failed = any("audio_quality_failed" in s["reasons"] for s in group)
        return [] if failed else [{"type": "retry", "code": "not_heard"}]
    if status == "practice":
        return [{"type": "tone", "char": s["character"], "tone": int(s["expected_surface_tone"])}
                for s in group if s["status"] == "practice"][:MAX_TONE_ADVICE]
    if status == "insufficient_evidence":
        reasons = {r for s in group if s["status"] == "insufficient_evidence" for r in s["reasons"]}
        if "audio_quality_failed" in reasons:
            return []
        for code, names in _RETRY_BY_REASON:
            if reasons & names:
                char = _doubtful_character(group, names)
                doubtful = _doubtful_syllable(group, names)
                if code == "slower" and doubtful is not None and doubtful["expected_surface_tone"] == 4:
                    code = "short_fall"        # tone 4 is the shortest tone: it must never be asked to be drawn out
                return [{"type": "retry", "code": code, **({"char": char} if char else {})}]
        char = _doubtful_character(group)
        return [{"type": "retry", "code": "again", **({"char": char} if char else {})}]
    return []


def pace_summary(syllables, config=None):
    """Was the recording read so fast that sounds could not be judged? `pace:` in the config: the typical aligned syllable is shorter than
    fast_median_syllable_ms (teachers read at 230-360 ms, learners at 240-410 ms), or a large share of the syllables was too short to analyse
    (syllable_too_short / short_analysis_window). The UI then asks the learner to read slower or to stretch each sound."""
    pace = (config or get_config()).get("pace") or {}
    skipped = set(pace.get("exclude_tones") or ())          # tone 4 and the light tone are naturally short: they never make a reading "too fast"
    timed = [s for s in syllables if s.get("duration_ms") and s.get("expected_surface_tone") not in skipped]
    if not pace or len(timed) < pace.get("min_syllables", 3):
        return {"too_fast": False}
    median = float(np.median([s["duration_ms"] for s in timed]))
    short = sum(1 for s in timed if {"syllable_too_short", "short_analysis_window"} & set(s.get("reasons") or ()))
    share = short / len(timed)
    return {"median_syllable_ms": round(median), "too_short_share": round(share, 2),
            "too_fast": bool(median < pace["fast_median_syllable_ms"] or share >= pace["too_short_share"])}


PARTIAL_CREDIT_SHARE = 0.5


def heard_nothing(group):
    """No measurable voice anywhere in the phrase (silence, or a recording that was rejected): nothing can be said about
    its tones, so it is neither good, nor okay, nor something to practise."""
    return bool(group) and all(
        s["status"] == "insufficient_evidence" and {"too_few_voiced_frames", "audio_quality_failed"} & set(s.get("reasons") or ())
        for s in group)


def display_status(group, partial_credit=False):
    """The status a word card shows. Sentence practice (`partial_credit`): a phrase that would show `practice` shows
    `uncertain` (yellow) instead when MORE than half of its scored syllables are likely_correct, so one doubtful
    syllable does not turn a mostly-right phrase red. The syllable verdicts themselves are never changed, and the
    advice for the doubtful syllable is still given. Neutral syllables are not scored, so they do not count.
    A phrase in which no voice was measured at all shows `not_heard`."""
    if heard_nothing(group):
        return "not_heard"
    status = tp.aggregate_status(group)
    if status != "practice" or not partial_credit:
        return status
    scored = [s for s in group if s["expected_surface_tone"] != 5] or list(group)
    correct = sum(s["status"] == "likely_correct" for s in scored)
    return "uncertain" if correct / len(scored) > PARTIAL_CREDIT_SHARE else "practice"


def build_payload(result, *, groups, word_annotation_status, boundaries_from, practice_mode, sample_id,
                  target_script, practice_item=None, audio_quality=None, partial_credit=False):
    syllables = result["syllables"]
    baseline = (result.get("f0_extraction") or {}).get("speaker_f0_median")
    phrases = []
    for label, start, end in groups:
        group = syllables[start:end]
        strict = tp.aggregate_status(group)
        status = display_status(group, partial_credit)
        window = (group[0]["start_ms"] / 1000.0, group[-1]["end_ms"] / 1000.0) if group and "start_ms" in group[0] else None
        flagged = next((s for s in group if s["status"] == "practice"), None)
        note = None
        if strict == "practice":
            note = flagged["feedback"]
        elif strict == "insufficient_evidence":
            note = (group[0]["feedback"] if all(s["status"] == "insufficient_evidence" for s in group)
                    else result["config"]["feedback"]["insufficient_evidence"])
        phrases.append({
            "word": "".join(s["character"] for s in group),
            "annotation": label,
            "expected_tone_sequence": [_tone_or_zero(s["expected_surface_tone"]) for s in group],
            "lexical_tone_sequence": [_tone_or_zero(s["lexical_tone"]) for s in group],
            "observed_tone_sequence": [None] * len(group),   # the verifier checks expected tones; it does not name a tone
            "status": status,
            "phrase_status": status,
            "strict_status": strict,                         # what the syllables alone say (status can be softer)
            "phrase_window_sec": [round(window[0], 3), round(window[1], 3)] if window else None,
            "phrase_f0": phrase_f0(result.get("contour"), *window, baseline) if window else None,
            "sequence_evidence": [
                {"character": s["character"], "status": s["status"], "tone_match": s["tone_match"],
                 "acoustic_quality": s["acoustic_quality"]} for s in group],
            "syllable_diagnostics": group,
            "assessment_note": note,
            "feedback_items": phrase_feedback(group),
        })

    counts = {k: sum(p["status"] == k for p in phrases) for k in (*tp.STATUSES, "uncertain", "not_heard")}
    yellow = counts["insufficient_evidence"] + counts["uncertain"] + counts["not_heard"]   # neither good nor practice
    total = len(syllables)
    return {
        "success": True,
        "mode": MODE,
        "practice_mode": practice_mode,
        "sample_id": sample_id,
        "reference_audio_used": False,
        "reference_audio_role": "optional_playback_only",
        "practice_item": practice_item,
        "target_script": target_script,
        "phrases": phrases,
        "word_segmentation": {
            "status": word_annotation_status,
            "scored_phrases": len(phrases),
            "phrase_boundaries_from": boundaries_from,
        },
        "transcript_checked": False,
        "transcript_check": {
            "status": "script_provided", "recognized_text": None,
            "matched_syllables": None, "total_syllables": total, "match_ratio": None,
        },
        "feedback": result["feedback"],
        "audio_quality": audio_quality,
        "tone_verification": {
            "method": result["method"],
            "calibrated": result["calibrated"],
            "calibration_version": result["calibration_version"],
            "acoustic_profile": result["acoustic_profile"],
            "alignment": {
                "aligner": {"mfa": "MFA (fixed transcript, no ASR)", "voiced_run": "voiced run (one syllable, no aligner)",
                            "injected": "supplied alignment"}.get(result["aligner"], result["aligner"]),
                "error": result["alignment_error"],
                "error_kind": alignment_error_kind(result["alignment_error"]),
                "boundary_search": result["boundary_search"],
            },
            "f0_extraction": result["f0_extraction"],
            "syllable_summary": result["summary"],
            "syllables": syllables,
        },
        "score_semantics": {
            "status": "conservative expected-tone verification from the learner's Praat F0; not a calibrated probability",
            "gating": "practice requires high acoustic quality and a high mismatch; otherwise insufficient_evidence",
            "transcript_check": "ASR is not used; the supplied script is force-aligned",
        },
        "summary": {
            "n_phrases": len(phrases),
            "n_assessable": counts["likely_correct"] + counts["practice"] + counts["uncertain"],
            "n_with_audio_evidence": sum(1 for p in phrases if p["phrase_f0"]),
            "likely_correct": counts["likely_correct"],
            "uncertain": yellow,
            "insufficient_evidence": yellow,
            "partly_correct": counts["uncertain"],
            "not_heard": counts["not_heard"],
            "practice": counts["practice"],
            "text_mismatch": 0,
            "transcript_unchecked": 0,
            "n_syllables": total,
            "syllables": result["summary"],
        },
    }


# ---------------------------------------------------------------- entry points ----

def sentence_contexts(clauses, groups, annotation_status):
    """Expected tones for sentence practice: word ends from the annotation no longer block sandhi (see capt_core)."""
    return capt_core.expected_tone_contexts(
        clauses, phrase_end_indices=[end for _, _, end in groups] if annotation_status == "matched" else None,
        cross_word_sandhi=True)


def _verify(prepared, clauses, contexts, calibration, force_mfa=False, config=None, model_voice_contexts=None,
            model_voice_reference_inputs=None):
    """`prepared` is audio_prep.PreparedAudio: a recording that failed the quality check is not analysed."""
    mode_contexts = {}
    if model_voice_contexts is not None:
        mode_contexts["model_voice_contexts"] = model_voice_contexts
    if model_voice_reference_inputs is not None:
        mode_contexts["model_voice_reference_inputs"] = model_voice_reference_inputs
    return tp.verify_utterance(prepared.samples, prepared.sample_rate, "".join(clauses), calibration,
                               contexts=contexts, config=config or get_config(), rejected_reasons=prepared.rejected_reasons,
                               force_mfa=force_mfa, **mode_contexts)


_REFERENCE_CONTOUR_LOCK = threading.RLock()


@lru_cache(maxsize=256)
def _cached_reference_voice_inputs(path, mtime_ns, size, script, profile, level, contexts_json, model_contexts_json):
    """Extract one aligned teacher contour per syllable; cache successful and unavailable references."""
    try:
        contexts = json.loads(contexts_json)
        model_contexts = json.loads(model_contexts_json) if model_contexts_json else None
        config = deepcopy(get_config(profile, level, model_voice.MODE_MODEL_VOICE))
        config.setdefault("model_voice", {})["capture_reference_inputs"] = True
        prepared = audio_prep.prepare_recording(path, config)
        if not prepared.passed:
            return ()
        calibration = tp.pooled_calibration([(prepared.samples, prepared.sample_rate)], config)
        result = tp.verify_utterance(
            prepared.samples, prepared.sample_rate, script, calibration,
            contexts=contexts, model_voice_contexts=model_contexts, config=config,
        )
        captured = result.get("model_voice_reference_inputs") or []
        if len(captured) != len(contexts):
            return ()
        return tuple(None if not item else (tuple(float(value) for value in item["values"]), float(item["spread"]))
                     for item in captured)
    except Exception:
        # Reference matching is optional: an unavailable or unalignable sample must not stop ordinary scoring.
        return ()


def model_voice_reference_inputs(audio_path, script, contexts, model_voice_contexts=None, *, level=None, profile="sentence"):
    """Return cached per-syllable contours for a known book recording, or None if it cannot be aligned."""
    try:
        path = Path(audio_path).resolve()
        stat = path.stat()
        contexts_json = json.dumps(contexts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        model_contexts_json = (json.dumps(model_voice_contexts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                               if model_voice_contexts is not None else "")
        with _REFERENCE_CONTOUR_LOCK:
            cached = _cached_reference_voice_inputs(
                str(path), stat.st_mtime_ns, stat.st_size, str(script), profile, level or STANDARD,
                contexts_json, model_contexts_json)
        if not cached:
            return None
        return [None if item is None else {"values": np.asarray(item[0], dtype=float), "spread": item[1]}
                for item in cached]
    except (OSError, TypeError, ValueError):
        return None


def _fuse_verification_result(result):
    """Combine the already-computed Rules and Model Voice verdicts without changing either scorer."""
    counts = {key: 0 for key in ("rules_only", "model_voice_only", "both", "both_practice", "unresolved")}
    for syllable in result["syllables"]:
        rules_status = syllable.get("rule_status")
        if rules_status is None:
            raise ValueError("Fusion requires a Rules verdict for every syllable.")

        model_status = syllable["status"]
        model_tone = syllable.get("expected_surface_tone")
        model_judged = model_tone == 5 or "model_score" in syllable
        syllable["rules_expected_surface_tone"] = syllable.get("rule_expected_surface_tone")
        syllable["model_voice_expected_surface_tone"] = model_tone
        syllable["model_voice_tone_rule"] = syllable.get("tone_rule")
        syllable["model_voice_status"] = model_status if model_judged else "insufficient_evidence"
        syllable["model_voice_judged"] = bool(model_judged and model_tone != 5)

        rules_good = rules_status == "likely_correct"
        model_good = model_judged and model_status == "likely_correct"
        both_practice = rules_status == "practice" and model_judged and model_status == "practice"
        if rules_good and model_good:
            basis, final = "both", "likely_correct"
        elif rules_good:
            basis, final = "rules_only", "likely_correct"
        elif model_good:
            basis, final = "model_voice_only", "likely_correct"
        elif both_practice:
            basis, final = "both_practice", "practice"
        else:
            basis, final = "unresolved", "insufficient_evidence"

        syllable["fusion_basis"] = basis
        syllable["status"] = final
        if basis == "rules_only":
            syllable["expected_surface_tone"] = syllable.get("rule_expected_surface_tone", model_tone)
            syllable["lexical_tone"] = syllable.get("rule_lexical_tone", syllable.get("lexical_tone"))
            syllable["tone_rule"] = syllable.get("rule_tone_rule", syllable.get("tone_rule"))
            syllable["tone_source"] = "rules"
            for key in ("tone_match", "tone_mismatch", "detected_alternative", "tone_scores", "also_accepted_tones", "clause_final_half_t3"):
                rule_key = f"rule_{key}"
                if rule_key in syllable:
                    syllable[key] = syllable[rule_key]
        counts[basis] += 1

    summary = {status: sum(item["status"] == status for item in result["syllables"])
               for status in tp.STATUSES}
    result["summary"] = summary
    overall = tp.aggregate_status(result["syllables"])
    if overall == "practice":
        expected = next(item["expected_surface_tone"] for item in result["syllables"] if item["status"] == "practice")
        result["feedback"] = result["config"]["feedback"]["practice"].get(expected)
    else:
        result["feedback"] = result["config"]["feedback"][overall]
    result["fusion"] = counts
    return result


def _fusion_payload_metadata(payload, result):
    payload["fusion"] = result.get("fusion") or {}
    payload["score_semantics"] = {
        **payload.get("score_semantics", {}),
        "status": "Fusion keeps both original verdicts: either likely-correct verdict is accepted; practice requires both scorers to say practice; otherwise evidence is insufficient.",
        "expected_tone": "A Rules-only acceptance displays the Rules target. Otherwise the displayed target follows the book Pinyin used by Model Voice.",
    }
    return payload


def assess_recording(prepared, *, clauses, contexts, groups, word_annotation_status,
                     calibration, practice_mode, sample_id, target_script, level=None, scoring_mode=None,
                     model_voice_contexts=None, model_voice_reference_inputs=None):
    """Book / custom practice: one script, grouped into phrases by word annotation or clause."""
    verify_args = {}
    if model_voice_contexts is not None:
        verify_args["model_voice_contexts"] = model_voice_contexts
    if model_voice_reference_inputs is not None:
        verify_args["model_voice_reference_inputs"] = model_voice_reference_inputs
    result = _verify(prepared, clauses, contexts, calibration, config=get_config("sentence", level, scoring_mode),
                     **verify_args)
    if scoring_mode == model_voice.MODE_FUSION:
        result = _fuse_verification_result(result)
    payload = build_payload(
        result, groups=groups, word_annotation_status=word_annotation_status,
        boundaries_from=("word-level JSON grouped over MFA syllable boundaries"
                         if word_annotation_status == "matched" else "punctuation clauses"),
        practice_mode=practice_mode, sample_id=sample_id, target_script=target_script,
        audio_quality=prepared.report(), partial_credit=True)
    payload["level"] = level or STANDARD
    payload["pace"] = pace_summary(payload["tone_verification"]["syllables"])
    payload["length_check"] = (payload["tone_verification"].get("f0_extraction") or {}).get("length_check")
    if scoring_mode == model_voice.MODE_FUSION:
        _fusion_payload_metadata(payload, result)
    _mark_scoring_mode(payload, scoring_mode)
    return payload


def _decided_fraction(result):
    syllables = result["syllables"]
    return sum(s["status"] != "insufficient_evidence" for s in syllables) / max(1, len(syllables))


def _candidate_key(result):
    """Rank (reading, path) pairs; higher is better.

    1. an alignment that worked, 2. the least voiced speech left unexplained (a path that covers what was said),
    3. fewest `practice` syllables, 4. most of the scored syllables accepted, 5. most syllables decided, 6. quality.
    Unexplained speech comes before the verdict, so a learner who says the right word with a wrong tone is not moved
    onto some other variant just because that one happens to look better."""
    syllables = result["syllables"]
    scored = [s for s in syllables if s.get("expected_surface_tone", 0) != 5]
    quality = float(np.mean([s["acoustic_quality"] for s in syllables])) if syllables else 0.0
    outside = (result.get("f0_extraction") or {}).get("voiced_outside_syllables")
    unexplained = 1.0 if outside is None else outside
    accepted = sum(s["status"] == "likely_correct" for s in scored) / max(1, len(scored))
    return (0 if result["alignment_error"] else 1, -round(unexplained / 0.05),
            -sum(s["status"] == "practice" for s in syllables), round(accepted, 2), _decided_fraction(result), quality)


def best_variant(prepared, candidates, verify, config=None):
    """Pick ONE (reading, path) to score: the part of the recording that matches one of the variants.

    A clip may hold more than one reading (裡面 then 裡, 哪裡 then 哪兒) and a path never has to explain all of it.
    With several variants the whole recording and each separate reading are scored against each path, and the best
    pair wins (alignment succeeded, least speech left unexplained, most syllables decided, best quality).
    With one variant, or one reading, this is a single plain run. `verify(audio, candidate)` returns a result."""
    config = config or get_config()
    windows = [None]                                           # None = the whole recording
    if len(candidates) > 1 and prepared.passed:
        chunks = audio_prep.speech_chunks(prepared, config)
        if len(chunks) > 1:
            windows += chunks
    runs = []
    for window in windows:
        audio = prepared if window is None else audio_prep.slice_prepared(prepared, *window)
        for candidate in candidates:
            runs.append((candidate, window, verify(audio, candidate)))
    candidate, window, result = max(runs, key=lambda run: _candidate_key(run[2]))
    tried = [{"script_variant": c["script_variant"], "reading": None if w is None else [round(w[0], 2), round(w[1], 2)],
              "decided_fraction": round(_decided_fraction(r), 3),
              "voiced_outside_syllables": (r.get("f0_extraction") or {}).get("voiced_outside_syllables"),
              "alignment_error": r["alignment_error"]} for c, w, r in runs]
    return candidate, window, result, tried


def _mark_scoring_mode(payload, scoring_mode):
    """Say which verdict the syllables carry, in the payload and in the summary that is saved with the attempt (the boards read it there)."""
    mode = scoring_mode or model_voice.MODE_RULES
    payload["scoring_mode"] = mode
    payload["summary"]["scoring_mode"] = mode


def assess_chapter5_item(prepared, *, item, sample_id, target_script, calibration, normalize, level=None, scoring_mode=None):
    """Chapter 5 word: score the reading in the recording that matches one of the allowed variants."""
    candidates = [c for c in chapter5_candidates(item, target_script, normalize)
                  if len(chapter5_contexts(c["clauses"], item)) == len(c["chars"])]
    if not prepared.passed:
        candidates = candidates[:1]   # nothing is analysed, so there is nothing to choose between
    if not candidates:
        raise ValueError("The script contains no valid Han characters to score pronunciation.")

    def verify(audio, candidate):
        # With several variants every one is force-aligned so they compare fairly.
        contexts = chapter5_contexts(candidate["clauses"], item)
        model_contexts = None
        if scoring_mode in model_voice.MODES_USING_MODEL_VOICE:
            model_contexts = chapter5_book_contexts(candidate["clauses"], item, candidate["script_variant"], normalize)
        verify_args = {} if model_contexts is None else {"model_voice_contexts": model_contexts}
        reference_inputs = None
        reference_script = "".join(capt_core.prepare_target(str(item.get("script") or "")))
        if (model_contexts is not None and "".join(candidate["chars"]) == reference_script
                and item.get("audio_path")):
            reference_inputs = model_voice_reference_inputs(
                item["audio_path"], reference_script, contexts, model_contexts,
                level=level, profile=None)
        if reference_inputs is not None:
            verify_args["model_voice_reference_inputs"] = reference_inputs
        result = _verify(audio, candidate["clauses"], contexts, calibration,
                         force_mfa=len(candidates) > 1, config=get_config(None, level, scoring_mode),
                         **verify_args)
        return _fuse_verification_result(result) if scoring_mode == model_voice.MODE_FUSION else result

    candidate, window, result, tried = best_variant(prepared, candidates, verify)
    payload = build_payload(
        result, groups=[(item["traditional_chinese"], 0, len(candidate["chars"]))],
        word_annotation_status="selected_chapter5_item",
        boundaries_from="MFA forced alignment of the supplied or allowed text variant",
        practice_mode="chapter5_word", sample_id=sample_id, target_script=target_script,
        practice_item={key: item[key] for key in
                       ("word_key", "item_type", "traditional_chinese", "script", "pinyin", "section")},
        audio_quality=prepared.report())
    payload["level"] = level or STANDARD
    payload["pace"] = pace_summary(payload["tone_verification"]["syllables"])
    payload["length_check"] = (payload["tone_verification"].get("f0_extraction") or {}).get("length_check")
    payload["aligned_script_variant"] = candidate["script_variant"]
    payload["tone_verification"]["scored_reading"] = None if window is None else {
        "start_sec": round(window[0], 2), "end_sec": round(window[1], 2),
        "note": "the recording holds more than one reading; only this one was scored"}
    payload["tone_verification"]["script_variants_tried"] = tried
    if scoring_mode == model_voice.MODE_FUSION:
        _fusion_payload_metadata(payload, result)
    _mark_scoring_mode(payload, scoring_mode)
    return payload
