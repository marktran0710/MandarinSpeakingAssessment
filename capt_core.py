"""
Script and tone-context helpers shared by the API (main.py), tone_assessment and tone_pipeline: reduce a script to
Han clauses and work out each syllable's expected (surface) tone. No audio model is loaded here.
"""
import re

import numpy as np
import opencc
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_tone

_T2S = opencc.OpenCC("t2s")      # character for character; only used to look phrases up in pypinyin

# ---------------------------------------------------------------- script ----

def normalize_audio(samples):
    """Chuẩn hóa biên độ âm thanh (Peak Normalization) về khoảng [-1.0, 1.0] 
       để đồng bộ mức âm lượng giữa các học viên nói to hoặc nhỏ khác nhau."""
    samples = np.asarray(samples, dtype=np.float32)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        return samples / max_val
    return samples

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

def simplified(text):
    """The Simplified form of `text` for dictionary look-ups (pypinyin, jieba); the text itself if the length would change."""
    converted = _T2S.convert(text)
    return converted if len(converted) == len(text) else text


def pinyin_per_character(text):
    """One entry per character of `text`, so a page can put it above the character: pinyin for Han, '' for the rest."""
    out = []
    for run in re.findall(r"[一-龥]+|[^一-龥]+", text):
        out.extend(display_pinyin(run) if is_chinese_char(run[0]) else [""] * len(run))
    return out


def display_pinyin(text, tones=None):
    """Pinyin with tone marks to show next to a script: one entry per character, like lazy_pinyin(text, style=TONE).

    pypinyin's phrase dictionary is keyed by Simplified characters, so Traditional 音樂 comes out as yīn lè. The letters
    are therefore read from the Simplified form (yīn yuè), while the tones are the ones this app expects and scores
    (_base_tones: Taiwan readings, 個 gè, 隻 zhī), so what is shown matches the tone sequence the learner is checked on.
    Anything that is not a Han character is passed through as before. `tones` (one per Han character, in order) replaces
    the default tones, e.g. the catalog's own reading of 沒（有）空 (kòng); it is ignored if the count does not match."""
    runs = re.findall(r"[一-龥]+|[^一-龥]+", text)
    han = sum(len(run) for run in runs if is_chinese_char(run[0]))
    if tones is not None and len(tones) != han:
        tones = None
    out, cursor = [], 0
    for run in runs:
        if not is_chinese_char(run[0]):
            out.extend(lazy_pinyin(run, style=Style.TONE))
            continue
        simplified = _T2S.convert(run)
        letters = lazy_pinyin(simplified if len(simplified) == len(run) else run, style=Style.NORMAL, v_to_u=True)
        run_tones = list(tones[cursor:cursor + len(run)]) if tones is not None else _base_tones(run)
        cursor += len(run)
        for letter, tone in zip(letters, run_tones):
            out.append(letter if tone == 5 else to_tone(f"{letter}{tone}"))
    return out


# Taiwan Mandarin readings differ from mainland Pinyin readings.
TAIWAN_WORD_TONES = {
    "星期": [1, 2],
    "垃圾": [4, 4],
    "覺得": [2, 5],
    "睡覺": [4, 4],             # shuìjiào; pypinyin reads Traditional 覺 as jué
    "衣服": [1, 5],
    # Reduplicated kinship terms: neutral second syllable (book Pinyin, MOE).
    "哥哥": [1, 5],
    "姐姐": [3, 5],
    "姊姊": [3, 5],             # variant character of 姐姐
    "妹妹": [4, 5],
    "弟弟": [4, 5],
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

def _clause_tones(clause, additional_phrase_breaks=None):
    base = _base_tones(clause)
    tones = list(base)
    n = len(clause)
    break_after = set()
    for word in PHRASE_BREAK_AFTER:
        pos = clause.find(word)
        while pos != -1:
            break_after.add(pos + len(word) - 1)
            pos = clause.find(word, pos + len(word))
    break_after.update(
        index for index in (additional_phrase_breaks or ())
        if isinstance(index, int) and 0 <= index < n
    )
    for i, ch in enumerate(clause):
        nxt = base[i + 1] if i + 1 < n and i not in break_after else None
        prev_ch = clause[i - 1] if i > 0 else ""
        next_ch = clause[i + 1] if i + 1 < n and i not in break_after else ""
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

def expected_tones(clauses, *, phrase_end_indices=None):
    """Return expected tones; sandhi does not cross punctuation or supplied phrase ends."""
    return [
        item["target_tone"]
        for item in expected_tone_contexts(clauses, phrase_end_indices=phrase_end_indices)
    ]


def expected_tone_contexts(clauses, *, phrase_end_indices=None, cross_word_sandhi=False):
    """Return lexical/surface tones and context, respecting phrase ends in flattened text.

    By default every phrase end blocks tone sandhi. With `cross_word_sandhi` (sentence practice) the ends that come from
    word annotations stop blocking it: 不 and 一 take their sandhi tone before the next word, and a third tone before
    another third tone keeps T3 as the target but also accepts T2 (`also_accept_tones`), because a teacher reads
    我也 / 你想 / 很好 with sandhi even though the words are annotated separately. Punctuation and the fixed phrase
    breaks (所以, 哪裡) still block sandhi."""
    result = []
    global_index = 0
    phrase_end_indices = {
        int(index) for index in (phrase_end_indices or ())
        if isinstance(index, (int, np.integer)) and int(index) > 0
    }
    for clause_index, clause in enumerate(clauses):
        lexical = _base_tones(clause)
        break_after = set()
        taiwan_override_positions = set()
        for phrase in PHRASE_BREAK_AFTER:
            pos = clause.find(phrase)
            while pos != -1:
                break_after.add(pos + len(phrase) - 1)
                pos = clause.find(phrase, pos + len(phrase))
        explicit_breaks = {
            phrase_end - global_index - 1
            for phrase_end in phrase_end_indices
            if global_index < phrase_end <= global_index + len(clause)
        }
        hard_breaks = set(break_after)
        break_after.update(explicit_breaks)
        surface = _clause_tones(clause, additional_phrase_breaks=break_after)
        crossed, also_accept = set(), {}
        if cross_word_sandhi and explicit_breaks:
            across = _clause_tones(clause, additional_phrase_breaks=hard_breaks)
            for i, ch in enumerate(clause):
                if ch in "不一" and across[i] != surface[i]:
                    surface[i] = across[i]
                    crossed.add(i)
                elif lexical[i] == 3 and surface[i] == 3 and across[i] == 2:
                    also_accept[i] = 2
        for word in TAIWAN_WORD_TONES:
            pos = clause.find(word)
            while pos != -1:
                taiwan_override_positions.update(range(pos, pos + len(word)))
                pos = clause.find(word, pos + len(word))

        for index, char in enumerate(clause):
            starts_phrase = index == 0 or index - 1 in break_after
            ends_phrase = index in break_after
            following = lexical[index + 1] if index + 1 < len(clause) and not ends_phrase else None
            preceding_char = clause[index - 1] if index and not starts_phrase else None
            following_char = clause[index + 1] if index + 1 < len(clause) and not ends_phrase else None
            rule = "lexical_tone"
            if index in taiwan_override_positions:
                rule = "taiwan_lexical_reading"
            if char == "不" and preceding_char and preceding_char == following_char:
                rule = "bu_reduplication_neutral"
            elif char == "不" and lexical[index] == 4 and surface[index] != lexical[index]:
                rule = "bu_sandhi"
            elif char == "一" and surface[index] != lexical[index]:
                rule = "yi_sandhi"
            elif lexical[index] == 3 and surface[index] == 2:
                rule = "third_tone_sandhi"
            elif index in break_after and lexical[index] == 3:
                rule = "phrase_final_third_tone"
            if index in crossed:
                rule = ("bu_reduplication_neutral" if surface[index] == 5
                        else "bu_sandhi" if char == "不" else "yi_sandhi")
            elif index in also_accept:
                rule = "third_tone_sandhi_optional"

            result.append({
                "index": global_index,
                "char": char,
                "lexical_tone": lexical[index],
                "target_tone": surface[index],
                "tone_rule": rule,
                "previous_char": preceding_char,
                "previous_tone": surface[index - 1] if index and not starts_phrase else None,
                "next_char": following_char,
                "next_tone": surface[index + 1] if index + 1 < len(clause) and not ends_phrase else None,
                "clause_index": clause_index,
                "index_in_clause": index,
                "clause_length": len(clause),
                "is_clause_final": index == len(clause) - 1,
                "is_phrase_final": ends_phrase,
                "allow_half_t3": (
                    lexical[index] == 3
                    and surface[index] == 3
                    and index + 1 < len(clause)
                    and not ends_phrase
                    and following in {1, 2, 4, 5}
                ),
                **({"also_accept_tones": (also_accept[index],)} if index in also_accept else {}),
            })
            global_index += 1
    return result

