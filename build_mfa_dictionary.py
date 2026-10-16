"""
Build the MFA dictionary used by tone_pipeline: the stock `mandarin_taiwan_mfa` dictionary plus a
single-character entry for every Han character the app can ask for that the stock dictionary lacks.

The pipeline feeds MFA one character per "word" (no tokenization), so each character needs its own entry;
a missing one aligns as <unk> and the whole utterance becomes `insufficient_evidence`.

    python build_mfa_dictionary.py            # rebuilds mfa/mandarin_taiwan_catalog.dict from the app's catalog
    python build_mfa_dictionary.py --xlsx question_bank.xlsx   # ... and every word of the question-bank workbook
    python build_mfa_dictionary.py 巧克力 咖啡  # also covers extra characters

Pronunciations are borrowed, not invented: for a missing character with pinyin (syllable, tone) we take the
phones of an existing single-character entry with the same syllable and swap the tone letters for the target
tone's letters (learned from the stock dictionary itself). Review the printed list before relying on it.
"""
from __future__ import annotations

import collections
import os
import re
import sys
from pathlib import Path

from pypinyin import Style, lazy_pinyin

BASE_DICT = Path(os.environ.get(
    "MFA_BASE_DICT", r"D:\CodexData\mfa-runtime\models\pretrained_models\dictionary\mandarin_taiwan_mfa.dict"))
OUTPUT = Path(__file__).with_name("mfa") / "mandarin_taiwan_catalog.dict"
TONE_LETTERS = "˥˦˧˨˩"
DEFAULT_COLUMNS = ("1.0", "0.25", "1.0", "1.0")   # pron prob, silence-after prob, sil / non-sil correction
HAN = re.compile(r"^[一-鿿]$")
# Syllables with no single-character entry to borrow from, composed by hand from existing entries.
MANUAL = {
    "桌": "ʈʂ w o˥",   # zhuo1 = initial of 週 (ʈʂ ow˥) + final of 說 (ʂ w o˥)
}


def read_dictionary(path):
    entries = collections.defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 6:
            entries[parts[0]].append((float(parts[1]), parts[-1]))
    return entries


def tone_letters(phones):
    """The tone-letter string on the tone-bearing phone, e.g. 'aw˨˩˦' -> '˨˩˦' ('' if none)."""
    for phone in phones.split():
        letters = "".join(c for c in phone if c in TONE_LETTERS)
        if letters:
            return letters
    return ""


def swap_tone(phones, new_letters):
    out = []
    for phone in phones.split():
        if any(c in TONE_LETTERS for c in phone):
            phone = "".join(c for c in phone if c not in TONE_LETTERS) + new_letters
        out.append(phone)
    return " ".join(out)


def syllable_and_tone(char):
    reading = lazy_pinyin(char, style=Style.TONE3, neutral_tone_with_five=True)[0]
    tone = int(reading[-1]) if reading[-1].isdigit() else 5
    return reading.rstrip("012345"), tone


def tone_letter_map(entries):
    """tone number -> most common tone-letter string among single characters whose default reading has that tone."""
    votes = collections.defaultdict(collections.Counter)
    for char, pronunciations in entries.items():
        if HAN.match(char):
            _, tone = syllable_and_tone(char)
            for _, phones in pronunciations:
                if tone_letters(phones):
                    votes[tone][tone_letters(phones)] += 1
    return {tone: counter.most_common(1)[0][0] for tone, counter in votes.items()}


def pronunciation_for(char, entries, by_syllable, letters):
    syllable, tone = syllable_and_tone(char)
    target = letters.get(tone)
    if target is None:
        return None
    candidates = by_syllable.get(syllable, [])
    same_tone = [p for c, t, p in candidates if t == tone and tone_letters(p) == target]
    pool = same_tone or [swap_tone(p, target) for _, _, p in candidates]
    return collections.Counter(pool).most_common(1)[0][0] if pool else None


def catalog_characters():
    """Every Han character the app can ask to align (lesson samples, Chapter 5 items and their variants)."""
    import capt_core
    import main
    import tone_assessment

    scripts = [s["script"] for s in main.SAMPLE_CATALOG.values()]
    for item in main.CHAPTER5_WORD_ITEMS.values():
        scripts += [item["script"], *item.get("script_variants", [])]
    for words in main.WORD_ANNOTATIONS.values():
        scripts += words
    chars = set()
    for script in scripts:
        for variant in tone_assessment.chapter5_optional_text_variants(main.converter.convert(script)):
            chars.update("".join(capt_core.prepare_target(variant)))
    return chars


def workbook_characters(xlsx_path):
    """Han characters of every vocabulary / phrase / question word in the question-bank workbook (all chapters)."""
    import capt_core
    import xlsx_reader

    chars = set()
    for name, rows in xlsx_reader.read_xlsx(xlsx_path).items():
        if name not in {"Questions"} and not name.startswith("Chapter"):
            continue
        header = next((i for i, row in enumerate(rows) if "Traditional Chinese" in row), None)
        if header is None:
            continue
        column = rows[header].index("Traditional Chinese")
        for row in rows[header + 1:]:
            if len(row) > column and row[column]:
                for variant in row[column].split("/"):
                    chars.update("".join(capt_core.prepare_target(variant.replace("(", "").replace(")", "")
                                                                  .replace("（", "").replace("）", ""))))
    return chars


def build(extra_chars=(), xlsx_path=None):
    entries = read_dictionary(BASE_DICT)
    letters = tone_letter_map(entries)
    by_syllable = collections.defaultdict(list)
    for char, pronunciations in entries.items():
        if HAN.match(char):
            syllable, tone = syllable_and_tone(char)
            for _, phones in sorted(pronunciations, reverse=True):
                by_syllable[syllable].append((char, tone, phones))
    wanted = catalog_characters() | set("".join(extra_chars)) | (workbook_characters(xlsx_path) if xlsx_path else set())
    missing = sorted(c for c in wanted if HAN.match(c) and c not in entries)
    added, unresolved = [], []
    for char in missing:
        phones = MANUAL.get(char) or pronunciation_for(char, entries, by_syllable, letters)
        (added if phones else unresolved).append((char, phones))
    lines = BASE_DICT.read_text(encoding="utf-8").rstrip("\n").split("\n")
    lines += ["\t".join([char, *DEFAULT_COLUMNS, phones]) for char, phones in added]
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return added, unresolved, len(wanted)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = sys.argv[1:]
    workbook = None
    if "--xlsx" in arguments:
        at = arguments.index("--xlsx")
        workbook = arguments[at + 1]
        arguments = arguments[:at] + arguments[at + 2:]
    added, unresolved, wanted = build(arguments, workbook)
    print(f"{wanted} distinct characters needed; {len(added)} added to {OUTPUT}")
    for char, phones in added:
        print(f"  {char}  {'/'.join(lazy_pinyin(char, style=Style.TONE3))}  {phones}")
    if unresolved:
        print("COULD NOT RESOLVE (no same-syllable entry to borrow from):", "".join(c for c, _ in unresolved))
