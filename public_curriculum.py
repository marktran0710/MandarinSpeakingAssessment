"""Versioned comprehension exercises built from the verified textbook translations."""
from collections import defaultdict

PROMPTS = {
    'en': {'listen': 'What does the line in scene {scene} mean?', 'read': 'What does the sentence in scene {scene} mean?', 'explain': 'Compare the answer with the original line and its translation.'},
    'zh-TW': {'listen': '第 {scene} 幕聽到的句子是什麼意思？', 'read': '第 {scene} 幕的句子是什麼意思？', 'explain': '請對照原句和譯文複習。'},
    'vi': {'listen': 'Câu nghe ở cảnh {scene} có nghĩa là gì?', 'read': 'Câu ở cảnh {scene} có nghĩa là gì?', 'explain': 'Đối chiếu đáp án với câu gốc và bản dịch.'},
    'id': {'listen': 'Apa arti kalimat yang terdengar pada adegan {scene}?', 'read': 'Apa arti kalimat pada adegan {scene}?', 'explain': 'Bandingkan jawaban dengan kalimat asli dan terjemahannya.'},
    'hi': {'listen': 'दृश्य {scene} में सुने गए वाक्य का क्या अर्थ है?', 'read': 'दृश्य {scene} के वाक्य का क्या अर्थ है?', 'explain': 'उत्तर की तुलना मूल वाक्य और अनुवाद से करें।'},
}


def stories(samples):
    groups = defaultdict(list)
    for sample in samples.values():
        groups[f"{sample['lesson']}-{sample['story']}"].append(sample)
    return {key: sorted(rows, key=lambda row: row['scene']) for key, rows in sorted(groups.items(), key=lambda item: tuple(map(int, item[0].split('-'))))}


def translated(sample, locale, meanings):
    if locale == 'zh-TW':
        return sample['script']
    return meanings.get(sample['id'], {}).get(locale) or meanings.get(sample['id'], {}).get('en') or sample['script']


def questions(story_id, skill, locale, samples, meanings):
    rows = stories(samples).get(story_id)
    if not rows or skill not in ('listen', 'read'):
        return None
    locale = locale if locale in PROMPTS else 'en'
    # Three distinct lines spread through the story. Four answer choices each use
    # published translations of real lines, with no fabricated textbook content.
    targets = [rows[0], rows[len(rows)//2], rows[-1]]
    result = []
    for index, target in enumerate(targets):
        alternatives = [row for row in rows if row['id'] != target['id'] and translated(row, locale, meanings) != translated(target, locale, meanings)]
        alternatives.sort(key=lambda row: (abs(row['scene'] - target['scene']), row['scene']))
        chosen = [target] + alternatives[:3]
        chosen.sort(key=lambda row: (row['scene'] * 7 + index * 11) % 13)
        result.append({
            'id': f'{skill}-{story_id}-{index+1}',
            'scene': target['scene'],
            'prompt': PROMPTS[locale][skill].format(scene=target['scene']),
            'options': [{'id': row['id'], 'text': translated(row, locale, meanings)} for row in chosen],
            'answer': target['id'],
            'explanation': PROMPTS[locale]['explain'],
        })
    return result
