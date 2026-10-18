import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
NODE = shutil.which("node")


def function_source(name):
    start = SOURCE.index(f"function {name}(")
    end = SOURCE.index("\n            }\n", start) + len("\n            }\n")
    return SOURCE[start:end]


def detect(languages=None, language=None, broken=False):
    """Run the page's own detectBrowserLanguage() in node against a fake navigator."""
    navigator = "{ get languages() { throw new Error('no'); } }" if broken else json.dumps({"languages": languages, "language": language})
    script = f"const navigator = {navigator};\n{function_source('detectBrowserLanguage')}\nconsole.log(detectBrowserLanguage());"
    return subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout.strip()


@unittest.skipUnless(NODE, "node is needed to run the page's JavaScript")
class BrowserLanguageTests(unittest.TestCase):
    def test_each_supported_language_is_found_from_the_browser_settings(self):
        for languages, expected in ((["vi-VN", "vi"], "vi"), (["vi"], "vi"), (["id-ID"], "id"), (["in"], "id"), (["hi-IN"], "hi"),
                                    (["zh-TW"], "zh-TW"), (["zh-Hant-TW"], "zh-TW"), (["zh-HK"], "zh-TW"), (["zh-CN"], "zh-TW"), (["en-US"], "en"), (["vi;q=0.9"], "vi"), (["VI_vn"], "vi")):
            self.assertEqual(detect(languages), expected, languages)

    def test_the_first_language_of_the_list_that_the_page_has_wins(self):
        self.assertEqual(detect(["fr-FR", "vi", "en"]), "vi")            # French is not offered: the next one counts
        self.assertEqual(detect(["en-GB", "vi"]), "en")                  # someone who puts English first reads English
        self.assertEqual(detect(["de", "ja"]), "en")                     # nothing offered: English

    def test_an_old_browser_with_only_one_language_value_still_works(self):
        self.assertEqual(detect(None, "vi-VN"), "vi")
        self.assertEqual(detect([], "hi"), "hi")
        self.assertEqual(detect(None, None), "en")

    def test_a_browser_that_throws_gets_english(self):
        self.assertEqual(detect(broken=True), "en")


class PageWiringTests(unittest.TestCase):
    def test_a_first_visit_uses_the_detected_language_without_storing_it_and_a_chosen_one_is_kept(self):
        self.assertIn('applyLanguage(savedLanguage() || detectBrowserLanguage(), Boolean(savedLanguage()));', SOURCE)
        self.assertIn("function applyLanguage(language, remember = true) {", SOURCE)
        self.assertRegex(SOURCE, r"function changeLanguage\(language\) \{\s*applyLanguage\(language\);")           # the picker still stores the learner's choice

    def test_the_speaker_profile_window_has_its_own_language_picker_with_the_same_languages_in_step_with_the_page(self):
        dialog = SOURCE[SOURCE.index('<dialog id="speakerDialog"'): SOURCE.index("</dialog>", SOURCE.index('<dialog id="speakerDialog"'))]
        page = SOURCE[SOURCE.index('<select id="languageSelect"'): SOURCE.index("</select>", SOURCE.index('<select id="languageSelect"'))]
        picker = dialog[dialog.index('<select id="dialogLanguage"'): dialog.index("</select>", dialog.index('<select id="dialogLanguage"'))]
        self.assertIn('onchange="changeLanguage(this.value)"', picker)                          # the same function: the choice is remembered and applied at once
        self.assertEqual(re.findall(r'value="([^"]+)"', picker), re.findall(r'value="([^"]+)"', page))
        self.assertIn('document.getElementById("dialogLanguage").value = currentLanguage;', SOURCE)   # it starts on the language the page has (the browser's on a first visit)
        opening = SOURCE[SOURCE.index("function openStudentDialog("): SOURCE.index("function closeStudentDialog(")]
        self.assertLess(opening.index("form.reset();"), opening.index("dialogLanguage"))                   # ... also after the form was reset (which picks the first option)

    def test_the_server_never_looks_at_the_ip_address_for_the_language(self):
        self.assertNotIn("geoip", SOURCE.lower())
        self.assertNotIn("x-forwarded-for", SOURCE.lower())


if __name__ == "__main__":
    unittest.main()
