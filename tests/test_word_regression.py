import unittest

import word_regression as wr


class EntryTests(unittest.TestCase):
    def test_numbers_are_rounded_and_missing_values_stay_none(self):
        self.assertEqual(wr.entry("practice", 0.12345, 0.98765, ["a", "b"]), ["practice", 0.123, 0.988, "a|b"])
        self.assertEqual(wr.entry("insufficient_evidence", None, "", []), ["insufficient_evidence", None, None, ""])

    def test_a_csv_style_reason_string_is_kept_as_is(self):
        self.assertEqual(wr.entry("likely_correct", "1.0", "0.9", "x|y")[3], "x|y")


class CompareTests(unittest.TestCase):
    GOLDEN = {"s": {"a#0": ["likely_correct", 1.0, 0.9, ""], "a#1": ["practice", 0.1, 0.95, ""]}}

    def test_identical_snapshots_have_no_differences(self):
        self.assertEqual(wr.compare(self.GOLDEN, {"s": dict(self.GOLDEN["s"])}), {})

    def test_a_changed_status_is_reported(self):
        current = {"s": {**self.GOLDEN["s"], "a#1": ["insufficient_evidence", 0.1, 0.95, "x"]}}
        report = wr.compare(self.GOLDEN, current)
        self.assertEqual(list(report["s"]["changed"]), ["a#1"])
        self.assertIn("a#1", wr.describe(report))

    def test_even_a_tiny_quality_change_fails(self):
        current = {"s": {**self.GOLDEN["s"], "a#0": ["likely_correct", 1.0, 0.901, ""]}}
        self.assertIn("a#0", wr.compare(self.GOLDEN, current)["s"]["changed"])

    def test_appearing_and_disappearing_syllables_fail(self):
        current = {"s": {"a#0": self.GOLDEN["s"]["a#0"], "b#0": ["likely_correct", 1.0, 0.9, ""]}}
        report = wr.compare(self.GOLDEN, current)["s"]
        self.assertEqual((report["missing"], report["extra"]), (["a#1"], ["b#0"]))

    def test_a_suite_absent_from_the_current_run_is_reported(self):
        self.assertIn("absent", wr.describe(wr.compare(self.GOLDEN, {})))


if __name__ == "__main__":
    unittest.main()
