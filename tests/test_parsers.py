"""Tests for the pure functions that earned them: the teacher-output parsers
(three JSONDecodeError rounds in production), path sanitisation, and the
mastery math shared by solo and cohort runs.

Run:  python -m pytest tests/ -q   (or python -m unittest discover tests)
"""
import unittest

from rl_learning.llm.teacher import _fix_escapes, _parse_json_array, _parse_qa, _mask_secrets
from rl_learning.university import GENERAL_SET, _overlap, _safe, mastery_from


class TestParseQA(unittest.TestCase):
    def test_valid_array(self):
        raw = '[{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}]'
        self.assertEqual(len(_parse_qa(raw)), 2)

    def test_code_fenced(self):
        raw = '```json\n[{"question": "Q?", "answer": "A."}]\n```'
        qa = _parse_qa(raw)
        self.assertEqual(qa, [{"question": "Q?", "answer": "A."}])

    def test_latex_escapes_survive(self):
        # \nabla, \(x\), \epsilon are invalid JSON escapes — must not crash
        raw = '[{"question": "grad?", "answer": "Use \\nabla f, \\(x\\), \\epsilon"}]'
        qa = _parse_qa(raw)
        self.assertEqual(len(qa), 1)
        self.assertIn("epsilon", qa[0]["answer"])

    def test_literal_newline_in_string(self):
        raw = '[{"question": "q", "answer": "line1\nline2"}]'
        qa = _parse_qa(raw)
        self.assertEqual(qa[0]["answer"], "line1\nline2")

    def test_truncated_array_salvaged(self):
        raw = '[{"question": "q1", "answer": "a1"}, {"question": "q2", "ans'
        qa = _parse_qa(raw)
        self.assertEqual(len(qa), 1)
        self.assertEqual(qa[0]["question"], "q1")

    def test_preamble_before_array(self):
        raw = 'Here is your JSON:\n[{"question": "q", "answer": "a"}]'
        self.assertEqual(len(_parse_qa(raw)), 1)

    def test_missing_keys_skipped(self):
        raw = '[{"question": "q"}, {"answer": "a"}, {"question": "q2", "answer": "a2"}]'
        self.assertEqual(len(_parse_qa(raw)), 1)

    def test_garbage_returns_empty(self):
        self.assertEqual(_parse_qa("I cannot answer that."), [])
        self.assertEqual(_parse_qa(""), [])


class TestParseJsonArray(unittest.TestCase):
    def test_curriculum_objects(self):
        raw = '[{"subject": "A", "why": "w", "depth_topics": ["x"]}]'
        out = _parse_json_array(raw)
        self.assertEqual(out[0]["subject"], "A")

    def test_truncated_with_latex(self):
        raw = '[{"subject": "A", "why": "uses \\alpha"}, {"subject": "B", "wh'
        out = _parse_json_array(raw)
        self.assertEqual(len(out), 1)


class TestFixEscapes(unittest.TestCase):
    def test_doubles_invalid_only(self):
        self.assertEqual(_fix_escapes(r'"\epsilon"'), r'"\\epsilon"')
        # valid escapes untouched
        self.assertEqual(_fix_escapes(r'"\n \t \" \\ é"'), r'"\n \t \" \\ é"')


class TestMaskSecrets(unittest.TestCase):
    def test_masks_api_keys(self):
        # NOT a real key — an obviously-fake fixture so nobody (or a scanner)
        # mistakes it for a credential. It only has to match the masker's regex.
        msg = "Error code 429 for key sk-proj-THIS-IS-A-FAKE-EXAMPLE-KEY-000 retry later"
        self.assertNotIn("FAKE-EXAMPLE", _mask_secrets(msg))
        self.assertIn("sk-***", _mask_secrets(msg))


class TestSafe(unittest.TestCase):
    def test_strips_path_traversal(self):
        self.assertNotIn("..", _safe("../../etc/passwd"))
        self.assertNotIn("/", _safe("a/b"))
        self.assertNotIn("\\", _safe("a\\b"))

    def test_empty_defaults(self):
        self.assertEqual(_safe("///"), "model")

    def test_keeps_normal_names(self):
        self.assertEqual(_safe("rl-graduate"), "rl-graduate")
        self.assertEqual(_safe("q-Ph 2"), "q-Ph 2")


class TestMastery(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(mastery_from(["pass", "pass", "pass", "pass"]), 1.0)
        self.assertEqual(mastery_from(["fail", "fail"]), 0.0)
        self.assertEqual(mastery_from(["pass", "partial"]), 0.75)
        self.assertEqual(mastery_from(["pass", "partial", "fail", "fail"]), 0.375)

    def test_empty_is_zero(self):
        self.assertEqual(mastery_from([]), 0.0)


class TestLibraryRetrieval(unittest.TestCase):
    def test_overlap_relevance_ordering(self):
        q = "What is the Bellman equation in reinforcement learning?"
        relevant = "Explain the Bellman equation and how reinforcement learning uses it."
        irrelevant = "Describe the French Revolution's main causes."
        self.assertGreater(_overlap(q, relevant), _overlap(q, irrelevant))

    def test_overlap_ignores_stopwords(self):
        self.assertEqual(_overlap("what is the a of", "how does the an in"), 0.0)

    def test_overlap_bounded(self):
        s = "quantum entanglement basics"
        self.assertEqual(_overlap(s, s), 1.0)


class TestGeneralSet(unittest.TestCase):
    def test_shape(self):
        self.assertGreaterEqual(len(GENERAL_SET), 5)
        for q, a in GENERAL_SET:
            self.assertTrue(q.strip() and a.strip())


if __name__ == "__main__":
    unittest.main()
