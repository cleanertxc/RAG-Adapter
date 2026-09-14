import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('evaluate_mc', Path(__file__).parents[1]/'scripts/evaluate_mc.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationTests(unittest.TestCase):
    def test_formats_and_option_sets(self):
        for response in ['B', '(B)', 'B. A person is running.', 'The best answer is: B', '**B**']:
            self.assertEqual(MODULE.parse_answer(response, 'ABCD'), 'B')
        self.assertEqual(MODULE.parse_answer('E', 'ABCDE'), 'E')
        self.assertIsNone(MODULE.parse_answer('D', 'ABC'))

    def test_no_guessing_and_conflicting_answers(self):
        for response in ['', None, 'A person is standing.', 'I do not know.', 'Answer: A. Answer: B.']:
            self.assertIsNone(MODULE.parse_answer(response, 'ABCD'))

    def test_missing_response_stays_in_denominator(self):
        rows = [{'question_id': str(i), 'options': ['one', 'two', 'three'],
                 'answer': 'B', 'response': response} for i, response in enumerate(['B', 'C', None])]
        result = MODULE.score_records(rows)
        self.assertEqual(result['questions'], 3)
        self.assertEqual(result['unparsed'], 1)
        self.assertAlmostEqual(result['accuracy_percent'], 100/3)

    def test_reject_duplicate_ids(self):
        row = {'question_id': 'q1', 'options': ['one', 'two'], 'answer': 'A', 'response': 'A'}
        with self.assertRaises(ValueError):
            MODULE.score_records([row, row])


if __name__ == '__main__':
    unittest.main()
