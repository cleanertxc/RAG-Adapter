import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def test_notebook_datasets_count_unparsed_answers_as_incorrect(self):
        notebook = json.loads((Path(__file__).parents[1] /
                               'notebooks/legacy_evaluation.ipynb').read_text())
        conditions = [
            (14, 'Video-MME', 'questions_video_mme',
             ['short', 'medium', 'long'], 'A'),
            (16, 'Perception_Test', 'questions_perception_test', [None], 'B'),
            (18, 'Egoschema', 'questions_egoschema', [None], 'E'),
            (20, 'MLVU', 'questions_mlvu',
             ['1_plotQA', '2_needle', '3_ego', '4_count', '5_order',
              '6_anomaly_reco', '7_topic_reasoning'], 'D'),
        ]
        # Keep model dependencies, API calls and notebook initialization out of the test.
        for source_cell, dataset, variable, groups, expected in conditions:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                questions = {}
                results = root / 'results' / dataset / 'test-model' / 'test-sampling'
                for group_idx, group in enumerate(groups):
                    # Uppercase prose, an empty response, and one valid option.
                    for index, response in enumerate(['A person is standing.', '', expected]):
                        vid = f'video-{group_idx}-{index}'
                        qid = 'q1'
                        (results / vid).mkdir(parents=True)
                        (results / vid / f'{qid}_pred.txt').write_text(response)
                        if source_cell == 14:
                            questions[vid] = {qid: {'answer': expected, 'domain': 'Knowledge',
                                                    'duration': group}}
                        elif source_cell == 20:
                            questions[vid] = {qid: {'answer': expected, 'question_type': group}}
                        else:
                            questions[vid] = {'id': [qid], 'answer_id': [expected], 'pred': []}
                cell = next(c for c in notebook['cells']
                            if c.get('metadata', {}).get('source_cell') == source_cell)
                definition = next(n for n in ast.parse(''.join(cell['source'])).body
                                  if isinstance(n, ast.FunctionDef) and n.name == 'acc')
                namespace = {'os': os, 'DATA_ROOT': str(root), 'model': 'test-model',
                             'rag_type': 'test-sampling', 'parse_answer': MODULE.parse_answer,
                             variable: questions}
                exec(compile(ast.Module(body=[definition], type_ignores=[]),
                             f'legacy_evaluation:source_cell_{source_cell}', 'exec'), namespace)
                output = io.StringIO()
                original_dir = Path.cwd()
                try:
                    os.chdir(root)
                    with contextlib.redirect_stdout(output):
                        namespace['acc']()
                finally:
                    os.chdir(original_dir)
                overall = float(output.getvalue().strip().splitlines()[-1].split(':')[-1])
                self.assertAlmostEqual(overall, 1 / 3)
                if source_cell in (14, 20):
                    predictions = [r['pred'] for v in questions.values() for r in v.values()]
                elif source_cell == 16:
                    predictions = [p for v in questions.values() for _, p in v['pred']]
                else:
                    predictions = [p for v in questions.values() for p in v['pred']]
                self.assertEqual(predictions.count(None), 2 * len(groups))
                self.assertEqual(predictions.count(expected), len(groups))

    def test_perception_predictions_match_question_ids_regardless_of_file_order(self):
        notebook = json.loads((Path(__file__).parents[1] /
                               'notebooks/legacy_evaluation.ipynb').read_text())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('source_cell') == 16)
        definition = next(n for n in ast.parse(''.join(cell['source'])).body
                          if isinstance(n, ast.FunctionDef) and n.name == 'acc')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Question IDs may repeat in different videos and be stored as integers.
            questions = {'v1': {'id': [1, 2], 'answer_id': ['A', 'B'], 'pred': []},
                         'v2': {'id': [1, 2], 'answer_id': ['C', 'A'], 'pred': []}}
            for vid, records in questions.items():
                folder = root / 'results' / 'Perception_Test' / 'model' / 'sampling' / vid
                folder.mkdir(parents=True)
                for qid, answer in zip(records['id'], records['answer_id']):
                    (folder / f'{qid}_pred.txt').write_text(answer)
            namespace = {'os': os, 'DATA_ROOT': str(root), 'model': 'model',
                         'rag_type': 'sampling', 'parse_answer': MODULE.parse_answer,
                         'questions_perception_test': questions}
            exec(compile(ast.Module(body=[definition], type_ignores=[]),
                         'perception_acc', 'exec'), namespace)
            old_dir = Path.cwd()
            try:
                os.chdir(root)
                for order in [['1_pred.txt', '2_pred.txt'], ['2_pred.txt', '1_pred.txt']]:
                    for records in questions.values():
                        records['pred'] = []
                    output = io.StringIO()
                    with patch.object(os, 'listdir', return_value=order), contextlib.redirect_stdout(output):
                        namespace['acc']()
                    self.assertEqual(float(output.getvalue().strip().split(':')[-1]), 1.0)
                with patch.object(os, 'listdir', return_value=['unknown_pred.txt']):
                    with self.assertRaisesRegex(ValueError, 'Unknown question ID'):
                        namespace['acc']()
            finally:
                os.chdir(old_dir)


if __name__ == '__main__':
    unittest.main()
