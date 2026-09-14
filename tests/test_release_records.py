import csv
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


class RecordTests(unittest.TestCase):
    def test_each_benchmark_has_90_matching_video_ids(self):
        for name in ['video_mme', 'mlvu', 'egoschema', 'perception_test']:
            with self.subTest(dataset=name):
                folder = ROOT / 'data/sampled_records'
                ids = (folder/f'{name}_video_ids.txt').read_text(encoding='utf-8-sig').splitlines()
                with (folder/f'{name}_nif.csv').open(encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(ids), 90)
                self.assertEqual(len(set(ids)), 90)
                self.assertEqual({row['video_id'] for row in rows}, set(ids))

    def test_notebooks_have_clean_outputs_and_valid_python(self):
        for path in (ROOT/'notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text())
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    with self.subTest(notebook=path.name, cell=i):
                        self.assertFalse(cell['outputs'])
                        self.assertIsNone(cell['execution_count'])
                        compile(''.join(cell['source']), str(path), 'exec')


if __name__ == '__main__':
    unittest.main()
