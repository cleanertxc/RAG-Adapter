"""Check the notebook's CLIP labels independently of model training."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image


class ClipGroupingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / 'notebooks/rag_adapter_experiments.ipynb'
        notebook = json.loads(path.read_text())
        cell = next(c for c in notebook['cells']
                    if c.get('metadata', {}).get('source_cell') == 58)
        definitions = [n for n in ast.parse(''.join(cell['source'])).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        self.scope = {'Path': Path, 'Image': Image, 'Dataset': object,
                      'clip': SimpleNamespace(tokenize=lambda queries: list(queries)),
                      'train_hash_map': {}, 'test_hash_map': {},
                      'train_label': 0, 'test_label': 0}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), self.scope)

    def test_frames_from_same_video_share_a_group_in_both_partitions(self):
        for name in ['generate_train_labels', 'generate_test_labels']:
            with self.subTest(partition=name):
                label = self.scope[name]
                first = label('/data/MSVD-QA/frames/video1/000001.jpg')
                second = label('/data/MSVD-QA/frames/video1/000020.jpg')
                other_video = label('/data/MSVD-QA/frames/video2/000001.jpg')
                other_source = label('/data/MSRVTT-QA/frames/video1/000001.jpg')
                self.assertEqual(first, second)
                self.assertNotEqual(first, other_video)
                self.assertNotEqual(first, other_source)

    def test_dataset_looks_up_video_group_for_each_sampled_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / 'MSVD-QA' / 'frames' / video / frame
                     for video, frame in [('video1', '000001.png'),
                                          ('video1', '000020.png'),
                                          ('video2', '000001.png')]]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (2, 2)).save(path)

            def processor(image):
                with image:
                    return image.size

            for split in ['train', 'test']:
                with self.subTest(partition=split):
                    generate = self.scope[f'generate_{split}_labels']
                    expected = [generate(str(path)) for path in paths]
                    dataset = self.scope['ClipFintuneDataset'](
                        ['question1', 'question2', 'question3'], [str(p) for p in paths],
                        processor, self.scope[f'{split}_hash_map'])
                    actual = [dataset[i][2] for i in range(len(dataset))]
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual[0], actual[1])
                    self.assertNotEqual(actual[0], actual[2])


if __name__ == '__main__':
    unittest.main()
