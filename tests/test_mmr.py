"""Isolated tests of the notebook MMR functions using CPU embeddings."""
import ast
import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'MMR tests require PyTorch.')
class MMRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / 'notebooks/rag_adapter_experiments.ipynb'
        notebook = json.loads(path.read_text())
        cell = next(c for c in notebook['cells']
                    if c.get('metadata', {}).get('source_cell') == 20)
        nodes = [n for n in ast.parse(''.join(cell['source'])).body
                 if isinstance(n, ast.FunctionDef) and n.name in ('mmr', 'mmr_selection')]
        scope = {'torch': torch, 'device': 'cpu'}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        cls.mmr = staticmethod(scope['mmr'])
        cls.select = staticmethod(scope['mmr_selection'])

    @staticmethod
    def vector(x, y):
        return torch.tensor([x, y, math.sqrt(max(0, 1 - x*x - y*y))], dtype=torch.float64)

    def test_penalizes_similarity_to_any_selected_frame_without_duplicates(self):
        embeddings = {'s1': self.vector(1, 0), 's2': self.vector(0, 1),
                      'c': self.vector(.9, .2), 'd': self.vector(.6, .6)}
        choice = self.mmr({'c': .8, 'd': .8}, ['s1', 's2'], ['c', 'd'],
                          .7, embeddings, embeddings)
        # C has less similarity to s2 but substantially more redundancy with s1.
        self.assertEqual(choice, 'd')

    def test_takes_maximum_after_summing_modalities_for_each_selected_frame(self):
        frames = {'s1': self.vector(1, 0), 's2': self.vector(0, 1),
                  'c': self.vector(.9, .1), 'd': self.vector(.6, .2)}
        captions = dict(frames, c=self.vector(.1, .9))
        choice = self.mmr({'c': .8, 'd': .8}, ['s1', 's2'], ['c', 'd'],
                          .7, frames, captions)
        # C's pairwise sums are 1.0 and 1.0. D's are 1.2 and 0.4.
        # Taking separate maxima in the two modalities would choose D incorrectly.
        self.assertEqual(choice, 'c')

    def test_negative_similarities_are_not_clamped_to_zero(self):
        frames = {'s1': self.vector(1, 0), 's2': self.vector(0, 1),
                  'c': self.vector(-.6, -.7), 'd': self.vector(-.1, -.2)}
        choice = self.mmr({'c': .8, 'd': .8}, ['s1', 's2'], ['d', 'c'], .7, frames)
        self.assertEqual(choice, 'c')

    def test_first_frame_uses_highest_relevance_and_theta_one_ignores_redundancy(self):
        frames = {'s': self.vector(1, 0), 'low': self.vector(0, 1),
                  'high': self.vector(.9, .1)}
        scores = {'low': .2, 'high': .9}
        self.assertEqual(self.mmr(scores, [], ['low', 'high'], .7, frames), 'high')
        self.assertEqual(self.mmr(scores, ['s'], ['low', 'high'], 1., frames), 'high')

    def test_empty_and_zero_budget_do_not_load_embeddings(self):
        self.assertEqual(self.select({}, None, None), [])
        self.assertEqual(self.select({'unused.jpg': .8}, None, None, top_k=0), [])
        with self.assertRaises(ValueError):
            self.select({}, None, None, top_k=-1)
        with self.assertRaises(ValueError):
            self.select({}, None, None, lambda_param=1.1)

    def test_full_selection_returns_unique_frames_and_handles_small_candidate_pools(self):
        class Encoder:
            def __init__(self, embeddings):
                self.embeddings = embeddings

            def get_text_embedding(self, text):
                return self.embeddings[text]

            def _get_image_embedding(self, path):
                return self.embeddings[Path(path).stem]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'frames').mkdir()
            (root / 'captions').mkdir()
            vectors = {'a': [1., 0., 0.], 'b': [.9, .2, math.sqrt(.15)],
                       'c': [0., 1., 0.]}
            encoder = Encoder(vectors)
            paths = {key: str(root / 'frames' / f'{key}.jpg') for key in vectors}
            for key in vectors:
                (root / 'captions' / f'{key}.txt').write_text(key)
            scores = {paths['a']: .9, paths['b']: .8, paths['c']: .8}
            self.assertEqual(self.select(scores, encoder, encoder, top_k=2),
                             [paths['a'], paths['c']])
            result = self.select(scores, encoder, encoder, top_k=10)
            self.assertEqual(result, [paths['a'], paths['c'], paths['b']])
            self.assertEqual(len(result), len(set(result)))


if __name__ == '__main__':
    unittest.main()
