"""Replay must cover the catalog before repeating and reproduce on resume."""
import unittest
from audiollm.data import replay_epoch_indices, normalize_training_text, SpeechDataset


class ReplayTests(unittest.TestCase):
    def test_rotation_and_resume(self):
        items = [{} for _ in range(40)] + [dict(replay=True) for _ in range(23)]
        stream = []
        for epoch in range(4):
            indices = replay_epoch_indices(items, epoch, .2, 4000)
            self.assertEqual(indices, replay_epoch_indices(items, epoch, .2, 4000))
            self.assertEqual(indices[:40], list(range(40)))
            self.assertEqual(len(indices), 50)
            stream.extend(indices[40:])
        self.assertEqual(len(set(stream[:23])), 23)
        self.assertEqual(len(set(stream[23:])), 17)

    def test_english_not_vietnamese_correction(self):
        self.assertEqual(normalize_training_text('I have 42 apples.', 'en'), 'i have 42 apples')
        dataset = SpeechDataset([], None, None, noiser=object())
        self.assertIsNone(dataset.correction_hypothesis(dict(replay=True, text='hello')))


if __name__ == '__main__':
    unittest.main()
