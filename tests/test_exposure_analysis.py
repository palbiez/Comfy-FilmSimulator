import json
import unittest

import numpy as np
import torch

from exposure_analysis import ImageExposureAnalysis


class ImageExposureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.node = ImageExposureAnalysis()

    def test_image_is_returned_unchanged_and_metrics_are_deterministic(self):
        image = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                              [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]]], dtype=torch.float32)
        result = self.node.analyze(image, 0.05, 0.99)
        self.assertIs(result[0], image)
        self.assertEqual(result[3], 25.0)
        self.assertEqual(result[4], 50.0)
        self.assertAlmostEqual(result[5], (0.0 + 1.0 + 0.5 + 0.2126) / 4.0, places=6)
        self.assertEqual(tuple(result[1].shape), (1, 128, 256, 3))
        self.assertEqual(tuple(result[2].shape), tuple(image.shape))
        payload = json.loads(result[9])
        self.assertEqual(payload["records"][0]["highlight_clipping_percent_full_rgb"], 25.0)

    def test_mask_limits_statistics_and_batches_are_supported(self):
        image = torch.tensor([
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
            [[[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]],
        ], dtype=torch.float32)
        mask = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=torch.float32)
        result = self.node.analyze(image, 0.1, 0.9, mask)
        self.assertEqual(result[3], [0.0, 0.0])
        self.assertEqual(result[4], [100.0, 0.0])
        self.assertEqual(result[5], [1.0, 0.25])
        self.assertEqual(tuple(result[1].shape), (2, 128, 256, 3))
        self.assertEqual(len(json.loads(result[9])["records"]), 2)

    def test_channel_and_full_rgb_clipping_are_distinguished(self):
        image = torch.tensor([[[[1.0, 0.2, 0.2], [1.0, 1.0, 1.0]]]], dtype=torch.float32)
        result = self.node.analyze(image, 0.01, 0.99)
        payload = json.loads(result[9])["records"][0]
        self.assertEqual(payload["highlight_clipping_percent_any_channel"], 100.0)
        self.assertEqual(payload["highlight_clipping_percent_full_rgb"], 50.0)


if __name__ == "__main__":
    unittest.main()
