import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import dng_reader


class _FakeRaw:
    white_level = 65535
    camera_whitebalance = [2.0, 1.0, 1.0, 1.0]
    daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
    black_level = [0, 0, 0, 0]
    color_desc = b"RGBG"
    model = b"Test Camera"
    camera_make = b"Test Maker"
    sizes = SimpleNamespace(width=2, height=1, filters=9)

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def postprocess(self, **params):
        self.calls.append(params)
        return np.array([[[1000, 2000, 3000], [4000, 5000, 6000]]], dtype=np.uint16)


class DNGImageReaderTests(unittest.TestCase):
    def test_new_controls_are_passed_to_rawpy_in_linear_light(self):
        raw = _FakeRaw()
        reader = dng_reader.DNGImageReader()
        with tempfile.NamedTemporaryFile(suffix=".dng", delete=False) as handle:
            path = handle.name
        try:
            with patch.object(dng_reader.rawpy, "imread", return_value=raw):
                image, metadata = reader.read_dng(
                    path, True, "As Shot", 0.5, False, 0.02, 1.0, "blend", 5
                )
            params = raw.calls[0]
            self.assertTrue(params["no_auto_bright"])
            self.assertEqual(params["auto_bright_thr"], 0.02)
            self.assertEqual(params["exp_shift"], 2.0)
            self.assertEqual(params["highlight_mode"], getattr(dng_reader.rawpy.HighlightMode, "Blend", 2))
            self.assertEqual(tuple(image.shape), (1, 1, 2, 3))
            self.assertIn("Exposure EV: +1.00", metadata)
        finally:
            os.unlink(path)

    def test_default_arguments_preserve_sdr_auto_brightness(self):
        raw = _FakeRaw()
        reader = dng_reader.DNGImageReader()
        with tempfile.NamedTemporaryFile(suffix=".dng", delete=False) as handle:
            path = handle.name
        try:
            with patch.object(dng_reader.rawpy, "imread", return_value=raw):
                reader.read_dng(path, False, "As Shot", 0.5)
            self.assertFalse(raw.calls[0]["no_auto_bright"])
            self.assertEqual(raw.calls[0]["exp_shift"], 1.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
