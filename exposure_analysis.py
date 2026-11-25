"""Deterministic exposure analysis node for ComfyUI images.

IMAGE values are treated as normalized RGB values in [0, 1]. ComfyUI IMAGE
does not carry transfer-function metadata, so the node does not infer whether
the source is linear RAW-derived data or gamma-encoded PNG/JPEG data. Luminance
is the Rec.709 proxy ``0.2126 R + 0.7152 G + 0.0722 B`` in the input encoding.
The input IMAGE is returned unchanged.
"""

import json

import numpy as np
import torch


class ImageExposureAnalysis:
    HISTOGRAM_HEIGHT = 128
    HISTOGRAM_WIDTH = 256

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "highlight_threshold": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
        "FLOAT", "FLOAT", "STRING"
    )
    RETURN_NAMES = (
        "image", "histogram_image", "clipping_preview",
        "shadow_clipping_percent", "highlight_clipping_percent",
        "mean_luminance", "p1_luminance", "p50_luminance", "p99_luminance",
        "analysis_json"
    )
    FUNCTION = "analyze"
    CATEGORY = "Image/Analysis"

    @staticmethod
    def _as_numpy_image(image):
        array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[-1] < 3:
            raise ValueError("IMAGE must have shape [batch, height, width, 3]")
        return np.asarray(array[..., :3], dtype=np.float32)

    @staticmethod
    def _as_numpy_mask(mask, batch, height, width):
        if mask is None:
            return np.ones((batch, height, width), dtype=bool)
        array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError("MASK must have shape [batch, height, width]")
        if array.shape[0] == 1 and batch > 1:
            array = np.broadcast_to(array, (batch, height, width))
        if array.shape != (batch, height, width):
            raise ValueError("MASK dimensions must match IMAGE dimensions")
        return np.isfinite(array) & (array > 0.0)

    @classmethod
    def _histogram_image(cls, rgb, luma, selected):
        canvas = np.zeros((cls.HISTOGRAM_HEIGHT, cls.HISTOGRAM_WIDTH, 3), dtype=np.float32)
        for channel in range(3):
            histogram, _ = np.histogram(np.clip(rgb[..., channel][selected], 0.0, 1.0), bins=cls.HISTOGRAM_WIDTH, range=(0.0, 1.0))
            if histogram.max() > 0:
                histogram = histogram.astype(np.float32) / float(histogram.max())
            for x, value in enumerate(histogram):
                top = cls.HISTOGRAM_HEIGHT - 1 - int(round(value * (cls.HISTOGRAM_HEIGHT - 1)))
                canvas[top:, x, channel] = np.maximum(canvas[top:, x, channel], 0.75)
        histogram, _ = np.histogram(np.clip(luma[selected], 0.0, 1.0), bins=cls.HISTOGRAM_WIDTH, range=(0.0, 1.0))
        if histogram.max() > 0:
            histogram = histogram.astype(np.float32) / float(histogram.max())
        for x, value in enumerate(histogram):
            top = cls.HISTOGRAM_HEIGHT - 1 - int(round(value * (cls.HISTOGRAM_HEIGHT - 1)))
            canvas[top:, x, :] = np.maximum(canvas[top:, x, :], 1.0)
        return canvas

    @staticmethod
    def _preview(rgb, luma, selected, shadow_threshold, highlight_threshold):
        preview = np.clip(rgb, 0.0, 1.0).copy()
        shadow = selected & (luma <= shadow_threshold)
        any_highlight = selected & np.any(rgb >= highlight_threshold, axis=-1)
        full_highlight = selected & np.all(rgb >= highlight_threshold, axis=-1)
        preview[shadow] = (0.10, 0.35, 1.0)
        preview[any_highlight] = (1.0, 0.10, 0.10)
        preview[full_highlight] = (1.0, 1.0, 0.10)
        return preview

    @staticmethod
    def _batch_value(values):
        values = [float(value) for value in values]
        return values[0] if len(values) == 1 else values

    def analyze(self, image, shadow_threshold, highlight_threshold, mask=None):
        if not 0.0 <= shadow_threshold <= 1.0 or not 0.0 <= highlight_threshold <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        rgb = self._as_numpy_image(image)
        batch, height, width, _ = rgb.shape
        selected = self._as_numpy_mask(mask, batch, height, width)
        finite_rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        luma = (0.2126 * finite_rgb[..., 0] + 0.7152 * finite_rgb[..., 1] + 0.0722 * finite_rgb[..., 2]).astype(np.float32)
        histograms, previews, records = [], [], []
        shadows, highlights, means, p1s, p50s, p99s = [], [], [], [], [], []
        for index in range(batch):
            chosen = selected[index]
            values = luma[index][chosen]
            if values.size == 0:
                values = np.array([0.0], dtype=np.float32)
            channel_clipping = np.any(finite_rgb[index] >= highlight_threshold, axis=-1)
            full_rgb_clipping = np.all(finite_rgb[index] >= highlight_threshold, axis=-1)
            shadow_mask = luma[index] <= shadow_threshold
            denominator = max(int(np.count_nonzero(chosen)), 1)
            shadow_percent = 100.0 * np.count_nonzero(shadow_mask & chosen) / denominator
            highlight_percent = 100.0 * np.count_nonzero(channel_clipping & chosen) / denominator
            shadows.append(shadow_percent); highlights.append(highlight_percent)
            means.append(float(np.mean(values))); p1s.append(float(np.percentile(values, 1)))
            p50s.append(float(np.percentile(values, 50))); p99s.append(float(np.percentile(values, 99)))
            histograms.append(self._histogram_image(finite_rgb[index], luma[index], chosen))
            previews.append(self._preview(finite_rgb[index], luma[index], chosen, shadow_threshold, highlight_threshold))
            records.append({
                "batch_index": index, "selected_pixel_count": int(np.count_nonzero(chosen)),
                "shadow_clipping_percent": shadow_percent,
                "highlight_clipping_percent_any_channel": highlight_percent,
                "highlight_clipping_percent_full_rgb": 100.0 * np.count_nonzero(full_rgb_clipping & chosen) / denominator,
                "mean_luminance": means[-1], "p1_luminance": p1s[-1],
                "p50_luminance": p50s[-1], "p99_luminance": p99s[-1],
            })
        analysis_json = json.dumps({
            "color_space": "normalized RGB IMAGE values; source transfer function is unknown",
            "luminance": "Rec.709 proxy: 0.2126*R + 0.7152*G + 0.0722*B in input encoding",
            "thresholds": {"shadow": float(shadow_threshold), "highlight": float(highlight_threshold)},
            "clipping_definition": "highlight any-channel is channel clipping; highlight full-rgb requires all channels",
            "records": records,
        }, sort_keys=True, separators=(",", ":"))
        return (
            image, torch.from_numpy(np.stack(histograms, axis=0)), torch.from_numpy(np.stack(previews, axis=0)),
            self._batch_value(shadows), self._batch_value(highlights), self._batch_value(means),
            self._batch_value(p1s), self._batch_value(p50s), self._batch_value(p99s), analysis_json,
        )
