from .dng_reader import DNGImageReader
from .film_sim import FilmSimNode
from .save_avif_hdr import SaveAVIF_HDR
from .hdr_merge import HDRMergeNode
from .save_sdr import SaveSDR
from .film_pipeline import FilmPipelineNode
from .exposure_analysis import ImageExposureAnalysis

NODE_CLASS_MAPPINGS = {
    "DNG Image Reader": DNGImageReader,
    "FilmSimNode": FilmSimNode,
    "SaveAVIF_HDR": SaveAVIF_HDR,
    "HDR Merge (Gainmap)": HDRMergeNode,
    "SaveSDR": SaveSDR,
    "FilmPipelineNode": FilmPipelineNode,
    "ImageExposureAnalysis": ImageExposureAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DNG Image Reader": "DNG Image Reader (Rawpy)",
    "FilmSimNode": "Film Simulation V4.1 (HDR Capable)",
    "SaveAVIF_HDR": "Save AVIF/HEIC HDR (Native)",
    "HDR Merge (Gainmap)": "HDR Merge (Gainmap)",
    "SaveSDR": "Save SDR Preview (PNG/JPEG)",
    "FilmPipelineNode": "Film Pipeline (RAW -> Film -> Save)",
    "ImageExposureAnalysis": "Image Exposure Analysis",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
