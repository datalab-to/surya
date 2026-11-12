"""Surya: VLM-backed document OCR, layout analysis, and HTML table recognition."""

from __future__ import annotations

__version__ = "0.24.0"

__all__ = [
    "__version__",
    "SuryaInferenceManager",
    "get_default_manager",
    "LayoutPredictor",
    "RecognitionPredictor",
    "TableRecPredictor",
]


def __getattr__(name: str):
    if name in ("SuryaInferenceManager", "get_default_manager"):
        from surya.inference import SuryaInferenceManager, get_default_manager

        return (
            SuryaInferenceManager
            if name == "SuryaInferenceManager"
            else get_default_manager
        )
    if name == "LayoutPredictor":
        from surya.layout import LayoutPredictor

        return LayoutPredictor
    if name == "RecognitionPredictor":
        from surya.recognition import RecognitionPredictor

        return RecognitionPredictor
    if name == "TableRecPredictor":
        from surya.table_rec import TableRecPredictor

        return TableRecPredictor
    raise AttributeError(f"module 'surya' has no attribute {name!r}")
