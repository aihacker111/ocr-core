"""
ocr-core — local GLM-OCR pipeline without any external API calls.

Quick start:
    from ocr_core import OCRPipeline, PipelineConfig, LayoutConfig, OCRConfig

    # Test with dummy models (no weights):
    pipe = OCRPipeline(PipelineConfig(
        layout=LayoutConfig(detector_name="dummy"),
        ocr=OCRConfig(model_name="dummy"),
    ))
    result = pipe.run_image(my_pil_image)
    print(result.formatted)

    # Full pipeline:
    pipe   = OCRPipeline()
    doc    = pipe.run_file("document.pdf")
    print(doc.merged_text)
"""

from .pipeline            import OCRPipeline, PageResult, DocumentResult
from .config.pipeline_config import PipelineConfig
from .config.layout_config   import LayoutConfig
from .config.ocr_config      import OCRConfig

__all__ = [
    "OCRPipeline",
    "PageResult",
    "DocumentResult",
    "PipelineConfig",
    "LayoutConfig",
    "OCRConfig",
]
