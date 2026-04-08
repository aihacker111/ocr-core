# ocr-core

Local GLM-OCR pipeline — no API calls, runs entirely on your machine.

Based on [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR), adapted for
local-model execution with a clean, extensible architecture.

## Pipeline stages

```
PDF / Image  ──►  PageLoader  ──►  LayoutDetector  ──►  WorkerPool (OCR)  ──►  Formatter
                  (PIL images)    (PP-DocLayout-V3)     (GLM-OCR threads)      (MD/JSON/text)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
from ocr_core import OCRPipeline

# Full pipeline (downloads weights on first run)
pipeline = OCRPipeline()
doc      = pipeline.run_file("document.pdf")
print(doc.merged_text)

# Single image
from PIL import Image
img    = Image.open("page.png")
result = pipeline.run_image(img)
print(result.formatted)
```

## Configuration

```python
from ocr_core import OCRPipeline, PipelineConfig, LayoutConfig, OCRConfig

cfg = PipelineConfig(
    layout=LayoutConfig(
        detector_name="pp_doclay",   # or "dummy" for testing
        device="auto",               # "cpu" | "cuda" | "cuda:0"
        score_threshold=0.30,
    ),
    ocr=OCRConfig(
        model_name="glm_ocr",        # or "dummy" for testing
        model_id="zai-org/GLM-OCR",  # HF model ID
        device="auto",
        dtype="auto",                # "float32" | "float16" | "bfloat16"
        max_new_tokens=2048,
    ),
    max_workers=4,
    output_format="markdown",        # "markdown" | "json" | "text"
)

pipeline = OCRPipeline(cfg)
```

## Adding a custom OCR model

1. Subclass `BaseOCRModel` and implement `_load()` + `_recognize()`.
2. Register in `ocr/models/__init__.py`:
   ```python
   MODEL_REGISTRY["my_model"] = MyOCRModel
   ```
3. Use via `OCRConfig(model_name="my_model")`.

## Adding a custom layout detector

1. Subclass `BaseLayoutDetector` and implement `_load()` + `_predict()`.
2. Register in `layout/detectors/__init__.py`:
   ```python
   DETECTOR_REGISTRY["my_detector"] = MyDetector
   ```
3. Use via `LayoutConfig(detector_name="my_detector")`.
