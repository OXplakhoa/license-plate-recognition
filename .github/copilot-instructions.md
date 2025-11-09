# Copilot / Agent Instructions for Vietnamese License Plate Recognition

This short guide helps AI coding agents be productive in this repository. Focus on concrete, discoverable patterns and commands.

## Big picture (what this repo does)
- Purpose: Traditional (non-Deep-Learning) Vietnamese license-plate recognition using OpenCV + Tesseract. See `README.md` for the pipeline diagram (Preprocess → Detect → Segment → OCR).
- Entry points:
  - `main.py` — CLI entry (example: `python main.py --image data/test_images/sample.jpg`). It prints TODOs and is primarily a developer-run demo.
  - `notebooks/lp_recognition_exploration.ipynb` — exploratory development and visual debugging.
- Expected modules (some are referenced in docs but may be missing): `src/lp_detector.py`, `src/character_segmenter.py`, `src/ocr_engine.py`, `src/utils.py`. Treat `src/__init__.py` as package metadata.

## Developer workflows & commands (use PowerShell on Windows)
- Install deps (recommended env Python 3.11/3.12):
  ```powershell
  cd f:\code\XLA
  pip install -r requirements.txt
  ```
- Tesseract: must be installed on the OS (Windows recommended installer). Configure environment via `.env` using `.env.example` values:
  - `TESSERACT_PATH` points to `tesseract.exe` (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- Quick checks:
  ```powershell
  python -c "import cv2, numpy, pytesseract; print(cv2.__version__)"
  ```
- Run main demo:
  ```powershell
  python main.py --image data/test_images/sample.jpg
  ```
- Tests (if/when present):
  ```powershell
  pytest tests/
  ```

## Project-specific conventions & patterns
- Pipeline-first development: heavy use of the Jupyter notebook for exploration. Changes often start in `notebooks/` and then are converted into `src/` modules.
- No deep-learning models: all detection and segmentation use OpenCV transforms (Canny, morphology, contours), Otsu thresholding, CLAHE, etc. Search for these keywords when implementing detection/segmentation.
- Tesseract is used for character recognition — expect a wrapper module like `ocr_engine.py` that sets `pytesseract.pytesseract.tesseract_cmd` from `.env`.
- File paths: examples and docs use Windows absolute-like paths (drive letter). When writing cross-platform code, use `pathlib.Path` consistently (see `main.py`).

## Where to look for examples and important files
- `README.md` — pipeline, commands, and expected module names. Use it as canonical quick-reference.
- `SETUP.md` — environment, Python version recommendations, and troubleshooting steps (compilers, tesseract path, pip flags).
- `.env.example` — required Tesseract env variables and expected locations on Windows.
- `notebooks/lp_recognition_exploration.ipynb` — image-level debugging and parameter tuning; copy working cells into `src/` when stabilised.

## Typical edits and tests agents should perform
- When adding detection/segmentation code: write small functions with clear inputs (BGR image or grayscale numpy array) and outputs (ROI box coords, binary mask, list of character image arrays).
- Add a unit test under `tests/` for each module: test a small image in `data/test_images/` and assert expected ROI or number of segmented characters (ground truth in `data/ground_truth.txt`).
- If you change Tesseract usage, add a short integration test that sets `pytesseract.pytesseract.tesseract_cmd` to the `.env` value (mock when Tesseract is not installed).

## Examples (concrete snippets you may need)
- Set tesseract cmd (pattern expected in `ocr_engine.py`):
  ```python
  from os import getenv
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = getenv('TESSERACT_PATH')
  ```
- CLI usage (from `main.py`):
  ```python
  # process args
  python main.py --image data/test_images/sample.jpg --output out.png
  ```

## Integration points & external dependencies
- OpenCV (`cv2`) for all image processing.
- pytesseract / Tesseract OCR (external system dependency; instruct users to install OS-level Tesseract).
- NumPy and Pillow for image handling.
- Jupyter for interactive exploration.

## What not to change without verifying
- Do not assume existence of `src/lp_detector.py` etc. — verify and add new modules rather than renaming undocumented files.
- Avoid introducing deep-learning libraries; project explicitly forbids DL solutions.

## Reporting & iteration
- After making module changes, run the notebook to re-check visual outputs and run `pytest tests/` (or add tests if missing).

---
If any important file or workflow is missing from this doc (for example, a CI test or a favored local debugging command), tell me which area to expand and I will update this file accordingly.