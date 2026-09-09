"""
scanned_pdf_parser.py
----------------------
OCR path for scanned/image-only PDF pages and standalone image files
(.jpg/.png/.tiff/.bmp), using pytesseract + Pillow. This is the module
pdf_parser.py's parse_pdf() routes to when classify_page() decides a page
has no usable text layer, and the module app/dispatcher.py routes to for
standalone image inputs.

WHY THIS MODULE RETURNS THE SAME PDFPageBlock SHAPE
------------------------------------------------------
pdf_parser.py's PDFPageBlock (text, word_spans, block_ranges) exists so
that detect_pii_in_pdf() and pdf_redactor.py can treat every page
identically regardless of how its text was obtained - this is the payoff
of keeping the detection core (merger.detect_all) format-agnostic. This
module produces the exact same shape from OCR output instead of PyMuPDF's
native text layer, so nothing downstream needs to know or care that a
page was scanned.

ORIENTATION CORRECTION (90-degree only) - IMPLEMENTED
--------------------------------------------------------
A scanned page can come in sideways or upside-down (fed into a scanner
rotated, or a phone photo taken in the wrong orientation) - a clean
90/180/270-degree error, not an arbitrary tilt. Tesseract's own OSD
(orientation and script detection, image_to_osd()) detects exactly this:
it reports how many degrees clockwise the image needs to be rotated to
be upright, and only ever reports one of 0/90/180/270 - it does not
detect or report arbitrary skew angles.

_correct_orientation() runs this check and rotates the image BEFORE
Tesseract's real word-level OCR pass (image_to_data()) runs. Because the
rotation happens pre-OCR, every bounding box Tesseract returns afterward
is already correct for the now-upright image - there is no separate
"transform word boxes back to the original orientation" step needed, and
none is done here. This is the key simplification that makes 90-degree
correction safe to ship without the coordinate-mapping risk full
arbitrary-angle deskew would carry.

If OSD can't determine an orientation (common on sparse, mostly-blank,
or very low-text images - Tesseract needs a minimum amount of detected
text to make this call), the image is used as-is rather than failing the
whole page.

WHAT'S STILL DELIBERATELY OUT OF SCOPE (see conversation / design notes)
--------------------------------------------------------------------------
- Arbitrary-angle deskew (e.g. a page tilted 6 degrees, not a clean 90-
  degree multiple): NOT implemented. That requires detecting a continuous
  skew angle AND an inverse-rotation transform to map OCR bounding boxes
  back onto the original, untilted page for redaction - getting that
  transform wrong silently misplaces a redaction box on a real scan, a
  correctness bug, not just a cosmetic one. Left as an explicit open
  decision, to be revisited if real Nigerian test samples show meaningful
  non-90-degree tilt.
- Page rotation matrices (page.rotation): coordinate mapping assumes a
  standard unrotated page, same category of accepted simplification as
  the NUBAN/NHIS collision elsewhere in this project.

COORDINATE SPACES
------------------
- extract_scanned_page_block() (pages embedded in a PDF): OCR runs on a
  pixmap rendered at `dpi`. Pixel coordinates are scaled back to PDF
  point-space (1 point = 1/72 inch) via factor 72/dpi, so word_spans
  stay directly comparable to the ones _build_page_block() produces for
  text-layer pages - pdf_redactor.py needs no changes.
- extract_image_block() (standalone image files): there is no PDF page to
  map back onto, so rects stay in raw image pixel space. The
  yet-to-be-built scanned_pdf_redactor.py draws directly on the image in
  that same space.
- In both cases, orientation correction (if it fires) rotates the image
  itself prior to word extraction, so word_spans are always relative to
  the final, upright image dimensions - not the original file's raw
  orientation. Downstream code doesn't need to know whether a rotation
  happened.

CONFIDENCE / LOW-CONFIDENCE FLAGGING
--------------------------------------
pytesseract.image_to_data() returns a 0-100 confidence per recognized
word. Each WordSpan carries its word's confidence. After detection,
annotate_entity_confidence() resolves each detected entity's character
span back to the underlying word(s) and attaches:
    entity["ocr_confidence"]  - min confidence among overlapping words
    entity["low_confidence"]  - True if below LOW_OCR_CONFIDENCE_THRESHOLD
matching the earlier decision: low-confidence OCR words get flagged for
manual review rather than silently redacted or silently trusted.
"""

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from app.parsers.pdf_parser import PDFPageBlock, WordSpan
from app.detectors.merger import detect_all

# Tesseract word-confidence threshold (0-100 scale). Below this, a word is
# flagged low_confidence=True on any entity it overlaps, surfaced in the
# GUI for manual review rather than trusted outright. Adjustable.
LOW_OCR_CONFIDENCE_THRESHOLD = 60

# pytesseract's Tesseract-level "no confidence" sentinel for structural
# (non-word) rows in image_to_data output.
_NO_CONFIDENCE = -1


# ---------------------------------------------------------------------------
# Orientation correction (90-degree only - see module docstring)
# ---------------------------------------------------------------------------

def _correct_orientation(image: Image.Image) -> Image.Image:
    """
    Detects and corrects a clean 90/180/270-degree rotation using
    Tesseract's OSD (orientation and script detection), run BEFORE the
    real word-level OCR pass. OSD only ever reports one of these four
    angles - never an arbitrary tilt - so no bbox transform is needed
    afterward; Tesseract simply re-reads the now-upright image and its
    word boxes are correct for it from the start.

    Falls back to returning `image` unchanged if OSD can't confidently
    determine an orientation (e.g. too little text on the page/image) -
    this is a normal, expected outcome on sparse scans, not an error
    worth surfacing to the caller.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
    except pytesseract.TesseractError:
        # OSD couldn't run (typically: too little text to analyze).
        # Proceed with the image as-is rather than failing the page.
        return image

    rotate_degrees = osd.get("rotate", 0)
    if not rotate_degrees:
        return image

    # osd["rotate"] is how many degrees CLOCKWISE the image must be
    # turned to be upright. PIL's Image.rotate() turns COUNTER-clockwise
    # for positive angles, so the equivalent PIL call is the negated
    # angle. expand=True so the canvas is resized to fit the rotated
    # image instead of cropping it (matters for the 90/270 cases, where
    # width and height swap).
    return image.rotate(-rotate_degrees, expand=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Grayscale + autocontrast only (see module docstring for why
    arbitrary-angle deskew is intentionally excluded here; 90-degree
    orientation correction is handled separately in
    _correct_orientation(), before this function runs). Tesseract 4/5
    does its own internal binarization (Otsu), so a hard manual threshold
    is deliberately skipped too - it tends to help less than it risks
    losing faint scan detail.
    """
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray)


# ---------------------------------------------------------------------------
# Core OCR -> PDFPageBlock construction (shared by both entry points)
# ---------------------------------------------------------------------------

def _ocr_image_to_block(
    image: Image.Image,
    page_num: int,
    pixel_to_point_scale: Optional[float],
) -> PDFPageBlock:
    """
    Runs Tesseract on `image` and reconstructs a PDFPageBlock the same way
    _build_page_block() does for text-layer PDFs: word-level text is
    concatenated in reading order, with a per-word (char_start, char_end)
    -> bounding box map, plus block_ranges (one per Tesseract block_num)
    so detection still runs per structural unit rather than on one giant
    page string (same NER-context-contamination fix as pdf_parser.py).

    Orientation is corrected (see _correct_orientation()) before
    preprocessing/OCR, so word_spans are always relative to the final,
    upright image - callers never need to account for rotation
    themselves.

    pixel_to_point_scale: multiply pixel coords by this to get PDF point
    coords (72/dpi). Pass None to keep raw pixel coordinates (standalone
    image files - see module docstring on coordinate spaces).
    """
    oriented = _correct_orientation(image)
    preprocessed = _preprocess_for_ocr(oriented)
    ocr_data = pytesseract.image_to_data(preprocessed, output_type=Output.DICT)

    n = len(ocr_data["text"])
    words = []
    for i in range(n):
        text = ocr_data["text"][i]
        conf = float(ocr_data["conf"][i])
        # level 5 = word-level row in Tesseract's hierarchy (1=page,
        # 2=block, 3=paragraph, 4=line, 5=word). Higher levels carry no
        # text and conf == -1; skip them, and skip blank/whitespace words.
        if int(ocr_data["level"][i]) != 5 or not text.strip() or conf < 0:
            continue
        words.append({
            "text": text,
            "conf": conf,
            "left": ocr_data["left"][i],
            "top": ocr_data["top"][i],
            "width": ocr_data["width"][i],
            "height": ocr_data["height"][i],
            "block_num": ocr_data["block_num"][i],
            "line_num": ocr_data["line_num"][i],
            "word_num": ocr_data["word_num"][i],
        })

    # Reading order, mirroring pdf_parser._build_page_block's
    # (block_no, line_no, word_no) sort.
    words.sort(key=lambda w: (w["block_num"], w["line_num"], w["word_num"]))

    scale = pixel_to_point_scale if pixel_to_point_scale is not None else 1.0

    text_parts = []
    word_spans = []
    block_ranges = []
    cursor = 0
    prev_block_line = None
    prev_block_num = None
    block_start = 0

    for w in words:
        current_block_line = (w["block_num"], w["line_num"])

        if prev_block_num is not None and w["block_num"] != prev_block_num:
            block_ranges.append((block_start, cursor))
            block_start = cursor

        if prev_block_line is not None:
            separator = "\n" if current_block_line != prev_block_line else " "
            text_parts.append(separator)
            cursor += len(separator)

        start = cursor
        text_parts.append(w["text"])
        cursor += len(w["text"])
        end = cursor

        rect = fitz.Rect(
            w["left"] * scale,
            w["top"] * scale,
            (w["left"] + w["width"]) * scale,
            (w["top"] + w["height"]) * scale,
        )
        word_spans.append(WordSpan(start=start, end=end, rect=rect, confidence=w["conf"]))

        prev_block_line = current_block_line
        prev_block_num = w["block_num"]

    if words:
        block_ranges.append((block_start, cursor))

    return PDFPageBlock(
        page_num=page_num,
        text="".join(text_parts),
        word_spans=word_spans,
        block_ranges=block_ranges,
        page_type="scanned",
    )


# ---------------------------------------------------------------------------
# Entry point 1: a scanned page embedded in an otherwise-normal PDF
# ---------------------------------------------------------------------------

def extract_scanned_page_block(page: fitz.Page, page_num: int, dpi: int = 300) -> PDFPageBlock:
    """
    Rasterizes one PDF page at `dpi` and OCRs it, returning a PDFPageBlock
    in PDF point-space so it's directly interchangeable with
    _build_page_block()'s output for text-layer pages. Called from
    pdf_parser.parse_pdf() when classify_page() finds a scanned page.

    Detection is NOT run here - detect_pii_in_pdf() runs detect_all()
    against block.block_ranges uniformly for every page regardless of
    page_type, exactly as it already does for text pages.
    """
    pix = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    pixel_to_point_scale = 72.0 / dpi
    return _ocr_image_to_block(image, page_num, pixel_to_point_scale)


# ---------------------------------------------------------------------------
# Entry point 2: a standalone image file (not part of any PDF)
# ---------------------------------------------------------------------------

def extract_image_block(image_path: str, dpi: int = 300) -> Tuple[PDFPageBlock, list]:
    """
    Opens a standalone image file, OCRs it, and runs detection - unlike
    extract_scanned_page_block(), this does both parsing AND detection in
    one call, because there is no pdf_parser.detect_pii_in_pdf()-style
    wrapper for non-PDF inputs; app/dispatcher.py calls this directly.

    `dpi` is accepted for interface symmetry with the PDF path but has no
    rasterization effect here - a standalone image is already a fixed
    raster at whatever resolution it was captured/scanned at. If OCR
    quality on very low-resolution images turns out to be a problem,
    upscaling before OCR would be the fix - not implemented here.

    Returns (block, entities) - entities shaped identically to
    detect_pii_in_pdf()'s output ({"text","entity_type","start","end",
    "score","source"}, plus "ocr_confidence"/"low_confidence"), with
    start/end as offsets into block.text.
    """
    image = Image.open(image_path).convert("RGB")

    # No PDF page to map back onto - keep raw pixel coordinates.
    block = _ocr_image_to_block(image, page_num=0, pixel_to_point_scale=None)

    entities = []
    for seg_start, seg_end in block.block_ranges:
        segment_text = block.text[seg_start:seg_end]
        if not segment_text.strip():
            continue

        segment_entities = detect_all(segment_text)
        for e in segment_entities:
            shifted = dict(e)
            shifted["start"] = e["start"] + seg_start
            shifted["end"] = e["end"] + seg_start
            entities.append(shifted)

    entities.sort(key=lambda e: e["start"])
    annotate_entity_confidence(block, entities)

    return block, entities


# ---------------------------------------------------------------------------
# Confidence annotation (used by extract_image_block above, and called
# from pdf_parser.detect_pii_in_pdf() for embedded scanned pages)
# ---------------------------------------------------------------------------

def annotate_entity_confidence(block: PDFPageBlock, entities: list) -> None:
    """
    Mutates each entity dict in place, adding:
        ocr_confidence - the lowest confidence among OCR words whose
                          character span overlaps the entity's (start, end)
        low_confidence - True if ocr_confidence < LOW_OCR_CONFIDENCE_THRESHOLD

    No-op (entities untouched) if block.word_spans carry no confidence
    data, i.e. block.page_type != "scanned" - safe to call unconditionally.
    """
    if block.page_type != "scanned" or not block.word_spans:
        return

    for e in entities:
        overlapping = [
            ws.confidence
            for ws in block.word_spans
            if ws.confidence is not None and ws.start < e["end"] and ws.end > e["start"]
        ]
        if overlapping:
            e["ocr_confidence"] = min(overlapping)
            e["low_confidence"] = e["ocr_confidence"] < LOW_OCR_CONFIDENCE_THRESHOLD
        else:
            # No overlapping word found (shouldn't normally happen) -
            # fail toward caution rather than silently omitting the flag.
            e["ocr_confidence"] = None
            e["low_confidence"] = True


if __name__ == "__main__":
    # Run as: python -m app.parsers.scanned_pdf_parser tests\NigerianSamples\scanned_sample.png
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/scanned_sample.png"

    print(f"OCR-parsing: {test_path}\n")
    block, entities = extract_image_block(test_path)

    print(f"Text length: {len(block.text)} chars | {len(block.word_spans)} words | "
          f"{len(block.block_ranges)} structural blocks")
    print(f"Entities found: {len(entities)}")
    for e in entities:
        matched = block.text[e["start"]:e["end"]]
        flag = " ⚠ LOW CONFIDENCE" if e.get("low_confidence") else ""
        print(f"  [{e['entity_type']}] '{matched}' (score={e.get('score', '?')}, "
              f"source={e.get('source', '?')}, ocr_confidence={e.get('ocr_confidence')}){flag}")