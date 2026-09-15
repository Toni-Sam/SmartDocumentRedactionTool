"""
scanned_pdf_redactor.py
------------------------
Redaction for standalone image files (.jpg/.png/.tiff/.bmp) - NOT for
scanned pages embedded in a PDF. See note below on why this module is
much smaller than originally scoped.

WHY THIS MODULE DOESN'T ALSO HANDLE EMBEDDED SCANNED PDF PAGES
-------------------------------------------------------------------
The original plan assumed a scanned PDF page needs its own redaction path
because there's "no content stream to redact - the page is just an
image". That's true, but it turns out not to matter: PyMuPDF's
apply_redactions() defaults to images=2 ("blank out overlapping image
parts" - pixel-level redaction within the given rect, not whole-image
removal or a no-op). Verified directly: built a page containing only an
image, redacted a sub-rectangle of it via the existing
add_redact_annot()/apply_redactions() calls in pdf_redactor.py, and
confirmed via pixel inspection and a re-OCR pass that only the targeted
rectangle was blanked - everything outside it was untouched, and the
redacted text was unrecoverable afterward.

Since scanned_pdf_parser.py's word_spans already carry rects in the same
PDF point-space as text-layer pages (72/dpi scaling - see that module),
pdf_redactor.py's redact_pdf_from_detections() needs ZERO changes to
correctly redact embedded scanned pages. app/dispatcher.py's PDF branch
of apply_redactions() already calls it uniformly for every page
regardless of page_type, which is exactly correct as-is.

So this module's only job is the standalone-image case, where there is
no fitz.Page / content stream at all - just a raster image on disk that
needs pixels blanked directly and re-saved.

WHY PIXELS ARE BLANKED, NOT JUST BOXED OVER
-----------------------------------------------
Same reasoning as pdf_redactor.py: a visual-only box over an image is
fine as far as raster pixels go (there's no separate "text layer"
underneath a JPEG/PNG to leak), so a solid fill IS the equivalent of true
redaction here - once the pixels are overwritten and the file re-saved,
the original pixel data is gone. No copy-paste leak vector exists for a
flattened raster image the way it does for a PDF content stream.

WHY THE OUTPUT IS RE-ORIENTED BEFORE SAVING
-----------------------------------------------
word_spans are in the *raw* input image's pixel space (as documented on
redact_image_from_detections below), so drawing rects straight from
word_spans onto Image.open(input_path) already lands boxes on the
correct words even when the source photo/scan is rotated 90 deg from
upright - that part was already working. What wasn't happening: nothing
ever rotated the *final saved file* to upright, so a correctly-redacted
image would still come out sideways for the user.

_detect_rotation_angle() re-runs Tesseract OSD on the image *after*
redaction (so it can't shift box placement) purely to get the clockwise
correction needed, and the image is rotated to upright immediately
before the final save. This is deliberately independent of
scanned_pdf_parser.py's own OSD-based orientation correction (used
upstream to get accurate word_spans in the first place) - duplicating a
cheap OSD call here avoids coupling this module to that one's internal
signature/return shape.
"""

from typing import Dict, List

import pytesseract
from PIL import Image, ImageDraw

from app.parsers.pdf_parser import PDFPageBlock


def _rects_for_span(block: PDFPageBlock, start: int, end: int) -> list:
    """Returns every word rect in `block` whose char range overlaps [start, end)."""
    return [ws.rect for ws in block.word_spans if ws.start < end and ws.end > start]


def _detect_rotation_angle(image: Image.Image) -> int:
    """
    Returns the clockwise rotation (in degrees) Tesseract's orientation
    and script detection (OSD) says is needed to bring `image` upright.

    Returns 0 (no rotation) if OSD can't produce a confident reading -
    e.g. too little text left after redaction, or a scan too noisy for
    OSD - rather than raising, since a failed orientation check on an
    already-redacted image should never block saving the output.
    """
    try:
        osd = pytesseract.image_to_osd(image)
        rotate_line = next(
            line for line in osd.splitlines() if line.startswith("Rotate:")
        )
        return int(rotate_line.split(":")[1].strip())
    except Exception:
        return 0


def redact_image_from_detections(
    input_path: str,
    output_path: str,
    block: PDFPageBlock,
    entities: list,
    fill_color=(0, 0, 0),
) -> Dict[str, int]:
    """
    Draws solid-fill rectangles directly onto the source image's pixels
    for every approved entity in `entities`, rotates the result upright
    if needed, then saves it to output_path. Mirrors
    pdf_redactor.redact_pdf_from_detections()'s signature and per-word-rect
    approach (redacting each overlapping word rect individually rather
    than one merged bbox per entity, so a detected span that happens to
    wrap a line doesn't blank unrelated text in between).

    `block` is the PDFPageBlock returned by
    scanned_pdf_parser.extract_image_block() for this same input_path -
    its word_spans carry rects in raw image pixel space (not PDF
    point-space - there is no PDF page here), which is exactly what
    PIL.ImageDraw needs.

    Box placement happens entirely before any rotation, so a rotated
    source photo/scan still gets its boxes drawn in the correct spots;
    the upright-rotation pass afterward only affects how the final file
    is oriented for the user, never where the redactions land.

    Returns a summary dict: {"pages": 1, "entities_redacted": n} - kept
    in the same shape as pdf_redactor's summary dict (pages=1 always,
    since a standalone image has no page concept) so callers/GUI code
    don't need a separate code path just to print a summary line.
    """
    image = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    entities_redacted = 0
    for entity in entities:
        rects = _rects_for_span(block, entity["start"], entity["end"])
        if not rects:
            continue
        for rect in rects:
            # PIL expects (x0, y0, x1, y1) - fitz.Rect already stores
            # exactly that, so no conversion needed.
            draw.rectangle([rect.x0, rect.y0, rect.x1, rect.y1], fill=fill_color)
        entities_redacted += 1

    angle = _detect_rotation_angle(image)
    if angle:
        # Tesseract's "Rotate" value is the clockwise correction needed
        # to reach upright; PIL's rotate() turns counter-clockwise for
        # positive degrees, so negate it. expand=True so the canvas
        # swaps width/height for 90/270 corrections instead of cropping.
        image = image.rotate(-angle, expand=True)

    image.save(output_path)

    return {"pages": 1, "entities_redacted": entities_redacted}


if __name__ == "__main__":
    # Run as:
    # python -m app.redactors.scanned_pdf_redactor tests\NigerianSamples\scanned_sample.png tests\NigerianSamples\scanned_sample_redacted.png
    import sys

    from app.parsers.scanned_pdf_parser import extract_image_block

    input_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/scanned_sample.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tests/NigerianSamples/scanned_sample_redacted.png"

    block, entities = extract_image_block(input_path)
    summary = redact_image_from_detections(input_path, output_path, block, entities)
    print(f"Redacted {summary['entities_redacted']} entities from {input_path}")
    print(f"Output saved to: {output_path}")