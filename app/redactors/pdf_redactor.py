"""
pdf_redactor.py
---------------
Applies TRUE content-stream redaction to a text-based PDF using PyMuPDF's
add_redact_annot() / apply_redactions() - this actually strips the
underlying text/glyphs from the page, not just draws a black box over them.
A visual-only overlay is a well-known leak vector (the "redacted" text is
still copy-pasteable underneath); apply_redactions() avoids that.

TWO-STEP FLOW UPDATE
--------------------
This module used to only expose redact_pdf(), which ran detection and
redaction as one inseparable step. That's incompatible with a
detect-then-review-then-apply flow, since a human reviewing a manifest
needs to hand back *edited* entities, not trigger a fresh detection pass.

redact_pdf_from_detections() is now the core: it takes (PDFPageBlock,
entities) pairs - the same shape pdf_parser.detect_pii_in_pdf() returns -
and applies redactions for whatever entities it's given, no detection
call inside. app/dispatcher.py rebuilds this shape from a saved manifest
plus a fresh parse_pdf() call before invoking it.

redact_pdf() is kept as a convenience wrapper for one-pass usage (and so
this file's own inline test and any existing callers keep working
unchanged): it just calls detect_pii_in_pdf() and forwards the result.

Given the (PDFPageBlock, entities) pairs, this module:
  1. Maps each entity's [start, end) character span back to every word rect
     that overlaps it (via PDFPageBlock.word_spans)
  2. Adds a redaction annotation for each of those rects individually
  3. Applies all redactions on the page in one pass

Redacting per-word-rect (rather than one merged bounding box per entity)
matters when a matched entity spans a line break or column - a single
merged bbox could accidentally cover unrelated text in between.
"""

import fitz
from typing import List, Dict, Tuple

from app.parsers.pdf_parser import PDFPageBlock, detect_pii_in_pdf


def _rects_for_span(block: PDFPageBlock, start: int, end: int) -> List[fitz.Rect]:
    """Returns every word rect in `block` whose char range overlaps [start, end)."""
    return [ws.rect for ws in block.word_spans if ws.start < end and ws.end > start]


def redact_pdf_from_detections(
    input_path: str,
    output_path: str,
    page_results: List[Tuple[PDFPageBlock, list]],
    fill_color=(0, 0, 0),
) -> Dict[str, int]:
    """
    Applies redactions for a pre-computed set of (PDFPageBlock, entities)
    pairs - no detection happens here. `entities` for each block is a list
    of dicts with at least "start" and "end" keys (character offsets into
    that block's reconstructed page text).

    This is the entry point the two-step dispatcher flow uses: the
    dispatcher re-parses the PDF to regenerate PDFPageBlocks (with their
    word_spans/rects, which aren't serializable into a JSON manifest),
    then matches a human-reviewed manifest's spans onto those blocks by
    page number before calling this.

    Returns a summary dict: {"pages": n, "entities_redacted": n}
    """
    doc = fitz.open(input_path)
    entities_redacted = 0

    try:
        for block, entities in page_results:
            page = doc[block.page_num]

            for entity in entities:
                rects = _rects_for_span(block, entity["start"], entity["end"])
                if not rects:
                    continue
                for rect in rects:
                    page.add_redact_annot(rect, fill=fill_color)
                entities_redacted += 1

            # Burns the annotations into the content stream - permanently
            # removes the underlying text, not just a visual cover.
            page.apply_redactions()

        doc.save(output_path)
    finally:
        doc.close()

    return {"pages": len(page_results), "entities_redacted": entities_redacted}


def redact_pdf(input_path: str, output_path: str, fill_color=(0, 0, 0)) -> Dict[str, int]:
    """
    One-pass convenience wrapper: runs detection + redaction end-to-end on
    a text-based PDF. Kept for backward compatibility / quick one-off use;
    the two-step dispatcher flow calls redact_pdf_from_detections()
    directly instead, with a human-reviewed entity list.

    Returns a summary dict: {"pages": n, "entities_redacted": n}
    """
    page_results = detect_pii_in_pdf(input_path)
    return redact_pdf_from_detections(input_path, output_path, page_results, fill_color)


if __name__ == "__main__":
    # Run as:
    # python -m app.redactors.pdf_redactor tests\NigerianSamples\sample.pdf tests\NigerianSamples\sample_redacted.pdf
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/DOCX_Redaction_Test.pdf"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tests/NigerianSamples/DOCX_Redaction_Test_redacted.pdf"

    summary = redact_pdf(input_path, output_path)
    print(f"Redacted {summary['entities_redacted']} entities across {summary['pages']} pages")
    print(f"Output saved to: {output_path}")