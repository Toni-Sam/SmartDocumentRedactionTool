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

AUDIT TRAIL
-----------
redact_pdf_from_detections() now also returns an `audit_trail` list: one
entry per entity that was ACTUALLY redacted (i.e. it resolved to at least
one word rect and a redact_annot was added for it) - not just every entity
that was passed in as "approved". An approved entity whose span doesn't
resolve to any word rect (defensive edge case; shouldn't normally happen
given entities come from this same page's word_spans in the first place)
is silently skipped today, same as before, and is correctly excluded from
the audit trail too, since nothing was actually redacted for it.

Unlike docx_redactor.py's audit_trail (which aggregates by text value,
since DOCX redacts by text match document-wide), each PDF entity is
already a distinct, individually-tracked occurrence - so each audit_trail
entry here corresponds 1:1 with one entity, no aggregation needed.

Each entry is shaped like:
    {
        "entity_type": str,
        "score": Optional[float],
        "source": Optional[str],
        "page_num": int,  # 0-indexed, matches PDFPageBlock.page_num
    }

Deliberately excludes the entity's original text - same reasoning as
docx_redactor.py: an audit artifact shouldn't carry the PII it's
documenting the removal of.
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
) -> Dict:
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

    Returns a summary dict:
        {
            "pages": int,               # len(page_results)
            "entities_redacted": int,   # count of entities that resolved
                                         # to at least one rect and were
                                         # actually redacted
            "audit_trail": list[dict],  # one entry per entity actually
                                         # redacted - see module docstring
        }
    """
    doc = fitz.open(input_path)
    entities_redacted = 0
    audit_trail = []

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
                audit_trail.append({
                    "entity_type": entity.get("entity_type", "UNKNOWN"),
                    "score": entity.get("score"),
                    "source": entity.get("source"),
                    "page_num": block.page_num,
                })

            # Burns the annotations into the content stream - permanently
            # removes the underlying text, not just a visual cover.
            page.apply_redactions()

        doc.save(output_path)
    finally:
        doc.close()

    return {
        "pages": len(page_results),
        "entities_redacted": entities_redacted,
        "audit_trail": audit_trail,
    }


def redact_pdf(input_path: str, output_path: str, fill_color=(0, 0, 0)) -> Dict:
    """
    One-pass convenience wrapper: runs detection + redaction end-to-end on
    a text-based PDF. Kept for backward compatibility / quick one-off use;
    the two-step dispatcher flow calls redact_pdf_from_detections()
    directly instead, with a human-reviewed entity list.

    Returns a summary dict: {"pages", "entities_redacted", "audit_trail"}
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
    print("\naudit_trail:")
    for entry in summary["audit_trail"]:
        print(
            f"  [{entry['entity_type']}] page={entry['page_num'] + 1} "
            f"(score={entry['score']}, source={entry['source']})"
        )