"""
pdf_parser.py
--------------
Phase A: text-based PDFs (not scanned/image PDFs - those need OCR and get a
separate pdf_ocr_parser.py later).

Extracts text from a PDF page by page using PyMuPDF (fitz), reconstructing
each page's text from word-level extraction. Unlike docx, PDF redaction is
spatial - you can't "replace a run's text" the way python-docx lets you, you
have to know WHERE on the page each character sits. So alongside the plain
text, this module keeps a per-word map of (char_start, char_end) -> bounding
box (fitz.Rect), letting pdf_redactor.py convert a detected entity's
character span back into the rect(s) it needs to redact.

NER CONTEXT FIX
----------------
Earlier versions ran the detection pipeline once against an entire page's
reconstructed text. That meant a name sitting near a form label got
surrounded, in the same string spaCy analyzed, by table headers, numeric
IDs, and unrelated content from elsewhere on the page - and spaCy's NER
would frequently drop the name as a result (NER accuracy degrades sharply
when free-text content is embedded in dense tabular/numeric blobs).

This module now segments each page by PyMuPDF's own block_no (its notion
of a distinct visual region - roughly a paragraph, table cell, or header
line) and runs detection separately per segment, the same "detect per
structural unit" principle docx_parser.py already applies per-paragraph.

The full page text and word_spans are still built exactly as before (so
pdf_redactor.py's rect-mapping logic needs no changes), but PDFPageBlock
now also tracks each structural block's (start, end) character range
within that page text. detect_pii_in_pdf() slices the page text by those
ranges, detects each slice independently, then shifts the returned
start/end offsets back into page-level coordinates before merging results.
Because block ranges are disjoint and non-overlapping by construction,
there's no cross-block overlap to reconcile afterward.

The output shape of detect_pii_in_pdf() - a list of (PDFPageBlock,
entities) tuples, with page-level start/end offsets - is unchanged, so
pdf_redactor.py and app/dispatcher.py require no changes.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Tuple

from app.detectors.merger import detect_all


@dataclass
class WordSpan:
    start: int
    end: int
    rect: fitz.Rect


@dataclass
class PDFPageBlock:
    page_num: int  # 0-indexed
    text: str
    word_spans: List[WordSpan] = field(default_factory=list)
    block_ranges: List[Tuple[int, int]] = field(default_factory=list)
    # (start, end) character ranges within `text`, one per PyMuPDF
    # structural block (block_no), in reading order. Used to run
    # detection per structural unit instead of on the whole page string.


def _build_page_block(page, page_num: int) -> PDFPageBlock:
    """
    Reconstructs a page's text from PyMuPDF's word-level extraction,
    tracking:
      - the character offset range each word occupies in the
        reconstructed text, alongside its bounding box (word_spans)
      - the character offset range each PyMuPDF structural block
        (block_no) occupies in the reconstructed text (block_ranges)

    get_text("words") returns tuples:
        (x0, y0, x1, y1, word, block_no, line_no, word_no)
    """
    words = page.get_text("words")
    words.sort(key=lambda w: (w[5], w[6], w[7]))  # reading order: block, line, word

    text_parts = []
    word_spans = []
    block_ranges = []
    cursor = 0
    prev_block_line = None
    prev_block_no = None
    block_start = 0

    for (x0, y0, x1, y1, word, block_no, line_no, word_no) in words:
        current_block_line = (block_no, line_no)

        if prev_block_no is not None and block_no != prev_block_no:
            block_ranges.append((block_start, cursor))
            block_start = cursor

        if prev_block_line is not None:
            separator = "\n" if current_block_line != prev_block_line else " "
            text_parts.append(separator)
            cursor += len(separator)

        start = cursor
        text_parts.append(word)
        cursor += len(word)
        end = cursor

        word_spans.append(WordSpan(start=start, end=end, rect=fitz.Rect(x0, y0, x1, y1)))
        prev_block_line = current_block_line
        prev_block_no = block_no

    if words:
        block_ranges.append((block_start, cursor))  # close out the final block

    return PDFPageBlock(
        page_num=page_num,
        text="".join(text_parts),
        word_spans=word_spans,
        block_ranges=block_ranges,
    )


def parse_pdf(pdf_path: str) -> List[PDFPageBlock]:
    """Opens a PDF and returns one PDFPageBlock per non-empty page."""
    doc = fitz.open(pdf_path)
    blocks = []
    try:
        for page_num, page in enumerate(doc):
            block = _build_page_block(page, page_num)
            if block.text.strip():
                blocks.append(block)
    finally:
        doc.close()
    return blocks


def detect_pii_in_pdf(pdf_path: str):
    """
    Runs the full detection pipeline against each page of a PDF, one
    structural block (block_ranges) at a time rather than against the
    whole page string - this is the NER context fix. Each block's
    detection results are shifted from block-local offsets back to
    page-level offsets before being collected.

    Returns a list of (PDFPageBlock, entities) tuples, where `entities` is
    a list of dicts shaped like {"text", "entity_type", "start", "end",
    "score", "source"} with start/end as offsets into block.text (the
    full page text) - unchanged from before, so pdf_redactor.py's
    word_spans lookup keeps working without modification.
    """
    blocks = parse_pdf(pdf_path)
    results = []

    for block in blocks:
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
        results.append((block, entities))

    return results


if __name__ == "__main__":
    # Run as: python -m app.parsers.pdf_parser tests\NigerianSamples\sample.pdf
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/DOCX_Redaction_Test.pdf"

    print(f"Parsing: {test_path}\n")
    results = detect_pii_in_pdf(test_path)

    for block, entities in results:
        print(f"--- Page {block.page_num + 1} ---")
        print(f"Text length: {len(block.text)} chars | {len(block.word_spans)} words | "
              f"{len(block.block_ranges)} structural blocks")
        print(f"Entities found: {len(entities)}")
        for e in entities:
            matched = block.text[e["start"]:e["end"]]
            print(f"  [{e['entity_type']}] '{matched}' "
                  f"(score={e.get('score', '?')}, source={e.get('source', '?')})")
        print()