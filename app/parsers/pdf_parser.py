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

The detection pipeline itself (merger.merge_detections) is untouched - it
just receives plain text same as it does from docx_parser.py.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List

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


def _build_page_block(page, page_num: int) -> PDFPageBlock:
    """
    Reconstructs a page's text from PyMuPDF's word-level extraction,
    tracking the character offset range each word occupies in the
    reconstructed text alongside its bounding box.

    get_text("words") returns tuples:
        (x0, y0, x1, y1, word, block_no, line_no, word_no)
    """
    words = page.get_text("words")
    words.sort(key=lambda w: (w[5], w[6], w[7]))  # reading order: block, line, word

    text_parts = []
    word_spans = []
    cursor = 0
    prev_block_line = None

    for (x0, y0, x1, y1, word, block_no, line_no, word_no) in words:
        current_block_line = (block_no, line_no)

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

    return PDFPageBlock(page_num=page_num, text="".join(text_parts), word_spans=word_spans)


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
    Runs the full detection pipeline against each page of a PDF.

    Returns a list of (PDFPageBlock, entities) tuples, where `entities` is
    whatever merger.merge_detections(text) returns - a list of dicts shaped
    like {"text", "entity_type", "start", "end", "score", "source"}.
    """
    blocks = parse_pdf(pdf_path)
    results = []
    for block in blocks:
        entities = detect_all(block.text)
        results.append((block, entities))
    return results


if __name__ == "__main__":
    # Run as: python -m app.parsers.pdf_parser tests\nigerian_samples\sample.pdf
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/nigerian_samples/sample.pdf"

    print(f"Parsing: {test_path}\n")
    results = detect_pii_in_pdf(test_path)

    for block, entities in results:
        print(f"--- Page {block.page_num + 1} ---")
        print(f"Text length: {len(block.text)} chars | {len(block.word_spans)} words")
        print(f"Entities found: {len(entities)}")
        for e in entities:
            matched = block.text[e["start"]:e["end"]]
            print(f"  [{e['entity_type']}] '{matched}' "
                  f"(score={e.get('score', '?')}, source={e.get('source', '?')})")
        print()