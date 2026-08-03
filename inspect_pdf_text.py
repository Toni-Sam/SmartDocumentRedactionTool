"""
Quick diagnostic: dumps the raw text PyMuPDF extracts per page, plus a rough
"garbage-ness" signal, so we can tell whether a "scanned" PDF is truly
image-only (no text at all) or has an embedded, possibly low-quality OCR
text layer that's producing false-positive PII matches.

Usage (from project root, with venv active):
    python inspect_pdf_text.py path\to\scanned_file.pdf
"""

import re
import sys

import fitz  # PyMuPDF


def alpha_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are alphanumeric."""
    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    alnum = sum(c.isalnum() for c in stripped)
    return alnum / len(stripped)


def main(path: str) -> None:
    doc = fitz.open(path)
    print(f"'{path}' — {doc.page_count} page(s)\n")

    for i, page in enumerate(doc):
        text = page.get_text()
        char_count = len(text.strip())
        words = text.split()
        word_count = len(words)
        ratio = alpha_ratio(text)

        print(f"--- Page {i} ---")
        print(f"  chars extracted: {char_count}")
        print(f"  word count:      {word_count}")
        print(f"  alnum ratio:     {ratio:.2f}")

        if char_count == 0:
            print("  -> No text layer at all (pure image scan).")
        elif ratio < 0.5 or word_count < 3:
            print("  -> Text present but looks low-quality/garbled "
                  "(likely a rough embedded OCR layer).")
        else:
            print("  -> Text looks structurally normal.")

        preview = text.strip().replace("\n", " ")[:200]
        print(f"  preview: {preview!r}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pdf_text.py <path_to_pdf>")
        sys.exit(1)
    main(sys.argv[1])