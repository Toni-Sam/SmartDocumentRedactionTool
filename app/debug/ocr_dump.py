r"""
ocr_dump.py

Diagnostic script: dumps Tesseract's raw word-level OCR output for a
standalone image OR a page from an image-only PDF - text, full
block/paragraph/line/word grouping, bounding box, and confidence per
word.

v2 change: now captures block_num and par_num in addition to line_num.
Tesseract's line_num resets to 1 at the start of every new block, so
line_num alone cannot be used to tell whether two words on the page
are really on the same physical row - you need the (block_num,
par_num, line_num) triple together for that. This version exists to
diagnose exactly that: whether a label and its value (or two unrelated
rows) are being grouped into the same block/paragraph/line by
Tesseract, which is what downstream row-clustering logic keys off of.

Use this to check:
- Whether a specific PII string (phone, TIN, etc.) was OCR'd correctly
  or came out malformed (wrong spacing, misread character, or dropped
  entirely - dropped words simply won't appear in the output at all).
- Whether words that visually sit on the same row share the same
  (block_num, par_num, line_num) triple, or drifted into a different
  block/line due to skew - this is what determines row-clustering
  behavior downstream.

Usage (run as a module from the project root):
    python -m app.debug.ocr_dump --image tests\NigerianSamples\some_image.jpg
    python -m app.debug.ocr_dump --pdf tests\NigerianSamples\Scanned_Typed_PII_Sample.pdf --csv dump.csv
"""

import argparse
import csv
import io
import sys

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def get_image_from_pdf_page(pdf_path: str, page_num: int = 0) -> Image.Image:
    """Extracts the embedded image from a single-page image-only PDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    images = page.get_images(full=True)
    if not images:
        raise ValueError(f"No embedded images found on page {page_num} of {pdf_path}")
    xref = images[0][0]
    base_image = doc.extract_image(xref)
    doc.close()
    return Image.open(io.BytesIO(base_image["image"]))


def dump_ocr_data(image: Image.Image, csv_out: str | None = None) -> list[dict]:
    """Runs pytesseract.image_to_data() and prints one row per word."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    rows = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        rows.append({
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
            "line_num": data["line_num"][i],
            "word_num": data["word_num"][i],
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "conf": data["conf"][i],
            "text": word,
        })

    print(f"{'blk':>3} {'par':>3} {'line':>4} {'word#':>5} {'left':>5} {'top':>5} "
          f"{'w':>5} {'h':>5} {'conf':>5}  text")
    for r in rows:
        print(f"{r['block_num']:>3} {r['par_num']:>3} {r['line_num']:>4} {r['word_num']:>5} "
              f"{r['left']:>5} {r['top']:>5} {r['width']:>5} {r['height']:>5} "
              f"{r['conf']:>5}  {r['text']}")

    if csv_out and rows:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {csv_out}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump Tesseract word-level OCR data.")
    parser.add_argument("--image", help="Path to a standalone image file")
    parser.add_argument("--pdf", help="Path to an image-only PDF")
    parser.add_argument("--page", type=int, default=0, help="Page number (0-indexed) if using --pdf")
    parser.add_argument("--csv", help="Optional path to save results as CSV")
    args = parser.parse_args()

    if args.image:
        img = Image.open(args.image)
    elif args.pdf:
        img = get_image_from_pdf_page(args.pdf, args.page)
    else:
        print("Provide either --image or --pdf")
        sys.exit(1)

    dump_ocr_data(img, csv_out=args.csv)