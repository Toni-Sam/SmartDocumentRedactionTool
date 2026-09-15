r"""
pipeline_dump.py

Diagnostic script: runs the REAL pipeline (parse_pdf -> per-segment
detection) against a PDF and prints, for every structural segment:
  - the exact segment_text your code actually reconstructed
  - what local_context_detector finds in it, on its own
  - what masakhaner_detector finds in it, on its own
  - what merger.detect_all() returns AFTER merging the two (plus Presidio)

This exists because every detector has checked out correctly in
isolated unit tests so far, but two specific fields ("Chiamaka Ngozi
Eze" and "Nnewi North") still aren't surviving into the final redacted
output. The remaining explanation has to be something about how these
components interact on the REAL segment text/grouping for this
specific document - which this script surfaces directly instead of
guessing from static code reading.

Usage (run as a module from the project root):
    python -m app.debug.pipeline_dump tests\NigerianSamples\Scanned_Typed_PII_Sample.pdf
"""

import sys

from app.parsers.pdf_parser import parse_pdf
from app.detectors.local_context_detector import detect_local_context
from app.detectors.masakhaner_detector import detect_with_masakhaner
from app.detectors.merger import detect_all


def dump_pipeline(pdf_path: str) -> None:
    blocks = parse_pdf(pdf_path)

    for block in blocks:
        print(f"\n{'=' * 80}")
        print(f"PAGE {block.page_num + 1}  (page_type={block.page_type})")
        print(f"{'=' * 80}")

        for i, (seg_start, seg_end) in enumerate(block.block_ranges):
            segment_text = block.text[seg_start:seg_end]
            if not segment_text.strip():
                continue

            print(f"\n--- Segment {i}  (chars {seg_start}-{seg_end}) ---")
            print(f"segment_text: {segment_text!r}")

            local_results = detect_local_context(segment_text)
            print(f"\n  local_context_detector raw output:")
            if not local_results:
                print("    (none)")
            for r in local_results:
                print(f"    [{r['type']}] {r['text']!r} (pos {r['start']}-{r['end']}, "
                      f"score {r['score']:.2f})")

            masakhaner_results = detect_with_masakhaner(segment_text)
            print(f"\n  masakhaner_detector raw output:")
            if not masakhaner_results:
                print("    (none)")
            for r in masakhaner_results:
                print(f"    [{r['entity_type']}] {r['text']!r} (pos {r['start']}-{r['end']}, "
                      f"score {r['score']}, source={r['source']})")

            merged_results = detect_all(segment_text)
            print(f"\n  merger.detect_all() MERGED output:")
            if not merged_results:
                print("    (none)")
            for r in merged_results:
                print(f"    [{r['entity_type']}] {r['text']!r} (pos {r['start']}-{r['end']}, "
                      f"score {r['score']}, source={r['source']})")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/Scanned_Typed_PII_Sample.pdf"
    dump_pipeline(pdf_path)