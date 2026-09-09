from app.parsers.pdf_parser import detect_pii_in_pdf
import sys

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/Scanned_Redaction_Test.pdf"
results = detect_pii_in_pdf(pdf_path)

for block, entities in results:
    print(f"=== PAGE {block.page_num + 1} ({block.page_type}) ===")
    print(repr(block.text))
    print()
    print("DETECTED ENTITIES:")
    for e in entities:
        conf = e.get("ocr_confidence")
        low = " <<< LOW CONFIDENCE" if e.get("low_confidence") else ""
        print(f"  [{e['source']}] {e['entity_type']}: {e['text']!r} (ocr_confidence={conf}){low}")
    print()