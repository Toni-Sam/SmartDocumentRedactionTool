from app.parsers.docx_parser import parse_and_detect
from app.redactors.docx_redactor import redact_docx

input_path = "tests/NigerianSamples/DOCX_Redaction_Test.docx"
output_path = "tests/NigerianSamples/DOCX_Redaction_Test_redacted.docx"

entities = parse_and_detect(input_path)
print(f"Entities detected: {len(entities)}")
for e in entities:
    print(f"  [{e.get('entity_type')}] {e.get('text')!r}")

result = redact_docx(input_path, output_path, entities)
print(result)