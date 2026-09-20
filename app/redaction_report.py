"""
redaction_report.py

Generates a PDF audit-trail report summarizing what was redacted from a
document: entity type, confidence score, and location, per redacted
occurrence.

Design notes
------------
- This module is intentionally decoupled from DetectionSession / merger.py
  internals. It consumes a list of already-normalized RedactionRecord
  objects. The caller (GUI trigger, or a small collector function you add
  once the real DetectionSession structure is confirmed) is responsible
  for translating approved, kept-redacted entities into RedactionRecord
  instances.
- Generation is on-demand only: this module is never called automatically
  during the redact/approve pipeline. It's invoked when the user clicks a
  "Generate report" button in the GUI.
- The report deliberately does NOT include the original redacted text
  value. Only entity type, confidence, and location are shown. Including
  the actual PII in an "audit" artifact would defeat the purpose of the
  redaction.
- DOCX location model: since DOCX approval is text-value matching
  document-wide (see technical-learnings), a single DOCX entity value can
  correspond to many paragraphs. Per the confirmed decision, DOCX records
  carry BOTH the full list of paragraph indices where the value occurred
  AND a total occurrence count.
- PDF location model: PDF entities are tracked per-occurrence with live
  fitz.Rect bounding boxes, so each occurrence gets its own record with a
  page number (and optionally a bounding box, included for completeness
  but not required by the spec).

Run standalone for a smoke test:
    python -m app.reporting.redaction_report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"


@dataclass
class RedactionRecord:
    """
    One row in the audit report. Represents a single redacted occurrence
    (PDF) or a single redacted entity value (DOCX, which may span several
    paragraphs).
    """

    entity_type: str
    document_type: DocumentType
    confidence: Optional[float] = None  # 0.0-1.0, or None if unavailable

    # PDF-specific location fields
    page_number: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1

    # DOCX-specific location fields
    paragraph_indices: Optional[List[int]] = field(default=None)
    occurrence_count: Optional[int] = None

    def location_str(self) -> str:
        if self.document_type == DocumentType.PDF:
            if self.page_number is not None:
                return f"Page {self.page_number}"
            return "Page unknown"
        else:  # DOCX
            count = self.occurrence_count
            if self.paragraph_indices:
                para_list = ", ".join(str(i) for i in self.paragraph_indices)
                if len(self.paragraph_indices) > 6:
                    # keep the report readable if one value repeats a lot
                    shown = ", ".join(str(i) for i in self.paragraph_indices[:6])
                    para_list = f"{shown}, ... (+{len(self.paragraph_indices) - 6} more)"
                count_str = f" ({count} occurrence{'s' if count != 1 else ''})" if count else ""
                return f"Paragraph(s) {para_list}{count_str}"
            if count:
                return f"{count} occurrence{'s' if count != 1 else ''} (paragraph list unavailable)"
            return "Location unavailable"

    def confidence_str(self) -> str:
        if self.confidence is None:
            return "N/A"
        return f"{self.confidence * 100:.1f}%"


def _summarize_by_entity_type(records: Sequence[RedactionRecord]) -> List[Tuple[str, int]]:
    counts = {}
    for r in records:
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


class _AuditReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Redaction Audit Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_audit_report(
    records: Sequence[RedactionRecord],
    source_filename: str,
    output_path: str,
) -> str:
    """
    Build the audit-trail PDF and write it to output_path.

    Parameters
    ----------
    records : the redacted occurrences to report on (already filtered to
        approved / kept-redacted entities only — rejected entities should
        never reach this function).
    source_filename : original document name, shown in the report header.
    output_path : where to write the generated PDF.

    Returns
    -------
    The output_path, for convenience when chaining into a download step.
    """
    pdf = _AuditReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Metadata block ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Document Information", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Source file: {source_filename}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc_type = records[0].document_type.value.upper() if records else "N/A"
    pdf.cell(0, 6, f"Document type: {doc_type}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Total redactions: {len(records)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Summary by entity type ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Summary by Entity Type", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    summary = _summarize_by_entity_type(records)

    if summary:
        with pdf.table(col_widths=(70, 30), text_align=("LEFT", "CENTER")) as table:
            header_row = table.row()
            header_row.cell("Entity Type")
            header_row.cell("Count")
            for entity_type, count in summary:
                row = table.row()
                row.cell(entity_type)
                row.cell(str(count))
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No redactions recorded.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(6)

    # --- Detail table ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Redaction Detail", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if records:
        with pdf.table(
            col_widths=(60, 30, 90),
            text_align=("LEFT", "CENTER", "LEFT"),
        ) as table:
            header_row = table.row()
            header_row.cell("Entity Type")
            header_row.cell("Confidence")
            header_row.cell("Location")

            # Stable, readable ordering: by page/paragraph where available,
            # then by entity type.
            def sort_key(r: RedactionRecord):
                if r.document_type == DocumentType.PDF:
                    return (r.page_number if r.page_number is not None else 10**9, r.entity_type)
                else:
                    first_para = r.paragraph_indices[0] if r.paragraph_indices else 10**9
                    return (first_para, r.entity_type)

            for r in sorted(records, key=sort_key):
                row = table.row()
                row.cell(r.entity_type)
                row.cell(r.confidence_str())
                row.cell(r.location_str())
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No redactions recorded.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(output_path)
    return output_path


def build_docx_records(audit_trail: Sequence[dict]) -> List[RedactionRecord]:
    """
    Convert the "audit_trail" list returned by
    app.redactors.docx_redactor.redact_docx() into RedactionRecord objects
    ready for generate_audit_report().

    Each audit_trail entry is shaped like:
        {
            "text": str,               # dropped here - never enters the report
            "entity_type": str,
            "score": Optional[float],
            "source": Optional[str],   # dropped here - not part of the report spec
            "paragraph_indices": list[int],
            "occurrence_count": int,
        }
    """
    records = []
    for entry in audit_trail:
        records.append(
            RedactionRecord(
                entity_type=entry.get("entity_type", "UNKNOWN"),
                document_type=DocumentType.DOCX,
                confidence=entry.get("score"),
                paragraph_indices=entry.get("paragraph_indices") or None,
                occurrence_count=entry.get("occurrence_count"),
            )
        )
    return records


def build_pdf_records(audit_trail: Sequence[dict]) -> List[RedactionRecord]:
    """
    Convert the "audit_trail" list returned by
    app.redactors.pdf_redactor.redact_pdf_from_detections() into
    RedactionRecord objects ready for generate_audit_report().

    Each audit_trail entry is shaped like:
        {
            "entity_type": str,
            "score": Optional[float],
            "source": Optional[str],   # dropped here - not part of the report spec
            "page_num": int,           # 0-indexed, matches PDFPageBlock.page_num
        }

    Unlike DOCX, PDF entities are already one-per-occurrence (not
    deduplicated by text value), so this is a direct 1:1 mapping - no
    aggregation needed.
    """
    records = []
    for entry in audit_trail:
        page_num = entry.get("page_num")
        records.append(
            RedactionRecord(
                entity_type=entry.get("entity_type", "UNKNOWN"),
                document_type=DocumentType.PDF,
                confidence=entry.get("score"),
                # Report is human-facing: display 1-indexed page numbers,
                # even though page_num is stored 0-indexed everywhere else
                # in this codebase (matches PDFPageBlock.page_num).
                page_number=(page_num + 1) if page_num is not None else None,
            )
        )
    return records


if __name__ == "__main__":
    # Inline smoke test, per project convention — no separate test file.
    # Run with: python -m app.reporting.redaction_report
    sample_records = [
        RedactionRecord(
            entity_type="NIN",
            document_type=DocumentType.PDF,
            confidence=0.97,
            page_number=1,
            bbox=(72.0, 100.0, 180.0, 112.0),
        ),
        RedactionRecord(
            entity_type="PHONE_NUMBER",
            document_type=DocumentType.PDF,
            confidence=0.88,
            page_number=1,
            bbox=(72.0, 130.0, 160.0, 142.0),
        ),
        RedactionRecord(
            entity_type="PERSON_NAME",
            document_type=DocumentType.PDF,
            confidence=None,
            page_number=2,
            bbox=(50.0, 60.0, 140.0, 72.0),
        ),
        RedactionRecord(
            entity_type="PERSON_NAME",
            document_type=DocumentType.DOCX,
            confidence=0.91,
            paragraph_indices=[2, 5, 5, 11],
            occurrence_count=4,
        ),
        RedactionRecord(
            entity_type="BVN",
            document_type=DocumentType.DOCX,
            confidence=0.99,
            paragraph_indices=[7],
            occurrence_count=1,
        ),
    ]

    out = generate_audit_report(
        sample_records,
        source_filename="DOCX_Redaction_Test.docx",
        output_path="sample_audit_report.pdf",
    )
    print(f"Sample report written to: {out}")