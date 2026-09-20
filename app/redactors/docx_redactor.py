"""
docx_redactor.py
----------------
Replaces detected PII spans with a redaction label (default "[REDACTED]")
in .docx files, using python-docx.

THE CORE PROBLEM
----------------
Word doesn't store a paragraph's visible text as one continuous string. It
splits it across multiple <w:r> "run" elements based on formatting changes,
spell-check boundaries, autocorrect, and other internal reasons unrelated to
the text's actual content. So an entity like "Emeka Okonkwo" might physically
live in the XML as:

    Run 1: "Emeka Ok"
    Run 2: "onk"
    Run 3: "wo"

A naive `run.text.replace(...)` only catches an entity when it happens to sit
entirely inside a single run. This module reconstructs each paragraph's full
text, finds entity spans at the character level (across run boundaries), and
rewrites the paragraph so the redacted text replaces the original.

Trade-off: when a redaction spans multiple runs, this collapses the whole
paragraph down to a single run carrying the new text (using the first run's
formatting). That guarantees the redaction itself is correct, at the cost of
losing run-level formatting variety (e.g. only part of a paragraph being
bold) in paragraphs that get redacted. Paragraphs with no matches are left
completely untouched.

WHAT GETS SCANNED
-----------------
- Body paragraphs
- Table cells, including nested tables
- Headers and footers, including first-page/even-page variants, across all
  sections

Known limitation: text inside text boxes / floating shapes is not covered
(python-docx doesn't expose these through the normal paragraph API; they'd
need raw OOXML traversal via document.element as a follow-up).

AUDIT TRAIL / LOCATION TRACKING
--------------------------------
Every paragraph scanned (body, table cell, or header/footer) is assigned a
sequential 1-based `paragraph_index` as it's walked, in the same order this
module has always walked the document (body + tables, then headers/footers).
This index is a single flat counter across ALL of those sources - it does
NOT distinguish "this paragraph was in a table" from "this paragraph was in
the body". That's a deliberate simplification: it satisfies "paragraph index
per redaction" without needing a richer location taxonomy (table/row/column
addressing, per-section header naming, etc.). If finer-grained location
labels are wanted later, this is the place to extend.

Since redaction here matches by TEXT VALUE document-wide, one entity value
can legitimately be redacted in several different paragraphs - and even
multiple times within the same paragraph, if it appears more than once
there. `redact_docx()` now returns an `audit_trail` list: one entry per
unique entity text that was actually redacted, with the full list of
paragraph indices it was found in (with a repeated index if it occurred more
than once in that same paragraph) and the total occurrence count. This is
built from what was ACTUALLY redacted (i.e. actually replaced in the
document), not from a naive substring check against the original text -
fixing a prior gap where `entities_matched` could be inflated by an entity
whose text happened to appear as a substring of a longer entity that had
already claimed that span.

USAGE
-----
    from app.redactors.docx_redactor import redact_docx

    entities = [
        {"text": "Emeka Okonkwo", "entity_type": "PERSON_NAME", "score": 0.91},
        {"text": "08012345678",   "entity_type": "NG_PHONE", "score": 0.99},
    ]
    result = redact_docx("input.docx", "output_redacted.docx", entities)
    print(result)
    # {
    #     "entities_matched": 2,
    #     "paragraphs_redacted": 3,
    #     "output": "output_redacted.docx",
    #     "audit_trail": [
    #         {"text": "Emeka Okonkwo", "entity_type": "PERSON_NAME", "score": 0.91,
    #          "source": None, "paragraph_indices": [2, 2, 7], "occurrence_count": 3},
    #         {"text": "08012345678", "entity_type": "NG_PHONE", "score": 0.99,
    #          "source": None, "paragraph_indices": [5], "occurrence_count": 1},
    #     ],
    # }

Entities just need a "text" key to be redacted at all - this is designed to
slot directly into the output of docx_parser.parse_and_detect(). Including
"entity_type" and "score" keys lets the audit_trail carry them through for
the redaction report; if omitted, those fields fall back to "UNKNOWN" / None.
"""

from itertools import count

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


# ---------------------------------------------------------------------------
# Paragraph-level cross-run text replacement
# ---------------------------------------------------------------------------

def _paragraph_full_text(paragraph: Paragraph) -> str:
    """Full visible text of a paragraph, concatenated across all its runs."""
    return "".join(run.text for run in paragraph.runs)


def _find_spans_for_entities(full_text: str, entities: list[dict]) -> list[tuple[int, int, str]]:
    """
    Given a paragraph's full text and the entity list (matched by text
    value), return sorted, non-overlapping (start, end, text) spans to
    redact - the text value is carried alongside each span so callers can
    tell which entity a given occurrence belongs to.

    Entities are matched longest-text-first, so "Emeka Okonkwo" claims its
    span before "Emeka" gets a chance to - preventing a first name from
    being redacted on its own while the surname next to it is missed.
    """
    spans: list[tuple[int, int, str]] = []

    for entity in sorted(entities, key=lambda e: len(e.get("text", "")), reverse=True):
        needle = entity.get("text", "")
        if not needle:
            continue

        search_from = 0
        while True:
            idx = full_text.find(needle, search_from)
            if idx == -1:
                break
            candidate_start, candidate_end = idx, idx + len(needle)
            overlaps_existing = any(
                candidate_start < s_end and s_start < candidate_end
                for s_start, s_end, _ in spans
            )
            if not overlaps_existing:
                spans.append((candidate_start, candidate_end, needle))
            search_from = idx + len(needle)

    spans.sort(key=lambda s: s[0])
    return spans


def _redact_paragraph(paragraph: Paragraph, spans: list[tuple[int, int]], label: str) -> bool:
    """
    Replace the given [start, end) spans (positions in the paragraph's
    full text) with `label`. Spans must already be sorted and
    non-overlapping. Returns True if the paragraph text actually changed.
    """
    if not spans or not paragraph.runs:
        return False

    full_text = _paragraph_full_text(paragraph)
    if not full_text:
        return False

    pieces = []
    cursor = 0
    for start, end in spans:
        if start < cursor or start >= len(full_text):
            continue  # stale/invalid span, skip defensively
        end = min(end, len(full_text))
        pieces.append(full_text[cursor:start])
        pieces.append(label)
        cursor = end
    pieces.append(full_text[cursor:])
    new_text = "".join(pieces)

    if new_text == full_text:
        return False

    # Collapse to a single run so the new text renders correctly, keeping
    # the first run's formatting (font, bold, etc.) as the paragraph's style.
    first_run = paragraph.runs[0]
    first_run.text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""

    return True


def _redact_paragraph_against_entities(
    paragraph: Paragraph, entities: list[dict], label: str
) -> tuple[bool, dict[str, int]]:
    """
    Redact `paragraph` against `entities`. Returns (changed, occurrence_counts)
    where occurrence_counts maps each entity text that was actually redacted
    in this paragraph to how many times it occurred here (usually 1, but can
    be more if the same value appears twice in one paragraph).
    """
    full_text = _paragraph_full_text(paragraph)
    if not full_text:
        return False, {}

    spans = _find_spans_for_entities(full_text, entities)
    if not spans:
        return False, {}

    changed = _redact_paragraph(paragraph, [(s, e) for s, e, _ in spans], label)
    if not changed:
        return False, {}

    occurrence_counts: dict[str, int] = {}
    for _, _, text in spans:
        occurrence_counts[text] = occurrence_counts.get(text, 0) + 1

    return True, occurrence_counts


# ---------------------------------------------------------------------------
# Structural walkers: body, tables (incl. nested), headers/footers
# ---------------------------------------------------------------------------
#
# Each walker now yields (paragraph, paragraph_index) pairs. `counter` is a
# single itertools.count() shared across body, tables, AND headers/footers,
# so every paragraph scanned in the whole document gets a unique, sequential
# 1-based index - regardless of which of those three sources it came from.
# See the "AUDIT TRAIL / LOCATION TRACKING" note in the module docstring.

def _iter_table_paragraphs(table: Table, counter):
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                yield paragraph, next(counter)
            for nested_table in cell.tables:
                yield from _iter_table_paragraphs(nested_table, counter)


def _iter_body_and_table_paragraphs(document: Document, counter):
    for paragraph in document.paragraphs:
        yield paragraph, next(counter)
    for table in document.tables:
        yield from _iter_table_paragraphs(table, counter)


def _iter_header_footer_paragraphs(document: Document, counter):
    for section in document.sections:
        parts = (
            section.header,
            section.footer,
            section.first_page_header,
            section.first_page_footer,
            section.even_page_header,
            section.even_page_footer,
        )
        for part in parts:
            if part is None:
                continue
            for paragraph in part.paragraphs:
                yield paragraph, next(counter)
            for table in part.tables:
                yield from _iter_table_paragraphs(table, counter)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def redact_docx(input_path: str, output_path: str, entities: list[dict],
                 redact_label: str = "[REDACTED]") -> dict:
    """
    Redact every occurrence of `entities` (each a dict with at least a
    "text" key) throughout a .docx file's body, tables, and headers/footers,
    saving the result to `output_path`.

    Returns a summary dict:
        {
            "entities_matched": int,       # unique entity texts actually redacted
            "paragraphs_redacted": int,    # count of paragraphs that were changed
            "output": str,                 # output_path, echoed back
            "audit_trail": list[dict],     # one entry per unique entity text
                                            # actually redacted - see module
                                            # docstring for shape.
        }
    """
    document = Document(input_path)

    usable_entities = [e for e in entities if e.get("text")]
    entity_lookup = {e["text"]: e for e in usable_entities}

    paragraphs_redacted = 0
    matched_texts: set[str] = set()
    location_map: dict[str, list[int]] = {}

    counter = count(1)
    all_paragraphs_with_index = (
        list(_iter_body_and_table_paragraphs(document, counter))
        + list(_iter_header_footer_paragraphs(document, counter))
    )

    for paragraph, para_index in all_paragraphs_with_index:
        full_text_before = _paragraph_full_text(paragraph)
        if not full_text_before:
            continue

        changed, occurrence_counts = _redact_paragraph_against_entities(
            paragraph, usable_entities, redact_label
        )
        if changed:
            paragraphs_redacted += 1
            for text, count_in_paragraph in occurrence_counts.items():
                matched_texts.add(text)
                location_map.setdefault(text, []).extend([para_index] * count_in_paragraph)

    document.save(output_path)

    audit_trail = []
    for text in matched_texts:
        entity = entity_lookup.get(text, {})
        locations = location_map.get(text, [])
        audit_trail.append({
            "text": text,
            "entity_type": entity.get("entity_type", "UNKNOWN"),
            "score": entity.get("score"),
            "source": entity.get("source"),
            "paragraph_indices": locations,
            "occurrence_count": len(locations),
        })

    return {
        "entities_matched": len(matched_texts),
        "paragraphs_redacted": paragraphs_redacted,
        "output": output_path,
        "audit_trail": audit_trail,
    }


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample_entities = [
        {"text": "Emeka Okonkwo", "entity_type": "PERSON_NAME", "score": 0.91, "source": "masakhaner"},
        {"text": "08012345678", "entity_type": "NG_PHONE", "score": 0.99, "source": "presidio"},
        {"text": "Ikeja, Lagos", "entity_type": "ADDRESS", "score": 0.75, "source": "local_context"},
    ]

    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_input.docx"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sample_redacted.docx"

    print(f"Redacting {input_file} -> {output_file}")
    result = redact_docx(input_file, output_file, sample_entities)

    print(f"\nentities_matched: {result['entities_matched']}")
    print(f"paragraphs_redacted: {result['paragraphs_redacted']}")
    print(f"output: {result['output']}")
    print("\naudit_trail:")
    for entry in result["audit_trail"]:
        print(
            f"  [{entry['entity_type']}] {entry['text']!r} "
            f"(score={entry['score']}, source={entry['source']}) "
            f"-> paragraphs {entry['paragraph_indices']} "
            f"({entry['occurrence_count']} occurrence(s))"
        )