"""
Smart Document Redaction Tool — Streamlit GUI Prototype

Run from project root as:
    streamlit run gui/streamlit_app.py

Do NOT run this file directly with `python gui/streamlit_app.py` — it depends
on package-relative imports from `app`, same as every other module in this
project, so Streamlit must be launched from the project root.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------------------------
# `streamlit run gui/streamlit_app.py` only adds this script's own folder
# (gui/) to sys.path - unlike plain `python`, it does NOT add the current
# working directory. That means `from app.dispatcher import ...` below
# fails with ModuleNotFoundError: No module named 'app', even when you
# launch streamlit from the project root. Explicitly add the project root
# (one level up from this file) so the app package resolves correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# NOTE ON IMPORTS
# ---------------------------------------------------------------------------
# app/dispatcher.py exposes a DetectionSession dataclass plus a set of
# free functions that operate on it (NOT methods): detect_document,
# get_entities_for_review, set_approval, set_all_approved, apply_redactions,
# redact_document. DetectionSession itself is never constructed directly -
# always go through detect_document(input_path).
from app.dispatcher import (
    DetectionSession,
    detect_document,
    get_entities_for_review,
    set_approval,
    set_all_approved,
    apply_redactions,
)



# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Document Redaction Tool",
    page_icon="🛡️",
    layout="wide",
)

SUPPORTED_EXTENSIONS = {".docx", ".pdf"}  # xlsx/txt/csv/scanned-pdf pending

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
# DetectionSession is expensive to (re)create because MasakhaNER loads on
# first call. We keep exactly one DetectionSession per uploaded document in
# st.session_state so Streamlit reruns (which happen on every widget
# interaction) don't retrigger detect_document().
if "session" not in st.session_state:
    st.session_state.session = None
if "session_filename" not in st.session_state:
    st.session_state.session_filename = None
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False
if "redacted_bytes" not in st.session_state:
    st.session_state.redacted_bytes = None
if "redacted_filename" not in st.session_state:
    st.session_state.redacted_filename = None
if "redaction_summary" not in st.session_state:
    st.session_state.redaction_summary = None


def reset_session():
    st.session_state.session = None
    st.session_state.session_filename = None
    st.session_state.detection_done = False
    st.session_state.redacted_bytes = None
    st.session_state.redacted_filename = None
    st.session_state.redaction_summary = None


# ---------------------------------------------------------------------------
# Sidebar — upload & document info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Redaction Tool")
    st.caption("Nigerian PII detection & redaction")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["docx", "pdf"],
        help="Text-based PDF and DOCX are supported today. "
        "Scanned PDFs/images (OCR), XLSX, TXT/CSV, and RTF are not yet implemented.",
    )

    if uploaded_file is not None:
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            st.error(f"'{ext}' is not supported yet. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        elif st.session_state.session_filename != uploaded_file.name:
            # New file uploaded — reset any prior session state.
            reset_session()
            st.session_state.session_filename = uploaded_file.name

    st.divider()
    if st.button("↺ Reset / start over", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    with st.expander("ℹ️ Known behavior differences"):
        st.markdown(
            "- **DOCX:** rejecting one match un-redacts *every* occurrence "
            "of that exact text in the document (text-value matching).\n"
            "- **PDF:** each occurrence is tracked and approved "
            "individually — rejecting one leaves the others redacted."
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("Smart Document Redaction Tool")

if uploaded_file is None:
    st.info("Upload a DOCX or text-based PDF from the sidebar to get started.")
    st.stop()

ext = Path(uploaded_file.name).suffix.lower()
if ext not in SUPPORTED_EXTENSIONS:
    st.stop()  # error already shown in sidebar

# ---------------------------------------------------------------------------
# Step 1 — Run detection (once per uploaded file)
# ---------------------------------------------------------------------------
if not st.session_state.detection_done:
    # Persist the upload to a temp file since DetectionSession expects a path.
    suffix = ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner(
        "Running detection — Presidio, local context heuristics, and "
        "MasakhaNER (first load can take a while)…"
    ):
        session = detect_document(tmp_path)

    st.session_state.session = session
    st.session_state.detection_done = True
    st.rerun()

session: DetectionSession = st.session_state.session

# ---------------------------------------------------------------------------
# Step 2 — Review detected entities
# ---------------------------------------------------------------------------
entities = get_entities_for_review(session)

if not entities:
    st.warning("No PII detected in this document.")
    st.stop()

st.subheader(f"Detected entities ({len(entities)})")

col_a, col_b, col_c = st.columns([1, 1, 4])
with col_a:
    if st.button("✅ Approve all"):
        set_all_approved(session, True)
        st.rerun()
with col_b:
    if st.button("❌ Reject all"):
        set_all_approved(session, False)
        st.rerun()

st.caption(
    "Approved entities will be redacted. Rejected entities are left as-is. "
    "Remember: for DOCX, rejecting a match un-redacts *all* matching text "
    "in the document; for PDF, rejection is per-occurrence only."
)

# Group by entity_type for easier scanning, but keep each occurrence
# individually actionable.
entity_types = sorted({e.get("entity_type", "UNKNOWN") for e in entities})
type_filter = st.multiselect("Filter by type", entity_types, default=entity_types)

filtered = [e for e in entities if e.get("entity_type", "UNKNOWN") in type_filter]

# Flag ambiguous NUBAN/NHIS labels for the user rather than presenting
# the type as certain fact.
AMBIGUOUS_TYPES = {"NUBAN", "NHIS"}

header_cols = st.columns([0.5, 2, 1.5, 1, 1])
header_cols[0].markdown("**Approve**")
header_cols[1].markdown("**Text**")
header_cols[2].markdown("**Type**")
header_cols[3].markdown("**Location**")
header_cols[4].markdown("**Scope**")

for entity in filtered:
    entity_id = entity.get("id")
    text_val = entity.get("text", "")
    etype = entity.get("entity_type", "UNKNOWN")
    # page_num only exists on PDF entities (0-indexed); docx has no page concept.
    location = f"page {entity['page_num'] + 1}" if "page_num" in entity else "—"
    is_docx_scope = ext == ".docx"
    scope_label = "All occurrences" if is_docx_scope else "This occurrence"

    row = st.columns([0.5, 2, 1.5, 1, 1])
    approved = row[0].checkbox(
        "", value=entity.get("approved", True), key=f"approve_{entity_id}"
    )
    if approved != entity.get("approved", True):
        set_approval(session, entity_id, approved)

    row[1].write(text_val)

    type_display = etype
    if etype in AMBIGUOUS_TYPES:
        type_display += " ⚠️"
    row[2].write(type_display)
    if etype in AMBIGUOUS_TYPES:
        row[2].caption("Label uncertain — NUBAN/NHIS share the same 10-digit pattern")

    row[3].write(str(location))
    row[4].write(scope_label)

st.divider()

# ---------------------------------------------------------------------------
# Step 3 — Apply redactions & export
# ---------------------------------------------------------------------------
st.subheader("Export")

if st.button("🔒 Apply redactions", type="primary"):
    with st.spinner("Applying redactions…"):
        out_name = f"redacted_{uploaded_file.name}"
        out_path = str(Path(tempfile.gettempdir()) / out_name)
        summary = apply_redactions(session, out_path)

        with open(out_path, "rb") as f:
            st.session_state.redacted_bytes = f.read()
        st.session_state.redacted_filename = out_name
        st.session_state.redaction_summary = summary

    st.success("Redaction complete.")

if st.session_state.get("redaction_summary"):
    with st.expander("Redaction summary", expanded=True):
        st.json(st.session_state.redaction_summary)

if st.session_state.redacted_bytes is not None:
    st.download_button(
        label=f"⬇️ Download {st.session_state.redacted_filename}",
        data=st.session_state.redacted_bytes,
        file_name=st.session_state.redacted_filename,
        mime=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if ext == ".docx"
            else "application/pdf"
        ),
    )