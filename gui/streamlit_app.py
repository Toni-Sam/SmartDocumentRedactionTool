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
# get_entities_for_review, set_approval, set_all_approved,
# set_approval_by_types, apply_redactions, redact_document.
# DetectionSession itself is never constructed directly - always go
# through detect_document(input_path).
from app.dispatcher import (
    DetectionSession,
    detect_document,
    get_entities_for_review,
    set_approval,
    set_all_approved,
    set_approval_by_types,
    apply_redactions,
)

# app/redaction_policies.py defines named subsets of entity types
# ("Financial IDs only", "Names + Location only", "Full NDPA profile")
# plus a "Custom" path for an arbitrary user-chosen set. See that
# module's docstring for how a policy relates to the approval flow above.
from app.redaction_policies import (
    PRESET_POLICIES,
    DEFAULT_POLICY_NAME,
    ALL_ENTITY_TYPES,
    resolve_policy_types,
)



# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Document Redaction Tool",
    page_icon="🛡️",
    layout="wide",
)

SUPPORTED_EXTENSIONS = {
    ".docx", ".pdf",
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
}  # xlsx/txt/csv pending

# Maps an uploaded file's extension to the MIME type used for the download
# button, and to the extension used to persist the redacted output (image
# outputs preserve the original extension - a .jpg upload produces a .jpg
# download, not a normalized format).
MIME_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}

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
if "approval_revision" not in st.session_state:
    st.session_state.approval_revision = 0
if "applied_policy_signature" not in st.session_state:
    # Tracks which (policy_name, custom_types) combo was last APPLIED to
    # session.entities via set_approval_by_types() - not just selected in
    # the widgets. Compared against the widgets' current value each rerun
    # so a policy is only (re-)applied when the user actually changes it,
    # not on every unrelated rerun (e.g. toggling one checkbox in the
    # review table below shouldn't stomp on other rows' manual overrides).
    st.session_state.applied_policy_signature = None


def friendly_error_message(exc: Exception, ext: str) -> str:
    """
    Maps common low-level exceptions (corrupted files, password-protected
    PDFs, bad zip structures in DOCX) to a plain-language message. Falls
    back to a generic message for anything unrecognized - the raw
    exception is still shown separately in a collapsed "technical
    details" expander for debugging, it's just not the headline message.
    """
    msg = str(exc).lower()

    password_signals = ("password", "encrypt", "decrypt", "needs_pass")
    corruption_signals = (
        "bad zip", "not a zip file", "packagenotfounderror", "damaged",
        "cannot open", "file data error", "invalid pdf", "syntax error",
    )

    if any(sig in msg for sig in password_signals):
        return (
            "This file appears to be password-protected. Please remove "
            "the password (or provide an unprotected copy) and upload again — "
            "this tool cannot open encrypted documents."
        )

    if any(sig in msg for sig in corruption_signals):
        return (
            f"This {ext.upper().lstrip('.')} file appears to be corrupted or "
            "isn't a valid document of that type. Try re-saving or "
            "re-exporting it, then upload again."
        )

    return (
        "Something went wrong while processing this file, and it doesn't "
        "match a known cause (password protection or corruption). See "
        "'Technical details' below, or try a different file."
    )


def reset_session():
    st.session_state.session = None
    st.session_state.session_filename = None
    st.session_state.detection_done = False
    st.session_state.redacted_bytes = None
    st.session_state.redacted_filename = None
    st.session_state.redaction_summary = None
    st.session_state.approval_revision = 0
    st.session_state.applied_policy_signature = None
    # Clear the policy widgets' own persisted values too, so a newly
    # uploaded document starts back at the default policy rather than
    # carrying over whatever the previous document had selected.
    st.session_state.pop("policy_name", None)
    st.session_state.pop("policy_custom_types", None)


# ---------------------------------------------------------------------------
# Sidebar — upload & document info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Redaction Tool")
    st.caption("Nigerian PII detection & redaction")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["docx", "pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        help="Text-based PDF, DOCX, and standalone images (JPG/PNG/TIFF/BMP) "
        "are supported today. XLSX, TXT/CSV, and RTF are not yet implemented.",
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
    st.info("Upload a DOCX, text-based PDF, or image (JPG/PNG/TIFF/BMP) from the sidebar to get started.")
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

    try:
        with st.spinner(
            "Running detection — Presidio, local context heuristics, and "
            "MasakhaNER (first load can take a while)…"
        ):
            session = detect_document(tmp_path)
    except Exception as exc:
        st.error(friendly_error_message(exc, ext))
        with st.expander("Technical details"):
            st.exception(exc)
        st.stop()

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

# ---------------------------------------------------------------------------
# Redaction policy — sets default approval per entity type
# ---------------------------------------------------------------------------
# This runs BEFORE the manual approve/reject controls below, and only
# changes the STARTING approval state for each entity type - every row is
# still individually reviewable afterward via its own checkbox, exactly
# as before this feature existed.
st.markdown("**Redaction policy**")

policy_options = list(PRESET_POLICIES.keys()) + ["Custom"]
default_index = policy_options.index(DEFAULT_POLICY_NAME)

policy_name = st.selectbox(
    "Choose which categories of PII to redact",
    policy_options,
    index=default_index,
    key="policy_name",
    help="Sets which entity types are approved for redaction by default. "
    "You can still uncheck or check any individual match in the table below.",
)

if policy_name == "Custom":
    custom_types = st.multiselect(
        "Select entity types to redact",
        sorted(ALL_ENTITY_TYPES),
        key="policy_custom_types",
    )
    if not custom_types:
        st.info(
            "No entity types selected yet — nothing will be approved for "
            "redaction until you pick some here, or switch to a preset above."
        )
else:
    custom_types = None

policy_signature = (policy_name, tuple(sorted(custom_types)) if custom_types else None)

if st.session_state.applied_policy_signature != policy_signature:
    allowed_types = resolve_policy_types(policy_name, custom_types)
    set_approval_by_types(session, allowed_types)
    st.session_state.applied_policy_signature = policy_signature
    # New approval state per entity - bump the revision counter so the
    # checkboxes below (keyed on approval_revision) re-render with their
    # new default values instead of showing stale widget state.
    st.session_state.approval_revision += 1
    st.rerun()

st.caption(
    f"Policy '{policy_name}' pre-approves "
    f"{sum(1 for e in entities if e.get('approved', True))}/{len(entities)} "
    "detected entities. Adjust individual rows below as needed."
)

st.divider()

col_a, col_b, col_c = st.columns([1, 1, 4])
with col_a:
    if st.button("Approve all"):
        set_all_approved(session, True)
        st.session_state.approval_revision += 1
        st.rerun()
with col_b:
    if st.button("Reject all"):
        set_all_approved(session, False)
        st.session_state.approval_revision += 1
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
        "",
        value=entity.get("approved", True),
        key=f"approve_{entity_id}_{st.session_state.approval_revision}",
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

if st.button("Apply redactions", type="primary"):
    out_name = f"redacted_{uploaded_file.name}"
    out_path = str(Path(tempfile.gettempdir()) / out_name)

    try:
        with st.spinner("Applying redactions…"):
            summary = apply_redactions(session, out_path)

        with open(out_path, "rb") as f:
            st.session_state.redacted_bytes = f.read()
        st.session_state.redacted_filename = out_name
        st.session_state.redaction_summary = summary

        st.success("Redaction complete.")
    except Exception as exc:
        st.error(friendly_error_message(exc, ext))
        with st.expander("Technical details"):
            st.exception(exc)

if st.session_state.get("redaction_summary"):
    summary = st.session_state.redaction_summary
    rejected_count = sum(1 for e in entities if not e.get("approved", True))

    # DOCX and PDF redactors return differently-named keys for the same
    # concepts (entities_matched vs entities_redacted, paragraphs_redacted
    # vs pages) rather than a shared schema - normalize here.
    matched_count = summary.get("entities_matched", summary.get("entities_redacted"))
    unit_count = summary.get("paragraphs_redacted", summary.get("pages"))
    unit_label = "Paragraphs redacted" if "paragraphs_redacted" in summary else "Pages"

    st.subheader("Redaction summary")
    metric_cols = st.columns(3)
    if matched_count is not None:
        metric_cols[0].metric("Entities matched", matched_count)
    metric_cols[1].metric("Rejected (left un-redacted)", rejected_count)
    if unit_count is not None:
        metric_cols[2].metric(unit_label, unit_count)

    known_keys = {"entities_matched", "entities_redacted", "paragraphs_redacted", "pages", "output"}
    extra = {k: v for k, v in summary.items() if k not in known_keys}
    if extra:
        with st.expander("Additional details"):
            st.json(extra)

if st.session_state.redacted_bytes is not None:
    st.download_button(
        label=f"Download {st.session_state.redacted_filename}",
        data=st.session_state.redacted_bytes,
        file_name=st.session_state.redacted_filename,
        mime=MIME_TYPES.get(ext, "application/octet-stream"),
    )