"""
Smart Document Redaction Tool — Streamlit GUI Prototype

Run from project root as:
    streamlit run gui/streamlit_app.py

Do NOT run this file directly with `python gui/streamlit_app.py` — it depends
on package-relative imports from `app`, same as every other module in this
project, so Streamlit must be launched from the project root.

LAYOUT
------
Three tabs: Upload / Review & Preview / Export. Tab bodies all run on
every rerun (Streamlit doesn't lazy-load tab content) - only the ACTIVE
tab's output is visually shown, so `session`/`entities`/`ext` are
computed once, right after the Upload tab's body, and reused by the
other two. Each of the latter two tabs guards itself with "upload a
document first" if no session exists yet, since there's no single
`st.stop()` gate anymore (a global stop would prevent the tab bar
itself from rendering, which defeats the point of tabs existing before
a file is uploaded).

THEME
-----
Colors come from PROJECT_ROOT/.streamlit/config.toml (create the
.streamlit folder if it doesn't exist yet - see that file's own header
comment). A small CSS block below layers on header/tab/metric spacing
that plain theme config can't reach. The `button[data-baseweb="tab"]`
selector targets Streamlit's current tab-bar implementation and could
need adjusting on a future Streamlit version upgrade if tabs stop
picking up the styling.
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
# set_approval_by_types, add_manual_entity, apply_redactions.
# DetectionSession itself is never constructed directly - always go
# through detect_document(input_path).
from app.dispatcher import (
    DetectionSession,
    detect_document,
    get_entities_for_review,
    set_approval,
    set_all_approved,
    set_approval_by_types,
    add_manual_entity,
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

# app/preview.py renders the document (with detected PII highlighted) for
# the Review & Preview tab - read-only, never touches the source file.
from app.preview import (
    get_page_count,
    render_page_with_highlights,
    render_docx_preview_html,
    render_legend_html,
)


# ---------------------------------------------------------------------------
# Page config + theme polish
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Xpunge",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Streamlit renders its own fixed toolbar (hamburger menu / Deploy
       button) pinned to the top of the viewport, and reserves a generous
       chunk of .block-container's top padding by default specifically so
       page content clears it. An earlier version of this file trimmed
       that down to 1.4rem for a tighter look, which was LESS than the
       toolbar's height - so the top of the custom header below (icon +
       title) rendered partly underneath/behind that fixed bar and got
       visually clipped. 3.5rem clears it with a bit of room to spare; if
       a future Streamlit version changes the toolbar's height, this may
       need nudging up or down. */
    .block-container { padding-top: 3.5rem; padding-bottom: 3rem; }
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid #E4E9E8;
        margin-bottom: 1.3rem;
    }
    .app-header .icon { font-size: 2.1rem; line-height: 1.2; }
    .app-header h1 { margin: 0; padding: 0; font-size: 1.55rem; }
    .app-header p { margin: 0; padding: 0; color: #5B6B68; font-size: 0.92rem; }
    button[data-baseweb="tab"] { font-size: 1rem; font-weight: 600; padding: 0.55rem 1.05rem; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-header"><span class="icon">🛡️</span>'
    "<div><h1>Xpunge</h1>"
    "<p>Find and redact sensitive personal information like ID numbers, "
    "phone numbers, and names in Nigerian DOCX, PDF, and image documents.</p>"
    "</div></div>",
    unsafe_allow_html=True,
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
# Plain-language labels for entity_type codes
# ---------------------------------------------------------------------------
# The detectors (see app/detectors/nigerian_patterns.py) store entity types
# as short internal codes like "NG_NIN" or "NG_NUBAN" - accurate for the
# code, but not something most reviewers will recognize on sight. This maps
# each known code to a plain-language label for display ONLY - the raw code
# is still what's stored on the entity dict and used for filtering/policy
# matching, so nothing downstream needs to change.
FRIENDLY_TYPE_LABELS = {
    "NG_NIN": "National ID Number (NIN)",
    "NG_BVN": "Bank Verification Number (BVN)",
    "NG_PASSPORT": "Passport Number",
    "NG_PHONE": "Phone Number",
    "NG_PLATE": "Vehicle Plate Number",
    "NG_VIN": "Vehicle ID Number (VIN)",
    "NG_DRIVERS_LICENSE": "Driver's License Number",
    "NG_TIN": "Tax ID Number (TIN)",
    "NG_NUBAN": "Bank Account Number (NUBAN)",
    "NG_RSA_PIN": "Pension Number (RSA PIN)",
    "NG_NYSC": "NYSC Number",
    "NG_NYSC_STATE_CODE": "NYSC State Code",
    "NG_CAC_RC": "Company Registration Number (CAC/RC)",
    "NG_NHIS": "Health Insurance Number (NHIS)",
    "PERSON_NAME": "Person's Name",
    "ADDRESS": "Address",
    "LGA": "Local Government Area (LGA)",
    "UNKNOWN": "Unknown",
}


def friendly_type(etype: str) -> str:
    """Plain-language label for an entity_type code, falling back to the
    raw code itself for anything not in the table above (e.g. a custom
    label someone typed into the manual-mark box)."""
    return FRIENDLY_TYPE_LABELS.get(etype, etype)


# Sentinel used by the manual-mark entity-type dropdown for "let me type my
# own label" - a dedicated sentinel (rather than reusing a display string
# like "Other") means it can never collide with a real entity_type code.
_CUSTOM_TYPE_SENTINEL = "__custom__"

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
    st.session_state.pop("preview_page", None)


# ---------------------------------------------------------------------------
# Sidebar — global controls only; the uploader itself now lives in the
# "Upload" tab so it's part of the tabbed flow rather than always-visible
# chrome.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### 🛡️ Smart Document Redaction Tool")
    st.caption("Finds and removes personal information from documents")

    if st.button("↺ Reset / start over", use_container_width=True):
        reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_review, tab_export = st.tabs(
    ["📤 Upload", "🔍 Review & Preview", "📦 Export"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Upload (runs detection once per uploaded file)
# ---------------------------------------------------------------------------
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["docx", "pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        help="PDF, Word (.docx), and image files (JPG/PNG/TIFF/BMP) are "
        "supported today. Excel, plain text, and RTF files aren't yet.",
    )

    ext = None
    if uploaded_file is not None:
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            st.error(f"'{ext}' is not supported yet. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            ext = None
        elif st.session_state.session_filename != uploaded_file.name:
            # New file uploaded — reset any prior session state.
            reset_session()
            st.session_state.session_filename = uploaded_file.name

    if uploaded_file is not None and ext is not None and not st.session_state.detection_done:
        # Persist the upload to a temp file since DetectionSession expects a path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner(
                "Scanning the document for personal information — this can "
                "take a little longer the first time…"
            ):
                new_session = detect_document(tmp_path)
            st.session_state.session = new_session
            st.session_state.detection_done = True
            st.rerun()
        except Exception as exc:
            st.error(friendly_error_message(exc, ext))
            with st.expander("Technical details"):
                st.exception(exc)

    if st.session_state.session is not None:
        _session_for_summary: DetectionSession = st.session_state.session
        _entities_for_summary = get_entities_for_review(_session_for_summary)
        _ext_for_summary = Path(st.session_state.session_filename).suffix.lower()

        st.success(
            f"**{st.session_state.session_filename}** processed — "
            f"found {len(_entities_for_summary)} item(s) of personal information."
        )
        meta_cols = st.columns(3)
        meta_cols[0].metric("File type", _ext_for_summary.lstrip(".").upper())
        meta_cols[1].metric("Items found", len(_entities_for_summary))
        if _session_for_summary.file_type in ("pdf", "image"):
            try:
                meta_cols[2].metric("Pages", get_page_count(_session_for_summary.input_path))
            except Exception:
                pass

        flagged = getattr(_session_for_summary, "flagged_pages", None)
        if flagged:
            st.warning(
                f"{len(flagged)} page(s) look like scanned images and may need "
                "extra checking — review them carefully in **Review & Preview**."
            )

        st.info("Head over to **🔍 Review & Preview** to check the results.")
    else:
        st.info(
            "Upload a Word document, PDF, or image (JPG/PNG/TIFF/BMP) above "
            "to get started."
        )

# ---------------------------------------------------------------------------
# Shared state for the two tabs below — computed once, right after Upload's
# body runs, so both tabs see the same session/entities/ext this rerun.
# ---------------------------------------------------------------------------
session: DetectionSession = st.session_state.session
entities = get_entities_for_review(session) if session is not None else []
ext = Path(st.session_state.session_filename).suffix.lower() if st.session_state.session_filename else None

# ---------------------------------------------------------------------------
# Tab 2 — Review & Preview
# ---------------------------------------------------------------------------
with tab_review:
    if session is None:
        st.info("Upload a document in the **📤 Upload** tab first.")
    elif not entities:
        st.warning("No personal information was found in this document.")
    else:
        preview_col, controls_col = st.columns([1.15, 1], gap="large")

        # -------------------------------------------------------------
        # Left: document preview, highlighted by current approval state
        # -------------------------------------------------------------
        with preview_col:
            st.markdown("**Document preview**")
            st.caption(
                "Personal information the tool found is highlighted here. Use "
                "this to spot anything it may have missed — like a "
                "registration number it doesn't recognize — then add it in "
                "the panel on the right."
            )
            st.markdown(render_legend_html(), unsafe_allow_html=True)

            if session.file_type in ("pdf", "image"):
                try:
                    page_count = get_page_count(session.input_path)
                except Exception as exc:
                    st.error("Couldn't open this file for preview.")
                    with st.expander("Technical details"):
                        st.exception(exc)
                    page_count = 0

                selected_page = 0
                if page_count > 1:
                    selected_page = st.number_input(
                        "Page", min_value=1, max_value=page_count, value=1, step=1,
                        key="preview_page",
                    ) - 1

                if page_count:
                    try:
                        with st.spinner("Rendering page…"):
                            page_image = render_page_with_highlights(
                                session.input_path, session._page_results, selected_page,
                            )
                        st.image(page_image, use_container_width=True)
                    except Exception as exc:
                        st.error("Couldn't render this page.")
                        with st.expander("Technical details"):
                            st.exception(exc)

            elif session.file_type == "docx":
                try:
                    docx_html = render_docx_preview_html(session.input_path, session.entities)
                    st.markdown(
                        '<div style="max-height: 640px; overflow-y: auto; '
                        'border: 1px solid #E4E9E8; border-radius: 8px; '
                        'padding: 16px; background: #FFFFFF;">'
                        f"{docx_html}</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error("Couldn't render a preview for this document.")
                    with st.expander("Technical details"):
                        st.exception(exc)
            else:
                st.info("Preview isn't available for this file type yet.")

        # -------------------------------------------------------------
        # Right: policy, approve/reject, manual mark, filter + table
        # -------------------------------------------------------------
        with controls_col:
            st.markdown("**What to redact**")

            policy_options = list(PRESET_POLICIES.keys()) + ["Custom"]
            default_index = policy_options.index(DEFAULT_POLICY_NAME)

            policy_name = st.selectbox(
                "Choose which types of personal information to redact",
                policy_options,
                index=default_index,
                key="policy_name",
                help="Sets which types of personal information are "
                "automatically approved for redaction. You can still check "
                "or uncheck any individual match in the list below.",
            )

            if policy_name == "Custom":
                custom_types = st.multiselect(
                    "Select the types of personal information to redact",
                    sorted(ALL_ENTITY_TYPES),
                    key="policy_custom_types",
                    format_func=friendly_type,
                )
                if not custom_types:
                    st.info(
                        "No types selected yet — nothing will be approved "
                        "for redaction until you pick some here, or switch "
                        "to a preset above."
                    )
            else:
                custom_types = None

            policy_signature = (policy_name, tuple(sorted(custom_types)) if custom_types else None)

            if st.session_state.applied_policy_signature != policy_signature:
                allowed_types = resolve_policy_types(policy_name, custom_types)
                set_approval_by_types(session, allowed_types)
                st.session_state.applied_policy_signature = policy_signature
                # New approval state per entity - bump the revision counter so
                # the checkboxes below (keyed on approval_revision) re-render
                # with their new default values instead of showing stale
                # widget state.
                st.session_state.approval_revision += 1
                st.rerun()

            st.caption(
                f"The '{policy_name}' option pre-approves "
                f"{sum(1 for e in entities if e.get('approved', True))} of {len(entities)} "
                "matches for redaction. Adjust individual rows below as needed."
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Approve all", use_container_width=True):
                    set_all_approved(session, True)
                    st.session_state.approval_revision += 1
                    st.rerun()
            with col_b:
                if st.button("Reject all", use_container_width=True):
                    set_all_approved(session, False)
                    st.session_state.approval_revision += 1
                    st.rerun()

            st.divider()

            # -----------------------------------------------------------
            # Manually mark additional PII (the pipeline missed it - e.g.
            # a unique registration number with no pattern recognizer)
            # -----------------------------------------------------------
            st.markdown("**Add something the tool missed**")
            st.caption(
                "Spotted something in the preview that wasn't caught "
                "automatically? Type it in exactly as it appears — "
                "capitalization doesn't matter, and every matching spot in "
                "the document will be redacted. What you add here shows up "
                "already approved, both below and in the preview."
            )

            manual_col1, manual_col2, manual_col3 = st.columns([2, 1.5, 1])

            with manual_col1:
                manual_value = st.text_input(
                    "Value to add",
                    key="manual_value",
                    label_visibility="collapsed",
                    placeholder="e.g. a registration number the tool didn't catch",
                )

            with manual_col2:
                manual_type_options = sorted(ALL_ENTITY_TYPES) + [_CUSTOM_TYPE_SENTINEL]
                manual_type_choice = st.selectbox(
                    "Entity type",
                    manual_type_options,
                    key="manual_type_choice",
                    label_visibility="collapsed",
                    format_func=lambda t: (
                        "Other (type your own)" if t == _CUSTOM_TYPE_SENTINEL else friendly_type(t)
                    ),
                )
                if manual_type_choice == _CUSTOM_TYPE_SENTINEL:
                    manual_custom_type = st.text_input(
                        "Custom label",
                        key="manual_custom_type",
                        placeholder="e.g. Membership Number",
                        label_visibility="collapsed",
                    )
                    resolved_manual_type = manual_custom_type.strip() or "Custom"
                else:
                    resolved_manual_type = manual_type_choice

            with manual_col3:
                add_clicked = st.button("Find & mark", use_container_width=True)

            if add_clicked:
                if not manual_value.strip():
                    st.warning("Enter a value to search for first.")
                else:
                    newly_added = add_manual_entity(session, manual_value, resolved_manual_type)
                    if newly_added:
                        st.session_state.approval_revision += 1
                        st.success(
                            f"Found and marked {len(newly_added)} spot(s) of "
                            f"{manual_value!r} as {friendly_type(resolved_manual_type)}."
                        )
                        st.rerun()
                    else:
                        st.warning(
                            f"No matches found for {manual_value!r} in this "
                            "document. Try a different value."
                        )

            st.caption(
                "Note: changing the 'What to redact' option above re-checks "
                "every match again, including ones you added yourself — if a "
                "match's type isn't included in the option you switch to, it "
                "will be unapproved along with everything else outside it."
            )

            st.divider()

            # Group by entity_type for easier scanning, but keep each
            # occurrence individually actionable.
            entity_types = sorted({e.get("entity_type", "UNKNOWN") for e in entities})
            type_filter = st.multiselect(
                "Filter by type", entity_types, default=entity_types, format_func=friendly_type,
            )

            filtered = [e for e in entities if e.get("entity_type", "UNKNOWN") in type_filter]

            # Flag ambiguous labels for the user rather than presenting the
            # type as certain fact. NG_NUBAN (bank account) and NG_NHIS
            # (health insurance) both come out as a bare 10-digit number, so
            # the detector genuinely can't always tell them apart.
            AMBIGUOUS_TYPES = {"NG_NUBAN", "NG_NHIS"}

            header_cols = st.columns([0.5, 2, 1.5, 1, 1])
            header_cols[0].markdown("**Approve**")
            header_cols[1].markdown("**Text**")
            header_cols[2].markdown("**Type**")
            header_cols[3].markdown("**Location**")
            header_cols[4].markdown("**Applies to**")

            for entity in filtered:
                entity_id = entity.get("id")
                text_val = entity.get("text", "")
                etype = entity.get("entity_type", "UNKNOWN")
                # page_num only exists on PDF entities (0-indexed); docx has no page concept.
                location = f"page {entity['page_num'] + 1}" if "page_num" in entity else "—"
                is_docx_scope = ext == ".docx"
                scope_label = "Everywhere in the document" if is_docx_scope else "Just this spot"

                row = st.columns([0.5, 2, 1.5, 1, 1])
                approved = row[0].checkbox(
                    "",
                    value=entity.get("approved", True),
                    key=f"approve_{entity_id}_{st.session_state.approval_revision}",
                )
                if approved != entity.get("approved", True):
                    set_approval(session, entity_id, approved)

                row[1].write(text_val)

                type_display = friendly_type(etype)
                if entity.get("source") == "manual":
                    type_display += " ✍️"
                if etype in AMBIGUOUS_TYPES:
                    type_display += " ⚠️"
                row[2].write(type_display)
                if entity.get("source") == "manual":
                    row[2].caption("Manually added by reviewer")
                if etype in AMBIGUOUS_TYPES:
                    row[2].caption(
                        "Label uncertain — Bank Account Numbers (NUBAN) and "
                        "Health Insurance Numbers (NHIS) look the same (both "
                        "are 10 digits), so this label might not be exactly right."
                    )

                row[3].write(str(location))
                row[4].write(scope_label)

# ---------------------------------------------------------------------------
# Tab 3 — Apply redactions & export
# ---------------------------------------------------------------------------
with tab_export:
    if session is None:
        st.info("Upload a document in the **📤 Upload** tab first.")
    elif not entities:
        st.warning("No personal information was found in this document — nothing to export yet.")
    else:
        filename = st.session_state.session_filename

        if st.button("Apply redactions", type="primary"):
            out_name = f"redacted_{filename}"
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

            # DOCX and PDF redactors return differently-named keys for the
            # same concepts (entities_matched vs entities_redacted,
            # paragraphs_redacted vs pages) rather than a shared schema -
            # normalize here.
            matched_count = summary.get("entities_matched", summary.get("entities_redacted"))
            unit_count = summary.get("paragraphs_redacted", summary.get("pages"))
            unit_label = "Paragraphs redacted" if "paragraphs_redacted" in summary else "Pages"

            st.subheader("Redaction summary")
            metric_cols = st.columns(3)
            if matched_count is not None:
                metric_cols[0].metric("Items redacted", matched_count)
            metric_cols[1].metric("Not redacted (left as-is)", rejected_count)
            if unit_count is not None:
                metric_cols[2].metric(unit_label, unit_count)

        if st.session_state.redacted_bytes is not None:
            st.download_button(
                label=f"Download {st.session_state.redacted_filename}",
                data=st.session_state.redacted_bytes,
                file_name=st.session_state.redacted_filename,
                mime=MIME_TYPES.get(ext, "application/octet-stream"),
            )