"""
app/redaction_policies.py

Named subsets of entity types a user can choose to redact, e.g.
"Financial IDs only" instead of everything the detectors found.

WHERE THIS SITS IN THE PIPELINE
--------------------------------
Filtering happens AFTER detection/merging. detect_all() in merger.py
still runs every detector and returns everything it finds, exactly as
it does today - no detector is ever skipped, and merger.py's overlap/
priority-resolution logic is completely untouched by this module.

A policy only changes which of those already-detected entities are
pre-approved for redaction: it sets DetectionSession.entities[i]["approved"]
via dispatcher.set_approval_by_types(). A human reviewer can still flip
any individual entity afterward via dispatcher.set_approval(), exactly
as with any other approval change - a policy is a bulk starting point,
not a restriction on what can be reviewed or redacted.

WHY THIS SHAPE (decided with Oluwatoni before implementation)
----------------------------------------------------------------
- Filter after merge, not before/during detection - lower risk, and
  every detector's output stays available for review even under a
  narrow policy (e.g. switching policy later doesn't require re-detecting).
- Fixed named presets PLUS a "Custom" option (arbitrary multiselect of
  entity types) - the GUI exposes both via resolve_policy_types() below.
"""

from app.detectors.merger import PATTERN_PREFERRED_TYPES, CONTEXT_PREFERRED_TYPES


# All entity types this pipeline can ever produce, derived directly from
# merger.py's own priority-rule constants rather than a separate hardcoded
# list - this can't silently drift out of sync with merger.py as new types
# are added (e.g. if Igbo/Hausa name gazetteers or new NG_* recognizers are
# registered there later, they automatically appear here too, and in the
# "Custom" multiselect, with no change needed in this file).
ALL_ENTITY_TYPES = frozenset(PATTERN_PREFERRED_TYPES | CONTEXT_PREFERRED_TYPES)


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# Each preset is a plain set of entity_type strings (see merger.py's
# PATTERN_PREFERRED_TYPES / CONTEXT_PREFERRED_TYPES for the full type
# vocabulary). These groupings are judgment calls - review and adjust
# freely. This is the ONLY place that needs to change to redefine a preset.

FINANCIAL_IDS_ONLY = {
    "NG_BVN",          # Bank Verification Number
    "NG_NUBAN",        # bank account number
    "NG_TIN",          # Tax Identification Number
    "NG_RSA_PIN",      # Retirement Savings Account PIN (pension)
    "ACCOUNT_NUMBER",  # generic account number (Presidio default recognizer)
    "NG_CAC_RC",       # CAC business registration number - included here
                       # as a financial/business identifier; pull it out
                       # into its own type of preset if that doesn't match
                       # your definition of "financial" for the writeup.
}

NAMES_AND_LOCATION_ONLY = {
    "PERSON_NAME",
    "LGA",
    "STATE_OF_ORIGIN",
    "ADDRESS",
    # ETHNIC_GROUP deliberately excluded - "location" reads as
    # place-of-residence/origin, not ethnicity. Add it in if your
    # definition of this preset should cover it too.
}

FULL_NDPA_PROFILE = set(ALL_ENTITY_TYPES)  # every known type - no restriction


PRESET_POLICIES = {
    "Financial IDs only": FINANCIAL_IDS_ONLY,
    "Names + Location only": NAMES_AND_LOCATION_ONLY,
    "Full NDPA profile": FULL_NDPA_PROFILE,
}

DEFAULT_POLICY_NAME = "Full NDPA profile"


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_policy_types(policy_name: str, custom_types: set | None = None) -> set:
    """
    Returns the set of entity_type strings a given policy allows.

    policy_name: one of PRESET_POLICIES' keys, or "Custom".
    custom_types: required when policy_name == "Custom" - the user's own
        selection of entity types (e.g. from a GUI multiselect). Any
        value not in ALL_ENTITY_TYPES is silently dropped rather than
        raising, since a GUI multiselect is already constrained to valid
        types - this just guards against a stale/unexpected value.

    Raises ValueError for any policy_name that's neither a known preset
    nor "Custom", since that's a programming error (a typo in a caller),
    not a normal runtime condition to guard silently against.
    """
    if policy_name == "Custom":
        if not custom_types:
            return set()
        return set(custom_types) & ALL_ENTITY_TYPES

    if policy_name not in PRESET_POLICIES:
        raise ValueError(
            f"Unknown policy {policy_name!r}. "
            f"Known presets: {sorted(PRESET_POLICIES)}, or 'Custom'."
        )

    return set(PRESET_POLICIES[policy_name])


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"ALL_ENTITY_TYPES ({len(ALL_ENTITY_TYPES)}): {sorted(ALL_ENTITY_TYPES)}\n")

    for name in PRESET_POLICIES:
        types = resolve_policy_types(name)
        print(f"[{name}] -> {sorted(types)}")

    custom = resolve_policy_types("Custom", {"NG_BVN", "NOT_A_REAL_TYPE", "LGA"})
    print(f"\n[Custom] with one invalid type mixed in -> {sorted(custom)}")
    assert custom == {"NG_BVN", "LGA"}, "invalid type should have been dropped"

    empty_custom = resolve_policy_types("Custom", None)
    print(f"[Custom] with no selection -> {sorted(empty_custom)}")
    assert empty_custom == set()

    try:
        resolve_policy_types("Not A Real Policy")
        print("FAIL: expected ValueError for unknown policy name")
    except ValueError as e:
        print(f"\nOK: unknown policy name correctly raised: {e}")