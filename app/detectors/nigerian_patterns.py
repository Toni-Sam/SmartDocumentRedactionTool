"""
nigerian_patterns.py
---------------------
Presidio PatternRecognizers for Nigerian-specific PII.

Each recognizer is deliberately scoped to what could actually be verified
against an official source (FRSC, CBN, PenCom, NYSC, CAC, NIMC, NHIS). Where
a format is genuinely ambiguous or collides with another Nigerian ID length,
that's called out in a comment rather than silently guessed at - see the
NUBAN/NHIS note near the bottom.
"""

import json
from pathlib import Path

from presidio_analyzer import PatternRecognizer, Pattern


# ---------------------------------------------------------------------------
# Existing recognizers
# ---------------------------------------------------------------------------

NIN_PATTERN = Pattern("NIN", r"\b[0-9]{11}\b", 0.6)
BVN_PATTERN = Pattern("BVN", r"\b[0-9]{11}\b", 0.6)  # context-disambiguated
PASSPORT_NG = Pattern("NG_PASSPORT", r"\bA[0-9]{8}\b", 0.85)

nin_recognizer = PatternRecognizer(
    supported_entity="NG_NIN",
    patterns=[NIN_PATTERN],
    # National ID Card displays the NIN itself - no separate card-serial format exists,
    # so "national id card" is included here rather than as its own recognizer.
    context=["NIN", "national identification", "national id", "national id card", "eID"]
)

bvn_recognizer = PatternRecognizer(
    supported_entity="NG_BVN",
    patterns=[BVN_PATTERN],
    context=["BVN", "bank verification", "verification number"]
)

passport_recognizer = PatternRecognizer(
    supported_entity="NG_PASSPORT",
    patterns=[PASSPORT_NG],
    context=["passport", "international passport"]
)


# ---------------------------------------------------------------------------
# NG_PHONE - validator-based (see fix note below)
# ---------------------------------------------------------------------------
# FIX (OCR scanned-document testing, typed-scan sample):
# The original pattern - r"\b(\+?234|0)[789][01]\d{8}\b" - required an
# unbroken run of digits with zero separator tolerance. That's not just an
# OCR artifact: Nigerian phone numbers are conventionally WRITTEN with
# separators in the first place (e.g. "0803 123 4567", "+234 803 123 4567",
# "0803-123-4567"), so the old pattern would miss correctly-typed, un-OCR'd
# text too, not only scanned/noisy text. And OCR tokenization can introduce
# its own separators independent of source formatting (Tesseract splits on
# whitespace, so "0803 123 4567" always arrives as 3 separate word tokens
# regardless of how the original document grouped them).
#
# Rather than hand-enumerate every grouping convention in the regex itself
# (fragile - the next document format breaks it again), this uses the same
# strategy as VinRecognizer below: a loose candidate match, then a
# validate_result() check that strips all separators and verifies the
# normalized digit string is a structurally valid Nigerian mobile number.
# This tolerates whatever separator convention (or none) a given document
# uses without needing to special-case each one.

PHONE_NG = Pattern("NG_PHONE", r"\b(?:\+?234|0)[\d\s-]{10,14}\b", 0.6)


class PhoneRecognizer(PatternRecognizer):
    """
    Validates a loosely-matched NG_PHONE candidate by stripping all
    separators (spaces, hyphens) and checking the normalized digit string
    against the real Nigerian mobile number shape: a 10-digit local number
    starting with [789][01], after either a "234"/"+234" or a leading "0"
    trunk prefix has been removed.
    """

    def validate_result(self, pattern_text: str):
        digits = "".join(c for c in pattern_text if c.isdigit())

        if digits.startswith("234"):
            local = digits[3:]
        elif digits.startswith("0"):
            local = digits[1:]
        else:
            return False

        if len(local) != 10:
            return False

        return local[0] in "789" and local[1] in "01"


phone_recognizer = PhoneRecognizer(
    supported_entity="NG_PHONE",
    patterns=[PHONE_NG],
    context=["phone", "mobile", "contact number", "tel"]
)


# ---------------------------------------------------------------------------
# NG_PLATE - direct patch (see fix note below)
# ---------------------------------------------------------------------------
# FIX (OCR scanned-document testing, typed-scan sample):
# The original pattern allowed an optional separator between the letter
# prefix and the digit group (r"[-\s]?" right after the letters), but
# required the trailing 2 letters to follow the 3 digits with ZERO
# separator - r"\d{3}[A-Z]{2}". A plate written as "KJA 456 XY" (space
# before the trailing letters, as printed on the actual plate and as OCR
# tokenizes it) never matched. Added the same optional [-\s]? before the
# trailing letter group, matching the tolerance already given to the first
# separator.
PLATE_NG = Pattern("NG_PLATE", r"\b[A-Z]{2,3}[-\s]?\d{3}[-\s]?[A-Z]{2}\b", 0.8)

plate_recognizer = PatternRecognizer(
    supported_entity="NG_PLATE",
    patterns=[PLATE_NG]
)


# ---------------------------------------------------------------------------
# New recognizers
# ---------------------------------------------------------------------------

# Voter's Identification Number (INEC PVC) - 19-character grouped alphanumeric,
# e.g. "804R DFT9 2617 3628 1038". First draft based on limited real samples -
# revisit if you see VINs formatted differently.
#
# KNOWN ISSUE (fixed below via validate_result, not by tightening the regex):
# Presidio compiles patterns with re.IGNORECASE by default, and the optional
# [\s-]? separators mean this shape - "roughly 17-20 alnum chars, loosely
# chunked into 4s" - is satisfied by ordinary English prose purely by
# coincidence (e.g. "from underrepresented" breaks perfectly into
# from|unde|rrep|rese|nted). The regex can't fix this by itself without
# becoming unreadable, so VinRecognizer below adds a post-match digit-density
# check instead: real VINs are digit-heavy (the sample above is ~84% digits),
# ordinary prose isn't.
VIN_NG = Pattern(
    "NG_VIN",
    r"\b[A-Z0-9]{4}[\s-]?[A-Z0-9]{4}[\s-]?[A-Z0-9]{4}[\s-]?[A-Z0-9]{4}[\s-]?[A-Z0-9]{1,4}\b",
    0.7
)


class VinRecognizer(PatternRecognizer):
    """
    Adds a digit-density check on top of the plain VIN_NG regex match.
    The regex alone only constrains length and character class (grouped
    4-char alnum blocks), which ordinary prose satisfies by coincidence
    often enough to flood results. validate_result() runs after the regex
    matches and rejects anything that isn't digit-dense enough, without
    weakening what the regex itself accepts (so it still tolerates
    formatting variants you haven't seen yet).
    """

    MIN_DIGIT_RATIO = 0.5  # real sample is ~0.84; 0.5 leaves margin for format variants

    def validate_result(self, pattern_text: str):
        alnum_chars = [c for c in pattern_text if c.isalnum()]
        if not alnum_chars:
            return False
        digit_ratio = sum(c.isdigit() for c in alnum_chars) / len(alnum_chars)
        return digit_ratio >= self.MIN_DIGIT_RATIO


vin_recognizer = VinRecognizer(
    supported_entity="NG_VIN",
    patterns=[VIN_NG],
    context=["VIN", "voter", "PVC", "permanent voter card", "voter registration"]
)

# Driver's License (FRSC) - 12 characters: 3 letters, 5 digits, 2 letters, 2 digits.
# Format confirmed via FRSC's own published example (ghi12345bb89).
DRIVERS_LICENSE_NG = Pattern(
    "NG_DRIVERS_LICENSE",
    r"\b[A-Za-z]{3}[0-9]{5}[A-Za-z]{2}[0-9]{2}\b",
    0.75
)

drivers_license_recognizer = PatternRecognizer(
    supported_entity="NG_DRIVERS_LICENSE",
    patterns=[DRIVERS_LICENSE_NG],
    context=["driver's license", "driving license", "FRSC"]
)

# Tax Identification Number - FIRS hyphenated format only (8 digits-4 digits,
# e.g. 12345678-0001). The plain 10-digit (JTB legacy) and 13-digit (2026 NRS)
# formats are deliberately NOT included here - see the note below.
#
# FIX (OCR scanned-document testing, typed-scan sample):
# Tesseract tokenizes "12345678-0001" as TWO words when the hyphen sits at a
# natural OCR word break ("12345678" and "-0001"), and the word-joining logic
# in scanned_pdf_parser.py inserts a single space between same-line word
# tokens - so the reconstructed text becomes "12345678 -0001" (space before
# the hyphen). The original pattern required the hyphen immediately after
# the 8th digit with no tolerance for that space. Added \s? on both sides of
# the hyphen to absorb it without loosening the pattern enough to match an
# unrelated 12-digit run (the hyphen itself is still mandatory).
TIN_NG = Pattern("NG_TIN", r"\b[0-9]{8}\s?-\s?[0-9]{4}\b", 0.85)

tin_recognizer = PatternRecognizer(
    supported_entity="NG_TIN",
    patterns=[TIN_NG],
    context=["TIN", "tax identification", "taxpayer"]
)

# NUBAN - 10-digit bank account number. Low base score because a bare 10-digit
# string is inherently ambiguous (see note below) - relies on Presidio's
# context-word score boost to disambiguate in practice.
NUBAN_NG = Pattern("NG_NUBAN", r"\b[0-9]{10}\b", 0.4)

nuban_recognizer = PatternRecognizer(
    supported_entity="NG_NUBAN",
    patterns=[NUBAN_NG],
    context=["account number", "NUBAN", "bank account"]
)

# RSA PIN (PenCom) - "PEN" followed by 12 digits. Distinctive prefix, high confidence.
RSA_PIN_NG = Pattern("NG_RSA_PIN", r"\bPEN[0-9]{12}\b", 0.9)

rsa_pin_recognizer = PatternRecognizer(
    supported_entity="NG_RSA_PIN",
    patterns=[RSA_PIN_NG],
    context=["RSA", "pension", "PenCom", "retirement savings"]
)

# NYSC call-up number - NYSC/<institution or state code>/<year>/<serial>,
# e.g. NYSC/STC/2024/000000 or NYSC/OY/BEN/2011/142356.
NYSC_CALLUP_NG = Pattern(
    "NG_NYSC",
    r"\bNYSC/[A-Z]{2,3}(?:/[A-Z]{2,3})?/[0-9]{4}/[0-9]{5,6}\b",
    0.85
)

# NYSC state code, assigned separately at camp - e.g. IM/25A/1023.
# Lower score: 2 letters + 2 digits + 1 letter + 3-4 digits is a fairly
# generic shape, so this leans more heavily on context than NYSC_CALLUP_NG.
NYSC_STATE_CODE_NG = Pattern(
    "NG_NYSC_STATE_CODE",
    r"\b[A-Z]{2}/[0-9]{2}[A-Z]/[0-9]{3,4}\b",
    0.6
)

nysc_recognizer = PatternRecognizer(
    supported_entity="NG_NYSC",
    patterns=[NYSC_CALLUP_NG, NYSC_STATE_CODE_NG],
    context=["NYSC", "call-up", "corps member", "state code", "orientation camp"]
)

# CAC registration number - RC (company), BN (business name), IT (incorporated
# trustees), or LP (limited partnership) prefix + 5-8 digits.
CAC_RC_NG = Pattern("NG_CAC_RC", r"\b(RC|BN|IT|LP)[\s-]?[0-9]{5,8}\b", 0.75)

cac_rc_recognizer = PatternRecognizer(
    supported_entity="NG_CAC_RC",
    patterns=[CAC_RC_NG],
    context=["CAC", "corporate affairs", "registration number", "certificate of incorporation", "RC number"]
)

# NHIS enrollee number - 10 digits, no distinctive prefix or checksum pattern
# publicly documented. Base score kept deliberately very low; this recognizer
# is realistically only useful when "NHIS" appears nearby to trigger Presidio's
# context boost. See the collision note below before relying on this one.
NHIS_NG = Pattern("NG_NHIS", r"\b[0-9]{10}\b", 0.3)

nhis_recognizer = PatternRecognizer(
    supported_entity="NG_NHIS",
    patterns=[NHIS_NG],
    context=["NHIS", "health insurance", "enrollee", "NHIA"]
)


# ---------------------------------------------------------------------------
# NOTE: NUBAN vs NHIS vs legacy TIN - all bare 10-digit numbers
# ---------------------------------------------------------------------------
# NG_NUBAN and NG_NHIS use the exact same regex shape (10 plain digits), and
# it's the same shape as the legacy JTB TIN format (which is why that TIN
# variant isn't implemented at all - see TIN_NG above). Presidio can't tell
# these apart from the digits alone; only the nearby context word does that.
# In a well-labeled document (a table with "NUBAN" or "NHIS" as a column
# header, like your BVN/NIN table) this works fine. In free-flowing prose
# without a nearby label, expect these two to be indistinguishable by type -
# the number will still get redacted, but it may be tagged as the wrong one
# of the two. This is a real limitation, not a bug to chase.


# ---------------------------------------------------------------------------
# NG_NAME_GAZETTEER - Yoruba name gazetteer (deny-list, not regex)
# ---------------------------------------------------------------------------
# PROBLEM THIS SOLVES: MasakhaNER's model (masakhaner_detector.py) infers
# names from sentence context, so an isolated name with no surrounding
# sentence structure (e.g. sitting alone on a signature line, with no
# "Name:" label for local_context_detector.py to trigger on either) can be
# missed by both of those layers entirely. This recognizer adds a third,
# independent check: an exact-match lookup against a real list of names,
# which needs no context at all to fire.
#
# DATA SOURCE: PER-tagged spans mined from MasakhaNER's Yoruba CoNLL splits
# (1.0 + 2.0 combined) via scripts/build_yoruba_name_gazetteer.py. Dataset
# license: CC-BY-NC-4.0 (Adelani et al., 2021/2022) - non-commercial use
# only, fine for this capstone. See that script's docstring for the full
# citation and a note on scope: these are names as they occur in
# Yoruba-language news text, not a curated Yoruba-ethnicity name list, so
# coverage includes some non-Yoruba public figures too. That's treated as
# acceptable extra recall for this project's purpose, not noise.
#
# WHY deny_list INSTEAD OF A CUSTOM REGEX: Presidio's PatternRecognizer
# supports deny_list natively - an exact phrase/token match against a
# list, no regex needed. This is the correct tool for "does this text
# contain one of these known strings", as opposed to "does this text match
# this shape" (which is what every other recognizer above does).
#
# WHY THIS NEEDS NO CHANGES TO merger.py: this recognizer's source is
# "presidio" and its entity_type is "PERSON_NAME", but PERSON_NAME isn't
# in PATTERN_PREFERRED_TYPES - so on any overlap with an actual
# masakhaner or local_context detection, those still win outright via
# their own source-based priority checks, unconditionally, regardless of
# score. This recognizer only ever contributes a NEW detection when
# nothing else caught the name at all - exactly the gap it exists to
# patch - or corroborates/widens an existing same-type span via the
# existing union-widening logic on genuine overlap.
#
# CONFIDENCE SCORE: 0.65. High enough that it isn't routinely discarded
# against Presidio's own default spaCy PERSON detections (which use a
# different entity_type, "PERSON", and never actually compete with this
# one - see the note in merger.py's docstring/comments on that if it's
# ever unified), but deliberately below MasakhaNER's typical confidence
# so it never looks more authoritative than a genuine model inference in
# any tie-break scenario.
YORUBA_NAME_GAZETTEER_PATH = Path(__file__).resolve().parent.parent / "data" / "yoruba_names_gazetteer.json"


def _load_yoruba_name_gazetteer() -> list[str]:
    with open(YORUBA_NAME_GAZETTEER_PATH, encoding="utf-8") as f:
        return json.load(f)


YORUBA_NAME_GAZETTEER = _load_yoruba_name_gazetteer()

yoruba_name_recognizer = PatternRecognizer(
    supported_entity="PERSON_NAME",
    deny_list=YORUBA_NAME_GAZETTEER,
    deny_list_score=0.65,
)


if __name__ == "__main__":
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

    registry = RecognizerRegistry()
    registry.add_recognizer(nin_recognizer)
    registry.add_recognizer(bvn_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(passport_recognizer)
    registry.add_recognizer(plate_recognizer)
    registry.add_recognizer(vin_recognizer)
    registry.add_recognizer(drivers_license_recognizer)
    registry.add_recognizer(tin_recognizer)
    registry.add_recognizer(nuban_recognizer)
    registry.add_recognizer(rsa_pin_recognizer)
    registry.add_recognizer(nysc_recognizer)
    registry.add_recognizer(cac_rc_recognizer)
    registry.add_recognizer(nhis_recognizer)
    registry.add_recognizer(yoruba_name_recognizer)

    analyzer = AnalyzerEngine(registry=registry)

    test_text = (
        "Applicant NIN: 49340714800, BVN: 22334455667, "
        "Phone: 08112500229, Passport: A01234567, Plate: FST317EG, "
        "Driver's License: ghi12345bb89, TIN: 12345678-0001, "
        "NUBAN account number: 0123456789, RSA PIN: PEN123456789012, "
        "NYSC call-up number NYSC/STC/2024/000123, State code: IM/25A/1023, "
        "CAC registration number RC 1234567, NHIS enrollee number: 0987654321, "
        "Voter's Identification Number (VIN): 804R DFT9 2617 3628 1038"
    )

    print("--- True positive check ---")
    results = analyzer.analyze(text=test_text, language="en")
    for r in sorted(results, key=lambda x: x.start):
        print(f"{r.entity_type}: '{test_text[r.start:r.end]}' (score: {r.score:.2f})")

    # Regression test for the false-positive flood: ordinary prose that
    # happens to be dense with 4-letter-ish word chunks and short spacing,
    # similar to what triggered the original NG_VIN bug against real PDF text.
    prose_text = (
        "Existing solutions for automated redaction risk exposing sensitive "
        "personal information, particularly for underrepresented groups in "
        "Nigeria, which makes reducing regional bias an increasingly critical "
        "goal when working with specific datasets and specific identifiers, "
        "even without reliable internet connectivity."
    )

    print("\n--- False-positive regression check (expect NO NG_VIN hits) ---")
    prose_results = [
        r for r in analyzer.analyze(text=prose_text, language="en", entities=["NG_VIN"])
    ]
    if not prose_results:
        print("OK: no false NG_VIN matches in plain prose.")
    else:
        for r in prose_results:
            print(f"UNEXPECTED MATCH: '{prose_text[r.start:r.end]}' (score: {r.score:.2f})")

    # Regression test for the scanned-document format-tolerance fixes:
    # OCR-reconstructed text with spaced phone numbers, a space-before-hyphen
    # TIN, and a spaced plate number - all of which the ORIGINAL patterns
    # missed against the typed-scan test sample.
    scanned_style_text = (
        "Phone Number: +234 803 245 6721 "
        "Tax Identification No. (TIN): 10293847 -0001 "
        "Vehicle Plate Number: KJA 456 XY "
        "Next of Kin Phone: 0706 112 3345"
    )

    print("\n--- Scanned-format tolerance check (expect 4 hits: 2x NG_PHONE, NG_TIN, NG_PLATE) ---")
    scanned_results = analyzer.analyze(
        text=scanned_style_text,
        language="en",
        entities=["NG_PHONE", "NG_TIN", "NG_PLATE"],
    )
    if not scanned_results:
        print("FAIL: no matches found - fixes did not take effect.")
    else:
        for r in sorted(scanned_results, key=lambda x: x.start):
            print(f"{r.entity_type}: '{scanned_style_text[r.start:r.end]}' (score: {r.score:.2f})")

    # New: isolated-name gazetteer check - a bare name with NO surrounding
    # sentence context and NO "Name:" label, simulating a signature line.
    # This is the exact failure mode the gazetteer recognizer exists to fix.
    isolated_name_text = "Adebayo"

    print("\n--- Isolated-name gazetteer check (expect PERSON_NAME hit, no context needed) ---")
    isolated_results = analyzer.analyze(
        text=isolated_name_text, language="en", entities=["PERSON_NAME"]
    )
    if isolated_results:
        for r in isolated_results:
            print(f"{r.entity_type}: '{isolated_name_text[r.start:r.end]}' (score: {r.score:.2f})")
    else:
        print("FAIL: gazetteer did not fire on an isolated known name.")