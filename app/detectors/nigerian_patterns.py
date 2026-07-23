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

from presidio_analyzer import PatternRecognizer, Pattern


# ---------------------------------------------------------------------------
# Existing recognizers
# ---------------------------------------------------------------------------

NIN_PATTERN = Pattern("NIN", r"\b[0-9]{11}\b", 0.6)
BVN_PATTERN = Pattern("BVN", r"\b[0-9]{11}\b", 0.6)  # context-disambiguated
PASSPORT_NG = Pattern("NG_PASSPORT", r"\bA[0-9]{8}\b", 0.85)
PHONE_NG = Pattern("NG_PHONE", r"\b(\+?234|0)[789][01]\d{8}\b", 0.9)
PLATE_NG = Pattern("NG_PLATE", r"\b[A-Z]{2,3}[-\s]?\d{3}[A-Z]{2}\b", 0.8)

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

phone_recognizer = PatternRecognizer(
    supported_entity="NG_PHONE",
    patterns=[PHONE_NG]
)

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
    context=["driver's license", "driving licence", "FRSC", "DL number"]
)

# Tax Identification Number - FIRS hyphenated format only (8 digits-4 digits,
# e.g. 12345678-0001). The plain 10-digit (JTB legacy) and 13-digit (2026 NRS)
# formats are deliberately NOT included here - see the note below.
TIN_NG = Pattern("NG_TIN", r"\b[0-9]{8}-[0-9]{4}\b", 0.85)

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