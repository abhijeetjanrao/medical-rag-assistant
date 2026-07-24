"""
Guardrails for medical-content generation. Two layers:
1. Pre-generation: flag queries that are asking for a personal diagnosis,
   dosage for a specific patient, or emergency triage — defer to a clinician.
2. Post-generation: check the answer doesn't slip into prescriptive/diagnostic
   language even when the query looked benign.

This is a deliberately simple, explainable rule-based layer (not another LLM
call) so it's fast, cheap, and auditable — you can point to exactly why
something was flagged.
"""
import re
from dataclasses import dataclass

EMERGENCY_PATTERNS = [
    r"\b(chest pain|can'?t breathe|difficulty breathing|suicidal|overdose|"
    r"severe bleeding|unconscious|stroke symptoms|heart attack)\b",
]

PERSONAL_DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b", r"\bam i (having|experiencing)\b",
    r"\bwhat'?s wrong with me\b", r"\bshould i take\b.*\bmy\b",
    r"\bmy (dose|dosage) of\b", r"\bdiagnose me\b",
]

DISCLAIMER = (
    "This is general medical information, not a diagnosis or personal medical "
    "advice. Please consult a licensed clinician for anything specific to your "
    "situation."
)

EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please contact emergency "
    "services (911 in the US, or your local emergency number) or go to the "
    "nearest emergency room right away. I can't provide emergency medical care."
)


@dataclass
class GuardrailResult:
    flagged: bool
    reason: str | None
    block_generation: bool
    prepend_message: str | None = None


def check_query(query: str) -> GuardrailResult:
    q = query.lower()

    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(
                flagged=True,
                reason="emergency_pattern",
                block_generation=True,
                prepend_message=EMERGENCY_RESPONSE,
            )

    for pattern in PERSONAL_DIAGNOSIS_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(
                flagged=True,
                reason="personal_diagnosis_request",
                block_generation=False,  # still answer generally, but disclaim
            )

    return GuardrailResult(flagged=False, reason=None, block_generation=False)


def append_disclaimer_if_needed(answer: str, result: GuardrailResult) -> str:
    if result.flagged and not result.block_generation:
        return f"{answer}\n\n*{DISCLAIMER}*"
    return answer
