"""Enterprise guardrails: role-based access control (RBAC), prompt-injection screening and PII redaction.

Defence in depth - the same rules are enforced in several places:
  * input guard      -> blocks obvious prompt-injection / abusive input before any LLM call
  * RAG retriever    -> only searches documents the role is allowed to see
  * SQL validator    -> blocks restricted columns and anything that is not a single read-only SELECT
  * output guard     -> redacts PII (emails, phones, card / government ids) before the answer leaves the system
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_QUESTION_CHARS = 1500


@dataclass(frozen=True)
class AccessPolicy:
    role: str
    description: str
    doc_access: tuple[str, ...]                       # document classifications this role may read
    blocked_columns: dict[str, frozenset[str]] = field(default_factory=dict)   # table -> columns
    mask_contact_pii: bool = True                     # mask emails / phone numbers in answers

    @property
    def all_blocked_column_names(self) -> frozenset[str]:
        names: set[str] = set()
        for cols in self.blocked_columns.values():
            names |= set(cols)
        return frozenset(names)


POLICIES: dict[str, AccessPolicy] = {
    "employee": AccessPolicy(
        role="employee",
        description="Standard employee: public documents; no salary or personal contact data.",
        doc_access=("public",),
        blocked_columns={"employees": frozenset({"salary", "email"})},
        mask_contact_pii=True,
    ),
    "hr_admin": AccessPolicy(
        role="hr_admin",
        description="HR administrator: all documents and employee compensation data.",
        doc_access=("public", "hr_only"),
        blocked_columns={},
        mask_contact_pii=False,
    ),
}


def get_policy(role: str | None) -> AccessPolicy:
    """Unknown roles fall back to the most restrictive policy (fail closed)."""
    return POLICIES.get((role or "employee").lower(), POLICIES["employee"])


# --------------------------------------------------------------------------- input guard
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|system|above)",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system|hidden|initial)\s+(prompt|instructions?|message)",
    r"you\s+are\s+now\s+(dan|in\s+developer\s+mode|jailbroken|unrestricted)",
    r"(act|behave)\s+as\s+(an?\s+)?(unrestricted|jailbroken|uncensored)",
    r"bypass\s+(the\s+)?(guardrails?|safety|security|access\s+control|restrictions?)",
    r"\bdrop\s+table\b|\bdelete\s+from\b|\btruncate\s+table\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def check_input(text: str) -> GuardResult:
    """Screen a user question. Cheap, deterministic and runs before any LLM call."""
    if not text or not text.strip():
        return GuardResult(False, "Empty question.")
    if len(text) > MAX_QUESTION_CHARS:
        return GuardResult(False, f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            return GuardResult(False, "Possible prompt-injection or destructive-command attempt detected.")
    return GuardResult(True)


# --------------------------------------------------------------------------- output guard
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?![\w])")
_AADHAAR_RE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_cards(text: str, found: list[str]) -> str:
    """Mask card numbers - the Luhn checksum avoids false positives on dates and ids."""

    def repl(match: re.Match) -> str:
        if _luhn_ok(match.group(0)):
            found.append("card number")
            return "[REDACTED-CARD]"
        return match.group(0)

    return _CARD_RE.sub(repl, text)


def redact_pii(text: str, policy: AccessPolicy) -> tuple[str, list[str]]:
    """Mask sensitive data. Returns (clean_text, list_of_redaction_kinds)."""
    found: list[str] = []

    def sub(pattern: re.Pattern, repl: str, kind: str, s: str) -> str:
        new, n = pattern.subn(repl, s)
        if n:
            found.append(kind)
        return new

    text = _redact_cards(text, found)            # cards first: a 16-digit card would otherwise partly match the id rule
    text = sub(_AADHAAR_RE, "[REDACTED-ID]", "government id", text)
    text = sub(_PAN_RE, "[REDACTED-ID]", "government id", text)
    if policy.mask_contact_pii:
        text = sub(_EMAIL_RE, "[REDACTED-EMAIL]", "email", text)
        text = sub(_PHONE_RE, "[REDACTED-PHONE]", "phone", text)
    return text, sorted(set(found))
