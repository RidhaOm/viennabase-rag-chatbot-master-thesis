# input_guardrails.py
# Input validation for the chatbot:
#   - Unicode normalization and whitespace cleanup
#   - Basic length and format checks (no URLs or email addresses)
#   - Heuristic detection of prompt injection / policy-bypass attempts (DE/EN)
#   - Rejection of shell-style token chains

from __future__ import annotations
import re
import unicodedata

MAX_QUESTION_CHARS = 800


def normalize(text: str) -> str:
    """
    Normalize free-text input for consistent pattern matching:
    strips diacritics, lowercases, trims, and collapses whitespace.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# Heuristic patterns for prompt injection and policy-bypass attempts (DE/EN).
# Intentionally broad to catch variations; false positives are possible.
INJECTION_PATTERNS = [
    # English
    r"\bignore\b.*\b(previous|earlier)\b.*\b(instruction|message|rule)s?\b",
    r"\bforget\b.*\b(previous|earlier)\b.*\b(instruction|message|rule)s?\b",
    r"\boverride\b.*\b(rule|policy|instruction)s?\b",
    r"\bbypass\b.*\b(safety|guard|filter|policy|rule)s?\b",
    r"\breset\b.*\b(instruction|context|chat)\b",
    r"\b(do\s*anything\s*now|dan\s*prompt)\b",
    r"\byou\s+are\s+chatgpt\b",
    r"\b(system\s*prompt|developer\s*message|dev\s*mode)\b",
    # German
    r"\bignoriere\b.*\b(vorherig|frueher|früher)\b.*\b(anweisung|nachricht|regel)n?\b",
    r"\bvergiss\b.*\b(vorherig|frueher|früher)\b.*\b(anweisung|nachricht|regel)n?\b",
    r"\boverride\b|\bereg.*uebergehen\b|\buebergehen\b|\büberg[eä]n\b",
    r"\bumgehe\b.*\b(sicherheit|schutz|filter|richtlinie|regel)n?\b",
    r"\b(setze|reset)\b.*\b(anweisung|kontext|chat)\b.*\bzurueck\b|\bzurück\b",
    r"\bhandle\s+als\b|\btu\s+so\s+als\b|\brolle:\s*system\b",
    r"\bdu\s+bist\s+chatgpt\b",
    r"\b(system\s*prompt|entwickler\s*nachricht|entwicklernachricht)\b",
]

# Token patterns commonly associated with shell injection or code execution.
BANNED_TOKENS = [r"\s&&\s", r"\|\|", r";", r"`", r"\$\(", r"\|"]


def validate_query(q: str) -> str:
    """
    Validate and lightly normalize a user query.

    Raises ValueError for empty input, inputs that are too long, inputs
    containing URLs or email addresses, inputs matching injection patterns,
    or inputs containing banned shell-style tokens.

    Returns the original text with only whitespace normalization applied.
    """
    if q is None:
        raise ValueError("Empty input.")

    raw = q
    qn = normalize(q)

    if not qn:
        raise ValueError("Empty input.")
    if len(qn) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question too long (>{MAX_QUESTION_CHARS} characters).")

    if re.search(r"https?://|www\.", qn) or re.search(r"\b\S+@\S+\.\S+\b", qn):
        raise ValueError("Please do not include links or email addresses.")

    for pat in INJECTION_PATTERNS:
        if re.search(pat, qn, flags=re.IGNORECASE):
            raise ValueError("Input contains disallowed patterns.")

    for tok in BANNED_TOKENS:
        if re.search(tok, q):
            raise ValueError("Input contains disallowed patterns.")

    return re.sub(r"\s+", " ", raw).strip()
