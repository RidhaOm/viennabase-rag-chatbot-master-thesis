# output_guardrails.py
# Output moderation for the chatbot.
# Uses the OpenAI moderation API to check generated answers for harmful content.
# Fail-open strategy: if the moderation API is unavailable, the answer is accepted.

from __future__ import annotations

try:
    from openai import OpenAI
    _oa_client = OpenAI()
except Exception:
    _oa_client = None


def is_safe_text(text: str) -> bool:
    """
    Check whether a generated answer passes the OpenAI moderation filter.

    Returns False if the content is flagged, True otherwise.
    If the moderation API is unavailable, the function returns True (fail-open)
    to prevent the chatbot from going completely offline due to an API issue.

    Only the first 5000 characters are sent to stay within API limits.
    """
    if not text:
        return True

    try:
        if _oa_client is None:
            return True

        resp = _oa_client.moderations.create(
            model="omni-moderation-latest",
            input=text[:5000],
        )
        return not (getattr(resp, "results", [])[0].flagged)

    except Exception:
        return True
