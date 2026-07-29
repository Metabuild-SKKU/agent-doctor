"""Document type detection used by Ingest and retrieval-time routing."""
from __future__ import annotations

import re
from typing import Any

_PI_FRACTION_RE = re.compile(r"(π|pi)\s*/\s*([0-9]+)", re.IGNORECASE)
_MATH_SIGNAL_TERMS = (
    "분의",
    "수식",
    "수열",
    "급수",
    "공비",
    "극한",
    "접선",
    "기울기",
    "적분",
    "미분",
)
_FINANCE_SIGNAL_TERMS = (
    "자산총계",
    "부채총계",
    "손익계산서",
    "재무상태표",
    "당분기말",
    "전기말",
    "투자보고서",
)
_POLICY_SIGNAL_TERMS = (
    "제",
    "조",
    "시행",
    "규정",
    "약관",
    "별표",
)
_MATH_DOCUMENT_TYPES = {"math", "formula", "stem"}


def has_math_signal(value: str) -> bool:
    """Return True only for math-like notation or Korean math terms."""
    value = value or ""
    if any(marker in value for marker in ("\\frac", "^", "π", "∑", "∞")):
        return True
    if any(term in value for term in _MATH_SIGNAL_TERMS):
        return True
    if _PI_FRACTION_RE.search(value):
        return True
    if re.search(r"(\\lim\b|\blim\s*[_({])", value):
        return True
    return bool(re.search(r"\b[a-zA-Z]\s*[=<>]|[0-9]+\s*/\s*[0-9]+", value))


def document_type_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    metadata = metadata or {}
    raw = (
        metadata.get("document_type")
        or metadata.get("retrieval_profile")
        or metadata.get("domain")
    )
    if not raw:
        return None

    value = str(raw).strip().lower()
    if value in _MATH_DOCUMENT_TYPES or "math" in value or "formula" in value:
        return "math"
    if value in {"finance", "finance_table", "financial"}:
        return "finance_table"
    if value in {"policy", "legal", "rule"}:
        return "policy"
    return value or None


def detect_document_type(text: str, metadata: dict[str, Any] | None = None) -> str:
    """Return a conservative retrieval profile for type-specific preprocessing."""
    explicit = document_type_from_metadata(metadata)
    if explicit:
        return explicit

    value = text or ""
    if not value.strip():
        return "general"

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    sample = "\n".join(lines[:80])
    math_hits = sum(1 for line in lines[:80] if has_math_signal(line))
    finance_hits = sum(1 for term in _FINANCE_SIGNAL_TERMS if term in sample)
    policy_hits = len(re.findall(r"제\s*\d+\s*조", sample))
    policy_hits += sum(1 for term in _POLICY_SIGNAL_TERMS if term in sample)

    if math_hits >= 3 or (
        math_hits >= 1
        and re.search(r"\b\d+\s*번\b|SET\s*\d+", sample, re.IGNORECASE)
    ):
        return "math"
    if finance_hits >= 2 or re.search(r"\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|", sample):
        return "finance_table"
    if policy_hits >= 3:
        return "policy"
    return "general"


def is_math_document(metadata: dict[str, Any] | None) -> bool:
    return document_type_from_metadata(metadata) == "math"


def annotate_document_metadata(
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach document type metadata during ingest without overwriting explicit input."""
    annotated = dict(metadata or {})
    document_type = detect_document_type(text, annotated)
    annotated.setdefault("document_type", document_type)
    if document_type == "math":
        annotated.setdefault("retrieval_profile", "math_formula")
    return annotated
