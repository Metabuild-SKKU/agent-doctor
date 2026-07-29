"""Document type detection used by Ingest and retrieval-time routing."""
from __future__ import annotations

import re
from typing import Any

_PI_FRACTION_RE = re.compile(r"(π|pi)\s*/\s*([0-9]+)", re.IGNORECASE)
_KOREAN_FRAC_RE = re.compile(
    r"([0-9]+)\s*분의\s*(파이|π|pi|[A-Za-z가-힣0-9]+)",
    re.IGNORECASE,
)
_MATH_DOCUMENT_TYPES = {"math", "formula", "stem"}


def has_math_signal(value: str) -> bool:
    """Return True only for low-ambiguity math notation."""
    value = value or ""
    if any(marker in value for marker in ("\\frac", "^", "π", "∑", "∞")):
        return True
    if _KOREAN_FRAC_RE.search(value):
        return True
    if _PI_FRACTION_RE.search(value):
        return True
    if re.search(r"(\\lim\b|\blim\s*[_({])", value):
        return True
    if re.search(r"\b[a-zA-Z]\s*[=<>]", value):
        return True
    return any(_looks_like_numeric_fraction(match) for match in re.findall(r"\d+\s*/\s*\d+", value))


def _looks_like_numeric_fraction(value: str) -> bool:
    left, right = [part.strip() for part in value.split("/", 1)]
    if len(left) == 4 or len(right) == 4:
        return False
    if left == "0" or right == "0":
        return False
    return len(left) <= 3 and len(right) <= 3


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
    """Return an explicitly declared document type, otherwise general."""
    explicit = document_type_from_metadata(metadata)
    if explicit:
        return explicit
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
