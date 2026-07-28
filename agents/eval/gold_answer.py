"""Gold-answer calibration helpers.

These helpers keep lexical F1 useful without letting it dominate cases where
the answer is grounded, relevant, and contains the important numeric/keyword
facts from the reference.
"""
from __future__ import annotations

import re
from collections import Counter

from agents.eval.types import (
    EvalRecord,
    F1_PASS_THRESHOLD,
    RAGAS_FAITHFULNESS_MIN,
    RAGAS_RESPONSE_RELEVANCY_MIN,
)


_TOKEN_RE = re.compile(r"[\w.%,-]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+(?:[,\d]*\d)?(?:\.\d+)?%?")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "about",
}


def calibrated_answer_score(record: EvalRecord) -> float:
    """Answer score for reporting/optimization."""
    score = max(record.f1_score, record.ragas_answer_correctness or 0.0)
    if gold_answer_calibrated_match(record):
        return max(score, F1_PASS_THRESHOLD)
    return score


def gold_answer_calibrated_match(record: EvalRecord) -> bool:
    """Whether low-F1 answer should still count as matching the gold answer."""
    if not record.probe.ground_truth:
        return False
    if record.f1_score >= F1_PASS_THRESHOLD:
        return False
    if record.ragas_answer_correctness is not None and record.ragas_answer_correctness >= F1_PASS_THRESHOLD:
        return False
    if record.recall_at_k < 1:
        return False

    faith = _metric(record.ragas, "faithfulness")
    rel = _metric(record.ragas, "response_relevancy")
    if faith is None or rel is None:
        return False
    if faith < max(0.9, RAGAS_FAITHFULNESS_MIN):
        return False
    if rel < max(0.85, RAGAS_RESPONSE_RELEVANCY_MIN):
        return False

    evidence = gold_answer_overlap(record.probe.ground_truth or "", record.generated_answer or "")
    return evidence["numeric_match"] or evidence["keyword_recall"] >= 0.65


def gold_answer_overlap(reference: str, prediction: str) -> dict:
    """Return numeric and keyword containment signals from reference to answer."""
    ref_numbers = _numbers(reference)
    pred_numbers = _numbers(prediction)
    numeric_recall = 0.0
    numeric_match = False
    if ref_numbers:
        hits = sum(1 for n in ref_numbers if n in pred_numbers)
        numeric_recall = hits / len(ref_numbers)
        numeric_match = numeric_recall >= 0.8

    ref_keywords = _keywords(reference)
    pred_keywords = _keywords(prediction)
    keyword_recall = 0.0
    if ref_keywords:
        overlap = sum((Counter(ref_keywords) & Counter(pred_keywords)).values())
        keyword_recall = overlap / len(ref_keywords)

    return {
        "numeric_recall": numeric_recall,
        "numeric_match": numeric_match,
        "keyword_recall": keyword_recall,
    }


def _metric(values: dict, key: str) -> float | None:
    value = values.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _numbers(text: str) -> set[str]:
    return {_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(text or "")}


def _normalize_number(value: str) -> str:
    value = value.strip().replace(",", "")
    if value.endswith("%"):
        value = value[:-1]
    return value


def _keywords(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        token = raw.strip(".,;:!?()[]{}\"'`~")
        if not token:
            continue
        if _NUMBER_RE.fullmatch(token):
            continue
        low = token.lower()
        if low in _STOPWORDS:
            continue
        if len(low) < 2:
            continue
        out.append(low)
    return out
