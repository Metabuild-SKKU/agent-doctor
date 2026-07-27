"""
agents/optimize/evidence_window.py
gold span을 chunk_size 계산용 evidence window로 확장하는 결정적 도구.

[읽는 것]  Document.content, gold span(doc_id/start/end), evidence_window_policy
[쓰는 것]  state를 수정하지 않고 evidence window dict 목록만 반환

정답 문자열 자체는 매우 짧을 수 있으므로 chunk_size의 근거로 바로 쓰지 않는다.
같은 원문 좌표계에서 정답이 속한 문장·문단·표 행·리스트 항목과 가까운 문맥을
포함하는 최소 연속 구간을 만든다. 멀티홉의 서로 떨어진 span은 합치지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from core.schema import Document


_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\(\d+\)\s+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# 오래된 저장 상태나 최소 단위 테스트처럼 정책 필드가 없는 입력의 하위호환 기본값.
# 정상 파이프라인에서는 core/state.py의 index_config 값이 이 값을 덮어쓴다.
DEFAULT_EVIDENCE_WINDOW_POLICY: dict[str, int] = {
    "min_chars": 100,
    "max_chars": 1000,
    "heading_max_distance": 200,
    "adjacent_context_blocks": 1,
}


@dataclass(frozen=True)
class _Range:
    """원문 내 반열린 문자 구간."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def build_evidence_windows(
    documents: list[Document],
    gold_spans: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """유효한 gold span마다 구조 기반 evidence window를 하나 만든다."""

    resolved = _validate_policy({
        **DEFAULT_EVIDENCE_WINDOW_POLICY,
        **(policy if isinstance(policy, dict) else {}),
    })
    by_id = {document.doc_id: document for document in documents}
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for span in gold_spans:
        if not isinstance(span, dict):
            continue
        doc_id = span.get("doc_id")
        start = span.get("start")
        end = span.get("end")
        document = by_id.get(doc_id)
        if not _valid_span(document, start, end):
            continue

        window, kind = _window_for_span(
            document.content,
            _Range(int(start), int(end)),
            resolved,
        )
        identity = (str(doc_id), window.start, window.end)
        if identity in seen:
            continue
        seen.add(identity)
        windows.append(
            {
                "doc_id": str(doc_id),
                "start": window.start,
                "end": window.end,
                "length": window.length,
                "kind": kind,
                "gold_start": int(start),
                "gold_end": int(end),
                "source": "structural_evidence_window",
            }
        )
    return windows


def _validate_policy(policy: dict[str, Any]) -> dict[str, int]:
    """상태에서 받은 evidence window 정책을 검증하고 정수 설정으로 정규화한다."""

    if not isinstance(policy, dict):
        raise ValueError("evidence_window_policy는 dict여야 합니다.")
    keys = (
        "min_chars",
        "max_chars",
        "heading_max_distance",
        "adjacent_context_blocks",
    )
    values: dict[str, int] = {}
    for key in keys:
        value = policy.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"evidence_window_policy.{key}가 유효하지 않습니다.")
        values[key] = value
    if values["min_chars"] <= 0:
        raise ValueError("evidence_window_policy.min_chars는 1 이상이어야 합니다.")
    if values["max_chars"] < values["min_chars"]:
        raise ValueError("evidence window 최대 길이는 최소 길이 이상이어야 합니다.")
    return values


def _valid_span(
    document: Document | None,
    start: Any,
    end: Any,
) -> bool:
    """span이 원문 좌표계에서 안전한지 확인한다."""

    return bool(
        document is not None
        and not isinstance(start, bool)
        and not isinstance(end, bool)
        and isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(document.content)
    )


def _window_for_span(
    content: str,
    gold: _Range,
    policy: dict[str, int],
) -> tuple[_Range, str]:
    """span이 속한 구조를 판별해 최소 의미 문맥으로 확장한다."""

    lines = _line_ranges(content)
    line_index = _containing_index(lines, gold)
    line = lines[line_index] if line_index is not None else gold
    line_text = content[line.start:line.end].strip()

    if _is_table_line(line_text):
        base = _table_window(content, lines, line_index, policy["max_chars"])
        kind = "table"
    elif _LIST_RE.match(line_text):
        base = _list_window(content, lines, line_index)
        kind = "list"
    else:
        base = _prose_window(content, gold, policy["adjacent_context_blocks"])
        kind = "prose"

    base = _include_nearby_heading(
        content,
        lines,
        base,
        policy["heading_max_distance"],
        policy["max_chars"],
    )
    expanded = _expand_to_minimum(
        content,
        lines,
        base,
        gold,
        policy["min_chars"],
        policy["max_chars"],
    )
    return _clamp_around_gold(
        content,
        expanded,
        gold,
        policy["max_chars"],
    ), kind


def _line_ranges(content: str) -> list[_Range]:
    """개행을 포함하는 원문 행 좌표를 만든다."""

    ranges = [
        _Range(match.start(), match.end())
        for match in re.finditer(r".*(?:\n|$)", content)
        if match.end() > match.start()
    ]
    return ranges or [_Range(0, len(content))]


def _paragraph_range(content: str, position: int) -> _Range:
    """빈 줄 경계로 position이 속한 문단을 찾는다."""

    before = content[:position]
    previous = list(re.finditer(r"\n\s*\n", before))
    start = previous[-1].end() if previous else 0
    following = re.search(r"\n\s*\n", content[position:])
    end = position + following.start() if following else len(content)
    return _trim_whitespace(content, _Range(start, end))


def _sentence_ranges(content: str, paragraph: _Range) -> list[_Range]:
    """문단을 문장부호와 행 경계 기준의 문장 좌표로 나눈다."""

    text = content[paragraph.start:paragraph.end]
    boundaries = [0]
    for match in _SENTENCE_END_RE.finditer(text):
        boundaries.append(match.end())
    boundaries.extend(match.end() for match in re.finditer(r"\n+", text))
    boundaries.append(len(text))
    ordered = sorted(set(boundaries))
    ranges: list[_Range] = []
    for left, right in zip(ordered, ordered[1:]):
        candidate = _trim_whitespace(
            content,
            _Range(paragraph.start + left, paragraph.start + right),
        )
        if candidate.length > 0:
            ranges.append(candidate)
    return ranges or [paragraph]


def _prose_window(content: str, gold: _Range, adjacent: int) -> _Range:
    """gold 문장과 가까운 앞뒤 문장을 evidence로 묶는다."""

    paragraph = _paragraph_range(content, gold.start)
    sentences = _sentence_ranges(content, paragraph)
    index = _containing_index(sentences, gold)
    if index is None:
        return paragraph
    left = max(0, index - adjacent)
    right = min(len(sentences) - 1, index + adjacent)
    return _Range(sentences[left].start, sentences[right].end)


def _table_window(
    content: str,
    lines: list[_Range],
    index: int | None,
    max_chars: int,
) -> _Range:
    """표가 작으면 전체 표, 크면 gold 행 주변의 연속 행만 사용한다."""

    if index is None:
        return _Range(0, min(len(content), max_chars))
    left = index
    right = index
    while left > 0 and _is_table_line(content[lines[left - 1].start:lines[left - 1].end]):
        left -= 1
    while right + 1 < len(lines) and _is_table_line(
        content[lines[right + 1].start:lines[right + 1].end]
    ):
        right += 1
    whole = _Range(lines[left].start, lines[right].end)
    if whole.length <= max_chars:
        return whole

    # 큰 표 전체가 chunk_size를 끌어올리지 않도록 대상 행 앞뒤부터 제한적으로 확장한다.
    selected_left = index
    selected_right = index
    while True:
        candidates: list[tuple[int, int]] = []
        if selected_left > left:
            candidates.append((selected_left - 1, selected_right))
        if selected_right < right:
            candidates.append((selected_left, selected_right + 1))
        fitting = [
            pair
            for pair in candidates
            if lines[pair[1]].end - lines[pair[0]].start <= max_chars
        ]
        if not fitting:
            break
        selected_left, selected_right = min(
            fitting,
            key=lambda pair: lines[pair[1]].end - lines[pair[0]].start,
        )
    return _Range(lines[selected_left].start, lines[selected_right].end)


def _list_window(
    content: str,
    lines: list[_Range],
    index: int | None,
) -> _Range:
    """대상 리스트 항목과 바로 앞의 도입 행을 묶는다."""

    if index is None:
        return _Range(0, len(content))
    start = lines[index].start
    end = lines[index].end
    if index > 0:
        previous_text = content[lines[index - 1].start:lines[index - 1].end].strip()
        if previous_text and not _LIST_RE.match(previous_text):
            start = lines[index - 1].start
    # 줄바꿈된 리스트 항목의 연속 본문 한 행을 포함한다.
    if index + 1 < len(lines):
        next_text = content[lines[index + 1].start:lines[index + 1].end].strip()
        if next_text and not _LIST_RE.match(next_text):
            end = lines[index + 1].end
    return _Range(start, end)


def _include_nearby_heading(
    content: str,
    lines: list[_Range],
    window: _Range,
    max_distance: int,
    max_chars: int,
) -> _Range:
    """가까운 Markdown 제목이 있으면 연속 evidence에 포함한다."""

    candidate: _Range | None = None
    for line in lines:
        if line.end > window.start:
            break
        text = content[line.start:line.end].strip()
        if _HEADING_RE.match(text):
            candidate = line
    if (
        candidate is not None
        and window.start - candidate.end <= max_distance
        and window.end - candidate.start <= max_chars
    ):
        return _Range(candidate.start, window.end)
    return window


def _expand_to_minimum(
    content: str,
    lines: list[_Range],
    window: _Range,
    gold: _Range,
    min_chars: int,
    max_chars: int,
) -> _Range:
    """짧은 evidence에 인접 행을 붙여 최소 문맥량을 확보한다."""

    if window.length >= min_chars:
        return window
    left = _first_overlapping_index(lines, window.start)
    right = _last_overlapping_index(lines, window.end)
    if left is None or right is None:
        return _clamp_around_gold(content, window, gold, min(max_chars, min_chars))

    while lines[right].end - lines[left].start < min_chars:
        options: list[tuple[int, int]] = []
        if left > 0:
            options.append((left - 1, right))
        if right + 1 < len(lines):
            options.append((left, right + 1))
        options = [
            pair
            for pair in options
            if lines[pair[1]].end - lines[pair[0]].start <= max_chars
        ]
        if not options:
            break
        left, right = min(
            options,
            key=lambda pair: lines[pair[1]].end - lines[pair[0]].start,
        )
    return _trim_whitespace(content, _Range(lines[left].start, lines[right].end))


def _clamp_around_gold(
    content: str,
    window: _Range,
    gold: _Range,
    max_chars: int,
) -> _Range:
    """최대 길이를 넘는 window를 gold를 보존하며 자른다."""

    if window.length <= max_chars:
        return _trim_whitespace(content, window)
    if gold.length >= max_chars:
        return gold
    spare = max_chars - gold.length
    left_room = min(gold.start - window.start, spare // 2)
    right_room = min(window.end - gold.end, spare - left_room)
    remaining = spare - left_room - right_room
    if remaining:
        extra_left = min(gold.start - window.start - left_room, remaining)
        left_room += extra_left
        remaining -= extra_left
        right_room += min(window.end - gold.end - right_room, remaining)
    return _trim_whitespace(
        content,
        _Range(gold.start - left_room, gold.end + right_room),
    )


def _trim_whitespace(content: str, value: _Range) -> _Range:
    """좌표를 보존하면서 구간 양끝 공백만 제거한다."""

    start = value.start
    end = value.end
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return _Range(start, end)


def _containing_index(ranges: list[_Range], target: _Range) -> int | None:
    for index, value in enumerate(ranges):
        if value.start <= target.start and value.end >= target.end:
            return index
    return None


def _first_overlapping_index(ranges: list[_Range], position: int) -> int | None:
    for index, value in enumerate(ranges):
        if value.end > position:
            return index
    return None


def _last_overlapping_index(ranges: list[_Range], position: int) -> int | None:
    for index in range(len(ranges) - 1, -1, -1):
        if ranges[index].start < position:
            return index
    return None


def _is_table_line(text: str) -> bool:
    """Markdown 표 행으로 볼 수 있는 최소 pipe 수를 검사한다."""

    return text.count("|") >= 2
