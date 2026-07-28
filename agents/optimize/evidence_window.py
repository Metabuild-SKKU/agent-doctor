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

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re
from typing import Any

from core.schema import Document


_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\(\d+\)\s+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_SENTENCE_END_RE = re.compile(r"""[.!?。！？]+["'”’」』)]*(?=\s|$)""")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)

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


@dataclass(frozen=True)
class _RangeIndex:
    """정렬된 구간을 문자 좌표로 빠르게 찾기 위한 인덱스."""

    ranges: tuple[_Range, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    @classmethod
    def from_ranges(cls, ranges: list[_Range]) -> "_RangeIndex":
        values = tuple(ranges)
        return cls(
            ranges=values,
            starts=tuple(value.start for value in values),
            ends=tuple(value.end for value in values),
        )

    def overlapping_bounds(self, target: _Range) -> tuple[int, int] | None:
        """target과 겹치는 첫/마지막 구간 인덱스를 반환한다."""

        if not self.ranges:
            return None
        left = bisect_right(self.ends, target.start)
        right = bisect_left(self.starts, target.end) - 1
        if left >= len(self.ranges) or right < left:
            return None
        return left, right


@dataclass(frozen=True)
class _LineBlock:
    """행 인덱스 기준의 연속 구조 블록."""

    start: int
    end: int


@dataclass(frozen=True)
class _DocumentLayout:
    """문서마다 한 번만 계산해 모든 gold span이 공유하는 구조 정보."""

    content: str
    lines: _RangeIndex
    line_texts: tuple[str, ...]
    headings: tuple[_Range, ...]
    heading_ends: tuple[int, ...]
    table_blocks: tuple[_LineBlock | None, ...]
    paragraphs: _RangeIndex
    sentences_by_paragraph: tuple[_RangeIndex, ...]


def resolve_evidence_window_policy(
    policy: dict[str, Any] | None,
) -> dict[str, int]:
    """부분 정책에 기본값을 합치고 잘못된 입력은 명시적으로 거부한다."""

    if policy is None:
        overrides: dict[str, Any] = {}
    elif isinstance(policy, dict):
        overrides = policy
    else:
        raise ValueError("evidence_window_policy는 dict여야 합니다.")
    return _validate_policy({
        **DEFAULT_EVIDENCE_WINDOW_POLICY,
        **overrides,
    })


def build_evidence_windows(
    documents: list[Document],
    gold_spans: list[dict[str, Any]],
    policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """유효한 gold span마다 gold를 포함하는 구조 기반 window를 하나 만든다."""

    resolved = resolve_evidence_window_policy(policy)
    by_id = {document.doc_id: document for document in documents}
    layouts: dict[str, _DocumentLayout] = {}
    windows: list[dict[str, Any]] = []

    for sample_index, span in enumerate(gold_spans):
        if not isinstance(span, dict):
            continue
        doc_id = span.get("doc_id")
        start = span.get("start")
        end = span.get("end")
        document = by_id.get(doc_id)
        if not _valid_span(document, start, end):
            continue

        resolved_doc_id = str(doc_id)
        layout = layouts.get(resolved_doc_id)
        if layout is None:
            layout = _build_layout(document.content)
            layouts[resolved_doc_id] = layout
        gold = _Range(int(start), int(end))
        window, kind = _window_for_span(layout, gold, resolved)
        windows.append(
            {
                "doc_id": resolved_doc_id,
                "start": window.start,
                "end": window.end,
                "length": window.length,
                "kind": kind,
                "gold_start": gold.start,
                "gold_end": gold.end,
                "sample_index": sample_index,
                "source": "structural_evidence_window",
            }
        )
    return windows


def _validate_policy(policy: dict[str, Any]) -> dict[str, int]:
    """상태에서 받은 evidence window 정책을 검증하고 정수 설정으로 정규화한다."""

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


def _build_layout(content: str) -> _DocumentLayout:
    """행·문단·문장·표 구조를 문서당 한 번 분석한다."""

    line_values = _line_ranges(content)
    lines = _RangeIndex.from_ranges(line_values)
    line_texts = tuple(
        content[value.start:value.end].strip()
        for value in line_values
    )
    headings = tuple(
        value
        for value, text in zip(line_values, line_texts)
        if _HEADING_RE.match(text)
    )
    paragraph_values = _paragraph_ranges(content)
    paragraphs = _RangeIndex.from_ranges(paragraph_values)
    return _DocumentLayout(
        content=content,
        lines=lines,
        line_texts=line_texts,
        headings=headings,
        heading_ends=tuple(value.end for value in headings),
        table_blocks=_table_blocks(line_texts),
        paragraphs=paragraphs,
        sentences_by_paragraph=tuple(
            _RangeIndex.from_ranges(_sentence_ranges(content, paragraph))
            for paragraph in paragraph_values
        ),
    )


def _window_for_span(
    layout: _DocumentLayout,
    gold: _Range,
    policy: dict[str, int],
) -> tuple[_Range, str]:
    """span과 겹치는 모든 행을 기준으로 최소 의미 문맥을 확장한다."""

    line_bounds = layout.lines.overlapping_bounds(gold)
    if line_bounds is None:
        return gold, "prose"
    first_line, last_line = line_bounds
    table_block = layout.table_blocks[first_line]

    if table_block is not None and table_block.end >= last_line:
        base = _table_window(
            layout,
            table_block,
            first_line,
            last_line,
            policy["max_chars"],
        )
        kind = "table"
    elif any(
        _LIST_RE.match(layout.line_texts[index])
        for index in range(first_line, last_line + 1)
    ):
        base = _list_window(layout, first_line, last_line)
        kind = "list"
    else:
        base = _prose_window(
            layout,
            gold,
            policy["adjacent_context_blocks"],
        )
        kind = "prose"

    base = _include_nearby_heading(
        layout,
        base,
        policy["heading_max_distance"],
        policy["max_chars"],
    )
    expanded = _expand_to_minimum(
        layout,
        base,
        gold,
        policy["min_chars"],
        policy["max_chars"],
    )
    window = _clamp_preserving_gold(
        layout.content,
        expanded,
        gold,
        policy["max_chars"],
    )
    return window, kind


def _line_ranges(content: str) -> list[_Range]:
    """개행을 포함하는 원문 행 좌표를 만든다."""

    ranges = [
        _Range(match.start(), match.end())
        for match in re.finditer(r".*(?:\n|$)", content)
        if match.end() > match.start()
    ]
    return ranges or [_Range(0, len(content))]


def _paragraph_ranges(content: str) -> list[_Range]:
    """빈 줄 경계로 문서의 문단 좌표를 한 번에 만든다."""

    ranges: list[_Range] = []
    cursor = 0
    for separator in re.finditer(r"\n[ \t]*\n+", content):
        candidate = _trim_whitespace(
            content,
            _Range(cursor, separator.start()),
        )
        if candidate.length:
            ranges.append(candidate)
        cursor = separator.end()
    candidate = _trim_whitespace(content, _Range(cursor, len(content)))
    if candidate.length:
        ranges.append(candidate)
    return ranges or [_Range(0, len(content))]


def _sentence_ranges(content: str, paragraph: _Range) -> list[_Range]:
    """문단을 한국어 인용부호·전각 문장부호와 행 경계까지 고려해 나눈다."""

    text = content[paragraph.start:paragraph.end]
    boundaries = [0]
    boundaries.extend(match.end() for match in _SENTENCE_END_RE.finditer(text))
    boundaries.extend(match.end() for match in re.finditer(r"\n+", text))
    boundaries.append(len(text))
    ordered = sorted(set(boundaries))
    ranges: list[_Range] = []
    for left, right in zip(ordered, ordered[1:]):
        candidate = _trim_whitespace(
            content,
            _Range(paragraph.start + left, paragraph.start + right),
        )
        if candidate.length:
            ranges.append(candidate)
    return ranges or [paragraph]


def _prose_window(
    layout: _DocumentLayout,
    gold: _Range,
    adjacent: int,
) -> _Range:
    """gold가 걸친 문장 전체와 가까운 앞뒤 문장을 evidence로 묶는다."""

    paragraph_bounds = layout.paragraphs.overlapping_bounds(gold)
    if paragraph_bounds is None:
        return gold
    first_paragraph, last_paragraph = paragraph_bounds
    if first_paragraph != last_paragraph:
        return _Range(
            layout.paragraphs.ranges[first_paragraph].start,
            layout.paragraphs.ranges[last_paragraph].end,
        )

    sentences = layout.sentences_by_paragraph[first_paragraph]
    sentence_bounds = sentences.overlapping_bounds(gold)
    if sentence_bounds is None:
        return layout.paragraphs.ranges[first_paragraph]
    first_sentence, last_sentence = sentence_bounds
    left = max(0, first_sentence - adjacent)
    right = min(len(sentences.ranges) - 1, last_sentence + adjacent)
    return _Range(sentences.ranges[left].start, sentences.ranges[right].end)


def _table_window(
    layout: _DocumentLayout,
    block: _LineBlock,
    first_line: int,
    last_line: int,
    max_chars: int,
) -> _Range:
    """표가 작으면 전체 표, 크면 gold와 겹치는 행 주변을 연속 선택한다."""

    lines = layout.lines.ranges
    whole = _Range(lines[block.start].start, lines[block.end].end)
    if whole.length <= max_chars:
        return whole

    selected_left = first_line
    selected_right = last_line
    while True:
        candidates: list[tuple[int, int]] = []
        if selected_left > block.start:
            candidates.append((selected_left - 1, selected_right))
        if selected_right < block.end:
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
    layout: _DocumentLayout,
    first_line: int,
    last_line: int,
) -> _Range:
    """다중 행 gold가 걸친 리스트 항목과 앞의 도입 문맥을 묶는다."""

    lines = layout.lines.ranges
    left = first_line
    right = last_line

    # 앞선 리스트 항목까지 거슬러 올라가야 도입 문장과 연속 구간으로 묶을 수 있다.
    while left > 0 and _LIST_RE.match(layout.line_texts[left - 1]):
        left -= 1
    if left > 0:
        previous = layout.line_texts[left - 1]
        if (
            previous
            and not _LIST_RE.match(previous)
            and not _HEADING_RE.match(previous)
            and layout.table_blocks[left - 1] is None
        ):
            left -= 1

    # 줄바꿈된 마지막 항목의 연속 본문 한 행을 포함한다.
    if right + 1 < len(lines):
        following = layout.line_texts[right + 1]
        if (
            following
            and not _LIST_RE.match(following)
            and not _HEADING_RE.match(following)
            and layout.table_blocks[right + 1] is None
        ):
            right += 1
    return _Range(lines[left].start, lines[right].end)


def _include_nearby_heading(
    layout: _DocumentLayout,
    window: _Range,
    max_distance: int,
    max_chars: int,
) -> _Range:
    """가까운 Markdown 제목이 있으면 연속 evidence에 포함한다."""

    heading_index = bisect_right(layout.heading_ends, window.start) - 1
    if heading_index < 0:
        return window
    candidate = layout.headings[heading_index]
    if (
        window.start - candidate.end <= max_distance
        and window.end - candidate.start <= max_chars
    ):
        return _Range(candidate.start, window.end)
    return window


def _expand_to_minimum(
    layout: _DocumentLayout,
    window: _Range,
    gold: _Range,
    min_chars: int,
    max_chars: int,
) -> _Range:
    """짧은 evidence에 인접 행을 붙여 최소 문맥량을 확보한다."""

    if window.length >= min_chars:
        return window
    line_bounds = layout.lines.overlapping_bounds(window)
    if line_bounds is None:
        return _clamp_preserving_gold(
            layout.content,
            window,
            gold,
            min(max_chars, min_chars),
        )
    left, right = line_bounds
    lines = layout.lines.ranges

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
    return _trim_whitespace(
        layout.content,
        _Range(lines[left].start, lines[right].end),
    )


def _clamp_preserving_gold(
    content: str,
    window: _Range,
    gold: _Range,
    max_chars: int,
) -> _Range:
    """가능하면 max_chars로 제한하되 긴 gold 자체를 보존하는 일을 우선한다."""

    window = _Range(min(window.start, gold.start), max(window.end, gold.end))
    if window.length <= max_chars:
        return _trim_while_preserving_gold(content, window, gold)
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
    return _trim_while_preserving_gold(
        content,
        _Range(gold.start - left_room, gold.end + right_room),
        gold,
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


def _trim_while_preserving_gold(
    content: str,
    value: _Range,
    gold: _Range,
) -> _Range:
    """공백을 제거하되 gold의 원문 좌표는 절대 잘라내지 않는다."""

    trimmed = _trim_whitespace(content, value)
    return _Range(
        min(trimmed.start, gold.start),
        max(trimmed.end, gold.end),
    )


def _table_blocks(
    line_texts: tuple[str, ...],
) -> tuple[_LineBlock | None, ...]:
    """Markdown 표로 확인된 연속 행 블록을 각 행에 연결한다."""

    blocks: list[_LineBlock | None] = [None] * len(line_texts)
    index = 0
    while index < len(line_texts):
        if not _is_table_row_candidate(line_texts[index]):
            index += 1
            continue
        end = index
        while (
            end + 1 < len(line_texts)
            and _is_table_row_candidate(line_texts[end + 1])
        ):
            end += 1
        texts = line_texts[index:end + 1]
        confirmed = (
            any(_is_table_separator(text) for text in texts)
            or (
                len(texts) >= 2
                and all(_is_bordered_table_row(text) for text in texts)
            )
        )
        if confirmed:
            block = _LineBlock(index, end)
            for line_index in range(index, end + 1):
                blocks[line_index] = block
        index = end + 1
    return tuple(blocks)


def _is_table_row_candidate(text: str) -> bool:
    """표 후보 행인지 보되 단일 pipe 일반 문장은 제외한다."""

    stripped = text.strip()
    if not stripped or "|" not in stripped:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return len(cells) >= 2 and any(cells)


def _is_bordered_table_row(text: str) -> bool:
    """구분행이 없는 표를 허용할 때 요구하는 명시적 바깥 pipe 형식."""

    stripped = text.strip()
    return (
        stripped.startswith("|")
        and stripped.endswith("|")
        and _is_table_row_candidate(stripped)
    )


def _is_table_separator(text: str) -> bool:
    """Markdown 표의 열 정렬 구분행인지 검사한다."""

    return bool(_TABLE_SEPARATOR_RE.match(text))
