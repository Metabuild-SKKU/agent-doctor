"""
agents/ingest/preprocess.py
PDF 텍스트 전처리 — pdfplumber 로 뽑은 페이지별 원문을 청킹하기 좋은 형태로 정리한다.

[왜 별도 모듈인가]
  추출(pdfplumber I/O)과 정리(순수 문자열 변환)를 분리해두면 정리 로직만 파일 없이
  단위 테스트할 수 있다. agent.py 의 _ingest_file 은 추출만 하고 여기에 넘긴다.

[처리 순서] — 순서가 결과를 바꾸므로 바꾸지 말 것
  1. 페이지 반복 요소(머리말/꼬리말/페이지 번호) 제거   ← 페이지 경계가 살아있어야 가능
  2. 페이지별 본문 정리(하이픈 줄바꿈 복원 등)
  3. 페이지 이어붙이기 + 페이지별 char_span 기록

OCR 은 범위 밖 — 스캔본 감지만 하고(is_probably_scanned) 실제 인식은 하지 않는다.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# 페이지 구분자. Index 의 markdown_recursive 청킹이 빈 줄을 문단 경계로 보므로
# 페이지 사이도 빈 줄 두 개로 둔다(문단 중간에 페이지가 끊겨도 별도 청크가 되지 않게).
PAGE_SEPARATOR = "\n\n"

# 머리말/꼬리말 후보로 볼 줄 길이 상한. 본문 문장이 우연히 여러 페이지에 반복될 때
# 지워버리는 사고를 막는다(반복되는 본문 한 문단은 보통 이보다 길다).
_MAX_RUNNING_HEAD_LEN = 80

# 반복 요소로 판정할 최소 페이지 수·비율. 3페이지 미만 문서는 "반복"을 신뢰할 수 없어
# 아예 제거하지 않는다.
_MIN_PAGES_FOR_HEADER_STRIP = 3
_HEADER_REPEAT_RATIO = 0.6

# 페이지 번호로만 이루어진 줄: "12", "- 12 -", "12 / 34", "Page 12", "p. 12"
_PAGE_NUMBER_RE = re.compile(
    r"""^\s*(?:
        [-–—~<\[(]*\s*\d+\s*[-–—~>\])]*        # 12 · - 12 - · <12>
      | \d+\s*(?:/|of)\s*\d+                   # 12 / 34 · 12 of 34
      | (?:page|p\.?|페이지)\s*\d+(?:\s*(?:/|of)\s*\d+)?
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# 줄 "끝"에 꼬리처럼 붙은 페이지 마커: "전자공시시스템 dart.fss.or.kr Page 522".
# _PAGE_NUMBER_RE 는 줄 전체가 페이지 번호일 때만 매치하므로(^...$) 앞에 텍스트가 붙은
# 이런 꼬리말은 못 잡는다. 실제 DART 사업보고서에서 이 푸터가 그대로 청크에 남아
# Probe 의 주제 문구로 쓰였다 — "dart.fss.or.kr Page 522 의 관계를 설명해줘" 같은
# 답할 수 없는 질문이 만들어졌다. 줄을 통째로 지우지 않고 꼬리만 잘라 본문을 지킨다.
_TRAILING_PAGE_MARKER_RE = re.compile(
    r"\s*(?:page|p\.|페이지)\s*\d+\s*$",
    re.IGNORECASE,
)

# 줄 끝 하이픈 + 줄바꿈 = 조판상 단어가 잘린 것. 영문 PDF 에서 흔하다.
# 앞뒤가 모두 알파벳일 때만 붙인다("- 항목" 같은 목록 기호나 숫자 범위는 건드리지 않음).
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])[-­]\n([a-z])")

# 한글은 하이픈 없이 줄이 바뀌므로 줄바꿈만 보고는 단어 중간인지 단어 사이인지 알 수 없다.
# ("회복세가\n뚜렷하게" 는 단어 사이, "회복세\n가" 는 단어 중간).
# 그래서 붙이지 않고 공백으로 바꾼다 — 잘못 넣은 공백은 토크나이저가 흡수하지만,
# 잘못 붙이면 "회복세가뚜렷하게" 같은 없는 단어가 생겨 임베딩과 F1 채점이 모두 망가진다.
_HANGUL_LINE_BREAK_RE = re.compile(r"([가-힣])\n([가-힣])")
_HANGUL_LINE_BREAK_SUB = r"\1 \2"

# 단어 중간에 낀 소프트 하이픈·제로폭 문자 — 임베딩 토크나이저를 망가뜨린다.
_INVISIBLE_RE = re.compile(r"[­​‌‍﻿]")

# 3줄 이상 연속된 빈 줄 → 2줄. 페이지 여백 때문에 생기는 과도한 공백 정리.
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")

# 본문이 이 정도도 안 되면 텍스트 레이어가 없는 스캔본으로 본다(페이지당 평균 문자 수).
_SCANNED_CHARS_PER_PAGE = 50


@dataclass
class PreprocessResult:
    """전처리 결과 — content 와 그 content 를 설명하는 메타데이터."""

    content: str
    page_spans: list[tuple[int, int]] = field(default_factory=list)
    # page_spans[i] = content 내 (i+1)페이지의 (start, end) char offset.
    # Index 가 만드는 Chunk.char_span 과 같은 좌표계라, 청크 → 페이지 역산이 가능하다.
    removed_headers: list[str] = field(default_factory=list)
    page_count: int = 0
    empty_page_count: int = 0
    table_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()

    @property
    def is_probably_scanned(self) -> bool:
        """텍스트 레이어가 없는 스캔본으로 의심되는가 (OCR 필요 신호)."""
        if self.page_count == 0:
            return False
        return len(self.content.strip()) < _SCANNED_CHARS_PER_PAGE * self.page_count

    def page_of(self, char_offset: int) -> int | None:
        """content 기준 char offset 이 몇 페이지인지 (1-based). 못 찾으면 None."""
        for idx, (start, end) in enumerate(self.page_spans, start=1):
            if start <= char_offset < end:
                return idx
        return None


def _normalize_line(line: str) -> str:
    """비교용 정규화 — 페이지마다 바뀌는 숫자를 지워 반복 여부를 판정한다.

    "3장 서론 · 12" 와 "3장 서론 · 13" 은 같은 머리말이므로 같은 키가 되어야 한다.
    숫자를 지우면 서로 다른 본문까지 같아 보일 수 있는데("본문 1 입니다."·
    "본문 2 입니다."), 그건 _find_repeated_lines 가 위치 고정성으로 걸러낸다.
    """
    return re.sub(r"\d+", "#", line).strip().lower()


# 머리말/꼬리말로 인정할 위치. 페이지의 첫 줄이거나 마지막 줄 — "위에서 두 번째"까지
# 넓히면 본문 첫 문장이 휩쓸려 나간다.
#
# [실측] 이 범위를 위아래 2줄로 넓히고 반복 비율을 0.4 로 낮춰봤다가 되돌렸다. 위에서 두
# 번째 줄의 본문(80자 미만이라 길이 제한도 통과)이 통째로 지워지면서
# test_repeated_body_text_is_not_stripped 가 깨졌다. 푸터가 가장자리 밖으로 밀리는 문제는
# 범위를 넓혀서가 아니라 _TRAILING_PAGE_MARKER_RE(페이지 마커가 붙은 줄은 위치 무관 처리)로
# 푼다 — 본문 손실 위험이 없는 쪽.
def _edge_lines(page: str) -> tuple[str | None, str | None]:
    """페이지의 (첫 내용 줄, 마지막 내용 줄). 내용이 없으면 (None, None)."""
    lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
    if not lines:
        return None, None
    return lines[0], lines[-1]


def _find_repeated_lines(pages: list[str]) -> set[str]:
    """페이지의 같은 자리(첫 줄/마지막 줄)에 반복되는 줄을 찾는다.

    위치를 고정해서 세는 게 핵심이다. "어느 가장자리든" 으로 느슨하게 잡으면,
    페이지마다 다른 본문 문장이 우연히 가장자리에 놓였을 때 머리말로 오인돼
    본문이 통째로 지워진다. 머리말은 정의상 항상 맨 위, 꼬리말은 항상 맨 아래다.
    """
    if len(pages) < _MIN_PAGES_FOR_HEADER_STRIP:
        return set()

    head_counter: Counter[str] = Counter()
    foot_counter: Counter[str] = Counter()

    for page in pages:
        head, foot = _edge_lines(page)
        if head is not None and len(head) <= _MAX_RUNNING_HEAD_LEN:
            head_counter[_normalize_line(head)] += 1
        # 한 줄짜리 페이지면 head 와 foot 이 같다 — 중복 카운트하지 않는다.
        if foot is not None and foot != head and len(foot) <= _MAX_RUNNING_HEAD_LEN:
            foot_counter[_normalize_line(foot)] += 1

    threshold = max(_MIN_PAGES_FOR_HEADER_STRIP, int(len(pages) * _HEADER_REPEAT_RATIO))
    repeated = {key for key, count in head_counter.items() if count >= threshold}
    repeated |= {key for key, count in foot_counter.items() if count >= threshold}
    return repeated


def _strip_page_furniture(page: str, repeated: set[str]) -> tuple[str, list[str]]:
    """한 페이지에서 머리말/꼬리말/페이지 번호 줄을 제거한다."""
    kept: list[str] = []
    removed: list[str] = []

    lines = page.splitlines()
    # 가장자리 = 내용이 있는 첫 줄과 마지막 줄. _find_repeated_lines 와 같은 기준이어야
    # 후보로 센 자리와 실제로 지우는 자리가 어긋나지 않는다.
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    if not filled:
        return "", []
    edge_idx = {filled[0], filled[-1]}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        is_edge = i in edge_idx
        # 페이지 번호는 어디에 있든(가장자리 판정 실패해도) 지운다.
        if _PAGE_NUMBER_RE.match(stripped):
            removed.append(stripped)
            continue
        # 반복 판정은 "자르기 전" 원본 줄로 해야 한다 — 꼬리를 먼저 자르면
        # "...dart.fss.or.kr Page 501" 이 "...dart.fss.or.kr" 이 되면서 반복 키와
        # 어긋나 되레 살아남는다.
        is_repeated = _normalize_line(stripped) in repeated
        # 페이지 마커가 꼬리에 붙은 줄은 그 자체가 페이지 가구(furniture)이므로,
        # 반복으로 확인됐다면 가장자리가 아니어도 줄째 지운다. 표가 섞인 페이지에서
        # 푸터가 마지막 줄 자리를 벗어나는 경우가 실제로 있었다(DART 사업보고서).
        has_page_marker = bool(_TRAILING_PAGE_MARKER_RE.search(stripped))
        if is_repeated and (is_edge or has_page_marker):
            removed.append(stripped)
            continue
        # 반복까지는 아니어도 꼬리에 페이지 마커가 붙어 있으면 그 부분만 잘라낸다.
        # 줄 전체를 지우지 않으므로 본문 손실 위험이 없다.
        trimmed = _TRAILING_PAGE_MARKER_RE.sub("", stripped)
        if trimmed != stripped:
            removed.append(stripped[len(trimmed):].strip())
            if not trimmed:
                continue
            kept.append(trimmed)
            continue
        kept.append(line)

    return "\n".join(kept), removed


def _clean_body(text: str) -> str:
    """페이지 본문 정리 — 줄바꿈 복원과 공백 정규화."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 하이픈 줄바꿈 복원은 소프트 하이픈 제거보다 먼저 — 제거하면 단서가 사라진다.
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _HANGUL_LINE_BREAK_RE.sub(_HANGUL_LINE_BREAK_SUB, text)
    text = _INVISIBLE_RE.sub("", text)
    # 줄 끝 공백 제거(청크 텍스트에 그대로 남으면 임베딩에 노이즈)
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    text = _EXCESS_BLANK_RE.sub("\n\n", text)
    return text.strip()


def preprocess_pages(
    pages: list[str | None],
    *,
    page_tables: list[list[str]] | None = None,
) -> PreprocessResult:
    """페이지별 원문 텍스트 리스트 → 정리된 단일 텍스트 + 페이지 span.

    pages 항목이 None 이어도 된다 — pdfplumber 의 extract_text() 는 텍스트 레이어가
    없는 페이지에서 None 을 돌려준다. 빈 페이지로 세고 넘어간다.

    page_tables[i] 는 i 페이지에서 뽑은 직렬화된 표 목록(agents/ingest/tables.py).
    본문 정리가 끝난 뒤에 붙인다 — _clean_body 를 태우면 Markdown 표의 줄 구조와
    파이프가 망가지기 때문이다.
    """
    raw_pages = [p or "" for p in pages]
    tables_per_page = page_tables or []
    repeated = _find_repeated_lines(raw_pages)

    cleaned_pages: list[str] = []
    removed_all: list[str] = []
    empty_count = 0
    table_count = 0

    for idx, page in enumerate(raw_pages):
        stripped, removed = _strip_page_furniture(page, repeated)
        removed_all.extend(removed)
        body = _clean_body(stripped)

        tables = tables_per_page[idx] if idx < len(tables_per_page) else []
        if tables:
            table_count += len(tables)
            # 빈 줄로 띄워야 청킹이 표를 본문 문단과 섞지 않는다.
            body = "\n\n".join([body, *tables]) if body else "\n\n".join(tables)

        if not body:
            empty_count += 1
        cleaned_pages.append(body)

    # 이어붙이면서 페이지별 span 기록. 빈 페이지도 span 을 갖는다(길이 0),
    # 그래야 page_spans 인덱스와 실제 페이지 번호가 어긋나지 않는다.
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, body in enumerate(cleaned_pages):
        if i > 0:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)
        spans.append((cursor, cursor + len(body)))
        parts.append(body)
        cursor += len(body)

    # strip() 은 span 좌표계를 바꾼다. spans 는 strip 이전 기준으로 쌓았으므로
    # 앞에서 잘려나간 만큼 전부 당겨줘야 한다. 표지처럼 첫 페이지가 전처리 후
    # 비는 문서(머리말만 있던 페이지)에서 실제로 모든 페이지가 밀렸다 —
    # Chunk.page 가 조용히 틀린 번호를 기록하므로 발견이 어려운 종류의 버그다.
    joined = "".join(parts)
    content = joined.strip()
    lead = len(joined) - len(joined.lstrip())
    limit = len(content)
    spans = [
        (min(max(0, start - lead), limit), min(max(0, end - lead), limit))
        for start, end in spans
    ]
    if not content:
        spans = [(0, 0) for _ in cleaned_pages]

    return PreprocessResult(
        content=content,
        page_spans=spans,
        removed_headers=sorted(set(removed_all)),
        page_count=len(raw_pages),
        empty_page_count=empty_count,
        table_count=table_count,
    )
