"""
agents/ingest/tables.py
PDF 표 추출 — pdfplumber 가 찾은 표를 검색 가능한 텍스트로 직렬화한다.

[왜 필요한가]
  extract_text() 는 표를 좌표 순서대로 흘려서 "2024 3.5 2025 4.1" 같은 숫자 나열로
  만든다. 헤더와 값의 관계가 사라져서 "2024년 실업률" 로는 절대 검색되지 않는다.
  이건 단순한 기능 누락이 아니라 진단 왜곡이다 — 수집 단계에서 깨진 것을 Eval 이
  청킹/검색 실패로 오진하고, Optimize 가 chunk_size 를 아무리 돌려도 낫지 않는다.

[직렬화 형식]
  Markdown 표를 쓰되, 표 앞에 "행 단위 문장"을 함께 넣는다.

      | 연도 | 실업률 |
      | --- | --- |
      | 2024 | 3.5 |

      연도: 2024, 실업률: 3.5

  이유: 청킹이 표를 자르면 한 행만 남을 수 있는데, "| 2024 | 3.5 |" 만으로는
  무슨 값인지 알 수 없다. 행 문장은 헤더를 매 행에 반복해 넣어 조각나도 자립한다.
  임베딩 검색에도 자연어 쪽이 유리하다.
"""
from __future__ import annotations

# 표로 인정할 최소 크기. 1행짜리나 1열짜리는 pdfplumber 가 레이아웃 선을 표로
# 오인한 경우가 대부분이라 버린다(본문이 표로 둔갑하면 피해가 크다).
_MIN_ROWS = 2
_MIN_COLS = 2

# 셀 값이 이보다 길면 표가 아니라 본문 문단이 선 안에 들어간 것으로 본다.
_MAX_CELL_LEN = 200

# 행 문장에서 값이 빈 칸은 건너뛴다 — "연도: 2024, 실업률: , 비고: " 같은 노이즈 방지.
_EMPTY_CELLS = {"", "-", "–", "—", "n/a", "N/A"}


def _clean_cell(cell: str | None) -> str:
    """셀 하나 정리 — None 처리와 셀 안 줄바꿈 제거."""
    if cell is None:
        return ""
    # 셀 안의 줄바꿈은 조판 때문이지 의미가 아니다. 공백으로 펴야 Markdown 표가 깨지지 않는다.
    text = " ".join(str(cell).split())
    # 파이프는 Markdown 표 구분자와 충돌하므로 escape.
    return text.replace("|", "\\|")


def _is_meaningful_table(rows: list[list[str]]) -> bool:
    """pdfplumber 가 넘긴 후보가 실제 표인지 판정한다."""
    if len(rows) < _MIN_ROWS:
        return False
    if max((len(r) for r in rows), default=0) < _MIN_COLS:
        return False
    # 셀이 전부 비어 있으면 선만 있는 장식이다.
    if not any(cell for row in rows for cell in row):
        return False
    # 셀 하나가 지나치게 길면 본문 문단을 표로 오인한 것.
    if any(len(cell) > _MAX_CELL_LEN for row in rows for cell in row):
        return False
    return True


def _row_sentences(header: list[str], body: list[list[str]]) -> list[str]:
    """각 데이터 행을 "헤더: 값" 문장으로 편다 — 청크가 잘려도 자립하도록."""
    sentences: list[str] = []
    for row in body:
        pairs = []
        for i, cell in enumerate(row):
            if cell.strip().lower() in _EMPTY_CELLS:
                continue
            label = header[i] if i < len(header) else ""
            label = label.strip()
            # 헤더가 비면 값만 넣는다(헤더 없는 표도 있다).
            pairs.append(f"{label}: {cell}" if label else cell)
        if pairs:
            sentences.append(", ".join(pairs))
    return sentences


def serialize_table(raw_rows: list[list[str | None]], *, caption: str = "") -> str:
    """pdfplumber extract_tables() 항목 하나 → 검색 가능한 텍스트. 표가 아니면 ""."""
    rows = [[_clean_cell(c) for c in row] for row in (raw_rows or [])]
    # 폭을 최대 열 수로 맞춘다 — 병합 셀 때문에 행마다 길이가 다를 수 있다.
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]

    if not _is_meaningful_table(rows):
        return ""

    header, body = rows[0], rows[1:]

    lines: list[str] = []
    if caption:
        lines.append(caption)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    sentences = _row_sentences(header, body)
    if sentences:
        lines.append("")
        lines.extend(sentences)

    return "\n".join(lines)


def extract_page_tables(page) -> list[str]:
    """pdfplumber Page → 직렬화된 표 텍스트 목록. 표가 없으면 빈 리스트.

    page 는 pdfplumber 의 Page 객체(덕 타이핑 — extract_tables() 만 있으면 된다).
    추출 자체가 실패해도 수집 전체를 죽이지 않는다 — 표는 부가 정보이고,
    본문 텍스트는 이미 확보돼 있기 때문이다.
    """
    try:
        raw_tables = page.extract_tables() or []
    except Exception as exc:  # noqa: BLE001 — pdfplumber 내부 오류 종류가 다양하다
        print(f"[Ingest] 표 추출 실패(본문은 유지): {exc}")
        return []

    out: list[str] = []
    for raw in raw_tables:
        text = serialize_table(raw)
        if text:
            out.append(text)
    return out
