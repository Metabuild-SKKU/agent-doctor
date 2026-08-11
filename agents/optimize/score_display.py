"""
agents/optimize/score_display.py
표시용 점수 변환 규약의 단일 출처.

[왜 이 파일이 있나]
  파이프라인에는 스케일이 다른 점수 두 축이 흐른다.

    - **탐색 신호** `before_score`/`after_score` (0~1) — `MIN_IMPROVEMENT_MARGIN`
      비교와 이력 측정 기록이 쓰는 값. `history._read_score` 가 composite÷100 으로
      만든다. 판정용이지 표시용이 아니다.
    - **표시용 종합점수** `before_composite`/`after_composite` (0~100) — 사용자가
      리포트 상단에서 보는 그 숫자와 같은 축.

  표시 소비처는 셋이다(reporter 요약 · web_api 티커 · report_view 카드). 규약이
  각 파일에 복사돼 있으면 새 소비처가 생길 때마다 같은 종류의 버그가 난다 —
  실제로 report_view 만 규약을 알고 나머지 둘이 0~1 을 그대로 찍던 것이 이 모듈이
  생긴 이유다. **표시 점수를 만드는 규칙은 여기 한 곳에만 둔다.**

[규약]  (`resolve_display_scores`)
  1. composite 쌍이 **둘 다** 있으면 그대로 쓴다(이미 0~100).
  2. **둘 다** 없으면 탐색 신호×100 으로 복원한다. 탐색 신호가 composite÷100 이라
     이 복원은 정확하다 — composite 미기록 구버전 이력을 위한 경로다.
  3. **한쪽만** 있으면 점수를 만들지 않는다(`None`). 이것이 이 모듈의 핵심 규칙이다.
     한쪽은 composite, 다른 쪽은 다른 축의 값×100 인 상황이 실제로 존재하기 때문이다
     (chunk prescreener 의 `best_score` 는 종합점수가 아니라 정답 span 포함률이다).
     서로 다른 축 두 개를 화살표로 이으면 **틀린 숫자를 사용자가 믿게 된다** — 표시를
     생략하는 쪽이 낫다.
  4. `proxy_only` 이력의 점수는 eval 이 매긴 종합점수가 아니다. 값이 있어도
     "종합점수"라는 이름을 붙이지 않는다(`DisplayScores.is_composite` 로 구분).

[쓰는 곳]  optimize/reporter.py, serve/web_api.py, serve/report_view.py
[쓰지 않는 곳]  history.judge 의 판정 — 판정은 탐색 신호(0~1)로 한다. 스케일을 바꾸면
           마진 상수의 의미가 통째로 흔들린다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# 표시용 점수가 없을 때 점수 문장 대신 쓰는 문구. 소비처가 제각각 문장을 지어내면
# 사용자가 "측정이 없다"와 "0점이다"를 구분하지 못한다.
UNMEASURED_CAPTION = "실측 종합점수 미측정"


@dataclass(frozen=True)
class DisplayScores:
    """표시용 점수 한 쌍과, 그 숫자를 뭐라고 불러도 되는지에 대한 판단.

    Attributes:
        before: 처방 전 표시 점수(0~100). 표시할 수 없으면 None.
        after: 처방 후 표시 점수(0~100). 표시할 수 없으면 None.
        is_composite: True 면 eval 이 매긴 종합점수라 "종합점수"로 불러도 된다.
            False 면 프록시 지표(prescreener 포함률 등)라 그 이름을 쓰면 안 된다.
        unavailable_reason: 점수가 없을 때 그 이유(디버깅·캡션용). 있으면 None.
    """

    before: Optional[float] = None
    after: Optional[float] = None
    is_composite: bool = True
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        """점수 문장을 만들어도 되는지. 두 값이 같은 축으로 갖춰졌을 때만 True."""
        return self.before is not None and self.after is not None


def resolve_display_scores(
    before_score: Optional[float],
    after_score: Optional[float],
    before_composite: Optional[float] = None,
    after_composite: Optional[float] = None,
    *,
    proxy_only: bool = False,
) -> DisplayScores:
    """탐색 신호와 composite 쌍에서 표시용 점수 쌍(0~100)을 정한다.

    규약은 모듈 docstring 참조. 폴백은 **쌍 단위**로만 한다 — 한쪽만 composite 가
    없을 때 다른 축의 값으로 채우지 않는 것이 이 함수의 존재 이유다.
    """
    # proxy_only 이력의 점수는 종합점수가 아니다. 값이 있어도 이름을 빌려주지 않는다.
    if proxy_only:
        return DisplayScores(
            is_composite=False,
            unavailable_reason="프록시 지표로 선택된 후보라 종합점수가 측정되지 않았습니다",
        )

    has_before = before_composite is not None
    has_after = after_composite is not None

    if has_before and has_after:
        return DisplayScores(
            before=round(float(before_composite), 1),
            after=round(float(after_composite), 1),
        )

    # 한쪽만 있는 경우. 없는 쪽을 탐색 신호×100 으로 채우면 그 값이 composite 라는
    # 보장이 없다(prescreener 경로가 정확히 이 경우다) → 축이 섞이느니 표시를 접는다.
    if has_before or has_after:
        return DisplayScores(
            is_composite=False,
            unavailable_reason=(
                "처방 전후 중 한쪽만 종합점수가 측정돼 비교할 수 없습니다"
            ),
        )

    # 둘 다 없다 = composite 를 기록하지 않던 구버전 이력. 이때의 탐색 신호는
    # composite÷100 이라 ×100 복원이 정확하다.
    if before_score is not None and after_score is not None:
        return DisplayScores(
            before=round(float(before_score) * 100, 1),
            after=round(float(after_score) * 100, 1),
        )

    return DisplayScores(unavailable_reason="점수가 기록되지 않았습니다")


def display_scores_from_verdict(verdict: Any) -> DisplayScores:
    """`Verdict` 에서 표시용 점수 쌍을 뽑는다(reporter 용 래퍼)."""
    return resolve_display_scores(
        getattr(verdict, "before_score", None),
        getattr(verdict, "after_score", None),
        getattr(verdict, "before_composite", None),
        getattr(verdict, "after_composite", None),
        proxy_only=bool(getattr(verdict, "proxy_only", False)),
    )


def display_scores_from_metadata(metadata: dict) -> DisplayScores:
    """이력 항목 metadata 에서 표시용 점수 쌍을 뽑는다(티커·카드 용 래퍼).

    권위 있는 표시값은 이력 metadata 에 확정 기록된다(`history.finalize_item`,
    sweep 종료 시 `agent._finish_internal_study`). 같은 trial 의 표시면들이 서로
    다른 출처를 보지 않도록, metadata 를 읽는 소비처는 모두 이 함수를 거친다.
    """
    metadata = metadata or {}
    return resolve_display_scores(
        metadata.get("before_score"),
        metadata.get("after_score"),
        metadata.get("before_composite"),
        metadata.get("after_composite"),
        proxy_only=bool(metadata.get("proxy_only")),
    )
