"""
agents/eval/scoring.py
STEP5 종합점수 — 품질·신뢰도를 결합한 0~100 점수.

모듈 설계(교체 용이 — 이 파일의 존재 이유):
  · 점수 성분(component): records -> 0~1 (또는 None=측정불가). 독립 함수 1개 = 성분 1개.
  · COMPONENTS 레지스트리: (key, label, fn) 의 순서 목록. 성분 추가/삭제는 여기 한 줄.
  · combine(): 성분들을 하나로 합치는 방식(현재 조화평균). 이 함수만 바꾸면 결합식이 교체된다.
  · compute_composite(): report 가 부르는 유일한 진입점. 성분 계산 → 결합 → CompositeScore.

이 기능을 통째로 버리려면 report.py 의 compute_composite 호출과 import 를 지우면 된다
(DiagnosticReport.composite_score 는 None 으로 남아 무해).

왜 조화평균인가:
  품질(답이 얼마나 좋은가)과 신뢰도(얼마나 자주 맞히는가)는 직교하는 축이라, 산술평균은
  '가끔 완벽'과 '항상 그럭저럭'을 못 가른다. 조화평균은 한 축만 낮아도 총점을 크게 끌어내려
  (precision·recall → F1 과 같은 결합) 약한 축을 숨기지 않는다.

RAGAS 가중평균(_ragas_means)과 계산이 겹치지만, 이 모듈은 report 에 의존하지 않고 홀로
서도록 자체 계산한다(순환 import 방지 + 통째로 드롭 가능). 중복보다 독립성을 택한 것.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from agents.eval.types import EvalRecord, RAGAS_WEIGHTS


# ── 점수 성분 (component) — 각 함수는 records 를 받아 0~1 또는 None ─────────────

def quality_score(records: list[EvalRecord]) -> Optional[float]:
    """품질 축 — 실제 트랙 RAGAS 4지표 가중평균(측정된 것만 재정규화).
    RAGAS 미측정(DEEP 미만 등)으로 한 지표도 없으면 None."""
    present: dict[str, float] = {}
    for key in RAGAS_WEIGHTS:
        vals = [r.ragas[key] for r in records if r.ragas.get(key) is not None]
        if vals:
            present[key] = sum(vals) / len(vals)
    if not present:
        return None
    wsum = sum(RAGAS_WEIGHTS[k] for k in present)
    return sum(present[k] * RAGAS_WEIGHTS[k] for k in present) / wsum


def reliability_score(records: list[EvalRecord]) -> Optional[float]:
    """신뢰도 축 — 판정 가능한 probe 중 finding 이 없는(=통과) 비율.
    diagnose 는 판정 불가 probe(정답셋 없음 등)엔 finding 을 만들지 않으므로,
    판정 가능한 probe 에서 finding 유무가 곧 실패/통과다. 판정 가능한 probe 가 없으면 None."""
    evaluable = [r for r in records if _is_evaluable(r)]
    if not evaluable:
        return None
    passed = sum(1 for r in evaluable if not r.findings)
    return passed / len(evaluable)


def _is_evaluable(record: EvalRecord) -> bool:
    """pass/fail 판정이 가능한 probe: 정답셋이 있거나(정답 대조 가능),
    무응답 기대(answer_exists=False, 올바른 회피인지 판정 가능)."""
    return bool(record.probe.ground_truth) or record.probe.answer_exists is False


# ── 성분 레지스트리 — 추가/삭제는 이 목록 한 줄 (순서 = 표시 순서) ────────────
COMPONENTS: list[tuple[str, str, Callable[[list[EvalRecord]], Optional[float]]]] = [
    ("quality", "품질", quality_score),
    ("reliability", "신뢰도", reliability_score),
]


# ── 결합식 — 이 함수만 바꾸면 조화평균 → 산술/가중/최소 등으로 교체 ───────────

def combine(values: list[float]) -> float:
    """성분(0~1)들을 조화평균으로 결합. 한 축이라도 0 이면 0(사슬이 끊긴 것으로 본다)."""
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


# ── 결과 자료구조 + 진입점 ───────────────────────────────────────────────────

@dataclass
class _Component:
    key: str
    label: str
    value: Optional[float]     # 0~1 (None=측정불가)


@dataclass
class CompositeScore:
    total: Optional[int]              # 0~100 (측정 가능한 성분이 하나도 없으면 None)
    components: list[_Component]

    def as_dict(self) -> dict:
        """DiagnosticReport.composite_score 에 실을 직렬화 형태(성분별 0~100 포함)."""
        return {
            "total": self.total,
            "components": [
                {"key": c.key, "label": c.label,
                 "score": (round(c.value * 100) if c.value is not None else None)}
                for c in self.components
            ],
        }


def compute_composite(records: list[EvalRecord]) -> CompositeScore:
    """records 로부터 종합점수(0~100)와 성분별 점수를 계산. report 의 유일한 진입점.
    측정 불가 성분(None)은 결합에서 제외 → 남은 성분만으로 총점을 낸다(우아한 저하).
    측정 가능한 성분이 하나도 없으면 total=None."""
    components = [_Component(key, label, fn(records)) for key, label, fn in COMPONENTS]
    present = [c.value for c in components if c.value is not None]
    total = round(combine(present) * 100) if present else None
    return CompositeScore(total=total, components=components)


def format_composite(d: Optional[dict]) -> str:
    """composite_score dict → '68/100 (품질 79 / 신뢰도 60)' 로그·표시용 한 줄.
    (dict 에서 복원하므로 CompositeScore 객체 없이도 리포트만 있으면 출력 가능.)"""
    if not d:
        return "-"
    total = d.get("total")
    head = f"{total}/100" if total is not None else "-/100"
    parts = " / ".join(
        f"{c['label']} {c['score']}" if c.get("score") is not None else f"{c['label']} -"
        for c in d.get("components", [])
    )
    return f"{head} ({parts})" if parts else head
