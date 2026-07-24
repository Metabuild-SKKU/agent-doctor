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
    """신뢰도 축 — 판정 가능한 probe 별 신뢰도([0,1] 연속값)의 평균.
    판정 가능한 probe 가 없으면 None.

    과거엔 finding 유무로 통과/실패를 이진 카운트(passed/total)했으나, 그 계단 함수가
    최적화 탐색 신호로 못 쓰이는 근본 원인이었다(값이 통과↔실패로 flip 되기 전엔 파라미터를
    조금 개선해도 신호가 전혀 안 움직임 → composite 이 평평 → overall 을 따로 둘 수밖에 없었음).
    각 probe 를 [0,1] 연속 신뢰도로 바꾸면 '거의 통과'가 부분점수를 받아 신호가 매끄러워지고,
    composite 하나로 탐색·표시를 통일할 수 있다(_probe_reliability 참고)."""
    evaluable = [r for r in records if _is_evaluable(r)]
    if not evaluable:
        return None
    return sum(_probe_reliability(r) for r in evaluable) / len(evaluable)


def _probe_reliability(record: EvalRecord) -> float:
    """probe 1개의 신뢰도를 [0,1] 연속값으로 — 이진 판정(_is_success)의 매끄러운 대응물.

    · 무응답 기대(answer_exists=False): 연속 축이 없어 finding 유무로 1/0(옳게 기권=1).
    · gold 대조 probe: 검색축(recall@k) × 답변축. 답변축은 lexical f1_score 와 의미
      answer_correctness 중 높은 값 — char-F1 이 낮아도 의미가 맞으면 부분점수를 준다.
      (게이트 _f1_ok 는 answer_correctness 로 강등만 하지만, 신뢰도 축은 승격까지 허용해
       긴 서술형 gold 에서 char-F1 이 구조적으로 저평가되는 문제를 완화한다.)
    두 축의 곱이라 검색·답변 어느 한쪽이 무너지면 신뢰도가 낮게 나온다(이진 게이트의 AND 대응)."""
    if record.probe.answer_exists is False:
        return 0.0 if record.findings else 1.0
    retrieval = _clamp01(record.recall_at_k)
    ac = record.ragas_answer_correctness
    answer = _clamp01(max(record.f1_score, ac if ac is not None else 0.0))
    return retrieval * answer


def _clamp01(value: float) -> float:
    """[0,1] 로 자른다(recall_at_k 의 -1 sentinel·부동소수 초과 방어)."""
    return 0.0 if value < 0 else (1.0 if value > 1 else value)


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
    """성분(0~1)들을 조화평균으로 결합. 한 축이 낮을수록 총점을 강하게 끌어내린다.

    조화평균은 유지한다 — 성분(신뢰도)이 이제 연속값이라, 조화평균은 저신뢰도 구간에서
    오히려 기울기가 가장 커(H(0.8,0.1→0.2): 0.18→0.32) 탐색이 구멍에서 빠져나올 방향을
    또렷하게 가리킨다. 과거 우려했던 '저신뢰도에서 평평/붕괴'는 조화평균이 아니라 신뢰도가
    이진 카운트였기 때문(flip 전엔 안 움직임)이었고, 그 원인은 reliability_score 에서 제거됐다.
    다만 값이 정확히 0 일 때 총점을 계단처럼 0 으로 떨구던 하드 컷은 제거한다(eps clamp) —
    탐색 신호가 0 근방에서도 매끄럽게 이어지도록."""
    if not values:
        return 0.0
    eps = 1e-6
    return len(values) / sum(1.0 / max(v, eps) for v in values)


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
