"""
agents/optimize/ext_advisor.py
외부 RAG(리플레이 모드) 전용 "권고 카드" 계층.

[왜 Optimize 에 있나]
  라벨 판정은 Eval(replay_labels) 소관이지만, "그 라벨에 무슨 처방이 붙고 그게
  사람 말로 무엇인가"는 rules.py 를 아는 Optimize 의 지식이다. reporter.py 가
  내부 모드에서 하는 일(결정 → 사람이 읽는 번역)의 외부 모드 짝이다.
  카드 배치·헤드라인은 serve/report_view 소관 — 여기서는 재료만 만든다.

[내부 모드와 다른 점 — 자동 적용이 없다]
  Optimize 의 처방은 index_config 를 바꿔 재색인하는 것인데, 외부 RAG 는 남의
  인덱스라 우리가 못 바꾼다. 그래서 planner/optimizer/history 를 타지 않고
  rules 의 처방을 **읽기 전용으로 재참조**해 권고 문구로만 번역한다.
  LABEL_TO_PRESCRIPTIONS 에 ext_ 를 섞지 않는 이유와 같다(replay_labels 주석 참고).

[읽는 것]  Finding(ext_ 라벨), 로그의 config(있으면), rules.py
[쓰는 것]  없음 — 카드 dict 리스트를 만들어 반환만 한다(reporter 와 동일 규칙).
"""
from __future__ import annotations

from typing import Any, Optional

from core.schema import Finding

from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS

# ── ext_ 라벨 → 기존 rules 라벨 ──────────────────────────────────
# 재참조 매핑. replay_labels.EXT_RECOMMENDATIONS 와 같은 대응이지만, 저쪽은
# 처방 id 만 필요한 CLI 출력용이고 여기는 patch/detail 까지 읽어 카드를 만든다.
EXT_LABEL_TO_RULE: dict[str, str] = {
    "ext_generation_hallucination": "generation_hallucination",
    "ext_answer_off_topic": "generation_misinterpretation",
    "ext_context_overflow": "too_long_context",
    "ext_grounded_but_wrong": "corpus_gap",
}

# 라벨 자체의 한 줄 설명 — 카드 제목 밑에 붙는다.
LABEL_SUMMARY: dict[str, str] = {
    "ext_generation_hallucination": "답변이 검색된 근거에 기반하지 않습니다(환각)",
    "ext_answer_off_topic": "답변이 질문을 비껴갑니다(동문서답)",
    "ext_context_overflow": "컨텍스트가 과다해 근거 사용을 방해합니다",
    "ext_grounded_but_wrong": "답변은 근거에 충실하나 근거 자체가 틀렸거나 부족합니다",
}

# 배지·칩처럼 좁은 자리에 쓰는 짧은 이름. 라벨 코드(ext_grounded_but_wrong)를 그대로
# 노출하면 우리 코드를 아는 사람만 읽을 수 있다 — 진단서는 상대 팀이 보는 문서다.
LABEL_SHORT: dict[str, str] = {
    "ext_generation_hallucination": "환각",
    "ext_answer_off_topic": "동문서답",
    "ext_context_overflow": "컨텍스트 과다",
    "ext_grounded_but_wrong": "근거 부족",
}


def label_summary(label: str) -> str:
    """라벨 → 사람이 읽는 한 줄. 모르는 라벨이면 코드를 그대로 돌려준다
    (지어내는 것보다 rules/replay_labels 로 되짚을 수 있는 게 낫다)."""
    return LABEL_SUMMARY.get(label) or label


def label_short(label: str) -> str:
    """라벨 → 배지용 짧은 이름."""
    return LABEL_SHORT.get(label) or label

# 처방 id → 사람이 읽는 조치 문구. rules.py 의 manual 처방은 action/detail 을
# 이미 갖고 있지만 ready 처방은 기계용 patch 뿐이라, 그 빈자리를 여기서 메운다.
# ext_ 4라벨이 참조하는 10개만 담는다 — 전체 45개를 미리 채우면 쓰이지 않는
# 문구가 rules 변경과 어긋난 채 남는다.
_PRESCRIPTION_TEXT: dict[str, dict[str, str]] = {
    "lower_temperature": {
        "action": "생성 온도를 낮추세요",
        "detail": "temperature 를 내리면 모델이 근거에서 벗어나 지어낼 여지가 줄어듭니다.",
    },
    "strengthen_abstention": {
        "action": "기권 조건을 강화하세요",
        "detail": "근거가 부족하면 '모른다'고 답하도록 프롬프트에 기권 규칙을 넣습니다. "
                  "틀린 답보다 모른다는 답이 낫습니다.",
    },
    "require_citation": {
        "action": "인용을 강제하세요",
        "detail": "답변에 사용한 컨텍스트 문장을 함께 제시하도록 요구하면, "
                  "근거 없는 문장이 생성 단계에서 걸러집니다.",
    },
    "upgrade_generation_model": {
        "action": "더 강한 생성 모델로 올리세요",
        "detail": "근거를 읽고 종합하는 능력이 모자라 환각이 나는 경우입니다. "
                  "프롬프트 조정이 듣지 않으면 마지막으로 검토하세요.",
    },
    "restate_question": {
        "action": "질문 재진술을 켜세요",
        "detail": "답변 전에 질문을 다시 정리하게 하면 동문서답이 줄어듭니다.",
    },
    "decrease_top_k": {
        "action": "top_k 를 낮추세요",
        "detail": "검색 결과를 적게 넣어 컨텍스트를 줄입니다. 관련 없는 청크가 "
                  "근거 사용을 방해하는 상황입니다.",
    },
    "context_compression": {
        "action": "컨텍스트 압축을 켜세요",
        "detail": "청크 수를 줄이지 않고 각 청크에서 질문과 관련된 부분만 남깁니다.",
    },
    "shrink_chunk_size": {
        "action": "청크 크기를 줄이세요",
        "detail": "청크 하나가 커서 컨텍스트가 넘칩니다. 재색인이 필요합니다.",
    },
}


def _prescription_entries(rule_label: str) -> list[dict]:
    entry = LABEL_TO_PRESCRIPTIONS.get(rule_label) or {}
    return list(entry.get("prescriptions") or [])


def _current_value(patch_key: str, config: dict) -> Optional[Any]:
    """로그 config 에서 이 patch 가 건드리는 현재값을 찾는다.

    상대 로그의 config 는 자유 키라 우리 축 이름과 완전히 일치하지 않는다.
    'retriever.top_k' 처럼 점이 있는 축은 마지막 조각으로도 한 번 더 찾는다."""
    if not config:
        return None
    if patch_key in config:
        return config[patch_key]
    tail = patch_key.rsplit(".", 1)[-1]
    return config.get(tail)


def _direction_hint(patch: dict, config: dict) -> Optional[str]:
    """'top_k 5 → 낮추기' 처럼 현재값을 실은 한 조각.

    구체적인 목표값은 제시하지 않는다 — 남의 인덱스라 우리가 탐색해 본 적이
    없고, 근거 없는 숫자를 적으면 권고가 사실보다 단정적으로 읽힌다."""
    for key, want in (patch or {}).items():
        cur = _current_value(key, config)
        if cur is None:
            continue
        name = key.rsplit(".", 1)[-1]
        if want == "decrease":
            return f"현재 {name}={cur} → 낮추기"
        if want == "increase":
            return f"현재 {name}={cur} → 올리기"
        if isinstance(want, bool):
            return f"현재 {name}={cur} → {str(want).lower()}"
    return None


def _steps(rule_label: str, config: dict) -> list[dict[str, Any]]:
    """처방 목록 → 카드에 실을 조치 단계."""
    steps: list[dict[str, Any]] = []
    for p in _prescription_entries(rule_label):
        pid = p.get("id") or ""
        text = _PRESCRIPTION_TEXT.get(pid)
        if text:
            action, detail = text["action"], text["detail"]
        else:
            # manual 처방은 rules.py 가 이미 사람용 문구를 갖고 있다. 둘 다 없으면
            # id 를 그대로 노출한다 — 지어내는 것보다 rules 로 되짚을 수 있는 게 낫다.
            action = p.get("action") or pid
            detail = p.get("detail") or ""
        steps.append({
            "prescription_id": pid,
            "action": action,
            "detail": detail,
            "current": _direction_hint(p.get("patch") or {}, config),
            "needs_reindex": bool(p.get("reindex")),
            "manual": bool(p.get("manual")),
        })
    return steps


def build_ext_recommendations(
    findings: list[Finding],
    config: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """ext_ 소견 → 권고 카드 재료. 심각도·확정 건수 순으로 정렬한다.

    같은 라벨의 probe 별 소견은 카드 하나로 묶는다(라벨이 처방의 단위라서).
    확정(confirmed)과 예비를 따로 세어 카드에 싣는다 — 예비를 확정처럼 보여주면
    증거 수준을 속이는 것이고, 빼버리면 gold_contexts 가 없는 로그에서 카드가
    통째로 사라진다.

    반환 형식(serve/report_view 가 배치에만 쓰는 재료):
      [{label, summary, severity, confirmed, tentative, evidence, steps[]}, ...]
    """
    config = config or {}
    grouped: dict[str, dict[str, Any]] = {}

    for f in findings or []:
        label = getattr(f, "label", "") or ""
        if label not in EXT_LABEL_TO_RULE:
            continue                      # 내부 라벨은 이 경로가 다루지 않는다
        card = grouped.setdefault(label, {
            "label": label,
            "summary": label_summary(label),
            "severity": getattr(f, "severity", "") or "warning",
            "confirmed": 0,
            "tentative": 0,
            "evidence": [],
            "steps": _steps(EXT_LABEL_TO_RULE[label], config),
        })
        if getattr(f, "confirmed", False):
            card["confirmed"] += 1
        else:
            card["tentative"] += 1
        # metadata['reason'] 은 "[리플레이 소견] 라벨 - " 접두어가 없는 원문이다.
        # description 을 그대로 실으면 라벨 코드가 화면에 노출된다.
        why = (str(getattr(f, "metadata", {}).get("reason") or "").strip()
               or str(getattr(f, "description", "") or "").strip())
        if why and why not in card["evidence"] and len(card["evidence"]) < 3:
            card["evidence"].append(why)

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(
        grouped.values(),
        key=lambda c: (order.get(c["severity"], 9), -c["confirmed"], -c["tentative"]),
    )
