# tests/test_ragas_eval.py
# ragas_eval 을 **실제 OpenAI API** 로 검증. (모듈 코드 변경 없이 함수 직접 호출)
#
# 실행:
#   1) .env 에 OPENAI_API_KEY 설정 (또는 환경변수)
#   2) python tests/test_ragas_eval.py
#      - LLM 입출력을 보고 싶으면:  EVAL_DEBUG=1 python tests/test_ragas_eval.py
#   ※ 실제 API 호출이라 소액 비용 발생(~$0.01~0.05). 키 없으면 자동 스킵.
#
# 검증 전략: LLM은 temperature=0 이어도 완벽 결정적이진 않으므로, 절대값 대신
#           "정답을 아는 케이스의 상대 관계"(할루시네이션<정확, 모른다=0, 관련청크 앞>뒤)를 확인.

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_key = os.getenv("OPENAI_API_KEY", "").strip()
if not _key or "..." in _key or len(_key) < 20:
    # 비었거나 .env.example 의 placeholder("sk-...") 면 실제 API 불가 → 스킵
    print("실제 OPENAI_API_KEY 없음(비었거나 placeholder) → 스킵")
    print("  .env 의 OPENAI_API_KEY 를 진짜 키로 바꾸고 다시 실행하세요.")
    sys.exit(0)

os.environ["EVAL_ENABLE_LLM"] = "1"   # RAGAS 진단 활성화

import agents.eval.metrics_ragas as R
from core.schema import Probe
from agents.eval.types import EvalRecord

judge = R._judge()
assert judge is not None, "심판 LLM 로드 실패"

# ── (선택) LLM 입출력 로깅 spy — 모듈은 안 건드리고 여기서 _chat 만 감싼다 ──
if os.getenv("EVAL_DEBUG"):
    _orig_chat = R._chat

    def _spy(jg, prompt):
        out = _orig_chat(jg, prompt)
        head = prompt.split("\n", 1)[0][:70]
        print(f"    · LLM← {head}...\n      LLM→ {str(out)[:160]}")
        return out

    R._chat = _spy


def show(name, val, note=""):
    print(f"  {name:26s} = {val}   {note}")


print("=" * 60)
print("ragas_eval 실측 테스트 (실제 API)")
print("=" * 60)

# ── 1) Faithfulness: 할루시네이션이 감점되는가 ──────────────────
CTX = ["Albert Einstein (born 14 March 1879) was a German-born physicist."]
print("\n[1] Faithfulness")
faith_bad = R._faithfulness(judge, "When and where was Einstein born?",
                            "He was born in Germany on 20 March 1879.", CTX)   # 날짜 지어냄
faith_ok = R._faithfulness(judge, "When and where was Einstein born?",
                           "He was born in Germany.", CTX)                     # 근거 그대로
show("faithfulness(날짜 지어냄)", faith_bad, "기대: < 1.0")
show("faithfulness(정확)", faith_ok, "기대: ~1.0, 위보다 높거나 같음")
assert faith_bad is not None and faith_bad < 1.0, "지어낸 날짜가 감점돼야 함"
assert faith_ok is None or faith_ok >= faith_bad

# ── 2) Response Relevancy: 동문서답/모른다 감지 ────────────────
print("\n[2] Response Relevancy")
rel_good = R._response_relevancy(judge, "Where is the Eiffel Tower located?",
                                 "The Eiffel Tower is located in Paris, France.")
rel_none = R._response_relevancy(judge, "Where is the Eiffel Tower located?",
                                 "I don't know.")
show("relevancy(정상 답변)", rel_good, "기대: 높음(>0.5)")
show("relevancy('모른다')", rel_none, "기대: 0.0 (noncommittal)")
assert rel_good is not None and rel_good > 0.5
assert rel_none == 0.0, "회피성 답변은 0 이어야 함"

# ── 3) Context Precision: 관련 청크가 앞에 있을수록 높은가 ──────
print("\n[3] Context Precision (순위 가중)")
REF = "The Eiffel Tower is in Paris."
prec_first = R._context_precision(judge, "Where is the Eiffel Tower?", REF,
    ["The Eiffel Tower is in Paris.", "The Brandenburg Gate is in Berlin."])
prec_last = R._context_precision(judge, "Where is the Eiffel Tower?", REF,
    ["The Brandenburg Gate is in Berlin.", "The Eiffel Tower is in Paris."])
show("precision(관련 청크 1위)", prec_first, "기대: ~1.0")
show("precision(관련 청크 2위)", prec_last, "기대: ~0.5 (더 낮음)")
assert prec_first is not None and prec_last is not None and prec_first >= prec_last

# ── 4) Context Recall: 정답이 컨텍스트로 뒷받침되는가 ──────────
print("\n[4] Context Recall")
recall = R._context_recall(judge, "Where is the Eiffel Tower?",
    "The Eiffel Tower is in Paris.",
    ["Paris is the capital of France and home to the Eiffel Tower."])
show("context_recall", recall, "기대: 높음(1.0 근처)")
assert recall is not None and recall > 0.5

# ── 5) 트랙 함수 엔드투엔드 (실제+오라클+aspect) ──────────────
print("\n[5] 트랙 함수 엔드투엔드")
probe = Probe(probe_id="t0", question="Where is the Eiffel Tower located?",
              source="llm_generated", ground_truth="The Eiffel Tower is in Paris.",
              gold_chunk_ids=["c0"])
rec = EvalRecord(probe=probe)
rec.retrieved_context = ["The Eiffel Tower is in Paris.", "Berlin is in Germany."]
rec.generated_answer = "The Eiffel Tower is in Paris."
rec.oracle_context = ["The Eiffel Tower is in Paris."]
rec.oracle_answer = "The Eiffel Tower is in Paris."
rec.ragas = R.evaluate_real_track(rec, judge)
rec.oracle_ragas = R.evaluate_oracle_track(rec, judge) if rec.oracle_answer is not None else {}
rec.aspect = R.evaluate_aspect_critics(rec, judge)
print(f"  ragas        = { {k: round(v, 3) for k, v in rec.ragas.items()} }")
print(f"  oracle_ragas = { {k: round(v, 3) for k, v in rec.oracle_ragas.items()} }")
print(f"  aspect       = {rec.aspect}")
assert rec.ragas, "트랙 함수가 ragas 를 채워야 함"

print("\n실측 테스트 통과 [OK]")
