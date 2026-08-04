"""
tools/compare_ragas_fused.py
fused(트랙당 chat 1회) vs legacy(지표별 개별 호출) RAGAS 점수 비교.

파이프라인을 타지 않는다 — 질문/답변/컨텍스트/정답을 고정해두고 같은 record 에
evaluate_real_track 을 두 경로로 각각 부른다. 검색·답변생성이 끼지 않으므로 두 결과의
차이는 오직 채점 경로 하나에서 나온다(파이프라인을 두 번 돌리면 답변이 매번 달라져
그 흔들림과 경로 차이가 섞인다).

실행:
    python tools/compare_ragas_fused.py            # 내장 케이스 4건
    python tools/compare_ragas_fused.py --repeat 3 # 케이스당 3회 (판정 흔들림 폭 확인)
    python tools/compare_ragas_fused.py --case 2   # 특정 케이스만

읽는 법:
    Δ 는 fused − legacy. 지표 스케일이 0~1 이므로 |Δ|<0.1 이면 판정 흔들림 수준이고,
    그보다 크게, 그리고 여러 케이스에서 **한 방향으로** 쏠리면 그 블록의 지시문 문제다.
    LLM 은 temperature=0 이어도 완전 결정적이지 않으니 --repeat 로 legacy 자기 자신의
    흔들림 폭을 먼저 보고, Δ 가 그보다 큰지로 판단할 것.

※ 실제 API 호출이라 비용이 든다. 케이스 1건당 fused 1콜 + legacy 12콜 수준.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["EVAL_ENABLE_LLM"] = "1"

import agents.eval.metrics_ragas as R          # noqa: E402
from agents.eval.llm_provider import _provider  # noqa: E402
from agents.eval.types import EvalRecord       # noqa: E402
from core.schema import Probe                  # noqa: E402

# 활성 provider 기준 키 확인. has_key() 는 .env.example 의 placeholder("sk-...")도 통과시켜서
# 401 스택트레이스로 죽는다 — test_ragas_eval.py 와 같은 기준으로 미리 걸러 안내한다.
_PROVIDER = _provider()
_KEY_ENV = {"gemini": "GEMINI_API_KEY", "github": "GITHUB_TOKEN"}.get(_PROVIDER, "OPENAI_API_KEY")


def _real_key() -> bool:
    key = os.getenv(_KEY_ENV, "").strip()
    return bool(key) and "..." not in key and len(key) >= 20


# ── 비교 케이스 ────────────────────────────────────────────────
#   지표가 서로 다른 방향으로 움직이는 상황을 하나씩 담는다. 전부 만점인 케이스만 보면
#   두 경로가 똑같이 1.0 을 뱉어 '차이 없음'이 공짜로 나온다.
CASES = [
    {
        "name": "정상 (전 지표 높음)",
        "question": "Where is the Eiffel Tower located?",
        "answer": "The Eiffel Tower is located in Paris, France.",
        "contexts": ["The Eiffel Tower is a landmark in Paris, France.",
                     "The Brandenburg Gate is in Berlin, Germany."],
        "ground_truth": "The Eiffel Tower is in Paris, France.",
    },
    {
        "name": "할루시네이션 (faithfulness 감점)",
        "question": "When and where was Einstein born?",
        "answer": "Einstein was born in Germany on 20 March 1879, in the city of Munich.",
        "contexts": ["Albert Einstein (born 14 March 1879) was a German-born physicist."],
        "ground_truth": "Albert Einstein was born in Germany on 14 March 1879.",
    },
    {
        "name": "관련 청크가 뒤 (precision 감점)",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "contexts": ["Berlin is the capital of Germany.",
                     "Tokyo is the capital of Japan.",
                     "Paris is the capital of France."],
        "ground_truth": "Paris is the capital of France.",
    },
    {
        "name": "부분 답변 (correctness FN 발생)",
        "question": "What are the primary colors of light?",
        "answer": "One of the primary colors of light is red.",
        "contexts": ["The primary colors of light are red, green, and blue.",
                     "Mixing them in equal parts produces white light."],
        "ground_truth": "The primary colors of light are red, green, and blue.",
    },
]

METRICS = ["faithfulness", "response_relevancy", "context_precision",
           "context_recall", "answer_correctness"]


def _record(case: dict) -> EvalRecord:
    rec = EvalRecord(probe=Probe(probe_id="cmp", question=case["question"],
                                 source="llm_generated", ground_truth=case["ground_truth"]))
    rec.retrieved_context = list(case["contexts"])
    rec.generated_answer = case["answer"]
    return rec


def _score(case: dict, judge, fused: bool) -> tuple[dict, int, int]:
    """한 경로로 채점 → (점수 dict, chat 호출수, embed 호출수). 호출수는 _chat/_embed 를
    잠깐 감싸 센다(모듈 코드는 건드리지 않는다)."""
    os.environ["EVAL_RAGAS_FUSED"] = "1" if fused else "0"
    orig_chat, orig_embed = R._chat, R._embed
    counts = [0, 0]

    def chat(jg, prompt, max_output_tokens=None):
        counts[0] += 1
        return orig_chat(jg, prompt, max_output_tokens)

    def embed(jg, texts):
        counts[1] += 1
        return orig_embed(jg, texts)

    R._chat, R._embed = chat, embed
    try:
        scores = R.evaluate_real_track(_record(case), judge)
    finally:
        R._chat, R._embed = orig_chat, orig_embed
    return scores, counts[0], counts[1]


def _fmt(v):
    return "  --  " if v is None else f"{v:6.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="케이스당 반복 횟수(판정 흔들림 확인)")
    ap.add_argument("--case", type=int, action="append", help="케이스 번호(1-base). 여러 번 지정 가능")
    args = ap.parse_args()

    judge = R._judge()
    if judge is None or not _real_key():
        print(f"실제 {_KEY_ENV} 없음(비었거나 placeholder) → 중단  [provider={_PROVIDER}]")
        print(f"  .env 의 {_KEY_ENV} 를 진짜 키로 바꾸고 다시 실행하세요.")
        return 1

    cases = [CASES[i - 1] for i in args.case] if args.case else CASES
    deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    total_calls = {"fused": [0, 0], "legacy": [0, 0]}

    for n, case in enumerate(cases, start=1):
        for r in range(args.repeat):
            tag = f"[{n}] {case['name']}" + (f"  (#{r + 1})" if args.repeat > 1 else "")
            print("\n" + "=" * 72)
            print(tag)
            print("=" * 72)
            fu, fc, fe = _score(case, judge, fused=True)
            le, lc, le_ = _score(case, judge, fused=False)
            total_calls["fused"][0] += fc
            total_calls["fused"][1] += fe
            total_calls["legacy"][0] += lc
            total_calls["legacy"][1] += le_

            print(f"  {'지표':<22}{'fused':>8}{'legacy':>9}{'diff':>9}")
            for m in METRICS:
                a, b = fu.get(m), le.get(m)
                d = f"{a - b:+7.3f}" if (a is not None and b is not None) else "   --  "
                if a is not None and b is not None:
                    deltas[m].append(a - b)
                print(f"  {m:<22}{_fmt(a):>8}{_fmt(b):>9}{d:>9}")
            fn_f, fn_l = fu.get("answer_correctness_fn"), le.get("answer_correctness_fn")
            print(f"  {'FN(누락 요소 수)':<22}{str(fn_f):>8}{str(fn_l):>9}")
            print(f"  호출  fused chat {fc} + embed {fe}   /   legacy chat {lc} + embed {le_}")

    print("\n" + "=" * 72)
    print("요약 (diff = fused - legacy)")
    print("=" * 72)
    for m in METRICS:
        ds = deltas[m]
        if not ds:
            print(f"  {m:<22} 비교 불가(한쪽 미측정)")
            continue
        mean = sum(ds) / len(ds)
        print(f"  {m:<22} 평균 {mean:+.3f}   최대 |Δ| {max(abs(d) for d in ds):.3f}   n={len(ds)}")
    fc, fe = total_calls["fused"]
    lc, le_ = total_calls["legacy"]
    saved = f"   → chat {(1 - fc / lc) * 100:.0f}% 절감" if lc else ""
    print(f"\n  총 호출  fused chat {fc} + embed {fe}   /   legacy chat {lc} + embed {le_}{saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
