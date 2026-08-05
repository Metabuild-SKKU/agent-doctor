"""임베딩 처리량 벤치마크 — 로컬 bge-m3(CPU) vs OpenRouter 임베딩 API.

색인 임베딩을 OpenRouter 로 옮길지 판단하는 데 필요한 세 가지를 실측한다.

  1) 처리량   : 로컬 batch_size 별 chunks/sec, API 는 동시 요청 수를 올리며 chunks/sec
  2) 동시성 한도: 429 가 처음 나는 동시 요청 수 (OpenRouter 는 문서로 공개하지 않아
                 부딪혀 봐야 안다 — /api/v1/key 로는 크레딧만 보이고 concurrency 는 안 나온다)
  3) 벡터 호환성: 같은 텍스트를 로컬/API 로 각각 임베딩해 코사인 비교. 1.0 에 가깝지
                 않으면 "색인은 API, 질의는 로컬" 같은 혼용은 불가능하다.

비용은 추정하지 않고 OpenRouter 가 응답 usage 에 실어 보내는 실제 과금액을 읽는다
(core.llm_clients.openrouter_reported_cost 와 같은 경로). --max-cost 를 넘으면 즉시 멈춘다.

사용 예:
    # 비용 없이 계획만 확인
    python -m tools.bench_embedding --dry-run

    # 로컬만 (API 호출 0건, 과금 없음)
    python -m tools.bench_embedding --no-api

    # 전체 (기본 1,000청크 × 5라운드 ≈ $0.04)
    python -m tools.bench_embedding --corpus data/corpus.txt

주의: 이 스크립트는 .env 의 진짜 키로 실제 과금 호출을 한다. 기본값은 실행 전
      예상 비용을 보여주고 확인을 받는다(--yes 로 생략).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 한글 Windows 콘솔(cp949)에서는 출력의 '—' 하나에 UnicodeEncodeError 로 죽는다.
# 계획 출력에서 죽으면 그나마 낫지만, 같은 문자가 report() 에도 있어 API 호출을
# 모두 끝낸 뒤 결과를 찍는 순간 죽는다 — 과금은 하고 측정값은 잃는다.
from core.console import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)   # 다른 엔트리포인트(run_local_pipeline 등)와 같은 규약
except ImportError:
    pass                         # python-dotenv 없으면 셸 환경변수로만 동작

from core.llm_clients import OPENROUTER_BASE_URL, openrouter_reported_cost  # noqa: E402

DEFAULT_API_MODEL = "baai/bge-m3"      # OpenRouter 철자. 로컬은 "BAAI/bge-m3"
DEFAULT_LOCAL_MODEL = "BAAI/bge-m3"

# 합성 코퍼스용 문장 풀. 실제 코퍼스(--corpus)가 있으면 쓰지 않는다.
_KO_POOL = [
    "검색 증강 생성은 외부 문서를 근거로 답변을 만들어 환각을 줄이는 기법이다.",
    "청크 크기와 오버랩은 검색 품질과 색인 비용을 동시에 좌우하는 핵심 파라미터다.",
    "임베딩 모델의 차원이 달라지면 벡터 컬렉션을 다시 만들어야 한다.",
    "재순위 모델은 1차 검색 결과를 다시 정렬해 상위 정밀도를 끌어올린다.",
    "평가 지표가 검색기와 같은 모델을 쓰면 실패가 상관되어 진단이 무뎌진다.",
]
_EN_POOL = [
    "Retrieval augmented generation grounds answers in retrieved documents.",
    "Chunk size and overlap jointly determine recall and indexing cost.",
    "Changing the embedding dimension forces a full collection rebuild.",
    "A reranker reorders first stage results to improve top-k precision.",
    "Sharing a model between retriever and judge correlates their failures.",
]


# ---------------------------------------------------------------- 코퍼스 준비

def build_chunks(args: argparse.Namespace) -> list[str]:
    """벤치마크에 쓸 청크 리스트. 실제 코퍼스가 있으면 그걸 자르고, 없으면 합성한다."""
    if args.corpus:
        with open(args.corpus, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        step = args.chunk_size - args.chunk_overlap
        chunks = [
            text[i : i + args.chunk_size]
            for i in range(0, max(len(text) - args.chunk_overlap, 1), step)
        ]
        chunks = [c for c in chunks if c.strip()][: args.chunks]
        if not chunks:
            raise SystemExit(f"코퍼스 '{args.corpus}' 에서 청크를 못 만들었다 (내용이 비었나?).")
        if len(chunks) < args.chunks:
            print(f"[!] 코퍼스가 짧아 {len(chunks)}청크만 확보 (요청 {args.chunks}).")
        return chunks

    # 합성: 같은 문자열이 반복되면 provider 캐시에 걸려 처리량이 부풀 수 있으므로
    # 청크마다 고유 번호를 심고 문장 순서를 섞는다.
    pool = _KO_POOL if args.lang == "ko" else _EN_POOL
    rng = random.Random(args.seed)
    out = []
    for i in range(args.chunks):
        buf = [f"[{i:06d}]"]
        while sum(len(s) for s in buf) < args.chunk_size:
            buf.append(rng.choice(pool))
        out.append(" ".join(buf)[: args.chunk_size])
    return out


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------- 로컬 측정

@dataclass
class LocalResult:
    batch_size: int
    seconds: float
    chunks: int

    @property
    def cps(self) -> float:
        return self.chunks / self.seconds if self.seconds else 0.0


def run_local(chunks: list[str], batch_sizes: list[int], model_name: str) -> list[LocalResult]:
    """로컬 sentence-transformers 처리량. 모델 로드 시간은 워밍업으로 분리한다."""
    from agents.index.qdrant_store import embed_batch, embedding_is_fallback

    print(f"\n[로컬] 모델 로드 중: {model_name} …")
    t0 = time.perf_counter()
    # provider 를 못 박는다 — 기본값이 openrouter 라 그냥 부르면 "로컬 측정" 이
    # 그대로 API 호출이 되어 비교 자체가 무의미해진다.
    warm = embed_batch(chunks[:2], model_name=model_name, batch_size=2,
                       provider="local")
    load_sec = time.perf_counter() - t0

    # ★ 중요: 모델 로드에 실패하면 embed_batch 는 조용히 해시 fallback 벡터를 낸다.
    #   그건 행렬곱을 아예 안 하므로 "로컬이 API 보다 100배 빠르다"는 완전히 틀린
    #   결과가 나온다. 반드시 여기서 걸러야 벤치마크가 의미를 갖는다.
    if embedding_is_fallback(model_name, provider="local"):
        raise SystemExit(
            f"[로컬] 모델 '{model_name}' 로드 실패 → 해시 fallback 임베딩 상태다.\n"
            "       이 상태의 측정치는 무의미하므로 중단한다. sentence-transformers 설치와\n"
            "       모델 다운로드(약 2.2GB)를 먼저 확인할 것."
        )
    print(f"[로컬] 로드 완료 ({load_sec:.1f}초, dim={len(warm[0])}) — 아래 측정에서 제외됨")

    results = []
    for bs in batch_sizes:
        t0 = time.perf_counter()
        embed_batch(chunks, model_name=model_name, batch_size=bs,
                    provider="local")
        dt = time.perf_counter() - t0
        r = LocalResult(bs, dt, len(chunks))
        results.append(r)
        print(f"  batch={bs:<4} {dt:7.1f}초  {r.cps:8.1f} chunks/sec")
    return results


# ---------------------------------------------------------------- API 측정

@dataclass
class ApiResult:
    concurrency: int
    seconds: float
    ok_chunks: int
    requests_ok: int
    rate_limited: int = 0
    errors: list[str] = field(default_factory=list)
    tokens: int = 0
    cost_usd: float | None = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def cps(self) -> float:
        return self.ok_chunks / self.seconds if self.seconds else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


def _make_client(keepalive: int = 0):
    """keepalive>0 이면 커넥션 풀을 그만큼 키운 http_client 를 쓴다.

    OpenAI SDK 기본값은 max_keepalive_connections=100 이라, 동시 요청을 그보다 높이면
    초과분이 매 요청 TLS 핸드셰이크를 새로 한다. 그러면 처리량이 떨어지는데 원인이
    서버 한도가 아니라 클라이언트 풀이므로, 한도를 재려면 이걸 먼저 올려야 한다.
    """
    from openai import OpenAI

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY 가 없다. .env 를 확인할 것.")
    http_client = None
    if keepalive:
        import httpx

        http_client = httpx.Client(
            limits=httpx.Limits(max_connections=max(keepalive * 2, 1000),
                                max_keepalive_connections=keepalive)
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=0,
                  http_client=http_client)


def _one_request(client, model: str, texts: list[str]) -> dict:
    """임베딩 요청 1건. 재시도하지 않는다 — 429 지점을 찾는 게 목적이라 삼키면 안 된다."""
    import openai

    t0 = time.perf_counter()
    try:
        resp = client.embeddings.create(
            model=model,
            input=texts,
            # OpenRouter 는 usage.include 를 주면 응답 usage 에 실제 과금액(cost)을 싣는다.
            # 단가표 추정과 달리 모델을 바꿔도 값이 틀릴 일이 없다.
            extra_body={"usage": {"include": True}},
        )
    except openai.RateLimitError:
        return {"status": "429", "latency": time.perf_counter() - t0}
    except Exception as exc:                      # noqa: BLE001 — 어떤 실패든 라운드는 계속
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}",
                "latency": time.perf_counter() - t0}

    usage = getattr(resp, "usage", None)
    return {
        "status": "ok",
        "latency": time.perf_counter() - t0,
        "n": len(resp.data),
        "dim": len(resp.data[0].embedding) if resp.data else 0,
        "tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "cost": openrouter_reported_cost(usage) if usage else None,
    }


def run_api_round(client, model: str, chunks: list[str], batch: int, concurrency: int) -> ApiResult:
    batches = batched(chunks, batch)
    res = ApiResult(concurrency=concurrency, seconds=0.0, ok_chunks=0, requests_ok=0)
    cost_unknown = False

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, client, model, b) for b in batches]
        for fut in as_completed(futures):
            r = fut.result()
            res.latencies.append(r["latency"])
            if r["status"] == "429":
                res.rate_limited += 1
            elif r["status"] == "error":
                res.errors.append(r["detail"])
            else:
                res.requests_ok += 1
                res.ok_chunks += r["n"]
                res.tokens += r["tokens"]
                if r["cost"] is None:
                    cost_unknown = True
                elif res.cost_usd is not None:
                    res.cost_usd += r["cost"]
    res.seconds = time.perf_counter() - t0
    if cost_unknown:
        res.cost_usd = None            # $0 으로 뭉개지 않고 "모름"으로 드러낸다
    return res


# ---------------------------------------------------------------- 벡터 호환성

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def compare_vectors(client, api_model: str, local_model: str, texts: list[str]) -> None:
    """같은 텍스트의 로컬 벡터 vs API 벡터. 색인/질의 경로를 섞어도 되는지의 근거."""
    from agents.index.qdrant_store import embed_batch

    from agents.index.qdrant_store import embedding_is_fallback

    print(f"\n[벡터 호환성] 같은 텍스트 {len(texts)}건을 양쪽에서 임베딩해 비교")
    local_vecs = embed_batch(texts, model_name=local_model,
                             batch_size=len(texts), provider="local")
    if embedding_is_fallback(local_model):
        print(f"  로컬 '{local_model}' 로드 실패(해시 fallback) — 비교는 무의미하므로 생략")
        return
    try:
        resp = client.embeddings.create(model=api_model, input=texts,
                                        extra_body={"usage": {"include": True}})
    except Exception as exc:                      # noqa: BLE001
        print(f"  API 실패({type(exc).__name__}: {exc}) — 비교 생략")
        return
    api_vecs = [d.embedding for d in resp.data]

    if len(api_vecs[0]) != len(local_vecs[0]):
        print(f"  ✗ 차원 불일치: 로컬 {len(local_vecs[0])} vs API {len(api_vecs[0])}")
        print("    → 혼용 불가. 컬렉션 재생성이 필요하다.")
        return

    sims = [_cosine(l, a) for l, a in zip(local_vecs, api_vecs)]
    lo, mean = min(sims), sum(sims) / len(sims)
    print(f"  차원 {len(api_vecs[0])}  코사인 평균 {mean:.5f}  최저 {lo:.5f}")
    if lo >= 0.999:
        print("  → 사실상 동일. 색인/질의 경로를 섞어도 순위가 흔들릴 위험은 낮다.")
    elif lo >= 0.99:
        print("  → 미세하게 다르다. 혼용은 권하지 않지만 한쪽으로 통일하면 문제없다.")
    else:
        print("  ✗ 유의미하게 다르다. 경로를 섞으면 검색 품질이 무너진다 — 반드시 통일할 것.")


# ---------------------------------------------------------------- 리포트

def report(local: list[LocalResult], api: list[ApiResult], args, chunks: list[str]) -> None:
    print("\n" + "=" * 72)
    print("결과 요약")
    print("=" * 72)

    best_local = max(local, key=lambda r: r.cps) if local else None
    best_api = max((r for r in api if r.ok_chunks), key=lambda r: r.cps, default=None)

    if local:
        print("\n[로컬 CPU]")
        for r in local:
            print(f"  batch={r.batch_size:<4} {r.cps:8.1f} chunks/sec")

    if api:
        print("\n[OpenRouter API]")
        print(f"  {'동시':>4} {'chunks/sec':>11} {'p50 지연':>9} {'429':>5} {'에러':>5} {'비용':>10}")
        for r in api:
            cost = "모름" if r.cost_usd is None else f"${r.cost_usd:.5f}"
            print(f"  {r.concurrency:>4} {r.cps:>11.1f} {r.p50:>8.2f}s "
                  f"{r.rate_limited:>5} {len(r.errors):>5} {cost:>10}")
        first_429 = next((r.concurrency for r in api if r.rate_limited), None)
        if first_429:
            print(f"\n  → 429 최초 발생: 동시 {first_429}. 실사용은 그 아래로 잡을 것.")
        else:
            print(f"\n  → 시험한 최대 동시 {api[-1].concurrency} 까지 429 없음. "
                  "더 올려서 한도를 찾으려면 --concurrency 확장.")
        # 동시성을 올려도 처리량이 안 늘면 어딘가가 상한이다 — 다만 이 측정만으로는
        # 서버 큐인지 클라이언트 커넥션 풀인지 구분할 수 없으므로 단정하지 않는다.
        if len(api) >= 2 and best_api and best_api.concurrency < api[-1].concurrency:
            print(f"  → 처리량 최대는 동시 {best_api.concurrency} 지점 "
                  f"({best_api.cps:.0f} chunks/sec). 그 이상에서는 오히려 떨어진다.")
            if not args.keepalive:
                print("     주의: --keepalive 미지정이라 SDK 기본 풀(100)이 상한일 수 있다. "
                      "동시 100 이상 구간의 열화는 --keepalive 를 올려 재측정할 것.")

    if best_local and best_api:
        print(f"\n[배수] API {best_api.cps:.1f} / 로컬 {best_local.cps:.1f} "
              f"= {best_api.cps / best_local.cps:.1f}배")

    # 실측 chunks/sec 로 목표 코퍼스 색인 시간을 환산
    if args.extrapolate_mb:
        bytes_per_char = 3.0 if args.lang == "ko" else 1.0
        total_chars = args.extrapolate_mb * 1024 * 1024 / bytes_per_char
        n_chunks = total_chars / (args.chunk_size - args.chunk_overlap)
        print(f"\n[환산] {args.extrapolate_mb}MB ({args.lang}) ≈ {n_chunks:,.0f}청크 색인 시")
        for label, r in (("로컬 CPU", best_local), ("OpenRouter", best_api)):
            if not r:
                continue
            sec = n_chunks / r.cps
            unit = f"{sec/3600:.1f}시간" if sec >= 3600 else f"{sec/60:.1f}분"
            print(f"  {label:<12} {unit}")
        if best_api and api:
            known = [r.cost_usd for r in api if r.cost_usd is not None]
            measured_chunks = sum(r.ok_chunks for r in api if r.cost_usd is not None)
            if known and measured_chunks:
                per_chunk = sum(known) / measured_chunks
                print(f"  예상 비용     ${per_chunk * n_chunks:.2f} (실측 단가 기준)")


# ---------------------------------------------------------------- main

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="로컬 bge-m3(CPU) vs OpenRouter 임베딩 처리량 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--corpus", help="실제 텍스트 파일. 없으면 합성 코퍼스 사용")
    p.add_argument("--chunks", type=int, default=1000, help="측정에 쓸 청크 수 (기본 1000)")
    p.add_argument("--chunk-size", type=int, default=600, help="청크 문자 수 (기본 600)")
    p.add_argument("--chunk-overlap", type=int, default=50, help="오버랩 (기본 50)")
    p.add_argument("--lang", choices=["ko", "en"], default="ko",
                   help="합성 코퍼스 언어 및 환산 시 바이트/문자 가정")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--no-local", dest="local", action="store_false", help="로컬 측정 생략")
    p.add_argument("--no-api", dest="api", action="store_false",
                   help="API 측정 생략 (과금 0)")
    p.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    p.add_argument("--api-model", default=DEFAULT_API_MODEL)
    p.add_argument("--local-batch", default="8,32,128",
                   help="로컬 batch_size 목록 (기본 8,32,128)")
    p.add_argument("--batch", type=int, default=64,
                   help="API 요청당 텍스트 수 (기본 64). RPM 한도가 상한이면 이걸 먼저 올릴 것")
    p.add_argument("--concurrency", default="1,2,4,8,16",
                   help="API 동시 요청 수 램프 (기본 1,2,4,8,16)")

    p.add_argument("--compare-vectors", type=int, default=8, metavar="N",
                   help="로컬/API 벡터 코사인 비교에 쓸 텍스트 수 (0이면 생략)")
    p.add_argument("--extrapolate-mb", type=float, default=26.0,
                   help="측정 처리량으로 색인 시간을 환산할 코퍼스 크기 (0이면 생략)")
    p.add_argument("--keepalive", type=int, default=0, metavar="N",
                   help="HTTP keep-alive 커넥션 풀 크기. 기본 0=SDK 기본값(100). "
                        "동시 요청을 100 초과로 올릴 때는 반드시 그 이상으로 지정할 것 "
                        "— 안 그러면 클라이언트 병목을 서버 한도로 오독한다")
    p.add_argument("--cooldown", type=float, default=0.0, metavar="SEC",
                   help="라운드 사이 대기(초). 한도가 분당 요청 수(RPM) 기준이면 직전 "
                        "라운드가 같은 창에 남아 다음 라운드 429 로 잘못 잡힌다 — 램프를 "
                        "올릴 때는 15 이상 권장")
    p.add_argument("--max-cost", type=float, default=0.20,
                   help="누적 과금 상한(USD). 넘으면 즉시 중단 (기본 0.20)")
    p.add_argument("--dry-run", action="store_true", help="계획과 예상 비용만 출력하고 종료")
    p.add_argument("-y", "--yes", action="store_true", help="비용 확인 프롬프트 생략")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local_batches = [int(x) for x in args.local_batch.split(",") if x.strip()]
    concurrencies = [int(x) for x in args.concurrency.split(",") if x.strip()]

    chunks = build_chunks(args)
    # 한글은 문자당 약 1.3토큰, 영문은 문자당 약 0.25토큰.
    tok_per_char = 1.3 if args.lang == "ko" else 0.25
    est_tokens = len(chunks) * args.chunk_size * tok_per_char * len(concurrencies)
    est_cost = est_tokens / 1_000_000 * 0.01      # bge-m3 $0.01/M 기준

    print(f"청크 {len(chunks)}개 × {args.chunk_size}자"
          f"{' (합성 ' + args.lang + ')' if not args.corpus else f' (코퍼스: {args.corpus})'}")
    if args.local:
        print(f"로컬  : {args.local_model}, batch {local_batches} — 과금 없음")
    if args.api:
        print(f"API   : {args.api_model}, batch {args.batch}, 동시 {concurrencies}")
        print(f"        요청 {math.ceil(len(chunks)/args.batch) * len(concurrencies)}건, "
              f"약 {est_tokens/1e6:.2f}M 토큰 → 예상 ${est_cost:.3f} (상한 ${args.max_cost:.2f})")

    if args.dry_run:
        print("\n--dry-run: 여기까지. 실제 호출은 하지 않았다.")
        return 0
    if args.api and not args.yes:
        if input("\n실제 과금 호출을 진행할까? [y/N] ").strip().lower() not in ("y", "yes"):
            print("중단했다.")
            return 1

    local_results: list[LocalResult] = []
    if args.local:
        local_results = run_local(chunks, local_batches, args.local_model)

    api_results: list[ApiResult] = []
    if args.api:
        client = _make_client(args.keepalive)
        print(f"\n[API] {args.api_model} — 동시 요청 수를 올리며 429 지점을 찾는다")
        spent = 0.0
        for idx, conc in enumerate(concurrencies):
            if idx and args.cooldown:
                print(f"  … 쿨다운 {args.cooldown:.0f}초 (RPM 창 이월 방지)")
                time.sleep(args.cooldown)
            r = run_api_round(client, args.api_model, chunks, args.batch, conc)
            api_results.append(r)
            cost = "모름" if r.cost_usd is None else f"${r.cost_usd:.5f}"
            print(f"  동시={conc:<3} {r.seconds:6.1f}초  {r.cps:8.1f} chunks/sec  "
                  f"p50={r.p50:.2f}s  429={r.rate_limited}  err={len(r.errors)}  {cost}")
            if r.errors:
                print(f"        첫 에러: {r.errors[0]}")
            spent += r.cost_usd or 0.0
            if spent > args.max_cost:
                print(f"  [중단] 누적 ${spent:.4f} 가 상한 ${args.max_cost:.2f} 초과.")
                break
            if r.rate_limited and r.rate_limited >= len(batched(chunks, args.batch)) // 2:
                print("  [중단] 요청 절반 이상이 429 — 한도를 넘었으므로 램프를 멈춘다.")
                break

        # 로컬 처리량 측정(--no-local)을 건너뛰어도 벡터 비교는 따로 할 수 있다.
        # 그때는 여기서 모델을 처음 로드하므로 수십 초가 더 걸린다.
        if args.compare_vectors:
            compare_vectors(client, args.api_model, args.local_model,
                            chunks[: args.compare_vectors])

    report(local_results, api_results, args, chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
