"""Generate a report-ready KorQuAD corpus visualization without API calls.

The script reads a KorQuAD 2.x JSON file, cleans each article context, creates
local chunks, projects hashed TF-IDF vectors to two dimensions, and writes a
standalone HTML report plus a compact JSON summary.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_OUTPUT_DIR = Path("output/corpus_visualization")
DEFAULT_INPUT_NAME = "korquad2.1_train_02.json"

TOKEN_RE = re.compile(
    r"[\uac00-\ud7a3]{2,}|[A-Za-z][A-Za-z0-9_+-]{1,}|\d+(?:년|월|일|개|명|회|차|위|대)?"
)
STOPWORDS = {
    "그리고",
    "그러나",
    "또한",
    "있는",
    "없는",
    "한다",
    "했다",
    "있다",
    "대한",
    "에서",
    "으로",
    "에게",
    "위키백과",
    "우리",
    "모두",
    "백과사전",
    "문서",
    "편집",
    "보기",
    "분류",
    "원본",
    "주소",
    "같이",
    "도구",
    "메뉴",
}
TERM_STOPWORDS = STOPWORDS | {
    "the",
    "and",
    "for",
    "of",
    "to",
    "in",
    "on",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "있다",
    "있다.",
    "한다",
    "한다.",
    "했다",
    "했다.",
    "되었다",
    "되었다.",
    "확인함",
    "확인함.",
    "가기",
    "링크",
    "외부",
    "각주",
    "목차",
    "문서",
    "대한",
    "그는",
    "그의",
    "이후",
    "것을",
    "것은",
    "것이",
    "것이다",
    "것으로",
    "등을",
    "다른",
    "같은",
    "함께",
    "위해",
    "가장",
    "된다",
    "있었다",
    "부터",
    "하는",
    "따라",
    "의해",
    "다시",
    "통해",
    "자신의",
    "또는",
    "라고",
    "하였다",
    "현재",
    "영어",
    "많은",
    "에는",
    "으로",
    "에서",
    "에게",
    "모두의",
    "둘러보기로",
    "때문에",
    "com",
    "www",
    "http",
    "https",
    "보존된",
    "전거",
    "통제",
    "isbn",
    "lccn",
    "viaf",
    "ebay",
    "chapter",
    "displaystyle",
    "빈칸",
    "검색하러",
}


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    url: str
    text: str
    context_chars: int
    qas: int


@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    token_count: int
    char_count: int
    qa_count: int
    index: int


class _TextExtractor(HTMLParser):
    block_tags = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    skip_tags = {"head", "script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self.block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.skip_tags and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self.block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def main() -> None:
    args = _parse_args()
    input_path = args.input or _find_default_input()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_korquad(input_path)
    docs = _build_documents(raw["data"], limit=args.limit_docs)
    chunks = _build_chunks(docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    data = _build_visualization_data(
        input_path=input_path,
        version=str(raw.get("version", "")),
        docs=docs,
        chunks=chunks,
        max_points=args.max_points,
        vector_dim=args.vector_dim,
    )

    html_path = output_dir / "corpus_visualization.html"
    json_path = output_dir / "corpus_summary.json"
    html_path.write_text(_render_html(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote_html={html_path.resolve()}")
    print(f"wrote_json={json_path.resolve()}")
    print(f"documents={data['summary']['documents']}")
    print(f"qas={data['summary']['qas']}")
    print(f"chunks={data['summary']['chunks']}")
    print(f"scatter_points={data['projection']['point_count']}")
    print(f"clusters={len(data['projection']['clusters'])}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="KorQuAD JSON path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=3500)
    parser.add_argument("--vector-dim", type=int, default=384)
    return parser.parse_args()


def _find_default_input() -> Path:
    roots = [Path.cwd(), Path.cwd().parent, Path.home() / "OneDrive", Path.home()]
    for root in roots:
        if not root.exists():
            continue
        matches = list(root.rglob(DEFAULT_INPUT_NAME))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"{DEFAULT_INPUT_NAME} 파일을 찾지 못했습니다. --input으로 경로를 넘겨주세요."
    )


def _read_korquad(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("data"), list):
        raise ValueError("KorQuAD JSON에서 data 배열을 찾지 못했습니다.")
    return data


def _build_documents(items: list[dict], *, limit: int) -> list[SourceDoc]:
    selected = items[:limit] if limit and limit > 0 else items
    docs: list[SourceDoc] = []
    for index, item in enumerate(selected, start=1):
        title = str(item.get("title") or f"document-{index}")
        context = str(item.get("context") or item.get("raw_html") or "")
        text = _clean_html_context(context)
        if not text:
            continue
        docs.append(
            SourceDoc(
                doc_id=f"korquad-{index:04d}",
                title=title,
                url=str(item.get("url") or ""),
                text=text,
                context_chars=len(context),
                qas=len(item.get("qas") or []),
            )
        )
    return docs


def _clean_html_context(value: str) -> str:
    # KorQuAD contexts are full Wikipedia HTML snapshots. The report should focus
    # on article content, so remove obvious page chrome before extracting text.
    value = re.sub(r"(?is)<head.*?</head>", " ", value)
    value = re.split(r"(?is)<h2[^>]*>\s*둘러보기 메뉴\s*</h2>|원본 주소", value, maxsplit=1)[0]
    value = re.sub(r"(?is)<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>", " ", value)

    parser = _TextExtractor()
    parser.feed(value)
    text = html.unescape(parser.text())
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\b편집\b", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_chunks(docs: list[SourceDoc], *, chunk_size: int, overlap: int) -> list[ChunkRow]:
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")

    chunks: list[ChunkRow] = []
    for doc in docs:
        for chunk_index, text in enumerate(_chunk_text(doc.text, chunk_size, overlap), start=1):
            token_count = _count_tokens(text)
            if token_count < 5:
                continue
            chunks.append(
                ChunkRow(
                    chunk_id=f"{doc.doc_id}-chunk-{chunk_index:04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    url=doc.url,
                    text=text,
                    token_count=token_count,
                    char_count=len(text),
                    qa_count=doc.qas,
                    index=len(chunks),
                )
            )
    return chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if len(text) <= chunk_size:
        yield text
        return

    start = 0
    while start < len(text):
        hard_end = min(len(text), start + chunk_size)
        end = hard_end
        if hard_end < len(text):
            end = _preferred_boundary(text, start, hard_end)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
        while start < len(text) and text[start].isspace():
            start += 1


def _preferred_boundary(text: str, start: int, hard_end: int) -> int:
    minimum = start + max(1, (hard_end - start) // 2)
    separators = ("\n\n", "\n", "다. ", "요. ", ". ", "? ", "! ", " ")
    for separator in separators:
        position = text.rfind(separator, minimum, hard_end)
        if position >= minimum:
            return position + len(separator)
    return hard_end


def _build_visualization_data(
    *,
    input_path: Path,
    version: str,
    docs: list[SourceDoc],
    chunks: list[ChunkRow],
    max_points: int,
    vector_dim: int,
) -> dict:
    tokens = [chunk.token_count for chunk in chunks]
    chars = [chunk.char_count for chunk in chunks]
    chunks_by_doc = Counter(chunk.doc_id for chunk in chunks)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    duplicate_hashes = Counter(_stable_hash(chunk.text) for chunk in chunks)
    duplicate_chunks = sum(count - 1 for count in duplicate_hashes.values() if count > 1)
    empty_docs = sum(1 for doc in docs if not doc.text.strip())

    sampled_chunks = _sample_evenly(chunks, max_points)
    projection = _project_chunks(sampled_chunks, vector_dim=vector_dim)

    doc_rows = [
        {
            "doc_id": doc_id,
            "title": docs_by_id[doc_id].title,
            "url": docs_by_id[doc_id].url,
            "chunks": count,
            "qas": docs_by_id[doc_id].qas,
            "chars": len(docs_by_id[doc_id].text),
        }
        for doc_id, count in chunks_by_doc.items()
    ]
    doc_rows.sort(key=lambda row: (-row["chunks"], row["title"]))

    summary = {
        "source_file": str(input_path),
        "version": version,
        "documents": len(docs),
        "qas": sum(doc.qas for doc in docs),
        "contexts_with_qas": sum(1 for doc in docs if doc.qas),
        "empty_after_cleaning": empty_docs,
        "chunks": len(chunks),
        "sampled_points": len(sampled_chunks),
        "total_context_chars": sum(doc.context_chars for doc in docs),
        "total_clean_chars": sum(len(doc.text) for doc in docs),
        "avg_clean_chars_per_doc": _round(_mean([len(doc.text) for doc in docs])),
        "avg_chunks_per_doc": _round(_mean(list(chunks_by_doc.values()))),
        "max_chunks_in_doc": max(chunks_by_doc.values(), default=0),
        "avg_tokens_per_chunk": _round(_mean(tokens)),
        "median_tokens_per_chunk": _round(_median(tokens)),
        "p90_tokens_per_chunk": _round(_percentile(tokens, 90)),
        "min_tokens_per_chunk": min(tokens, default=0),
        "max_tokens_per_chunk": max(tokens, default=0),
        "avg_chars_per_chunk": _round(_mean(chars)),
        "short_chunks": sum(1 for value in tokens if value < 40),
        "long_chunks": sum(1 for value in tokens if value > 180),
        "duplicate_chunks": duplicate_chunks,
        "duplicate_ratio": _round(duplicate_chunks / len(chunks), 4) if chunks else 0,
        "projection_method": "local hashed TF-IDF + SVD",
        "axis_note": "X/Y축은 의미가 붙은 축이 아니라 고차원 텍스트 벡터를 2차원으로 압축한 좌표입니다.",
    }

    return {
        "summary": summary,
        "documents": doc_rows[:25],
        "token_bins": _histogram(tokens, bins=12),
        "chunk_doc_bins": _histogram(list(chunks_by_doc.values()), bins=10),
        "qa_bins": _qa_bins(docs),
        "top_terms": _top_terms(chunk.text for chunk in chunks),
        "projection": projection,
    }


def _project_chunks(chunks: list[ChunkRow], *, vector_dim: int) -> dict:
    if not chunks:
        return {
            "method": "none",
            "point_count": 0,
            "explained_variance": [0, 0],
            "points": [],
            "clusters": [],
        }

    tokenized = [_filtered_tokens(chunk.text) for chunk in chunks]
    vectors = _hashed_tfidf(tokenized, dim=vector_dim)
    coords, explained = _svd_2d(vectors)
    normalized = _normalize_coords(coords)
    labels = _kmeans(normalized, _cluster_count(len(chunks)))

    points = []
    cluster_terms: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_docs: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_tokens: dict[int, list[int]] = defaultdict(list)

    for chunk, tokens, coord, label in zip(chunks, tokenized, normalized, labels):
        cluster = int(label) + 1
        cluster_terms[cluster].update(tokens)
        cluster_docs[cluster][chunk.title] += 1
        cluster_tokens[cluster].append(chunk.token_count)
        points.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "url": chunk.url,
                "tokens": chunk.token_count,
                "chars": chunk.char_count,
                "qa_count": chunk.qa_count,
                "x": _round(float(coord[0]), 4),
                "y": _round(float(coord[1]), 4),
                "cluster": cluster,
                "preview": _preview(chunk.text),
            }
        )

    clusters = []
    for cluster in sorted(cluster_tokens):
        tokens = cluster_tokens[cluster]
        clusters.append(
            {
                "cluster": cluster,
                "chunks": len(tokens),
                "share": _round(len(tokens) / len(chunks), 4),
                "avg_tokens": _round(_mean(tokens)),
                "top_terms": [term for term, _ in cluster_terms[cluster].most_common(6)],
                "top_documents": [
                    {"title": title, "chunks": count}
                    for title, count in cluster_docs[cluster].most_common(3)
                ],
            }
        )

    return {
        "method": "local hashed TF-IDF + SVD",
        "point_count": len(points),
        "explained_variance": [_round(value, 4) for value in explained],
        "points": points,
        "clusters": clusters,
    }


def _hashed_tfidf(tokenized: list[list[str]], *, dim: int) -> np.ndarray:
    doc_count = len(tokenized)
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    matrix = np.zeros((doc_count, dim), dtype=np.float32)
    for row, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for term, count in counts.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            column = raw % dim
            sign = 1.0 if raw & 1 else -1.0
            tf = 1.0 + math.log(count)
            idf = math.log((1 + doc_count) / (1 + df[term])) + 1.0
            matrix[row, column] += sign * tf * idf
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return matrix / norms[:, None]


def _svd_2d(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:2].T
    variances = singular_values**2
    total = float(variances.sum()) or 1.0
    explained = [(float(value) / total) for value in variances[:2]]
    return coords, explained


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    result = np.zeros_like(coords, dtype=np.float32)
    for axis in range(2):
        values = coords[:, axis]
        low, high = np.percentile(values, [1, 99])
        if math.isclose(float(low), float(high)):
            result[:, axis] = 0
            continue
        clipped = np.clip(values, low, high)
        result[:, axis] = ((clipped - low) / (high - low)) * 2 - 1
    return result


def _kmeans(points: np.ndarray, k: int, *, iterations: int = 30) -> list[int]:
    if len(points) <= 1:
        return [0 for _ in points]
    k = min(k, len(points))
    order = np.argsort(points[:, 0] + points[:, 1])
    centers = points[order[np.linspace(0, len(points) - 1, k, dtype=int)]].copy()
    labels = np.zeros(len(points), dtype=np.int32)

    for _ in range(iterations):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(k):
            mask = labels == cluster
            if np.any(mask):
                centers[cluster] = points[mask].mean(axis=0)
    return labels.tolist()


def _cluster_count(size: int) -> int:
    if size < 20:
        return max(1, min(3, size))
    return min(9, max(4, round(math.sqrt(size) / 7)))


def _sample_evenly(chunks: list[ChunkRow], max_points: int) -> list[ChunkRow]:
    if max_points <= 0 or len(chunks) <= max_points:
        return chunks
    if max_points == 1:
        return [chunks[0]]
    step = (len(chunks) - 1) / (max_points - 1)
    return [chunks[round(index * step)] for index in range(max_points)]


def _histogram(values: list[int], *, bins: int) -> list[dict]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        return [{"label": str(low), "start": low, "end": high, "count": len(values)}]
    width = max(1, math.ceil((high - low + 1) / bins))
    rows = []
    for start in range(low, high + 1, width):
        end = min(high, start + width - 1)
        rows.append(
            {
                "label": f"{start}-{end}",
                "start": start,
                "end": end,
                "count": sum(1 for value in values if start <= value <= end),
            }
        )
    return rows


def _qa_bins(docs: list[SourceDoc]) -> list[dict]:
    counts = Counter(doc.qas for doc in docs)
    return [
        {"label": f"{qa_count}개", "qas": qa_count, "count": count}
        for qa_count, count in sorted(counts.items())
    ]


def _top_terms(texts: Iterable[str], *, limit: int = 16) -> list[dict]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_filtered_tokens(text))
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]


def _filtered_tokens(text: str) -> list[str]:
    terms = []
    for token in _tokens(text):
        normalized = token.strip("._-+").lower()
        if _is_noise_term(normalized):
            continue
        terms.append(normalized)
    return terms


def _is_noise_term(token: str) -> bool:
    if len(token) < 2 or token in TERM_STOPWORDS:
        return True
    if re.fullmatch(r"\d+(?:년|월|일|개|명|회|차|위|대)?", token):
        return True
    if re.fullmatch(r"[a-z]{1,2}", token):
        return True
    return False


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _count_tokens(text: str) -> int:
    return len(_tokens(text))


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _mean(values: list[int] | list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[int] | list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _percentile(values: list[int] | list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percentile / 100)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - index) + ordered[high] * (index - low))


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _format_int(value: int | float) -> str:
    return f"{int(round(value)):,}"


def _format_float(value: int | float, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}"


def _render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    summary = data["summary"]

    stats = "\n".join(
        [
            _stat_card("원본 문서", _format_int(summary["documents"]), f"QA 포함 문서 {_format_int(summary['contexts_with_qas'])}개"),
            _stat_card("평가 QA", _format_int(summary["qas"]), "질문-정답 세트 수"),
            _stat_card("생성 청크", _format_int(summary["chunks"]), f"평균 {_format_float(summary['avg_chunks_per_doc'])}개/문서"),
            _stat_card("산점도 점", _format_int(summary["sampled_points"]), "전체 청크에서 균등 샘플링"),
            _stat_card("평균 청크 길이", _format_float(summary["avg_tokens_per_chunk"]), f"중앙값 {_format_float(summary['median_tokens_per_chunk'])} 토큰"),
            _stat_card("긴 청크", _format_int(summary["long_chunks"]), "180 토큰 초과"),
            _stat_card("짧은 청크", _format_int(summary["short_chunks"]), "40 토큰 미만"),
            _stat_card("완전 중복 청크", _format_int(summary["duplicate_chunks"]), f"{summary['duplicate_ratio'] * 100:.2f}%"),
        ]
    )

    doc_rows = "\n".join(_bar_row(row["title"], row["chunks"], data["documents"][0]["chunks"]) for row in data["documents"][:15])
    token_rows = "\n".join(_bar_row(row["label"], row["count"], max(item["count"] for item in data["token_bins"])) for row in data["token_bins"])
    chunk_doc_rows = "\n".join(_bar_row(row["label"], row["count"], max(item["count"] for item in data["chunk_doc_bins"])) for row in data["chunk_doc_bins"])
    qa_rows = "\n".join(_bar_row(row["label"], row["count"], max(item["count"] for item in data["qa_bins"])) for row in data["qa_bins"])
    term_rows = "\n".join(
        f'<span class="term"><b>{_escape(row["term"])}</b><small>{_format_int(row["count"])}</small></span>'
        for row in data["top_terms"]
    )
    cluster_rows = "\n".join(_cluster_row(row, summary["sampled_points"]) for row in data["projection"]["clusters"])

    return HTML_TEMPLATE.replace("__PAYLOAD__", payload).replace("__STATS__", stats).replace(
        "__DOC_ROWS__", doc_rows
    ).replace("__TOKEN_ROWS__", token_rows).replace("__CHUNK_DOC_ROWS__", chunk_doc_rows).replace(
        "__QA_ROWS__", qa_rows
    ).replace("__TERM_ROWS__", term_rows).replace("__CLUSTER_ROWS__", cluster_rows).replace(
        "__SOURCE_FILE__", _escape(Path(summary["source_file"]).name)
    ).replace(
        "__VERSION__", _escape(summary["version"])
    ).replace(
        "__PROJECTION_METHOD__", _escape(summary["projection_method"])
    ).replace(
        "__EXPLAINED__", ", ".join(f"{value * 100:.1f}%" for value in data["projection"]["explained_variance"])
    )


def _stat_card(label: str, value: str, note: str) -> str:
    return (
        '<article class="stat-card">'
        f'<div class="stat-label">{_escape(label)}</div>'
        f'<div class="stat-value">{_escape(value)}</div>'
        f'<div class="stat-note">{_escape(note)}</div>'
        "</article>"
    )


def _bar_row(label: str, value: int, max_value: int) -> str:
    width = (value / max(max_value, 1)) * 100
    return (
        '<div class="bar-row">'
        f'<div class="bar-label" title="{_escape(label)}">{_escape(label)}</div>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{width:.2f}%"></div>'
        "</div>"
        f'<div class="bar-value">{_format_int(value)}</div>'
        "</div>"
    )


def _cluster_row(row: dict, total: int) -> str:
    terms = ", ".join(row["top_terms"])
    docs = ", ".join(item["title"] for item in row["top_documents"])
    share = row["chunks"] / max(total, 1) * 100
    return (
        "<tr>"
        f"<td>{row['cluster']}</td>"
        f"<td>{_format_int(row['chunks'])}</td>"
        f"<td>{share:.1f}%</td>"
        f"<td>{_format_float(row['avg_tokens'])}</td>"
        f"<td>{_escape(terms)}</td>"
        f"<td>{_escape(docs)}</td>"
        "</tr>"
    )


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Doctor KorQuAD DB 시각화</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #18202f;
      --muted: #647084;
      --line: #d7deea;
      --soft: #eef2f8;
      --blue: #2563eb;
      --green: #079669;
      --orange: #d97706;
      --pink: #cf2777;
      --violet: #7c3aed;
      --cyan: #0787a5;
      --red: #dc2626;
      --shadow: 0 10px 28px rgba(18, 32, 58, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Pretendard, Arial, sans-serif;
      line-height: 1.5;
    }
    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(24px, 3vw, 34px);
      font-weight: 600;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 18px;
      font-weight: 600;
      letter-spacing: 0;
    }
    h3 {
      margin: 0 0 8px;
      font-size: 16px;
      font-weight: 600;
      letter-spacing: 0;
    }
    p { margin: 0; color: var(--muted); }
    code {
      border-radius: 5px;
      background: var(--soft);
      padding: 2px 5px;
      font-family: Consolas, monospace;
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      padding: 6px 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .notice {
      margin: 0 0 18px;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      background: #fff7ed;
      color: #7c2d12;
      padding: 12px 14px;
    }
    .grid {
      display: grid;
      gap: 16px;
    }
    .stats {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .two {
      grid-template-columns: minmax(0, 1.05fr) minmax(340px, 0.95fr);
      align-items: start;
      margin-top: 16px;
    }
    .panel, .stat-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel { padding: 18px; }
    .stat-card { padding: 16px; }
    .stat-label {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 27px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }
    .stat-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(132px, 1fr) minmax(170px, 2fr) 72px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
    }
    .bar-label {
      overflow: hidden;
      color: var(--ink);
      font-size: 13px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .bar-track {
      height: 11px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--soft);
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--blue), var(--cyan));
    }
    .bar-value {
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .length-bars .bar-fill { background: linear-gradient(90deg, var(--green), var(--orange)); }
    .qa-bars .bar-fill { background: linear-gradient(90deg, var(--violet), var(--pink)); }
    .terms {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .term {
      display: inline-flex;
      gap: 7px;
      align-items: baseline;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fafbfe;
      padding: 6px 9px;
      color: var(--ink);
      font-size: 13px;
    }
    .term b { font-weight: 600; }
    .term small {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .scatter-panel { margin-top: 16px; }
    .scatter-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .legend button {
      cursor: pointer;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--ink);
      padding: 6px 10px;
      font: inherit;
      font-size: 13px;
    }
    .legend button[aria-pressed="true"] {
      border-color: var(--ink);
      background: var(--soft);
    }
    .legend .swatch {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: 0;
    }
    .scatter-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      align-items: start;
    }
    .chart-box {
      position: relative;
      min-height: 460px;
    }
    svg {
      display: block;
      width: 100%;
      height: auto;
    }
    .axis, .gridline {
      stroke: var(--line);
      stroke-width: 1;
    }
    .gridline { opacity: 0.75; }
    .axis-label, .tick-label {
      fill: var(--muted);
      font-size: 12px;
    }
    .point {
      cursor: pointer;
      opacity: 0.82;
      stroke: var(--panel);
      stroke-width: 1.2;
    }
    .point:hover, .point.active {
      opacity: 1;
      stroke: var(--ink);
      stroke-width: 2;
    }
    .point.dimmed {
      opacity: 0.08;
      pointer-events: none;
    }
    .detail {
      min-height: 300px;
      border-left: 3px solid var(--line);
      padding-left: 14px;
    }
    .detail p {
      color: var(--ink);
      font-size: 14px;
    }
    .detail dl {
      display: grid;
      grid-template-columns: 90px minmax(0, 1fr);
      gap: 6px 10px;
      margin: 14px 0 0;
      font-size: 13px;
    }
    .detail dt { color: var(--muted); }
    .detail dd {
      margin: 0;
      overflow-wrap: anywhere;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px 6px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
    td:nth-child(1), td:nth-child(2), td:nth-child(3), td:nth-child(4) {
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    details {
      margin-top: 16px;
      color: var(--muted);
    }
    summary {
      cursor: pointer;
      color: var(--ink);
      font-weight: 600;
    }
    @media (max-width: 980px) {
      main { padding: 20px; }
      header, .scatter-head {
        display: grid;
        align-items: start;
      }
      .badge-row, .legend { justify-content: flex-start; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two, .scatter-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      main { padding: 14px; }
      .stats { grid-template-columns: 1fr; }
      .bar-row {
        grid-template-columns: 1fr;
        gap: 5px;
      }
      .bar-value { text-align: left; }
      .chart-box { min-height: auto; }
      .detail { border-left: 0; padding-left: 0; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Agent Doctor KorQuAD DB 시각화</h1>
      <p>KorQuAD의 <code>context</code>를 문서 본문으로 보고, 청크 단위의 길이·분포·군집 상태를 리포트용으로 확인합니다.</p>
    </div>
    <div class="badge-row">
      <span class="badge">데이터: __SOURCE_FILE__</span>
      <span class="badge">버전: __VERSION__</span>
    </div>
  </header>

  <div class="notice">
    이 산점도는 API 비용이 들지 않는 로컬 분석입니다. 실제 Qdrant에 저장된 BAAI/Gemini 임베딩 분포와 완전히 같다고 단정하면 안 되고,
    교수님/멘토님께 보여줄 때는 “본문 청크의 의미적 분포를 빠르게 점검하는 보조 자료”라고 설명하는 편이 안전합니다.
  </div>

  <section class="grid stats">
    __STATS__
  </section>

  <section class="grid two">
    <div class="panel">
      <h2>문서별 청크 수 상위 15개</h2>
      __DOC_ROWS__
    </div>
    <div class="panel length-bars">
      <h2>청크 길이 분포</h2>
      __TOKEN_ROWS__
    </div>
  </section>

  <section class="panel scatter-panel">
    <div class="scatter-head">
      <div>
        <h2>2차원 청크 산점도</h2>
        <p>X축/Y축은 고정된 주제명이 아니라 <code>__PROJECTION_METHOD__</code>로 압축한 좌표입니다. 가까운 점일수록 비슷한 단어 구성을 가진 청크로 해석하면 됩니다.</p>
      </div>
      <div class="legend" id="cluster-legend" aria-label="군집 필터"></div>
    </div>
    <div class="scatter-layout">
      <div class="chart-box">
        <svg id="scatter" viewBox="0 0 820 500" role="img" aria-label="KorQuAD 청크의 2차원 분포 산점도"></svg>
      </div>
      <aside class="detail" id="point-detail">
        <h3>청크를 선택하세요</h3>
        <p>점 하나는 생성된 청크 하나입니다. 색은 군집, 점 크기는 청크 길이를 뜻합니다.</p>
      </aside>
    </div>
  </section>

  <section class="grid two">
    <div class="panel">
      <h2>문서당 청크 수 분포</h2>
      __CHUNK_DOC_ROWS__
    </div>
    <div class="panel qa-bars">
      <h2>문서당 QA 수 분포</h2>
      __QA_ROWS__
    </div>
  </section>

  <section class="grid two">
    <div class="panel">
      <h2>전체 상위 단어</h2>
      <div class="terms">__TERM_ROWS__</div>
    </div>
    <div class="panel">
      <h2>군집별 요약</h2>
      <table>
        <thead>
          <tr>
            <th>군집</th>
            <th>점</th>
            <th>비율</th>
            <th>평균 토큰</th>
            <th>대표 단어</th>
            <th>대표 문서</th>
          </tr>
        </thead>
        <tbody>__CLUSTER_ROWS__</tbody>
      </table>
    </div>
  </section>

  <details>
    <summary>산점도 계산 방식</summary>
    <p>
      청크 텍스트를 로컬 hashed TF-IDF 벡터로 바꾼 뒤 SVD로 2차원에 투영했습니다.
      산점도에 표시한 첫 두 축의 설명분산 비율은 __EXPLAINED__입니다. 축 이름 자체보다 점 사이의 거리와 색으로 표시된 군집 구조를 보는 용도입니다.
    </p>
  </details>
</main>

<script type="application/json" id="corpus-data">__PAYLOAD__</script>
<script>
  const data = JSON.parse(document.getElementById("corpus-data").textContent);
  const points = data.projection.points;
  const clusters = data.projection.clusters;
  const colors = ["#2563eb", "#079669", "#d97706", "#cf2777", "#7c3aed", "#0787a5", "#dc2626", "#6b7280", "#0f766e"];
  const svg = document.getElementById("scatter");
  const detail = document.getElementById("point-detail");
  const legend = document.getElementById("cluster-legend");
  let activeCluster = 0;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function makeSvg(name) {
    return document.createElementNS("http://www.w3.org/2000/svg", name);
  }

  function pointPosition(point) {
    const pad = { left: 58, right: 28, top: 30, bottom: 54 };
    const width = 820;
    const height = 500;
    return {
      x: pad.left + ((point.x + 1) / 2) * (width - pad.left - pad.right),
      y: height - pad.bottom - ((point.y + 1) / 2) * (height - pad.top - pad.bottom)
    };
  }

  function showPoint(point, node) {
    svg.querySelectorAll(".point.active").forEach((item) => item.classList.remove("active"));
    if (node) node.classList.add("active");
    detail.innerHTML = `
      <h3>${escapeHtml(point.title)}</h3>
      <p>${escapeHtml(point.preview)}</p>
      <dl>
        <dt>청크 ID</dt><dd>${escapeHtml(point.chunk_id)}</dd>
        <dt>문서 ID</dt><dd>${escapeHtml(point.doc_id)}</dd>
        <dt>군집</dt><dd>${point.cluster}</dd>
        <dt>토큰</dt><dd>${point.tokens.toLocaleString("ko-KR")}</dd>
        <dt>문자</dt><dd>${point.chars.toLocaleString("ko-KR")}</dd>
        <dt>QA 수</dt><dd>${point.qa_count.toLocaleString("ko-KR")}</dd>
        <dt>URL</dt><dd>${point.url ? `<a href="${escapeHtml(point.url)}" target="_blank" rel="noreferrer">원문 보기</a>` : "-"}</dd>
      </dl>
    `;
  }

  function drawAxes() {
    const width = 820;
    const height = 500;
    const pad = { left: 58, right: 28, top: 30, bottom: 54 };
    for (const value of [-1, -0.5, 0, 0.5, 1]) {
      const x = pad.left + ((value + 1) / 2) * (width - pad.left - pad.right);
      const y = height - pad.bottom - ((value + 1) / 2) * (height - pad.top - pad.bottom);

      const vline = makeSvg("line");
      vline.setAttribute("class", value === 0 ? "axis" : "gridline");
      vline.setAttribute("x1", x);
      vline.setAttribute("x2", x);
      vline.setAttribute("y1", pad.top);
      vline.setAttribute("y2", height - pad.bottom);
      svg.appendChild(vline);

      const hline = makeSvg("line");
      hline.setAttribute("class", value === 0 ? "axis" : "gridline");
      hline.setAttribute("x1", pad.left);
      hline.setAttribute("x2", width - pad.right);
      hline.setAttribute("y1", y);
      hline.setAttribute("y2", y);
      svg.appendChild(hline);

      const xtick = makeSvg("text");
      xtick.setAttribute("class", "tick-label");
      xtick.setAttribute("x", x);
      xtick.setAttribute("y", height - 24);
      xtick.setAttribute("text-anchor", "middle");
      xtick.textContent = value.toFixed(value === 0 ? 0 : 1);
      svg.appendChild(xtick);

      const ytick = makeSvg("text");
      ytick.setAttribute("class", "tick-label");
      ytick.setAttribute("x", 44);
      ytick.setAttribute("y", y + 4);
      ytick.setAttribute("text-anchor", "end");
      ytick.textContent = value.toFixed(value === 0 ? 0 : 1);
      svg.appendChild(ytick);
    }

    const xlabel = makeSvg("text");
    xlabel.setAttribute("class", "axis-label");
    xlabel.setAttribute("x", width / 2);
    xlabel.setAttribute("y", height - 4);
    xlabel.setAttribute("text-anchor", "middle");
    xlabel.textContent = "X축: 2D 투영 축 1";
    svg.appendChild(xlabel);

    const ylabel = makeSvg("text");
    ylabel.setAttribute("class", "axis-label");
    ylabel.setAttribute("x", -height / 2);
    ylabel.setAttribute("y", 14);
    ylabel.setAttribute("text-anchor", "middle");
    ylabel.setAttribute("transform", "rotate(-90)");
    ylabel.textContent = "Y축: 2D 투영 축 2";
    svg.appendChild(ylabel);
  }

  function drawPoints() {
    const group = makeSvg("g");
    group.setAttribute("id", "points");
    svg.appendChild(group);
    points.forEach((point, index) => {
      const pos = pointPosition(point);
      const circle = makeSvg("circle");
      circle.setAttribute("class", "point");
      circle.setAttribute("cx", pos.x.toFixed(1));
      circle.setAttribute("cy", pos.y.toFixed(1));
      circle.setAttribute("r", Math.max(3.2, Math.min(8.5, 3.2 + point.tokens / 65)).toFixed(1));
      circle.setAttribute("fill", colors[(point.cluster - 1) % colors.length]);
      circle.dataset.cluster = String(point.cluster);
      circle.dataset.index = String(index);
      circle.setAttribute("aria-label", `${point.title} ${point.chunk_id}`);
      circle.addEventListener("click", () => showPoint(point, circle));
      group.appendChild(circle);
    });
    if (points.length) {
      showPoint(points[0], group.querySelector(".point"));
      group.querySelector(".point").classList.add("active");
    }
  }

  function drawLegend() {
    const all = document.createElement("button");
    all.type = "button";
    all.setAttribute("aria-pressed", "true");
    all.textContent = "전체";
    all.addEventListener("click", () => setCluster(0));
    legend.appendChild(all);

    clusters.forEach((cluster) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.cluster = String(cluster.cluster);
      button.setAttribute("aria-pressed", "false");
      button.innerHTML = `<span class="swatch" style="background:${colors[(cluster.cluster - 1) % colors.length]}"></span>${cluster.cluster} (${cluster.chunks.toLocaleString("ko-KR")})`;
      button.addEventListener("click", () => setCluster(cluster.cluster));
      legend.appendChild(button);
    });
  }

  function setCluster(cluster) {
    activeCluster = cluster;
    legend.querySelectorAll("button").forEach((button) => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.cluster || 0) === cluster));
    });
    if (cluster === 0) {
      legend.querySelector("button").setAttribute("aria-pressed", "true");
    }
    svg.querySelectorAll(".point").forEach((node) => {
      node.classList.toggle("dimmed", activeCluster !== 0 && Number(node.dataset.cluster) !== activeCluster);
    });
  }

  drawLegend();
  drawAxes();
  drawPoints();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
