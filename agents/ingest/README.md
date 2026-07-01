# Ingest Agent

데이터 소스(Notion, 파일 등)에서 문서를 수집해 `Document` 리스트로 반환하는 에이전트.

---

## 역할

파이프라인에서 가장 첫 번째로 실행됨.

```
[Ingest Agent]  ←  state.source_url, state.source_type
      ↓
state.documents 채움  →  [Index Agent]로 전달
```

---

## 출력 데이터 (`state.documents`)

Index Agent는 이 데이터를 읽어서 청크 분할 + 벡터화를 진행함.

```python
state.documents: list[Document]
```

### Document 구조

```python
@dataclass
class Document:
    doc_id    : str       # UUID (자동 생성)
    source    : str       # 원본 URL 또는 파일 경로
    format    : str       # "notion" | "pdf" | "md" | "txt"
    content   : str       # 순수 텍스트 (HTML 태그 없음)
    metadata  : dict      # 아래 참고
    ingested_at: datetime
```

### metadata 예시

| source_type | metadata 키 |
|-------------|-------------|
| notion | `title`, `page_id` |
| pdf | `filename` |
| md / txt | `filename` |

---

## 지원 소스

| source_type | 설명 | 필요 설정 |
|-------------|------|-----------|
| `"notion"` | Notion 페이지 전체 수집 | `NOTION_TOKEN` 또는 OAuth |
| `"file"` | 로컬 파일 (.txt / .md / .pdf) | 없음 |
| `"gdrive"` | Google Drive | 미구현 (TODO) |

---

## 인증 방식

### 방식 1 — Access Token (개발/테스트용)

`.env` 파일에 토큰 직접 입력:

```
NOTION_TOKEN=secret_xxxxx
```

### 방식 2 — OAuth (서비스 배포용)

이용자가 직접 본인 Notion을 연결하는 방식.

```
NOTION_CLIENT_ID=xxx
NOTION_CLIENT_SECRET=xxx
```

`get_notion_token(use_oauth=True)` 호출 시 브라우저 자동 오픈 → 로그인 → 승인 → 토큰 자동 저장.

---

## 테스트

```bash
cd C:\SKKU\agent_doctor_v2

# Access Token 방식 테스트
python tests/test_ingest.py

# OAuth 방식 테스트
python tests/test_oauth.py
```

### OAuth 테스트 사전 준비

1. [notion.so/my-integrations](https://notion.so/my-integrations) → **Public** integration 생성
2. Redirect URI: `http://localhost:8765/callback` 등록
3. `.env`에 `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` 추가

---

## 새 소스 추가하는 법

`agent.py` 에 함수 하나 추가하고 라우팅 테이블에 등록하면 됨.

```python
# 1. 수집 함수 작성
def _ingest_slack(source_url: str) -> list[Document]:
    ...
    return [Document(...)]

# 2. 라우팅 테이블에 추가
_INGESTERS = {
    "notion": _ingest_notion,
    "file":   _ingest_file,
    "gdrive": _ingest_gdrive,
    "slack":  _ingest_slack,   # ← 추가
}
```

---

## 파일 구조

```
agents/ingest/
├── agent.py    # run(state) 진입점 + 소스별 수집 함수
├── oauth.py    # Notion 인증 처리 (Access Token / OAuth)
└── README.md   # 이 파일
```
