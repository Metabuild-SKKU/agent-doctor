"""
tests/test_console_encoding.py
검색 경로의 콘솔 문구가 cp949 콘솔에서 죽지 않는지 고정한다.

왜 필요한가:
    진입점은 core.console.force_utf8_stdio() 로 보호되지만, 그걸 거치지 않는 경로
    (모듈 단독 import·일부 워커)에서는 한국어 Windows 콘솔이 cp949 라 '—'·'⚠' 같은
    문자가 UnicodeEncodeError 를 낸다. 그 예외가 어디서 나느냐에 따라 두 가지로 터진다.

      1) try 안이면 바깥 except 가 삼켜 **폴백·복구 경로가 무력화**된다(실제로 리랭커
         로드 폴백이 그렇게 죽었다: 상한 인자 미지원 → 상한 없이 로드해야 하는데,
         경고 print 가 터져 '로드 실패'로 기록되고 300초 쿨다운까지 걸렸다).
      2) try 밖이면 그대로 위로 올라가 **검색 자체가 죽는다**. 리랭커 로드
         (_load_reranker)가 rerank_with_status 의 try 바깥이라, 거기서 난 인코딩
         예외는 search_with_details 까지 올라간다. GPU 가 모자란 머신을 구하려던
         CPU 폴백 경고가 정확히 그 머신에서 검색을 깨뜨리는 모양이 된다.

    pytest 는 stdout 을 utf-8 로 캡처하므로 일반 테스트로는 이 경로가 안 잡힌다.

범위:
    저장소 전체가 아니라 검색 경로 모듈만 본다. 다만 그 안에서는 try 안팎을 가리지
    않는다 — 위 2) 때문이다. 처음에는 try 안만 봤는데, 그 좁은 규칙이 실제로 위
    두 경우를 다 놓쳤다(리뷰에서 잡혔다).

    print 뿐 아니라 _notify_route_once 처럼 문구를 받아 콘솔에 찍는 헬퍼도 본다.
    한 번만 찍는다는 성질이 인코딩 위험을 줄여 주지는 않는다.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 검색 경로 — 여기서 나는 인코딩 예외는 복구를 삼키거나(try 안) 검색을 죽인다(try 밖).
GUARDED_MODULES = (
    "agents/index/qdrant_store.py",
    "agents/rag/retriever.py",
)
# 문구를 받아 콘솔에 찍는 함수들. print 와 같은 위험이다.
CONSOLE_CALLS = ("print", "_notify_route_once")


def _string_literals(node: ast.AST) -> list[str]:
    """노드 아래의 문자열 리터럴(f-string 고정 부분 포함)."""
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def _console_literals(tree: ast.AST) -> list[tuple[int, str]]:
    """콘솔 출력 호출에 넘기는 문자열 리터럴 (줄번호, 값). try 안팎을 가리지 않는다."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        if name not in CONSOLE_CALLS:
            continue
        for text in _string_literals(node):
            found.append((node.lineno, text))
    return found


def _cp949_offenders(text: str) -> list[str]:
    bad = []
    for char in text:
        if char.isascii():
            continue
        try:
            char.encode("cp949")
        except UnicodeEncodeError:
            bad.append(char)
    return bad


class SearchPathConsoleEncodingTest(unittest.TestCase):
    def test_guarded_modules_have_cp949_safe_console_text(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for rel in GUARDED_MODULES:
            path = root / rel
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for lineno, text in _console_literals(tree):
                offenders = _cp949_offenders(text)
                self.assertEqual(
                    offenders, [],
                    f"{rel}:{lineno} 콘솔 문구에 cp949 불가 문자 {offenders} - "
                    f"try 안이면 폴백이 삼켜지고, 밖이면 검색이 죽는다"
                    f"(AGENTS.md 코드 컨벤션)",
                )


if __name__ == "__main__":
    unittest.main()
