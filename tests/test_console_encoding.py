"""
tests/test_console_encoding.py
try 블록 안의 print 문구가 cp949 콘솔에서 죽지 않는지 고정한다.

왜 필요한가:
    진입점은 core.console.force_utf8_stdio() 로 보호되지만, 그걸 거치지 않는 경로
    (모듈 단독 import·일부 워커)에서는 한국어 Windows 콘솔이 cp949 라 '—'·'⚠' 같은
    문자가 UnicodeEncodeError 를 낸다. 그 print 가 try 안이면 바깥 except 가 예외를
    삼켜 **폴백·복구 경로 자체가 무력화된다**(실제로 리랭커 로드 폴백이 그렇게 죽었다:
    상한 인자 미지원 → 상한 없이 로드해야 하는데, 경고 print 가 터져 '로드 실패'로
    기록되고 300초 쿨다운까지 걸렸다).

    pytest 는 stdout 을 utf-8 로 캡처하므로 일반 테스트로는 이 경로가 안 잡힌다.

범위:
    저장소 전체가 아니라 '폴백을 try 로 감싸는' 검색 경로 모듈만 본다. 기존 위반이
    남아 있는 파일까지 한 번에 묶으면 테스트가 계속 빨간 상태로 방치되기 때문이다.
    새 위반을 이 경로에서 막는 것이 목적이고, 나머지는 별도 정리 대상이다.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 폴백을 try/except 로 감싸는 검색 경로 — 여기서 나는 인코딩 예외가 복구를 삼킨다.
GUARDED_MODULES = (
    "agents/index/qdrant_store.py",
    "agents/rag/retriever.py",
)


def _string_literals(node: ast.AST) -> list[str]:
    """노드 아래의 문자열 리터럴(f-string 고정 부분 포함)."""
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def _print_literals_in_try(tree: ast.AST) -> list[tuple[int, str]]:
    """try 블록 안에서 print() 에 넘기는 문자열 리터럴 (줄번호, 값)."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name) and func.id == "print":
                for text in _string_literals(child):
                    found.append((child.lineno, text))
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


class TryBlockPrintEncodingTest(unittest.TestCase):
    def test_guarded_modules_have_cp949_safe_print(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for rel in GUARDED_MODULES:
            path = root / rel
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for lineno, text in _print_literals_in_try(tree):
                offenders = _cp949_offenders(text)
                self.assertEqual(
                    offenders, [],
                    f"{rel}:{lineno} print 문구에 cp949 불가 문자 {offenders} — "
                    f"try 안이라 인코딩 예외가 폴백을 삼킨다(AGENTS.md 코드 컨벤션)",
                )


if __name__ == "__main__":
    unittest.main()
