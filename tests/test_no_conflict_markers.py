"""
tests/test_no_conflict_markers.py
미해결 병합·stash 충돌 마커가 추적 파일에 커밋되는 것을 막는다.

왜 필요한가:
    실제로 한 번 새어 들어왔다. PR #79 리뷰 반영을 `git add -A` 로 커밋하면서 워킹트리에
    남아 있던 이전 세션의 미해결 stash pop 충돌 파일이 함께 딸려 갔고, `commit -q` 라
    파일 목록이 안 찍혀 넘어갔다(cf2a495 → PARAM_TUNING_PROPOSAL.md).

    **문서에 들어가면 테스트가 전부 초록이라 아무도 못 잡는다.** 실제로 다른 사람이
    다른 PR 작업 중 main 을 머지하다 우연히 발견할 때까지 남아 있었다. 코드 파일이면
    SyntaxError 로 즉시 죽지만, 마크다운은 렌더링만 깨진 채 조용히 통과한다.

    이 테스트는 그 조용한 경로를 막는다 — 커밋 습관(파일 명시 add)에만 기대지 않는다.

범위:
    `git ls-files` 로 추적 중인 파일 전부. 인코딩 검사(test_console_encoding)와 달리
    기존 위반이 남아 있지 않아(0건 확인) 저장소 전체를 한 번에 묶어도 빨간 상태로
    방치되지 않는다.

'=======' 오탐에 대하여:
    마크다운 setext 제목 밑줄(`===...`)이 우연히 7 글자면 마커와 구분이 안 된다.
    그래서 `=======` 는 **`<<<<<<<` 가 열려 있는 동안에만** 마커로 센다. 여는 쪽과
    닫는 쪽은 뒤에 라벨이 붙어(`<<<<<<< HEAD`) 오탐 여지가 없다.
"""
from __future__ import annotations

import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 이 파일 자신이 마커를 품지 않도록 런타임에 조립한다 — 리터럴로 적으면 스스로를
# 예외 처리해야 하고, 그 예외가 곧 다음 사고의 빈틈이 된다.
OPEN = "<" * 7 + " "
MIDDLE = "=" * 7
CLOSE = ">" * 7 + " "

MAX_BYTES = 2_000_000        # 그 이상은 텍스트로 안 보고 건너뛴다


def _tracked_files() -> list[pathlib.Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest(f"git 저장소가 아니거나 git 을 못 찾음: {exc}")
    return [ROOT / name for name in out.decode("utf-8").split("\0") if name]


def _markers_in(text: str) -> list[tuple[int, str]]:
    """(줄번호, 줄) 목록. '=======' 는 여는 마커가 열려 있는 동안만 센다."""
    found: list[tuple[int, str]] = []
    inside = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith(OPEN):
            found.append((lineno, line))
            inside = True
        elif inside and line.rstrip() == MIDDLE:
            found.append((lineno, line))
        elif line.startswith(CLOSE):
            found.append((lineno, line))
            inside = False
    return found


class NoConflictMarkersTest(unittest.TestCase):
    def test_tracked_files_have_no_conflict_markers(self):
        offenders: list[str] = []
        for path in _tracked_files():
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue          # 바이너리·읽을 수 없는 파일은 대상이 아니다
            for lineno, line in _markers_in(text):
                rel = path.relative_to(ROOT).as_posix()
                offenders.append(f"{rel}:{lineno}: {line[:60]}")

        self.assertEqual(
            offenders, [],
            "미해결 충돌 마커가 커밋돼 있습니다. 해당 구간을 손으로 해소한 뒤 다시 커밋하세요"
            " (문서 파일이면 다른 테스트는 전부 통과하므로 여기서만 걸립니다):\n  "
            + "\n  ".join(offenders),
        )


class MarkerDetectionTest(unittest.TestCase):
    """탐지기 자체 — 스캐너가 조용히 아무것도 안 찾게 되는 회귀를 막는다."""

    def test_detects_a_real_conflict_block(self):
        text = "\n".join([
            "본문",
            OPEN + "Updated upstream",
            "이쪽",
            MIDDLE,
            "저쪽",
            CLOSE + "Stashed changes",
        ])
        self.assertEqual([n for n, _ in _markers_in(text)], [2, 4, 6])

    def test_setext_heading_underline_is_not_a_marker(self):
        """충돌 밖의 '=======' 는 마크다운 제목 밑줄이라 세지 않는다."""
        self.assertEqual(_markers_in("제목\n" + MIDDLE + "\n본문"), [])

    def test_repo_scan_actually_reads_files(self):
        """ls-files 가 빈 목록을 주면 위 테스트가 무의미하게 통과한다."""
        self.assertGreater(len(_tracked_files()), 50)


if __name__ == "__main__":
    unittest.main()
