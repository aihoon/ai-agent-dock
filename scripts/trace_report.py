"""추적 리포트: logs/sessions/의 세션 파일에서 호출 수, 걸린 시간, 토큰 수를 뽑아 보여 준다(내용은 보여 주지 않는다).

사용법:
  uv run python scripts/trace_report.py                 # 세션 목록
  uv run python scripts/trace_report.py show 세션번호    # 세션 하나의 턴별 통계(앞부분만 적어도 된다)
  uv run python scripts/trace_report.py show 세션이름    # 채팅의 세션 이름으로도 찾는다(data/sessions/, C9 D26)
  uv run python scripts/trace_report.py steps 세션번호   # 세션 하나를 스텝 순서대로: 프롬프트, 모델 출력, Tool 결과, 워크플로 노드(C13)
  uv run python scripts/trace_report.py steps           # 세션을 적지 않으면 가장 최근 세션
  uv run python scripts/trace_report.py total           # 모든 세션의 합계와 턴당 평균
  uv run python scripts/trace_report.py compare         # 세션별 비교(세션마다 한 줄)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_agent_dock.core.observability.report import all_stats, format_compare, format_list, format_session, format_steps, format_total, load_sessions
from ai_agent_dock.core.session.store import SessionStore

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "logs" / "sessions"
SAVED_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"  # 채팅 세션 저장(이름은 여기에만 있다, C9 D10)


def main() -> None:
    parser = argparse.ArgumentParser(description="추적 리포트")
    parser.add_argument("command", nargs="?", default="list", choices=["list", "show", "steps", "total", "compare"])
    parser.add_argument("session", nargs="?", help="show, steps에서 볼 세션 이름 또는 세션 번호(번호는 앞부분만 적어도 된다)")
    args = parser.parse_args()

    stats = all_stats(SESSIONS_DIR)
    names = {r.id: r.name for r in SessionStore(SAVED_DIR).list()}
    if not stats:
        print(f"기록된 세션이 없습니다({SESSIONS_DIR}). TRACE_ENABLED=true로 채팅이나 평가를 실행하세요.")
        return
    if args.command == "list":
        print(format_list(stats, names))
    elif args.command == "total":
        print(format_total(stats))
    elif args.command == "compare":
        print(format_compare(stats))
    elif args.command == "steps" and not args.session:
        latest = max(stats, key=lambda s: s.start)
        print(format_steps(load_sessions(SESSIONS_DIR)[latest.session]))
    else:
        if not args.session:
            raise SystemExit("사용법: show 세션이름 또는 세션번호 (목록은 인자 없이 실행하면 나온다)")
        by_name = [s for s in stats if names.get(s.session) == args.session]  # 이름이 정확히 같으면 이름을 먼저 쓴다.
        matches = by_name or [s for s in stats if args.session in s.session]  # 번호 일부만 적어도 찾는다(eval- 생략 등).
        if len(matches) != 1:
            raise SystemExit(f"세션을 하나로 정할 수 없습니다: {args.session!r} (일치 {len(matches)}개). 목록은 인자 없이 실행해 보세요.")
        if args.command == "steps":
            print(format_steps(load_sessions(SESSIONS_DIR)[matches[0].session]))
        else:
            print(format_session(matches[0]))


if __name__ == "__main__":
    main()
