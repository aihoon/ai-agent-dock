"""패턴 비교표(C90): 평가 결과 파일의 compare 케이스를 CSV로 쓴다.

사용법: uv run python scripts/compare_report.py data/evals/실행번호.json
결과: data/evals/실행번호-compare.csv(엑셀로 열린다). 한 줄이 "질문 하나 × 실행 방법 하나"다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_agent_dock.core.evals.compare import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="패턴 비교표 CSV")
    parser.add_argument("result", type=Path, help="평가 결과 파일(data/evals/실행번호.json)")
    path = parser.parse_args().result
    data = json.loads(path.read_text(encoding="utf-8"))
    print(write_csv(path.parent, data))


if __name__ == "__main__":
    main()
