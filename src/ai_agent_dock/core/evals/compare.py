"""패턴 비교표(C90): 결과 파일의 compare 케이스를 "질문 하나 × 실행 방법 하나" 줄로 만들어 CSV로 쓴다.

케이스 이름은 compare-모양-번호-실행방법이다(예: compare-noevidence-1-corrective-rag). 모양은 한 단어라
앞의 세 조각을 떼면 나머지가 실행 방법이다. 질문 글은 CSV에 넣지 않는다(Notion으로 옮겨 색인을 오염시키지 않게, C90 4.4절).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

COLUMNS = [
    "run", "shape", "level", "question", "runner", "passes", "runs", "judge",
    "llm_calls", "input_tokens", "output_tokens", "seconds", "calls_x", "tokens_x", "seconds_x",
]
JUDGED = re.compile(r"^통과 (\d+)/(\d+)")  # verdict_line(C17)의 "통과 3/4 …", "통과 2/2, 판정 안 함 2". "오류: …"는 맞지 않는다.


def _judge(lines: list[str]) -> float | str:
    """실행마다 LLM 판정에서 판정한 기준 중 통과한 비율(0~1)의 평균. 판정이 없으면 빈 칸.

    비율로 본다: corrective-rag는 근거를 판정자에게 넘기지 않아 기준 2개만 판정하므로 개수로는 main(4개)과 견줄 수 없다(C90 Q17).
    """
    shares = [int(m.group(1)) / int(m.group(2)) for m in map(JUDGED.match, lines) if m and int(m.group(2))]
    return round(sum(shares) / len(shares), 2) if shares else ""


def _ratio(value: float, base: float | None) -> float | str:
    return round(value / base, 1) if base else ""


def compare_rows(data: dict) -> list[dict]:
    """compare 케이스마다 한 줄. 모양, 질문 번호 순이고 같은 질문 안에서는 main이 먼저다. 배수는 같은 질문의 main 대비."""
    rows = []
    for case in data["cases"]:
        if case["category"] != "compare":
            continue
        _, shape, number, runner = case["id"].split("-", 3)
        rows.append({
            "run": data["run"], "shape": case.get("shape") or shape, "level": case.get("level") or "",
            "question": int(number), "runner": runner, "passes": case["passes"], "runs": case["runs"],
            "judge": _judge(case["llm_verdicts"]), "llm_calls": case["llm_calls"], "input_tokens": case["input_tokens"],
            "output_tokens": case["output_tokens"], "seconds": case["seconds"],
        })
    main = {(r["shape"], r["question"]): r for r in rows if r["runner"] == "main"}
    for r in rows:
        base = main.get((r["shape"], r["question"]), {})
        r["calls_x"] = _ratio(r["llm_calls"], base.get("llm_calls"))
        r["tokens_x"] = _ratio(r["input_tokens"], base.get("input_tokens"))
        r["seconds_x"] = _ratio(r["seconds"], base.get("seconds"))
    return sorted(rows, key=lambda r: (r["shape"], r["question"], r["runner"] != "main"))


def write_csv(directory: Path, data: dict) -> Path:
    """data/evals/실행번호-compare.csv. UTF-8 BOM으로 써서 엑셀에서 한글이 깨지지 않게 한다."""
    path = directory / f"{data['run']}-compare.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(compare_rows(data))
    return path
