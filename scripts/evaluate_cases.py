from __future__ import annotations

from pathlib import Path

from app.agent.diagnosis_agent import run_diagnosis
from app.evaluation.judge import score_report


CASES = [
    "车厢温度持续上升，乘客反馈闷热，检查建议是什么？",
    "温度波动大且出风忽冷忽热，可能原因与处理步骤？",
]


def main() -> None:
    csv_path = Path("app/data/sample_temperature.csv")
    for i, case in enumerate(CASES, start=1):
        report = run_diagnosis(case, csv_path)
        score = score_report(report)
        print(f"Case {i}: {case}")
        print(f"Score(total={score.total}): {score}")
        print("-" * 80)


if __name__ == "__main__":
    main()
