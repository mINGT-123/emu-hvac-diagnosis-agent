from __future__ import annotations

from pathlib import Path

from app.agent.diagnosis_agent import run_diagnosis


def main() -> None:
    sample_csv = Path("app/data/sample_temperature.csv")
    question = "列车3号车厢温度持续升高，乘客反馈闷热，请给出诊断建议。"
    output = run_diagnosis(question=question, csv_path=sample_csv)
    print("=" * 80)
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    main()
