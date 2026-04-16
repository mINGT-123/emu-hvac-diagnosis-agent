from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from app.config import SETTINGS


@dataclass
class ScoreResult:
    accuracy: int
    actionability: int
    safety: int
    traceability: int

    @property
    def total(self) -> int:
        return self.accuracy + self.actionability + self.safety + self.traceability


def _rule_based_score(report: str) -> ScoreResult:
    text = report.lower()
    accuracy = 8 if any(k in text for k in ["温度", "趋势", "传感器"]) else 5
    actionability = 9 if any(k in report for k in ["步骤", "检查", "复测"]) else 6
    safety = 9 if any(k in report for k in ["断电", "安全", "隔离"]) else 6
    traceability = 9 if any(k in report for k in ["规程", "source=", "依据"]) else 5
    return ScoreResult(accuracy, actionability, safety, traceability)


def _deepseek_score(report: str) -> ScoreResult:
    prompt = f"""
你是轨道交通设备诊断评审专家。请对以下诊断报告按0-10分打分，并只返回JSON:
{{"accuracy":x,"actionability":x,"safety":x,"traceability":x}}

诊断报告:
{report}
""".strip()

    headers = {
        "Authorization": f"Bearer {SETTINGS.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SETTINGS.deepseek_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    resp = requests.post(
        f"{SETTINGS.deepseek_base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    return ScoreResult(
        accuracy=int(data.get("accuracy", 0)),
        actionability=int(data.get("actionability", 0)),
        safety=int(data.get("safety", 0)),
        traceability=int(data.get("traceability", 0)),
    )


def score_report(report: str) -> ScoreResult:
    if SETTINGS.deepseek_api_key:
        try:
            return _deepseek_score(report)
        except Exception:
            return _rule_based_score(report)
    return _rule_based_score(report)
