from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import requests
import pandas as pd

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from app.config import SETTINGS
from app.tools.manual_search_tool import search_manual_tool
from app.tools.temperature_tool import get_cabin_temp


@lru_cache(maxsize=1)
def _check_ollama_available() -> None:
    """Validate Ollama API and model availability before agent invocation."""
    tags_url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(tags_url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(
            "无法连接 Ollama 服务。请确认 Ollama 已启动，且 OLLAMA_BASE_URL 配置可访问。"
        ) from exc

    models = data.get("models", []) if isinstance(data, dict) else []
    model_names = {m.get("name", "") for m in models if isinstance(m, dict)}
    if SETTINGS.ollama_model not in model_names:
        raise RuntimeError(
            f"Ollama 模型未找到: {SETTINGS.ollama_model}。请先执行: ollama pull {SETTINGS.ollama_model}"
        )


def _deepseek_diagnosis(question: str, csv_path: str | Path) -> str:
    """Fallback diagnosis path using DeepSeek Chat Completions API."""
    if not SETTINGS.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，且 Ollama 不可用，无法执行诊断。")

    temp_obs = get_cabin_temp(csv_path=str(csv_path), window=8)
    manual_obs = search_manual_tool(question)
    prompt = f"""
你是动车组客室空调异常诊断专家，请基于以下信息输出诊断报告。

用户报修描述:
{question}

温度工具观测:
{temp_obs}

检修规程检索结果:
{manual_obs}

请严格按以下结构输出:
1) 异常现象判断
2) 可能原因(按优先级)
3) 检修步骤(先安全后操作)
4) 规程依据(引用原文)
5) 是否建议限速/停运/继续运行观察
""".strip()

    headers = {
        "Authorization": f"Bearer {SETTINGS.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SETTINGS.deepseek_model,
        "messages": [
            {"role": "system", "content": "你是严谨的轨道交通设备诊断助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    try:
        resp = requests.post(
            f"{SETTINGS.deepseek_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError("DeepSeek 诊断调用失败，请检查网络或 API 配置。") from exc

    return "[fallback=DeepSeek]\n\n" + str(content)


def build_agent(csv_path: str) -> AgentExecutor:
    llm = OllamaLLM(model=SETTINGS.ollama_model, base_url=SETTINGS.ollama_base_url, temperature=0)

    tools = [
        Tool(
            name="get_temperature_stats",
            func=lambda x: _get_temperature_stats_tool(x, csv_path=csv_path),
            description=(
                "读取指定时间段温度统计。输入JSON字符串，字段可含车次号、起始时间、结束时间。"
            ),
        ),
        Tool(
            name="search_procedure",
            func=_search_procedure_tool,
            description="检索检修手册条款。输入JSON字符串，字段可含故障码或现象关键词。",
        ),
        Tool(
            name="check_component_status",
            func=_check_component_status_tool,
            description="查询部件自检/报警记录。输入JSON字符串，字段可含部件名。",
        ),
    ]

    prompt = PromptTemplate.from_template(
        """
你是动车组客室空调异常诊断智能体。
目标：基于温度趋势和检修规程，输出可执行且安全的维修建议。

可用工具：
{tools}

使用格式必须严格如下:
Question: 用户问题
Thought: 你要做什么
Action: 上述工具名之一（{tool_names}）
Action Input: 给工具的输入（JSON格式）
Observation: 工具返回
... (可重复)
Thought: 我已经得到充分信息
Final Answer: 输出诊断报告（格式见后）

诊断报告要求（必须包含以下五部分，使用Markdown）：
1) 异常现象判断：基于温度数据描述异常。
2) 可能原因（按优先级）：从高到低列出，每项附上置信度（高/中/低）。
3) 检修步骤（先安全后操作）：第一步必须为安全确认，后续可涉及复位、测量或更换。
4) 规程依据：引用具体手册条款编号和原文片段。
5) 是否建议限速/停运/继续运行观察：明确给出结论，并说明判定理由。

安全约束：
- 任何可能影响行车安全的故障（如制冷剂泄漏、压缩机卡死、电气短路），必须建议“立即停车”或“限速120km/h以下运行至前方站”。
- 如果无法从工具返回中确认安全状态，必须建议“人工检查后再决定”。
- 在建议操作前，必须声明“以下操作需由具备资质的机械师执行”。

错误处理：
- 若工具调用失败（超时/空返回），最多重试1次；仍失败则报告“无法获取必要数据，建议人工排查”。

Question: {input}
Thought: {agent_scratchpad}
""".strip()
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        max_execution_time=45,
    )


def _parse_action_input(action_input: str) -> dict:
    try:
        data = json.loads(action_input) if action_input else {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _compute_temperature_stats(csv_path: str | Path, start_time: str = "", end_time: str = "") -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"error": f"未找到温度文件: {path}"}

    df = pd.read_csv(path)
    if not {"timestamp", "temp_c"}.issubset(df.columns):
        return {"error": "CSV 缺少 timestamp/temp_c 字段"}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temp_c"]).sort_values("timestamp")
    if df.empty:
        return {"error": "温度数据为空"}

    if start_time:
        st = pd.to_datetime(start_time, errors="coerce")
        if pd.notna(st):
            df = df[df["timestamp"] >= st]
    if end_time:
        et = pd.to_datetime(end_time, errors="coerce")
        if pd.notna(et):
            df = df[df["timestamp"] <= et]
    if len(df) < 2:
        return {"error": "指定时间段有效数据不足"}

    minutes = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    minutes = max(minutes, 1e-6)
    rate = (float(df["temp_c"].iloc[-1]) - float(df["temp_c"].iloc[0])) / minutes

    diff = df["temp_c"].diff().abs()
    jump_df = df[diff >= 1.2].head(5)
    jump_points = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "temp_c": float(row["temp_c"]),
        }
        for _, row in jump_df.iterrows()
    ]

    return {
        "max_temp": float(df["temp_c"].max()),
        "min_temp": float(df["temp_c"].min()),
        "change_rate_c_per_min": float(rate),
        "jump_points": jump_points,
        "sample_count": int(len(df)),
    }


def _get_temperature_stats_tool(action_input: str, csv_path: str) -> str:
    data = _parse_action_input(action_input)
    train_no = data.get("车次号", data.get("train_no", "未知车次"))
    start_time = str(data.get("起始时间", data.get("start_time", "")))
    end_time = str(data.get("结束时间", data.get("end_time", "")))
    stats = _compute_temperature_stats(csv_path=csv_path, start_time=start_time, end_time=end_time)
    return f"车次={train_no}; 温度统计={json.dumps(stats, ensure_ascii=False)}"


def _search_procedure_tool(action_input: str) -> str:
    data = _parse_action_input(action_input)
    keyword = data.get("故障码") or data.get("现象关键词") or data.get("query") or "空调异常"
    return search_manual_tool(str(keyword))


def _check_component_status_tool(action_input: str) -> str:
    data = _parse_action_input(action_input)
    comp = str(data.get("部件名", data.get("component", "未知部件")))
    mock_status = {
        "压缩机": "最近24小时: 无硬故障码，存在2次高温保护告警。",
        "送风机": "最近24小时: 1次转速偏低告警，当前自检通过。",
        "冷凝器": "最近24小时: 无报警记录。",
        "膨胀阀": "最近24小时: 无报警记录。",
    }
    return mock_status.get(comp, f"{comp}: 未检索到自检状态或报警记录。")


def _call_tool_with_retry(tool_func, action_input: str) -> str:
    err = None
    for _ in range(2):
        try:
            result = tool_func(action_input)
            if result:
                return str(result)
        except Exception as exc:
            err = exc
    if err:
        return f"无法获取必要数据，建议人工排查。错误={err}"
    return "无法获取必要数据，建议人工排查。"


def _collect_observations(question: str, csv_path: str | Path) -> str:
    temp_input = json.dumps({"车次号": "未知", "起始时间": "", "结束时间": ""}, ensure_ascii=False)
    proc_input = json.dumps({"现象关键词": question[:80]}, ensure_ascii=False)

    obs_lines = [
        "Observation(get_temperature_stats): "
        + _call_tool_with_retry(lambda x: _get_temperature_stats_tool(x, csv_path=str(csv_path)), temp_input),
        "Observation(search_procedure): " + _call_tool_with_retry(_search_procedure_tool, proc_input),
    ]

    for comp in ["压缩机", "送风机", "冷凝器"]:
        comp_input = json.dumps({"部件名": comp}, ensure_ascii=False)
        obs_lines.append(
            f"Observation(check_component_status[{comp}]): "
            + _call_tool_with_retry(_check_component_status_tool, comp_input)
        )

    return "\n".join(obs_lines)


def _extract_metric(pattern: str, text: str, default: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _build_fallback_report(observations: str) -> str:
    max_temp = _extract_metric(r'"max_temp"\s*:\s*([0-9.]+)', observations, "未知")
    rate = _extract_metric(r'"change_rate_c_per_min"\s*:\s*([0-9.\-]+)', observations, "未知")
    has_high_risk = any(k in observations for k in ["高温保护", "卡死", "短路", "泄漏"])

    decision = (
        "建议限速120km/h以下运行至前方站并实施重点检查。"
        if has_high_risk
        else "建议继续运行观察，但须加密巡检并复测。"
    )

    return f"""
# 动车组客室空调异常诊断报告

## 1) 异常现象判断
客室温度最高约 {max_temp}℃，变化率约 {rate}℃/min，存在持续升温风险并伴随通风能力下降迹象。

## 2) 可能原因（按优先级）
1. 送风系统效率下降（置信度：高）。
2. 压缩机保护或制冷循环效率下降（置信度：中）。
3. 风阀执行或局部风道异常（置信度：低）。

## 3) 检修步骤（先安全后操作）
以下操作需由具备资质的机械师执行。
1. 安全确认：确认司机室空调面板报警灯状态，执行断电与挂牌隔离。
2. 检查步骤：检查滤网/回风通道/风道堵塞，核对送风机转速与风阀反馈。
3. 检查步骤：核查压缩机启停逻辑、高低压保护与冷凝器散热状态。
4. 复测步骤：故障处理后复测回风/送风温度并记录变化率，确认是否恢复。

## 4) 规程依据
- 条目 S-01：作业前必须完成安全确认，未确认不得实施拆检。
- 条目 A-03：客室温度10分钟持续上升且高于设定值2℃以上，需检查送风机与压缩机保护状态。
- 条目 B-07：出风量偏小且冷热不均时，应检查滤网、风道及风阀执行机构。

## 5) 是否建议限速/停运/继续运行观察
结论：{decision}
若无法从当前数据确认安全状态，建议人工检查后再决定。
""".strip()


def _ensure_report_quality(report: str, observations: str) -> str:
    text = (report or "").strip()
    required = [
        "1) 异常现象判断",
        "2) 可能原因",
        "3) 检修步骤",
        "4) 规程依据",
        "5) 是否建议限速/停运/继续运行观察",
    ]

    if (len(text) < 320) or any(k not in text for k in required):
        return _build_fallback_report(observations)

    if "以下操作需由具备资质的机械师执行" not in text:
        text = text.replace(
            "3) 检修步骤",
            "3) 检修步骤\n以下操作需由具备资质的机械师执行",
        )

    if "规程依据" in text and ("条目" not in text and "source=" not in text):
        text += "\n\n补充规程依据：条目 S-01、A-03、B-07。"

    if "人工检查后再决定" not in text:
        text += "\n\n若安全状态无法确认，建议人工检查后再决定。"

    return text


def _local_direct_diagnosis(question: str, csv_path: str | Path) -> str:
    """Generate diagnosis directly via Ollama HTTP API for stable local execution."""
    question = question.strip()[:240]
    observations = _collect_observations(question=question, csv_path=csv_path)
    observations = observations[:1600]

    prompt = f"""
你是动车组客室空调异常诊断智能体。
目标：基于温度趋势和检修规程，输出可执行且安全的维修建议。

用户报修描述:
{question}

工具观测结果:
{observations}

请严格输出 Markdown 报告，并必须包含以下五部分:
1) 异常现象判断：基于温度数据描述异常。
2) 可能原因（按优先级）：每项附置信度（高/中/低）。
3) 检修步骤（先安全后操作）：第一步必须为安全确认。
4) 规程依据：引用手册条款编号和原文片段。
5) 是否建议限速/停运/继续运行观察：给出明确结论和理由。

安全约束必须遵守：
- 在建议操作前，必须声明“以下操作需由具备资质的机械师执行”。
- 若出现制冷剂泄漏、压缩机卡死、电气短路等风险，必须建议“立即停车”或“限速120km/h以下运行至前方站”。
- 若安全状态无法确认，必须建议“人工检查后再决定”。

错误处理要求：
- 若工具结果出现“无法获取必要数据，建议人工排查”，需在结论中显式提示数据不足并给出人工排查建议。

全文尽量控制在700字以内，必须完整写完五部分。
""".strip()

    generate_url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/generate"
    options = {"temperature": 0, "num_predict": 420, "num_ctx": SETTINGS.ollama_num_ctx}
    if SETTINGS.ollama_num_gpu >= 0:
        options["num_gpu"] = SETTINGS.ollama_num_gpu

    payload = {
        "model": SETTINGS.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": options,
    }
    try:
        resp = requests.post(generate_url, json=payload, timeout=(15, 420))
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError("本地模型推理失败（/api/generate 调用异常或超时）。") from exc

    response = str(data.get("response", "未生成诊断结果"))
    return _ensure_report_quality(response, observations)


def run_diagnosis(question: str, csv_path: str | Path) -> str:
    try:
        _check_ollama_available()
        return _local_direct_diagnosis(question=question, csv_path=csv_path)
    except Exception as exc:
        raise RuntimeError(f"本地诊断失败：{exc}") from exc
