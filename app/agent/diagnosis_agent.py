from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import requests

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
            name="get_cabin_temp",
            func=lambda _: get_cabin_temp(csv_path=csv_path, window=8),
            description="读取当前客室温度数据并返回滑动窗口统计。输入任意文本即可。",
        ),
        Tool(
            name="search_manual",
            func=search_manual_tool,
            description="从《铁路客车空调装置检修规程》检索维修条目。输入应为故障描述或查询问题。",
        ),
    ]

    prompt = PromptTemplate.from_template(
        """
你是动车组客室空调异常诊断智能体。
你的目标是: 基于温度趋势和检修规程，输出可执行且安全的维修建议。

工具列表:
{tools}

使用格式必须严格如下:
Question: 用户问题
Thought: 你要做什么
Action: 工具名[{tool_names}] 中的一个
Action Input: 给工具的输入
Observation: 工具返回
... (Thought/Action/Action Input/Observation 可重复)
Thought: 我已经得到充分信息
Final Answer: 输出诊断报告

诊断报告要求:
1) 异常现象判断
2) 可能原因(按优先级)
3) 检修步骤(先安全后操作)
4) 规程依据(引用原文)
5) 是否建议限速/停运/继续运行观察

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


def _local_direct_diagnosis(question: str, csv_path: str | Path) -> str:
    """Generate diagnosis directly via Ollama HTTP API for stable local execution."""
    question = question.strip()[:240]
    temp_obs = get_cabin_temp(csv_path=str(csv_path), window=8)
    manual_obs = search_manual_tool(question)
    # Cap retrieved manual text to keep prompt compact and latency predictable.
    manual_obs = manual_obs[:700]

    prompt = f"""
你是动车组客室空调异常诊断专家，请基于输入信息输出可执行报告。

用户报修描述:
{question}

温度工具观测:
{temp_obs}

检修规程检索结果:
{manual_obs}

请严格按以下结构输出，全文控制在500字以内:
1) 异常现象判断
2) 可能原因(按优先级)
3) 检修步骤(先安全后操作)
4) 规程依据(引用原文)
5) 是否建议限速/停运/继续运行观察
""".strip()

    generate_url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/generate"
    options = {"temperature": 0, "num_predict": 240, "num_ctx": SETTINGS.ollama_num_ctx}
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

    return str(data.get("response", "未生成诊断结果"))


def run_diagnosis(question: str, csv_path: str | Path) -> str:
    try:
        _check_ollama_available()
        return _local_direct_diagnosis(question=question, csv_path=csv_path)
    except Exception as exc:
        raise RuntimeError(f"本地诊断失败：{exc}") from exc
