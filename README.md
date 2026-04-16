# 基于本地 LLM 与 RAG 的动车组客室空调异常诊断 Agent

本项目面向“旅客报修空调异常、随车机械师排查效率低”的实际场景，提供一个可离线运行的轻量化智能诊断系统。

## 功能概览

- 本地化 Agent: 基于 LangChain ReAct，驱动工具调用。
- 温度趋势分析工具: 从 CSV 温度流中提取滑动窗口统计。
- 本地 RAG: 使用 ChromaDB + all-MiniLM-L6-v2 检索维修规程条目。
- 可溯源建议: 诊断结果包含引用手册原文。
- Streamlit 演示: 上传温度文件并实时生成诊断报告。
- 自动评测: 支持 DeepSeek API 的 LLM-as-a-Judge 四维评分。

## 目录结构

```text
app/
  agent/                # Agent 编排
  tools/                # 工具封装
  rag/                  # 向量库构建与检索
  evaluation/           # 自动化评测
  ui/                   # Streamlit 页面
  data/                 # 样例数据与数据模拟
knowledge/manual/       # 手册原文
scripts/                # 启动与运维脚本
```

## 快速开始

1) 安装依赖

```powershell
pip install -r requirements.txt
```

2) 配置环境变量

```powershell
copy .env.example .env
```

3) 准备本地模型（Ollama）

```powershell
ollama pull qwen2.5:1.5b
```

4) 构建知识库

```powershell
python scripts/build_kb.py
```

5) 启动 Streamlit

```powershell
streamlit run app/ui/streamlit_app.py
```

## 评测

使用内置样例执行自动评分：

```powershell
python scripts/evaluate_cases.py
```

若配置了 `DEEPSEEK_API_KEY`，会自动启用 LLM-as-a-Judge 评测。

## 说明

- 默认只依赖 CPU。
- 如本地无 Ollama，也可切换到任意兼容的本地模型服务。
