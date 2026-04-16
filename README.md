# 基于本地 LLM 与 RAG 的动车组客室空调异常诊断 Agent

本项目面向“旅客报修空调异常、随车机械师排查效率低”的实际场景，提供一个可离线运行的轻量化智能诊断系统。

## 功能概览

- 本地诊断链路：基于本地 Ollama 模型生成诊断报告。
- 温度趋势分析：从 CSV 温度流中提取滑动窗口统计。
- 本地 RAG 检索：使用 ChromaDB + all-MiniLM-L6-v2 检索维修规程条目。
- 可溯源输出：诊断结果包含规程引用片段。
- Streamlit 演示：上传温度文件并实时生成诊断报告。
- 自动评测：支持 DeepSeek API 的 LLM-as-a-Judge 四维评分。

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

1. 安装依赖

```powershell
pip install -r requirements.txt
```

2. 配置环境变量

```powershell
copy .env.example .env
```

3. 准备本地模型（Ollama）

```powershell
ollama pull qwen3.5:4b
```

4. 构建知识库

```powershell
python scripts/build_kb.py
```

5. 启动 Streamlit

```powershell
streamlit run app/ui/streamlit_app.py
```

## 环境变量说明

以下变量在 .env 中配置：

- OLLAMA_BASE_URL：本地 Ollama 地址，默认 http://localhost:11434。
- OLLAMA_MODEL：本地模型名，当前推荐 qwen3.5:4b。
- OLLAMA_NUM_GPU：GPU 层数参数。999 表示尽量走 GPU，-1 表示自动。
- OLLAMA_NUM_CTX：上下文长度上限，默认 1024。
- CHROMA_DIR：本地向量库目录，默认 ./.chroma。
- DEEPSEEK_API_KEY：可选。用于评测，不用于本地诊断主链路。

示例：

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_NUM_GPU=999
OLLAMA_NUM_CTX=1024
CHROMA_DIR=./.chroma
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## GPU 推理确认

可通过以下命令确认是否使用显卡：

```powershell
Invoke-WebRequest http://localhost:11434/api/ps -UseBasicParsing
nvidia-smi
```

判定标准：

- /api/ps 返回中 size_vram 大于 0，表示模型已进入显存。
- nvidia-smi 中看到 ollama.exe 且显存占用上升，表示 GPU 正在参与推理。

## 评测

使用内置样例执行自动评分：

```powershell
python scripts/evaluate_cases.py
```

若配置了 DEEPSEEK_API_KEY，会自动启用 LLM-as-a-Judge 评测。

## 性能调优

- 降低 OLLAMA_NUM_CTX（例如 768）可显著降低延迟。
- 减少检索文本长度和生成长度会加快响应。
- 首次请求通常慢于后续请求（模型与向量检索组件预热）。

## 常见问题

1. 报错 No module named app

- 请在项目根目录运行命令，或使用 python -m scripts.build_kb。

2. 报错 Ollama 模型未找到

- 执行 ollama pull qwen3.5:4b，并确认 .env 里 OLLAMA_MODEL 与本地模型一致。

3. 推理很慢或超时

- 优先检查是否已启用 GPU。
- 适当降低 OLLAMA_NUM_CTX（如 768）。
- 关闭其他占用显卡的高负载程序。

4. 上传 GitHub 时担心泄露密钥

- .env 已加入 .gitignore，不会被提交。
- 请仅提交 .env.example，真实密钥保留在本地。
