from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st

# Ensure imports work no matter where Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.diagnosis_agent import run_diagnosis
from app.evaluation.judge import score_report

st.set_page_config(page_title="动车组空调诊断 Agent", layout="wide")
st.title("动车组客室空调异常诊断 Agent")
st.caption("本地 LLM + RAG + 工具调用")

uploaded = st.file_uploader("上传温度CSV (timestamp,temp_c)", type=["csv"])
question = st.text_area(
    "报修描述",
    value="3号车厢持续升温，乘客反馈闷热，出风量偏小，请给出诊断建议。",
    height=90,
)

if st.button("开始诊断", type="primary"):
    if uploaded is None:
        st.warning("请先上传温度 CSV 文件。")
    else:
        tmp_path = Path("app/data/_uploaded.csv")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())

        try:
            with st.spinner("诊断中..."):
                report = run_diagnosis(question=question, csv_path=tmp_path)
                score = score_report(report)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("诊断报告")
                st.markdown(report)
            with col2:
                st.subheader("自动评测")
                st.metric("准确性", score.accuracy)
                st.metric("可执行性", score.actionability)
                st.metric("安全性", score.safety)
                st.metric("溯源能力", score.traceability)
                st.metric("总分", score.total)
        except Exception as exc:
            st.error(f"诊断失败: {exc}")
            st.info(
                "排查建议: 1) 启动 Ollama 服务; 2) 确认 OLLAMA_BASE_URL; 3) 拉取模型并保持可用。"
            )
