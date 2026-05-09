"""
app/streamlit_app.py
Chatbot Hỏi Đáp Môn Pháp Luật Đại Cương
Trường Đại học Mở Thành phố Hồ Chí Minh
"""

import os, sys, time, html
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rag.rag_pipeline import RAGPipeline
    from llm.chain import LegalChatChain
    from config.settings import (
        RAG_DATA_PATH, PHOBERT_MODEL_PATH,
        PHOGPT_MODEL, OLLAMA_BASE_URL,
        LLM_TEMPERATURE, LLM_MEMORY_K, RAG_TOP_K,
    )
    _USE_LANGCHAIN = True
except ImportError:
    from app.pipeline import answer_dict
    _USE_LANGCHAIN = False

import json as _json, os as _os
_META_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "model", "train_meta.json"
)
_train_meta = {}
if _os.path.exists(_META_PATH):
    with open(_META_PATH, "r", encoding="utf-8") as _f:
        _train_meta = _json.load(_f)

_TRAIN_COUNT = _train_meta.get("splits", {}).get("train", "1.283")
_DOC_COUNT   = _train_meta.get("rag_docs", 299)
_HIT_RATE    = _train_meta.get("hit_rate_top3", 78)

def escape_html(text: str) -> str:
    """Escape HTML entities, giữ xuống dòng."""
    return html.escape(str(text))


# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Chatbot Pháp Luật Đại Cương – OU",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# CSS – Flex layout toàn trang, sticky input, dark theme
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

html, body, #root, .stApp { height: 100%; margin: 0; padding: 0; overflow: hidden; }
.stApp { background: #0f1117; display: flex; flex-direction: column; }

header { visibility: visible !important; background: #0f1117; flex-shrink: 0; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

.ou-header {
    text-align: center; padding: 1.2rem 0 0.8rem;
    border-bottom: 1px solid #1e2130; flex-shrink: 0;
}
.ou-school { font-size: 0.7rem; font-weight: 500; color: #8b949e; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.3rem; }
.ou-title { font-size: 1.3rem; font-weight: 600; color: #f0f0f0; }
.ou-sub { font-size: 0.8rem; color: #6e7681; margin-top: 0.2rem; }

.chat-wrap {
    flex: 1; overflow-y: auto; padding: 0.5rem 1rem 0;
    display: flex; flex-direction: column;
}

.msg-user { display: flex; justify-content: flex-end; margin: 0.8rem 0; }
.msg-user-bubble {
    background: #1f6feb; color: white; border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem; max-width: 75%; font-size: 0.9rem;
    line-height: 1.6; box-shadow: 0 2px 8px rgba(31,111,235,0.25);
}

.msg-bot { display: flex; gap: 0.8rem; margin: 0.8rem 0; align-items: flex-start; }
.bot-avatar {
    width: 30px; height: 30px; background: #2a2d35; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px; font-size: 0.7rem; color: #8b949e; font-weight: 600;
}
.bot-body { flex: 1; max-width: calc(100% - 40px); }

.bot-meta {
    font-size: 0.7rem; color: #8b949e; margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
}
.chapter-tag {
    background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb44;
    border-radius: 4px; padding: 0.1rem 0.4rem; font-size: 0.65rem; font-weight: 500;
}
.tech-badge {
    font-size: 0.6rem; color: #22d3ee; background: #0c1a20;
    border: 1px solid #164e63; border-radius: 4px; padding: 0.05rem 0.35rem;
}
.tech-badge-amber {
    color: #fbbf24; background: #1c1409; border-color: #78350f;
}
.tech-badge-route {
    color: #d2a843; background: #2a2415; border: 1px solid #5c4a1f;
    border-radius: 4px; padding: 0.05rem 0.35rem; font-size: 0.6rem;
}

.answer-box {
    background: #161923; border: 1px solid #2a2d3a;
    border-radius: 4px 14px 14px 14px; padding: 0.8rem 1rem;
    font-size: 0.9rem; color: #d2d6dc; line-height: 1.65;
    white-space: pre-wrap; margin-bottom: 0.6rem;
}

.source-item {
    background: #111318; border: 1px solid #1e2130;
    border-radius: 6px; padding: 0.5rem 0.8rem; margin-top: 0.3rem;
    font-size: 0.8rem; color: #8b949e; line-height: 1.55;
}
.source-header {
    font-size: 0.65rem; font-weight: 600; color: #6b7280;
    display: flex; justify-content: space-between; margin-bottom: 0.2rem;
}

.typing-cursor::after {
    content: "▌"; animation: blink 1s step-end infinite; color: #1f6feb;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

.loading-dots { display: flex; gap: 4px; padding: 0.5rem 0; }
.loading-dot {
    width: 6px; height: 6px; background: #8b949e; border-radius: 50%;
    animation: bounce 1.4s infinite;
}
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%,60%,100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-5px); opacity: 1; }
}

.input-area {
    position: sticky; bottom: 0; background: #0f1117;
    border-top: 1px solid #1e2130; padding: 0.8rem 0 1rem;
    flex-shrink: 0; z-index: 10;
}

.stTextInput > div > div > input {
    background: #161923 !important; border: 1px solid #2a2d3a !important;
    border-radius: 10px !important; padding: 0.7rem 1rem !important;
    font-size: 0.9rem !important; color: #e5e7eb !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1f6feb !important; box-shadow: 0 0 0 2px rgba(31,111,235,0.25) !important;
}
.stTextInput > label { display: none !important; }

.stButton > button {
    border-radius: 10px !important; background: #1f6feb !important;
    border: none !important; color: white !important; font-weight: 500 !important;
    padding: 0.7rem 1.3rem !important; font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #1158c7 !important; }
.stButton > button:disabled { background: #2a2d35 !important; color: #8b949e !important; }

section[data-testid="stSidebar"] {
    background: #0d0f14 !important; border-right: 1px solid #1e2130 !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.sb-title {
    font-size: 0.7rem; font-weight: 600; color: #8b949e !important;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px solid #1e2130;
}
.stSelectbox > div > div {
    background: #161923 !important; border: 1px solid #2a2d3a !important;
    border-radius: 8px !important; color: #e5e7eb !important; font-size: 0.85rem !important;
}
.stToggle label { font-size: 0.8rem !important; color: #c9d1d9 !important; }

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.8rem; }
.stat-card {
    background: #161923; border: 1px solid #2a2d3a;
    border-radius: 8px; padding: 0.55rem 0.7rem;
}
.stat-value { font-size: 1rem; font-weight: 600; color: #e6edf3; }
.stat-label { font-size: 0.6rem; color: #8b949e; margin-top: 0.1rem; }

.empty-state { text-align: center; padding: 3rem 1rem; color: #6e7681; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# LOAD PIPELINE (cache)
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Đang tải mô hình PhoBERT & E5…")
def _load_chain():
    if not _USE_LANGCHAIN:
        return None
    rag = RAGPipeline(
        rag_data_path=str(RAG_DATA_PATH),
        phobert_model_path=str(PHOBERT_MODEL_PATH),
    )
    chain = LegalChatChain(
        rag_pipeline=rag,
        model_name=PHOGPT_MODEL,
        ollama_base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        top_k_docs=RAG_TOP_K,
        memory_k=LLM_MEMORY_K,
    )
    return chain

chain = _load_chain()

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
if "messages"   not in st.session_state: st.session_state.messages   = []
if "total_q"    not in st.session_state: st.session_state.total_q    = 0
if "prefill"    not in st.session_state: st.session_state.prefill    = ""
if "tech_mode"  not in st.session_state: st.session_state.tech_mode  = False
if "processing" not in st.session_state: st.session_state.processing = False

prefill = st.session_state.get("prefill", "")
if prefill:
    st.session_state._prefill_temp = prefill
    st.session_state.prefill = ""
real_prefill = st.session_state.get("_prefill_temp", "")
if "_prefill_temp" in st.session_state:
    del st.session_state["_prefill_temp"]


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0;text-align:center;border-bottom:1px solid #1e2130;margin-bottom:1rem">
      <div style="font-weight:700;font-size:1rem;color:#f0f0f0">Pháp Luật Đại Cương</div>
      <div style="font-size:0.7rem;color:#8b949e;margin-top:0.2rem">Đại học Mở TP.HCM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-title">📚 Câu hỏi theo chương</div>', unsafe_allow_html=True)

    CHAPTERS = {
        "Chương 1 – Nhà nước": [
            "Nhà nước là gì?", "Nguồn gốc nhà nước theo Mác-Lênin?",
            "Chức năng đối nội của nhà nước?", "Bộ máy nhà nước gồm những cơ quan nào?",
        ],
        "Chương 2 – Bộ máy NN VN": [
            "Quốc hội Việt Nam có chức năng gì?", "Chủ tịch nước có quyền hạn gì?",
            "Chính phủ là cơ quan gì?", "UBND có nhiệm vụ gì?",
        ],
        "Chương 3 – Pháp luật": [
            "Pháp luật là gì?", "Quy phạm pháp luật là gì?",
            "Quan hệ pháp luật là gì?", "Năng lực pháp luật là gì?",
        ],
        "Chương 4 – Hình thức PL": [
            "Hình thức pháp luật là gì?", "Tập quán pháp là gì?",
            "Văn bản quy phạm pháp luật là gì?", "Hiệu lực của văn bản pháp luật?",
        ],
        "Chương 5 – Hệ thống PL VN": [
            "Hệ thống pháp luật Việt Nam gồm gì?", "Vi phạm pháp luật là gì?",
            "Trách nhiệm pháp lý là gì?", "Cấu thành vi phạm pháp luật?",
        ],
        "Chương 6 – Luật Hành chính": [
            "Luật hành chính điều chỉnh gì?", "Vi phạm hành chính là gì?",
            "Xử phạt vi phạm hành chính?",
        ],
        "Chương 7 – Luật Hình sự": [
            "Tội phạm là gì?", "Trách nhiệm hình sự là gì?",
            "Hình phạt gồm những loại nào?",
        ],
        "Chương 8 – Luật Dân sự": [
            "Luật dân sự điều chỉnh gì?", "Hợp đồng dân sự có hiệu lực khi nào?",
            "Thừa kế theo pháp luật là gì?",
        ],
        "Chương 9 – Luật Lao động": [
            "Hợp đồng lao động là gì?", "Quyền của người lao động?",
            "Tranh chấp lao động là gì?",
        ],
        "Chương 10 – Luật Hôn nhân": [
            "Điều kiện kết hôn hợp lệ?", "Ly hôn dựa trên căn cứ nào?",
            "Quyền nuôi con sau ly hôn?",
        ],
    }

    selected = st.selectbox("Chọn chương", list(CHAPTERS.keys()), label_visibility="collapsed")
    for q in CHAPTERS[selected]:
        if st.button(q, key=f"q_{q}"):
            st.session_state.prefill = q
            st.rerun()

    st.markdown("---")

    st.markdown(f"""
    <div style="font-size:0.8rem;color:#8b949e;line-height:2">
      Câu hỏi đã hỏi: <b style="color:#e6edf3">{st.session_state.total_q}</b><br>
      Dữ liệu huấn luyện: <b style="color:#e6edf3">{_TRAIN_COUNT} câu</b><br>
      Kho kiến thức: <b style="color:#e6edf3">{_DOC_COUNT} đoạn / 10 chương</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    tech_mode = st.toggle("🔧 Hiện thông tin kỹ thuật", value=st.session_state.tech_mode)
    st.session_state.tech_mode = tech_mode

    if tech_mode:
        _test_f1 = _train_meta.get("test_macro_f1", "—")
        _f1_show = f"{_test_f1:.2f}" if isinstance(_test_f1, float) else str(_test_f1)

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-value">{_f1_show}</div><div class="stat-label">Macro F1</div></div>
            <div class="stat-card"><div class="stat-value">{_HIT_RATE}%</div><div class="stat-label">Hit Rate (top-3)</div></div>
            <div class="stat-card"><div class="stat-value">{_TRAIN_COUNT}</div><div class="stat-label">Câu Training</div></div>
            <div class="stat-card"><div class="stat-value">{_DOC_COUNT}</div><div class="stat-label">Đoạn</div></div>
        </div>
        <div style="font-size:0.65rem;color:#8b949e;margin-top:0.5rem">
            PhoBERT + E5 Multilingual<br>PhoGPT-4B via Ollama
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_q = 0
        if chain: chain.reset()
        st.rerun()


# ════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="ou-header">
  <div class="ou-school">Trường Đại học Mở Thành phố Hồ Chí Minh</div>
  <div class="ou-title">Chatbot Hỏi Đáp Môn Pháp Luật Đại Cương</div>
  <div class="ou-sub">Hệ thống trả lời câu hỏi tự động dành cho sinh viên</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-wrap" id="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:2rem;margin-bottom:0.5rem">⚖️</div>
      <div style="font-size:1rem;font-weight:600;color:#8b949e">Bắt đầu đặt câu hỏi</div>
      <div style="font-size:0.85rem;color:#6e7681;margin-top:0.3rem">
        Chọn câu hỏi gợi ý bên trái hoặc nhập câu hỏi bên dưới
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            safe_content = escape_html(msg['content'])
            st.markdown(f"""<div class="msg-user"><div class="msg-user-bubble">{safe_content}</div></div>""",
                        unsafe_allow_html=True)
        else:
            data = msg.get("data", {})
            passages = data.get("passages") or data.get("sources") or []
            chs = data.get("predicted_chapters", [])
            ch_num = chs[0] if chs else (passages[0]["chapter"] if passages else "?")
            conf = data.get("confidence", 0.0)
            lat_ms = msg.get("latency_ms", 0)
            lat_label = f"{lat_ms:.0f}ms" if lat_ms < 1000 else f"{lat_ms/1000:.1f}s"
            route_mode = data.get("route_mode", "global")
            llm_answer = msg.get("content", "").strip()

            meta_html = f'<span class="chapter-tag">Chương {ch_num}</span>'
            if st.session_state.tech_mode:
                meta_html += f'<span class="tech-badge">conf {conf:.2f}</span>'
                meta_html += f'<span class="tech-badge-amber">{lat_label}</span>'
                meta_html += f'<span class="tech-badge-route">{route_mode}</span>'

            st.markdown(f"""
            <div class="msg-bot">
                <div class="bot-avatar">PL</div>
                <div class="bot-body">
                    <div class="bot-meta">{meta_html}</div>
            """, unsafe_allow_html=True)

            if llm_answer:
                safe_llm = escape_html(llm_answer)
                st.markdown(f"""<div class="answer-box">{safe_llm}</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="answer-box" style="color:#6e7681;">Hệ thống chưa tạo được câu trả lời.</div>""",
                            unsafe_allow_html=True)

            if passages:
                with st.expander(f"📚 Nguồn tham khảo ({len(passages)} đoạn)"):
                    for p in passages:
                        sp = int(min(p.get("score", 0), 1.0) * 100)
                        ch_name = escape_html(str(p.get("chapter_name", "N/A")))
                        content = escape_html(str(p.get("content", "")))
                        st.markdown(f"""
                        <div class="source-item">
                            <div class="source-header">
                                <span>Ch.{p['chapter']} – {ch_name}</span>
                                <span>Độ liên quan: {sp}%</span>
                            </div>
                            <div>{content}</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

# Input sticky
st.markdown('<div class="input-area">', unsafe_allow_html=True)
col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "q", value=real_prefill,
        placeholder="Nhập câu hỏi về Pháp Luật Đại Cương…",
        label_visibility="collapsed",
        key="q_main",
        disabled=st.session_state.processing,
    )
with col_btn:
    send = st.button("Gửi", use_container_width=True, type="primary", disabled=st.session_state.processing)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # đóng chat-wrap


# ════════════════════════════════════════════════════════════════
# HANDLE SUBMIT 
# ════════════════════════════════════════════════════════════════
if send and question.strip() and not st.session_state.processing:
    q = question.strip()
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": q})

    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("""
        <div class="msg-bot">
            <div class="bot-avatar">PL</div>
            <div class="bot-body">
                <div class="bot-meta"><span class="chapter-tag">Đang xử lý...</span></div>
                <div class="answer-box">
                    <div class="loading-dots">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    t0 = time.perf_counter()
    full_answer = ""
    final_result = None

    try:
        if chain and hasattr(chain, 'ask_stream'):
            loading_placeholder.empty()
            stream_placeholder = st.empty()
            # Stream từng token ra giao diện
            for chunk in chain.ask_stream(q):
                full_answer += chunk
                stream_placeholder.markdown(f"""
                <div class="msg-bot">
                    <div class="bot-avatar">PL</div>
                    <div class="bot-body">
                        <div class="bot-meta">
                            <span class="chapter-tag">Đang trả lời...</span>
                        </div>
                        <div class="answer-box typing-cursor">{escape_html(full_answer)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Sau khi streaming kết thúc, lấy kết quả từ _last_result
            final_result = chain._last_result
            if final_result is None:
                # fallback: dùng full_answer thô
                answer = full_answer.strip()
                passages = []
                conf = 0.0
                route_mode = "global"
            else:
                answer = final_result["answer"]
                passages = final_result["sources"]
                conf = final_result["confidence"]
                route_mode = final_result["route_mode"]
            stream_placeholder.empty()
        else:
            # Non‑streaming
            if chain:
                result = chain.ask(q)
            else:
                raw = answer_dict(q, top_k=3)
                passages = raw.get("passages", [])
                answer = passages[0].get("content", "") if passages else ""
                result = {
                    "answer": answer,
                    "sources": passages,
                    "confidence": raw.get("confidence", 0.0),
                    "route_mode": raw.get("route_mode", "global"),
                }
            answer = result.get("answer", "")
            passages = result.get("sources", [])
            conf = result.get("confidence", 0.0)
            route_mode = result.get("route_mode", "global")
            loading_placeholder.empty()
    except Exception as e:
        answer = f"Đã xảy ra lỗi khi xử lý câu hỏi: {escape_html(str(e))}"
        passages = []
        conf = 0.0
        route_mode = "error"
        loading_placeholder.empty()

    bot_msg = {
        "role": "bot",
        "content": answer,
        "data": {
            "passages": passages,
            "sources": passages,
            "confidence": conf,
            "route_mode": route_mode,
            "predicted_chapters": [passages[0]["chapter"]] if passages else [],
        },
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
    st.session_state.messages.append(bot_msg)
    st.session_state.total_q += 1
    st.session_state.processing = False
    st.rerun()