"""
llm/prompts.py
Tất cả prompt templates dùng trong chatbot
"""

from langchain_core.prompts import PromptTemplate

# ── 1. Hàm hỗ trợ xử lý dữ liệu JSON (Metadata) ────────────────
def format_rag_docs(docs):
    formatted_texts = []
    for doc in docs:
        chapter = doc.metadata.get("chapter", "Không rõ")
        topic = doc.metadata.get("topic", "Không rõ chủ đề")
        content = doc.page_content
        formatted_doc = f"[Chương {chapter} - Chủ đề: {topic}]\nNội dung: {content}"
        formatted_texts.append(formatted_doc)
    return "\n\n".join(formatted_texts)


# ── 2. System prompt (Định hình nhân vật) ──────────────────────
SYSTEM_TEMPLATE = """Bạn là giảng viên môn Pháp luật Đại cương tại Đại học Mở TP.HCM.

Nhiệm vụ của bạn:
- Giải thích các khái niệm pháp lý một cách rõ ràng, dễ hiểu với văn phong sư phạm.
- Dựa HOÀN TOÀN vào "Tài liệu tham khảo" được cung cấp để trả lời.
- Nếu có thể, hãy nhắc đến thông tin tài liệu nằm ở Chương nào, Chủ đề nào để sinh viên dễ tra cứu.
- TUYỆT ĐỐI KHÔNG bịa đặt thông tin pháp lý. Nếu tài liệu không có thông tin, hãy trả lời chính xác câu này: "Dựa trên giáo trình hiện tại, thầy/cô chưa có thông tin chính xác về vấn đề này." """


# ── 3. RAG QA prompt (Hỏi đáp 1 lượt) ──────────────────────────
RAG_QA_TEMPLATE = """Bạn hãy đóng vai giảng viên (được định nghĩa trong System Prompt) để trả lời câu hỏi sau.

Tài liệu tham khảo:
{context}

---
Câu hỏi của sinh viên: {question}

Trả lời:"""

RAG_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_QA_TEMPLATE,
)


# ── 4. Conversation + RAG prompt ─────────
CONV_RAG_TEMPLATE = """Bạn là trợ lý học tập môn Pháp luật Đại cương của Đại học Mở TP.HCM.
Sinh viên hỏi: "{question}"

Hãy làm theo các bước sau để trả lời:
1. Đọc kỹ [TÀI LIỆU THAM KHẢO] bên dưới.
2. TÓM TẮT ý chính liên quan đến câu hỏi bằng 2-3 câu, dùng NGÔN NGỮ CỦA BẠN.
3. TUYỆT ĐỐI KHÔNG lặp lại nguyên văn bất kỳ câu nào trong tài liệu.
4. Trả lời như đang giảng bài cho sinh viên năm nhất.

Nếu tài liệu không có thông tin, hãy trả lời: "Dựa trên giáo trình hiện tại, tôi chưa tìm thấy thông tin chính xác về vấn đề này."

[LỊCH SỬ HỘI THOẠI]
{chat_history}

[TÀI LIỆU THAM KHẢO]
{context}

Trợ lý trả lời (bằng lời lẽ tự nhiên của bạn):"""

CONV_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=CONV_RAG_TEMPLATE,
)
CONV_RAG_TEMPLATE_STRING = CONV_RAG_TEMPLATE


# ==============================================================================
# PROMPT dùng cho PhoGPT
# ==============================================================================
SYSTEM_PROMPT = (
    "Bạn là trợ lý học tập môn Pháp luật Đại cương - Đại học Mở TP.HCM.\n"
    "Chỉ trả lời dựa trên TÀI LIỆU THAM KHẢO được cung cấp.\n"
    "Trả lời ngắn gọn, súc tích, 3-4 câu. Không copy nguyên văn tài liệu.\n"
    "Nếu tài liệu không có thông tin: trả lời 'Nội dung này chưa có trong giáo trình.'\n"
)

def build_phogpt_prompt(
    system: str = SYSTEM_PROMPT,
    context: str = "",
    history: list = None,
    question: str = ""
) -> str:
    parts = []

    # System context
    parts.append(system)
    parts.append("")

    # Lịch sử (chỉ user turn)
    # if history:
    #   for user_msg, _ in history[-2:]:
    #        parts.append(f"<s>USER: {user_msg}")
    #        parts.append("ASSISTANT: [Đã trả lời]</s>")
    #    parts.append("")

    # RAG context + câu hỏi — format PhoGPT
    user_turn = (
        f"Tài liệu:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Trả lời câu hỏi dựa vào tài liệu trên:"
    )
    parts.append(user_turn)
    # parts.append("ASSISTANT:")  

    return "\n".join(parts)