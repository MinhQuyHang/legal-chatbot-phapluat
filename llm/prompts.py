"""
llm/prompts.py
Tất cả prompt templates dùng trong chatbot
"""

from langchain_core.prompts import PromptTemplate

# ── 1. Hàm hỗ trợ xử lý dữ liệu JSON (Metadata) ────────────────
def format_rag_docs(docs):
    """
    Hàm này nhận đầu vào là danh sách các tài liệu (Document) lấy ra từ VectorDB.
    Nó sẽ bóc tách metadata (chapter, topic) từ file JSON và ráp nối với nội dung.
    """
    formatted_texts = []
    for doc in docs:
        # Lấy thông tin từ metadata (nếu không có thì để trống)
        chapter = doc.metadata.get("chapter", "Không rõ")
        topic = doc.metadata.get("topic", "Không rõ chủ đề")
        
        # Lấy nội dung chính (trường "content" trong JSON của bạn)
        content = doc.page_content
        
        # Ráp lại thành một đoạn văn bản rõ ràng cho PhoGPT đọc
        formatted_doc = f"[Chương {chapter} - Chủ đề: {topic}]\nNội dung: {content}"
        formatted_texts.append(formatted_doc)
    
    # Nối các tài liệu bằng 2 dấu xuống dòng để tách biệt rõ ràng
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
# ChatML PROMPT dùng cho LegalChatChain
# ==============================================================================
SYSTEM_PROMPT = (
    "Bạn là trợ lý học tập môn Pháp luật Đại cương - Đại học Mở TP.HCM.\n"
    "Chỉ trả lời dựa trên tài liệu được cung cấp.\n"
    "Trả lời ngắn gọn, súc tích, 3-4 câu. Không copy nguyên văn tài liệu.\n"
    "Nếu tài liệu không có thông tin: trả lời 'Nội dung này chưa có trong giáo trình.'\n"
    "Không lặp lại câu hỏi. Chỉ văn bản thuần."
)

def build_chatml_prompt(
    system: str = SYSTEM_PROMPT,
    context: str = "",
    history: list[tuple[str, str]] = None,
    question: str = ""
) -> str:
    """
    Xây dựng prompt chuẩn ChatML cho PhoGPT.
    """
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]

    if history:
        for user_msg, assistant_msg in history:
            parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")

    user_prompt = (
        f"Tài liệu tham khảo:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Yêu cầu: Tóm tắt trong 3-4 câu dựa trên tài liệu trên. "
        f"Không được copy nguyên văn tài liệu. Trả lời trực tiếp, không nhắc lại câu hỏi."
    )
    parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)