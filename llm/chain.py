"""
llm/chain.py
LegalChatChain – ChatML prompt, stop tokens, giới hạn chunk, streaming,
"""

import json
import logging
import re
import time
import requests
from typing import Dict, List, Generator

from langchain_community.llms.ollama import Ollama

from llm.prompts import SYSTEM_PROMPT, build_chatml_prompt
from llm.memory import get_recent_history

logger = logging.getLogger(__name__)


class LegalChatChain:
    """
    Chain chính: RAG → PhoGPT → Answer
    Hỗ trợ non‑streaming (ask) và streaming (ask_stream).
    """

    def __init__(
        self,
        rag_pipeline,
        model_name: str = "mrjacktung/phogpt-4b-chat-gguf",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        top_k_docs: int = 3,
        memory_k: int = 5,
        max_chunk_length: int = 800,   # None = không giới hạn
        streaming: bool = False,
    ):
        self.rag = rag_pipeline
        self.top_k_docs = top_k_docs
        self.memory_k = memory_k
        self.max_chunk_length = max_chunk_length
        self.streaming = streaming
        self.history = []              # List of (question, answer)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self._last_result = None       # lưu kết quả cuối của ask_stream

        # LLM với stop tokens (đã thêm "[Tài liệu")
        logger.info(f"Loading LLM: {model_name} via Ollama at {ollama_base_url}")
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>"]
        )

        logger.info("LegalChatChain initialized (ChatML, stop tokens, max_chunk=%s)", max_chunk_length)

    # ── Hàm làm sạch đầu ra (thêm cắt [Tài liệu) ─────────────
    @staticmethod
    def _clean_response(text: str) -> str:
        """Loại bỏ token ChatML, HTML, JSON rác và cắt bỏ phần [Tài liệu thừa."""
        # 1. Xóa token ChatML
        for token in ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]:
            text = text.replace(token, "")

        # 2. Xóa tất cả thẻ HTML
        text = re.sub(r'<[^>]+>', '', text)

        # 3. Xóa dấu ngoặc nhọn đơn độc (JSON hỏng)
        text = text.replace('{', '').replace('}', '')

        # 4. Nếu câu trả lời bắt đầu bằng "Dang" (dấu hiệu của streaming lỗi),
        #    cắt bỏ cho đến khi gặp "Theo giáo trình" hoặc "Chào" hoặc "Dựa trên"
        if text.lstrip().startswith("Dang"):
            match = re.search(r'(Theo giáo trình|Chào bạn|Dựa trên)', text)
            if match:
                text = text[match.start():]
            else:
                lines = text.split('\n')
                if len(lines) > 1:
                    text = '\n'.join(lines[1:])

        # 5. Cắt bỏ toàn bộ phần từ [Tài liệu trở đi (phòng trường hợp stop token không kịp)
        #if "[Tài liệu" in text:
        #    text = text.split("[Tài liệu")[0].strip()

        # 6. Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()

        # 7. Fallback nếu kết quả rỗng
        if not text:
            text = "Theo giáo trình hiện tại, tôi chưa tìm thấy thông tin chính xác về vấn đề này."

        return text

    # ── Non‑streaming ─────────────────────────────────────────
    def ask(self, question: str) -> Dict:
        start = time.time()
        logger.info(f"Processing: {question[:60]}")

        # 1. RAG
        rag_result = self.rag.answer(question, top_k=self.top_k_docs, verbose=False)
        passages   = rag_result.get("passages", [])
        confidence = rag_result.get("confidence", 0.0)
        route_mode = rag_result.get("route_mode", "global")

        # 2. Format context (có giới hạn chunk)
        context = self._format_context(passages)

        # 3. Lấy lịch sử hội thoại gần nhất
        recent = get_recent_history(self.history, self.memory_k)

        # 4. Tạo prompt ChatML
        prompt_text = build_chatml_prompt(
            system=SYSTEM_PROMPT,
            context=context,
            history=recent,
            question=question,
        )

        # 5. Gọi LLM
        try:
            raw_answer = self.llm.invoke(prompt_text).strip()
            answer = self._clean_response(raw_answer)
            self.history.append((question, answer))
        except Exception as e:
            logger.error(f"Chain error: {e}")
            answer = "Xin lỗi, trợ lý đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

        latency = time.time() - start
        logger.info(f"Done in {latency:.2f}s | confidence={confidence:.2f}")

        return {
            "question":   question,
            "answer":     answer,
            "sources":    passages,
            "confidence": confidence,
            "route_mode": route_mode,
            "latency":    round(latency, 2),
        }

    # ── Streaming (đã sửa: không yield dict nữa, lưu vào _last_result) ─
    def ask_stream(self, question: str) -> Generator:
        start = time.time()
        logger.info(f"Streaming: {question[:60]}")

        # 1. RAG
        rag_result = self.rag.answer(question, top_k=self.top_k_docs, verbose=False)
        passages   = rag_result.get("passages", [])
        confidence = rag_result.get("confidence", 0.0)
        route_mode = rag_result.get("route_mode", "global")

        # 2. Format context
        context = self._format_context(passages)

        # 3. Lấy lịch sử
        recent = get_recent_history(self.history, self.memory_k)

        # 4. Tạo ChatML prompt
        prompt_text = build_chatml_prompt(
            system=SYSTEM_PROMPT,
            context=context,
            history=recent,
            question=question,
        )

        # 5. Gửi request stream
        tokens = []
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm.model,
                    "prompt": prompt_text,
                    "stream": True,
                    "options": {
                        "temperature": self.llm.temperature,
                        "stop": ["<|im_end|>", "<|im_start|>", "[Tài liệu"],
                    },
                },
                stream=True,
                timeout=None,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        tokens.append(token)
                        yield token
                    if data.get("done"):
                        break

        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            fallback = self.ask(question)
            yield fallback["answer"]
            tokens = [fallback["answer"]]
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            yield "\nKhông thể kết nối Ollama."
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\nLỗi: {str(e)}"

        full_answer = "".join(tokens).strip()
        full_answer = self._clean_response(full_answer)

        if full_answer:
            self.history.append((question, full_answer))

        latency = time.time() - start
        logger.info(f"Streaming done in {latency:.2f}s")

        # LƯU kết quả cuối cùng thay vì yield
        self._last_result = {
            "question":   question,
            "answer":     full_answer,
            "sources":    passages,
            "confidence": confidence,
            "route_mode": route_mode,
            "latency":    round(latency, 2),
        }
        # KHÔNG yield dict

    # ── Helpers ────────────────────────────────────────────────
    def reset(self) -> None:
        self.history = []
        self._last_result = None
        logger.info("Memory cleared")

    def _format_context(self, passages: List[Dict]) -> str:
        if not passages:
            return "(Không tìm thấy tài liệu liên quan trong giáo trình)"
        parts = []
        for i, p in enumerate(passages, 1):
            content = p.get("content", "")
            if self.max_chunk_length and len(content) > self.max_chunk_length:
                content = content[:self.max_chunk_length] + "…"
            ch = p.get("chapter", "?")
            topic = p.get("chapter_name", "Chủ đề không xác định")
            parts.append(f"(Đoạn {i} – Chương {ch}: {topic})\n{content}")
        return "\n\n".join(parts)

    def __call__(self, question: str) -> str:
        return self.ask(question)["answer"]