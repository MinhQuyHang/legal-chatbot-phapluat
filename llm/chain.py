"""
llm/chain.py
LegalChatChain – Plain-text prompt cho PhoGPT-4B
"""

import json
import logging
import re
import time
import requests
from typing import Dict, List, Generator

from langchain_community.llms.ollama import Ollama

from llm.prompts import SYSTEM_PROMPT, build_phogpt_prompt   # ← đổi tên hàm
from llm.memory import get_recent_history

logger = logging.getLogger(__name__)


class LegalChatChain:
    """
    Chain chính: RAG → PhoGPT-4B → Answer
    Hỗ trợ non-streaming (ask) và streaming (ask_stream).
    """

    def __init__(
        self,
        rag_pipeline,
        model_name: str = "mrjacktung/phogpt-4b-chat-gguf",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.3,       # giảm để ít hallucinate hơn
        top_k_docs: int = 3,
        memory_k: int = 2,
        max_chunk_length: int = 600,    #RAG context đủ dài
        streaming: bool = False,
    ):
        self.rag = rag_pipeline
        self.top_k_docs = top_k_docs
        self.memory_k = memory_k
        self.max_chunk_length = max_chunk_length
        self.streaming = streaming
        self.history = []               # List of (question, answer)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self._last_result = None

        logger.info(f"Loading LLM: {model_name} via Ollama at {ollama_base_url}")
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        logger.info(
            "LegalChatChain initialized | plain-text prompt | "
            "max_chunk=%s | memory_k=%s", max_chunk_length, memory_k
        )

    # ── Markers để phát hiện prompt bị echo lại trong output ─────
    LEAK_MARKERS = [
        "Không được copy",
        "Không nhắc lại",
        "Không dùng thông tin",
        "Chỉ dựa vào",
        "<|im_start|>",
        "<|im_end|>",
        "<s>USER:",
        "ASSISTANT:",
        "USER:",                    
        "Tài liệu tham khảo:", 
        "TÀI LIỆU THAM KHẢO:",    
    ]

    # ── Làm sạch đầu ra ──────────────────────────────────────────
    @staticmethod
    def _clean_response(text: str) -> str:
        # 1. Xóa token ChatML
        for token in ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]:
            text = text.replace(token, "")

        # 2. Cắt tại marker sớm nhất bị leak
        earliest_cut = len(text)
        for marker in LegalChatChain.LEAK_MARKERS:
            idx = text.find(marker)
            if idx != -1 and idx < earliest_cut:
                earliest_cut = idx
        if earliest_cut < len(text):
            text = text[:earliest_cut].strip()

        # 3. Xóa "Trả lời:" ở đầu output
        text = re.sub(r'^\s*Trợ lý trả lời\s*:\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^\s*Trả lời\s*:\s*', '', text, flags=re.IGNORECASE).strip()

        # 4. Xóa thẻ HTML
        text = re.sub(r'<[^>]+>', '', text)

        # 5. Cắt nếu model lặp lại tài liệu nguồn
        if "[Tài liệu" in text:
            text = text.split("[Tài liệu")[0].strip()

        # 6. Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()

        # 7. Fallback nếu output rỗng
        if not text:
            text = "Theo giáo trình hiện tại, tôi chưa tìm thấy thông tin chính xác về vấn đề này."

        return text

    # ── Phát hiện response bị copy từ history ────────────────────
    def _is_hallucinated_repeat(self, answer: str) -> bool:
        """
        Trả về True nếu answer overlap > 70% với một câu trả lời cũ trong history.
        """
        if not self.history:
            return False
        answer_words = set(answer.lower().split())
        if len(answer_words) < 5:
            return False
        for _, old_answer in self.history:
            old_words = set(old_answer.lower().split())
            if not old_words:
                continue
            overlap = len(answer_words & old_words) / len(old_words)
            if overlap > 0.7:
                logger.warning("Phát hiện repeat từ history (overlap=%.0f%%). Fallback.", overlap * 100)
                return True
        return False

    # ── Fallback từ RAG passages khi LLM bị repeat ───────────────
    @staticmethod
    def _fallback_from_passages(passages: List[Dict]) -> str:
        if not passages:
            return "Nội dung này chưa có trong giáo trình."
        best = passages[0]
        content = best.get("content", "")
        ch = best.get("chapter", "?")
        ch_name = best.get("chapter_name", "")
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        summary = " ".join(sentences[:2]) if sentences else content[:200]
        return f"(Chương {ch} – {ch_name}) {summary}"

    # ── Non-streaming ─────────────────────────────────────────────
    def ask(self, question: str) -> Dict:
        start = time.time()
        logger.info("Processing: %s", question[:80])

        # 1. RAG
        rag_result = self.rag.answer(question, top_k=self.top_k_docs, verbose=False)

        if rag_result.get("rejected"):
            return {
                "question": question,
                "answer": rag_result["message"],
                "sources": [],
                "confidence": rag_result["confidence"],
                "route_mode": "rejected",
                "latency": 0.0,
            }

        passages   = rag_result.get("passages", [])
        confidence = rag_result.get("confidence", 0.0)
        route_mode = rag_result.get("route_mode", "global")

        # 2. Format context từ passages
        context = self._format_context(passages)


        # 3. Lấy lịch sử hội thoại gần nhất
        recent = get_recent_history(self.history, self.memory_k)

        # 4. Tạo plain-text prompt (KHÔNG dùng ChatML)
        prompt_text = build_phogpt_prompt(
            system=SYSTEM_PROMPT,
            context=context,
            history=recent,
            question=question,
        )

        # log prompt để kiểm tra model nhận được gì
        logger.debug("=== PROMPT GỬI VÀO LLM ===\n%s\n=== END PROMPT ===", prompt_text)

        # 5. Gọi LLM
        try:
            raw_answer = self.llm.invoke(prompt_text).strip()
            logger.debug("=== RAW OUTPUT ===\n%s\n=== END OUTPUT ===", raw_answer)

            answer = self._clean_response(raw_answer)

            if self._is_hallucinated_repeat(answer):
                answer = self._fallback_from_passages(passages)

            self.history.append((question, answer))

        except Exception as e:
            logger.error("Chain error: %s", e)
            answer = "Xin lỗi, trợ lý đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

        latency = time.time() - start
        logger.info("Done in %.2fs | confidence=%.2f", latency, confidence)

        return {
            "question":   question,
            "answer":     answer,
            "sources":    passages,
            "confidence": confidence,
            "route_mode": route_mode,
            "latency":    round(latency, 2),
        }

    # ── Streaming ─────────────────────────────────────────────────
    def ask_stream(self, question: str) -> Generator:
        start = time.time()
        logger.info("Streaming: %s", question[:80])

        # 1. RAG
        rag_result = self.rag.answer(question, top_k=self.top_k_docs, verbose=False)

        logger.info("RAG result: rejected=%s | confidence=%.3f | passages=%d | route=%s",
        rag_result.get("rejected"),
        rag_result.get("confidence", 0),
        len(rag_result.get("passages", [])),
        rag_result.get("route_mode")
    )
        
        

        if rag_result.get("rejected"):
            message = rag_result["message"]
            self._last_result = {
                "question": question,
                "answer": message,
                "sources": [],
                "confidence": rag_result["confidence"],
                "route_mode": "rejected",
                "latency": 0.0,
            }
            yield message
            return
        passages   = rag_result.get("passages", [])
        confidence = rag_result.get("confidence", 0.0)
        route_mode = rag_result.get("route_mode", "global")

        # 2. Format context
        context = self._format_context(passages)
        logger.info("CONTEXT:\n%s", context)

        # 3. Lịch sử
        recent = get_recent_history(self.history, self.memory_k)

        # 4. Plain-text prompt
        prompt_text = build_phogpt_prompt(
            system=SYSTEM_PROMPT,
            context=context,
            history=recent,
            question=question,
        )
        logger.info("PROMPT:\n%s", prompt_text)

        # 5. Stream từ Ollama
        tokens: List[str] = []
        stream_buffer = ""
        stream_stopped = False

        def _check_leak(buf: str):
            """
            Kiểm tra buffer có chứa leak marker không.
            Trả về (phần safe để yield, should_stop).
            Giữ lại 40 ký tự cuối buffer phòng marker bị split qua token.
            """
            for marker in LegalChatChain.LEAK_MARKERS:
                idx = buf.find(marker)
                if idx != -1:
                    return buf[:idx], True
            return buf, False

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm.model,
                    "prompt": prompt_text,
                    "stream": True,
                    "options": {
                        "temperature": self.llm.temperature,
                        "num_predict": 512,
                        "stop": ["<|im_end|>", "<|im_start|>"],
                    },
                    "raw": True,
                },
                stream=True,
                timeout=None,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if stream_stopped:
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("response", "")
                if token:
                    stream_buffer += token
                    safe_text, should_stop = _check_leak(stream_buffer)
                    if safe_text:
                        tokens.append(safe_text)
                        yield safe_text
                        stream_buffer = stream_buffer[len(safe_text):]
                    if should_stop:
                        stream_stopped = True
                        break

                if data.get("done"):
                    # Flush buffer còn lại
                    if stream_buffer:
                        tokens.append(stream_buffer)
                        yield stream_buffer
                    break

        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout — fallback to non-streaming ask()")
            fallback = self.ask(question)
            yield fallback["answer"]
            tokens = [fallback["answer"]]

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.ollama_base_url)
            yield "\nKhông thể kết nối Ollama. Vui lòng kiểm tra server."

        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield f"\nLỗi không xác định: {str(e)}"

        # Hậu xử lý sau khi stream xong
        full_answer = "".join(tokens).strip()
        full_answer = self._clean_response(full_answer)

        # if self._is_hallucinated_repeat(full_answer):
        #    full_answer = self._fallback_from_passages(passages)

        if full_answer:
            self.history.append((question, full_answer))

        latency = time.time() - start
        logger.info("Streaming done in %.2fs", latency)

        self._last_result = {
            "question":           question,
            "answer":             full_answer,
            "sources":            passages,
            "passages":           passages,   
            "confidence":         confidence,
            "route_mode":         route_mode,
            "predicted_chapters": route_mode != "rejected" and passages  
                                and [passages[0]["chapter"]] or [],
            "latency":            round(latency, 2),
        }

    # ── Helpers ───────────────────────────────────────────────────
    def reset(self) -> None:
        """Xóa toàn bộ lịch sử hội thoại."""
        self.history = []
        self._last_result = None
        logger.info("Memory cleared")

    def _format_context(self, passages: List[Dict]) -> str:
        """Format danh sách passages thành plain text đưa vào prompt."""
        if not passages:
            return "(Không tìm thấy tài liệu liên quan trong giáo trình)"
        parts = []
        for p in passages:
            content = p.get("content", "")
            if self.max_chunk_length and len(content) > self.max_chunk_length:
                content = content[:self.max_chunk_length] + "…"
            parts.append(content)
        return "\n\n".join(parts)

    def __call__(self, question: str) -> str:
        return self.ask(question)["answer"]
