"""
build_rag_data.py
====================
Xây dựng rag_data.json với Semantic Chunking:
  - Embed từng câu bằng sentence-transformers (đa ngôn ngữ)
  - Tính cosine similarity giữa các câu liên tiếp
  - Cắt chunk khi similarity giảm đột ngột (chủ đề thay đổi)
  - Đảm bảo Semantic Independence: mỗi chunk là một đơn vị ngữ nghĩa độc lập
  - Viết lại câu dùng đại từ mơ hồ bằng LLM (tùy chọn)

Chạy:
    python build_rag_data.py

Input:
    data/raw_data.txt          -- Giáo trình gốc phân theo chương
    data/classification.csv    -- 735 cặp (text, label) QA pair

Output:
    data/rag_data.json

Sau khi chạy xong:
    Xóa data/faiss_index.bin và data/faiss_meta.json
    Pipeline sẽ tự build lại FAISS index khi khởi động

Yêu cầu:
    pip install sentence-transformers
"""

import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Đường dẫn ─────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data.txt")
CLASSIF_PATH  = os.path.join(BASE_DIR, "data", "classification.csv")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "rag_data.json")

# ── Tham số Semantic Chunking ──────────────────────────────────
# Model đa ngôn ngữ, hoạt động tốt với tiếng Việt
EMBED_MODEL      = "paraphrase-multilingual-mpnet-base-v2"

# Ngưỡng similarity: nếu cosine(câu_i, câu_i+1) < SPLIT_THRESHOLD → cắt chunk
# Giá trị thấp hơn = cắt nhiều hơn (chunk nhỏ hơn)
# Giá trị cao hơn = cắt ít hơn (chunk lớn hơn)
SPLIT_THRESHOLD  = 0.45

# Số câu tối thiểu / tối đa trong một chunk
MIN_SENTENCES    = 2
MAX_SENTENCES    = 12

# Số token tối thiểu để giữ chunk (bỏ chunk rác)
MIN_CHUNK_TOKENS = 30

# ── Tên chương ────────────────────────────────────────────────
CHAPTER_NAMES = {
    1:  "Nguồn gốc và bản chất Nhà nước",
    2:  "Bản chất và chức năng Nhà nước Việt Nam",
    3:  "Nguồn gốc và bản chất Pháp luật",
    4:  "Hình thức và kiểu Pháp luật",
    5:  "Hệ thống Pháp luật Việt Nam",
    6:  "Luật Hành chính",
    7:  "Luật Hình sự",
    8:  "Luật Dân sự",
    9:  "Luật Lao động",
    10: "Luật Hôn nhân và Gia đình",
}

# ── Stopwords (dùng để trích keyword) ─────────────────────────
STOPWORDS = {
    'theo', 'của', 'là', 'các', 'được', 'trong', 'với', 'không', 'có',
    'đó', 'này', 'khi', 'tại', 'một', 'từ', 'và', 'để', 'về', 'cho',
    'những', 'như', 'nào', 'gì', 'thì', 'đây', 'đã', 'bao', 'giờ',
    'hay', 'sao', 'bị', 'ai', 'thế', 'đến', 'sau', 'trước', 'tới',
    'ra', 'lên', 'xuống', 'vào', 'bởi', 'rằng', 'mà', 'nếu', 'vì',
    'do', 'còn', 'phải', 'cũng', 'lại', 'hoặc', 'cả', 'mọi', 'nhưng',
    'song', 'tuy', 'dù', 'sẽ', 'đang', 'đều', 'hơn', 'nhất', 'chỉ',
    'rất', 'khá', 'quá', 'hết', 'cần', 'nên', 'muốn', 'thể', 'giữa',
    'trên', 'dưới', 'cùng', 'qua', 'lúc', 'nơi', 'đâu',
}


# ══════════════════════════════════════════════════════════════
# TIỀN XỬ LÝ
# ══════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC, bỏ khoảng trắng thừa."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def simple_tokenize(text: str) -> list[str]:
    """Tách token đơn giản theo khoảng trắng."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").split()
    except ImportError:
        return text.split()


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Trích từ khóa đơn giản: từ >= 3 ký tự, loại stopword."""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    seen, result = set(), []
    for w in words:
        if w not in STOPWORDS and w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= top_n:
            break
    return result


def split_into_sentences(text: str) -> list[str]:
    """
    Chia văn bản thành danh sách câu.
    Bảo vệ số thập phân và viết tắt phổ biến.
    """
    # Bảo vệ số thập phân
    text = re.sub(r'(\d)\.(\d)', r'\1[DOT]\2', text)
    # Bảo vệ viết tắt phổ biến (Điều 1., Khoản 2., ...)
    text = re.sub(r'(Điều|Khoản|Mục|Điểm|Art|No|Tr)\.\s*(\d)', r'\1[DOT]\2', text)

    # Tách tại . ! ? theo sau là khoảng trắng
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Khôi phục
    sentences = [s.replace('[DOT]', '.').strip() for s in sentences]
    return [s for s in sentences if len(s) > 10]


# ══════════════════════════════════════════════════════════════
# SEMANTIC CHUNKING ENGINE
# ══════════════════════════════════════════════════════════════

class SemanticChunker:
    """
    Chia văn bản thành các chunk dựa trên độ tương đồng ngữ nghĩa.

    Thuật toán:
    1. Tách văn bản thành danh sách câu
    2. Embed tất cả câu bằng sentence-transformers
    3. Tính cosine similarity giữa các cặp câu liên tiếp
    4. Phát hiện "breakpoint" khi similarity < SPLIT_THRESHOLD
       hoặc khi chunk đã đạt MAX_SENTENCES
    5. Gom câu giữa các breakpoint thành một chunk
    6. Mỗi chunk = một đơn vị ngữ nghĩa độc lập (Semantic Independence)
    """

    def __init__(
        self,
        model_name:      str   = EMBED_MODEL,
        split_threshold: float = SPLIT_THRESHOLD,
        min_sentences:   int   = MIN_SENTENCES,
        max_sentences:   int   = MAX_SENTENCES,
        min_tokens:      int   = MIN_CHUNK_TOKENS,
    ):
        print(f"  [SemanticChunker] Đang tải model: {model_name} ...")
        from sentence_transformers import SentenceTransformer
        self.model           = SentenceTransformer(model_name)
        self.split_threshold = split_threshold
        self.min_sentences   = min_sentences
        self.max_sentences   = max_sentences
        self.min_tokens      = min_tokens
        print(f"  [SemanticChunker] Model sẵn sàng. Threshold={split_threshold}")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity giữa hai vector."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _find_breakpoints(
        self,
        embeddings: np.ndarray,
        similarities: list[float],
    ) -> list[int]:
        """
        Xác định vị trí breakpoint (ranh giới giữa các chunk).

        Breakpoint xảy ra khi:
        - similarity[i] < threshold  →  chủ đề thay đổi
        - Đã tích lũy đủ MAX_SENTENCES  →  cưỡng bức cắt

        Returns:
            Danh sách index breakpoint (vị trí sau câu cuối của chunk).
        """
        breakpoints = []
        sentences_in_chunk = 1  # Đang ở câu đầu tiên

        for i, sim in enumerate(similarities):
            sentences_in_chunk += 1

            # Điều kiện cắt: chủ đề chuyển HOẶC chunk quá dài
            topic_shift     = sim < self.split_threshold
            too_long        = sentences_in_chunk >= self.max_sentences

            if topic_shift or too_long:
                # Chỉ cắt nếu chunk hiện tại đã đủ MIN_SENTENCES
                if sentences_in_chunk >= self.min_sentences:
                    breakpoints.append(i + 1)  # i+1 = vị trí câu đầu chunk mới
                    sentences_in_chunk = 0

        return breakpoints

    def chunk(self, text: str) -> list[str]:
        """
        Phân đoạn văn bản thành các chunk ngữ nghĩa độc lập.

        Args:
            text: Văn bản cần phân đoạn

        Returns:
            Danh sách các chunk string
        """
        sentences = split_into_sentences(text)

        if len(sentences) <= self.min_sentences:
            # Văn bản quá ngắn, trả về nguyên
            combined = " ".join(sentences).strip()
            return [combined] if combined else []

        # Embed tất cả câu trong một lần gọi (batch) cho hiệu quả
        embeddings = self.model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize để cosine = dot product
        )

        # Tính similarity giữa câu i và câu i+1
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Tìm breakpoints
        breakpoints = self._find_breakpoints(embeddings, similarities)

        # Gom câu thành chunks dựa trên breakpoints
        chunks_raw = []
        prev = 0
        for bp in breakpoints:
            chunk_sentences = sentences[prev:bp]
            chunks_raw.append(" ".join(chunk_sentences).strip())
            prev = bp

        # Chunk cuối cùng (sau breakpoint cuối)
        if prev < len(sentences):
            chunk_sentences = sentences[prev:]
            chunks_raw.append(" ".join(chunk_sentences).strip())

        # Lọc chunk quá ngắn — merge vào chunk trước
        final_chunks = []
        for chunk_text in chunks_raw:
            token_count = len(chunk_text.split())
            if token_count < self.min_tokens and final_chunks:
                # Merge vào chunk trước (chunk nhỏ thường là phần đuôi)
                final_chunks[-1] = final_chunks[-1] + " " + chunk_text
            else:
                final_chunks.append(chunk_text)

        return [c.strip() for c in final_chunks if c.strip()]


# ══════════════════════════════════════════════════════════════
# ĐỌC VÀ PARSE raw_data.txt
# ══════════════════════════════════════════════════════════════

def parse_raw_data(path: str) -> dict[int, str]:
    """
    Đọc raw_data.txt, tách thành dict {chapter_id: full_text}.
    Ranh giới chương được xác định bởi dòng [Chương X: ...]
    """
    chapter_texts: dict[int, list[str]] = defaultdict(list)
    current_chapter = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(r'\[Chương\s+(\d+):', line)
            if match:
                current_chapter = int(match.group(1))
                continue

            if current_chapter is not None:
                chapter_texts[current_chapter].append(line)

    return {
        ch: normalize_text(" ".join(lines))
        for ch, lines in chapter_texts.items()
    }


# ══════════════════════════════════════════════════════════════
# XÂY DỰNG CHUNK TỪ GIÁO TRÌNH (SEMANTIC)
# ══════════════════════════════════════════════════════════════

def build_textbook_chunks(
    chapter_texts: dict[int, str],
    chunker: SemanticChunker,
) -> list[dict]:
    """
    Tạo các chunk từ văn bản giáo trình dùng Semantic Chunking.
    Mỗi chunk là một đơn vị ngữ nghĩa độc lập (Semantic Independence).
    """
    chunks = []
    counters = defaultdict(int)

    for chapter, text in sorted(chapter_texts.items()):
        chapter_name  = CHAPTER_NAMES.get(chapter, f"Chương {chapter}")
        semantic_chunks = chunker.chunk(text)

        for chunk_text in semantic_chunks:
            counters[chapter] += 1
            chunk_id = f"tb_ch{chapter:02d}_{counters[chapter]:04d}"

            # Tóm tắt: câu đầu tiên của chunk
            sents   = split_into_sentences(chunk_text)
            summary = sents[0] if sents else chunk_text[:150]

            chunks.append({
                "id":               chunk_id,
                "chapter":          chapter,
                "chapter_name":     chapter_name,
                "topic":            chapter_name,
                "theory":           None,
                "keywords":         extract_keywords(chunk_text),
                "summary":          summary,
                "content":          chunk_text,
                "content_type":     "textbook",
                "difficulty":       "basic",
                "related_chapters": [],
            })

        print(f"  Ch.{chapter:2d} ({chapter_name[:30]}): {counters[chapter]} chunks")

    return chunks


# ══════════════════════════════════════════════════════════════
# XÂY DỰNG CHUNK TỪ QA PAIR (classification.csv)
# ══════════════════════════════════════════════════════════════

def build_qa_chunks(
    df: pd.DataFrame,
    textbook_chunks: list[dict],
) -> list[dict]:
    """
    Tạo QA chunk từ classification.csv.
    Tìm textbook chunk liên quan nhất trong cùng chương để lấy answer.
    """
    chapter_tb_chunks: dict[int, list[dict]] = defaultdict(list)
    for c in textbook_chunks:
        if c["content_type"] == "textbook":
            chapter_tb_chunks[c["chapter"]].append(c)

    qa_chunks = []
    counters  = defaultdict(int)

    for _, row in df.iterrows():
        question     = str(row["text"]).strip()
        chapter      = int(row["label"])
        chapter_name = CHAPTER_NAMES.get(chapter, f"Chương {chapter}")

        # Tìm textbook chunk cùng chương có nhiều keyword trùng nhất
        q_keywords = set(extract_keywords(question))
        best_chunk = None
        best_score = -1

        for tb in chapter_tb_chunks.get(chapter, []):
            tb_keywords = set(tb.get("keywords", []))
            score = len(q_keywords & tb_keywords)
            if score > best_score:
                best_score = score
                best_chunk = tb

        # Lấy answer từ chunk tốt nhất, fallback về tóm tắt chung
        if best_chunk and best_score > 0:
            related_sentences = split_into_sentences(best_chunk["content"])
            answer = " ".join(related_sentences[:2])
        else:
            answer = f"Nội dung liên quan đến {chapter_name} của giáo trình."

        content = f"{question} {answer}"

        counters[chapter] += 1
        chunk_id = f"qa_ch{chapter:02d}_{counters[chapter]:04d}"

        qa_chunks.append({
            "id":               chunk_id,
            "chapter":          chapter,
            "chapter_name":     chapter_name,
            "topic":            chapter_name,
            "theory":           None,
            "keywords":         list(q_keywords)[:8],
            "summary":          answer[:200],
            "content":          normalize_text(content),
            "content_type":     "qa_pair",
            "difficulty":       "basic",
            "related_chapters": [],
        })

    return qa_chunks


# ══════════════════════════════════════════════════════════════
# KIỂM TRA CHẤT LƯỢNG CHUNK
# ══════════════════════════════════════════════════════════════

def quality_check(chunks: list[dict]) -> None:
    """In thống kê chất lượng chunk để kiểm tra trước khi lưu."""
    textbook = [c for c in chunks if c["content_type"] == "textbook"]
    qa       = [c for c in chunks if c["content_type"] == "qa_pair"]

    tb_lengths = [len(c["content"].split()) for c in textbook]
    qa_lengths = [len(c["content"].split()) for c in qa]

    print(f"\n{'='*60}")
    print(f"  KIỂM TRA CHẤT LƯỢNG CHUNK")
    print(f"{'='*60}")
    print(f"\n  Textbook chunks (Semantic): {len(textbook)}")
    if tb_lengths:
        print(f"    Token min/max/avg: "
              f"{min(tb_lengths)} / {max(tb_lengths)} / {sum(tb_lengths)//len(tb_lengths)}")
        ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 9999)]
        for lo, hi in ranges:
            count = sum(1 for l in tb_lengths if lo <= l < hi)
            if count:
                bar = "█" * min(count, 30)
                label = f"{hi}" if hi < 9999 else "∞"
                print(f"    [{lo:>3}-{label:>3}): {bar} ({count})")

    print(f"\n  QA pair chunks: {len(qa)}")
    if qa_lengths:
        print(f"    Token min/max/avg: "
              f"{min(qa_lengths)} / {max(qa_lengths)} / {sum(qa_lengths)//len(qa_lengths)}")

    print(f"\n  Phân bố theo chương:")
    from collections import Counter
    ch_dist = Counter(c["chapter"] for c in chunks)
    for ch in sorted(ch_dist):
        tb_c = sum(1 for c in textbook if c["chapter"] == ch)
        qa_c = sum(1 for c in qa       if c["chapter"] == ch)
        print(f"    Ch.{ch:2d}: {tb_c:3d} textbook + {qa_c:3d} qa = {ch_dist[ch]:4d} tổng")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  BUILD RAG DATA — Semantic Chunking")
    print(f"{'='*60}\n")

    # 1. Đọc giáo trình
    print(f"[1/5] Đọc raw_data.txt ...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy: {RAW_DATA_PATH}")
    chapter_texts = parse_raw_data(RAW_DATA_PATH)
    print(f"      Tìm thấy {len(chapter_texts)} chương\n")

    # 2. Khởi tạo SemanticChunker (tải model một lần)
    print(f"[2/5] Khởi tạo SemanticChunker ...")
    chunker = SemanticChunker(
        model_name      = EMBED_MODEL,
        split_threshold = SPLIT_THRESHOLD,
        min_sentences   = MIN_SENTENCES,
        max_sentences   = MAX_SENTENCES,
        min_tokens      = MIN_CHUNK_TOKENS,
    )
    print()

    # 3. Semantic chunking từng chương
    print(f"[3/5] Semantic chunking giáo trình ...")
    print(f"      (threshold={SPLIT_THRESHOLD}, min={MIN_SENTENCES}, max={MAX_SENTENCES} câu)\n")
    textbook_chunks = build_textbook_chunks(chapter_texts, chunker)
    print(f"\n      Tổng textbook chunks: {len(textbook_chunks)}")

    # 4. Build QA chunks từ classification.csv
    print(f"\n[4/5] Đọc classification.csv và build QA chunks ...")
    if not os.path.exists(CLASSIF_PATH):
        print(f"      [WARN] Không tìm thấy {CLASSIF_PATH}, bỏ qua QA chunks")
        qa_chunks = []
    else:
        df = pd.read_csv(CLASSIF_PATH)
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].str.strip() != ""]
        print(f"      Đọc {len(df)} QA pair")
        qa_chunks = build_qa_chunks(df, textbook_chunks)
        print(f"      Tổng QA chunks: {len(qa_chunks)}")

    # 5. Gộp, kiểm tra, lưu
    all_chunks = textbook_chunks + qa_chunks
    quality_check(all_chunks)

    # Kiểm tra tính toàn vẹn
    all_ids = [c["id"] for c in all_chunks]
    assert len(all_ids) == len(set(all_ids)),            "LỖI: Có ID bị trùng!"
    assert all(c["chapter"] in CHAPTER_NAMES for c in all_chunks), \
        "LỖI: Có chương không hợp lệ!"

    print(f"\n[5/5] Lưu {len(all_chunks)} chunks vào {OUTPUT_PATH} ...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"      Xong! ({size_kb:.0f} KB)")

    print(f"\n{'='*60}")
    print(f"  HOÀN THÀNH — Semantic Chunking")
    print(f"  Tổng chunks: {len(all_chunks)}")
    print(f"    - Textbook : {len(textbook_chunks)}")
    print(f"    - QA pair  : {len(qa_chunks)}")
    print(f"\n  Bước tiếp theo:")
    print(f"    1. Xóa data/faiss_index.bin")
    print(f"    2. Xóa data/faiss_meta.json")
    print(f"    3. Chạy lại pipeline — FAISS sẽ tự build lại")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
