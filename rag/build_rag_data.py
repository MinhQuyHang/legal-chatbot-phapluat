"""
build_rag_data.py
====================
Xây dựng rag_data.json với chunking :
  - Sliding window chunking (chunk_size=200 token, overlap=50 token)
  - Luôn cắt tại ranh giới câu, không cắt giữa câu
  - QA pair (classification.csv) được tách riêng thành loại chunk độc lập
  - Tất cả chunk có kích thước tương đối đồng đều
  - Giữ nguyên schema JSON cũ để pipeline không cần sửa

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
"""

import json
import os
import re
import unicodedata
import pandas as pd
from collections import defaultdict

# ── Đường dẫn ─────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data.txt")
CLASSIF_PATH  = os.path.join(BASE_DIR, "data", "classification.csv")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "rag_data.json")

# ── Tham số chunking ──────────────────────────────────────────
CHUNK_SIZE    = 200   # số token tối đa mỗi chunk (tính theo từ sau tách)
OVERLAP_SIZE  = 50    # số token overlap giữa các chunk liền kề
MIN_CHUNK_LEN = 30    # bỏ chunk quá ngắn (ít hơn 30 token)

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
    """
    Tách token đơn giản theo khoảng trắng (sau normalize).
    Dùng underthesea nếu có, fallback về split().
    """
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

            # Phát hiện header chương
            match = re.match(r'\[Chương\s+(\d+):', line)
            if match:
                current_chapter = int(match.group(1))
                continue

            if current_chapter is not None:
                chapter_texts[current_chapter].append(line)

    # Ghép các dòng thành văn bản liên tục cho mỗi chương
    return {
        ch: normalize_text(" ".join(lines))
        for ch, lines in chapter_texts.items()
    }


# ══════════════════════════════════════════════════════════════
# SLIDING WINDOW CHUNKING
# ══════════════════════════════════════════════════════════════

def split_into_sentences(text: str) -> list[str]:
    """
    Chia văn bản thành danh sách câu.
    Cắt tại dấu chấm câu tiếng Việt: . ! ?
    Nhưng giữ nguyên các số thập phân và viết tắt phổ biến.
    """
    # Bảo vệ số thập phân và viết tắt
    text = re.sub(r'(\d)\.(\d)', r'\1[DOT]\2', text)

    # Tách câu tại . ! ? theo sau là khoảng trắng + chữ hoa/chữ thường
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Khôi phục
    sentences = [s.replace('[DOT]', '.').strip() for s in sentences]
    return [s for s in sentences if s]


def sliding_window_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
    min_len: int = MIN_CHUNK_LEN
) -> list[str]:
    """
    Chia văn bản thành các chunk bằng sliding window.

    Thuật toán:
    1. Tách văn bản thành danh sách câu
    2. Tokenize từng câu để đếm token
    3. Gom câu vào chunk cho đến khi đủ chunk_size token
    4. Chunk tiếp theo bắt đầu từ vị trí (chunk_size - overlap) token trước
    5. Không bao giờ cắt giữa câu

    Returns:
        List các chuỗi chunk, mỗi chunk có kích thước gần đồng đều.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Tokenize từng câu một lần
    tokenized = [simple_tokenize(s) for s in sentences]
    token_counts = [len(t) for t in tokenized]

    chunks = []
    start_sent = 0  # index câu bắt đầu của chunk hiện tại

    while start_sent < len(sentences):
        # Gom câu vào chunk hiện tại
        current_tokens = 0
        end_sent = start_sent

        while end_sent < len(sentences):
            current_tokens += token_counts[end_sent]
            end_sent += 1
            if current_tokens >= chunk_size:
                break

        # Tạo nội dung chunk
        chunk_text = " ".join(sentences[start_sent:end_sent]).strip()

        if len(simple_tokenize(chunk_text)) >= min_len:
            chunks.append(chunk_text)

        # Tính vị trí bắt đầu chunk tiếp theo (có overlap)
        # Lùi lại overlap token tính từ cuối chunk hiện tại
        overlap_tokens = 0
        next_start = end_sent
        for i in range(end_sent - 1, start_sent - 1, -1):
            overlap_tokens += token_counts[i]
            if overlap_tokens >= overlap:
                next_start = i
                break

        # Tránh vòng lặp vô hạn nếu không tiến được
        if next_start <= start_sent:
            next_start = start_sent + 1

        start_sent = next_start

    return chunks


# ══════════════════════════════════════════════════════════════
# XÂY DỰNG CHUNK TỪ GIÁO TRÌNH
# ══════════════════════════════════════════════════════════════

def build_textbook_chunks(chapter_texts: dict[int, str]) -> list[dict]:
    """
    Tạo các chunk từ văn bản giáo trình dùng sliding window.
    Mỗi chunk giữ đúng schema cũ để pipeline không cần sửa.
    """
    chunks = []
    counters = defaultdict(int)

    for chapter, text in sorted(chapter_texts.items()):
        chapter_name = CHAPTER_NAMES.get(chapter, f"Chương {chapter}")
        window_chunks = sliding_window_chunk(text)

        for chunk_text in window_chunks:
            counters[chapter] += 1
            chunk_id = f"tb_ch{chapter:02d}_{counters[chapter]:04d}"

            # Tóm tắt đơn giản: lấy câu đầu tiên
            first_sentence = split_into_sentences(chunk_text)
            summary = first_sentence[0] if first_sentence else chunk_text[:100]

            chunks.append({
                "id":               chunk_id,
                "chapter":          chapter,
                "chapter_name":     chapter_name,
                "topic":            chapter_name,
                "theory":           None,
                "keywords":         extract_keywords(chunk_text),
                "summary":          summary,
                "content":          chunk_text,
                "content_type":     "textbook",   # phân biệt với qa_pair
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
    textbook_chunks: list[dict]
) -> list[dict]:
    """
    Tạo QA chunk từ classification.csv.
    Khác v2: QA pair được đánh tag riêng, không ghép lẫn với textbook chunk.
    Tìm textbook chunk liên quan nhất trong cùng chương để lấy answer.
    """
    # Index textbook chunk theo chương để tìm nhanh
    chapter_tb_chunks: dict[int, list[dict]] = defaultdict(list)
    for c in textbook_chunks:
        if c["content_type"] == "textbook":
            chapter_tb_chunks[c["chapter"]].append(c)

    qa_chunks = []
    counters = defaultdict(int)

    for _, row in df.iterrows():
        question = str(row["text"]).strip()
        chapter  = int(row["label"])
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
            # Lấy 2 câu đầu của chunk liên quan nhất
            related_sentences = split_into_sentences(best_chunk["content"])
            answer = " ".join(related_sentences[:2])
        else:
            answer = f"Nội dung liên quan đến {chapter_name} của giáo trình."

        # Nội dung chunk là câu hỏi + câu trả lời (rõ ràng hơn v2)
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
            "content_type":     "qa_pair",   # tag riêng
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

    tb_lengths = [len(simple_tokenize(c["content"])) for c in textbook]
    qa_lengths = [len(simple_tokenize(c["content"])) for c in qa]

    print(f"\n{'='*60}")
    print(f"  KIỂM TRA CHẤT LƯỢNG CHUNK")
    print(f"{'='*60}")
    print(f"\n  Textbook chunks: {len(textbook)}")
    if tb_lengths:
        print(f"    Token min/max/avg: {min(tb_lengths)} / {max(tb_lengths)} / {sum(tb_lengths)//len(tb_lengths)}")
        # Phân phối kích thước
        ranges = [(0,50),(50,100),(100,150),(150,200),(200,300),(300,9999)]
        for lo, hi in ranges:
            count = sum(1 for l in tb_lengths if lo <= l < hi)
            if count:
                bar = "█" * min(count, 30)
                print(f"    [{lo:>3}-{hi if hi<9999 else '∞':>3}): {bar} ({count})")

    print(f"\n  QA pair chunks: {len(qa)}")
    if qa_lengths:
        print(f"    Token min/max/avg: {min(qa_lengths)} / {max(qa_lengths)} / {sum(qa_lengths)//len(qa_lengths)}")

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
    print(f"  BUILD RAG DATA V3 — Sliding Window Chunking")
    print(f"{'='*60}\n")

    # 1. Đọc giáo trình
    print(f"[1/4] Đọc raw_data.txt ...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy: {RAW_DATA_PATH}")
    chapter_texts = parse_raw_data(RAW_DATA_PATH)
    print(f"      Tìm thấy {len(chapter_texts)} chương\n")

    # 2. Build textbook chunks bằng sliding window
    print(f"[2/4] Sliding window chunking (size={CHUNK_SIZE}, overlap={OVERLAP_SIZE})...")
    textbook_chunks = build_textbook_chunks(chapter_texts)
    print(f"\n      Tổng textbook chunks: {len(textbook_chunks)}")

    # 3. Build QA chunks từ classification.csv
    print(f"\n[3/4] Đọc classification.csv và build QA chunks ...")
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

    # 4. Gộp và kiểm tra
    all_chunks = textbook_chunks + qa_chunks
    quality_check(all_chunks)

    # 5. Kiểm tra tính toàn vẹn
    all_ids = [c["id"] for c in all_chunks]
    assert len(all_ids) == len(set(all_ids)), "LỖI: Có ID bị trùng!"
    assert all(c["chapter"] in CHAPTER_NAMES for c in all_chunks), "LỖI: Có chương không hợp lệ!"

    # 6. Lưu file
    print(f"\n[4/4] Lưu {len(all_chunks)} chunks vào {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"      Xong! ({size_kb:.0f} KB)")

    print(f"\n{'='*60}")
    print(f"  HOÀN THÀNH")
    print(f"  Tổng chunks: {len(all_chunks)}")
    print(f"    - Textbook: {len(textbook_chunks)}")
    print(f"    - QA pair : {len(qa_chunks)}")
    print(f"\n  Bước tiếp theo:")
    print(f"    1. Xóa data/faiss_index.bin")
    print(f"    2. Xóa data/faiss_meta.json")
    print(f"    3. Chạy lại pipeline — FAISS sẽ tự build lại")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
