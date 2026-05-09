"""
build_rag_data_v2.py
====================
Mở rộng file RAG gốc (299 chunk cứng) bằng 735 QA chunk
lấy từ classification.csv.

Chạy:
    python3 build_rag_data_v2.py

Input  (cùng thư mục):
    rag_data_original.json – file JSON gốc 299 chunk
    classification.csv     – 735 cặp (text, label)

Output (cùng thư mục):
    rag_data.json          – 1034 chunk = 299 cũ + 735 QA mới  ← FILE DÙNG CHO PRODUCTION

Lưu ý:
    Input và output KHÁC TÊN để tránh ghi đè file gốc.
    Nếu muốn đặt tên khác, chỉnh 2 biến RAG_INPUT và RAG_OUTPUT bên dưới.
"""

import json, re, os
import pandas as pd
from collections import defaultdict

# ── Đường dẫn ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAG_INPUT       = os.path.join(BASE_DIR, 'rag_data_original.json')  # file gốc 299 chunk
CLASSIF_INPUT   = os.path.join(BASE_DIR, 'classification.csv')
RAG_OUTPUT      = os.path.join(BASE_DIR, 'rag_data.json')           # output dùng cho production

# ── Stopwords tiếng Việt (giữ nhỏ gọn) ────────────────────────────────────
STOPWORDS = {
    'theo','của','là','các','được','trong','với','không','có','đó','này','khi',
    'tại','một','từ','và','để','về','cho','những','như','nào','gì','thì','đây',
    'đã','bao','giờ','hay','sao','bị','ai','thế','đến','sau','trước','tới','ra',
    'lên','xuống','vào','bởi','rằng','mà','nếu','vì','do','còn','phải','cũng',
    'lại','hoặc','cả','mọi','nhưng','song','tuy','dù','sẽ','đang','đều','hơn',
    'nhất','chỉ','rất','khá','quá','hết','cần','nên','muốn','thể','giữa','trên',
    'dưới','cùng','qua','lúc','nơi','đâu',
}

def extract_kw(text: str) -> list[str]:
    """Trích từ khóa đơn giản: từ ≥3 ký tự, loại stopword."""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


def find_best_match(question: str, chapter: int,
                    chapter_chunks: dict) -> tuple:
    """
    Tìm chunk cùng chương khớp nhất với câu hỏi (keyword overlap).
    Trả về (best_chunk, score).
    """
    q_kws = set(extract_kw(question))
    if not q_kws:
        return None, 0

    best_score, best_chunk = -1, None
    for chunk in chapter_chunks.get(chapter, []):
        pool = ' '.join([
            chunk.get('content', ''),
            chunk.get('topic', ''),
            ' '.join(chunk.get('keywords', [])),
            chunk.get('summary', ''),
        ]).lower()
        score = len(q_kws & set(extract_kw(pool)))
        if score > best_score:
            best_score, best_chunk = score, chunk

    return best_chunk, best_score


def build_qa_chunks(existing: list, df: pd.DataFrame) -> list:
    """Tạo danh sách QA chunk từ classification.csv."""
    # Index theo chương
    chapter_chunks: dict = defaultdict(list)
    for item in existing:
        chapter_chunks[item['chapter']].append(item)

    qa_chunks = []
    chapter_counters: dict = defaultdict(int)

    for _, row in df.iterrows():
        question = str(row['text']).strip()
        chapter  = int(row['label'])

        best, _ = find_best_match(question, chapter, chapter_chunks)

        if best:
            answer        = best.get('summary', '')
            matched_topic = best.get('topic', '')
            matched_kws   = best.get('keywords', [])
        else:
            answer        = f"Nội dung liên quan đến chương {chapter} của giáo trình."
            matched_topic = ''
            matched_kws   = []

        # Kết hợp keyword câu hỏi + keyword chunk tốt nhất (tối đa 8)
        q_kws    = list(dict.fromkeys(extract_kw(question)))[:5]
        kw_merge = list(dict.fromkeys(q_kws + matched_kws))[:8]

        chapter_counters[chapter] += 1
        qa_id = f"qa_ch{chapter:02d}_{chapter_counters[chapter]:04d}"

        qa_chunks.append({
            "id":               qa_id,
            "chapter":          chapter,
            "topic":            matched_topic,
            "theory":           None,
            "keywords":         kw_merge,
            "summary":          answer,
            "content":          f"Câu hỏi: {question}\nTrả lời: {answer}",
            "content_type":     "qa_pair",
            "difficulty":       "basic",
            "related_chapters": [],
        })

    return qa_chunks


def main():
    # 1. Load dữ liệu gốc
    print(f"Đọc {RAG_INPUT} ...")
    with open(RAG_INPUT, encoding='utf-8') as f:
        existing = json.load(f)
    print(f"  → {len(existing)} chunk gốc")

    print(f"Đọc {CLASSIF_INPUT} ...")
    df = pd.read_csv(CLASSIF_INPUT)
    print(f"  → {len(df)} câu hỏi classification")

    # 2. Tạo QA chunk
    print("Tạo QA chunk...")
    qa_chunks = build_qa_chunks(existing, df)
    print(f"  → {len(qa_chunks)} QA chunk")

    # 3. Gộp và ghi
    combined = existing + qa_chunks
    print(f"Ghi {RAG_OUTPUT} ...")
    with open(RAG_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # 4. Kiểm tra tính toàn vẹn
    old_ids = {item['id'] for item in existing}
    all_ids  = [item['id'] for item in combined]
    assert old_ids.issubset(set(all_ids)), "LỖI: Mất chunk cũ!"
    assert len(all_ids) == len(set(all_ids)), "LỖI: ID bị trùng!"

    # 5. Báo cáo
    from collections import Counter
    size_kb = os.path.getsize(RAG_OUTPUT) / 1024
    ct = Counter(item['content_type'] for item in combined)
    ch = Counter(item['chapter'] for item in combined)

    print(f"\n{'='*50}")
    print(f"✅ Hoàn thành: {len(combined)} chunk → {RAG_OUTPUT} ({size_kb:.0f} KB)")
    print(f"\nChunk cũ (giữ nguyên): {len(existing)}")
    print(f"QA chunk mới:          {len(qa_chunks)}")
    print(f"\ncontent_type:")
    for k, v in sorted(ct.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}")
    print(f"\nPhân bố chương:")
    for chapter in sorted(ch):
        n_old = sum(1 for item in existing if item['chapter'] == chapter)
        n_qa  = ch[chapter] - n_old
        print(f"  Ch{chapter:2d}: {n_old:3d} cũ + {n_qa:3d} QA = {ch[chapter]:4d} tổng")
    print(f"\n✅ Integrity check: OK")
    print(f"{'='*50}")
    print(f"\nBước tiếp theo: rebuild FAISS index với file mới.")


if __name__ == '__main__':
    main()
