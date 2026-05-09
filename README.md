# LawChat — Hệ thống Hỏi đáp Pháp luật Đại cương dựa trên RAG

Hệ thống hỏi đáp tự động cho môn Pháp luật Đại cương tại Trường Đại học Mở TP.HCM, xây dựng trên kiến trúc Retrieval-Augmented Generation (RAG) 3 tầng. Hệ thống truy xuất các đoạn văn bản liên quan từ giáo trình gốc và sinh câu trả lời được ràng buộc hoàn toàn vào ngữ cảnh đó, không phụ thuộc vào bất kỳ API bên ngoài nào.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Chi tiết từng tầng](#chi-tiết-từng-tầng)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cấu hình](#cấu-hình)
- [Hạn chế đã biết](#hạn-chế-đã-biết)

---

## Tổng quan

Chatbot tổng quát không phù hợp cho bài toán hỏi đáp pháp luật vì hai lý do cốt lõi: chúng có xu hướng bịa đặt thông tin pháp lý (hallucination) và không thể trích dẫn nguồn gốc cụ thể từ giáo trình. LawChat giải quyết vấn đề này bằng cách ràng buộc toàn bộ quá trình sinh câu trả lời vào các đoạn văn bản được truy xuất trực tiếp từ giáo trình, đồng thời hiển thị minh bạch nguồn trích dẫn cho mỗi câu trả lời.

Toàn bộ quá trình suy luận chạy cục bộ qua Ollama. Không có dữ liệu nào được gửi ra ngoài máy chủ.

---

## Kiến trúc hệ thống

```
Câu hỏi người dùng
        |
        v
[Tầng 1] PhoBERT Classifier
    - Dự đoán chương (1-10)
    - Tính confidence và margin
    - Quyết định phạm vi tìm kiếm: Scoped hoặc Global
        |
        v
[Tầng 2] Hybrid Retrieval
    - Dense: multilingual-e5-large + FAISS (1.034 vectors)
    - Sparse: BM25 với tokenizer underthesea
    - Điểm tổng hợp: alpha * dense + (1 - alpha) * bm25
    - Lọc theo chương khi ở chế độ Scoped
        |
        v
[Tầng 3] Sinh câu trả lời
    - PhoGPT-4B qua Ollama (định dạng ChatML)
    - Stop tokens ngăn model sinh thêm lượt hội thoại
    - Giới hạn 800 ký tự/chunk để kiểm soát context window
    - Hậu xử lý loại bỏ artifact định dạng
        |
        v
Câu trả lời + Danh sách đoạn tham khảo
```

---

## Chi tiết từng tầng

### Tầng 1 — Phân loại câu hỏi (PhoBERT)

Bộ phân loại sử dụng `vinai/phobert-base` được fine-tune trên 735 câu hỏi có nhãn, bao phủ 10 chương của giáo trình. Văn bản đầu vào được tách từ bằng `underthesea` trước khi tokenize, nhất quán với cách pre-train của PhoBERT.

Với mỗi câu hỏi, bộ phân loại sinh ra phân phối xác suất trên 10 lớp. Hai giá trị dẫn xuất điều khiển hành vi ở tầng sau:

- `confidence` — xác suất của lớp được dự đoán cao nhất
- `margin` — khoảng cách giữa xác suất top-1 và top-2

Khi cả hai vượt ngưỡng tương ứng (`CONFIDENCE_THRESHOLD`, `MARGIN_THRESHOLD`), pipeline chuyển sang chế độ **Scoped**: chỉ các chunk thuộc chương được dự đoán mới được đưa vào tính điểm. Ngược lại, hệ thống fallback sang chế độ **Global**, tìm kiếm toàn bộ 1.034 chunk. Cơ chế này tránh thu hẹp phạm vi quá mức khi bộ phân loại chưa đủ tự tin.

Các ngưỡng được chọn qua grid search trên tập validation, tối ưu hóa tích của tỷ lệ Scoped và độ chính xác top-2.

### Tầng 2 — Truy xuất lai (Hybrid Retrieval)

Kho tri thức gồm 1.034 chunk văn bản trích từ giáo trình, được bổ sung thêm 735 cặp hỏi-đáp dạng FAQ để cải thiện recall trên các câu hỏi ngắn, trực tiếp.

**Dense retrieval** mã hóa câu hỏi và đoạn văn bằng `intfloat/multilingual-e5-large` (prefix câu hỏi: `query:`, prefix đoạn văn: `passage:`). Embedding được lưu trong FAISS `IndexFlatIP` và tính sẵn khi khởi động. Độ tương đồng là cosine trên vector đã L2-normalize.

**Sparse retrieval** dùng BM25 (`rank-bm25`) trên token đã tách từ. Phương pháp này bổ trợ cho dense retriever với các câu hỏi chứa thuật ngữ pháp lý cụ thể, số điều luật, hoặc tên riêng mà embedding có thể không nắm bắt tốt.

Điểm tổng hợp được tính theo công thức:

```
hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score_normalized
```

Điểm BM25 được min-max normalize theo từng câu hỏi trước khi kết hợp. Trọng số `alpha` và các tham số khác (`score_gate`, `chapter_boost`) được tối ưu tự động qua `evaluate_rag.py`.

Ở chế độ Scoped, hệ số nhân `chapter_boost` được áp dụng lên điểm của các chunk thuộc chương được dự đoán trước khi xếp hạng cuối.

### Tầng 3 — Sinh câu trả lời (PhoGPT)

Các đoạn văn được truy xuất ghép thành một khối ngữ cảnh và đưa vào prompt theo định dạng ChatML. System prompt yêu cầu model chỉ trả lời dựa trên ngữ cảnh được cung cấp, và chỉ được phép trả lời đúng một câu fallback cố định nếu ngữ cảnh không đủ thông tin. Các ràng buộc bổ sung ngăn model sinh HTML, JSON, hoặc các câu xin lỗi kiểu AI-disclaimer mà model gốc có xu hướng tạo ra.

Stop token (`<|im_end|>`, `<|im_start|>`) ngăn model tiếp tục sinh thêm các lượt hội thoại mới. Mỗi chunk được cắt ngắn còn 800 ký tự trước khi đưa vào prompt để kiểm soát độ trễ và tránh vượt quá context window hiệu dụng.

Đầu ra được lọc qua bộ hậu xử lý loại bỏ các artifact định dạng còn sót lại.

---

## Kết quả đánh giá

Đánh giá thực hiện trên 28 câu hỏi soạn riêng, bao phủ đủ 10 chương (tính đến 09/05/2026, với corpus 1.034 chunk và hybrid search).

### Độ chính xác truy xuất

| Chỉ số | Giá trị |
|---|---|
| Keyword Hit@1 | 89,29% |
| Chapter Hit@1 | 53,57% |
| Chapter Hit@3 | 82,14% |
| Chapter Hit@5 | 92,86% |
| MRR | 0,678 |
| Điểm hybrid trung bình (top-1) | 0,748 |

### Phân bổ chiến lược tìm kiếm

| Chế độ | Số câu | Tỷ lệ |
|---|---|---|
| Scoped | 2 | 7,1% |
| Global | 26 | 92,9% |

Tỷ lệ Global cao phản ánh xu hướng của bộ phân loại hiện tại thường không vượt ngưỡng confidence trên tập đánh giá. Điều này được kỳ vọng với tập train 735 mẫu trên 10 lớp và tính chất đa dạng của câu hỏi pháp luật.

### Tham số tối ưu từ grid search

| Tham số | Giá trị |
|---|---|
| `score_gate` | 0,25 |
| `chapter_boost` | 1,10 |
| `alpha` | 0,60 |
| Hit@3 | 82,14% |
| MRR | 0,655 |
| Điểm tổng hợp (0,6 x Hit@3 + 0,4 x MRR) | 0,755 |

---

## Cấu trúc dự án

```
.
├── app/
│   ├── streamlit_app.py      # Giao diện web Streamlit
│   └── pipeline.py           # Singleton wrapper thread-safe cho RAGPipeline
├── config/
│   ├── settings.yaml         # Tham số runtime (tự động cập nhật từ grid search)
│   └── settings.py           # Đọc và load config
├── data/
│   ├── rag_data.json         # 1.034 chunk (giáo trình + cặp FAQ)
│   ├── classification.csv    # 735 câu hỏi có nhãn để train PhoBERT
│   ├── faiss_index.bin       # FAISS index đã build sẵn
│   ├── faiss_meta.json       # Metadata căn chỉnh theo thứ tự FAISS index
│   └── sample_questions.txt  # 50 câu hỏi mẫu để test batch
├── llm/
│   ├── chain.py              # LegalChatChain (ChatML, stop tokens, streaming)
│   ├── prompts.py            # Prompt template và system prompt
│   └── memory.py             # Quản lý lịch sử hội thoại
├── model/
│   ├── phobert_classifier/   # Trọng số PhoBERT đã fine-tune
│   ├── predict.py            # PhoBERTPredictor với interface predict_safe
│   ├── train_model.py        # Script fine-tune (thiết kế cho Colab T4)
│   └── train_meta.json       # Ngưỡng và chỉ số đánh giá đã lưu
├── rag/
│   ├── rag_pipeline.py       # Pipeline truy xuất chính (FAISS, BM25, hybrid)
│   └── evaluate_rag.py       # Đánh giá retrieval và grid search tham số
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## Cài đặt

Yêu cầu: Python 3.10 trở lên, Ollama đã cài đặt và đang chạy.

```bash
git clone https://github.com/MinhQuyHang/legal-chatbot-phapluat.git
cd legal-chatbot-phapluat
pip install -r requirements.txt
```

Tải mô hình ngôn ngữ:

```bash
ollama pull mrjacktung/phogpt-4b-chat-gguf
```

FAISS index (`faiss_index.bin`) và metadata (`faiss_meta.json`) được build tự động lần đầu chạy nếu chưa có, không cần thao tác thủ công.

### Huấn luyện lại PhoBERT (tùy chọn)

Nếu cần train lại bộ phân loại sau khi bổ sung dữ liệu:

1. Upload `model/train_model.py` và `data/classification.csv` lên Google Colab với GPU T4.
2. Chạy `!python model/train_model.py`.
3. Download `phobert_classifier.zip` về và giải nén vào thư mục `model/`.

---

## Hướng dẫn sử dụng

Giao diện web:

```bash
streamlit run app/streamlit_app.py
```

CLI tương tác:

```bash
python main.py
```

CLI batch:

```bash
python main.py --demo              # 10 câu hỏi mẫu có sẵn
python main.py --sample            # 50 câu từ sample_questions.txt
python main.py -q "Luật dân sự là gì?"
```

Chạy đánh giá retrieval và grid search:

```bash
python rag/evaluate_rag.py
```

Lệnh này tự động cập nhật `config/settings.yaml` với bộ tham số tốt nhất tìm được.

---

## Cấu hình

Toàn bộ tham số có thể điều chỉnh trong `config/settings.yaml`.

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `confidence_threshold` | 0,45 | Ngưỡng confidence tối thiểu để kích hoạt chế độ Scoped |
| `margin_threshold` | 0,15 | Khoảng cách tối thiểu giữa xác suất top-1 và top-2 |
| `score_gate` | 0,25 | Điểm hybrid tối thiểu để một chunk được chấp nhận |
| `chapter_boost` | 1,10 | Hệ số nhân điểm cho chunk thuộc chương được dự đoán |
| `alpha` | 0,60 | Trọng số của dense score trong công thức hybrid |
| `max_chunk_length` | 800 | Số ký tự tối đa của mỗi chunk đưa vào prompt LLM |

Sau khi thay đổi kho tri thức hoặc bộ phân loại, nên chạy lại `evaluate_rag.py` để tối ưu lại các tham số này.

---

## Hạn chế đã biết

**Bộ phân loại kích hoạt Scoped thấp.** Với 735 mẫu train trên 10 lớp, PhoBERT chỉ kích hoạt chế độ Scoped trên khoảng 7% câu hỏi trong tập đánh giá. Câu hỏi diễn đạt theo nhiều cách hoặc liên quan đến nhiều chương thường rơi vào Global search.

**Giới hạn context window.** PhoGPT-4B có context window hiệu dụng hạn chế. Với 3 chunk tối đa 800 ký tự mỗi chunk, các câu hỏi đòi hỏi tổng hợp thông tin từ nhiều mục trong giáo trình có thể nhận được câu trả lời chưa đầy đủ.

**Hành vi của model ngôn ngữ.** PhoGPT-4B là model được instruction-tune với các xu hướng có sẵn như sinh câu AI-disclaimer và artifact định dạng. System prompt hiện tại hạn chế phần lớn các hành vi này nhưng không loại bỏ hoàn toàn.

**Recall truy xuất chưa cao ở top-1.** Chapter Hit@3 đạt 82% có nghĩa là khoảng 1 trong 5 câu hỏi trong tập đánh giá không truy xuất được chunk đúng chương trong top 3 kết quả. Đây là bottleneck chính ảnh hưởng đến chất lượng câu trả lời.

**Không có cơ chế kiểm chứng câu trả lời.** Hệ thống không xác minh tính nhất quán của câu trả lời được sinh ra với các đoạn văn đã truy xuất. Hallucination trong phạm vi ngữ cảnh được cung cấp là hiếm nhưng không được phát hiện tự động.

---

*Phát triển bởi nhóm sinh viên Trường Đại học Mở TP.HCM, 2026.*
