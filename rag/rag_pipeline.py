"""
rag_pipeline.py 
"""

import os, json, threading, unicodedata, re, time, logging, yaml
import numpy as np
import faiss

from rank_bm25 import BM25Okapi

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.45   # lấy từ train_meta.json (PhoBERT)
MARGIN_THRESHOLD     = 0.15   # lấy từ train_meta.json
SCORE_GATE           = 0.25
CHAPTER_BOOST        = 1.10
MAX_SCORE            = 2.0

E5_MODEL_NAME  = "intfloat/multilingual-e5-large"
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_META_FILE  = "faiss_meta.json"

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
CHAPTER_KEYWORDS = {
        7:  ["tội phạm", "hình phạt", "tù", "hình sự", "tố tụng", "bị cáo"],
        10: ["ly hôn", "kết hôn", "hôn nhân", "vợ", "chồng", "nuôi con"],
        9:  ["lao động", "hợp đồng lao động", "tiền lương", "sa thải"],
        8:  ["dân sự", "thừa kế", "tài sản", "hợp đồng dân sự"],
    }

try:
    from underthesea import word_tokenize as _wt
    def _segment(text: str) -> str:
        return _wt(text, format="text")
    _HAS_SEG = True
except ImportError:
    def _segment(text: str) -> str:
        return text
    _HAS_SEG = False

def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]",
        " ", text
    )
    if _HAS_SEG:
        text = _segment(text)
    return re.sub(r"\s+", " ", text).strip()

def _e5_query(text: str) -> str:
    return f"query: {text}"

def _e5_passage(text: str) -> str:
    return f"passage: {text}"

def _load_config():
    """Đọc tham số RAG từ config/settings.yaml nếu có, trả về dict."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("rag", {})

# ==============================================================================
# RAG PIPELINE CLASS
# ==============================================================================
class RAGPipeline:
    def __init__(
        self,
        rag_data_path: str,
        phobert_model_path: str,
        top_k: int = 3,
        e5_model_name: str = E5_MODEL_NAME,
        cache_dir: str = None,
        alpha: float = 0.6,
        use_hybrid: bool = True,
    ):
        self.top_k   = top_k
        self.use_hybrid = use_hybrid
        self._lock   = threading.Lock()

        # Đọc cấu hình từ settings.yaml
        cfg = _load_config()
        self.score_gate    = cfg.get("score_gate", SCORE_GATE)
        self.chapter_boost = cfg.get("chapter_boost", CHAPTER_BOOST)
        self.alpha         = cfg.get("alpha", alpha)

        print("=" * 65)
        print(" Khởi tạo RAG Pipeline [Hybrid: E5 + BM25 + FAISS]")
        print("=" * 65)
        print(f"   score_gate={self.score_gate}  chapter_boost={self.chapter_boost}  alpha={self.alpha}")

        # 1. Classifier
        print(f"\n[1/5] Classifier: PhoBERT @ {phobert_model_path}")
        from model.predict import PhoBERTPredictor
        self._predictor = PhoBERTPredictor(model_dir=phobert_model_path, device="cpu")
        print("      [OK] PhoBERT classifier loaded")

        # 2. Load dữ liệu
        print(f"\n[2/5] RAG data: {os.path.basename(rag_data_path)}")
        with open(rag_data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.all_docs     : list[str] = []
        self.all_chapters : list[int] = []
        self.chapters     : dict[int, list[int]] = {}
        self.idx2meta     : list[dict] = []

        for item in raw:
            ch = int(item["chapter"])
            content = item["content"].strip()
            if not content:
                continue
            idx = len(self.all_docs)
            self.all_docs.append(content)
            self.all_chapters.append(ch)
            self.chapters.setdefault(ch, []).append(idx)
            self.idx2meta.append({
                "chapter": ch,
                "chapter_name": CHAPTER_NAMES.get(ch, f"Ch.{ch}"),
                "content": content,
                "summary": item.get("summary", ""),
                "keywords": item.get("keywords", []),
                "difficulty": item.get("difficulty", ""),
            })
        total = len(self.all_docs)
        print(f"      [OK] {total} đoạn | {len(self.chapters)} chương")

        # 3. Xây dựng BM25 index với tokenizer tiếng Việt (nếu có)
        print(f"\n[3/5] Xây dựng BM25 index...")
        if _HAS_SEG:
            tokenized_docs = [preprocess(doc).split() for doc in self.all_docs]
        else:
            tokenized_docs = [doc.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("      [OK] BM25 index built")

        # 4. Load E5 model
        print(f"\n[4/5] Load E5 model: {e5_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self._e5 = SentenceTransformer(e5_model_name)
            print("      [OK] E5 model loaded")
        except ImportError:
            raise ImportError("Thiếu sentence-transformers!\nChạy: pip install sentence-transformers faiss-cpu")

        # 5. Build hoặc load FAISS index
        print(f"\n[5/5] Build/Load FAISS index...")
        cache_dir = cache_dir or os.path.dirname(rag_data_path)
        self.index_path = os.path.join(cache_dir, FAISS_INDEX_FILE)
        self.meta_path  = os.path.join(cache_dir, FAISS_META_FILE)
        self._build_or_load_index()
        print(f"      [OK] FAISS index ready: {self.index.ntotal} vectors\n")
        print(f"      [OK] Hybrid search: E5 (alpha={self.alpha}) + BM25")

    def _build_or_load_index(self):
        if os.path.isfile(self.index_path) and os.path.isfile(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            if len(loaded_meta) == self.index.ntotal:
                self.idx2meta = loaded_meta
                return

        passages = [_e5_passage(doc) for doc in self.all_docs]
        embeddings = self._e5.encode(
            passages, batch_size=16, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype('float32'))

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.idx2meta, f, ensure_ascii=False, indent=2)
    
    def _route(self, query: str) -> dict:
        q = preprocess(query)
        if len(q) < 2:
            return {"mode": "global", "top_chapters": [], "confidence": 0.0, "margin": 0.0}

        # ── THÊM: Keyword override trước PhoBERT ──────────────────
        q_lower = query.lower()
        for ch, keywords in CHAPTER_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                logger.info(f"Keyword override → Chương {ch}")
                return {
                    "mode": "scoped",
                    "top_chapters": [ch],
                    "confidence": 0.99,
                    "margin": 0.99
                }
        # ──────────────────────────────────────────────────────────

        # Phần PhoBERT 
        result = self._predictor.predict_safe(query)
        if result.get("error") or result["chapter"] is None:
            return {"mode": "global", "top_chapters": [], "confidence": 0.0, "margin": 0.0}
        top3    = result["top3"]
        top1_ch = top3[0][0]
        top2_ch = top3[1][0] if len(top3) > 1 else (top1_ch % 10) + 1
        conf    = result["confidence"]
        margin  = result["margin"]
        mode = "scoped" if (conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD) else "global"
        return {"mode": mode, "top_chapters": [top1_ch, top2_ch], "confidence": round(conf, 4), "margin": round(margin, 4)}
    
    def _hybrid_scores(self, query: str, doc_indices: list[int] | None = None) -> np.ndarray:
        q_text = _e5_query(query)
        q_vec = self._e5.encode([q_text], normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=1)[0].astype('float32')
        dense_scores, _ = self.index.search(np.expand_dims(q_vec, axis=0), self.index.ntotal)
        dense_scores = dense_scores[0]

        if _HAS_SEG:
            tokenized_query = preprocess(query).split()
        else:
            tokenized_query = query.lower().split()
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))

        def minmax(arr):
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())

        dense_norm  = minmax(dense_scores)
        sparse_norm = minmax(sparse_scores)
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        if doc_indices is not None:
            mask = np.full(len(self.all_docs), -1.0)
            mask[doc_indices] = hybrid[doc_indices]
            return mask
        return hybrid

    def retrieve(self, question: str, chapter: int | None = None, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        doc_indices = None
        if chapter is not None:
            doc_indices = self.chapters.get(chapter, [])
            if not doc_indices:
                return []
        scores = self._hybrid_scores(question, doc_indices=doc_indices)
        top_idx = np.argsort(scores)[::-1][:k]
        top_idx = [i for i in top_idx if scores[i] >= 0][:k]
        return [
            {
                "rank": rank,
                "score": round(float(scores[i]), 4),
                "chapter": self.all_chapters[i],
                "chapter_name": CHAPTER_NAMES.get(self.all_chapters[i], f"Ch.{self.all_chapters[i]}"),
                "content": self.all_docs[i]
            }
            for rank, i in enumerate(top_idx, 1)
        ]

    def answer(self, question: str, top_k: int | None = None, verbose: bool = True) -> dict:
        q = question.strip() if isinstance(question, str) else ""
        k = top_k or self.top_k
        route = self._route(q)
        mode  = route["mode"]

        if mode == "scoped":
            combined_idx = []
            for ch in route["top_chapters"]:
                combined_idx.extend(self.chapters.get(ch, []))
            if not combined_idx:
                mode = "global"
            else:
                scores = self._hybrid_scores(q, doc_indices=combined_idx)
                valid_scores = [scores[i] for i in combined_idx if scores[i] >= 0]
                top_score = max(valid_scores) if valid_scores else 0.0
                if top_score < self.score_gate:
                    mode = "global_fallback"
                else:
                    top_idx = np.argsort(scores)[::-1]
                    top_idx = [i for i in top_idx if scores[i] >= self.score_gate][:k]
                    passages = [
                        {"rank": rank, "score": round(float(scores[i]), 4),
                         "chapter": self.all_chapters[i],
                         "chapter_name": CHAPTER_NAMES.get(self.all_chapters[i]),
                         "content": self.all_docs[i]}
                        for rank, i in enumerate(top_idx, 1)
                    ]

        if mode in ("global", "global_fallback"):
            scores = self._hybrid_scores(q)
            
            if route["confidence"] >= 0.55:
                for ch in route["top_chapters"]:
                    for i in self.chapters.get(ch, []):
                        scores[i] = min(scores[i] * self.chapter_boost, MAX_SCORE)

            top_idx = np.argsort(scores)[::-1]            
            top_idx = [i for i in top_idx if scores[i] >= self.score_gate][:k]

            passages = [
                {"rank": rank, "score": round(float(scores[i]), 4),
                 "chapter": self.all_chapters[i],
                 "chapter_name": CHAPTER_NAMES.get(self.all_chapters[i]),
                 "content": self.all_docs[i]}
                for rank, i in enumerate(top_idx, 1)
            ]

        return {
            "question": q,
            "route_mode": mode,
            "predicted_chapters": route["top_chapters"],
            "confidence": route["confidence"],
            "passages": passages
        }

    def get_context_string(self, question: str, top_k: int = 3) -> str:
        result = self.answer(question, top_k=top_k, verbose=False)
        if not result["passages"]:
            return "(Không tìm thấy nội dung liên quan.)"
        parts = [
            f"[Đoạn {p['rank']} – Ch.{p['chapter']}: {p['chapter_name']}]\n{p['content']}"
            for p in result["passages"]
        ]
        return "\n\n".join(parts)