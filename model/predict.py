"""
model/predict.py  ·  PhoBERT Classifier — Chạy local trên CMD
==============================================================
DÙNG ĐỂ:
  - Predict chapter từ câu hỏi người dùng
  - Tích hợp vào Streamlit app
  - Chạy hoàn toàn trên CPU, không cần GPU

YÊU CẦU:
  - Đã train xong trên Colab
  - Download model/phobert_classifier/ về máy
  - pip install transformers torch streamlit underthesea

CÀI ĐẶT:
  pip install transformers torch streamlit underthesea

CHẠY STREAMLIT:
  streamlit run predict.py

DÙNG NHƯ MODULE trong code khác:
  from model.predict import PhoBERTPredictor
  predictor = PhoBERTPredictor()
  result = predictor.predict("Điều kiện để được hưởng trợ cấp thất nghiệp?")
  print(result)  # {'chapter': 3, 'confidence': 0.87, 'status': 'scoped', ...}

NHỮNG GÌ ĐÃ BỔ SUNG (từ predict.py gốc SVM/LR):
  [ADD-1] _validate()     — kiểm tra input trước khi tokenize (TypeError, ValueError)
  [ADD-2] predict_safe()  — không bao giờ raise, dành cho UI/API
  [FIX-1] predict()       — gọi _validate() thay vì preprocess() trực tiếp
  [FIX-2] predict_batch() — dùng predict_safe() để 1 câu lỗi không crash cả batch
"""

import os, re, json, unicodedata, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────
_BASE            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR        = os.path.join(_BASE, "model")
PHOBERT_SAVE_DIR = os.path.join(MODEL_DIR, "phobert_classifier")
CONFIG_DIR       = os.path.join(_BASE, "config")
META_PATH        = os.path.join(MODEL_DIR, "train_meta.json")

# ── Word segmentation ─────────────────────────────────────────
try:
    from underthesea import word_tokenize as _wt
    def _segment(text: str) -> str:
        return _wt(text, format="text")
    _HAS_SEG = True
except ImportError:
    def _segment(text: str) -> str:
        return text
    _HAS_SEG = False


def preprocess(text: str, use_seg: bool = True) -> str:
    """Tiền xử lý — giống hệt train_model.py để không bị lệch."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    text = re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]",
        " ", text
    )
    if use_seg and _HAS_SEG:
        text = _segment(text)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════
class PhoBERTPredictor:
    """
    Load PhoBERT đã train và predict chapter từ câu hỏi.

    Dùng trong pipeline:
        predictor = PhoBERTPredictor()
        result = predictor.predict("câu hỏi")
        # result = {
        #   'chapter'   : 3,
        #   'confidence': 0.87,
        #   'margin'    : 0.23,
        #   'status'    : 'scoped',   # hoặc 'broad'
        #   'top3'      : [(3, 0.87), (5, 0.64), (1, 0.12)]
        # }

        # Không bao giờ raise:
        safe = predictor.predict_safe("câu hỏi")
        # safe = {..., 'error': None}  hoặc  {'chapter': None, ..., 'error': 'lý do'}
    """

    def __init__(self,
                 model_dir: str = PHOBERT_SAVE_DIR,
                 meta_path: str = META_PATH,
                 device: str = "cpu"):
        """
        Args:
            model_dir : thư mục chứa model đã train (phobert_classifier/)
            meta_path : file train_meta.json (chứa RAG thresholds)
            device    : "cpu" cho máy không có GPU (mặc định)
        """
        self.device = torch.device(device)
        self._load_meta(meta_path)
        self._load_model(model_dir)

    # ── Setup ──────────────────────────────────────────────────

    def _load_meta(self, meta_path: str):
        """Load RAG thresholds từ train_meta.json."""
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rag = meta.get("rag_thresholds", {})
            self.conf_threshold   = rag.get("conf_threshold",   0.45)
            self.margin_threshold = rag.get("margin_threshold", 0.15)
            self.use_seg          = meta.get("use_segment", True)
            self.max_len          = meta.get("max_len", 128)
            self.num_labels       = meta.get("num_labels", 10)
        else:
            # Fallback nếu không có file meta
            self.conf_threshold   = 0.45
            self.margin_threshold = 0.15
            self.use_seg          = True
            self.max_len          = 128
            self.num_labels       = 10

    def _load_model(self, model_dir: str):
        """Load tokenizer + model từ thư mục local."""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Không tìm thấy model tại: {model_dir}\n"
                f"Hãy train trên Colab và download phobert_classifier/ về thư mục model/"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    # ── Validation ────────────────────────────────────────────

    def _validate(self, question: str) -> str:
        """
        [ADD-1] Kiểm tra input và trả về text đã preprocess.
        Phải gọi trước khi tokenize — tokenizer không phát hiện được input rác.

        Raises:
            TypeError  : nếu question không phải str
            ValueError : nếu câu quá ngắn sau preprocessing
        """
        if not isinstance(question, str):
            raise TypeError(
                f"question phải là str, nhận {type(question).__name__}"
            )
        cleaned = preprocess(question, self.use_seg)
        if len(cleaned) < 2:
            raise ValueError(
                f"Câu hỏi quá ngắn sau preprocessing: '{cleaned}'"
            )
        return cleaned

    # ── Predict ───────────────────────────────────────────────

    def predict(self, question: str) -> dict:
        """
        Predict chapter từ câu hỏi.

        [FIX-1] Gọi _validate() thay vì preprocess() trực tiếp —
        phát hiện input xấu trước khi vào model.

        Args:
            question : câu hỏi tiếng Việt

        Returns:
            {
              'chapter'   : int   — chapter dự đoán (1-10)
              'confidence': float — xác suất cao nhất
              'margin'    : float — khoảng cách top1 và top2
              'status'    : str   — 'scoped' hoặc 'broad'
              'top3'      : list  — [(chapter, prob), ...] top 3
            }

        Raises:
            TypeError  : nếu question không phải str
            ValueError : nếu câu hỏi quá ngắn
        """
        # [FIX-1] validate + preprocess một lần, raise sớm nếu lỗi
        text = self._validate(question)

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference (không cần gradient → tiết kiệm bộ nhớ)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            proba  = F.softmax(logits, dim=-1)[0].cpu().numpy()

        # Lấy kết quả
        sorted_idx = proba.argsort()[::-1]
        top1_idx   = sorted_idx[0]
        top2_idx   = sorted_idx[1]
        confidence = float(proba[top1_idx])
        margin     = float(proba[top1_idx] - proba[top2_idx])

        # Quyết định scoped hay broad (dùng RAG threshold từ train)
        is_scoped = (confidence >= self.conf_threshold) and (margin >= self.margin_threshold)
        status    = "scoped" if is_scoped else "broad"

        top3 = [(int(sorted_idx[i]) + 1, float(proba[sorted_idx[i]])) for i in range(3)]

        return {
            "chapter":    int(top1_idx) + 1,   # 1-indexed
            "confidence": round(confidence, 4),
            "margin":     round(margin, 4),
            "status":     status,
            "top3":       top3,
        }

    def predict_safe(self, question: str) -> dict:
        """
        [ADD-2] Không bao giờ raise — dành cho UI/API và batch processing.

        Dùng khi không muốn try/except ở phía caller (Streamlit, FastAPI, v.v.)

        Returns:
            Nếu OK  : {**result, 'error': None}
            Nếu lỗi : {'chapter': None, 'confidence': 0.0, 'margin': 0.0,
                        'status': 'error', 'top3': [], 'error': str(e)}
        """
        try:
            result = self.predict(question)
            return {**result, "error": None}
        except (ValueError, TypeError) as e:
            return {
                "chapter":    None,
                "confidence": 0.0,
                "margin":     0.0,
                "status":     "error",
                "top3":       [],
                "error":      str(e),
            }

    def predict_batch(self, questions: list) -> list:
        """
        Predict nhiều câu cùng lúc.

        [FIX-2] Dùng predict_safe() thay vì predict() —
        1 câu lỗi sẽ trả {'error': ...} thay vì crash cả batch.
        """
        return [self.predict_safe(q) for q in questions]


# ══════════════════════════════════════════════════════════════
# STREAMLIT APP — chạy: streamlit run predict.py
# ══════════════════════════════════════════════════════════════
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Phân loại Chapter - PhoBERT",
        page_icon="📚",
        layout="centered",
    )

    st.title("📚 Phân loại Câu hỏi theo Chapter")
    st.caption("Sử dụng PhoBERT fine-tuned để phân loại câu hỏi pháp luật")

    # ── Load model (cache để không load lại mỗi lần) ──────────
    @st.cache_resource(show_spinner="Đang load PhoBERT model...")
    def load_predictor():
        return PhoBERTPredictor(device="cpu")

    # Kiểm tra model đã tồn tại chưa
    if not os.path.exists(PHOBERT_SAVE_DIR):
        st.error(
            f"⚠️ Chưa tìm thấy model tại `model/phobert_classifier/`\n\n"
            f"Hãy train trên Google Colab trước, sau đó download model về."
        )
        st.code(
            "# Trên Colab sau khi train xong:\n"
            "import shutil\n"
            "shutil.make_archive('phobert_classifier', 'zip', 'model/phobert_classifier')\n"
            "# Sau đó download phobert_classifier.zip về máy\n"
            "# Giải nén vào thư mục model/phobert_classifier/",
            language="python"
        )
        return

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return

    # ── Sidebar: thông tin model ───────────────────────────────
    with st.sidebar:
        st.header("ℹ️ Thông tin Model")
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.metric("Model", "PhoBERT")
            st.metric("Val Macro F1", f"{meta.get('val_macro_f1', 'N/A'):.4f}")
            st.metric("Test Macro F1", f"{meta.get('test_macro_f1', 'N/A'):.4f}")
            st.metric("Conf Threshold", predictor.conf_threshold)
            st.metric("Margin Threshold", predictor.margin_threshold)
        st.divider()
        st.caption("Device: CPU (local)")
        seg_status = "✅ ON" if _HAS_SEG else "❌ OFF (cài underthesea)"
        st.caption(f"Word segment: {seg_status}")

    # ── Input câu hỏi ─────────────────────────────────────────
    st.subheader("Nhập câu hỏi")
    question = st.text_area(
        label="Câu hỏi pháp luật:",
        placeholder="Ví dụ: Điều kiện để được hưởng trợ cấp thất nghiệp là gì?",
        height=120,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🔍 Phân loại", type="primary", use_container_width=True)
    with col2:
        show_debug = st.checkbox("Hiện chi tiết xác suất")

    # ── Kết quả ───────────────────────────────────────────────
    if predict_btn:
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi!")
        else:
            with st.spinner("Đang phân tích..."):
                # Dùng predict_safe → Streamlit không bao giờ crash
                result = predictor.predict_safe(question)

            if result["error"]:
                st.error(f"❌ Lỗi: {result['error']}")
            else:
                st.divider()

                # Kết quả chính
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("📖 Chapter", f"Chương {result['chapter']}")
                with col_b:
                    st.metric("🎯 Độ tin cậy", f"{result['confidence']:.1%}")
                with col_c:
                    status_icon = "✅ Scoped" if result["status"] == "scoped" else "🔄 Broad"
                    st.metric("📡 Trạng thái RAG", status_icon)

                # Giải thích status
                if result["status"] == "scoped":
                    st.success(
                        f"✅ **SCOPED** — Model tự tin cao. "
                        f"Pipeline sẽ search trong **Chương {result['chapter']}**."
                    )
                else:
                    top2_chapters = [f"Chương {ch}" for ch, _ in result["top3"][:2]]
                    st.info(
                        f"🔄 **BROAD** — Model chưa đủ tự tin. "
                        f"Pipeline sẽ search rộng hơn (có thể gồm {', '.join(top2_chapters)})."
                    )

                # Top-3 dự đoán
                st.subheader("Top 3 dự đoán")
                for i, (ch, prob) in enumerate(result["top3"]):
                    label = f"Chương {ch}"
                    icon  = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.progress(prob, text=f"{icon} {label}: {prob:.1%}")

                # Debug: xác suất raw
                if show_debug:
                    st.divider()
                    st.subheader("🔧 Chi tiết")
                    st.json({
                        "chapter":    result["chapter"],
                        "confidence": result["confidence"],
                        "margin":     result["margin"],
                        "status":     result["status"],
                        "conf_threshold":    predictor.conf_threshold,
                        "margin_threshold":  predictor.margin_threshold,
                        "preprocessed_text": preprocess(question, predictor.use_seg)[:100] + "...",
                    })

    # ── Batch test ────────────────────────────────────────────
    st.divider()
    with st.expander("🧪 Test nhiều câu cùng lúc"):
        batch_input = st.text_area(
            "Mỗi câu một dòng:",
            height=150,
            placeholder="Câu hỏi 1\nCâu hỏi 2\nCâu hỏi 3",
        )
        if st.button("Phân loại tất cả"):
            questions = [q.strip() for q in batch_input.strip().split("\n") if q.strip()]
            if questions:
                # predict_batch dùng predict_safe → không crash dù có câu lỗi
                results = predictor.predict_batch(questions)
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Câu hỏi":    q[:60] + "..." if len(q) > 60 else q,
                        "Chapter":    f"Chương {r['chapter']}" if r["chapter"] else "❌ Lỗi",
                        "Độ tin cậy": f"{r['confidence']:.1%}" if r["chapter"] else "—",
                        "Trạng thái": r["status"],
                        "Lỗi":        r["error"] or "",
                    }
                    for q, r in zip(questions, results)
                ])
                st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Kiểm tra có phải đang chạy qua streamlit không
    try:
        import streamlit as st
        run_streamlit_app()
    except Exception:
        # Chạy thử từ CMD thuần (không phải streamlit)
        print("=== PhoBERT Predictor — Test CLI ===")
        if not os.path.exists(PHOBERT_SAVE_DIR):
            print(f"⚠ Chưa có model tại: {PHOBERT_SAVE_DIR}")
            print("  Hãy train trên Colab trước!")
        else:
            predictor = PhoBERTPredictor(device="cpu")

            print("\n[A] predict() — câu bình thường:")
            test_questions = [
                "Điều kiện để được hưởng trợ cấp thất nghiệp là gì?",
                "Thủ tục đăng ký kết hôn như thế nào?",
                "Quyền và nghĩa vụ của người lao động?",
            ]
            for q in test_questions:
                result = predictor.predict(q)
                print(f"  Câu hỏi : {q}")
                print(f"  Chapter  : {result['chapter']} | Conf: {result['confidence']:.1%} | {result['status'].upper()}\n")

            print("[B] predict_safe() — edge cases (không bao giờ raise):")
            for q, label in [("", "empty"), ("  ", "whitespace"), (None, "None"), ("a", "1 char")]:
                r = predictor.predict_safe(q)
                print(f"  [{label}] → ch={r['chapter']} | err={r['error']}")

            print("\n[C] predict_batch() — hỗn hợp câu tốt và câu lỗi:")
            batch = ["Quyền bầu cử là gì?", "", "Luật hình sự áp dụng khi nào?"]
            results = predictor.predict_batch(batch)
            for q, r in zip(batch, results):
                q_display = repr(q) if not q.strip() else q
                print(f"  '{q_display}' → ch={r['chapter']} | err={r['error']}")
