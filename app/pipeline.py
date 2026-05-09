"""
app/pipeline.py  ·  Ngày 6 – Production-Grade Pipeline
=======================================================
NHỮNG GÌ ĐÃ SỬA:
  [FIX-1] Singleton cache PhoBERT classifier.
  [FIX-2] answer_dict() có schema nhất quán: luôn có đủ keys.
  [FIX-3] Route mode được expose ra ngoài để UI hiển thị.
  [FIX-4] Mọi exception đều được catch → không bao giờ crash UI.
  [FIX-5] get_context() luôn trả str (kể cả khi lỗi).
"""

import os, sys, threading
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from rag.rag_pipeline import RAGPipeline, CHAPTER_NAMES

_cache: dict[str, RAGPipeline] = {}
_lock  = threading.Lock()

def _get_rag() -> RAGPipeline:
    with _lock:
        if "phobert" not in _cache:
            _cache["phobert"] = RAGPipeline(
                rag_data_path=os.path.join(_BASE, "data", "rag_data.json"),
                phobert_model_path=os.path.join(_BASE, "model", "phobert_classifier"),
                top_k=3,
            )
    return _cache["phobert"]

_SEP = "─" * 62

def _format(result: dict) -> str:
    q        = result.get("question", "")
    chs      = result.get("predicted_chapters", [])
    conf     = result.get("confidence", 0.0)
    mode     = result.get("route_mode", "global")
    passages = result.get("passages", [])
    mode_label = {"scoped":"🎯 Scoped","global":"🌐 Global","global_fallback":"🌐 Global Fallback"}.get(mode, mode)
    lines = [
        f"❓ Câu hỏi : {q}",
        f"📖 Route   : {mode_label} | Chapters: {chs} | Conf: {conf:.2f}",
        "", _SEP, "📄 NỘI DUNG LIÊN QUAN:", _SEP,
    ]
    if not passages:
        lines.append("\n⚠️  Không tìm thấy nội dung phù hợp.")
    else:
        for p in passages:
            lines.append(f"\n[Đoạn {p['rank']} | Ch.{p['chapter']}: {p['chapter_name']} | {p['score']:.3f}]")
            lines.append(p["content"])
    lines += ["", _SEP]
    return "\n".join(lines)


def answer(question: str, top_k: int = 3) -> str:
    """Trả về string có định dạng. Không bao giờ raise."""
    q = question.strip() if isinstance(question, str) else ""
    if not q:
        return "⚠️  Vui lòng nhập câu hỏi."
    try:
        return _format(_get_rag().answer(q, top_k=top_k, verbose=False))
    except Exception as e:
        return f"⚠️  Lỗi hệ thống: {e}"


def answer_dict(question: str, top_k: int = 3) -> dict:
    """Trả về dict nhất quán — dùng cho Streamlit/API."""
    q = question.strip() if isinstance(question, str) else ""
    _empty = {"question": question, "predicted_chapters": [], "confidence": 0.0,
               "route_mode": "none", "passages": [], "error": None}
    if not q:
        return {**_empty, "error": "Câu hỏi trống"}
    try:
        r = _get_rag().answer(q, top_k=top_k, verbose=False)
        return {**r, "error": None}
    except Exception as e:
        return {**_empty, "question": q, "error": str(e)}


def get_context(question: str, top_k: int = 3) -> str:
    q = question.strip() if isinstance(question, str) else ""
    if not q:
        return "(Câu hỏi trống.)"
    try:
        return _get_rag().get_context_string(q, top_k=top_k)
    except Exception as e:
        return f"(Lỗi: {e})"


if __name__ == "__main__":
    print("=" * 62)
    for q in ["Pháp luật là gì", "Luật hình sự là gì", ""]:
        d = answer_dict(q)
        print(f"  Q: '{q}' | mode={d['route_mode']} | conf={d['confidence']:.2f} | err={d['error']}")
    print("=" * 62)
