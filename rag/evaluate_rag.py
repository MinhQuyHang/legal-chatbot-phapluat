"""
rag/evaluate_rag.py  ·  Ngày 8 – RAG Evaluation Professional
=============================================================
Metrics:
  - Keyword Hit@1     (proxy cho relevance)
  - Chapter Hit@K     (K=1,3,5) — biết expected chapter từ rag_data
  - MRR               (Mean Reciprocal Rank)
  - Route distribution (scoped vs global vs fallback)
  - Score stats per chapter

Chạy:
    python rag/evaluate_rag.py --top-k 3
"""
import os, sys, json, argparse, itertools
import numpy as np
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.rag_pipeline import RAGPipeline

# Chapter map đã xác nhận từ rag_data.json
EVAL_QUESTIONS = [
    # (question, expected_chapter, keywords)
    ("Nhà nước xuất hiện từ khi nào",           1, ["nhà nước","nguồn gốc","lịch sử"]),
    ("Thuyết thần học về nguồn gốc nhà nước",   1, ["thuyết","thần học","nhà nước"]),
    ("Nhà nước mang tính giai cấp là gì",        1, ["nhà nước","giai cấp"]),
    ("Chức năng đối nội nhà nước Việt Nam",      2, ["chức năng","đối nội","nhà nước"]),
    ("Bản chất nhà nước Cộng hòa XHCN VN",      2, ["nhà nước","việt nam","đảng","bản chất"]),
    ("Pháp luật là gì",                           3, ["pháp luật","quy tắc","nhà nước"]),
    ("Pháp luật ra đời theo Mác Lênin",          3, ["pháp luật","mác","giai cấp"]),
    ("Bản chất giai cấp của pháp luật",          3, ["bản chất","pháp luật","giai cấp"]),
    ("Hình thức pháp luật là gì",                4, ["hình thức","pháp luật","tập quán"]),
    ("Tập quán pháp là gì",                      4, ["tập quán","pháp"]),
    ("Văn bản quy phạm pháp luật là gì",         4, ["văn bản","quy phạm","pháp luật"]),
    ("Hệ thống pháp luật Việt Nam",              5, ["hệ thống","pháp luật","ngành luật"]),
    ("Ngành luật và chế định pháp luật",         5, ["ngành luật","chế định"]),
    ("Luật hành chính điều chỉnh quan hệ nào",   6, ["hành chính","điều chỉnh"]),
    ("Phương pháp điều chỉnh luật hành chính",   6, ["hành chính","phương pháp","mệnh lệnh"]),
    ("Vi phạm hành chính là gì",                 6, ["hành chính","vi phạm"]),
    ("Tội phạm được phân loại như thế nào",      7, ["tội phạm","phân loại","hình sự"]),
    ("Trách nhiệm hình sự là gì",                7, ["hình sự","trách nhiệm","hình phạt"]),
    ("Cấu thành tội phạm gồm gì",               7, ["cấu thành","tội phạm"]),
    ("Luật dân sự điều chỉnh quan hệ tài sản",  8, ["dân sự","tài sản","điều chỉnh"]),
    ("Nguyên tắc bình đẳng luật dân sự",         8, ["dân sự","bình đẳng"]),
    ("Hợp đồng dân sự có hiệu lực khi nào",     8, ["hợp đồng","dân sự","hiệu lực"]),
    ("Hợp đồng lao động là gì",                  9, ["hợp đồng","lao động"]),
    ("Luật lao động bảo vệ người lao động",      9, ["lao động","bảo vệ","người lao động"]),
    ("Công đoàn có vai trò gì trong lao động",   9, ["công đoàn","lao động"]),
    ("Điều kiện kết hôn theo pháp luật VN",      10, ["kết hôn","điều kiện","pháp luật"]),
    ("Quyền và nghĩa vụ vợ chồng",              10, ["vợ chồng","quyền","nghĩa vụ"]),
    ("Căn cứ ly hôn theo luật VN",              10, ["ly hôn","căn cứ","tòa án"]),
]


def evaluate(rag: RAGPipeline, top_k_list=[1,3,5]) -> dict:
    n = len(EVAL_QUESTIONS)
    kw_hits     = 0
    ch_hits     = {k:0 for k in top_k_list}
    mrr_list    = []
    scores      = []
    route_dist  = {}

    print(f"\n{'='*65}")
    print(f"  RAG Evaluation — {n} câu hỏi | Top-K = {top_k_list}")
    print(f"{'='*65}\n")
    print(f"  {'#':>2}  {'KW':3}  {'@1':3} {'@3':3} {'@5':3}  {'Score':6}  {'Route':10}  Câu hỏi")
    print(f"  {'─'*60}")

    for i, (q, exp_ch, kws) in enumerate(EVAL_QUESTIONS, 1):
        result   = rag.answer(q, top_k=max(top_k_list), verbose=False)
        passages = result["passages"]
        mode     = result.get("route_mode","?")
        route_dist[mode] = route_dist.get(mode, 0) + 1

        top_content = passages[0]["content"].lower() if passages else ""
        top_score   = passages[0]["score"] if passages else 0.0
        scores.append(top_score)

        # Keyword hit
        kw_ok = any(kw.lower() in top_content for kw in kws)
        if kw_ok: kw_hits += 1

        # Chapter hit@K
        ch_at = {}
        for k in top_k_list:
            hit = any(p["chapter"]==exp_ch for p in passages[:k])
            ch_at[k] = hit
            if hit: ch_hits[k] += 1

        # MRR
        rr = next((1/(r+1) for r,p in enumerate(passages) if p["chapter"]==exp_ch), 0.0)
        mrr_list.append(rr)

        ks = " ".join("✓" if ch_at.get(k) else "✗" for k in top_k_list)
        q_s = (q[:30]+"…") if len(q)>30 else q
        print(f"  {i:2d}  {'✓' if kw_ok else '✗'}  {ks}  {top_score:.4f}  {mode:<10}  {q_s}")

    # Summary
    kw_acc  = kw_hits / n
    ch_accs = {k: ch_hits[k]/n for k in top_k_list}
    mrr     = float(np.mean(mrr_list))
    avg_s   = float(np.mean(scores))

    print(f"\n{'='*65}\n  KẾT QUẢ ĐÁNH GIÁ\n{'='*65}")
    print(f"\n   Keyword Hit@1        : {kw_hits}/{n} = {kw_acc:.2%}")
    for k in top_k_list:
        print(f"   Chapter Hit@{k:<2}       : {ch_hits[k]}/{n} = {ch_accs[k]:.2%}")
    print(f"   MRR                  : {mrr:.4f}")
    print(f"   Score avg (top-1)    : {avg_s:.4f}")
    print(f"   Route distribution   : {route_dist}")

    # Per-chapter analysis
    print(f"\n   Chapter Hit@3 phân tích:")
    ch_result = {}
    for q, exp_ch, kws in EVAL_QUESTIONS:
        ch_result.setdefault(exp_ch, {"total":0,"hit":0})
        ch_result[exp_ch]["total"] += 1
    for q, exp_ch, kws in EVAL_QUESTIONS:
        result   = rag.answer(q, top_k=3, verbose=False)
        passages = result["passages"]
        if any(p["chapter"]==exp_ch for p in passages[:3]):
            ch_result[exp_ch]["hit"] += 1
    for ch in sorted(ch_result):
        d   = ch_result[ch]
        acc = d["hit"]/d["total"]
        bar = "█"*d["hit"] + "░"*(d["total"]-d["hit"])
        from rag.rag_pipeline import CHAPTER_NAMES
        print(f"    Ch.{ch:2d} {bar} {d['hit']}/{d['total']} ({acc:.0%}) — {CHAPTER_NAMES.get(ch,'')}")

    return {
        "keyword_hit_at_1": round(kw_acc,4),
        "chapter_hit_at_k": {str(k): round(ch_accs[k],4) for k in top_k_list},
        "mrr": round(mrr,4),
        "score_avg": round(avg_s,4),
        "route_distribution": route_dist,
        "n_questions": n,
    }


def _quick_eval(rag: RAGPipeline, top_k: int = 3) -> tuple[float, float]:
    """
    Chạy nhanh Hit@3 và MRR trên toàn bộ EVAL_QUESTIONS.
    Trả về (hit_at_3, mrr) — không in ra màn hình.
    """
    hits, mrr_list = 0, []
    for q, exp_ch, _ in EVAL_QUESTIONS:
        result   = rag.answer(q, top_k=top_k, verbose=False)
        passages = result["passages"]
        hit      = any(p["chapter"] == exp_ch for p in passages[:top_k])
        rr       = next((1/(r+1) for r, p in enumerate(passages) if p["chapter"] == exp_ch), 0.0)
        if hit: hits += 1
        mrr_list.append(rr)
    hit_rate = hits / len(EVAL_QUESTIONS)
    mrr      = float(np.mean(mrr_list))
    return hit_rate, mrr


def grid_search_rag(
    rag_data_path: str,
    phobert_model_path: str,
) -> dict:
    """
    Grid search score_gate và chapter_boost cho E5 Dense Retrieval.

    Metric tổng hợp: 0.6 × Hit@3 + 0.4 × MRR
        → Hit@3 quan trọng hơn vì cần tìm đúng chapter,
          không nhất thiết phải rank 1.

    16 tổ hợp = 4 × 4, mỗi tổ hợp chạy _quick_eval()
    trên 28 câu hỏi → tổng ~448 lần gọi retrieve, nhanh.
    """
    import rag.rag_pipeline as rp

    gate_grid   = [0.25, 0.30, 0.35, 0.40]
    boost_grid  = [1.10, 1.15, 1.20, 1.25]

    total = len(gate_grid) * len(boost_grid)
    best  = {"score": -1.0}
    results = []

    print(f"\n{'='*65}")
    print(f"  GRID SEARCH RAG — {total} tổ hợp")
    print(f"  Metric: 0.6×Hit@3 + 0.4×MRR")
    print(f"{'='*65}")
    print(f"\n  {'gate':>5}  {'boost':>6}  {'Hit@3':>6}  {'MRR':>6}  {'Score':>7}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")

    for gate, boost in itertools.product(gate_grid, boost_grid):
        # Patch constants trong module rag_pipeline tạm thời
        rp.SCORE_GATE      = gate
        rp.CHAPTER_BOOST   = boost

        # Tạo pipeline mới
        pipeline = RAGPipeline(
            rag_data_path      = rag_data_path,
            phobert_model_path = phobert_model_path,
        )

        hit3, mrr = _quick_eval(pipeline, top_k=3)
        score     = round(0.6 * hit3 + 0.4 * mrr, 4)

        results.append((score, gate, boost, hit3, mrr))
        tag = " ← best" if score > best["score"] else ""
        print(f"  {gate:>5.2f}  {boost:>6.2f}  {hit3:>6.1%}  {mrr:>6.4f}  {score:>7.4f}{tag}")

        if score > best["score"]:
            best = {
                "score":         score,
                "score_gate":    gate,
                "chapter_boost": boost,
                "hit_at_3":      round(hit3, 4),
                "mrr":           round(mrr, 4),
            }

    print(f"\n   Best config:")
    print(f"     score_gate    = {best['score_gate']}")
    print(f"     chapter_boost = {best['chapter_boost']}")
    print(f"     Hit@3 = {best['hit_at_3']:.1%}  |  MRR = {best['mrr']:.4f}  |  Score = {best['score']:.4f}")

    # Reset về giá trị tốt nhất
    rp.SCORE_GATE    = best["score_gate"]
    rp.CHAPTER_BOOST = best["chapter_boost"]

    return best


def update_config_yaml(base_dir: str, rag_best: dict) -> None:
    """
    Cập nhật phần rag: trong config/settings.yaml với thông số tốt nhất.
    Nếu file chưa tồn tại thì tạo mới. Nếu không có pyyaml thì ghi thủ công.
    """
    config_dir  = os.path.join(base_dir, "config")
    config_path = os.path.join(config_dir, "settings.yaml")
    os.makedirs(config_dir, exist_ok=True)

    # Đọc config cũ nếu có
    existing = {}
    if os.path.exists(config_path):
        if _HAS_YAML:
            with open(config_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_raw = f.read()

    # Cập nhật phần rag
    rag_section = {
        "confidence_threshold": existing.get("rag", {}).get("confidence_threshold", 0.45),
        "margin_threshold":     existing.get("rag", {}).get("margin_threshold", 0.15),
        "score_gate":           rag_best["score_gate"],
        "chapter_boost":        rag_best["chapter_boost"],
        "retriever":            "multilingual-e5",
        "e5_model":             "intfloat/multilingual-e5-large",
        "search_meta": {
            "hit_at_3":  rag_best["hit_at_3"],
            "mrr":       rag_best["mrr"],
            "score":     rag_best["score"],
        },
    }

    if _HAS_YAML:
        existing["rag"] = rag_section
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
    else:
        rag_lines = [
            "rag:",
            f"  retriever: multilingual-e5",
            f"  e5_model: intfloat/multilingual-e5-large",
            f"  confidence_threshold: {rag_section['confidence_threshold']}",
            f"  margin_threshold: {rag_section['margin_threshold']}",
            f"  score_gate: {rag_section['score_gate']}        # từ grid search",
            f"  chapter_boost: {rag_section['chapter_boost']}  # từ grid search",
            "  # search_meta:",
            f"  #   hit_at_3: {rag_best['hit_at_3']}",
            f"  #   mrr: {rag_best['mrr']}",
            f"  #   score: {rag_best['score']}",
        ]
        if _HAS_YAML is False and os.path.exists(config_path):
            lines = existing_raw.splitlines()
            model_lines = [l for l in lines if not l.startswith("rag:") and
                           not (l.startswith("  ") and any(
                               k in l for k in ["confidence","margin","score_gate",
                                                "chapter_boost","bm25","search_meta",
                                                "hit_at","mrr","retriever","e5_model"]))]
            final = "\n".join(model_lines).rstrip() + "\n" + "\n".join(rag_lines) + "\n"
        else:
            final = "\n".join(rag_lines) + "\n"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(final)

    print(f"\n  ✓ config/settings.yaml đã cập nhật  (pyyaml={'có' if _HAS_YAML else 'không — fallback'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phobert-model-path", type=str,
                        default=None,
                        help="Đường dẫn thư mục phobert_classifier/ (mặc định: model/phobert_classifier)")
    parser.add_argument("--top-k",     type=int, default=3)
    parser.add_argument("--no-search", action="store_true",
                        help="Bỏ qua grid search, chỉ chạy evaluate")
    args = parser.parse_args()

    BASE             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_data         = os.path.join(BASE, "data",  "rag_data.json")
    phobert_path     = args.phobert_model_path or os.path.join(BASE, "model", "phobert_classifier")

    # ── 1. Evaluate với thông số hiện tại ────────────────────
    rag = RAGPipeline(
        rag_data_path      = rag_data,
        phobert_model_path = phobert_path,
        top_k              = args.top_k,
    )
    metrics = evaluate(rag, top_k_list=[1, 3, 5])

    out = os.path.join(BASE, "rag", "eval_results.json")
    json.dump(metrics, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n   Lưu: rag/eval_results.json")

    # ── 2. Grid search thông số tốt nhất ─────────────────────
    if not args.no_search:
        print(f"\n{'─'*65}")
        print(f"  Bắt đầu grid search — dùng --no-search để bỏ qua")
        print(f"{'─'*65}")
        rag_best = grid_search_rag(
            rag_data_path      = rag_data,
            phobert_model_path = phobert_path,
        )

        search_out = os.path.join(BASE, "rag", "grid_search_results.json")
        json.dump(rag_best, open(search_out, "w"), ensure_ascii=False, indent=2)
        print(f"   Lưu: rag/grid_search_results.json")

        update_config_yaml(BASE, rag_best)
    else:
        print(f"\n    Grid search bị bỏ qua (--no-search)")