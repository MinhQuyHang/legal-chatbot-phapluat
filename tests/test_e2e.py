"""
tests/test_e2e.py  

Chạy:
    python tests/test_e2e.py
    python tests/test_e2e.py --quick
    python tests/test_e2e.py --sample
    python tests/test_e2e.py --save
"""
import os, sys, json, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import answer_dict

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_QUESTIONS = [
    # (question, expected_chapter, keywords)
    ("Nhà nước xuất hiện từ khi nào trong lịch sử",    1, ["nhà nước","nguồn gốc","lịch sử"]),
    ("Thuyết thần học giải thích nhà nước như thế nào", 1, ["thuyết","thần học","nhà nước"]),
    ("Chức năng của nhà nước Việt Nam là gì",           2, ["chức năng","nhà nước","đối nội"]),
    ("Nhà nước Việt Nam do ai lãnh đạo",                2, ["nhà nước","việt nam","đảng"]),
    ("Pháp luật là gì",                                 3, ["pháp luật","quy tắc","nhà nước"]),
    ("Pháp luật ra đời theo học thuyết Mác Lênin",     3, ["pháp luật","mác","giai cấp"]),
    ("Bản chất của pháp luật là gì",                   3, ["bản chất","pháp luật","giai cấp"]),
    ("Hình thức pháp luật là gì",                      4, ["hình thức","pháp luật","tập quán"]),
    ("Tập quán pháp là gì",                            4, ["tập quán","pháp"]),
    ("Hệ thống pháp luật Việt Nam gồm những ngành luật nào", 5, ["hệ thống","ngành luật"]),
    ("Ngành luật là gì",                               5, ["ngành luật","chế định"]),
    ("Luật hành chính điều chỉnh quan hệ nào",         6, ["hành chính","điều chỉnh"]),
    ("Phương pháp điều chỉnh của luật hành chính",     6, ["hành chính","phương pháp"]),
    ("Trách nhiệm hình sự là gì",                      7, ["hình sự","trách nhiệm","hình phạt"]),
    ("Luật hình sự xác định hành vi tội phạm như thế nào", 7, ["hình sự","tội phạm"]),
    ("Luật dân sự điều chỉnh quan hệ tài sản như thế nào", 8, ["dân sự","tài sản"]),
    ("Nguyên tắc bình đẳng trong luật dân sự",         8, ["dân sự","bình đẳng"]),
    ("Luật lao động bảo vệ người lao động như thế nào", 9, ["lao động","bảo vệ"]),
    ("Công đoàn có vai trò gì",                         9, ["công đoàn","lao động"]),
    ("Luật hôn nhân và gia đình điều chỉnh quan hệ nào", 10, ["hôn nhân","gia đình"]),
]
QUICK_QUESTIONS = ALL_QUESTIONS[:10]


def _load_sample() -> list[tuple[str, int, list[str]]]:
    path = os.path.join(_BASE, "data", "sample_questions.txt")
    if not os.path.isfile(path): return []
    result = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("["): continue
        if line[0].isdigit() and ". " in line:
            line = line.split(". ", 1)[1]
        result.append((line, 0, []))
    return result


def check_components() -> bool:
    print(f"\n{'='*65}\n  KIỂM TRA THÀNH PHẦN\n{'='*65}\n")

    # [FIX-1] Xóa pkl cũ → thêm PhoBERT + E5 cache
    files = [
        ("data/classification.csv",          "data/classification.csv"),
        ("data/rag_data.json",               "data/rag_data.json"),
        ("data/sample_questions.txt",        "data/sample_questions.txt"),
        ("data/e5_index_cache.npz",          "data/e5_index_cache.npz"),        # [NEW] E5 cache
        ("model/phobert_classifier/",        "model/phobert_classifier"),        # [NEW] PhoBERT dir
        ("model/train_meta.json",            "model/train_meta.json"),
    ]
    all_ok = True
    for name, rel in files:
        path = os.path.join(_BASE, rel)
        # phobert_classifier là thư mục
        ok   = os.path.isdir(path) if rel.endswith("phobert_classifier") else os.path.isfile(path)
        if ok and not rel.endswith("phobert_classifier"):
            size = f"({os.path.getsize(path)/1024:.1f} KB)"
        elif ok:
            n_files = len(os.listdir(path))
            size = f"({n_files} files)"
        else:
            size = "MISSING!"
            all_ok = False
        print(f"  {'✓' if ok else '✗'}  {name:<40} {size}")

    # train_meta
    meta_path = os.path.join(_BASE, "model", "train_meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        print(f"\n  Train metadata:")
        for k, v in meta.items():
            print(f"    {k}: {v}")

    print()
    for mod, attr in [
        ("app.pipeline",      "answer_dict"),
        ("rag.rag_pipeline",  "RAGPipeline"),
        ("model.predict",     "PhoBERTPredictor"),   # [FIX-3]
    ]:
        try:
            getattr(__import__(mod, fromlist=[attr]), attr)
            print(f"  ✓  import {mod}.{attr}")
        except Exception as e:
            print(f"  ✗  import {mod}.{attr} → {e}")
            all_ok = False

    # Edge case safety check
    print()
    for bad, label in [("","empty"),("  ","whitespace"),("!@#","garbage")]:
        d = answer_dict(bad)
        ok = isinstance(d, dict) and d.get("error") is not None
        print(f"  {'✓' if ok else '✗'}  Edge [{label}] → error='{d.get('error')}'")

    return all_ok


def run_tests(questions: list[tuple], label: str = "FULL") -> dict:
    n   = len(questions)
    kw_hits = ch_hits_k1 = ch_hits_k3 = 0
    scores, latencies, modes = [], [], []
    mrr_list, failed = [], []

    print(f"\n{'='*65}\n  E2E TEST [{label}] — {n} câu hỏi\n{'='*65}\n")
    print(f"  {'#':>2}  {'KW':3} {'Ch@1':4} {'Ch@3':4} {'Scr':6} {'ms':>5} {'Md':6}  Câu hỏi")
    print(f"  {'─'*2}  {'─'*3} {'─'*4} {'─'*4} {'─'*6} {'─'*5} {'─'*6}  {'─'*30}")

    for i, row in enumerate(questions, 1):
        q, exp_ch, kws = row if len(row) == 3 else (*row, [])
        t0  = time.perf_counter()
        d   = answer_dict(q, top_k=3)   
        lat = time.perf_counter() - t0
        latencies.append(lat)

        top   = d["passages"][0] if d["passages"] else {}
        score = top.get("score", 0.0); scores.append(score)
        mode  = d.get("route_mode", "?")[:6]; modes.append(mode)

        # Keyword hit
        content = top.get("content", "").lower()
        kw_ok = any(kw.lower() in content for kw in kws) if kws else True
        if kw_ok: kw_hits += 1

        # Chapter hit
        if exp_ch > 0:
            ch_top1 = [p["chapter"] for p in d["passages"][:1]]
            ch_top3 = [p["chapter"] for p in d["passages"][:3]]
            c1 = exp_ch in ch_top1; c3 = exp_ch in ch_top3
            if c1: ch_hits_k1 += 1
            if c3: ch_hits_k3 += 1
            rr = next((1/(r+1) for r, p in enumerate(d["passages"]) if p["chapter"]==exp_ch), 0.0)
            mrr_list.append(rr)
            c1s, c3s = ("✓" if c1 else "✗"), ("✓" if c3 else "✗")
        else:
            c1s = c3s = "-"

        if not kw_ok: failed.append((i, q, kws))
        sym = "✓" if kw_ok else "✗"
        q_s = (q[:30]+"…") if len(q) > 30 else q
        print(f"  {i:2d}  {sym:3} {c1s:4} {c3s:4} {score:.4f} {lat*1000:>5.0f} {mode:6}  {q_s}")

    n_ch    = sum(1 for _, e, _ in questions if e > 0)
    acc     = kw_hits / n
    ch_acc1 = ch_hits_k1 / n_ch if n_ch else 0
    ch_acc3 = ch_hits_k3 / n_ch if n_ch else 0
    mrr     = float(np.mean(mrr_list)) if mrr_list else 0
    avg_s   = float(np.mean(scores))
    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    mode_dist = {m: modes.count(m) for m in set(modes)}

    print(f"\n{'='*65}\n  KẾT QUẢ\n{'='*65}")
    print(f"\n  ✅ Keyword Hit@1       : {kw_hits}/{n} = {acc:.2%}")
    if n_ch:
        print(f"  📊 Chapter Hit@1       : {ch_hits_k1}/{n_ch} = {ch_acc1:.2%}")
        print(f"  📊 Chapter Hit@3       : {ch_hits_k3}/{n_ch} = {ch_acc3:.2%}")
        print(f"  📊 MRR (chapter-level) : {mrr:.4f}")
    print(f"  📊 Cosine Sim avg      : {avg_s:.4f}   ← E5 score (0~1)")
    print(f"  ⏱️  Latency avg/p95     : {avg_lat*1000:.1f} / {p95_lat*1000:.1f} ms")
    print(f"  🗺️  Route distribution  : {mode_dist}")

    print(f"\n  📈 Score histogram (cosine similarity):")
    for lo, hi in [(0.0,0.5),(0.5,0.7),(0.7,0.85),(0.85,1.01)]:
        c = sum(1 for s in scores if lo <= s < hi)
        print(f"     [{lo:.2f}–{hi:.2f}): {'█'*c}{'░'*(n-c)} ({c})")

    if failed:
        print(f"\n  ⚠️  Không hit keyword ({len(failed)}):")
        for idx, q, kws in failed:
            print(f"     [{idx}] {q} | kws={kws}")
    else:
        print(f"\n  🎉 Tất cả {n} câu đều hit keyword!")

    verdict = (
        "🏆 SẴN SÀNG DEPLOY!" if acc >= 1.0
        else "✅ Đạt yêu cầu"   if acc >= 0.85
        else f"⚠️  Cần cải thiện ({acc:.2%})"
    )
    print(f"\n  {verdict}\n{'='*65}\n")

    return {
        "retriever":           "multilingual-e5",
        "label":               label,
        "n":                   n,
        "keyword_hit_at_1":    round(acc, 4),
        "chapter_hit_at_1":    round(ch_acc1, 4),
        "chapter_hit_at_3":    round(ch_acc3, 4),
        "mrr":                 round(mrr, 4),
        "cosine_sim_avg":      round(avg_s, 4),
        "latency_avg_ms":      round(avg_lat*1000, 2),
        "latency_p95_ms":      round(p95_lat*1000, 2),
        "route_distribution":  mode_dist,
        "failed_questions":    [q for _, q, _ in failed],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",      action="store_true")
    parser.add_argument("--sample",     action="store_true")
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_check:
        ok = check_components()
        if not ok:
            print("  ❌ Thiếu file. Kiểm tra lại model/phobert_classifier/ và data/e5_index_cache.npz\n")
            sys.exit(1)

    if args.sample:
        questions, label = _load_sample(), "SAMPLE-50"
    elif args.quick:
        questions, label = QUICK_QUESTIONS, "QUICK-10"
    else:
        questions, label = ALL_QUESTIONS, "FULL-20"

    result = run_tests(questions, label=label)

    if args.save:
        out = os.path.join(_BASE, "tests", "e2e_results.json")
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"  💾 Lưu: {out}\n")
