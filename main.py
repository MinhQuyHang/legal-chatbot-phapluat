"""
main.py  ·  CLI Chatbot Production 
============================================
Cách chạy:
    python main.py                          # Interactive
    python main.py --demo                   # 10 câu mẫu tự động
    python main.py --sample                 # 50 câu sample_questions.txt
    python main.py -q "Pháp luật là gì?"    # 1 câu nhanh
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Khai báo đường dẫn
_BASE      = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE = os.path.join(_BASE, "data", "sample_questions.txt")

# Import Full Pipeline 
from app.pipeline import RAGPipeline
# from rag.rag_pipeline import PhoGPTChain   # không dùng nữa
from llm.chain import LegalChatChain       # sử dụng chain llm

BANNER = """
==============================================================
             CHATBOT HỎI ĐÁP PHÁP LUẬT ĐẠI CƯƠNG            
   RAG Pipeline: PhoBERT -> Multilingual-e5 -> PhoGPT    
   Gõ 'help' để xem lệnh  |  'quit' để thoát                 
=============================================================="""

HELP = """
  Lệnh:
    <câu hỏi>    Đặt câu hỏi về pháp luật
    demo         Chạy 10 câu mẫu
    sample       Chạy 50 câu sample_questions.txt
    quit / q     Thoát
"""

EXIT_KW = {"quit","exit","thoat","thoát","q","bye"}

DEMO_Q = [
    "Pháp luật là gì?",
    "Quy phạm pháp luật là gì?",
    "Nhà nước có chức năng gì?",
    "Vi phạm pháp luật là gì?",
    "Trách nhiệm hình sự là gì?",
    "Luật dân sự điều chỉnh gì?",
    "Luật lao động bảo vệ ai?",
    "Điều kiện kết hôn là gì?",
    "Hệ thống pháp luật VN gồm gì?",
    "Tập quán pháp là gì?",
]

# --- Khởi tạo Hệ thống Toàn cục ---
print("[INFO] Đang khởi tạo hệ thống (PhoBERT + E5 + PhoGPT)...")
rag_system = RAGPipeline(
    rag_data_path=os.path.join(_BASE, "data", "rag_data.json"), 
    phobert_model_path=os.path.join(_BASE, "model", "phobert_classifier"),  
    top_k=3
)

chatbot = LegalChatChain(
    rag_pipeline=rag_system,
    model_name="mrjacktung/phogpt-4b-chat-gguf",
    ollama_base_url="http://localhost:11434",
    temperature=0.1,
    top_k_docs=3,
    memory_k=5,
    max_chunk_length=500
)
# -----------------------------------

def _load_sample() -> list[str]:
    if not os.path.isfile(SAMPLE_FILE):
        print(f"  [ERROR] Không tìm thấy: {SAMPLE_FILE}")
        return []
    qs = []
    for line in open(SAMPLE_FILE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("["): continue
        if line[0].isdigit() and ". " in line:
            line = line.split(". ", 1)[1]
        qs.append(line)
    return qs

def run_batch(questions: list[str], label: str) -> None:
    print(f"\n{'='*62}\n  {label} — {len(questions)} câu\n{'='*62}\n")
    total_ms = 0
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i:2d}] Câu hỏi: {q}")
        
        # Gọi Full Pipeline
        result = chatbot.ask(q)
        ms = result['latency'] * 1000
        total_ms += ms
        
        ans = result['answer']
        conf = result['confidence']
        
       
        short_ans = f"{ans[:150]}..." if len(ans) > 150 else ans
        print(f"   Trợ lý: {short_ans}")
        print(f"   [PhoBERT Conf: {conf:.2f} | Time: {ms:>5.0f}ms]")
        
    n = len(questions)
    print(f"\n  [OK] Đã chạy xong {n} câu | Tốc độ trung bình {total_ms/n:.0f}ms/câu\n")

def run_interactive() -> None:
    print(BANNER)
    print("  [Gợi ý]: " + " | ".join(DEMO_Q[:3]))
    while True:
        try:
            raw = input(f"\nSinh viên: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nTạm biệt!")
            break
            
        if not raw: continue
        lower = raw.lower()
        if lower in EXIT_KW: print("\nTạm biệt!"); break
        if lower == "help": print(HELP); continue
        if lower == "demo": run_batch(DEMO_Q, "DEMO"); continue
        if lower == "sample":
            qs = _load_sample()
            if qs: run_batch(qs, "SAMPLE")
            continue
            
        # Gọi luồng RAG + LLM
        result = chatbot.ask(raw)
        print(f"\nTrợ lý: {result['answer']}")
        print(f"\n[Thời gian: {result['latency']}s | Tự tin PhoBERT: {result['confidence']}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("-q","--question", default=None)
    args = parser.parse_args()

    if args.question:
        res = chatbot.ask(args.question)
        print(f"\nTrợ lý: {res['answer']}")
    elif args.demo:
        print(BANNER); run_batch(DEMO_Q, "DEMO – 10 câu mẫu")
    elif args.sample:
        print(BANNER); qs = _load_sample()
        if qs: run_batch(qs, f"SAMPLE – {SAMPLE_FILE}")
    else:
        run_interactive()