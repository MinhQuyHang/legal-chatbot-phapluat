"""
config/settings.py
Toàn bộ constants & paths tập trung ở đây
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

RAG_DATA_PATH      = DATA_DIR / "rag_data.json"
CLASSIFY_CSV       = DATA_DIR / "classification.csv"
PHOBERT_MODEL_PATH = MODEL_DIR / "phobert_classifier"

# ── RAG ────────────────────────────────────────────────────────
RAG_TOP_K             = 3
RAG_CONFIDENCE_THRESH = 0.45
RAG_MARGIN_THRESH     = 0.15
RAG_SCORE_GATE        = 0.35
RAG_CHAPTER_BOOST     = 1.10 
RAG_E5_MODEL          = "intfloat/multilingual-e5-large" 

# ── LLM / Ollama ───────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PHOGPT_MODEL = os.getenv("PHOGPT_MODEL", "phogpt-legal") 
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MEMORY_K    = int(os.getenv("LLM_MEMORY_K", "2"))
