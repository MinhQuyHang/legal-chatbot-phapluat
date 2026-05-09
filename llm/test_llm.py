# test_llm.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.rag_pipeline import RAGPipeline
from llm.chain import LegalChatChain

rag = RAGPipeline('data/rag_data.json', 'model/phobert_classifier')
chain = LegalChatChain(rag, temperature=1.0, top_k_docs=2)

print('=== Testing LLM ===')
res = chain.ask('Pháp luật là gì')
print('LLM ANSWER:')
print(res['answer'])
print()
print('First RAG passage (truncated):')
print(res['sources'][0]['content'][:300])