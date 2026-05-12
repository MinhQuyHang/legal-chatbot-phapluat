"""
llm/memory.py
Quản lý lịch sử hội thoại 
"""

from typing import List, Tuple

def create_memory(k: int = 5) -> List[Tuple[str, str]]:
    """
    Tạo danh sách rỗng để lưu lịch sử hội thoại.
    Mỗi phần tử là (question, answer).
    """
    return []

def get_history_string(memory: List[Tuple[str, str]]) -> str:
    """Lấy lịch sử dạng string để debug."""
    if not memory:
        return ""
    return "\n".join([f"Sinh viên: {q}\nTrợ lý: {a}" for q, a in memory])

def clear_memory(memory: List[Tuple[str, str]]) -> None:
    """Xóa toàn bộ lịch sử."""
    memory.clear()

def get_recent_history(history: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
    """
    Trả về K hội thoại gần nhất để đưa vào prompt ChatML.
    """
    if not history:
        return []
    return history[-k:]