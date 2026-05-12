"""
setup_data.py
Tự động tải model và data nặng từ Google Drive về máy.

Cách dùng:
    pip install gdown
    python setup_data.py
"""
import os
import gdown

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    {
        "name": "data/rag_data.json",
        "url":  "https://drive.google.com/uc?id=1OFytX2ObvYCWc6KaNhigbDA_FkFBiP9V",
        "type": "file",
    },
    {
        "name": "data/faiss_index.bin",
        "url":  "https://drive.google.com/uc?id=1jkYPLENhpVnPMjrTzyFjRTlPMpqZ-mlx",
        "type": "file",
    },
    {
        "name": "data/faiss_meta.json",
        "url":  "https://drive.google.com/uc?id=1wpXE9rQPXWjrPfysX1pJEJpNGVcdlSw1",
        "type": "file",
    },
    {
        "name": "model/phobert_classifier",
        "url":  "https://drive.google.com/drive/folders/1pQTnvZ9kkdOoYLgDrQeM4BKLZnz1sj7n",
        "type": "folder",
    },
]


def main():
    print("\n" + "="*55)
    print("  Legal Chatbot — Tải model & data từ Google Drive")
    print("="*55 + "\n")

    for item in FILES:
        dest = os.path.join(BASE, item["name"])

        # Kiểm tra đã có chưa
        if item["type"] == "folder":
            already = os.path.isdir(dest) and len(os.listdir(dest)) > 0
        else:
            already = os.path.isfile(dest)

        if already:
            print(f"   Đã có sẵn: {item['name']}")
            continue

        print(f"  Đang tải: {item['name']} ...")

        # Tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(dest) if item["type"] == "file" else dest, exist_ok=True)

        try:
            if item["type"] == "folder":
                gdown.download_folder(item["url"], output=dest, quiet=False, use_cookies=False)
            else:
                gdown.download(item["url"], dest, quiet=False, fuzzy=True)
            print(f"   Xong: {item['name']}\n")
        except Exception as e:
            print(f"   Lỗi khi tải {item['name']}: {e}\n")

    print("="*55)
    print("  Hoàn tất! Chạy app bằng lệnh:")
    print("  streamlit run app/streamlit_app.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
