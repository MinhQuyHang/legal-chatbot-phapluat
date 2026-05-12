"""
model/train_model.py  ·  PhoBERT Version — Chạy trên Google Colab
"""

import os, sys, re, json, random, argparse, unicodedata, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)
warnings.filterwarnings("ignore")

# ── Cài thư viện nếu chạy trên Colab ─────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
except ImportError:
    print("Đang cài transformers + torch...")
    os.system("pip install transformers torch -q")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ── Paths ─────────────────────────────────────────────────────
_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(_BASE, "data", "classification.csv")
MODEL_DIR  = os.path.join(_BASE, "model")
CONFIG_DIR = os.path.join(_BASE, "config")
PHOBERT_SAVE_DIR = os.path.join(MODEL_DIR, "phobert_classifier")

PHOBERT_NAME = "vinai/phobert-base"   # Download tự động từ HuggingFace
NUM_LABELS   = 10                      # 10 chapters
MAX_LEN      = 128                     # Token length tối đa
BATCH_SIZE   = 16
EPOCHS       = 5
LEARNING_RATE = 2e-5
SEED         = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Word segmentation ─────────────────────────────────────────
try:
    from underthesea import word_tokenize as _wt
    def _segment(text: str) -> str:
        return _wt(text, format="text")
    _HAS_SEG = True
except ImportError:
    print("  underthesea chưa cài — cài tự động...")
    os.system("pip install underthesea -q")
    try:
        from underthesea import word_tokenize as _wt
        def _segment(text: str) -> str:
            return _wt(text, format="text")
        _HAS_SEG = True
    except:
        def _segment(text: str) -> str:
            return text
        _HAS_SEG = False

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(text: str, use_seg: bool = True) -> str:
    """
    NFC normalize → lowercase → clean → word segment
    PhoBERT cần word segment (underthesea) để nhận dạng từ tiếng Việt đúng.
    """
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


# ── PyTorch Dataset ───────────────────────────────────────────
class ChapterDataset(Dataset):
    """Dataset wrapper cho PhoBERT tokenizer."""
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.labels = labels
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Helpers ───────────────────────────────────────────────────
def _sep(title: str = "", w: int = 70):
    pad = (w - len(title) - 2) // 2
    print(f"\n{'═'*pad} {title} {'═'*pad}" if title else "─"*w)

def _print_cm(cm: np.ndarray) -> None:
    header = "       " + "".join(f"{i+1:>5}" for i in range(cm.shape[0]))
    print(header)
    for i, row in enumerate(cm):
        highlight = [f"\033[1m{v:>5}\033[0m" if j == i else f"{v:>5}" for j, v in enumerate(row)]
        print(f"  Ch.{i+1:>2} {''.join(highlight)}")


# ── Train 1 epoch ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total   += len(batch["labels"])

    return total_loss / len(loader), correct / total


# ── Evaluate ──────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return (
        total_loss / len(loader),
        np.array(all_preds),
        np.array(all_labels),
    )


# ── Lấy probabilities (thay thế predict_proba của sklearn) ────
def get_probabilities(model, loader):
    """Trả về softmax probabilities — dùng cho RAG threshold search."""
    import torch.nn.functional as F
    model.eval()
    all_proba, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            proba  = F.softmax(logits, dim=-1).cpu().numpy()
            all_proba.extend(proba)
            all_labels.extend(labels.numpy())
    return np.array(all_proba), np.array(all_labels)


# ── RAG threshold search  ──
def search_rag_thresholds(proba: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Grid search CONFIDENCE_THRESHOLD và MARGIN_THRESHOLD.
    Metric: scoped_ratio × top2_accuracy
    """
    conf_grid   = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    margin_grid = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]
    top1_idx   = sorted_idx[:, 0]
    top2_idx   = sorted_idx[:, 1]
    conf_arr   = proba[np.arange(len(proba)), top1_idx]
    margin_arr = conf_arr - proba[np.arange(len(proba)), top2_idx]

    in_top2 = np.array([
        y_val[i] in (top1_idx[i], top2_idx[i]) for i in range(len(y_val))
    ])

    best = {"score": -1.0, "conf_threshold": 0.45, "margin_threshold": 0.15,
            "scoped_ratio": 0.0, "top2_accuracy": 0.0}
    results = []

    for conf_t, margin_t in itertools.product(conf_grid, margin_grid):
        is_scoped    = (conf_arr >= conf_t) & (margin_arr >= margin_t)
        scoped_ratio = float(is_scoped.mean())
        top2_acc     = float(in_top2[is_scoped].mean()) if is_scoped.any() else 0.0
        score        = scoped_ratio * top2_acc
        results.append((score, conf_t, margin_t, scoped_ratio, top2_acc))
        if score > best["score"]:
            best = {
                "score":            round(score, 4),
                "conf_threshold":   conf_t,
                "margin_threshold": margin_t,
                "scoped_ratio":     round(scoped_ratio, 4),
                "top2_accuracy":    round(top2_acc, 4),
            }

    results.sort(reverse=True)
    print(f"\n  {'conf':>5}  {'margin':>7}  {'scoped%':>8}  {'top2_acc':>9}  {'score':>7}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*7}")
    for sc, ct, mt, sr, ta in results[:10]:
        tag = " ← best" if (ct == best["conf_threshold"] and mt == best["margin_threshold"]) else ""
        print(f"  {ct:>5.2f}  {mt:>7.2f}  {sr:>8.1%}  {ta:>9.1%}  {sc:>7.4f}{tag}")

    print(f"\n  ✓ Best: conf≥{best['conf_threshold']}  margin≥{best['margin_threshold']}")
    print(f"         Scoped ratio = {best['scoped_ratio']:.1%}  |  Top-2 accuracy = {best['top2_accuracy']:.1%}")
    return best


# ── Ghi config yaml ───────────────────────────────────────────
def write_config_yaml(best_epoch: int, best_val_f1: float,
                      rag_thresholds: dict, use_seg: bool) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, "settings.yaml")

    lines = [
        "# AUTO-GENERATED by train_model.py (PhoBERT) — không sửa tay",
        "model:",
        f"  type: phobert",
        f"  base: {PHOBERT_NAME}",
        f"  num_labels: {NUM_LABELS}",
        f"  max_len: {MAX_LEN}",
        f"  best_epoch: {best_epoch}",
        f"  val_macro_f1: {best_val_f1}",
        f"  use_segment: {str(use_seg).lower()}",
        f"  seed: {SEED}",
        "rag:",
        f"  confidence_threshold: {rag_thresholds['conf_threshold']}",
        f"  margin_threshold: {rag_thresholds['margin_threshold']}",
        f"  score_gate: 0.35",
        f"  chapter_boost: 1.15",
        f"  bm25_weight: 0.6",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f" config/settings.yaml")


# ── MAIN ──────────────────────────────────────────────────────
def main(use_seg: bool = True) -> None:

    print(f"\n    Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    Không có GPU — train sẽ chậm hơn nhiều. Khuyên dùng Colab T4.")

    # ── 1. Load ───────────────────────────────────────────────
    _sep("BƯỚC 1: TẢI & KIỂM TRA DỮ LIỆU")
    df = pd.read_csv(DATA_PATH)
    assert {"text", "label"}.issubset(df.columns), f"Thiếu cột: {df.columns.tolist()}"
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    assert df["label"].between(1, 10).all(), "Label ngoài range 1–10"

    print(f"  ✓ {df.shape[0]} samples | {df['label'].nunique()} classes")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        print(f"    Label {lbl:>2}: {'█'*(cnt//5)} ({cnt})")

    # ── 2. Preprocess ─────────────────────────────────────────
    _sep("BƯỚC 2: PREPROCESSING")
    seg_status = f"underthesea={'ON' if (use_seg and _HAS_SEG) else 'OFF'}"
    print(f"  Pipeline: NFC → lowercase → clean → word segment [{seg_status}]")
    df["text_clean"] = df["text"].apply(lambda t: preprocess(t, use_seg))
    for _, row in df.sample(2, random_state=SEED).iterrows():
        print(f"    RAW  : {row['text'][:65]}")
        print(f"    CLEAN: {row['text_clean'][:65]}")

    # ── 3. Split ──────────────────────────────────────────────
    _sep("BƯỚC 3: STRATIFIED SPLIT (70 / 15 / 15)")
    X = df["text_clean"].values
    y = (df["label"] - 1).values    # 0-indexed

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    print(f"  Train : {len(X_train):>4} ({len(X_train)/len(X):.0%})")
    print(f"  Val   : {len(X_val):>4} ({len(X_val)/len(X):.0%})")
    print(f"  Test  : {len(X_test):>4} ({len(X_test)/len(X):.0%})  ← chỉ chạm 1 lần")

    # ── 4. Tokenizer ──────────────────────────────────────────
    _sep("BƯỚC 4: LOAD PHOBERT TOKENIZER")
    print(f"  Downloading: {PHOBERT_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
    print(f"  Tokenizer loaded | max_len={MAX_LEN}")

    train_dataset = ChapterDataset(X_train, y_train, tokenizer)
    val_dataset   = ChapterDataset(X_val,   y_val,   tokenizer)
    test_dataset  = ChapterDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)
    print(f"   DataLoader: train={len(train_loader)} batches | val={len(val_loader)} | test={len(test_loader)}")

    # ── 5. Load model ─────────────────────────────────────────
    _sep("BƯỚC 5: LOAD PHOBERT MODEL")
    model = AutoModelForSequenceClassification.from_pretrained(
        PHOBERT_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)
    print(f"  ✓ PhoBERT + Classification Head (10 classes)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total params: {total_params:,} | Trainable: {trainable:,}")

    # ── 6. Training ───────────────────────────────────────────
    _sep("BƯỚC 6: FINE-TUNE PHOBERT")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1    = 0.0
    best_epoch     = 1
    best_val_preds = None

    print(f"  Epochs={EPOCHS} | LR={LEARNING_RATE} | Batch={BATCH_SIZE} | Warmup=10%")
    print(f"\n  {'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>7}  {'ValMacroF1':>10}  {'ValAcc':>6}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*6}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_preds, val_true = evaluate(model, val_loader)

        val_acc = accuracy_score(val_true, val_preds)
        val_f1  = f1_score(val_true, val_preds, average="macro", zero_division=0)
        tag     = " ← best" if val_f1 > best_val_f1 else ""

        print(f"  {epoch:>5}  {train_loss:>9.4f}  {train_acc:>8.4f}  {val_loss:>7.4f}  {val_f1:>10.4f}  {val_acc:>6.4f}{tag}")

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_epoch     = epoch
            best_val_preds = val_preds
            # Lưu checkpoint tốt nhất
            os.makedirs(PHOBERT_SAVE_DIR, exist_ok=True)
            model.save_pretrained(PHOBERT_SAVE_DIR)
            tokenizer.save_pretrained(PHOBERT_SAVE_DIR)

    print(f"\n  ✓ Best epoch: {best_epoch} | Val Macro F1: {best_val_f1:.4f}")
    print(f"  ✓ Model đã lưu tại: {PHOBERT_SAVE_DIR}")

    # ── 7. Load best model → đánh giá Test ───────────────────
    _sep("BƯỚC 7: ĐÁNH GIÁ TRÊN TEST SET [1 lần duy nhất]")
    best_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_SAVE_DIR).to(DEVICE)
    _, test_preds, test_true = evaluate(best_model, test_loader)

    class_labels = [f"Ch.{i+1}" for i in range(NUM_LABELS)]
    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average="macro",    zero_division=0)
    wf1 = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    kap = cohen_kappa_score(test_true, test_preds)
    verdict = (
        "🏆 Excellent (κ>0.80)" if kap > 0.80 else
        "✅ Good (κ>0.60)"      if kap > 0.60 else
        "⚠️  Fair (κ>0.40)"     if kap > 0.40 else
        "❌ Poor — cần cải thiện"
    )
    print(f"\n  ── PhoBERT Classifier ──────────────────────────────")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {mf1:.4f}   ← metric chính")
    print(f"  Weighted F1   : {wf1:.4f}")
    print(f"  Cohen's Kappa : {kap:.4f}   {verdict}")
    print(f"\n  Classification Report:")
    print(classification_report(test_true, test_preds, target_names=class_labels, digits=4, zero_division=0))
    cm = confusion_matrix(test_true, test_preds)
    print(f"  Confusion Matrix:")
    _print_cm(cm)

    errs = sorted(
        [(cm[i][j], i+1, j+1) for i in range(10) for j in range(10) if i != j and cm[i][j] > 0],
        reverse=True
    )[:5]
    if errs:
        print(f"\n  Top-5 nhầm lẫn:")
        for cnt, tc, pc in errs:
            print(f"    Thực Ch.{tc} → Dự đoán Ch.{pc}: {cnt} lần")

    # ── 8. RAG threshold search ───────────────────────────────
    _sep("BƯỚC 8: SEARCH RAG THRESHOLDS [dùng Val set]")
    val_proba, _ = get_probabilities(best_model, val_loader)
    rag_best = search_rag_thresholds(val_proba, y_val)

    # ── 9. Lưu metadata + config ──────────────────────────────
    _sep("BƯỚC 9: LƯU METADATA & CONFIG")
    meta = {
        "model_type":    "phobert",
        "base_model":    PHOBERT_NAME,
        "best_epoch":    best_epoch,
        "val_macro_f1":  round(best_val_f1, 4),
        "test_macro_f1": round(mf1, 4),
        "test_accuracy": round(acc, 4),
        "use_segment":   use_seg and _HAS_SEG,
        "num_labels":    NUM_LABELS,
        "max_len":       MAX_LEN,
        "splits":        {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "rag_thresholds": rag_best,
    }
    meta_path = os.path.join(MODEL_DIR, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ train_meta.json")
    print(f"  ✓ phobert_classifier/ (model + tokenizer)")

    write_config_yaml(best_epoch, round(best_val_f1, 4), rag_best, use_seg and _HAS_SEG)

    # ── 10. Zip để download ───────────────────────────────────
    _sep("BƯỚC 10: ZIP MODEL ĐỂ DOWNLOAD")
    import shutil
    zip_path = os.path.join(MODEL_DIR, "phobert_classifier")
    shutil.make_archive(zip_path, "zip", PHOBERT_SAVE_DIR)
    print(f"  ✓ phobert_classifier.zip — Download file này về máy!")
    print(f"  📁 Giải nén vào: model/phobert_classifier/")

    _sep("HOÀN THÀNH")
    print(f"  PhoBERT Macro F1 (Test) = {mf1:.4f}")
    print(f"  Bước tiếp theo:")
    print(f"    1. Download model/phobert_classifier.zip")
    print(f"    2. Giải nén vào thư mục model/ trên máy local")
    print(f"    3. Chạy: streamlit run app.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-segment", action="store_true",
                        help="Tắt word segmentation (underthesea)")
    args = parser.parse_args()
    main(use_seg=not args.no_segment)
