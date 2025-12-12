import numpy as np
import onnx
import onnxruntime as ort
from transformers import BertTokenizerFast

MODEL_DIR = "/workspace/app/data/bert-sst2"
ONNX_PATH = "/workspace/app/data/bert-sst2.onnx"

# -------------------------------
# 1) ONNX チェック
# -------------------------------
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model structure: OK")

# -------------------------------
# 2) ONNX Runtime セッション初期化
# -------------------------------
ort_session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"],
)
print("ONNX Runtime session initialized.")

# -------------------------------
# 3) Tokenizer 読み込み
# -------------------------------
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

# -------------------------------
# 4) ポジ・ネガ混在のテキスト
# -------------------------------
texts = [
    "This movie is great!",          # ポジ
    "This movie is terrible.",       # ネガ
    "I really loved this film.",     # ポジ寄り
    "I really hated this film.",     # ネガ寄り
    "The plot was boring and slow.", # ネガ
]

label_names = ["negative", "positive"]

# -------------------------------
# 5) 1文ずつ ONNX 推論
# -------------------------------
for text in texts:
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",  # numpy
    )

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    logits = ort_session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )[0]  # shape: (1, 2)

    pred_id = int(np.argmax(logits, axis=-1)[0])
    label = label_names[pred_id]

    print("text:", text)
    print("  logits:", logits)
    print("  pred_id:", pred_id, "->", label)
    print("-" * 40)

print("DONE")
