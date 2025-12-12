import torch
from transformers import BertForSequenceClassification

MODEL_DIR = "/workspace/app/data/bert-sst2"
ONNX_PATH = "/workspace/app/data/bert-sst2.onnx"

model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

dummy_input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
dummy_attention_mask = torch.ones_like(dummy_input_ids)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"},
    },
)
